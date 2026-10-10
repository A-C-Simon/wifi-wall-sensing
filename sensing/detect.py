"""Motion / presence decisions from the two measured channels.

Physical basis: a human body is a dielectric reflector of roughly 0.25 m^2
cross-section. When it enters the Fresnel zone of any TX-RX path it perturbs
the multipath, which shows up as (a) higher variance and more frequent
transitions in the associated-link RSSI, and (b) a correlated shift across many
of the multi-AP anchors at once.

A correlated shift across *many* anchors is the discriminative part: ordinary
drift moves all anchors slowly and coherently, whereas a moving body changes
different anchors by different amounts because their geometries differ.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import features as F


@dataclass
class Detection:
    t: float
    motion: float          # combined score (robust z, >~3 means active)
    occupancy: bool
    fine_score: float
    sweep_score: float
    sweep_l1: float


@dataclass
class DetectResult:
    detections: list[Detection] = field(default_factory=list)
    fine_rate_hz: float = 0.0
    sweep_interval_s: float = 0.0
    threshold: float = 0.0

    @property
    def occupancy_fraction(self) -> float:
        if not self.detections:
            return 0.0
        return sum(1 for d in self.detections if d.occupancy) / len(self.detections)

    def segments(self, min_len: float = 1.0):
        """Contiguous occupied intervals as (t_start, t_end) pairs."""
        out = []
        cur = None
        for d in self.detections:
            if d.occupancy:
                if cur is None:
                    cur = d.t
            else:
                if cur is not None and d.t - cur >= min_len:
                    out.append((cur, d.t))
                cur = None
        if cur is not None and self.detections and self.detections[-1].occupancy:
            out.append((cur, self.detections[-1].t))
        return out


def motion_score(session, win_s: float = 10.0, min_sweep_l1: float = 3.0,
                 calibration: dict | None = None):
    """Compute the fused motion score without thresholding.

    Split out from `detect` so an empty-room baseline can be characterised with
    exactly the same feature pipeline that will later be used for decisions --
    otherwise the calibration and the detector drift apart.

    Returns (t, motion, fine_score, sweep_score, sweep_l1, sweep_interval,
    fine_rate). sweep_l1 is the raw multi-AP change in dB, held constant between
    fresh sweeps, and is carried for diagnostics only.
    """
    if calibration is not None:
        validate_calibration(calibration, win_s)
    keys = (calibration["sweep_keys"] if calibration is not None else
            session.ap_keys(min_count=max(1, len(session.sweeps) // 3)))
    return motion_score_arrays(
        np.array([f.t for f in session.fine]),
        np.array([f.rssi for f in session.fine]),
        np.array([f.sd for f in session.fine]),
        np.array([f.chg for f in session.fine]),
        np.array([s.t for s in session.sweeps]),
        np.array([s.vector(keys) for s in session.sweeps], dtype=float)
        .reshape(len(session.sweeps), len(keys)),
        win_s, min_sweep_l1, calibration)


def _fine_energy(ft, rssi, sd, chg, win_s):
    dt = float(np.median(np.diff(ft))) if len(ft) > 1 else 0.05
    ff = F.fine_features(ft, rssi, sd, chg, win_s, dt)
    # Fixed across baseline and measurement; capture length is not a physical
    # filter parameter and must not change historical scores as data arrives.
    residual = F.detrend(ff[:, 0], ft, half_life_s=max(120.0, 3 * win_s))
    return F.envelope(residual), ff


def validate_calibration(calibration, win_s):
    if calibration.get("score_version") != 2:
        raise ValueError("baseline uses an old score pipeline; record a new baseline")
    if calibration.get("win_s") != win_s:
        raise ValueError("baseline window differs from analysis window; record a matching baseline")
    for name in ("fine_center", "fine_scale", "sweep_reference", "threshold"):
        if not np.isfinite(calibration.get(name, np.nan)):
            raise ValueError(f"invalid baseline {name}")
    if calibration["fine_scale"] <= 0 or calibration["sweep_reference"] <= 0:
        raise ValueError("baseline scales must be positive")
    if not isinstance(calibration.get("sweep_keys"), list):
        raise ValueError("baseline must include its sweep anchor set")


def motion_score_arrays(ft, rssi, sd, chg, st, mat, win_s=10.0,
                        min_sweep_l1=3.0, calibration=None):
    """Shared scoring implementation for recorded and live samples.

    A supplied empty-room calibration fixes normalization independently of
    the target recording. Uncalibrated scores are descriptive only.
    """
    if calibration is not None:
        validate_calibration(calibration, win_s)
    if not len(ft):
        z = np.zeros(0)
        return z, z, z, z, z, 0.0, 0.0
    energy, ff = _fine_energy(ft, rssi, sd, chg, win_s)
    motion, fine_score, sweep_score, sweep_l1, interval = fuse_scores(
        ft, energy, st, mat, min_sweep_l1, calibration)
    return (ft, motion, fine_score, sweep_score, sweep_l1, interval,
            float(np.mean(ff[:, 2])))


def fuse_scores(ft, energy, st, mat, min_sweep_l1=3.0, calibration=None):
    """Normalize fine energy and fuse it with timestamped sweep evidence."""
    if calibration is None:
        fine_score = F.zscore_robust(energy)
    else:
        fine_score = (energy - calibration["fine_center"]) / calibration["fine_scale"]

    sweep_score = np.zeros(len(ft))
    sweep_l1 = np.zeros(len(ft))
    interval = float(np.median(np.diff(st))) if len(st) > 1 else 0.0
    n_sweeps = len(st)
    if n_sweeps >= 2:
        if mat.shape[1]:
            sf = F.sweep_features(st, mat)
            if len(sf):
                l1 = sf[:, 0]
                if calibration is None:
                    l1z = F.ratio_score(l1)
                else:
                    l1z = (l1 / calibration["sweep_reference"] - 1.0) * 3.0
                l1z = np.where(l1 >= min_sweep_l1, l1z, 0.0)
                se = st[1:]
                # The sweep statistic is a z-score over only n_intervals
                # samples and is then held constant for ~12 s. With too little
                # history it is statistically meaningless yet would dominate
                # hundreds of fine buckets, so fade it in as evidence accrues.
                w = np.minimum(1.0, np.arange(1, len(l1z) + 1) / 12.0)
                # Each expiry uses only the interval that just completed.
                # A later scan must not change whether earlier evidence was
                # stale, even if later scans have a different cadence.
                max_age = np.maximum(2.0 * np.diff(st), 10.0)
                sweep_score = F.hold_sweeps(ft, se, w * l1z, max_age)
                sweep_l1 = F.hold_sweeps(ft, se, l1, max_age)

    motion = np.maximum(fine_score, sweep_score)
    # The threshold is measured on this fused statistic. Sweep ratio scores
    # and fine MAD scores are different units, so neither is a probability.
    return motion, fine_score, sweep_score, sweep_l1, interval


def baseline_threshold(session, win_s: float = 10.0, k: float = 6.0,
                       min_sweep_l1: float = 3.0) -> dict:
    """Characterise the score distribution of an EMPTY room.

    With RSSI this noisy, a threshold chosen from theory is meaningless; the
    only defensible threshold is one measured while the space is known to be
    unoccupied, then inflated by `k` robust sigmas to bound the false-alarm
    rate. Returns the distribution so the calibration can be audited.
    """
    if not session.fine:
        return {}
    ft = np.array([f.t for f in session.fine])
    energy, _ = _fine_energy(ft, np.array([f.rssi for f in session.fine]),
                              np.array([f.sd for f in session.fine]),
                              np.array([f.chg for f in session.fine]), win_s)
    center = float(np.median(energy))
    scale = float(np.median(np.abs(energy - center)))
    if scale <= 1e-9:
        scale = float(np.std(energy))
    scale = max(scale, 0.1)  # finite floor for a perfectly quiet baseline
    keys = session.ap_keys(min_count=max(1, len(session.sweeps) // 3))
    mat = np.array([sw.vector(keys) for sw in session.sweeps], dtype=float)
    sf = F.sweep_features(np.array([sw.t for sw in session.sweeps]), mat)
    reference = max(min_sweep_l1, float(np.median(sf[:, 0])) if len(sf) else 0., 1e-9)
    params = {"score_version": 2, "win_s": win_s, "fine_center": center,
              "fine_scale": scale, "sweep_reference": reference,
              "sweep_keys": keys, "threshold": 0.0}
    ft, motion, fine, sweep, _l1, interval, frate = motion_score(
        session, win_s, min_sweep_l1, params)
    thr = F.auto_threshold(motion, k=k)
    return {
        **params,
        "n_samples": int(len(motion)),
        "duration_s": float(ft[-1] - ft[0]) if len(ft) > 1 else 0.0,
        "threshold": float(thr),
        "k": k,
        "motion_p50": float(np.median(motion)),
        "motion_p95": float(np.percentile(motion, 95)),
        "motion_p99": float(np.percentile(motion, 99)),
        "motion_max": float(np.max(motion)),
        "fine_rate_hz": frate,
        "sweep_interval_s": interval,
        "win_s": win_s,
    }


def detect(session, win_s: float = 10.0, k: float = 6.0,
           min_sweep_l1: float = 3.0, threshold: float | None = None,
           calibration: dict | None = None) -> DetectResult:
    """Combine both channels into one motion decision per fine bucket.

    win_s should be several times the ~0.2 Hz information rate of the RSSI
    register, otherwise the statistic has too few real transitions to separate
    motion from drift. Pass `threshold` from a `baseline_threshold` measurement
    of an empty room rather than relying on the automatic fallback.
    """
    res = DetectResult()
    ft, motion, fine_score, sweep_score, sweep_l1, interval, frate = motion_score(
        session, win_s, min_sweep_l1, calibration)
    res.fine_rate_hz = frate
    res.sweep_interval_s = interval
    if len(ft) == 0:
        return res

    thr = (threshold if threshold is not None else calibration["threshold"]
           if calibration is not None else F.auto_threshold(motion, k=k))
    res.threshold = float(thr)

    for i in range(len(ft)):
        m = float(motion[i])
        res.detections.append(Detection(
            t=float(ft[i]),
            motion=m,
            occupancy=bool(m >= thr),
            fine_score=float(fine_score[i]),
            sweep_score=float(sweep_score[i]),
            sweep_l1=float(sweep_l1[i]),
        ))
    return res
