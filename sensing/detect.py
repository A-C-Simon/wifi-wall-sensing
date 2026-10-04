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


def motion_score(session, win_s: float = 10.0, min_sweep_l1: float = 3.0):
    """Compute the fused motion score without thresholding.

    Split out from `detect` so an empty-room baseline can be characterised with
    exactly the same feature pipeline that will later be used for decisions --
    otherwise the calibration and the detector drift apart.

    Returns (t, motion, fine_score, sweep_score, sweep_l1, sweep_interval,
    fine_rate). sweep_l1 is the raw multi-AP change in dB, held constant between
    fresh sweeps, and is carried for diagnostics only.
    """
    ft = np.array([f.t for f in session.fine])
    if len(ft) == 0:
        z = np.zeros(0)
        return z, z, z, z, z, 0.0, 0.0  # t, motion, fine, sweep, l1, interval, rate
    rssi = np.array([f.rssi for f in session.fine])
    sd = np.array([f.sd for f in session.fine])
    chg = np.array([f.chg for f in session.fine])
    dt = float(np.median(np.diff(ft))) if len(ft) > 1 else 0.05

    ff = F.fine_features(ft, rssi, sd, chg, win_s, dt)
    # Detrend half-life must cover the whole recording. With a short half-life
    # the baseline follows the motion itself and cancels the signal; with too
    # long a one, AGC drift accumulates faster than the body perturbs the link
    # and the score ends up tracking drift (observed: an empty phase scoring
    # higher than the walk phase). Tie it to the capture length.
    span = float(ft[-1] - ft[0]) if len(ft) > 1 else 600.0
    half_life = max(60.0, min(max(span * 0.6, 3 * win_s), 1800.0))
    fine_res = F.detrend(ff[:, 0], ft, half_life_s=half_life)
    fine_score = F.zscore_robust(F.envelope(fine_res))

    sweep_score = np.zeros(len(ft))
    sweep_l1 = np.zeros(len(ft))
    st = np.array([s.t for s in session.sweeps])
    interval = float(np.median(np.diff(st))) if len(st) > 1 else 0.0
    n_sweeps = len(session.sweeps)
    if n_sweeps >= 2:
        keys = session.ap_keys(min_count=max(1, n_sweeps // 3))
        if keys:
            mat = np.array([s.vector(keys) for s in session.sweeps], dtype=float)
            sf = F.sweep_features(st, mat)
            if len(sf):
                l1 = sf[:, 0]
                l1z = F.ratio_score(l1)
                se = st[1:]
                idx = np.clip(np.searchsorted(se, ft) - 1, 0, len(l1z) - 1)
                near = np.abs(se[idx] - ft) < max(2.0 * interval, 10.0)
                # The sweep statistic is a z-score over only n_intervals
                # samples and is then held constant for ~12 s. With too little
                # history it is statistically meaningless yet would dominate
                # hundreds of fine buckets, so fade it in as evidence accrues.
                w = min(1.0, len(l1z) / 12.0)
                sweep_score[near] = w * l1z[idx[near]]
                sweep_l1[near] = l1[idx[near]]

    motion = np.maximum(fine_score, sweep_score)
    # Both terms are robust z-scores, so they are directly comparable and the
    # max is a valid "either channel saw it" rule. Note sweep_score is left at
    # 0 (the median of its own distribution) where no fresh sweep is due, which
    # makes it neutral rather than falsely reassuring.
    return (ft, motion, fine_score, sweep_score, sweep_l1, interval,
            float(np.mean(ff[:, 2])))


def baseline_threshold(session, win_s: float = 10.0, k: float = 6.0,
                       min_sweep_l1: float = 3.0) -> dict:
    """Characterise the score distribution of an EMPTY room.

    With RSSI this noisy, a threshold chosen from theory is meaningless; the
    only defensible threshold is one measured while the space is known to be
    unoccupied, then inflated by `k` robust sigmas to bound the false-alarm
    rate. Returns the distribution so the calibration can be audited.
    """
    ft, motion, fine, sweep, _l1, interval, frate = motion_score(
        session, win_s, min_sweep_l1)
    if len(motion) == 0:
        return {}
    thr = F.auto_threshold(motion, k=k)
    return {
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
           min_sweep_l1: float = 3.0, threshold: float | None = None) -> DetectResult:
    """Combine both channels into one motion decision per fine bucket.

    win_s should be several times the ~0.2 Hz information rate of the RSSI
    register, otherwise the statistic has too few real transitions to separate
    motion from drift. Pass `threshold` from a `baseline_threshold` measurement
    of an empty room rather than relying on the automatic fallback.
    """
    res = DetectResult()
    ft, motion, fine_score, sweep_score, sweep_l1, interval, frate = motion_score(
        session, win_s, min_sweep_l1)
    res.fine_rate_hz = frate
    res.sweep_interval_s = interval
    if len(ft) == 0:
        return res

    thr = threshold if threshold is not None else F.auto_threshold(motion, k=k)
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