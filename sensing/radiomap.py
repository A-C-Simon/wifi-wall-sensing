"""Radio-map calibration and fingerprint localisation.

With a single receiver, localisation has to come from *spatial* differences in
the multi-AP RSSI vector rather than from timing. Each visible access point is
a passive anchor at a different physical location, so the vector
[rssi(ch1), rssi(ch2), ...] is a fingerprint whose value depends on where the
perturbing body is.

Procedure: survey the space on a grid, record the vector at each cell, then
match a live vector against the surveyed fingerprints. This is the classic
fingerprint approach (Mager/Lundrigan/Patwari, IEEE JSAC 2015) and is the only
localisation method that works on RSSI-only, single-receiver hardware.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class Fingerprint:
    x: float
    y: float
    label: str
    t: float
    # key -> rssi, only keys with valid readings
    rssi: dict[str, float]
    n_samples: int

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "Fingerprint":
        return Fingerprint(
            x=d["x"], y=d["y"], label=d.get("label", ""), t=d.get("t", 0.0),
            rssi={k: float(v) for k, v in d.get("rssi", {}).items()},
            n_samples=d.get("n_samples", 1),
        )


@dataclass
class RadioMap:
    points: list[Fingerprint]
    keys: list[str]
    meta: dict

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "keys": self.keys,
            "meta": self.meta,
            "points": [p.to_json() for p in self.points],
        }, indent=1))

    @staticmethod
    def load(path: str | Path) -> "RadioMap":
        d = json.loads(Path(path).read_text())
        return RadioMap(
            points=[Fingerprint.from_json(p) for p in d["points"]],
            keys=d["keys"],
            meta=d.get("meta", {}),
        )

    def matrix(self) -> np.ndarray:
        """(P, A) RSSI matrix, NaN where an AP was not heard."""
        m = np.full((len(self.points), len(self.keys)), np.nan)
        for i, p in enumerate(self.points):
            for j, k in enumerate(self.keys):
                if k in p.rssi:
                    m[i, j] = p.rssi[k]
        return m


def _impute(m: np.ndarray) -> np.ndarray:
    """Column-wise forward/backward fill, then column median.

    A missing anchor reading must not be treated as "infinitely far"; filling
    with the observed median keeps distance comparable across cells where a
    given AP was briefly not heard.
    """
    m = m.copy()
    for j in range(m.shape[1]):
        col = m[:, j]
        idx = np.where(~np.isnan(col))[0]
        if idx.size == 0:
            continue
        last = col[idx[0]]
        for i in range(m.shape[0]):
            if np.isnan(m[i, j]):
                m[i, j] = last
            else:
                last = m[i, j]
    return m


def distance(a: np.ndarray, b: np.ndarray, w: np.ndarray | None = None) -> float:
    """Weighted L1 distance in dB between two AP vectors.

    L1 rather than L2 because RSSI differences are heavy-tailed and a single
    badly multipathed anchor should not dominate the match.
    """
    d = np.abs(a - b)
    if not np.all(np.isfinite(d)):
        return float("inf")
    if w is None:
        return float(d.sum())
    return float((d * w).sum() / max(w.sum(), 1e-9))


def pairwise_summary(mat: np.ndarray) -> tuple[float, float]:
    """Closest and mean L1 separation with bounded temporary memory.

    Reduce unordered pairs in blocks rather than allocating a cells by cells
    by anchors tensor. Each pair is evaluated once.
    """
    if mat.ndim != 2 or len(mat) < 2 or not mat.shape[1] or not np.all(np.isfinite(mat)):
        raise ValueError("pairwise separation needs two finite, nonempty fingerprints")
    closest, total, count = float("inf"), 0.0, 0
    for i in range(len(mat) - 1):
        for j in range(i + 1, len(mat), 256):
            ds = np.abs(mat[j:j + 256] - mat[i]).sum(axis=1)
            closest = min(closest, float(ds.min()))
            total += float(ds.sum())
            count += len(ds)
    return closest, total / count


def anchor_weights(mat: np.ndarray) -> np.ndarray:
    """Down-weight anchors whose RSSI fluctuates wildly between survey cells.

    Such anchors are dominated by fast multipath rather than by position, so
    they localise poorly. Weight = 1 / (1 + spread).
    """
    with np.errstate(invalid="ignore"):
        spread = np.nanstd(mat, axis=0)
    spread = np.where(np.isfinite(spread), spread, 10.0)
    return 1.0 / (1.0 + spread)


def localize(rmap: RadioMap, live: dict[str, float],
             top: int = 3) -> list[tuple[Fingerprint, float]]:
    """Rank surveyed cells by distance to a live multi-AP vector."""
    return Matcher(rmap).localize(live, top)


class Matcher:
    """Prepared map for repeated queries, using measured shared anchors only.

    Cell order cannot be used to fill gaps: adjacent rows need not be adjacent
    in space. All candidate cells must use the same observed dimensions. At
    least five anchors and half the map's anchors must support a query.
    """

    def __init__(self, rmap: RadioMap):
        self.points = list(rmap.points)
        self.keys = list(rmap.keys)
        self.mat = rmap.matrix()
        self.shared = np.all(np.isfinite(self.mat), axis=0)

    def localize(self, live: dict[str, float],
                 top: int = 3) -> list[tuple[Fingerprint, float]]:
        if not self.points or top <= 0:
            return []
        lv = np.array([float(live.get(k, np.nan)) for k in self.keys])
        valid = self.shared & np.isfinite(lv)
        if valid.sum() < max(5, int(np.ceil(len(self.keys) / 2))):
            return []
        # Variation between cells is the signal being matched, not an estimate
        # of temporal noise. Do not suppress it with inverse spatial variance.
        ds = np.abs(self.mat[:, valid] - lv[valid]).mean(axis=1)
        order = np.argsort(ds, kind="stable")[:top]
        return [(self.points[i], float(ds[i])) for i in order]


def error_metrics(rmap: RadioMap, truth: list[tuple[float, float]],
                  pred: list[tuple[float, float]]) -> dict:
    """Median and mean localisation error over a sequence of predictions."""
    if not truth or not pred:
        return {}
    errs = []
    for (tx, ty), (px, py) in zip(truth, pred):
        errs.append(float(np.hypot(tx - px, ty - py)))
    a = np.array(errs)
    return {
        "n": len(errs),
        "median_err_m": float(np.median(a)),
        "mean_err_m": float(a.mean()),
        "p90_err_m": float(np.percentile(a, 90)),
        "max_err_m": float(a.max()),
    }
