"""Log-distance link model: converting RSSI to range.

Absolute distance is *not* obtainable from RSSI alone. The received power obeys

    RSSI(d) = A - 10 * n * log10(d)

where A folds in transmit power, antenna gains and cable loss, and n is the
environment's path-loss exponent. Two unknowns, one observable, so a single
RSSI reading gives a whole family of (A, n, d) solutions. Anyone quoting a
distance from one RSSI sample without stating their assumed model is guessing.

This module makes the assumption explicit and fittable: stand at several
*known* distances, record RSSI, and least-squares fit A and n. The fitted model
then inverts to place anchors by range.

Typical n: ~2.0 free space, 2.5-3.0 same-room with furniture, 3.5-4.0 through
one plaster wall, 4.5-6.0 through reinforced concrete.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LinkModel:
    A: float = -30.0     # RSSI (dBm) at 1 m
    n: float = 2.7       # path-loss exponent
    samples: int = 0
    note: str = ""

    def __post_init__(self):
        if not math.isfinite(self.A) or not math.isfinite(self.n) or self.n <= 0:
            raise ValueError("link model needs finite A and a positive finite path-loss exponent")

    def rssi_at(self, d: float) -> float:
        if not math.isfinite(d) or d <= 0:
            raise ValueError("distance must be finite and positive")
        return self.A - 10.0 * self.n * math.log10(d)

    def range_of(self, rssi: float) -> float:
        """Invert the model. Ill-conditioned when few samples were fitted:
        both A and n trade off against each other, so the result is only as
        good as the survey that produced it."""
        if not math.isfinite(rssi):
            raise ValueError("RSSI must be finite")
        arg = (self.A - rssi) / (10.0 * self.n)
        # Extreme extrapolation may exceed floating-point range. Returning
        # infinity lets a caller cap its display without an overflow crash.
        return float(max(10.0 ** arg, 0.05)) if arg <= 308 else float("inf")

    def fit(self, dists: list[float], rssis: list[float]) -> "LinkModel":
        """Least-squares fit of RSSI = A - 10 n log10(d)."""
        if len(dists) != len(rssis):
            raise ValueError("distance and RSSI counts must match")
        pts = [(d, r) for d, r in zip(dists, rssis)
               if math.isfinite(d) and math.isfinite(r) and d > 0 and r < 0]
        if len(pts) < 2:
            raise ValueError("need at least 2 (distance, rssi) pairs")
        xs = [math.log10(d) for d, _ in pts]
        ys = [r for _, r in pts]
        n_ = len(xs)
        mx = sum(xs) / n_
        my = sum(ys) / n_
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        if sxx <= 1e-12:
            raise ValueError("all distances identical - move between points")
        slope = sxy / sxx                 # dB per decade == -10 n
        a = my - slope * mx
        exponent = -slope / 10.0
        if not math.isfinite(a) or not math.isfinite(exponent) or exponent <= 1e-9:
            raise ValueError("RSSI must decrease with distance; cannot fit a positive path-loss exponent")
        self.A = a
        self.n = exponent
        self.samples = n_
        resid = [y - (a + slope * x) for x, y in zip(xs, ys)]
        rmse = math.sqrt(sum(r * r for r in resid) / n_)
        self.note = f"rmse {rmse:.2f} dB over {n_} points"
        return self

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(
            {"A": self.A, "n": self.n, "samples": self.samples, "note": self.note},
            indent=1))

    @staticmethod
    def load(path: str | Path) -> "LinkModel":
        d = json.loads(Path(path).read_text())
        return LinkModel(A=d["A"], n=d["n"], samples=d.get("samples", 0),
                         note=d.get("note", ""))


def estimate_uncertainty(model: LinkModel, rssi: float,
                         n_range=(2.0, 4.0)) -> tuple[float, float]:
    """Range of distances consistent with one RSSI reading, sweeping the
    plausible path-loss exponent. This is the honest error bar to show."""
    if not n_range:
        raise ValueError("need at least one path-loss exponent")
    distances = [LinkModel(A=model.A, n=n).range_of(rssi) for n in n_range]
    lo, hi = min(distances), max(distances)
    return float(min(lo, hi)), float(max(lo, hi))
