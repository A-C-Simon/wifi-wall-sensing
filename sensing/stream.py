"""Bounded state for the causal fine-channel filter used by live displays."""
from __future__ import annotations

from collections import deque
import math


class FineEnergy:
    """Update the rolling mean, detrended residual and envelope once per bucket.

    The plot buffer can discard old samples without resetting either IIR.
    Only the raw samples inside the current averaging window are retained.
    """

    def __init__(self, win_s: float):
        if not math.isfinite(win_s) or win_s <= 0:
            raise ValueError("win_s must be finite and positive")
        self.win_s = win_s
        self.half_life = max(120., 3 * win_s)
        self.window = deque()
        self.total = 0.
        self.base = None
        self.energy = 0.
        self.previous_t = None

    def update(self, t: float, rssi: float) -> float:
        if not math.isfinite(t) or not math.isfinite(rssi):
            raise ValueError("fine timestamp and RSSI must be finite")
        if self.previous_t is not None and t <= self.previous_t:
            raise ValueError("fine timestamps must be strictly increasing")
        self.window.append((t, rssi))
        self.total += rssi
        while self.window[0][0] <= t - self.win_s:
            self.total -= self.window.popleft()[1]
        mean = self.total / len(self.window)
        if self.base is None:
            self.base = mean
        else:
            alpha = -math.expm1(-math.log(2.) * (t - self.previous_t) / self.half_life)
            self.base += alpha * (mean - self.base)
        self.energy += .25 * (abs(mean - self.base) - self.energy)
        self.previous_t = t
        return self.energy
