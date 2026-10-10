"""Feature extraction tuned to what RSSI on this hardware can actually deliver.

Measured envelope driving these choices (see README):
  * associated-AP RSSI is quantised to 1 dB and its register transitions only
    ~0.2 times per second, so instantaneous values carry almost no information;
  * the multi-AP vector refreshes at most once per ~12 s (a global macOS limit).

Therefore the usable features are *temporal aggregates* over windows measured
in many seconds, never per-sample values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Window:
    t: float
    # fine channel
    rssi_mean: float
    rssi_sd: float        # within-bucket sd, averaged over the window
    change_rate: float    # register transitions per second
    span: float           # max-min of rssi_mean across the window
    # sweep channel
    l1: float             # sum |dRSSI| across APs since previous fresh sweep
    l2: float
    n_ap_changed: int
    n_ap: int


def _sliding(a: np.ndarray, w: int, fn) -> np.ndarray:
    """Centred sliding-window reduce with edge padding."""
    if a.size == 0:
        return np.zeros(0)
    w = max(1, min(w, a.size))
    pad = w // 2
    ap = np.pad(a.astype(float), (pad, pad), mode="edge")
    return np.array([fn(ap[i:i + w]) for i in range(a.size)], dtype=float)


def fine_features(t: np.ndarray, rssi: np.ndarray, sd: np.ndarray,
                  chg: np.ndarray, win_s: float, dt: float) -> np.ndarray:
    """Causal aggregates over the preceding `win_s` seconds.

    Returns an (N, 3) array of [mean rssi, mean within-bucket sd,
    transitions per second]. The window mean is retained because slow AGC drift
    is itself the dominant nuisance term and must be visible for detrending.
    """
    n = len(t)
    if n == 0:
        return np.zeros((0, 3))
    if not np.isfinite(win_s) or win_s <= 0:
        raise ValueError("win_s must be finite and positive")
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("fine timestamps must be finite and strictly increasing")
    if any(len(a) != n for a in (rssi, sd, chg)):
        raise ValueError("fine arrays must have matching lengths")
    # Prefix sums avoid reducing an overlapping window for every sample.
    # Actual timestamps keep a delayed bucket from stretching the window.
    left = np.searchsorted(t, t - win_s, side="right")
    right = np.arange(1, n + 1)
    counts = right - left
    out = np.zeros((n, 3))
    for j, values in enumerate((rssi, sd, chg)):
        cs = np.r_[0.0, np.cumsum(values, dtype=float)]
        out[:, j] = cs[right] - cs[left]
    out[:, :2] /= counts[:, None]
    # A startup window contains less than win_s seconds of observations.
    bucket_dt = max(float(dt), 1e-6)
    starts = np.r_[t[0] - bucket_dt, t[:-1]]
    elapsed = np.maximum(np.minimum(win_s, t - starts[left]), 1e-6)
    out[:, 2] /= elapsed
    return out


def detrend(x: np.ndarray, t: np.ndarray, half_life_s: float = 120.0) -> np.ndarray:
    """Subtract a slow exponentially-weighted baseline.

    RSSI drift from AGC and temperature is many times larger than the motion
    signature. Removing a slow baseline leaves only the fast residual, which is
    what correlates with a moving reflector. Implemented as a one-pole IIR so it
    is O(N) and needs no look-ahead (causal: usable in real time).
    """
    if x.size == 0:
        return x
    steps = np.r_[0.0, np.diff(t)]
    if len(t) != x.size or np.any(steps < 0):
        raise ValueError("detrend requires matching, increasing timestamps")
    alpha = -np.expm1(-math.log(2.0) * steps / max(half_life_s, 1e-6))
    out = np.empty_like(x, dtype=float)
    base = float(x[0])
    for i in range(x.size):
        base += alpha[i] * (float(x[i]) - base)
        out[i] = float(x[i]) - base
    return out


def envelope(x: np.ndarray) -> np.ndarray:
    """One-pole rectification: turns a signed residual into a magnitude."""
    if x.size == 0:
        return x
    a = float(np.abs(x[0]))
    out = np.empty(x.size, dtype=float)
    for i in range(x.size):
        a += 0.25 * (abs(float(x[i])) - a)
        out[i] = a
    return out


def cumsum_score(x: np.ndarray, drift: float = 0.0) -> np.ndarray:
    """Two-sided CUSUM: integrates a score with allowance for drift.

    A slow positive leak `drift` prevents accumulation from stalling, so the
    statistic can reset after a genuine absence period.
    """
    s = 0.0
    out = np.zeros(x.size, dtype=float)
    for i in range(x.size):
        v = float(x[i]) - drift
        s = max(0.0, s + v)
        out[i] = s
    return out


def _fill_columns(m: np.ndarray) -> np.ndarray:
    """Column-wise forward then backward fill, then median for leftovers.

    An AP missing from one sweep is not evidence of motion, so its last known
    value must be carried forward. Both directions are needed: the first sweep
    often has gaps, and leaving those unfilled seeds NaN into every subsequent
    difference.
    """
    m = m.copy()
    s, a = m.shape
    for j in range(a):
        col = m[:, j]
        idx = np.where(~np.isnan(col))[0]
        if idx.size == 0:
            continue
        # forward fill
        last = col[idx[0]]
        for i in range(s):
            if np.isnan(m[i, j]):
                m[i, j] = last
            else:
                last = m[i, j]
        # backward fill anything still missing (leading gaps)
        nxt = col[idx[-1]]
        for i in range(s - 1, -1, -1):
            if np.isnan(m[i, j]):
                m[i, j] = nxt
            else:
                nxt = m[i, j]
    # anything still NaN (all-NaN column) -> 0 so it contributes nothing
    bad = np.isnan(m)
    if bad.any():
        m[bad] = 0.0
    return m


def sweep_features(sweep_t: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Frame-to-frame change of the multi-AP RSSI matrix.

    matrix is (S, A) with NaN where an AP was absent. Result is (S-1, 3):
    L1 change, L2 change, and the number of APs that moved by >= 1 dB.
    """
    s = matrix.shape[0]
    if s < 2:
        return np.zeros((0, 3))
    filled = _fill_columns(matrix)
    d = np.diff(filled, axis=0)
    l1 = np.abs(d).sum(axis=1)
    l2 = np.sqrt((d ** 2).sum(axis=1))
    changed = (np.abs(d) >= 1.0).sum(axis=1)
    return np.column_stack([l1, l2, changed.astype(float)])


def hold_sweeps(t: np.ndarray, sweep_t: np.ndarray, values: np.ndarray,
                max_age: float) -> np.ndarray:
    """Hold each sweep statistic from its timestamp until it expires.

    Before the first completed interval there is no evidence. At an exact
    timestamp the new statistic applies, never the previous one.
    """
    out = np.zeros(len(t), dtype=float)
    if not len(sweep_t):
        return out
    idx = np.searchsorted(sweep_t, t, side="right") - 1
    safe = np.maximum(idx, 0)
    valid = (idx >= 0) & (t - sweep_t[safe] < max_age)
    out[valid] = values[safe[valid]]
    return out


def zscore_robust(x: np.ndarray) -> np.ndarray:
    """Median/MAD standardisation, NaN-safe.

    Safe with the tiny, tied value sets that 1 dB quantised RSSI produces
    (stdev would be degenerate), and preserves NaN positions rather than
    letting one gap contaminate the whole series through the median.
    """
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    v = x[finite]
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    scale = mad if mad > 1e-9 else float(np.std(v))
    if scale <= 1e-9:
        scale = 1.0
    out = np.zeros_like(x)
    out[finite] = (v - med) / scale
    out[~finite] = 0.0
    return out


def ratio_score(x: np.ndarray, gain: float = 3.0) -> np.ndarray:
    """Normalise a positive series by its own median, in units of `gain`.

    Used instead of a z-score for frame-to-frame change statistics. A change
    magnitude is physically stationary (the same multipath activity produces the
    same amount of frame-to-frame movement whether or not anyone is present),
    whereas its *mean level* drifts with AGC and temperature. Dividing by the
    median therefore yields a statistic that does not drift, while a z-score
    would slowly inflate a quiet stretch into looking anomalous.
    """
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    if x.size == 0:
        return x
    med = float(np.median(x))
    if med <= 1e-9:
        med = float(np.mean(x))
    if med <= 1e-9:
        return np.zeros_like(x)
    return (x / med - 1.0) * gain


def auto_threshold(x: np.ndarray, k: float = 6.0) -> float:
    """Robust noise-floor estimate for thresholding a score series.

    Uses the upper half of the distribution so that a trace which is mostly
    occupied still yields a threshold above the quiet baseline.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad <= 1e-9:
        mad = float(np.std(x))
    if mad <= 1e-9:
        return med + k
    return med + k * 1.4826 * mad
