"""Session I/O for wifisense_core JSONL captures."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Fine:
    """One bucket of the fast single-link channel (associated AP)."""
    t: float
    n: int
    rssi: float
    mn: float
    mx: float
    sd: float
    chg: float
    noise: float
    rate: float


@dataclass
class Sweep:
    """One fresh multi-AP scan. APs act as spatially distributed anchors."""
    t: float
    cost_ms: float
    # key -> rssi. Key is "ch/width"; BSSID is TCC-redacted on macOS.
    aps: dict[str, int] = field(default_factory=dict)

    def vector(self, keys: list[str]) -> list[float]:
        return [float(self.aps.get(k, float("nan"))) for k in keys]


@dataclass
class Session:
    meta: dict = field(default_factory=dict)
    fine: list[Fine] = field(default_factory=list)
    sweeps: list[Sweep] = field(default_factory=list)
    stats: list[dict] = field(default_factory=list)

    @property
    def duration(self) -> float:
        ts = [f.t for f in self.fine] + [s.t for s in self.sweeps]
        return max(ts) if ts else 0.0

    def ap_keys(self, min_count: int = 1) -> list[str]:
        """AP identities seen in at least `min_count` sweeps.

        Using a coverage threshold keeps one-off noise APs out of the
        fingerprint vector, which otherwise just adds NaN dimensions.
        """
        counts: dict[str, int] = {}
        for s in self.sweeps:
            for k in s.aps:
                counts[k] = counts.get(k, 0) + 1
        return sorted(k for k, c in counts.items() if c >= min_count)


def _key(ch: int, w: int) -> str:
    return f"{ch}/{w}"


def load(path: str | Path, dedupe_sweeps: bool = True) -> Session:
    """Parse a capture file.

    macOS often returns a second, near-identical scan result ~0.2s after a real
    sweep (the cache is repopulated with a slightly different AP count). Those
    are not independent samples, so by default keep only the first vector of
    each burst, separated by at least `burst_gap` seconds.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    s = Session()
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = o.get("type")
            if t == "meta":
                s.meta = o
            elif t == "fine":
                s.fine.append(Fine(
                    t=o["t"], n=o["n"], rssi=o["rssi"], mn=o["min"], mx=o["max"],
                    sd=o["sd"], chg=o["chg"], noise=o["noise"], rate=o["rate"]))
            elif t == "sweep":
                aps = {_key(a["ch"], a["w"]): a["rssi"] for a in o["aps"]}
                s.sweeps.append(Sweep(t=o["t"], cost_ms=o.get("cost_ms", 0.0), aps=aps))
            elif t == "stats":
                s.stats.append(o)

    if dedupe_sweeps:
        burst_gap = 1.0
        kept: list[Sweep] = []
        for sw in s.sweeps:
            if kept and (sw.t - kept[-1].t) < burst_gap:
                # Keep whichever saw more APs; it is the more complete sweep.
                if sw.aps and len(sw.aps) > len(kept[-1].aps):
                    kept[-1] = sw
                continue
            kept.append(sw)
        s.sweeps = kept
    return s


def median(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    n = len(ys)
    return ys[n // 2] if n % 2 else 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def mad(xs: list[float]) -> float:
    """Median absolute deviation -- robust spread, unlike stdev on 2-level data."""
    if len(xs) < 2:
        return 0.0
    m = median(xs)
    return median([abs(x - m) for x in xs])


def stdev(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))