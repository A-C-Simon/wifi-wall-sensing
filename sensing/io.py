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


def cell_capture_path(root: str | Path, x: float, y: float, *, for_write=False) -> Path:
    """Use round-trip coordinates; read old one-decimal filenames if needed."""
    root = Path(root)
    path = root / f"cell_{float(x)}_{float(y)}.jsonl"
    if not for_write and not path.exists():
        legacy = root / f"cell_{x:.1f}_{y:.1f}.jsonl"
        if legacy.exists():
            return legacy
    return path


def _number(value, name: str) -> float:
    try:
        valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite number")
    return value


def validate_record(record) -> dict:
    """Check a decoded capture before it can alter a filter or time axis."""
    if not isinstance(record, dict):
        raise ValueError("capture record must be an object")
    kind = record.get("type")
    if kind in ("fine", "sweep", "stats"):
        if _number(record.get("t"), "t") < 0:
            raise ValueError("capture timestamp must be nonnegative")
    if kind == "meta" and "started_at" in record:
        _number(record["started_at"], "started_at")
    if kind == "fine":
        for key in ("n", "rssi", "min", "max", "sd", "chg", "noise", "rate"):
            _number(record.get(key), key)
        if record["n"] <= 0 or int(record["n"]) != record["n"]:
            raise ValueError("fine sample count must be a positive integer")
        if not record["min"] <= record["rssi"] <= record["max"] < 0:
            raise ValueError("fine RSSI must lie within negative min/max bounds")
        if record["sd"] < 0 or record["rate"] < 0:
            raise ValueError("fine spread and transmit rate must be nonnegative")
        if not 0 <= record["chg"] <= record["n"] or int(record["chg"]) != record["chg"]:
            raise ValueError("fine changes must be an integer between zero and sample count")
    if kind == "sweep":
        cost = _number(record.get("cost_ms", 0), "cost_ms")
        if cost < 0:
            raise ValueError("scan cost must be nonnegative")
        scan_aps(record.get("aps"))
    return record


def scan_aps(rows: list[dict]) -> dict[str, int]:
    """Collapse redacted co-channel APs by strongest RSSI, independent of order."""
    if not isinstance(rows, list):
        raise ValueError("scan anchors must be a list")
    aps = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("scan anchor must be an object")
        for key in ("ch", "w", "rssi"):
            value = _number(row.get(key), key)
            if int(value) != value:
                raise ValueError(f"scan {key} must be an integer")
        key = _key(row["ch"], row["w"])
        value = row["rssi"]
        if math.isfinite(value) and value < 0:
            aps[key] = max(aps.get(key, value), value)
    return aps


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
        for line_no, line in enumerate(fh, 1):
            complete = line.endswith("\n")
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                validate_record(o)
            except (json.JSONDecodeError, ValueError) as e:
                # A stopped writer can leave an incomplete final JSON record.
                # Complete corrupt records must not silently change results.
                if not complete and isinstance(e, json.JSONDecodeError):
                    continue
                raise ValueError(f"{p}:{line_no}: {e}") from e
            t = o.get("type")
            if t == "meta":
                s.meta = o
            elif t == "fine":
                s.fine.append(Fine(
                    t=o["t"], n=o["n"], rssi=o["rssi"], mn=o["min"], mx=o["max"],
                    sd=o["sd"], chg=o["chg"], noise=o["noise"], rate=o["rate"]))
            elif t == "sweep":
                aps = scan_aps(o["aps"])
                s.sweeps.append(Sweep(t=o["t"], cost_ms=o.get("cost_ms", 0.0), aps=aps))
            elif t == "stats":
                s.stats.append(o)

    # Concurrent workers can finish writes out of timestamp order. Keep the
    # time axes monotonic before differencing or resampling them.
    s.fine.sort(key=lambda f: f.t)
    if any(a.t == b.t for a, b in zip(s.fine, s.fine[1:])):
        raise ValueError(f"{p}: duplicate fine timestamps")
    s.sweeps.sort(key=lambda sw: sw.t)
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
