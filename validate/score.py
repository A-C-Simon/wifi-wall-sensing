#!/usr/bin/env python3
"""Score a scripted walk-test capture against its known phase schedule."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sensing import detect, io, viz  # noqa: E402

OUT = ROOT / "out"


def score_file(cap, threshold: float, win: float, e1: float, w: float,
               s: float, e2: float) -> dict:
    path = Path(cap)
    s_obj = io.load(path)
    r = detect.detect(s_obj, win_s=win, threshold=threshold)
    if not r.detections:
        print("no data to score")
        return {}

    marks = [0.0, e1, e1 + w, e1 + w + s, e1 + w + s + e2]
    names = ["EMPTY", "WALK", "STILL", "EMPTY"]
    # Distinct keys per phase: "EMPTY" occurs twice, and a dict keyed by name
    # would silently merge the two empty phases into one list.
    keys = ["E1", "WALK", "STILL", "E2"]
    label = {"E1": "EMPTY-1", "WALK": "WALK", "STILL": "STILL", "E2": "EMPTY-2"}

    def phase_of(t: float) -> str:
        for i in range(4):
            if marks[i] <= t < marks[i + 1]:
                return keys[i]
        return keys[-1]

    by = {k: [] for k in keys}
    for d in r.detections:
        by[phase_of(d.t)].append(d.motion)

    print(f"{'phase':<9}{'n':>6}{'median':>9}{'p95':>9}{'max':>9}{'alarm%':>9}")
    print("-" * 51)
    for k in keys:
        v = np.array(by[k])
        if not len(v):
            continue
        print(f"{label[k]:<9}{len(v):>6}{np.median(v):>9.2f}"
              f"{np.percentile(v,95):>9.2f}{v.max():>9.2f}"
              f"{100*float(np.mean(v >= threshold)):>8.1f}%")

    walk = np.array(by["WALK"])
    still = np.array(by["STILL"])
    empty = np.concatenate([np.array(by["E1"]), np.array(by["E2"])])
    sd_e = float(np.std(empty)) if len(empty) else 1.0
    sd_e = sd_e if sd_e > 1e-9 else 1.0

    def sig(a, b):
        if not len(a) or not len(b):
            return float("nan")
        return (np.median(a) - np.median(b)) / sd_e

    lat = None
    for d in r.detections:
        if phase_of(d.t) == "WALK" and d.motion >= threshold:
            lat = d.t - marks[1]
            break

    print("\ndiscrimination (median score difference, empty-room sigmas):")
    print(f"  WALK  vs EMPTY : {sig(walk, empty):+.2f} sigma")
    print(f"  STILL vs EMPTY : {sig(still, empty):+.2f} sigma")
    print(f"  WALK  vs STILL : {sig(walk, still):+.2f} sigma")
    print(f"\ndetection latency after walk onset: "
          f"{'%.1f s' % lat if lat is not None else 'NOT DETECTED'}")

    res = {
        "capture": str(path), "threshold": threshold, "marks": marks,
        "phases": {label[k]: {"n": len(by[k]),
                              "median": float(np.median(by[k])) if len(by[k]) else None,
                              "p95": float(np.percentile(by[k], 95)) if len(by[k]) else None,
                              "max": float(np.max(by[k])) if len(by[k]) else None,
                              "alarm_pct": float(100 * np.mean(np.array(by[k]) >= threshold))
                              if len(by[k]) else None}
                   for k in keys},
        "sigma_walk_vs_empty": float(sig(walk, empty)),
        "sigma_still_vs_empty": float(sig(still, empty)),
        "sigma_walk_vs_still": float(sig(walk, still)),
        "detection_latency_s": lat,
        "fine_rate_hz": r.fine_rate_hz,
        "sweep_interval_s": r.sweep_interval_s,
    }
    (OUT / "walktest.json").write_text(json.dumps(res, indent=1))
    try:
        viz.plot_channels(s_obj, str(OUT / "walktest.png"), r)
    except Exception:
        pass
    print(f"wrote {OUT/'walktest.json'}")
    return res


if __name__ == "__main__":
    ap_args = sys.argv[1:]
    if not ap_args:
        print(__doc__)
        sys.exit(2)
    cap = ap_args[0]
    thr = float(ap_args[1]) if len(ap_args) > 1 else 3.0
    win = float(ap_args[2]) if len(ap_args) > 2 else 10.0
    e = float(ap_args[3]) if len(ap_args) > 3 else 45.0
    w = float(ap_args[4]) if len(ap_args) > 4 else 45.0
    stl = float(ap_args[5]) if len(ap_args) > 5 else 45.0
    score_file(cap, thr, win, e, w, stl, e)