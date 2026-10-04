#!/usr/bin/env python3
"""Scripted validation: does RSSI-only sensing actually detect a moving person?

Protocol (ground truth is the operator's own scripted schedule, so the result is
falsifiable rather than anecdotal):

  phase 0  EMPTY   room quiet for --empty seconds      -> should NOT alarm
  phase 1  WALK    operator crosses the sensing area    -> should alarm
  phase 2  STILL   operator stands motionless           -> should decay
  phase 3  EMPTY   operator has left the area          -> should NOT alarm

Reports detection latency, false-alarm rate and score separation using the
threshold calibrated from an empty-room baseline.

Usage: python3 validate/walk_test.py --empty 60 --walk 40 --still 40
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sensing import detect, io  # noqa: E402

OUT = ROOT / "out"
CORE = ROOT / "native" / "wifisense_core"


def countdown(sec: int, msg: str) -> None:
    print(f"\n>>> {msg}")
    for r in range(sec, 0, -1):
        print(f"    {r:3d}s  ", end="\r", flush=True)
        time.sleep(1)
    print("    go. ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="en0")
    ap.add_argument("--empty", type=int, default=60, help="seconds of empty room at start")
    ap.add_argument("--walk", type=int, default=40, help="seconds of walking")
    ap.add_argument("--still", type=int, default=40, help="seconds standing still")
    ap.add_argument("--empty2", type=int, default=60, help="seconds empty at end")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--out", default=str(OUT / "walktest.jsonl"))
    args = ap.parse_args()

    if not CORE.exists():
        subprocess.run(["swiftc", "-O", "-framework", "CoreWLAN", "-o", str(CORE),
                        str(ROOT / "native" / "wifisense_core.swift")], check=True)

    out = Path(args.out)
    total = args.empty + args.walk + args.still + args.empty2
    bp = OUT / "baseline.json"
    thr = args.threshold
    if thr is None and bp.exists():
        thr = json.loads(bp.read_text()).get("threshold")
    print(f"total {total}s, threshold = "
          f"{'calibrated ' + format(thr, '.3f') if thr else 'AUTO (no baseline found)'}")

    proc = subprocess.Popen([str(CORE), "--iface", args.iface, "--out", str(out),
                             "--duration", str(total), "--fine-hz", "20", "--quiet"])

    # Phase schedule in capture-relative seconds.
    marks = [0.0, float(args.empty), float(args.empty + args.walk),
             float(args.empty + args.walk + args.still), float(total)]
    names = ["EMPTY", "WALK", "STILL", "EMPTY"]

    time.sleep(1.0)
    for i in range(1, 4):
        if i == 1:
            countdown(args.empty, "PHASE 1 / WALK: walk through the area between "
                                  "the access point and the laptop, back and forth.")
        elif i == 2:
            countdown(args.walk, "PHASE 2 / STILL: stand still in the middle of "
                                 "the area, do not move.")
        else:
            countdown(args.still, "PHASE 3 / EMPTY: walk out of the sensing area "
                                  "and stay away.")
    proc.wait()

    s = io.load(out)
    r = detect.detect(s, threshold=thr)
    if not r.detections:
        print("no data captured")
        return 1

    # Score each detection with the phase it falls in.
    def phase_of(t: float) -> str:
        for i in range(4):
            if marks[i] <= t < marks[i + 1]:
                return names[i]
        return names[-1]

    rows = [(d.t, d.motion, phase_of(d.t)) for d in r.detections]
    by_phase: dict[str, list[float]] = {n: [] for n in names}
    for _t, m, p in rows:
        by_phase[p].append(m)

    print(f"\n{'phase':<8}{'n':>6}{'median':>9}{'p95':>9}{'max':>9}{'alarm%':>9}")
    print("-" * 50)
    for n in names:
        v = np.array(by_phase[n]) if by_phase[n] else np.zeros(0)
        if len(v) == 0:
            continue
        alarm = float(np.mean(v >= r.threshold)) * 100 if r.threshold is not None else float("nan")
        print(f"{n:<8}{len(v):>6}{np.median(v):>9.2f}{np.percentile(v,95):>9.2f}"
              f"{v.max():>9.2f}{alarm:>8.1f}%")

    walk = np.array(by_phase["WALK"])
    still = np.array(by_phase["STILL"])
    empty = np.concatenate([by_phase["EMPTY"]])

    # Separation: how many robust sigmas of empty-room spread separate walking
    # from standing still. This is the number that decides whether the system
    # has any real discriminating power.
    def sep(a, b):
        if len(a) == 0 or len(b) == 0:
            return float("nan")
        return (np.median(a) - np.median(b)) / (np.std(empty) + 1e-9)

    print("\ndiscrimination (median score difference in empty-room sigmas):")
    print(f"  WALK  vs EMPTY : {sep(walk, empty):+.2f} sigma")
    print(f"  STILL vs EMPTY : {sep(still, empty):+.2f} sigma")
    print(f"  WALK  vs STILL : {sep(walk, still):+.2f} sigma")

    # Detection latency: first alarm after the walk phase begins.
    lat = None
    if r.threshold is not None:
        for t, m, p in rows:
            if p == "WALK" and m >= r.threshold:
                lat = t - marks[1]
                break
    print(f"\ndetection latency after walk onset: "
          f"{'%.1f s' % lat if lat is not None else 'NOT DETECTED'}")

    summary = {
        "threshold": r.threshold,
        "marks": marks,
        "phases": {n: {"n": len(by_phase[n]),
                       "median": float(np.median(by_phase[n])) if by_phase[n] else None,
                       "p95": float(np.percentile(by_phase[n], 95)) if by_phase[n] else None,
                       "max": float(np.max(by_phase[n])) if by_phase[n] else None}
                   for n in names},
        "sigma_walk_vs_empty": float(sep(walk, empty)),
        "sigma_still_vs_empty": float(sep(still, empty)),
        "sigma_walk_vs_still": float(sep(walk, still)),
        "detection_latency_s": lat,
        "fine_rate_hz": r.fine_rate_hz,
        "sweep_interval_s": r.sweep_interval_s,
        "capture": str(out),
    }
    (OUT / "walktest.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {OUT/'walktest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())