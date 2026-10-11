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

from common import finite_float, positive_int

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
    ap.add_argument("--empty", type=positive_int, default=60, help="seconds of empty room at start")
    ap.add_argument("--walk", type=positive_int, default=40, help="seconds of walking")
    ap.add_argument("--still", type=positive_int, default=40, help="seconds standing still")
    ap.add_argument("--empty2", type=positive_int, default=60, help="seconds empty at end")
    ap.add_argument("--threshold", type=finite_float, default=None)
    ap.add_argument("--out", default=str(OUT / "walktest.jsonl"))
    args = ap.parse_args()

    from common import build_core
    build_core()

    out = Path(args.out)
    total = args.empty + args.walk + args.still + args.empty2
    bp = OUT / "baseline.json"
    thr = args.threshold
    calibration = None
    if thr is None and bp.exists():
        calibration = json.loads(bp.read_text())
        detect.validate_calibration(calibration, 10.0)
        thr = calibration["threshold"]
    print(f"total {total}s, threshold = "
          f"{'calibrated ' + format(thr, '.3f') if thr else 'AUTO (no baseline found)'}")

    if out.exists():
        out.unlink()
    proc = subprocess.Popen([str(CORE), "--iface", args.iface, "--out", str(out),
                             "--duration", str(total), "--fine-hz", "20", "--quiet"])

    # Read the native capture epoch before prompting. All deadlines refer to
    # that epoch, so subprocess startup and countdown overhead cannot shift
    # labels by an entire phase.
    from validate.score import score_file
    started = time.monotonic()
    try:
        deadline = started + 10
        while time.monotonic() < deadline:
            if out.exists():
                meta = io.load(out).meta
                if "started_at" in meta:
                    started = time.monotonic() - (time.time() - meta["started_at"])
                    break
            if proc.poll() is not None:
                raise RuntimeError("capture exited before publishing metadata")
            time.sleep(.05)
        else:
            raise RuntimeError("capture did not publish metadata")
        elapsed = 0.0
        phases = [(args.empty, "EMPTY: leave the sensing area"),
                  (args.walk, "WALK: walk back and forth across the area"),
                  (args.still, "STILL: stand still in the area"),
                  (args.empty2, "EMPTY: leave the sensing area")]
        for duration, prompt in phases:
            print(f"\n>>> {prompt}", flush=True)
            elapsed += duration
            while time.monotonic() - started < elapsed and proc.poll() is None:
                left = elapsed - (time.monotonic() - started)
                print(f"    {left:5.1f}s  ", end="\r", flush=True)
                time.sleep(.1)
        if proc.wait() != 0:
            raise RuntimeError("capture failed")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    score_file(out, thr, 10.0, args.empty, args.walk, args.still, args.empty2,
               calibration=calibration)

    return 0


if __name__ == "__main__":
    sys.exit(main())