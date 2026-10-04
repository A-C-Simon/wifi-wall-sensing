#!/usr/bin/env python3
"""Occupancy A/B test with a real effect size.

Earlier tests compared medians and that was not enough: a difference in medians
says nothing about whether the two distributions overlap. If an occupied and an
empty window have the same spread and are merely offset, or worse, if they
overlap almost completely, no threshold will ever separate them.

So this reports the quantity that actually decides usability:

    AUC = P(score from occupied window > score from empty window)

AUC 0.5 means the score carries no information about occupancy. AUC 0.5 to 0.7
is unusable in practice. Only above ~0.8 does a threshold start working. A
bootstrap confidence interval and a permutation p-value are included because
with a dozen windows per condition, a raw AUC difference is easy to overread.

Protocol is A/B/A/B/A/B with long dwells: the measured session drift was large
enough that short blocks are dominated by it rather than by occupancy.

Usage: python3 validate/occupancy_test.py --block 90 --reps 3
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

from sensing import detect, io, viz  # noqa: E402

OUT = ROOT / "out"
CORE = ROOT / "native" / "wifisense_core"


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(random positive > random negative), ties counted as half."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    uniq, inv = np.unique(allv, return_inverse=True)
    for u in uniq:
        m = inv == u
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    r_pos = ranks[: len(pos)].sum()
    n1, n0 = len(pos), len(neg)
    return float((r_pos - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_ci(pos, neg, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.empty(n)
    for i in range(n):
        vals[i] = auc(rng.choice(pos, len(pos), replace=True),
                      rng.choice(neg, len(neg), replace=True))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def perm_p(pos, neg, n=4000, seed=0):
    """Two sided permutation test on the difference of means."""
    rng = np.random.default_rng(seed)
    obs = abs(pos.mean() - neg.mean())
    pool = np.concatenate([pos, neg])
    k = len(pos)
    cnt = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(pool[:k].mean() - pool[k:].mean()) >= obs:
            cnt += 1
    return float((cnt + 1) / (n + 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="en0")
    ap.add_argument("--block", type=float, default=90.0, help="seconds per block")
    ap.add_argument("--reps", type=int, default=3, help="A/B pairs")
    ap.add_argument("--lead", type=float, default=20.0)
    ap.add_argument("--threshold", type=float, default=3.0)
    ap.add_argument("--win", type=float, default=10.0)
    ap.add_argument("--out", default=str(OUT / "occupancy.jsonl"))
    args = ap.parse_args()

    if not CORE.exists():
        subprocess.run(["swiftc", "-O", "-framework", "CoreWLAN", "-o", str(CORE),
                        str(ROOT / "native" / "wifisense_core.swift")], check=True)

    # schedule: A(empty) B(occupied) alternating, starting empty
    blocks = []
    for r in range(args.reps):
        blocks.append(("EMPTY", args.block))
        blocks.append(("OCCUPIED", args.block))
    total = args.lead + sum(b[1] for b in blocks)
    t0 = time.time()
    marks = [0.0, args.lead]
    for _n, d in blocks:
        marks.append(marks[-1] + d)

    print(f"occupancy A/B test: {args.reps} pairs x {args.block:.0f}s "
          f"= {total/60:.1f} min")
    print(f"capture starts {time.strftime('%H:%M:%S')}\n")
    for i, ((name, d), start) in enumerate(zip(blocks, marks[1:])):
        a = time.strftime("%H:%M:%S", time.localtime(t0 + start))
        b = time.strftime("%H:%M:%S", time.localtime(t0 + start + d))
        print(f"  block {i+1:2d}  {name:9s}  {a} -> {b}")
    print(f"\n  OCCUPIED means: stand in the sensing area and stay still.")
    print(f"  EMPTY means: be outside the room.\n")

    out = Path(args.out)
    if out.exists():
        out.unlink()
    proc = subprocess.Popen([str(CORE), "--iface", args.iface, "--out", str(out),
                             "--duration", str(int(total)), "--fine-hz", "20",
                             "--quiet"])
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()

    s = io.load(out)
    r = detect.detect(s, win_s=args.win, threshold=args.threshold)
    if not r.detections:
        print("no data")
        return 1

    def which(t):
        for i, (name, _d) in enumerate(blocks):
            if marks[i + 1] <= t < marks[i + 2]:
                return name
        return None

    pos, neg = [], []
    per_block = []
    for d in r.detections:
        n = which(d.t)
        if n == "OCCUPIED":
            pos.append(d.motion)
        elif n == "EMPTY":
            neg.append(d.motion)
    pos, neg = np.array(pos), np.array(neg)

    # per-block medians show whether the effect is stable or driven by one window
    for i, (name, _d) in enumerate(blocks):
        sel = [dd.motion for dd in r.detections
               if marks[i + 1] <= dd.t < marks[i + 2]]
        if sel:
            per_block.append((name, float(np.median(sel)), float(np.mean(sel)),
                              len(sel)))

    print(f"\n{'block':>5} {'cond':>9} {'median':>8} {'mean':>8} {'n':>6}")
    print("-" * 40)
    for i, (name, med, mean, n) in enumerate(per_block):
        print(f"{i+1:>5} {name:>9} {med:>8.2f} {mean:>8.2f} {n:>6}")

    if len(pos) < 5 or len(neg) < 5:
        print("not enough data")
        return 1

    a = auc(pos, neg)
    lo, hi = boot_ci(pos, neg)
    p = perm_p(pos, neg)
    print(f"\noccupied n={len(pos)}  empty n={len(neg)}")
    print(f"  AUC           : {a:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  permutation p : {p:.4f}")
    print(f"  median occ    : {np.median(pos):+.3f}")
    print(f"  median empty  : {np.median(neg):+.3f}")
    ov = max(0.0, min(pos.max(), neg.max()) - max(pos.min(), neg.min()))
    print(f"  range overlap : {ov:.2f} "
          f"(occ {pos.min():+.2f}..{pos.max():+.2f}, "
          f"empty {neg.min():+.2f}..{neg.max():+.2f})")

    print()
    if lo < 0.5:
        verdict = "NO SIGNAL - AUC CI includes 0.5"
    elif a < 0.7:
        verdict = "UNUSABLE - distributions overlap too much to threshold"
    elif a < 0.8:
        verdict = "WEAK - only large excursions would separate"
    else:
        verdict = "USABLE - a threshold can separate the conditions"
    print(f"  verdict: {verdict}")

    res = {"auc": a, "auc_ci": [lo, hi], "perm_p": p,
           "n_occupied": len(pos), "n_empty": len(neg),
           "median_occupied": float(np.median(pos)),
           "median_empty": float(np.median(neg)),
           "verdict": verdict, "block_s": args.block, "reps": args.reps,
           "threshold": args.threshold, "capture": str(out)}
    (OUT / "occupancy.json").write_text(json.dumps(res, indent=1))
    try:
        viz.plot_channels(s, str(OUT / "occupancy.png"), r)
    except Exception:
        pass
    print(f"\nwrote {OUT/'occupancy.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())