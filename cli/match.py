"""Match a capture against a surveyed radio map."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sensing import detect, io, radiomap, viz

from common import CORE, OUT, ROOT, build_core


def cmd_localize(args) -> int:
    """Attribute each fresh sweep to the surveyed cell it best matches."""
    mp = Path(args.radiomap)
    if not mp.exists():
        print(f"no radio map at {mp}; run `wifisense calibrate` first")
        return 1
    rm = radiomap.RadioMap.load(mp)
    if rm.meta.get("verdict") != "usable":
        print("localisation disabled: map must pass validate-map first")
        return 2
    matcher = radiomap.Matcher(rm)
    s = io.load(args.capture)
    if len(s.sweeps) < 2:
        print("capture has too few fresh sweeps to localise")
        return 1

    truth = []
    if args.truth:
        truth = [tuple(map(float, ln.split(",")))
                 for ln in Path(args.truth).read_text().split() if ln.strip()]
        if (len(truth) != len(s.sweeps)
                or any(len(p) != 2 or not np.all(np.isfinite(p)) for p in truth)):
            print("truth needs one finite x,y pair per fresh sweep, including unresolved sweeps")
            return 1

    rows = []
    matched_truth = []
    for i, sw in enumerate(s.sweeps):
        res = matcher.localize(sw.aps, top=2)
        if not res:
            continue
        best, d1 = res[0]
        d2 = res[1][1] if len(res) > 1 else float("nan")
        rows.append((sw.t, best, d1, d2))
        if truth:
            matched_truth.append(truth[i])
        print(f"  t={sw.t:7.1f}s -> cell {best.label:>6}  "
              f"d1={d1:6.1f} dB  margin={(d2-d1):5.1f} dB")

    if truth:
        preds = [(b.x, b.y) for _, b, _, _ in rows]
        n = len(preds)
        print(f"resolved {n}/{len(s.sweeps)} sweeps; {len(s.sweeps) - n} unresolved")
        if n:
            print(f"\nlocalisation error vs {n} ground-truth points:")
            print(json.dumps(radiomap.error_metrics(rm, matched_truth, preds), indent=1))

    series = [(t, d1) for t, _, d1, _ in rows]
    if series:
        viz.plot_trace(series, str(OUT / "localize_dist.png"),
                       "fingerprint distance to best-matching cell", "dB (L1 over anchors)")
        amb = [(t, d2 - d1) for t, _, d1, d2 in rows if np.isfinite(d2)]
        if amb:
            viz.plot_trace(amb, str(OUT / "localize_margin.png"),
                           "top1-top2 margin (small => ambiguous)", "dB")
    return 0


def cmd_heatmap(args) -> int:
    mp = Path(args.radiomap)
    if not mp.exists():
        print(f"no radio map at {mp}; run `wifisense calibrate` first"); return 1
    rm = radiomap.RadioMap.load(mp)
    if rm.meta.get("verdict") != "usable":
        print("localisation disabled: map must pass validate-map first")
        return 2
    matcher = radiomap.Matcher(rm)
    s = io.load(args.capture)
    visits: dict[tuple[float, float], int] = {}
    for sw in s.sweeps:
        res = matcher.localize(sw.aps, top=1)
        if not res:
            continue
        p, _ = res[0]
        key = (round(p.x, 2), round(p.y, 2))
        visits[key] = visits.get(key, 0) + 1
    viz.plot_occupancy(rm, visits, str(OUT / "occupancy.png"))
    tot = sum(visits.values())
    print(f"attributed {tot} sweeps to {len(visits)} cells")
    for k, v in sorted(visits.items(), key=lambda kv: -kv[1]):
        print(f"   cell {k}: {v}")
    print(f"wrote {OUT/'occupancy.png'}")
    return 0


# ------------------------------------------------------------------------ main
