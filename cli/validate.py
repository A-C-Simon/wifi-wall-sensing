"""Decide whether a radio map is fit for localisation."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from sensing import io, linkmodel, radiomap
from sensing.features import _fill_columns

from common import CORE, OUT, ROOT, build_core


def cmd_repeat_test(args) -> int:
    """Measure session-to-session repeatability by comparing two surveys.

    Re-survey the same grid at a later time and diff the two maps. The decisive
    number is whether a given cell measures the same twice: live vectors are
    always matched against a map built earlier, so that drift -- not the
    within-session noise -- is the noise floor localisation actually lives with.
    """
    import numpy as np
    from sensing.features import _fill_columns

    A = radiomap.RadioMap.load(args.a)
    B = radiomap.RadioMap.load(args.b)
    tol = args.tol

    def find(rm, x, y):
        for p in rm.points:
            if abs(p.x - x) <= tol and abs(p.y - y) <= tol:
                return p
        return None

    same, diff, details = [], [], []
    matched = []
    for pa in A.points:
        pb = find(B, pa.x, pa.y)
        if not pb:
            continue
        keys = [k for k in pa.rssi if k in pb.rssi]
        if len(keys) < 5:
            continue
        d = sum(abs(pa.rssi[k] - pb.rssi[k]) for k in keys)
        same.append(d)
        matched.append((pa.label, pb.label, d, len(keys)))
        details.append(f"    {pa.label:>7} re-measured as {d:5.1f} dB over "
                       f"{len(keys):2d} anchors ({d/len(keys):.2f} dB/anchor)")

    # The two surveys rarely see the same anchor set (APs come and go), so both
    # maps must be projected onto the common keys before they can be compared.
    common = sorted(set(A.keys) & set(B.keys))
    if len(common) < 5:
        print(f"only {len(common)} anchors common to both surveys - cannot compare")
        return 1
    print(f"\ncomparing over {len(common)} anchors common to both surveys "
          f"(A had {len(A.keys)}, B had {len(B.keys)})")

    def sub_matrix(rm):
        idx = [rm.keys.index(k) for k in common]
        return _fill_columns(rm.matrix()[:, idx])

    ma, mb = sub_matrix(A), sub_matrix(B)

    # Spatial spread must be measured WITHIN each survey, not across the two.
    # Comparing A[i] against B[j] for i != j mixes position with the
    # difference in which anchors each survey happened to hear, which inflates
    # the apparent spatial signal and can invert the verdict.
    def within_spread(m):
        v = [float(np.abs(m[i] - m[j]).sum())
             for i in range(len(m)) for j in range(i + 1, len(m))]
        return v

    diff = within_spread(ma) + within_spread(mb)

    print("same position, different session:")
    print("\n".join(details) if details else "    (no matching cells)")
    ms = float(np.mean(same)) if same else float("nan")
    md = float(np.mean(diff)) if diff else float("nan")
    print(f"\n  mean same-position drift      : {ms:.1f} dB  ({len(same)} positions)")
    print(f"  mean within-survey spatial spread: {md:.1f} dB")
    if same and diff:
        print(f"  worst same-position drift      : {max(same):.1f} dB")
        print(f"  largest within-survey separation: {max(diff):.1f} dB")
        overlap = sum(1 for v in same if v > max(diff))
        print(f"  same-position drifts exceeding ALL spatial separations: "
              f"{overlap}/{len(same)}")
    if not same or not diff:
        print("  insufficient overlap to judge")
        return 1
    ok = md > ms
    print(f"\n  verdict: {'VIABLE' if ok else 'NOT VIABLE - session drift exceeds spatial signal'}")
    print("  (a live vector is matched against an older map, so drift sets the floor)")

    B.meta["repeatability"] = {
        "compared_with": str(args.a),
        "mean_same_position_db": ms,
        "mean_different_position_db": md,
        "n_positions": len(same),
    }
    B.save(args.b)
    print(f"\nrecorded in {args.b}")
    return 0 if ok else 2


def cmd_cv_test(args) -> int:
    """Leave-one-out region classification accuracy from a survey's own sweeps.

    Each cell's sweeps are split in half: one half is the query, the centroids
    are built from the other half. This measures how often nearest-neighbour
    fingerprint matching picks the right cell, using only data from within one
    survey, so it is not confounded by session drift.

    Chance accuracy is 1/n_cells, which is the number to compare against -- for
    a 4-region map that is 25%.
    """
    import numpy as np
    from sensing.features import _fill_columns

    raw_dir = Path(args.raw_dir)
    cells = []
    for f in sorted(raw_dir.glob("cell_*.jsonl")):
        s_ = io.load(f)
        if len(s_.sweeps) < 4:
            continue
        cells.append((f, s_))
    if len(cells) < 2:
        print(f"need >=2 cells with >=4 sweeps in {raw_dir}")
        return 1

    # Anchor set: keys heard in EVERY sweep of EVERY cell. Intersecting only the
    # first sweep of each cell misses anchors that drop out mid-cell and then
    # raises KeyError when the halves are extracted.
    common = None
    for _f, s_ in cells:
        for sw in s_.sweeps:
            ks = set(sw.aps)
            common = ks if common is None else (common & ks)
    common = sorted(common or [])
    if len(common) < 5:
        print("too few anchors common to all cells")
        return 1

    def half_vec(s_, lo, hi):
        acc = {k: [] for k in common}
        for sw in s_.sweeps[lo:hi]:
            for k in common:
                acc[k].append(float(sw.aps.get(k, np.nan)))
        acc = {k: [v for v in vs if np.isfinite(v)] for k, vs in acc.items()}
        med = {k: (sum(v) / len(v) if v else float("nan")) for k, v in acc.items()}
        # any anchor missing from a whole half is imputed with the column median
        vals = [v for v in med.values() if np.isfinite(v)]
        fill = float(np.median(vals)) if vals else -70.0
        return np.array([med[k] if np.isfinite(med[k]) else fill for k in common])

    labels = []
    for f, _s in cells:
        labels.append(f.stem.replace("cell_", ""))

    print(f"{len(cells)} cells, {len(common)} common anchors, "
          f"leave-one-out over sweep halves\n")
    hits = 0
    total = 0
    for qi, (f, s_) in enumerate(cells):
        for lo, hi in ((0, 2), (2, 4)):
            q = half_vec(s_, lo, hi)
            # centroids from the opposite half of every cell
            best, bd = None, float("inf")
            for cj, (g, s2) in enumerate(cells):
                clo, chi = (2, 4) if (lo, hi) == (0, 2) else (0, 2)
                d = float(np.abs(q - half_vec(s2, clo, chi)).sum())
                if d < bd:
                    bd, best = d, cj
            ok = best == qi
            hits += ok
            total += 1
            print(f"  query {labels[qi]:>6} halves {lo}-{hi} -> matched "
                  f"{labels[best]:>6} ({bd:5.1f} dB)  {'OK' if ok else 'WRONG'}")
    acc = hits / total
    chance = 1.0 / len(cells)
    print(f"\n  accuracy      : {acc*100:.1f}%  ({hits}/{total})")
    print(f"  chance        : {chance*100:.1f}%")
    print(f"  lift          : {acc/chance:.2f}x")
    if acc <= chance * 1.5:
        print("\n  verdict: NOT USABLE - barely distinguishable from guessing")
    elif acc < 0.7:
        print("\n  verdict: MARGINAL - better than chance, expect frequent errors")
    else:
        print("\n  verdict: USABLE within a single session")
    return 0


# ------------------------------------------------------------- map validation


def cmd_validate_map(args) -> int:
    """Decide whether a radio map is fit for localisation, and record the verdict.

    Two independent failure modes are checked:

      separability  - are distinct cells further apart than the noise?
      repeatability - does the same cell measure the same twice?

    Repeatability is the one that matters most and the easiest to get wrong.
    A map can look excellent within a single session and be useless in
    production if the environment drifts faster than a body changes the
    vector, because every live vector is matched against a map built earlier.
    """
    mp = Path(args.radiomap)
    if not mp.exists():
        print(f"no map at {mp}")
        return 1
    rm = radiomap.RadioMap.load(mp)
    import numpy as np
    from sensing.features import _fill_columns

    M = _fill_columns(rm.matrix())
    n = len(rm.points)
    D = np.abs(M[:, None, :] - M[None, :, :]).sum(axis=2)
    np.fill_diagonal(D, np.inf)

    # within-cell temporal noise, measured from the survey captures if present
    noise = []
    raw_dir = Path(args.raw_dir)
    for p in rm.points:
        f = raw_dir / f"cell_{p.x:.1f}_{p.y:.1f}.jsonl"
        if not f.exists():
            continue
        s_ = io.load(f)
        if len(s_.sweeps) < 2:
            continue
        mat = np.array([[float(sw.aps.get(k, np.nan)) for k in rm.keys]
                        for sw in s_.sweeps], dtype=float)
        d = np.abs(np.diff(_fill_columns(mat), axis=0)).sum(axis=1)
        if len(d):
            noise.append(float(np.median(d)))
    wn = float(np.median(noise)) if noise else float(args.default_noise)

    closest = float(D.min())
    sep_ratio = closest / wn if wn > 1e-9 else float("inf")
    mean_between = float(D[np.isfinite(D)].mean())

    print(f"map: {n} regions x {len(rm.keys)} anchors")
    print(f"  within-cell temporal noise : {wn:.1f} dB")
    print(f"  closest region pair        : {closest:.1f} dB  ({sep_ratio:.2f}x noise)")
    print(f"  mean region pair           : {mean_between:.1f} dB")

    rep = rm.meta.get("repeatability")
    usable = sep_ratio >= args.sep_threshold
    if rep is not None:
        rp = rep.get("mean_same_position_db", float("nan"))
        rs = rep.get("mean_different_position_db", float("nan"))
        print(f"  session repeatability      : {rp:.1f} dB at the same spot "
              f"({rep.get('n_positions')} positions re-measured)")
        print(f"  spatial separation         : {rs:.1f} dB between spots")
        # A live vector is matched against an older map, so the drift between
        # sessions is the noise floor that actually matters.
        usable = usable and (rs > rp * args.repeat_margin)
        print(f"  usable                     : {usable}")
        if rs <= rp * args.repeat_margin:
            print("    -> SESSION DRIFT EXCEEDS SPATIAL SIGNAL. A body at the same")
            print("       spot changes the fingerprint more than a body at a")
            print("       different spot. Nearest-neighbour matching will pick the")
            print("       wrong region more often than not.")

    rm.meta["verdict"] = "usable" if usable else "unusable"
    rm.meta["separability_ratio"] = sep_ratio
    rm.meta["within_cell_noise_db"] = wn
    rm.save(mp)
    print(f"\nverdict written to {mp}: {rm.meta['verdict']}")
    return 0 if usable else 2


# ------------------------------------------------------------------- coarsen


def cmd_calibrate_distance(args) -> int:
    """Measure the RSSI<->range relationship so anchors can be placed by range.

    RSSI alone does not give distance; it needs this survey. The operator
    stands at a series of measured distances from the access point and the
    log-distance model is fitted, which then lets every other anchor's range
    be estimated from its RSSI.
    """
    build_core()
    pts = [float(x) for x in args.points.split(",")]
    hold = args.hold
    print(f"distance calibration at {pts} m, {hold:.0f}s each\n")
    print("Keep the laptop with you, face the access point, and hold still.")
    print("The AP must be the one you are associated with (the measured link).\n")

    dists, rssis = [], []
    for d in pts:
        f = OUT / f"dist_{d:.1f}m.jsonl"
        input(f"  stand {d:.1f} m from the access point, then press Enter... ")
        subprocess.run([str(CORE), "--iface", args.iface, "--out", str(f),
                        "--duration", str(int(hold)), "--fine-hz", "20",
                        "--no-sweep", "--quiet"], check=True)
        s_ = io.load(f)
        if not s_.fine:
            print("    no data, skipping")
            continue
        # Use the median: RSSI is heavy-tailed and the mean is pulled by fades.
        med = sorted(x.rssi for x in s_.fine)[len(s_.fine) // 2]
        spread = sorted(x.rssi for x in s_.fine)
        print(f"    RSSI median {med:.0f} dBm  (min {spread[0]}, max {spread[-1]})")
        dists.append(d)
        rssis.append(med)

    if len(dists) < 2:
        print("need at least 2 usable points")
        return 1
    try:
        m = linkmodel.LinkModel().fit(dists, rssis)
    except ValueError as e:
        print(f"fit failed: {e}")
        return 1
    m.note += f" | points {list(zip(dists, rssis))}"
    m.save(OUT / "linkmodel.json")
    print(f"\nfitted: RSSI(d) = {m.A:.2f} - 10*{m.n:.2f}*log10(d)   [{m.note}]")
    print(f"path-loss exponent n = {m.n:.2f}  "
          f"({'same room' if m.n < 3.2 else 'through structure' if m.n < 4.5 else 'heavy obstruction'})")
    lo, hi = linkmodel.estimate_uncertainty(m, float(rssis[-1]))
    print(f"example: the last point ({dists[-1]} m, {rssis[-1]:.0f} dBm) inverts to "
          f"{m.range_of(rssis[-1]):.2f} m; sweeping n in 2.0-4.0 gives {lo:.2f}-{hi:.2f} m")
    print(f"\nwrote {OUT/'linkmodel.json'}")
    return 0


# ------------------------------------------------------------------- heatmap
