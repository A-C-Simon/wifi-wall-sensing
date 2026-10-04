"""Survey a grid of positions to build a fingerprint radio map, and coarsen it."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np

from sensing import io, radiomap, viz
from sensing.features import _fill_columns

from common import CORE, OUT, ROOT, build_core


def cmd_calibrate(args) -> int:
    """Guided walk to build a multi-AP fingerprint radio map.

    Each cell gets its own capture file so cell attribution is unambiguous --
    correlating a single stream against operator prompts would depend on
    wall-clock alignment that does not survive buffering.

    The sweep channel refreshes only every ~12 s, so a cell must be held long
    enough to collect `args.sweeps` independent vectors before moving on.

    The grid lives in the same coordinate frame the radar draws (origin at the
    room corner), so `--laptop-x/y` must match what the radar is told,
    otherwise every contact lands in the wrong place.
    """
    build_core()
    cells = [(args.origin_x + x * args.extent, args.origin_y + y * args.extent)
             for y in range(args.rows) for x in range(args.cols)]
    hold = max(args.hold, args.sweeps * 14.0)

    print(f"calibration: {len(cells)} cells x {hold:.0f}s = "
          f"{hold*len(cells)/60:.1f} min total")
    print("The laptop must stay PUT for the whole survey -- it is the receiver.")
    print("Walk to each position and stand still. Ctrl-C aborts.\n")

    fps: list[radiomap.Fingerprint] = []
    raw_dir = OUT / "calib_raw"
    raw_dir.mkdir(exist_ok=True)
    t_start = time.time()
    try:
        for idx, (x, y) in enumerate(cells):
            f = raw_dir / f"cell_{x:.1f}_{y:.1f}.jsonl"
            at = time.time() - t_start
            print(f"[{idx+1}/{len(cells)}] stand at x={x:.1f}m y={y:.1f}m "
                  f"for {hold:.0f}s   (starts {at/60:.1f} min in)", flush=True)
            subprocess.run([str(CORE), "--iface", args.iface, "--out", str(f),
                            "--duration", str(int(hold)), "--fine-hz", "20", "--quiet"],
                           check=True)
            s = io.load(f)
            fp = fingerprint_from_session(s, x, y)
            if fp:
                fps.append(fp)
                print(f"      {fp.n_samples} sweeps, {len(fp.rssi)} anchors", flush=True)
            else:
                print("      WARNING: no usable fingerprint at this cell", flush=True)
    except KeyboardInterrupt:
        print("\naborted")

    if not fps:
        print("no fingerprints collected")
        return 1
    keys = sorted({k for p in fps for k in p.rssi})
    rm = radiomap.RadioMap(points=fps, keys=keys,
                           meta={"iface": args.iface, "created": time.time(),
                                 "laptop": [args.laptop_x, args.laptop_y],
                                 "room": [args.room_x, args.room_y],
                                 "hold_s": hold})
    mp = OUT / "radiomap.json"
    rm.save(mp)
    viz.plot_radio_map(rm, str(OUT / "radiomap.png"))

    print(f"\nbuilt radio map: {len(fps)} cells x {len(keys)} anchors -> {mp}")
    sep = separability_report(rm, raw_dir, cells)
    if sep:
        print("\n" + sep)
    return 0


def separability_report(rm, raw_dir: Path, cells) -> str:
    """Is the map actually usable, or are the cells indistinguishable?

    Within-cell noise is measurable directly: the sweeps recorded while the
    operator stood still at one cell differ from each other by pure temporal
    multipath. Between-cell distance is the L1 distance between cell means. If
    the closest pair of cells is not separated by more than that noise, nearest
    -neighbour matching will pick the wrong cell more often than not, and the
    map is decorative.
    """
    import numpy as np
    from sensing.features import _fill_columns

    # within-cell temporal noise
    noise = []
    for x, y in cells:
        f = raw_dir / f"cell_{x:.1f}_{y:.1f}.jsonl"
        if not f.exists():
            continue
        s = io.load(f)
        if len(s.sweeps) < 2:
            continue
        mat = np.array([[float(sw.aps.get(k, np.nan)) for k in rm.keys]
                        for sw in s.sweeps], dtype=float)
        fil = _fill_columns(mat)
        d = np.abs(np.diff(fil, axis=0)).sum(axis=1)
        if len(d):
            noise.append(float(np.median(d)))
    if not noise:
        return "  (could not measure within-cell noise)"
    wn = float(np.median(noise))

    # between-cell distances between cell means
    mat = _fill_columns(rm.matrix())
    D = np.abs(mat[:, None, :] - mat[None, :, :]).sum(axis=2)
    np.fill_diagonal(D, np.inf)
    closest = float(D.min())
    med = float(np.median(D[np.isfinite(D)]))

    ratio = closest / wn if wn > 1e-9 else float("inf")
    verdict = ("USABLE" if ratio >= 2.0 else
               "MARGINAL - expect frequent cell confusion" if ratio >= 1.2 else
               "NOT SEPARABLE - cells are within the noise; localisation "
               "cannot work at this grid spacing")
    lines = [
        "cell separability",
        f"  within-cell temporal noise (median) : {wn:.1f} dB summed",
        f"  closest pair of cells              : {closest:.1f} dB",
        f"  median pair of cells               : {med:.1f} dB",
        f"  separability ratio                 : {ratio:.2f}",
        f"  verdict                            : {verdict}",
    ]
    if ratio < 2.0:
        lines.append("  -> survey a wider grid (larger --extent), hold longer per "
                     "cell (--hold), or accept coarse region-level attribution")
    return "\n".join(lines)


def fingerprint_from_session(s, x: float, y: float) -> radiomap.Fingerprint | None:
    """Average the fresh sweeps of one cell into a single fingerprint.

    Averaging is what makes this work at all: a raw sweep is quantised to 1 dB
    with only a handful of samples, so the per-anchor mean is a far more stable
    descriptor than any individual scan.
    """
    if len(s.sweeps) < 2:
        return None
    acc: dict[str, list[float]] = {}
    for sw in s.sweeps:
        for k, v in sw.aps.items():
            acc.setdefault(k, []).append(float(v))
    rssi = {k: sum(v) / len(v) for k, v in acc.items() if v}
    if len(rssi) < 5:
        return None
    return radiomap.Fingerprint(x=x, y=y, label=f"{x:.0f},{y:.0f}",
                                t=s.sweeps[0].t, rssi=rssi, n_samples=len(s.sweeps))


# ------------------------------------------------------------------- localize


def cmd_coarsen(args) -> int:
    """Aggregate a fine fingerprint map into coarser regions.

    Adjacent cells in a fine grid are usually separated by less than the
    within-cell temporal noise, which makes nearest-neighbour matching pick the
    wrong cell more often than not. Merging cells into larger regions raises
    the between-region separation above the noise floor, trading resolution for
    reliability -- the only honest option when the fine grid cannot support it.

    Sub-cells are assigned to their nearest coarse centre and their fingerprints
    are averaged, so this needs no additional survey time.
    """
    src = Path(args.radiomap)
    if not src.exists():
        print(f"no map at {src}")
        return 1
    fine = radiomap.RadioMap.load(src)
    if len(fine.points) < 2:
        print("need at least 2 cells to coarsen")
        return 1

    step = args.extent
    ys = sorted({round(p.y, 3) for p in fine.points})
    xs = sorted({round(p.x, 3) for p in fine.points})
    # Index arithmetic, not float accumulation: accumulating step in a while
    # loop put the tolerance on the wrong side of the comparison and silently
    # produced a single region instead of a grid.
    nx = int(round((xs[-1] - xs[0]) / step)) + 1
    ny = int(round((ys[-1] - ys[0]) / step)) + 1
    centers = [(round(xs[0] + i * step, 3), round(ys[0] + j * step, 3))
               for j in range(ny) for i in range(nx)]
    print(f"coarse grid {nx}x{ny} at {step:.1f} m spacing -> {len(centers)} candidate regions")

    buckets: dict[tuple[float, float], list[radiomap.Fingerprint]] = {}
    for p in fine.points:
        c = min(centers, key=lambda c: (p.x - c[0]) ** 2 + (p.y - c[1]) ** 2)
        buckets.setdefault(c, []).append(p)

    out: list[radiomap.Fingerprint] = []
    for c, members in sorted(buckets.items()):
        keys = set()
        for m in members:
            keys |= set(m.rssi)
        avg = {}
        for k in keys:
            vals = [m.rssi[k] for m in members if k in m.rssi]
            avg[k] = sum(vals) / len(vals)
        out.append(radiomap.Fingerprint(
            x=c[0], y=c[1], label=f"{c[0]:.0f},{c[1]:.0f}", t=0.0,
            rssi=avg, n_samples=sum(m.n_samples for m in members)))
        print(f"  region {c[0]:.0f},{c[1]:.0f}  <- {len(members)} cells "
              f"({', '.join(m.label for m in members)})")

    keys = sorted({k for p in out for k in p.rssi})
    rm = radiomap.RadioMap(points=out, keys=keys,
                           meta=dict(fine.meta, coarsened_from=str(src),
                                     extent=step))
    mp = Path(args.out)
    rm.save(mp)
    viz.plot_radio_map(rm, str(OUT / "radiomap_coarse.png"))
    print(f"\nwrote {mp}  ({len(out)} regions x {len(keys)} anchors)")

    # separability, using the same within-cell noise measured during the survey
    raw_dir = OUT / "calib_raw"
    noise = []
    for x, y in cells_of(fine):
        f = raw_dir / f"cell_{x:.1f}_{y:.1f}.jsonl"
        if not f.exists():
            continue
        s_ = io.load(f)
        if len(s_.sweeps) < 2:
            continue
        import numpy as np
        from sensing.features import _fill_columns
        mat = np.array([[float(sw.aps.get(k, np.nan)) for k in rm.keys]
                        for sw in s_.sweeps], dtype=float)
        d = np.abs(np.diff(_fill_columns(mat), axis=0)).sum(axis=1)
        if len(d):
            noise.append(float(np.median(d)))
    if noise:
        wn = sorted(noise)[len(noise) // 2]
        import numpy as np
        M = _fill_columns(rm.matrix())
        D = np.abs(M[:, None, :] - M[None, :, :]).sum(axis=2)
        np.fill_diagonal(D, np.inf)
        print(f"\ncoarse separability: within-cell noise {wn:.1f} dB, "
              f"closest region pair {D.min():.1f} dB, ratio {D.min()/wn:.2f}")
        print("verdict:", "USABLE" if D.min() / wn >= 2 else "still not separable")
    return 0


def cells_of(rm: radiomap.RadioMap):
    return [(p.x, p.y) for p in rm.points]


# ------------------------------------------------------- distance calibration
