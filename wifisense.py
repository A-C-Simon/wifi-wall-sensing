#!/usr/bin/env python3
"""wifisense: device free presence sensing from a MacBook's own WiFi.

Subcommands live in cli/ and each module owns one concept. This file only
builds the argument parser and dispatches.
"""
from __future__ import annotations

import argparse
import sys

from common import finite_float, positive_float, nonnegative_float, positive_int, nonnegative_int
from common import CORE, OUT, ROOT, build_core, default_iface
from cli.measure import cmd_analyze, cmd_baseline, cmd_capture, cmd_doctor
from cli.survey import cmd_calibrate, cmd_coarsen
from cli.match import cmd_heatmap, cmd_localize
from cli.validate import (cmd_calibrate_distance, cmd_cv_test,
                          cmd_repeat_test, cmd_validate_map)


def main() -> int:
    ap = argparse.ArgumentParser(prog="wifisense", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default=default_iface())
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor"); d.add_argument("--duration", type=positive_int, default=60)
    d.set_defaults(func=cmd_doctor)

    c = sub.add_parser("capture"); c.add_argument("--duration", type=nonnegative_int, default=300)
    c.add_argument("--fine-hz", type=positive_float, default=20); c.add_argument("--out")
    c.add_argument("--no-sweep", action="store_true"); c.set_defaults(func=cmd_capture)

    b = sub.add_parser("baseline"); b.add_argument("--capture"); b.add_argument("--duration", type=positive_int, default=300)
    b.add_argument("--win", type=positive_float, default=10.0); b.add_argument("--k", type=positive_float, default=6.0)
    b.set_defaults(func=cmd_baseline)

    a = sub.add_parser("analyze"); a.add_argument("capture"); a.add_argument("--win", type=positive_float, default=10.0)
    a.add_argument("--no-baseline", action="store_true"); a.set_defaults(func=cmd_analyze)

    cal = sub.add_parser("calibrate")
    cal.add_argument("--cols", type=positive_int, default=3)
    cal.add_argument("--rows", type=positive_int, default=3)
    cal.add_argument("--extent", type=positive_float, default=2.0, help="metres between cells")
    cal.add_argument("--origin-x", type=finite_float, default=0.0)
    cal.add_argument("--origin-y", type=finite_float, default=0.0)
    cal.add_argument("--hold", type=positive_float, default=45.0)
    cal.add_argument("--sweeps", type=positive_int, default=3)
    cal.add_argument("--laptop-x", type=finite_float, default=0.0)
    cal.add_argument("--laptop-y", type=finite_float, default=0.0)
    cal.add_argument("--room-x", type=positive_float, default=7.0)
    cal.add_argument("--room-y", type=positive_float, default=6.0)
    cal.set_defaults(func=cmd_calibrate)

    lo = sub.add_parser("localize"); lo.add_argument("capture"); lo.add_argument("--radiomap", default=str(OUT / "radiomap.json"))
    lo.add_argument("--win", type=positive_float, default=10.0); lo.add_argument("--truth")
    lo.set_defaults(func=cmd_localize)

    cv = sub.add_parser("cv-test")
    cv.add_argument("--raw-dir", default=str(OUT / "calib_corner"))
    cv.set_defaults(func=cmd_cv_test)

    rt = sub.add_parser("repeat-test")
    rt.add_argument("--a", default=str(OUT / "radiomap_9cell.json"))
    rt.add_argument("--b", default=str(OUT / "radiomap.json"))
    rt.add_argument("--tol", type=nonnegative_float, default=0.05)
    rt.set_defaults(func=cmd_repeat_test)

    vm = sub.add_parser("validate-map")
    vm.add_argument("--radiomap", default=str(OUT / "radiomap.json"))
    vm.add_argument("--raw-dir", default=str(OUT / "calib_corner"))
    vm.add_argument("--default-noise", type=nonnegative_float, default=16.0)
    vm.add_argument("--sep-threshold", type=positive_float, default=2.0)
    vm.add_argument("--repeat-margin", type=positive_float, default=1.0)
    vm.set_defaults(func=cmd_validate_map)

    co = sub.add_parser("coarsen-map")
    co.add_argument("--radiomap", default=str(OUT / "radiomap.json"))
    co.add_argument("--out", default=str(OUT / "radiomap_coarse.json"))
    co.add_argument("--extent", type=positive_float, default=4.0,
                    help="coarse cell size in metres")
    co.set_defaults(func=cmd_coarsen)

    cd = sub.add_parser("calibrate-distance")
    cd.add_argument("--points", default="1,2,3.5,5,7")
    cd.add_argument("--hold", type=positive_float, default=20.0)
    cd.set_defaults(func=cmd_calibrate_distance)

    hm = sub.add_parser("heatmap"); hm.add_argument("capture"); hm.add_argument("--radiomap", default=str(OUT / "radiomap.json"))
    hm.set_defaults(func=cmd_heatmap)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
