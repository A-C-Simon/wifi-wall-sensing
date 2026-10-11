"""Paths and build steps shared by the command line tools and the displays."""
from __future__ import annotations

import os
import argparse
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
CORE = ROOT / "native" / "wifisense_core"
OUT.mkdir(exist_ok=True)


def build_core() -> None:
    """Compile the CoreWLAN capture daemon if it is missing or out of date."""
    src = ROOT / "native" / "wifisense_core.swift"
    if CORE.exists() and CORE.stat().st_mtime >= src.stat().st_mtime:
        return
    subprocess.run(["swiftc", "-O", "-framework", "CoreWLAN",
                    "-o", str(CORE), str(src)], check=True)


def default_iface() -> str:
    return os.environ.get("WIFISENSE_IFACE", "en0")


def finite_float(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("must be a finite number") from None
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("must be a finite number")
    return result


def positive_float(value) -> float:
    result = finite_float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def nonnegative_float(value) -> float:
    result = finite_float(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def positive_int(value) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def nonnegative_int(value) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return result
