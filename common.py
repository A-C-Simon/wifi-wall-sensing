"""Paths and build steps shared by the command line tools and the displays."""
from __future__ import annotations

import os
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
