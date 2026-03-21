#!/usr/bin/env python3
"""Smoke check: rclone on PATH and listremotes exits 0. Exits 0 if rclone missing (skip)."""
from __future__ import annotations

import shutil
import subprocess
import sys


def main() -> int:
    rclone = shutil.which("rclone")
    if not rclone:
        print("SKIP: rclone not on PATH")
        return 0
    p = subprocess.run(
        ["rclone", "listremotes"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if p.returncode != 0:
        print(p.stderr or p.stdout or "listremotes failed")
        return 1
    print(p.stdout.strip() or "(no remotes configured)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
