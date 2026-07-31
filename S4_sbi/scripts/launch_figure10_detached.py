"""Launch one long Figure-10 stage as a detached, hidden Windows process."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = PROJECT_ROOT / "S4_sbi" / "results" / "figure10_8d_7d" / "logs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage")
    args = parser.parse_args()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stdout_path = LOG_ROOT / f"{args.stage}_{stamp}_stdout.log"
    stderr_path = LOG_ROOT / f"{args.stage}_{stamp}_stderr.log"
    stdout = stdout_path.open("w", encoding="utf-8", buffering=1)
    stderr = stderr_path.open("w", encoding="utf-8", buffering=1)
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    process = subprocess.Popen(
        [
            sys.executable,
            str(PROJECT_ROOT / "S4_sbi" / "scripts" / "run_figure10_pipeline.py"),
            args.stage,
        ],
        cwd=PROJECT_ROOT,
        stdout=stdout,
        stderr=stderr,
        stdin=subprocess.DEVNULL,
        creationflags=flags,
        close_fds=True,
    )
    print(f"pid={process.pid}")
    print(f"stdout={stdout_path}")
    print(f"stderr={stderr_path}")


if __name__ == "__main__":
    main()
