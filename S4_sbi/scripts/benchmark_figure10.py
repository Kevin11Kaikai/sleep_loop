"""Measure paired Route-3 simulation throughput before protocol freezing.

The benchmark uses the frozen synthetic cortical-rate 14D extractor.  Its
outputs are resource-planning evidence only and are never part of training or
final diagnostics.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "S4_sbi" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sleep_sbi.route3_7d_experiment import fixed_c_ctx2th  # noqa: E402
from sleep_sbi.route3_global_robustness import (  # noqa: E402
    deterministic_seed_schedule,
    run_resumable_batch,
    sobol_theta,
)


RESULT_ROOT = PROJECT_ROOT / "S4_sbi" / "results" / "figure10_8d_7d"
BENCHMARK_ROOT = RESULT_ROOT / "resource_benchmark"
N_PAIRS = 32
SOBOL_SEED = 9424001
SIMULATOR_SEED_BASE = 9424101
MAX_WORKERS = 8


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    theta8 = sobol_theta(N_PAIRS, SOBOL_SEED)
    theta7_full = theta8.copy()
    theta7_full[:, 7] = fixed_c_ctx2th()
    seeds = deterministic_seed_schedule(N_PAIRS, SIMULATOR_SEED_BASE)

    tracks = {}
    wall_started = perf_counter()
    for label, theta in (("8d", theta8), ("7d_fixed", theta7_full)):
        started = perf_counter()
        rows = run_resumable_batch(
            theta,
            seeds,
            BENCHMARK_ROOT / label,
            max_workers=MAX_WORKERS,
            labels=[f"benchmark_{label}_{index:03d}" for index in range(N_PAIRS)],
        )
        wall_s = perf_counter() - started
        success = np.asarray([row["success"] for row in rows], bool)
        runtime = np.asarray([row["runtime_s"] for row in rows], float)
        tracks[label] = {
            "attempted": N_PAIRS,
            "valid": int(success.sum()),
            "failed": int((~success).sum()),
            "wall_s": float(wall_s),
            "throughput_valid_per_s": float(success.sum() / wall_s),
            "median_worker_runtime_s": float(np.nanmedian(runtime)),
            "sum_worker_runtime_s": float(np.nansum(runtime)),
        }

    measured_pair_rate = N_PAIRS / max(
        tracks["8d"]["wall_s"], tracks["7d_fixed"]["wall_s"]
    )
    projections = {}
    for scale in (8192, 32768, 131072, 524288, 1000000, 3000000):
        sequential_tracks_s = (
            scale / tracks["8d"]["throughput_valid_per_s"]
            + scale / tracks["7d_fixed"]["throughput_valid_per_s"]
        )
        projections[str(scale)] = {
            "paired_tracks_wall_hours": float(sequential_tracks_s / 3600),
            "paired_tracks_wall_days": float(sequential_tracks_s / 86400),
            "raw_array_gib_estimate": float(
                scale * 2 * (8 + 14 + 4) * 8 / 1024**3
            ),
        }

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "resource benchmark only; excluded from scientific datasets",
        "environment": {
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "sys_executable": sys.executable,
            "sys_prefix": sys.prefix,
            "python": sys.version,
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
        },
        "configuration": {
            "pairs": N_PAIRS,
            "workers": MAX_WORKERS,
            "sobol_seed": SOBOL_SEED,
            "simulator_seed_base": SIMULATOR_SEED_BASE,
            "fixed_c_ctx2th": fixed_c_ctx2th(),
        },
        "tracks": tracks,
        "paired_rate_per_s_conservative": float(measured_pair_rate),
        "total_wall_s": float(perf_counter() - wall_started),
        "projections": projections,
    }
    _atomic_json(BENCHMARK_ROOT / "resource_benchmark.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
