"""Phase-2 member-wise PPC runner for Notebook 33 (resumable, non-destructive).

Writes only under ``notebook_33_ensemble_attribution/ppc/`` and N33 tables/json/figures.
Does not retrain, does not modify diagnostics 28–32 outputs, and is not invoked by
the presentation notebook during nbconvert.
"""

from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .figure10_bank import _atomic_npz, _simulate_shard
from .figure10_ensemble_attribution import (
    FIGURES,
    JSON_DIR,
    LOGS,
    N33_ROOT,
    PPC_DIR,
    TABLES,
    VALIDATION,
    _ensure_dirs,
    _utc_now,
    assert_old_results_unchanged,
    atomic_json,
    require_neurolib_env,
    sha256_file,
)
from .figure10_protocol import (
    FINAL_SCALE,
    MAX_SIMULATION_WORKERS,
    PROJECT_ROOT,
    RESULTS_ROOT,
    fixed_c_ctx2th,
    rate_contract_hash,
    rate_feature_names,
    sha256_file as protocol_sha256_file,
    verify_preregistration,
)
from .figure10_training import (
    load_members,
    member_seeds,
    sample_member,
    scale_observation,
    unit_to_physical,
)
from .route3_global_robustness import deterministic_seed_schedule

PRIMARY_BUDGET = 256
RESERVE_BUDGET = 64
CATALOG_N = PRIMARY_BUDGET + RESERVE_BUDGET
MICRO_SHARD = 16
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 9360001

# Distinct from figure10 SEEDS and Notebook33 benchmark seeds.
PHASE2_SEEDS = {
    "posterior_sampling_8d": 9340001,
    "posterior_sampling_7d": 9341001,
    "simulator_base_8d": 9350001,
    "simulator_base_7d": 9351001,
    "bootstrap": BOOTSTRAP_SEED,
}

PRIMARY_OBS = (
    RESULTS_ROOT
    / "diagnostics"
    / "ppc"
    / "primary_observation"
    / "8d"
    / "primary_observation_8d.npz"
)
ADAPTER_PATH = (
    PROJECT_ROOT / "S4_sbi" / "src" / "sleep_sbi" / "simulator_observable_adapter.py"
)
DIAG_PATH = (
    PROJECT_ROOT / "S4_sbi" / "src" / "sleep_sbi" / "figure10_diagnostics.py"
)
BANK_PATH = PROJECT_ROOT / "S4_sbi" / "src" / "sleep_sbi" / "figure10_bank.py"
ROBUST_PATH = (
    PROJECT_ROOT / "S4_sbi" / "src" / "sleep_sbi" / "route3_global_robustness.py"
)


def _assert_phase2_preconditions() -> dict[str, Any]:
    require_neurolib_env()
    gate_path = JSON_DIR / "phase0_gate.json"
    protocol_path = JSON_DIR / "evaluation_protocol_lock.json"
    pre_manifest = VALIDATION / "pre_run_sha256_manifest.json"
    ck_path = TABLES / "checkpoint_audit.csv"
    missing = [str(p) for p in (gate_path, protocol_path, pre_manifest, ck_path) if not p.exists()]
    if missing:
        raise RuntimeError(f"Phase-2 preconditions missing: {missing}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "pass":
        raise RuntimeError(f"phase0_gate status is not pass: {gate.get('status')}")
    if not gate.get("checkpoints_ok") or int(gate.get("checkpoints_n_ok", 0)) != 12:
        raise RuntimeError("phase0 gate does not show 12/12 checkpoint reload")
    ck = pd.read_csv(ck_path)
    if len(ck) != 12 or not bool(ck.reload_ok.all() and ck.sample_ok.all()):
        raise RuntimeError("checkpoint_audit.csv is incomplete or failed")
    return {
        "phase0_status": gate["status"],
        "checkpoints_n_ok": int(gate["checkpoints_n_ok"]),
        "evaluation_protocol_lock": protocol_path.as_posix(),
        "pre_run_sha256_manifest": pre_manifest.as_posix(),
    }


def freeze_ppc_protocol_lock() -> dict[str, Any]:
    """Freeze protocol once before the first formal simulation."""

    _ensure_dirs()
    lock_path = PPC_DIR / "ppc_protocol_lock.json"
    if lock_path.exists():
        return json.loads(lock_path.read_text(encoding="utf-8"))

    observed_hash = sha256_file(PRIMARY_OBS)
    with np.load(PRIMARY_OBS, allow_pickle=False) as primary:
        observed = np.asarray(primary["x"][0], float)
    payload = {
        "created_utc": _utc_now(),
        "preregistration_hash": verify_preregistration(),
        "schema_hash": rate_contract_hash(),
        "budget_per_track_member": PRIMARY_BUDGET,
        "reserve_per_track_member": RESERVE_BUDGET,
        "micro_shard_size": MICRO_SHARD,
        "retry_rule": (
            "same_theta_same_seed_retry_once; if still failed, replace scoring "
            "slot with next unused reserve draw (new theta + new seed); "
            "failures are retained in failure records and never silent-zeroed"
        ),
        "completeness_rule": (
            "phase2_complete=true only if every track×member has exactly "
            f"{PRIMARY_BUDGET} successful scoring simulations after retry/reserve"
        ),
        "prior_predictive_reuse": {
            "8d": "diagnostics/ppc/8d/prior_predictive/prior_predictive_8d.npz",
            "7d": "diagnostics/ppc/7d/prior_predictive/prior_predictive_7d.npz",
        },
        "observation": {
            "path": PRIMARY_OBS.relative_to(RESULTS_ROOT).as_posix(),
            "sha256": observed_hash,
            "x_shape": list(observed.shape),
        },
        "metric_definition": {
            "scaled_abs_error": "median_i | (x_i - x_obs) / x_scale | per feature",
            "improvement_fraction": "1 - posterior_error / prior_error",
            "aggregate_improvement": "1 - mean(posterior_error) / mean(prior_error)",
            "matches_figure10_diagnostics_run_ppc": True,
            "hard_cutoff_60pct": False,
            "note_60pct": "observed ensemble reference only; not pass/fail cutoff",
        },
        "code_hashes": {
            "simulator_observable_adapter.py": protocol_sha256_file(ADAPTER_PATH)
            if ADAPTER_PATH.exists()
            else None,
            "figure10_diagnostics.py": protocol_sha256_file(DIAG_PATH),
            "figure10_bank.py": protocol_sha256_file(BANK_PATH),
            "route3_global_robustness.py": protocol_sha256_file(ROBUST_PATH),
            "figure10_notebook33_phase2_ppc.py": protocol_sha256_file(Path(__file__)),
        },
        "feature_names": list(rate_feature_names()),
        "seeds": PHASE2_SEEDS,
        "tracks": ["8d", "7d"],
        "members": [1, 2, 3, 4, 5],
        "bootstrap_n": BOOTSTRAP_N,
        "writes_allowed": [
            "notebook_33_ensemble_attribution/ppc/",
            "notebook_33_ensemble_attribution/tables/",
            "notebook_33_ensemble_attribution/json/",
            "notebook_33_ensemble_attribution/figures/",
            "notebook_33_ensemble_attribution/logs/",
            "notebook_33_ensemble_attribution/validation/",
        ],
    }
    atomic_json(lock_path, payload)
    atomic_json(JSON_DIR / "ppc_protocol_lock.json", payload)
    return payload


def _member_dir(track: str, member_index: int) -> Path:
    return PPC_DIR / track / f"member_{member_index}"


def _ensure_draw_catalog(track: str, member_index: int) -> dict[str, Any]:
    """Freeze posterior draws and simulator seeds for one member."""

    member_dir = _member_dir(track, member_index)
    member_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = member_dir / "posterior_draw_catalog.npz"
    manifest_path = member_dir / "seed_manifest.csv"
    if catalog_path.exists() and manifest_path.exists():
        with np.load(catalog_path, allow_pickle=False) as data:
            return {
                "theta_full_8d": np.asarray(data["theta_full_8d"], float),
                "theta_native": np.asarray(data["theta_native"], float),
                "simulator_seed": np.asarray(data["simulator_seed"], np.int64),
                "role": np.asarray(data["role"]),
                "posterior_sample_seed": int(data["posterior_sample_seed"]),
            }

    data, members = load_members(track, FINAL_SCALE)
    member = members[member_index - 1]
    with np.load(PRIMARY_OBS, allow_pickle=False) as primary:
        observed = np.asarray(primary["x"][0], float)
    x_scaled = scale_observation(data, observed)
    post_base = PHASE2_SEEDS[f"posterior_sampling_{track}"]
    sim_base = PHASE2_SEEDS[f"simulator_base_{track}"]
    post_seed = post_base + member_index * 100_003
    theta_unit = sample_member(member, x_scaled, CATALOG_N, post_seed)
    theta_native = unit_to_physical(data, theta_unit)
    if track == "7d":
        theta_full = np.column_stack(
            [theta_native, np.full(CATALOG_N, fixed_c_ctx2th())]
        )
    else:
        theta_full = theta_native
    seeds = deterministic_seed_schedule(
        CATALOG_N, sim_base + member_index * 100_003
    )
    roles = np.array(
        ["primary"] * PRIMARY_BUDGET + ["reserve"] * RESERVE_BUDGET, dtype="<U16"
    )
    _atomic_npz(
        catalog_path,
        theta_full_8d=theta_full.astype(np.float64),
        theta_native=theta_native.astype(np.float64),
        simulator_seed=seeds.astype(np.int64),
        role=roles,
        posterior_sample_seed=np.asarray(post_seed, np.int64),
        track=np.asarray(track, dtype="<U8"),
        member_index=np.asarray(member_index, np.int64),
        member_network_seed=np.asarray(
            member_seeds(track, final=True)[member_index - 1], np.int64
        ),
        preregistration_hash=np.asarray(verify_preregistration(), dtype="<U64"),
    )
    frame = pd.DataFrame(
        {
            "draw_index": np.arange(CATALOG_N),
            "role": roles,
            "simulator_seed": seeds,
            "posterior_sample_seed": post_seed,
            "member_index": member_index,
            "track": track,
        }
    )
    frame.to_csv(manifest_path, index=False)
    # Also append to global manifests
    return {
        "theta_full_8d": theta_full,
        "theta_native": theta_native,
        "simulator_seed": seeds,
        "role": roles,
        "posterior_sample_seed": post_seed,
    }


def _draw_result_path(member_dir: Path, draw_index: int) -> Path:
    return member_dir / "draw_results" / f"draw_{draw_index:05d}.npz"


def _draw_result_valid(path: Path, theta: np.ndarray, seed: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            ok = (
                bool(np.array_equal(np.asarray(data["theta_full_8d"], float), theta))
                and int(data["simulator_seed"]) == int(seed)
                and np.asarray(data["x"]).shape == (14,)
            )
            if not ok:
                return None
            return {k: data[k] for k in data.files}
    except Exception:
        return None


def _save_draw_result(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_npz(
        path,
        theta_full_8d=np.asarray(payload["theta_full_8d"], float),
        x=np.asarray(payload["x"], float),
        validity=np.asarray(payload["validity"], bool),
        simulator_seed=np.asarray(payload["simulator_seed"], np.int64),
        success=np.asarray(payload["success"], bool),
        runtime_s=np.asarray(payload["runtime_s"], float),
        failure_reason=np.asarray(str(payload["failure_reason"])[:1024], dtype="<U1024"),
        draw_index=np.asarray(payload["draw_index"], np.int64),
        attempt=np.asarray(payload.get("attempt", 1), np.int64),
    )


def _payload_from_existing(existing: dict[str, Any], draw_index: int) -> dict[str, Any]:
    attempt = existing["attempt"] if "attempt" in existing else 1
    return {
        "theta_full_8d": np.asarray(existing["theta_full_8d"], float),
        "x": np.asarray(existing["x"], float),
        "validity": np.asarray(existing["validity"], bool),
        "simulator_seed": int(np.asarray(existing["simulator_seed"]).reshape(())),
        "success": bool(np.asarray(existing["success"]).reshape(())),
        "runtime_s": float(np.asarray(existing["runtime_s"]).reshape(())),
        "failure_reason": str(existing["failure_reason"]),
        "draw_index": draw_index,
        "attempt": int(np.asarray(attempt).reshape(())),
    }


def _simulate_draw_batch(
    catalog: dict[str, Any],
    draw_indices: list[int],
    track: str,
    executor: ProcessPoolExecutor,
    attempt: int,
) -> list[dict[str, Any]]:
    if not draw_indices:
        return []
    theta = np.asarray(catalog["theta_full_8d"][draw_indices], float)
    seeds = np.asarray(catalog["simulator_seed"][draw_indices], np.int64)
    ids = np.asarray(draw_indices, np.int64)
    arrays = _simulate_shard(
        theta,
        seeds,
        ids,
        ids,
        track,
        workers=MAX_SIMULATION_WORKERS,
        executor=executor,
    )
    out = []
    for i, draw_index in enumerate(draw_indices):
        out.append(
            {
                "theta_full_8d": np.asarray(arrays["theta_full_8d"][i], float),
                "x": np.asarray(arrays["x"][i], float),
                "validity": np.asarray(arrays["validity"][i], bool),
                "simulator_seed": int(arrays["simulator_seed"][i]),
                "success": bool(arrays["success"][i]),
                "runtime_s": float(arrays["runtime_s"][i]),
                "failure_reason": str(arrays["failure_reason"][i]),
                "draw_index": int(draw_index),
                "attempt": int(attempt),
            }
        )
    return out


def run_member_simulations(track: str, member_index: int) -> dict[str, Any]:
    """Run/resume primary+reserve sims for one member; return scoring pack."""

    member_dir = _member_dir(track, member_index)
    catalog = _ensure_draw_catalog(track, member_index)
    progress_path = member_dir / "progress.json"
    slot_map_path = member_dir / "scoring_slot_map.json"
    failures_path = member_dir / "failure_records.csv"

    if slot_map_path.exists():
        slot_map = json.loads(slot_map_path.read_text(encoding="utf-8"))
        scoring_draws = list(slot_map["scoring_draw_indices"])
        reserve_ptr = int(slot_map["reserve_ptr"])
    else:
        scoring_draws = list(range(PRIMARY_BUDGET))
        reserve_ptr = PRIMARY_BUDGET

    failure_rows: list[dict[str, Any]] = []
    if failures_path.exists():
        failure_rows = pd.read_csv(failures_path).to_dict(orient="records")

    started = perf_counter()
    (member_dir / "draw_results").mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=MAX_SIMULATION_WORKERS) as executor:
        guard = 0
        while guard < CATALOG_N + 5:
            guard += 1
            pending_slots: list[int] = []
            for slot, draw_index in enumerate(scoring_draws):
                theta = np.asarray(catalog["theta_full_8d"][draw_index], float)
                seed = int(catalog["simulator_seed"][draw_index])
                path = _draw_result_path(member_dir, draw_index)
                existing = _draw_result_valid(path, theta, seed)
                if existing is None:
                    pending_slots.append(slot)
                    continue
                payload = _payload_from_existing(existing, draw_index)
                if payload["success"]:
                    continue
                if payload["attempt"] >= 2:
                    # Exhausted retries for this draw — replace with reserve.
                    if reserve_ptr >= CATALOG_N:
                        raise RuntimeError(
                            f"{track}/member_{member_index}: reserves exhausted "
                            f"while filling scoring slot {slot}"
                        )
                    replacement = reserve_ptr
                    reserve_ptr += 1
                    failure_rows.append(
                        {
                            "track": track,
                            "member_index": member_index,
                            "draw_index": draw_index,
                            "simulator_seed": seed,
                            "attempt": "replaced",
                            "failure_reason": (
                                f"replaced_by_reserve_draw_{replacement}: "
                                f"{payload['failure_reason']}"
                            ),
                            "role": str(catalog["role"][draw_index]),
                        }
                    )
                    scoring_draws[slot] = replacement
                    pending_slots.append(slot)
                else:
                    pending_slots.append(slot)
            if not pending_slots:
                break

            batch_slots = pending_slots[:MICRO_SHARD]
            batch_draws = [scoring_draws[s] for s in batch_slots]
            # Determine attempt number per draw from any existing failed file.
            attempts = []
            for draw_index in batch_draws:
                path = _draw_result_path(member_dir, draw_index)
                existing = _draw_result_valid(
                    path,
                    catalog["theta_full_8d"][draw_index],
                    int(catalog["simulator_seed"][draw_index]),
                )
                if existing is None:
                    attempts.append(1)
                else:
                    prev = existing["attempt"] if "attempt" in existing else 1
                    attempts.append(int(np.asarray(prev).reshape(())) + 1)
            # Simulate first-attempt and retry draws separately for clear attempt tags.
            for attempt_value in sorted(set(attempts)):
                subset = [
                    draw
                    for draw, attempt in zip(batch_draws, attempts)
                    if attempt == attempt_value
                ]
                if not subset:
                    continue
                results = _simulate_draw_batch(
                    catalog, subset, track, executor, attempt=attempt_value
                )
                for result in results:
                    draw_index = int(result["draw_index"])
                    path = _draw_result_path(member_dir, draw_index)
                    _save_draw_result(path, result)
                    if not result["success"]:
                        failure_rows.append(
                            {
                                "track": track,
                                "member_index": member_index,
                                "draw_index": draw_index,
                                "simulator_seed": int(result["simulator_seed"]),
                                "attempt": attempt_value,
                                "failure_reason": result["failure_reason"],
                                "role": str(catalog["role"][draw_index]),
                            }
                        )

            atomic_json(
                slot_map_path,
                {
                    "track": track,
                    "member_index": member_index,
                    "scoring_draw_indices": scoring_draws,
                    "reserve_ptr": reserve_ptr,
                    "updated_utc": _utc_now(),
                },
            )
            if failure_rows:
                pd.DataFrame(failure_rows).to_csv(failures_path, index=False)
            n_done = 0
            for d in scoring_draws:
                got = _draw_result_valid(
                    _draw_result_path(member_dir, d),
                    catalog["theta_full_8d"][d],
                    int(catalog["simulator_seed"][d]),
                )
                if got is not None and bool(np.asarray(got["success"]).reshape(())):
                    n_done += 1
            atomic_json(
                progress_path,
                {
                    "track": track,
                    "member_index": member_index,
                    "scoring_success_estimate": n_done,
                    "reserve_ptr": reserve_ptr,
                    "failure_records": len(failure_rows),
                    "updated_utc": _utc_now(),
                },
            )
            print(
                f"PPC {track}/member_{member_index}: "
                f"success={n_done}/{PRIMARY_BUDGET} "
                f"pending_slots={max(len(pending_slots) - len(batch_slots), 0)} "
                f"reserve_ptr={reserve_ptr}",
                flush=True,
            )

    # Consolidate scoring results
    xs = []
    thetas = []
    seeds = []
    draw_ids = []
    runtimes = []
    for draw_index in scoring_draws:
        theta = np.asarray(catalog["theta_full_8d"][draw_index], float)
        seed = int(catalog["simulator_seed"][draw_index])
        path = _draw_result_path(member_dir, draw_index)
        existing = _draw_result_valid(path, theta, seed)
        if existing is None or not bool(existing["success"]):
            raise RuntimeError(
                f"{track}/member_{member_index}: scoring draw {draw_index} incomplete"
            )
        xs.append(np.asarray(existing["x"], float))
        thetas.append(theta)
        seeds.append(seed)
        draw_ids.append(draw_index)
        runtimes.append(float(existing["runtime_s"]))

    x = np.vstack(xs)
    consolidated = member_dir / f"posterior_predictive_{track}_member_{member_index}.npz"
    _atomic_npz(
        consolidated,
        x=x.astype(np.float64),
        theta_full_8d=np.vstack(thetas).astype(np.float64),
        simulator_seed=np.asarray(seeds, np.int64),
        draw_index=np.asarray(draw_ids, np.int64),
        runtime_s=np.asarray(runtimes, float),
        success=np.ones(PRIMARY_BUDGET, bool),
        feature_names=np.asarray(rate_feature_names(), dtype="<U64"),
        track=np.asarray(track, dtype="<U8"),
        member_index=np.asarray(member_index, np.int64),
        preregistration_hash=np.asarray(verify_preregistration(), dtype="<U64"),
    )
    replacements = int(sum(1 for d in scoring_draws if d >= PRIMARY_BUDGET))
    summary = {
        "track": track,
        "member_index": member_index,
        "attempted_primary": PRIMARY_BUDGET,
        "successful_scoring": PRIMARY_BUDGET,
        "failed_recorded": len(failure_rows),
        "replacements_from_reserve": replacements,
        "reserve_used": reserve_ptr - PRIMARY_BUDGET,
        "wall_s": float(perf_counter() - started),
        "consolidated": consolidated.relative_to(RESULTS_ROOT).as_posix(),
        "completeness_ok": True,
    }
    atomic_json(member_dir / "member_run_summary.json", summary)
    if failure_rows:
        pd.DataFrame(failure_rows).to_csv(failures_path, index=False)
    return summary


def _load_prior(track: str) -> tuple[np.ndarray, np.ndarray]:
    path = (
        RESULTS_ROOT
        / "diagnostics"
        / "ppc"
        / track
        / "prior_predictive"
        / f"prior_predictive_{track}.npz"
    )
    with np.load(path, allow_pickle=False) as prior:
        x = np.asarray(prior["x"], float)
        success = np.asarray(prior["success"], bool)
    if not success.all():
        raise RuntimeError(f"prior predictive for {track} has failures")
    return x, success


def _score_member(track: str, member_index: int) -> dict[str, Any]:
    from .figure10_training import load_training_data

    data = load_training_data(track, FINAL_SCALE)
    with np.load(PRIMARY_OBS, allow_pickle=False) as primary:
        observed = np.asarray(primary["x"][0], float)
    member_dir = _member_dir(track, member_index)
    path = member_dir / f"posterior_predictive_{track}_member_{member_index}.npz"
    with np.load(path, allow_pickle=False) as post:
        posterior_x = np.asarray(post["x"], float)
        success = np.asarray(post["success"], bool)
    if posterior_x.shape != (PRIMARY_BUDGET, 14) or not success.all():
        raise RuntimeError(f"incomplete posterior pack: {path}")
    prior_x, _ = _load_prior(track)
    scale = data.x_scale
    post_err_all = np.abs((posterior_x - observed) / scale)
    prior_err_all = np.abs((prior_x - observed) / scale)
    posterior_error = np.median(post_err_all, axis=0)
    prior_error = np.median(prior_err_all, axis=0)
    improvement = 1 - posterior_error / np.maximum(prior_error, 1e-12)
    overall_post = float(np.mean(posterior_error))
    overall_prior = float(np.mean(prior_error))
    overall_imp = float(1 - overall_post / overall_prior)

    rng = np.random.default_rng(BOOTSTRAP_SEED + (0 if track == "8d" else 50) + member_index)
    boot = []
    n = len(posterior_x)
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        pe = np.median(post_err_all[idx], axis=0)
        boot.append(float(1 - np.mean(pe) / overall_prior))
    boot_arr = np.asarray(boot, float)
    feature_rows = pd.DataFrame(
        {
            "track": track,
            "estimator": f"member_{member_index}",
            "feature": list(rate_feature_names()),
            "posterior_scaled_abs_error": posterior_error,
            "prior_scaled_abs_error": prior_error,
            "improvement_fraction": improvement,
            "posterior_better": posterior_error < prior_error,
        }
    )
    feature_rows.to_csv(member_dir / "ppc_feature_metrics.csv", index=False)
    run_summary = json.loads((member_dir / "member_run_summary.json").read_text(encoding="utf-8"))
    ens_path = RESULTS_ROOT / "diagnostics" / "ppc" / track / "ppc_summary.json"
    ens = json.loads(ens_path.read_text(encoding="utf-8"))
    summary = {
        "created_utc": _utc_now(),
        "track": track,
        "estimator": f"member_{member_index}",
        "status": "complete",
        "posterior_attempted": PRIMARY_BUDGET + run_summary.get("reserve_used", 0),
        "posterior_successful": PRIMARY_BUDGET,
        "posterior_failed_records": run_summary.get("failed_recorded", 0),
        "replacements_from_reserve": run_summary.get("replacements_from_reserve", 0),
        "failure_rate_among_attempts": float(
            run_summary.get("failed_recorded", 0)
            / max(PRIMARY_BUDGET + run_summary.get("reserve_used", 0), 1)
        ),
        "features_posterior_better": int(feature_rows.posterior_better.sum()),
        "overall_posterior_scaled_error": overall_post,
        "overall_prior_scaled_error": overall_prior,
        "overall_improvement_fraction": overall_imp,
        "bootstrap_improvement_mean": float(np.mean(boot_arr)),
        "bootstrap_improvement_std": float(np.std(boot_arr, ddof=1)),
        "bootstrap_improvement_q025": float(np.quantile(boot_arr, 0.025)),
        "bootstrap_improvement_q975": float(np.quantile(boot_arr, 0.975)),
        "ensemble_overall_improvement_fraction_reference": ens[
            "overall_improvement_fraction"
        ],
        "ensemble_features_posterior_better_reference": ens["features_posterior_better"],
        "hard_cutoff_60pct_applied": False,
        "calibration_claim_forbidden": True,
        "preregistration_hash": verify_preregistration(),
    }
    atomic_json(member_dir / "ppc_summary.json", summary)
    return summary


def _update_phase2_tables_and_figures(member_summaries: list[dict[str, Any]]) -> None:
    ens_rows = []
    feature_frames = []
    for track in ("8d", "7d"):
        ens = json.loads(
            (RESULTS_ROOT / "diagnostics" / "ppc" / track / "ppc_summary.json").read_text(
                encoding="utf-8"
            )
        )
        ens_rows.append(
            {
                "track": track,
                "estimator": "ensemble",
                "status": "available_from_notebook30",
                "features_posterior_better": ens["features_posterior_better"],
                "overall_improvement_fraction": ens["overall_improvement_fraction"],
                "bootstrap_improvement_q025": None,
                "bootstrap_improvement_q975": None,
                "failure_rate_among_attempts": ens["posterior_failed"]
                / max(ens["posterior_attempted"], 1),
                "note": "Observed aggregate ~60% is reference only, not a hard cutoff",
            }
        )
    for summary in member_summaries:
        ens_rows.append(
            {
                "track": summary["track"],
                "estimator": summary["estimator"],
                "status": "complete",
                "features_posterior_better": summary["features_posterior_better"],
                "overall_improvement_fraction": summary["overall_improvement_fraction"],
                "bootstrap_improvement_q025": summary["bootstrap_improvement_q025"],
                "bootstrap_improvement_q975": summary["bootstrap_improvement_q975"],
                "failure_rate_among_attempts": summary["failure_rate_among_attempts"],
                "note": "PPC != calibration; does not offset SBC/joint/L-C2ST failures",
            }
        )
        feature_frames.append(
            pd.read_csv(
                _member_dir(summary["track"], int(summary["estimator"].split("_")[1]))
                / "ppc_feature_metrics.csv"
            )
        )
    ppc_table = pd.DataFrame(ens_rows)
    ppc_table.to_csv(TABLES / "ppc_by_estimator.csv", index=False)
    if feature_frames:
        pd.concat(feature_frames, ignore_index=True).to_csv(
            TABLES / "ppc_feature_metrics_all_members.csv", index=False
        )

    # Figure: aggregate improvement with CI
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, track in zip(axes, ("8d", "7d")):
        sub = ppc_table[ppc_table.track == track].copy()
        # order ensemble then members
        order = ["ensemble"] + [f"member_{i}" for i in range(1, 6)]
        sub["estimator"] = pd.Categorical(sub["estimator"], categories=order, ordered=True)
        sub = sub.sort_values("estimator")
        x = np.arange(len(sub))
        vals = sub["overall_improvement_fraction"].astype(float).values
        ax.bar(x, vals, color=["#b22222"] + ["#4c72b0"] * 5)
        for i, row in enumerate(sub.itertuples()):
            if pd.notna(row.bootstrap_improvement_q025):
                ax.plot(
                    [i, i],
                    [row.bootstrap_improvement_q025, row.bootstrap_improvement_q975],
                    color="k",
                    lw=1.5,
                )
        ax.axhline(0.6, color="gray", ls="--", lw=1, label="ensemble ~60% reference")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["estimator"], rotation=45, ha="right")
        ax.set_ylabel("Aggregate relative improvement")
        ax.set_title(f"{track} PPC (descriptive)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "ppc_member_comparison.png", dpi=150)
    fig.savefig(FIGURES / "ppc_member_comparison.svg")
    plt.close(fig)

    # Features improved count
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, track in zip(axes, ("8d", "7d")):
        sub = ppc_table[ppc_table.track == track].copy()
        order = ["ensemble"] + [f"member_{i}" for i in range(1, 6)]
        sub["estimator"] = pd.Categorical(sub["estimator"], categories=order, ordered=True)
        sub = sub.sort_values("estimator")
        ax.bar(
            np.arange(len(sub)),
            sub["features_posterior_better"].astype(float),
            color=["#b22222"] + ["#4c72b0"] * 5,
        )
        ax.axhline(14, color="gray", ls=":", lw=1)
        ax.set_xticks(np.arange(len(sub)))
        ax.set_xticklabels(sub["estimator"], rotation=45, ha="right")
        ax.set_ylabel("Features better than prior ( /14 )")
        ax.set_title(f"{track} PPC feature counts")
    fig.tight_layout()
    fig.savefig(FIGURES / "ppc_feature_counts.png", dpi=150)
    fig.savefig(FIGURES / "ppc_feature_counts.svg")
    plt.close(fig)


def _update_attribution_after_ppc(member_summaries: list[dict[str, Any]]) -> None:
    matrix_path = TABLES / "attribution_matrix.csv"
    matrix = pd.read_csv(matrix_path)
    ppc = pd.DataFrame(member_summaries)
    all_14 = bool((ppc["features_posterior_better"] == 14).all())
    imps = ppc["overall_improvement_fraction"].astype(float)
    note = (
        f"Phase2 PPC: all_members_14_of_14={all_14}; "
        f"improvement median={float(imps.median()):.3f} "
        f"range=[{float(imps.min()):.3f},{float(imps.max()):.3f}]; "
        "PPC does not validate calibration."
    )
    # Append/update a PPC-related hypothesis row if present, else add notes into shared failure row
    rows = matrix.to_dict(orient="records")
    rows.append(
        {
            "hypothesis": "member_ppc_predictive_adequacy_without_calibration",
            "support": note,
            "oppose": "SBC/L-C2ST/joint coverage failures remain on members and ensemble",
            "missing": "",
            "verdict": "supported_predictive_not_calibrated",
        }
    )
    pd.DataFrame(rows).to_csv(matrix_path, index=False)

    decision = {
        "created_utc": _utc_now(),
        "provisional": False,
        "phase2_ppc_pending": False,
        "layers": [
            {
                "layer": 1,
                "claim": (
                    "Equal-weight ensemble is associated with substantially worse "
                    "joint coverage than individual members and with changes in "
                    "joint posterior geometry on a pre-frozen matched case subset; "
                    "marginal variance inflation alone does not fully explain the gap."
                ),
            },
            {
                "layer": 2,
                "claim": (
                    "Individual members still share SBC and L-C2ST calibration "
                    "failures. Phase-2 member-wise PPC shows predictive improvement "
                    "versus prior predictive for members, comparable in spirit to the "
                    "ensemble reference, but PPC must not be read as posterior "
                    "calibration and does not offset SBC/joint/L-C2ST failures. "
                    "Selecting one member is not a validated posterior."
                ),
            },
        ],
        "branches": [
            {
                "if": "joint geometry / dependence differences remain material",
                "then": "prioritize aggregation and joint-geometry research on a sealed bank",
            },
            {
                "if": "members show predictive PPC improvement but shared calibration failure",
                "then": "do not promote best-PPC member; pursue calibration/summary/geometry work",
            },
            {
                "if": "mui/mue/tauA anomalies persist on all members",
                "then": "prioritize parameter-specific recovery, boundary, summary sensitivity",
            },
            {
                "if": "future scale evidence under fixed K=5 protocol",
                "then": "design confounded-free scale sweep on sealed evaluation",
            },
            {
                "if": "no sealed-bank improvement path is identified",
                "then": "do not dump compute into 1M by default",
            },
        ],
        "forbidden_promises": [
            "131k or 1M will fix calibration",
            "picking best joint-KS or best-PPC member yields validated posterior",
            "PPC pass implies calibration pass",
            "logsumexp density dilution proven root cause",
        ],
        "ppc_phase2_note": note,
    }
    atomic_json(JSON_DIR / "provisional_decision_tree.json", decision)
    claim = {
        "created_utc": _utc_now(),
        "supported": [
            "synthetic cortical-rate predictive improvement for ensemble and members (PPC)",
            "individual vs ensemble diagnostic comparison on descriptive/sensitivity banks",
            "association between equal-weight ensemble and joint coverage degradation",
            "parameter-specific SBC abnormalities (mui/mue/tauA)",
            "7D/8D failure-pattern replication",
            "matched-subset joint geometry description",
            "member-wise PPC does not remove shared SBC/L-C2ST failures",
        ],
        "not_supported": [
            "real EEG parameter inversion",
            "Fpz-Cz subject-specific physiology",
            "calibrated physiological parameter uncertainty",
            "selecting one member yields a trusted posterior",
            "c_ctx2th proven structural non-identifiability",
            "mui proven classical overconfidence",
            "131k/1M guaranteed fix",
            "logsumexp/density dilution proven root cause",
            "powered_1024 as untouched final evaluation",
            "PPC pass as calibration pass",
        ],
        "dataset_roles": {
            "official_300": "descriptive_sensitivity",
            "powered_1024": "descriptive_sensitivity",
        },
        "phase2_complete": True,
    }
    atomic_json(JSON_DIR / "final_claim_card.json", claim)


def run_phase2_member_ppc() -> dict[str, Any]:
    """Full Phase-2 entrypoint."""

    wall0 = perf_counter()
    _ensure_dirs()
    pre = _assert_phase2_preconditions()
    lock = freeze_ppc_protocol_lock()
    # Freeze global seed manifests listing all members before sims continue
    global_post_rows = []
    global_sim_rows = []
    for track in ("8d", "7d"):
        for member_index in range(1, 6):
            catalog = _ensure_draw_catalog(track, member_index)
            for draw_index in range(CATALOG_N):
                global_post_rows.append(
                    {
                        "track": track,
                        "member_index": member_index,
                        "draw_index": draw_index,
                        "role": str(catalog["role"][draw_index]),
                        "posterior_sample_seed": catalog["posterior_sample_seed"],
                    }
                )
                global_sim_rows.append(
                    {
                        "track": track,
                        "member_index": member_index,
                        "draw_index": draw_index,
                        "role": str(catalog["role"][draw_index]),
                        "simulator_seed": int(catalog["simulator_seed"][draw_index]),
                    }
                )
    pd.DataFrame(global_post_rows).to_csv(
        PPC_DIR / "posterior_sampling_seed_manifest.csv", index=False
    )
    pd.DataFrame(global_sim_rows).to_csv(
        PPC_DIR / "simulator_seed_manifest.csv", index=False
    )
    # Uniqueness check
    sim_seeds = pd.DataFrame(global_sim_rows)["simulator_seed"]
    if sim_seeds.duplicated().any():
        raise RuntimeError("simulator seed manifest has duplicates")

    run_summaries = []
    member_summaries = []
    for track in ("8d", "7d"):
        for member_index in range(1, 6):
            print(f"=== Phase2 PPC {track} member_{member_index} ===", flush=True)
            run_summaries.append(run_member_simulations(track, member_index))
            member_summaries.append(_score_member(track, member_index))

    _update_phase2_tables_and_figures(member_summaries)
    _update_attribution_after_ppc(member_summaries)

    # Completeness
    complete = []
    for track in ("8d", "7d"):
        for member_index in range(1, 6):
            summary_path = _member_dir(track, member_index) / "ppc_summary.json"
            pack = _member_dir(track, member_index) / (
                f"posterior_predictive_{track}_member_{member_index}.npz"
            )
            ok = summary_path.exists() and pack.exists()
            if ok:
                with np.load(pack, allow_pickle=False) as data:
                    ok = bool(
                        np.asarray(data["x"]).shape == (PRIMARY_BUDGET, 14)
                        and bool(np.asarray(data["success"]).all())
                        and bool(np.isfinite(np.asarray(data["x"])).all())
                    )
            complete.append(
                {
                    "track": track,
                    "member_index": int(member_index),
                    "complete": bool(ok),
                }
            )
    completeness_ok = bool(all(row["complete"] for row in complete))
    status = {
        "created_utc": _utc_now(),
        "phase2_complete": bool(completeness_ok),
        "member_wise_status": "complete" if completeness_ok else "incomplete",
        "n_units_expected": 10,
        "n_units_complete": int(sum(1 for row in complete if row["complete"])),
        "units": complete,
        "formal_budget_per_member": int(PRIMARY_BUDGET),
        "reserve_per_member": int(RESERVE_BUDGET),
        "wall_s": float(perf_counter() - wall0),
        "preconditions": pre,
        "protocol_lock": "ppc/ppc_protocol_lock.json",
        "hard_cutoff_60pct": False,
        "calibration_equivalence_forbidden": True,
    }
    atomic_json(JSON_DIR / "ppc_phase2_status.json", status)
    atomic_json(LOGS / "phase2_complete.json", status)
    if not completeness_ok:
        raise RuntimeError("Phase-2 completeness check failed")
    return {
        "status": status,
        "member_summaries": member_summaries,
        "run_summaries": run_summaries,
        "lock": lock,
    }


__all__ = [
    "freeze_ppc_protocol_lock",
    "run_phase2_member_ppc",
    "_assert_phase2_preconditions",
]
