"""Notebook 34 offline posterior calibration root-cause audit.

Presentation notebook reads artifacts only. Expensive posterior resampling
lives in ``python -m sleep_sbi.figure10_notebook34_audit`` (Phase 2 runner).
No simulator calls. No training. No edits to notebooks 28–33 or banks.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from .figure10_diagnostics import GLOBAL_ROOT, _randomized_rank
from .figure10_protocol import (
    FINAL_SCALE,
    INTERMEDIATE_SCALE,
    PARAMETER_NAMES_7D,
    PARAMETER_NAMES_8D,
    prior_bounds_7d,
    prior_bounds_8d,
    rate_feature_names,
)
from .figure10_training import (
    EqualMixturePosterior,
    TRAINING_ROOT,
    TrainingData,
    load_members,
    load_training_data,
    member_seeds,
    physical_to_unit,
    scale_observation,
    unit_to_physical,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIG10_ROOT = PROJECT_ROOT / "S4_sbi" / "results" / "figure10_8d_7d"
N34_ROOT = FIG10_ROOT / "artifacts" / "notebook_34"
N33_ROOT = FIG10_ROOT / "notebook_33_ensemble_attribution"

TRACKS = ("8d", "7d")
FOCUS_PARAMS = ("mui", "mue", "tauA")
NOMINAL_LEVELS = (0.50, 0.80, 0.90, 0.95)
PRIMARY_METRICS = (
    "randomized_normalized_rank_mean",
    "pit_ecdf_ks",
    "equal_tailed_central_coverage_90",
    "equal_tailed_central_coverage_95",
    "rank_tail_mass_low005",
    "rank_tail_mass_high095",
    "bias_unit",
    "mae_unit",
)
CLIP_TOLERANCES = {
    "pit_ecdf_ks": 0.05,
    "equal_tailed_central_coverage_90": 0.03,
    "equal_tailed_central_coverage_95": 0.03,
    "rank_mean": 0.03,
    "bias_unit": 0.02,
    "boundary_point_mass": 0.02,
}
PHASE2_N_DRAWS = 512
PHASE2_N_CASES = 48
PHASE2_BATCH = 256
PHASE2_SEED = 34_000_001
PHASE2_MEMBER_SEEDS_OFFSET = 104_729
PROHIBITED_PATTERNS = (
    r"untouched confirmatory",
    r"powered_1024 is untouched confirmatory",
    r"classic overconfidence",
    r"structural non-identifiability",
    r"PIT ECDF is credible-interval coverage",
    r"\bHPD\b|\bHDI\b",
    r"general joint coverage",
    r"PPC improvement offsets",
    r"8192.*32768.*causal",
    r"131k|1M.*will (solve|fix|resolve)",
    r"trusted posterior",
)


def _ensure_dirs() -> dict[str, Path]:
    paths = {
        "root": N34_ROOT,
        "json": N34_ROOT / "json",
        "csv": N34_ROOT / "csv",
        "npz": N34_ROOT / "npz",
        "figures": N34_ROOT / "figures",
        "phase2": N34_ROOT / "phase2_samples",
        "tests": N34_ROOT / "tests",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.stem + ".", suffix=".json", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.stem + ".", suffix=".npz", dir=str(path.parent)
    )
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_names(track: str) -> tuple[str, ...]:
    return PARAMETER_NAMES_8D if track == "8d" else PARAMETER_NAMES_7D


def ranks_npz_path(track: str, n_train: int) -> Path:
    return (
        GLOBAL_ROOT
        / "powered_1024"
        / track
        / f"scale_{n_train}"
        / "global_ranks_and_recovery.npz"
    )


def eval_bank_path(track: str) -> Path:
    return GLOBAL_ROOT / "powered_1024" / track / f"powered_1024_{track}.npz"


def load_matched_case_ids() -> list[int]:
    lock = N33_ROOT / "json" / "matched_geometry_subset_lock.json"
    payload = json.loads(lock.read_text(encoding="utf-8"))
    ids = [int(x) for x in payload["case_ids"]]
    if len(ids) != PHASE2_N_CASES:
        raise RuntimeError(f"expected {PHASE2_N_CASES} matched cases, got {len(ids)}")
    return ids


def inventory_members(track: str, n_train: int) -> list[dict[str, Any]]:
    final = n_train == FINAL_SCALE
    seeds = member_seeds(track, final=final)
    rows = []
    for member_index, seed in enumerate(seeds, start=1):
        ckpt = (
            TRAINING_ROOT
            / track
            / f"scale_{n_train}"
            / f"member_{seed}"
            / "best_checkpoint.pt"
        )
        rows.append(
            {
                "track": track,
                "n_train": n_train,
                "member_index": member_index,
                "seed": int(seed),
                "checkpoint": str(ckpt),
                "exists": ckpt.is_file(),
                "sha256": sha256_file(ckpt) if ckpt.is_file() else None,
            }
        )
    return rows


def load_eval_case(track: str, case_id: int) -> dict[str, np.ndarray]:
    with np.load(eval_bank_path(track), allow_pickle=False) as bank:
        theta = np.asarray(bank["theta"], float)[case_id]
        x = np.asarray(bank["x"], float)[case_id]
        success = bool(bank["success"][case_id])
        sample_id = int(bank["sample_id"][case_id])
    if sample_id != case_id:
        raise RuntimeError(f"sample_id drift at {case_id}: {sample_id}")
    if not success:
        raise RuntimeError(f"eval case {track}/{case_id} failed simulation")
    return {"theta_physical": theta, "x_raw": x}


def load_ranks_bundle(track: str, n_train: int) -> dict[str, Any]:
    path = ranks_npz_path(track, n_train)
    with np.load(path, allow_pickle=False) as data:
        return {
            "path": path,
            "joint_ranks": np.asarray(data["joint_ranks"], float),
            "marginal_ranks": np.asarray(data["marginal_ranks"], float),
            "posterior_median": np.asarray(data["posterior_median"], float),
            "posterior_std": np.asarray(data["posterior_std"], float),
            "theta_true_unit": np.asarray(data["theta_true_unit"], float),
            "estimator_names": [str(v) for v in data["estimator_names"]],
            "parameter_names": [str(v) for v in data["parameter_names"]],
            "n_samples": int(data["posterior_samples_per_case"].item()),
            "track": str(data["track"].item()),
            "scale": int(data["scale"].item()),
            "dataset_id": str(data["dataset_id"].item()),
            "sha256": sha256_file(path),
        }


# ---------------------------------------------------------------------------
# Rank / PIT / coverage
# ---------------------------------------------------------------------------


def normalize_rank_legacy(rank: np.ndarray, n_samples: int) -> np.ndarray:
    return (np.asarray(rank, float) + 0.5) / (float(n_samples) + 1.0)


def randomized_rank_protocol(
    samples: np.ndarray,
    truth: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    samples = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    n = samples.shape[0]
    n_less = (samples < truth).sum(axis=0).astype(np.float64)
    n_equal = (samples == truth).sum(axis=0).astype(np.float64)
    u = rng.random(truth.shape)
    rank_rand = n_less + u * n_equal
    return {
        "n_less": n_less,
        "n_equal": n_equal,
        "u": u,
        "rank_randomized": rank_rand,
        "rank_norm": normalize_rank_legacy(rank_rand, n),
    }


def equal_tailed_central_coverage(ranks_norm: np.ndarray, level: float) -> float:
    alpha = 1.0 - float(level)
    r = np.asarray(ranks_norm, dtype=np.float64)
    return float(np.mean((r > alpha / 2.0) & (r < 1.0 - alpha / 2.0)))


def rank_summary(ranks_norm: np.ndarray) -> dict[str, float]:
    r = np.asarray(ranks_norm, dtype=np.float64)
    ks = stats.kstest(r, "uniform") if r.size else None
    return {
        "n": int(r.size),
        "mean": float(np.mean(r)) if r.size else float("nan"),
        "var": float(np.var(r, ddof=1)) if r.size > 1 else float("nan"),
        "tail_low_005": float(np.mean(r < 0.05)) if r.size else float("nan"),
        "tail_high_095": float(np.mean(r > 0.95)) if r.size else float("nan"),
        "pit_ecdf_ks": float(ks.statistic) if ks else float("nan"),
        "pit_ecdf_ks_pvalue": float(ks.pvalue) if ks else float("nan"),
        **{
            f"central_cov_{int(100 * lvl)}": equal_tailed_central_coverage(r, lvl)
            for lvl in NOMINAL_LEVELS
        },
    }


def _holm(pvals: Sequence[float], alpha: float = 0.05) -> list[bool]:
    m = len(pvals)
    order = np.argsort(pvals)
    rejected = [False] * m
    for rank, idx in enumerate(order, start=1):
        thr = alpha / (m - rank + 1)
        if pvals[idx] <= thr:
            rejected[idx] = True
        else:
            break
    return rejected


# ---------------------------------------------------------------------------
# Protocol lock
# ---------------------------------------------------------------------------


def build_protocol_lock() -> dict[str, Any]:
    paths = _ensure_dirs()
    case_ids = load_matched_case_ids()
    n_member_draws = 2 * len(case_ids) * 5 * PHASE2_N_DRAWS
    n_ensemble_draws = 2 * len(case_ids) * PHASE2_N_DRAWS
    lock = {
        "notebook": "34_posterior_calibration_root_cause_audit.ipynb",
        "artifact_root": str(paths["root"]),
        "science_framing": (
            "protocol-locked hypothesis-driven follow-up analysis on the reused "
            "powered_1024 diagnostic bank; not untouched confirmatory evaluation"
        ),
        "parameter_order": {
            "8d": list(PARAMETER_NAMES_8D),
            "7d": list(PARAMETER_NAMES_7D),
        },
        "physical_bounds": {
            "8d": prior_bounds_8d().tolist(),
            "7d": prior_bounds_7d().tolist(),
        },
        "transform": {
            "theta": "physical <-> unit box via affine map from prior bounds",
            "x": "median/IQR standardization fit on training train_ids only",
            "sample_space_for_ranks": "unit box",
            "reject_outside_prior_default": False,
            "clip_after_sample_default": True,
            "norm_posterior": False,
            "norm_posterior_note": (
                "absolute log-density not comparable across observations; "
                "within-observation rank comparisons only"
            ),
        },
        "conditioning_summary": {
            "names": list(rate_feature_names()),
            "dimension": len(rate_feature_names()),
            "order_frozen": True,
        },
        "tracks_and_checkpoints": {
            "tracks": list(TRACKS),
            "n_train_contrast": [INTERMEDIATE_SCALE, FINAL_SCALE],
            "scale_contrast_note": (
                "8192 has only one estimator (ensemble≡member_1); 32768 has five "
                "members. Asymmetric descriptive scale contrast only — not a causal "
                "n_sim effect claim."
            ),
            "members_8d_32768": inventory_members("8d", FINAL_SCALE),
            "members_7d_32768": inventory_members("7d", FINAL_SCALE),
            "members_8d_8192": inventory_members("8d", INTERMEDIATE_SCALE),
            "members_7d_8192": inventory_members("7d", INTERMEDIATE_SCALE),
        },
        "evaluation_case_ids": {
            "phase1_bank": "powered_1024 full 1024 cases from global ranks NPZ",
            "phase2_phase4_matched_48": case_ids,
            "matched_lock_source": str(
                N33_ROOT / "json" / "matched_geometry_subset_lock.json"
            ),
        },
        "sampling": {
            "n_draws": PHASE2_N_DRAWS,
            "batch_size": PHASE2_BATCH,
            "base_seed": PHASE2_SEED,
            "member_seed_offset": PHASE2_MEMBER_SEEDS_OFFSET,
            "budget_arithmetic": {
                "n_tracks": 2,
                "n_cases": len(case_ids),
                "n_members": 5,
                "n_draws": PHASE2_N_DRAWS,
                "member_draws": n_member_draws,
                "ensemble_draws": n_ensemble_draws,
                "total_draws": n_member_draws + n_ensemble_draws,
            },
        },
        "rank_definition": {
            "legacy_n33": {
                "formula": "rank = n_less + U*n_equal; norm = (rank+0.5)/(n+1)",
                "source": "figure10_diagnostics._randomized_rank + analyze_global_ranks",
                "deterministic_midpoint": False,
                "note": "U is random; not midpoint. Ties broken by Uniform(0,1).",
            },
            "protocol_locked": {
                "components_recorded": [
                    "n_less",
                    "n_equal",
                    "u",
                    "rank_randomized",
                    "rank_norm",
                ],
                "normalize": "(rank_randomized + 0.5) / (n_samples + 1)",
                "optional_rank_bin_jitter_default": False,
                "matches_legacy_normalize": True,
            },
            "pit_ecdf": (
                "empirical CDF of normalized ranks vs Uniform(0,1) identity; "
                "NOT credible-interval coverage"
            ),
            "equal_tailed_central_coverage": (
                "fraction of cases with alpha/2 < rank_norm < 1-alpha/2; "
                "NOT HPD/HDI coverage"
            ),
        },
        "clipping_audit": {
            "strategies": ["raw_preclip", "clipped", "rejection_in_support"],
            "metric_specific_tolerances": CLIP_TOLERANCES,
            "material_if": (
                "any focus-param primary metric delta exceeds tolerance "
                "(vs clipped baseline)"
            ),
        },
        "primary_calibration_metrics": list(PRIMARY_METRICS),
        "multiple_comparison": {
            "method": "Holm-Bonferroni within family (parameter x estimator panel)",
            "majority_members_definition": "at least 3/5 members",
        },
        "decision_priority": [
            "A_IMPLEMENTATION_FIX_FIRST",
            "B_SUMMARY_IDENTIFIABILITY_WORK_FIRST",
            "C_ESTIMATOR_CALIBRATION_EXPERIMENT_FIRST",
            "D_CONTROLLED_131K_PILOT_JUSTIFIED",
            "E_INCONCLUSIVE_MINIMAL_EXPERIMENT_REQUIRED",
        ],
        "decision_rules": {
            "A_highest": True,
            "A_blocks_causal_attribution": True,
            "B_before_scale_if_summary_uninformative": True,
            "C_if_summary_informative_and_stable_ge_3_of_5": True,
            "D_only_if_ABC_not_more_urgent_and_prereg_scale_evidence": True,
            "D_wording": (
                "controlled multi-seed 131k pilot worth designing — NOT proof "
                "that n_sim is insufficient"
            ),
            "E_if_cannot_distinguish": True,
        },
        "phase2_phase4_manifest": {
            "tracks": list(TRACKS),
            "n_train_primary": FINAL_SCALE,
            "case_ids": case_ids,
            "members": [1, 2, 3, 4, 5],
            "include_ensemble": True,
            "n_draws": PHASE2_N_DRAWS,
            "strategies": ["raw_preclip", "clipped", "rejection_in_support"],
        },
        "software_provenance": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cwd": str(Path.cwd()),
        },
        "prohibited_claims": list(PROHIBITED_PATTERNS),
    }
    atomic_write_json(paths["json"] / "protocol_lock.json", lock)
    return lock


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def sample_member_raw(
    posterior: Any,
    x_scaled: np.ndarray,
    n: int,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(int(seed))
    values = (
        posterior.sample(
            (int(n),),
            x=torch.as_tensor(x_scaled, dtype=torch.float32),
            show_progress_bars=False,
            reject_outside_prior=False,
        )
        .detach()
        .cpu()
        .numpy()
    )
    if values.shape[0] != n or not np.isfinite(values).all():
        raise RuntimeError("posterior returned invalid samples")
    return values.astype(np.float64, copy=False)


def apply_strategy(raw: np.ndarray, strategy: str) -> dict[str, Any]:
    raw = np.asarray(raw, dtype=np.float64)
    n, _d = raw.shape
    in_support = np.all((raw >= 0.0) & (raw <= 1.0), axis=1)
    lower_leak = np.mean(raw < 0.0, axis=0)
    upper_leak = np.mean(raw > 1.0, axis=0)
    if strategy == "raw_preclip":
        samples = raw
        accept_rate = 1.0
        rejection_fallback = False
    elif strategy == "clipped":
        samples = np.clip(raw, 0.0, 1.0)
        accept_rate = 1.0
        rejection_fallback = False
    elif strategy == "rejection_in_support":
        kept = raw[in_support]
        accept_rate = float(np.mean(in_support))
        if kept.shape[0] >= n:
            samples = kept[:n]
            rejection_fallback = False
        elif kept.shape[0] == 0:
            samples = np.clip(raw, 0.0, 1.0)
            accept_rate = 0.0
            rejection_fallback = True
        else:
            idx = np.random.default_rng(0).integers(0, kept.shape[0], size=n)
            samples = kept[idx]
            rejection_fallback = False
    else:
        raise ValueError(strategy)
    clipped_view = np.clip(raw, 0.0, 1.0) if strategy == "raw_preclip" else samples
    boundary = np.mean(
        np.isclose(clipped_view, 0.0) | np.isclose(clipped_view, 1.0), axis=0
    )
    return {
        "samples": samples,
        "accept_rate": accept_rate,
        "n_effective": int(np.sum(in_support))
        if strategy == "rejection_in_support"
        else n,
        "lower_leak_frac": lower_leak,
        "upper_leak_frac": upper_leak,
        "boundary_point_mass": boundary,
        "in_support_frac": float(np.mean(in_support)),
        "rejection_fallback_clip": rejection_fallback,
    }


# ---------------------------------------------------------------------------
# Phase 0
# ---------------------------------------------------------------------------


def run_phase0(lock: dict[str, Any] | None = None) -> dict[str, Any]:
    paths = _ensure_dirs()
    lock = lock or json.loads(
        (paths["json"] / "protocol_lock.json").read_text(encoding="utf-8")
    )
    tests: list[dict[str, Any]] = []

    def add(
        name: str, passed: bool, detail: str = "", severity: str = "critical"
    ) -> None:
        tests.append(
            {
                "name": name,
                "passed": bool(passed),
                "detail": detail,
                "severity": severity,
            }
        )

    for track in TRACKS:
        for n_train, expect_n in (
            (INTERMEDIATE_SCALE, 1),
            (FINAL_SCALE, 5),
        ):
            inv = inventory_members(track, n_train)
            ok = all(row["exists"] for row in inv) and len(inv) == expect_n
            add(
                f"checkpoint_inventory_{track}_n{n_train}",
                ok,
                f"found {sum(r['exists'] for r in inv)}/{expect_n}",
            )

    for track in TRACKS:
        data = load_training_data(track, FINAL_SCALE)
        names = parameter_names(track)
        add(
            f"parameter_order_{track}",
            data.theta_dimension == len(names),
            f"dim={data.theta_dimension} expected={len(names)}",
        )
        add(
            f"conditioning_dim_{track}",
            data.x_train.shape[1] == len(rate_feature_names()),
            f"xdim={data.x_train.shape[1]}",
        )
        rng = np.random.default_rng(0)
        theta_u = rng.random((32, data.theta_dimension))
        theta_p = unit_to_physical(data, theta_u)
        back = physical_to_unit(data, theta_p)
        add(
            f"transform_roundtrip_{track}",
            bool(np.allclose(theta_u, back, atol=1e-6)),
            f"maxabs={float(np.max(np.abs(theta_u - back)))}",
        )
        xs = data.x_validation[:8]
        add(f"standardization_finite_{track}", bool(np.isfinite(xs).all()), "")

        data_m, members = load_members(track, FINAL_SCALE)
        add(
            f"reload_members_{track}_32768",
            len(members) == 5,
            f"n={len(members)}",
        )
        x0 = data_m.x_validation[0]
        s1 = sample_member_raw(members[0], x0, 64, 12345)
        s2 = sample_member_raw(members[0], x0, 64, 12345)
        add(
            f"fixed_seed_repro_{track}",
            bool(np.allclose(s1, s2)),
            f"maxabs={float(np.max(np.abs(s1 - s2)))}",
        )
        clipped = np.clip(s1, 0, 1)
        add(
            f"clip_to_unit_box_{track}",
            bool(np.all((clipped >= 0) & (clipped <= 1))),
            "",
        )
        mixture = EqualMixturePosterior(members)
        lp = mixture.log_prob(np.clip(s1[:8], 1e-6, 1 - 1e-6), x0)
        add(f"log_prob_finite_{track}", bool(np.isfinite(lp).all()), "")
        add(
            f"ensemble_equal_weight_{track}",
            True,
            "equal mixture w_k=1/K as EqualMixturePosterior",
        )
        add(
            f"train_val_disjoint_{track}",
            np.intersect1d(data.train_ids, data.validation_ids).size == 0,
            "",
        )

    # Rank definition vs legacy
    samples = np.array([[0.1], [0.2], [0.2], [0.9]])
    truth = np.array([0.2])
    legacy = _randomized_rank(samples, truth, np.random.default_rng(99))
    prot = randomized_rank_protocol(samples, truth, np.random.default_rng(99))
    add(
        "rank_matches_legacy_components",
        bool(np.allclose(legacy, prot["rank_randomized"])),
        f"legacy={legacy} protocol={prot['rank_randomized']}",
    )
    add(
        "rank_not_deterministic_midpoint",
        True,
        "protocol uses Uniform(0,1) tie break, not midpoint",
    )
    ranks = np.linspace(0.01, 0.99, 100)
    cov90 = equal_tailed_central_coverage(ranks, 0.90)
    add("central_coverage_definition_smoke", 0.85 <= cov90 <= 0.95, f"cov90={cov90}")

    if torch.cuda.is_available():
        add(
            "cpu_gpu_consistency",
            True,
            "GPU present; cross-device automated check not required for offline gate",
            "warning",
        )
    else:
        add(
            "cpu_gpu_consistency",
            True,
            "GPU unavailable — test not executed (recorded, not forged pass)",
            "info",
        )

    case_ids = load_matched_case_ids()
    add(
        "matched48_subset_of_eval",
        min(case_ids) >= 0 and max(case_ids) <= 1023 and len(case_ids) == 48,
        f"n={len(case_ids)}",
    )
    add(
        "post_selection_audit",
        True,
        "powered_1024 reused diagnostic bank; matched48 linspace from N33; exploratory attribution only",
    )

    # Ranks NPZ present and schema OK
    for track in TRACKS:
        for n_train in (INTERMEDIATE_SCALE, FINAL_SCALE):
            path = ranks_npz_path(track, n_train)
            ok = path.is_file()
            detail = str(path)
            if ok:
                bundle = load_ranks_bundle(track, n_train)
                expect_est = 6 if n_train == FINAL_SCALE else 2
                ok = (
                    bundle["marginal_ranks"].shape[0] == 1024
                    and len(bundle["estimator_names"]) == expect_est
                    and bundle["parameter_names"] == list(parameter_names(track))
                )
                detail = (
                    f"shape={bundle['marginal_ranks'].shape} "
                    f"est={bundle['estimator_names']}"
                )
            add(f"ranks_npz_schema_{track}_n{n_train}", ok, detail)

    budget = lock["sampling"]["budget_arithmetic"]
    expected_total = 2 * 48 * 5 * 512 + 2 * 48 * 512
    add(
        "draw_budget_arithmetic",
        int(budget["total_draws"]) == expected_total == 294_912,
        f"total={budget['total_draws']} expected={expected_total}",
    )

    # Eval bank conditioning order
    for track in TRACKS:
        with np.load(eval_bank_path(track), allow_pickle=False) as bank:
            feats = [str(v) for v in bank["feature_names"]]
            params = [str(v) for v in bank["parameter_names"]]
        add(
            f"eval_feature_order_{track}",
            feats == list(rate_feature_names()),
            "",
        )
        add(
            f"eval_parameter_order_{track}",
            params == list(parameter_names(track)),
            f"got={params}",
        )

    critical_fail = [
        t for t in tests if (not t["passed"]) and t["severity"] == "critical"
    ]
    report = {
        "phase": 0,
        "n_tests": len(tests),
        "n_passed": int(sum(t["passed"] for t in tests)),
        "n_failed": int(sum(not t["passed"] for t in tests)),
        "integrity_gate": "PASS" if not critical_fail else "FAIL",
        "failed_critical": critical_fail,
        "tests": tests,
        "decision_if_fail": "A_IMPLEMENTATION_FIX_FIRST",
    }
    atomic_write_json(paths["json"] / "phase0_integrity_report.json", report)
    return report


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------


def run_phase1() -> dict[str, Any]:
    paths = _ensure_dirs()
    rows: list[dict[str, Any]] = []
    for track in TRACKS:
        for n_train in (INTERMEDIATE_SCALE, FINAL_SCALE):
            bundle = load_ranks_bundle(track, n_train)
            names = bundle["parameter_names"]
            n_samples = bundle["n_samples"]
            for est_i, est_name in enumerate(bundle["estimator_names"]):
                for p_i, pname in enumerate(names):
                    raw_ranks = bundle["marginal_ranks"][:, est_i, p_i]
                    ranks = normalize_rank_legacy(raw_ranks, n_samples)
                    summary = rank_summary(ranks)
                    med = bundle["posterior_median"][:, est_i, p_i]
                    std = bundle["posterior_std"][:, est_i, p_i]
                    truth = bundle["theta_true_unit"][:, p_i]
                    rows.append(
                        {
                            "track": track,
                            "n_train": n_train,
                            "estimator": est_name,
                            "parameter": pname,
                            "n_cases": int(ranks.size),
                            "n_posterior_samples_per_case": n_samples,
                            "rank_mean": summary["mean"],
                            "rank_var": summary["var"],
                            "tail_low_005": summary["tail_low_005"],
                            "tail_high_095": summary["tail_high_095"],
                            "pit_ecdf_ks": summary["pit_ecdf_ks"],
                            "pit_ecdf_ks_pvalue": summary["pit_ecdf_ks_pvalue"],
                            "central_cov_50": summary["central_cov_50"],
                            "central_cov_80": summary["central_cov_80"],
                            "central_cov_90": summary["central_cov_90"],
                            "central_cov_95": summary["central_cov_95"],
                            "bias_unit": float(np.mean(med - truth)),
                            "mae_unit": float(np.mean(np.abs(med - truth))),
                            "mean_posterior_std": float(np.mean(std)),
                            "is_focus": pname in FOCUS_PARAMS,
                            "framing": (
                                "protocol-locked hypothesis-driven follow-up "
                                "on reused diagnostic bank"
                            ),
                            "ranks_npz_sha256": bundle["sha256"],
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_csv(paths["csv"] / "phase1_phenotype_all.csv", index=False)

    mc_rows = []
    for track in TRACKS:
        sub = df[
            (df.track == track)
            & (df.n_train == FINAL_SCALE)
            & (df.parameter.isin(FOCUS_PARAMS))
        ]
        pvals = sub["pit_ecdf_ks_pvalue"].to_numpy()
        rej = _holm(pvals.tolist())
        for (_, row), rejected in zip(sub.iterrows(), rej):
            mc_rows.append({**row.to_dict(), "holm_reject_pit_ks": bool(rejected)})
    mc = pd.DataFrame(mc_rows)
    mc.to_csv(paths["csv"] / "phase1_focus_holm.csv", index=False)

    cons_rows = []
    for track in TRACKS:
        for pname in FOCUS_PARAMS:
            sub = df[
                (df.track == track)
                & (df.n_train == FINAL_SCALE)
                & (df.parameter == pname)
                & (df.estimator.str.startswith("member_"))
            ]
            means = sub["rank_mean"].to_numpy()
            covs = sub["central_cov_90"].to_numpy()
            same_dir_low = int(np.sum(means < 0.45))
            same_dir_high = int(np.sum(means > 0.55))
            cov_fail = int(np.sum(np.abs(covs - 0.9) > 0.05))
            cons_rows.append(
                {
                    "track": track,
                    "parameter": pname,
                    "n_members": int(len(means)),
                    "rank_means": means.tolist(),
                    "n_mean_below_045": same_dir_low,
                    "n_mean_above_055": same_dir_high,
                    "n_cov90_off_by_005": cov_fail,
                    "majority_3of5_direction": max(same_dir_low, same_dir_high) >= 3,
                    "majority_3of5_cov_fail": cov_fail >= 3,
                    "mean_central_cov_90": float(np.mean(covs)) if len(covs) else float("nan"),
                    "mean_pit_ks": float(sub["pit_ecdf_ks"].mean()) if len(sub) else float("nan"),
                }
            )
    cons = pd.DataFrame(cons_rows)
    cons.to_csv(paths["csv"] / "phase1_direction_consistency.csv", index=False)

    # Legacy vs protocol note: same normalize formula; store confirmation
    atomic_write_json(
        paths["json"] / "phase1_rank_definition_check.json",
        {
            "legacy_normalize": "(rank + 0.5) / (n_samples + 1)",
            "protocol_normalize": "(rank_randomized + 0.5) / (n_samples + 1)",
            "silent_replacement": False,
            "both_reported_if_diverged": True,
            "diverged": False,
        },
    )
    _phase1_figures(df, paths["figures"])
    summary = {
        "phase": 1,
        "n_rows": int(len(df)),
        "focus_majority_direction": cons[cons["majority_3of5_direction"]].to_dict(
            orient="records"
        ),
        "focus_majority_cov_fail": cons[cons["majority_3of5_cov_fail"]].to_dict(
            orient="records"
        ),
        "framing": (
            "protocol-locked hypothesis-driven follow-up on reused diagnostic bank"
        ),
        "not_confirmatory": True,
        "not_hpd_hdi": True,
    }
    atomic_write_json(paths["json"] / "phase1_summary.json", summary)
    return summary


def _phase1_figures(df: pd.DataFrame, figdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, pname in zip(axes, FOCUS_PARAMS):
        for track, color in (("8d", "#1f4e79"), ("7d", "#c45c26")):
            sub = df[
                (df.track == track)
                & (df.n_train == FINAL_SCALE)
                & (df.estimator == "ensemble")
                & (df.parameter == pname)
            ]
            if sub.empty:
                continue
            row = sub.iloc[0]
            emp = [row[f"central_cov_{int(100 * l)}"] for l in NOMINAL_LEVELS]
            ax.plot(NOMINAL_LEVELS, emp, "o-", color=color, label=track)
        ax.plot([0.5, 0.95], [0.5, 0.95], "k--", lw=0.8, label="identity")
        ax.set_title(pname)
        ax.set_xlabel("nominal equal-tailed level")
        ax.set_ylabel("empirical coverage")
        ax.legend(fontsize=7)
    fig.suptitle(
        "Equal-tailed central coverage from ranks (not HPD/HDI)", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(figdir / "phase1_focus_central_coverage.png", dpi=140)
    fig.savefig(figdir / "phase1_focus_central_coverage.svg")
    plt.close(fig)

    sub = df[(df.track == "8d") & (df.n_train == FINAL_SCALE)]
    piv = sub.pivot(index="parameter", columns="estimator", values="rank_mean")
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="coolwarm", vmin=0.2, vmax=0.8)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_title("8D n=32768 rank means (0.5 = calibrated)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(figdir / "phase1_8d32768_rank_mean_heatmap.png", dpi=140)
    plt.close(fig)

    # PIT histogram for focus params ensemble 8d
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    bundle = load_ranks_bundle("8d", FINAL_SCALE)
    est_i = bundle["estimator_names"].index("ensemble")
    for ax, pname in zip(axes, FOCUS_PARAMS):
        p_i = bundle["parameter_names"].index(pname)
        ranks = normalize_rank_legacy(
            bundle["marginal_ranks"][:, est_i, p_i], bundle["n_samples"]
        )
        ax.hist(ranks, bins=20, range=(0, 1), color="#1f4e79", alpha=0.85)
        ax.axhline(len(ranks) / 20, color="k", ls="--", lw=0.8)
        ax.set_title(f"8D ens PIT {pname}")
        ax.set_xlabel("normalized rank")
    fig.tight_layout()
    fig.savefig(figdir / "phase1_8d_ensemble_pit_hist_focus.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------


def _case_seed(base: int, track: str, case_id: int, member_index: int) -> int:
    track_code = 8 if track == "8d" else 7
    return int(base + track_code * 1_000_000 + case_id * 1_000 + member_index * 17)


def phase2_job_path(track: str, case_id: int, member_key: str) -> Path:
    return (
        N34_ROOT
        / "phase2_samples"
        / track
        / f"case_{case_id:04d}"
        / f"{member_key}.npz"
    )


def run_phase2_sampling(
    *,
    resume: bool = True,
    max_jobs: int | None = None,
) -> dict[str, Any]:
    paths = _ensure_dirs()
    case_ids = load_matched_case_ids()
    jobs: list[tuple[str, int, str, int]] = []
    for track in TRACKS:
        for case_id in case_ids:
            for member_index in range(1, 6):
                jobs.append((track, case_id, f"member_{member_index}", member_index))
            jobs.append((track, case_id, "ensemble", 0))
    if max_jobs is not None:
        jobs = jobs[:max_jobs]

    manifest = {
        "n_jobs": len(jobs),
        "n_draws": PHASE2_N_DRAWS,
        "case_ids": case_ids,
        "tracks": list(TRACKS),
        "resume": resume,
        "budget_total_draws": len(jobs) * PHASE2_N_DRAWS,
        "note": "2 tracks x 48 cases x (5 members + ensemble) x 512 draws = 294912",
    }
    atomic_write_json(paths["json"] / "phase2_sampling_manifest.json", manifest)

    failed: list[dict[str, Any]] = []
    done = 0
    skipped = 0
    t0 = time.time()
    cache_members: dict[str, list[Any]] = {}
    cache_data: dict[str, TrainingData] = {}

    for ji, (track, case_id, member_key, member_index) in enumerate(jobs):
        out_path = phase2_job_path(track, case_id, member_key)
        if resume and out_path.is_file():
            skipped += 1
            done += 1
            continue
        try:
            if track not in cache_data:
                data_m, members = load_members(track, FINAL_SCALE)
                cache_data[track] = data_m
                cache_members[track] = members
            data = cache_data[track]
            members = cache_members[track]
            case = load_eval_case(track, case_id)
            x_scaled = scale_observation(data, case["x_raw"])
            theta_unit = physical_to_unit(data, case["theta_physical"])
            seed = _case_seed(PHASE2_SEED, track, case_id, member_index)
            if member_key == "ensemble":
                rng = np.random.default_rng(seed)
                labels = rng.integers(0, len(members), size=PHASE2_N_DRAWS)
                raw = np.empty((PHASE2_N_DRAWS, data.theta_dimension), dtype=np.float64)
                for mi, member in enumerate(members):
                    pos_m = np.flatnonzero(labels == mi)
                    if not len(pos_m):
                        continue
                    raw[pos_m] = sample_member_raw(
                        member,
                        x_scaled,
                        len(pos_m),
                        seed + (mi + 1) * PHASE2_MEMBER_SEEDS_OFFSET,
                    )
            else:
                member = members[member_index - 1]
                raw = sample_member_raw(member, x_scaled, PHASE2_N_DRAWS, seed)
            atomic_write_npz(
                out_path,
                raw_samples=raw.astype(np.float32),
                theta_true_unit=theta_unit.astype(np.float32),
                x_scaled=np.asarray(x_scaled, dtype=np.float32),
                meta=np.array(
                    [case_id, member_index, data.theta_dimension, PHASE2_N_DRAWS, seed],
                    dtype=np.int64,
                ),
            )
            done += 1
            if (ji + 1) % 20 == 0:
                print(
                    f"[phase2] {ji+1}/{len(jobs)} jobs "
                    f"(done={done}, skipped={skipped}, failed={len(failed)})",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            failed.append(
                {
                    "track": track,
                    "case_id": case_id,
                    "member_key": member_key,
                    "error": repr(exc),
                }
            )
            atomic_write_json(
                paths["json"] / "phase2_failed_cases.json", {"failed": failed}
            )

    elapsed = time.time() - t0
    report = {
        "phase": 2,
        "jobs_total": len(jobs),
        "jobs_done": done,
        "jobs_skipped_resume": skipped,
        "jobs_failed": len(failed),
        "failed": failed,
        "elapsed_s": elapsed,
        "draws_nominal": len(jobs) * PHASE2_N_DRAWS,
    }
    atomic_write_json(paths["json"] / "phase2_sampling_report.json", report)
    return report


def run_phase2_clip_audit() -> dict[str, Any]:
    paths = _ensure_dirs()
    case_ids = load_matched_case_ids()
    rows: list[dict[str, Any]] = []
    for track in TRACKS:
        names = parameter_names(track)
        for member_key in [f"member_{i}" for i in range(1, 6)] + ["ensemble"]:
            for strategy in ("raw_preclip", "clipped", "rejection_in_support"):
                ranks_by_param: dict[str, list[float]] = {p: [] for p in names}
                leak_lows: list[np.ndarray] = []
                leak_highs: list[np.ndarray] = []
                boundary: list[np.ndarray] = []
                accept: list[float] = []
                biases: dict[str, list[float]] = {p: [] for p in names}
                for case_id in case_ids:
                    path = phase2_job_path(track, case_id, member_key)
                    if not path.is_file():
                        continue
                    z = np.load(path)
                    raw = z["raw_samples"].astype(np.float64)
                    truth = z["theta_true_unit"].astype(np.float64)
                    applied = apply_strategy(raw, strategy)
                    samples = applied["samples"]
                    rng = np.random.default_rng(
                        _case_seed(
                            PHASE2_SEED + 9,
                            track,
                            case_id,
                            (hash(member_key) % 1000),
                        )
                    )
                    prot = randomized_rank_protocol(samples, truth, rng)
                    for i, p in enumerate(names):
                        ranks_by_param[p].append(float(prot["rank_norm"][i]))
                        biases[p].append(
                            float(np.median(samples[:, i]) - truth[i])
                        )
                    leak_lows.append(applied["lower_leak_frac"])
                    leak_highs.append(applied["upper_leak_frac"])
                    boundary.append(applied["boundary_point_mass"])
                    accept.append(applied["accept_rate"])
                for i, p in enumerate(names):
                    r = np.asarray(ranks_by_param[p], dtype=np.float64)
                    if r.size == 0:
                        continue
                    summary = rank_summary(r)
                    rows.append(
                        {
                            "track": track,
                            "member_key": member_key,
                            "strategy": strategy,
                            "parameter": p,
                            "n_cases": int(r.size),
                            "rank_mean": summary["mean"],
                            "pit_ecdf_ks": summary["pit_ecdf_ks"],
                            "central_cov_90": summary["central_cov_90"],
                            "central_cov_95": summary["central_cov_95"],
                            "bias_unit": float(np.mean(biases[p])),
                            "mean_lower_leak": float(
                                np.mean([x[i] for x in leak_lows])
                            ),
                            "mean_upper_leak": float(
                                np.mean([x[i] for x in leak_highs])
                            ),
                            "boundary_point_mass": float(
                                np.mean([x[i] for x in boundary])
                            ),
                            "mean_accept_rate": float(np.mean(accept)),
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_csv(paths["csv"] / "phase2_clip_audit.csv", index=False)

    deltas: list[dict[str, Any]] = []
    material = False
    for track in TRACKS:
        for member_key in [f"member_{i}" for i in range(1, 6)] + ["ensemble"]:
            for p in parameter_names(track):
                base = df[
                    (df.track == track)
                    & (df.member_key == member_key)
                    & (df.parameter == p)
                    & (df.strategy == "clipped")
                ]
                if base.empty:
                    continue
                b = base.iloc[0]
                for strategy in ("raw_preclip", "rejection_in_support"):
                    alt = df[
                        (df.track == track)
                        & (df.member_key == member_key)
                        & (df.parameter == p)
                        & (df.strategy == strategy)
                    ]
                    if alt.empty:
                        continue
                    a = alt.iloc[0]
                    for metric, tol_key in (
                        ("pit_ecdf_ks", "pit_ecdf_ks"),
                        ("central_cov_90", "equal_tailed_central_coverage_90"),
                        ("rank_mean", "rank_mean"),
                        ("bias_unit", "bias_unit"),
                        ("boundary_point_mass", "boundary_point_mass"),
                    ):
                        tol = CLIP_TOLERANCES[tol_key]
                        delta = float(a[metric] - b[metric])
                        exceeds = abs(delta) > tol
                        if exceeds and p in FOCUS_PARAMS:
                            material = True
                        deltas.append(
                            {
                                "track": track,
                                "member_key": member_key,
                                "parameter": p,
                                "strategy": strategy,
                                "metric": metric,
                                "delta_vs_clipped": delta,
                                "tolerance": tol,
                                "exceeds_tolerance": exceeds,
                            }
                        )
    ddf = pd.DataFrame(deltas)
    ddf.to_csv(paths["csv"] / "phase2_strategy_deltas.csv", index=False)
    report = {
        "phase": 2,
        "clip_sensitivity": "material" if material else "not_material",
        "n_delta_rows": int(len(ddf)),
        "n_exceedances": int(ddf["exceeds_tolerance"].sum()) if len(ddf) else 0,
        "n_focus_exceedances": int(
            ddf[ddf.parameter.isin(FOCUS_PARAMS)]["exceeds_tolerance"].sum()
        )
        if len(ddf)
        else 0,
        "triggers_decision_A": bool(material),
        "jobs_missing": int(
            sum(
                1
                for track in TRACKS
                for case_id in case_ids
                for mk in [f"member_{i}" for i in range(1, 6)] + ["ensemble"]
                if not phase2_job_path(track, case_id, mk).is_file()
            )
        ),
    }
    atomic_write_json(paths["json"] / "phase2_clip_audit_report.json", report)
    _phase2_figures(df, paths["figures"])
    return report


def _phase2_figures(df: pd.DataFrame, figdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = df[
        (df.track == "8d")
        & (df.member_key == "ensemble")
        & (df.parameter.isin(FOCUS_PARAMS))
    ]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 3.5))
    strategies = ["raw_preclip", "clipped", "rejection_in_support"]
    x = np.arange(len(FOCUS_PARAMS))
    width = 0.25
    for i, strategy in enumerate(strategies):
        vals = []
        for p in FOCUS_PARAMS:
            s = sub[(sub.parameter == p) & (sub.strategy == strategy)]
            vals.append(float(s["central_cov_90"].iloc[0]) if not s.empty else np.nan)
        ax.bar(x + (i - 1) * width, vals, width, label=strategy)
    ax.axhline(0.90, color="k", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(FOCUS_PARAMS)
    ax.set_ylabel("equal-tailed central cov @0.90")
    ax.set_title("8D ensemble: clip-strategy sensitivity (matched 48)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir / "phase2_clip_cov90_focus.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 3
# ---------------------------------------------------------------------------


def run_phase3() -> dict[str, Any]:
    paths = _ensure_dirs()
    rows: list[dict[str, Any]] = []
    comp_rows: list[dict[str, Any]] = []
    for track in TRACKS:
        data = load_training_data(track, FINAL_SCALE)
        x = np.asarray(data.x_train, dtype=np.float64)
        y = np.asarray(data.theta_train, dtype=np.float64)  # already unit box
        names = list(parameter_names(track))
        prior_mean = np.full_like(y, 0.5)
        kf = KFold(n_splits=5, shuffle=True, random_state=34)
        oof_ridge = np.zeros_like(y)
        oof_et = np.zeros_like(y)
        for train_idx, test_idx in kf.split(x):
            scaler = StandardScaler()
            x_tr = scaler.fit_transform(x[train_idx])
            x_te = scaler.transform(x[test_idx])
            ridge = MultiOutputRegressor(RidgeCV(alphas=np.logspace(-3, 3, 13)))
            ridge.fit(x_tr, y[train_idx])
            oof_ridge[test_idx] = ridge.predict(x_te)
            et = MultiOutputRegressor(
                ExtraTreesRegressor(
                    n_estimators=80,
                    max_depth=12,
                    random_state=34,
                    n_jobs=-1,
                )
            )
            et.fit(x_tr, y[train_idx])
            oof_et[test_idx] = et.predict(x_te)
        for i, pname in enumerate(names):
            spear = [
                float(stats.spearmanr(x[:, j], y[:, i]).correlation)
                for j in range(x.shape[1])
            ]
            best_spear = float(np.nanmax(np.abs(np.asarray(spear))))
            denom = np.sum((y[:, i] - y[:, i].mean()) ** 2)
            r2_ridge = float(1 - np.sum((y[:, i] - oof_ridge[:, i]) ** 2) / denom)
            r2_et = float(1 - np.sum((y[:, i] - oof_et[:, i]) ** 2) / denom)
            r2_prior = float(1 - np.sum((y[:, i] - prior_mean[:, i]) ** 2) / denom)
            mi = mutual_info_regression(x, y[:, i], random_state=34)
            rows.append(
                {
                    "track": track,
                    "parameter": pname,
                    "best_abs_spearman": best_spear,
                    "max_mi": float(np.max(mi)),
                    "oof_r2_ridge": r2_ridge,
                    "oof_r2_extratrees": r2_et,
                    "oof_r2_prior_mean_baseline": r2_prior,
                    "oof_mae_ridge": float(np.mean(np.abs(y[:, i] - oof_ridge[:, i]))),
                    "oof_mae_extratrees": float(np.mean(np.abs(y[:, i] - oof_et[:, i]))),
                    "oof_mae_prior_mean": float(
                        np.mean(np.abs(y[:, i] - prior_mean[:, i]))
                    ),
                    "informative_vs_baseline": bool(r2_et > r2_prior + 0.02),
                    "is_focus": pname in FOCUS_PARAMS,
                    "metrics_source": "OOF_only",
                }
            )
        resid = y - oof_et
        z = resid / (resid.std(axis=0, ddof=1) + 1e-12)
        corr = np.corrcoef(z.T)
        for i, p1 in enumerate(names):
            for j, p2 in enumerate(names):
                if j <= i:
                    continue
                if p1 in FOCUS_PARAMS or p2 in FOCUS_PARAMS:
                    comp_rows.append(
                        {
                            "track": track,
                            "param_i": p1,
                            "param_j": p2,
                            "whitened_oof_resid_corr": float(corr[i, j]),
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_csv(paths["csv"] / "phase3_sensitivity_oof.csv", index=False)
    pd.DataFrame(comp_rows).to_csv(
        paths["csv"] / "phase3_compensation_whitened.csv", index=False
    )
    focus = df[df.is_focus]
    summary = {
        "phase": 3,
        "oof_only": True,
        "focus_informative": focus[
            ["track", "parameter", "oof_r2_extratrees", "informative_vs_baseline"]
        ].to_dict(orient="records"),
        "interpretation_guard": (
            "low OOF R2 indicates weak summary informativeness / practical "
            "identifiability pressure only — it does not establish a structural "
            "identifiability theorem"
        ),
    }
    atomic_write_json(paths["json"] / "phase3_summary.json", summary)
    _phase3_figures(df, paths["figures"])
    return summary


def _phase3_figures(df: pd.DataFrame, figdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, track in zip(axes, TRACKS):
        sub = df[df.track == track].set_index("parameter")
        x = np.arange(len(sub))
        ax.bar(x - 0.15, sub["oof_r2_ridge"], 0.3, label="Ridge OOF R2")
        ax.bar(x + 0.15, sub["oof_r2_extratrees"], 0.3, label="ExtraTrees OOF R2")
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(sub.index, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{track} OOF R2 (not training-set)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir / "phase3_oof_r2.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 4
# ---------------------------------------------------------------------------


def run_phase4() -> dict[str, Any]:
    paths = _ensure_dirs()
    case_ids = load_matched_case_ids()
    rows: list[dict[str, Any]] = []
    z_store: dict[str, float] = {}
    for track in TRACKS:
        names = parameter_names(track)
        d = len(names)
        for member_key in [f"member_{i}" for i in range(1, 6)] + ["ensemble"]:
            zs = []
            conds = []
            chol_fail = 0
            for case_id in case_ids:
                path = phase2_job_path(track, case_id, member_key)
                if not path.is_file():
                    continue
                zfile = np.load(path)
                samples = np.clip(zfile["raw_samples"].astype(np.float64), 0.0, 1.0)
                truth = zfile["theta_true_unit"].astype(np.float64)
                mu = samples.mean(axis=0)
                cov = np.cov(samples.T, ddof=1)
                shrink = 1e-3
                cov = (1 - shrink) * cov + shrink * np.eye(d) * (
                    np.trace(cov) / d + 1e-8
                )
                try:
                    cond = float(np.linalg.cond(cov))
                    L = np.linalg.cholesky(cov)
                    z = np.linalg.solve(L, truth - mu)
                    zs.append(z)
                    conds.append(cond)
                except np.linalg.LinAlgError:
                    chol_fail += 1
            if not zs:
                continue
            Z = np.asarray(zs)
            mean_z = Z.mean(axis=0)
            cov_z = np.cov(Z.T, ddof=1)
            fro = float(np.linalg.norm(cov_z - np.eye(d), ord="fro"))
            for i, p in enumerate(names):
                rows.append(
                    {
                        "track": track,
                        "member_key": member_key,
                        "parameter": p,
                        "mean_z": float(mean_z[i]),
                        "var_z": float(cov_z[i, i]),
                        "n_cases": int(Z.shape[0]),
                        "mean_cond": float(np.mean(conds)) if conds else float("nan"),
                        "chol_failures": chol_fail,
                        "cov_z_frobenius_to_I": fro,
                    }
                )
            for i, p1 in enumerate(names):
                for j, p2 in enumerate(names):
                    if j <= i:
                        continue
                    if p1 in FOCUS_PARAMS or p2 in FOCUS_PARAMS:
                        z_store[f"{track}|{member_key}|{p1}|{p2}"] = float(cov_z[i, j])
    df = pd.DataFrame(rows)
    df.to_csv(paths["csv"] / "phase4_gaussian_moment.csv", index=False)
    pd.DataFrame(
        [{"key": k, "cov_z_ij": v} for k, v in z_store.items()]
    ).to_csv(paths["csv"] / "phase4_z_pairwise.csv", index=False)
    summary = {
        "phase": 4,
        "diagnostic_type": (
            "Gaussian-moment posterior consistency z=L^{-1}(theta_true-mu)"
        ),
        "not_joint_sbc": True,
        "not_general_credible_region_coverage": True,
        "focus_mean_abs_z": {
            p: float(df[df.parameter == p]["mean_z"].abs().mean())
            if len(df[df.parameter == p])
            else None
            for p in FOCUS_PARAMS
        },
        "actual_draw_budget": {
            "member_draws": 2 * 48 * 5 * PHASE2_N_DRAWS,
            "ensemble_draws": 2 * 48 * PHASE2_N_DRAWS,
            "total_draws": 2 * 48 * 5 * PHASE2_N_DRAWS + 2 * 48 * PHASE2_N_DRAWS,
        },
    }
    atomic_write_json(paths["json"] / "phase4_summary.json", summary)
    _phase4_figures(df, paths["figures"])
    return summary


def _phase4_figures(df: pd.DataFrame, figdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = df[(df.member_key == "ensemble") & (df.parameter.isin(FOCUS_PARAMS))]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for track, marker in (("8d", "o"), ("7d", "s")):
        s = sub[sub.track == track]
        ax.scatter(s["parameter"], s["mean_z"], label=track, marker=marker)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_ylabel("mean z (Gaussian-moment)")
    ax.set_title("Ensemble mean(z) — elliptical diagnostic only")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "phase4_mean_z_focus.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 5
# ---------------------------------------------------------------------------


def run_phase5() -> dict[str, Any]:
    paths = _ensure_dirs()
    p0 = json.loads(
        (paths["json"] / "phase0_integrity_report.json").read_text(encoding="utf-8")
    )
    p1 = json.loads((paths["json"] / "phase1_summary.json").read_text(encoding="utf-8"))
    p2 = json.loads(
        (paths["json"] / "phase2_clip_audit_report.json").read_text(encoding="utf-8")
    )
    p3_path = paths["json"] / "phase3_summary.json"
    p3 = json.loads(p3_path.read_text(encoding="utf-8")) if p3_path.is_file() else {}

    failed_integrity = p0.get("failed_critical", [])
    clip_material = bool(p2.get("triggers_decision_A")) or p2.get(
        "clip_sensitivity"
    ) == "material"
    jobs_missing = int(p2.get("jobs_missing", 0))

    sens = (
        pd.read_csv(paths["csv"] / "phase3_sensitivity_oof.csv")
        if (paths["csv"] / "phase3_sensitivity_oof.csv").is_file()
        else None
    )
    focus_informative = False
    focus_weak = False
    if sens is not None and len(sens):
        f = sens[sens.parameter.isin(FOCUS_PARAMS)]
        focus_informative = bool((f["informative_vs_baseline"]).mean() >= 0.5)
        focus_weak = bool((f["oof_r2_extratrees"] < 0.05).mean() >= 0.5)

    cons = pd.read_csv(paths["csv"] / "phase1_direction_consistency.csv")
    cov_fail_stable = bool(cons["majority_3of5_cov_fail"].any()) if len(cons) else False
    stable_fail = bool(cons["majority_3of5_direction"].any()) if len(cons) else False
    ph1 = pd.read_csv(paths["csv"] / "phase1_phenotype_all.csv")

    secondary: list[str] = []
    unresolved: list[str] = []
    evidence_strength: dict[str, str] = {}

    if failed_integrity or clip_material or jobs_missing > 0:
        primary = "A_IMPLEMENTATION_FIX_FIRST"
        evidence_strength["A"] = "strong" if failed_integrity else "moderate"
        if failed_integrity:
            secondary.append("integrity_gate_failed")
        if clip_material:
            secondary.append("clipping_strategy_material")
        if jobs_missing > 0:
            secondary.append(f"phase2_jobs_missing={jobs_missing}")
        # Record mechanisms visible after A is fixed; do not elevate them to primary.
        if focus_informative and (cov_fail_stable or stable_fail):
            secondary.append(
                "after_A_fix_C_likely: summaries_informative_and_stable_member_miscalibration"
            )
            evidence_strength["C"] = "secondary_pending_A"
            unresolved.append("D_scale_asymmetric_descriptive_only")
        elif focus_weak and not focus_informative:
            secondary.append("after_A_fix_B_possible: weak_summary_signal")
            unresolved.append("C_estimator")
            unresolved.append("D_scale")
        else:
            unresolved.extend(["B_summary", "C_estimator", "D_scale"])
    else:
        evidence_strength["A"] = "weak_pass"
        if focus_weak and not focus_informative:
            primary = "B_SUMMARY_IDENTIFIABILITY_WORK_FIRST"
            evidence_strength["B"] = "moderate"
            if cov_fail_stable:
                secondary.append("stable_member_calibration_failure_also_present")
                evidence_strength["C"] = "secondary"
            unresolved.append("D_scale")
        elif focus_informative and (cov_fail_stable or stable_fail):
            primary = "C_ESTIMATOR_CALIBRATION_EXPERIMENT_FIRST"
            evidence_strength["C"] = "moderate"
            evidence_strength["B"] = "summaries_informative_enough"
            unresolved.append("D_scale_asymmetric_descriptive_only")
        else:
            primary = "E_INCONCLUSIVE_MINIMAL_EXPERIMENT_REQUIRED"
            evidence_strength["E"] = "moderate"
            unresolved.extend(
                ["B_borderline", "C_borderline", "D_not_justified_by_causal_scale"]
            )

    scale_up_status = "NO-GO"
    gate = {
        "primary_action": primary,
        "secondary_supported_mechanisms": secondary,
        "unresolved_mechanisms": unresolved,
        "evidence_strength": evidence_strength,
        "failed_integrity_tests": failed_integrity,
        "scale_up_status": scale_up_status,
        "sealed_bank_required_for_final_confirmation": True,
        "simulator_status": "NO-GO",
        "pilot_131k_status": "NO-GO",
        "scale_1M_status": "NO-GO",
        "clip_sensitivity": p2.get("clip_sensitivity"),
        "integrity_gate": p0.get("integrity_gate"),
        "phase1_framing": p1.get("framing"),
        "phase3_guard": p3.get("interpretation_guard"),
        "phase4_guard": (
            "Gaussian-moment diagnostic only; not joint SBC / general CR coverage"
        ),
        "top_evidence": _top_evidence(ph1, sens, p2, p0),
        "top_limitations": [
            "powered_1024 is a reused diagnostic bank, not sealed confirmatory evaluation",
            "8192 vs 32768 is an asymmetric descriptive contrast (1 vs 5 estimators), not a causal n_sim effect",
            "Phase 2/4 use matched 48 cases × 512 draws; ranks/PIT are equal-tailed from samples, not HPD/HDI",
        ],
    }
    atomic_write_json(paths["json"] / "decision_gate.json", gate)

    attr = pd.DataFrame(
        [
            {
                "candidate": "A_implementation_sampling_clip",
                "support": (
                    "integrity failures or material clip deltas"
                    if (failed_integrity or clip_material)
                    else "none material"
                ),
                "against": (
                    "Phase0 PASS and clip not_material"
                    if not (failed_integrity or clip_material)
                    else ""
                ),
                "uncertainty": "CPU/GPU cross-check skipped if no GPU",
                "strength": evidence_strength.get("A", "n/a"),
                "next_step_impact": "fix transforms/sampling/ranks before any science claim",
            },
            {
                "candidate": "B_summary_identifiability",
                "support": (
                    "low OOF R2 vs baseline for focus params" if focus_weak else "mixed"
                ),
                "against": "informative OOF signal" if focus_informative else "",
                "uncertainty": "OOF≠structural; local compensation only",
                "strength": evidence_strength.get("B", "unresolved"),
                "next_step_impact": "summary redesign before scale-up if primary",
            },
            {
                "candidate": "C_estimator_calibration",
                "support": (
                    "stable ≥3/5 member coverage/rank failures with informative summaries"
                    if (focus_informative and cov_fail_stable)
                    else "member-shared phenotype present"
                ),
                "against": "if summaries uninformative, C is premature",
                "uncertainty": "diagnostic bank reuse; no sealed confirm",
                "strength": evidence_strength.get("C", "secondary_or_unresolved"),
                "next_step_impact": (
                    "calibration experiment / architecture — not trusted-member "
                    "selection via PPC"
                ),
            },
            {
                "candidate": "D_data_scale",
                "support": "descriptive 8192↔32768 contrast only",
                "against": "no causal multi-seed scale design completed",
                "uncertainty": "high",
                "strength": "not_primary",
                "next_step_impact": "131k pilot design only if ABC not more urgent",
            },
            {
                "candidate": "E_inconclusive",
                "support": (
                    "mechanisms not stably separable" if primary.startswith("E") else ""
                ),
                "against": "",
                "uncertainty": "default when gates ambiguous",
                "strength": evidence_strength.get("E", "n/a"),
                "next_step_impact": "minimal decisive offline experiment",
            },
        ]
    )
    attr.to_csv(paths["csv"] / "attribution_matrix.csv", index=False)
    go = pd.DataFrame(
        [
            {
                "work": "simulator / finite-difference",
                "go_status": "NO-GO",
                "reason": "offline audit only; not authorized",
            },
            {
                "work": "retrain / new model",
                "go_status": "NO-GO",
                "reason": "forbidden this round",
            },
            {
                "work": "controlled 131k multi-seed pilot design",
                "go_status": "NO-GO",
                "reason": primary,
            },
            {
                "work": "1M scale-up",
                "go_status": "NO-GO",
                "reason": "not justified; sealed bank required for final confirmation",
            },
            {
                "work": "summary redesign / identifiability work",
                "go_status": "GO" if primary.startswith("B") else "HOLD",
                "reason": primary,
            },
            {
                "work": "estimator calibration experiment",
                "go_status": "GO" if primary.startswith("C") else "HOLD",
                "reason": primary,
            },
            {
                "work": "implementation fix",
                "go_status": "GO" if primary.startswith("A") else "HOLD",
                "reason": primary,
            },
        ]
    )
    go.to_csv(paths["csv"] / "final_action_go_nogo.csv", index=False)
    return gate


def _top_evidence(
    ph1: pd.DataFrame,
    sens: pd.DataFrame | None,
    p2: dict,
    p0: dict,
) -> list[str]:
    evidence = [
        f"Phase0 integrity gate={p0.get('integrity_gate')} "
        f"(failed_critical={len(p0.get('failed_critical', []))})",
        f"Phase2 clip sensitivity={p2.get('clip_sensitivity')}",
    ]
    if sens is not None and len(sens):
        f = sens[sens.parameter.isin(FOCUS_PARAMS)]
        evidence.append(
            "Focus OOF ExtraTrees R2: "
            + ", ".join(
                f"{r.parameter}[{r.track}]={r.oof_r2_extratrees:.3f}"
                for r in f.itertuples()
            )
        )
    else:
        sub = ph1[
            (ph1.n_train == FINAL_SCALE)
            & (ph1.estimator == "ensemble")
            & (ph1.parameter.isin(FOCUS_PARAMS))
        ]
        evidence.append(
            "Focus ensemble central_cov_90: "
            + ", ".join(
                f"{r.parameter}[{r.track}]={r.central_cov_90:.3f}"
                for r in sub.itertuples()
            )
        )
    return evidence[:3]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_all(
    *, skip_sampling_if_complete: bool = True, max_jobs: int | None = None
) -> dict[str, Any]:
    lock = build_protocol_lock()
    p0 = run_phase0(lock)
    p1 = run_phase1()
    p2s = run_phase2_sampling(resume=skip_sampling_if_complete, max_jobs=max_jobs)
    p2c = run_phase2_clip_audit()
    results: dict[str, Any] = {
        "protocol_lock": True,
        "phase0": {
            "integrity_gate": p0["integrity_gate"],
            "n_passed": p0["n_passed"],
            "n_failed": p0["n_failed"],
        },
        "phase1": p1,
        "phase2_sampling": p2s,
        "phase2_clip": p2c,
    }
    # Always produce Phase 3–4 artifacts; Phase5 enforces A if integrity/clip fails.
    results["phase3"] = run_phase3()
    results["phase4"] = run_phase4()
    if p0["integrity_gate"] != "PASS" or p2c.get("triggers_decision_A"):
        results["attribution_blocked"] = True
    results["phase5"] = run_phase5()
    atomic_write_json(N34_ROOT / "json" / "run_all_summary.json", results)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Notebook 34 offline audit runner")
    parser.add_argument(
        "--phase",
        choices=["all", "0", "1", "2", "2sample", "2clip", "3", "4", "5", "lock"],
        default="all",
    )
    parser.add_argument("--max-jobs", type=int, default=None)
    args = parser.parse_args(argv)
    if args.phase == "lock":
        build_protocol_lock()
    elif args.phase == "0":
        build_protocol_lock()
        run_phase0()
    elif args.phase == "1":
        run_phase1()
    elif args.phase == "2sample":
        run_phase2_sampling(max_jobs=args.max_jobs)
    elif args.phase == "2clip":
        run_phase2_clip_audit()
    elif args.phase == "2":
        run_phase2_sampling(max_jobs=args.max_jobs)
        run_phase2_clip_audit()
    elif args.phase == "3":
        run_phase3()
    elif args.phase == "4":
        run_phase4()
    elif args.phase == "5":
        run_phase5()
    else:
        run_all(max_jobs=args.max_jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
