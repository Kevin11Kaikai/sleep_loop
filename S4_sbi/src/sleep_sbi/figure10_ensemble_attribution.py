"""Notebook 33 Phase 0–1 ensemble attribution helpers (non-destructive).

Presentation notebook only reads artifacts written under
``results/figure10_8d_7d/notebook_33_ensemble_attribution/``.
Simulator calls belong to a separate Phase-2 runner, not this module's
offline / geometry paths.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .figure10_protocol import (
    FINAL_SCALE,
    INTERMEDIATE_SCALE,
    PARAMETER_NAMES_7D,
    PARAMETER_NAMES_8D,
    RESULTS_ROOT,
    SEEDS,
    verify_preregistration,
)
from .figure10_training import (
    load_members,
    load_training_data,
    member_seeds,
    sample_member,
    scale_observation,
)

N33_ROOT = RESULTS_ROOT / "notebook_33_ensemble_attribution"
TABLES = N33_ROOT / "tables"
FIGURES = N33_ROOT / "figures"
JSON_DIR = N33_ROOT / "json"
VALIDATION = N33_ROOT / "validation"
GEOMETRY = N33_ROOT / "geometry"
PPC_DIR = N33_ROOT / "ppc"
LOGS = N33_ROOT / "logs"

# Pre-frozen matched subset for global geometry (not chosen by joint KS).
MATCHED_SUBSET_N = 48
MATCHED_SUBSET_SEED = 9330001
MATCHED_POSTERIOR_SAMPLES = 512
MATCHED_MEMBER_SAMPLE_SEED_BASE = 9331001

# Notebook33 seed namespace (must not collide with figure10 SEEDS).
N33_SEEDS = {
    "matched_subset": MATCHED_SUBSET_SEED,
    "matched_sampling_base": MATCHED_MEMBER_SAMPLE_SEED_BASE,
    "decomp_mc_test": 9332001,
    "ppc_benchmark_base": 9333001,
}

PROTECTED_GLOBS = (
    "diagnostics/**/*.csv",
    "diagnostics/**/*.json",
    "diagnostics/**/*.npz",
    "diagnostics/**/*.png",
    "diagnostics/**/*.svg",
    "diagnostics/**/*.pdf",
    "figures/**/*",
    "training/**/ensemble_manifest.json",
    "training/**/training_summary.csv",
    "matched_banks/**/*.json",
    "matched_banks/paired_split_and_scaling.npz",
    "matched_banks/*/figure10_*_bank_32768.npz",
)

OUTPUT_WHITELIST_PREFIXES = (
    "notebook_33_ensemble_attribution/",
)

CHECKPOINT_SPECS: tuple[dict[str, Any], ...] = (
    {"track": "8d", "scale": INTERMEDIATE_SCALE, "member_index": 1},
    {"track": "7d", "scale": INTERMEDIATE_SCALE, "member_index": 1},
    *[{"track": "8d", "scale": FINAL_SCALE, "member_index": i} for i in range(1, 6)],
    *[{"track": "7d", "scale": FINAL_SCALE, "member_index": i} for i in range(1, 6)],
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dirs() -> None:
    for path in (N33_ROOT, TABLES, FIGURES, JSON_DIR, VALIDATION, GEOMETRY, PPC_DIR, LOGS):
        path.mkdir(parents=True, exist_ok=True)


def require_neurolib_env() -> dict[str, str]:
    env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if env != "neurolib":
        raise RuntimeError(
            f"Notebook 33 requires conda env 'neurolib'; got {env!r}"
        )
    import sys

    return {"conda_default_env": env, "sys_executable": sys.executable}


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_protected_files() -> list[Path]:
    files: set[Path] = set()
    for pattern in PROTECTED_GLOBS:
        for path in RESULTS_ROOT.glob(pattern):
            if not path.is_file():
                continue
            rel = path.relative_to(RESULTS_ROOT).as_posix()
            # Skip per-case rank shards (thousands); consolidated NPZ/CSV/JSON remain.
            if "/checkpoints/" in f"/{rel}/":
                continue
            files.add(path.resolve())
    return sorted(files)


def build_sha256_manifest(label: str) -> dict[str, Any]:
    _ensure_dirs()
    rows = []
    for path in _iter_protected_files():
        rel = path.relative_to(RESULTS_ROOT.resolve()).as_posix()
        stat = path.stat()
        rows.append(
            {
                "relative_path": rel,
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "created_utc": _utc_now(),
        "label": label,
        "results_root": RESULTS_ROOT.as_posix(),
        "n_files": len(rows),
        "files": rows,
    }
    out = VALIDATION / f"{label}_sha256_manifest.json"
    atomic_json(out, payload)
    # Compact CSV for diffs
    pd.DataFrame(rows).to_csv(VALIDATION / f"{label}_sha256_manifest.csv", index=False)
    return payload


def assert_old_results_unchanged(
    pre_label: str = "pre_run", post_label: str = "post_run"
) -> dict[str, Any]:
    pre_path = VALIDATION / f"{pre_label}_sha256_manifest.json"
    if not pre_path.exists():
        raise RuntimeError(f"missing pre-run manifest: {pre_path}")
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    post = build_sha256_manifest(post_label)
    pre_map = {row["relative_path"]: row["sha256"] for row in pre["files"]}
    post_map = {row["relative_path"]: row["sha256"] for row in post["files"]}
    missing = sorted(set(pre_map) - set(post_map))
    added_protected = sorted(set(post_map) - set(pre_map))
    changed = sorted(
        path
        for path in set(pre_map) & set(post_map)
        if pre_map[path] != post_map[path]
    )
    report = {
        "created_utc": _utc_now(),
        "ok": not missing and not changed,
        "missing_count": len(missing),
        "changed_count": len(changed),
        "added_protected_count": len(added_protected),
        "missing": missing[:50],
        "changed": changed[:50],
        "added_protected": added_protected[:50],
    }
    atomic_json(VALIDATION / "sha256_compare_report.json", report)
    if not report["ok"]:
        raise RuntimeError(
            "Protected diagnostic artifacts changed: "
            f"missing={len(missing)} changed={len(changed)}"
        )
    return report


def write_output_whitelist() -> dict[str, Any]:
    _ensure_dirs()
    payload = {
        "created_utc": _utc_now(),
        "allowed_write_prefixes_under_results": list(OUTPUT_WHITELIST_PREFIXES),
        "allowed_repo_paths": [
            "S4_sbi/notebooks/notebook_33.ipynb",
            "S4_sbi/src/sleep_sbi/figure10_ensemble_attribution.py",
        ],
        "forbidden_write_prefixes_under_results": [
            "diagnostics/",
            "training/",
            "matched_banks/",
            "figures/",
            "preregistration/",
        ],
        "note": (
            "notebook_33.ipynb is presentation-only and must not launch Phase-2 "
            "simulator jobs during nbconvert."
        ),
    }
    atomic_json(VALIDATION / "output_whitelist.json", payload)
    return payload


def canonicalize_theta_x(
    theta: np.ndarray, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize arrays for stable case hashing (theta+x only)."""

    th = np.ascontiguousarray(np.round(np.asarray(theta, dtype=np.float64), 12))
    xx = np.ascontiguousarray(np.round(np.asarray(x, dtype=np.float64), 12))
    return th, xx


def case_canonical_hash(theta: np.ndarray, x: np.ndarray) -> str:
    th, xx = canonicalize_theta_x(theta, x)
    digest = hashlib.sha256()
    digest.update(th.tobytes())
    digest.update(b"|")
    digest.update(xx.tobytes())
    return digest.hexdigest()


def audit_evaluation_bank_independence() -> dict[str, Any]:
    """Audit official_300 vs powered_1024; both are descriptive/sensitivity only."""

    _ensure_dirs()
    rows = []
    track_reports = {}
    for track in ("8d", "7d"):
        official_path = (
            RESULTS_ROOT
            / "diagnostics"
            / "global"
            / "official_300"
            / track
            / f"official_300_{track}.npz"
        )
        powered_path = (
            RESULTS_ROOT
            / "diagnostics"
            / "global"
            / "powered_1024"
            / track
            / f"powered_1024_{track}.npz"
        )
        with np.load(official_path, allow_pickle=False) as official:
            o_theta = np.asarray(official["theta"], float)
            o_x = np.asarray(official["x"], float)
            o_seeds = np.asarray(official["simulator_seed"], np.int64)
            o_sample_id = np.asarray(official["sample_id"], np.int64)
            o_paired = np.asarray(official["paired_row_id"], np.int64)
        with np.load(powered_path, allow_pickle=False) as powered:
            p_theta = np.asarray(powered["theta"], float)
            p_x = np.asarray(powered["x"], float)
            p_seeds = np.asarray(powered["simulator_seed"], np.int64)
            p_sample_id = np.asarray(powered["sample_id"], np.int64)
            p_paired = np.asarray(powered["paired_row_id"], np.int64)

        o_hash = {case_canonical_hash(o_theta[i], o_x[i]) for i in range(len(o_theta))}
        p_hash = {case_canonical_hash(p_theta[i], p_x[i]) for i in range(len(p_theta))}
        exact = o_hash & p_hash
        o_theta_only = {
            hashlib.sha256(canonicalize_theta_x(o_theta[i], o_x[i])[0].tobytes()).hexdigest()
            for i in range(len(o_theta))
        }
        p_theta_only = {
            hashlib.sha256(canonicalize_theta_x(p_theta[i], p_x[i])[0].tobytes()).hexdigest()
            for i in range(len(p_theta))
        }
        o_x_only = {
            hashlib.sha256(canonicalize_theta_x(o_theta[i], o_x[i])[1].tobytes()).hexdigest()
            for i in range(len(o_x))
        }
        p_x_only = {
            hashlib.sha256(canonicalize_theta_x(p_theta[i], p_x[i])[1].tobytes()).hexdigest()
            for i in range(len(p_x))
        }
        seed_overlap = int(len(set(o_seeds.tolist()) & set(p_seeds.tolist())))
        # Local index namespaces intentionally collide; audit as metadata only.
        sample_id_overlap = int(len(set(o_sample_id.tolist()) & set(p_sample_id.tolist())))
        paired_overlap = int(len(set(o_paired.tolist()) & set(p_paired.tolist())))
        report = {
            "track": track,
            "official_n": int(len(o_theta)),
            "powered_n": int(len(p_theta)),
            "official_unique_canonical": int(len(o_hash)),
            "powered_unique_canonical": int(len(p_hash)),
            "exact_theta_x_overlap": int(len(exact)),
            "theta_only_overlap": int(len(o_theta_only & p_theta_only)),
            "x_only_overlap": int(len(o_x_only & p_x_only)),
            "simulator_seed_overlap": seed_overlap,
            "sample_id_overlap_metadata_only": sample_id_overlap,
            "paired_row_id_overlap_metadata_only": paired_overlap,
            "overlap_proportion_vs_official": float(len(exact) / max(len(o_hash), 1)),
            "canonical_hash_fields": ["normalized_theta", "normalized_x"],
            "metadata_fields_excluded_from_canonical_hash": [
                "dataset_id",
                "sample_id",
                "paired_row_id",
            ],
            "allowed_use_in_notebook_33": "descriptive_sensitivity_only",
            "untouched_final_evaluation": False,
            "selection_confirmation_split_allowed": False,
            "future_selection_requires_sealed_bank": True,
            "independent_cases_by_theta_x": bool(len(exact) == 0),
        }
        track_reports[track] = report
        rows.append(report)

    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES / "evaluation_bank_overlap_audit.csv", index=False)
    payload = {
        "created_utc": _utc_now(),
        "preregistration_hash": verify_preregistration(),
        "policy": {
            "official_300_role": "descriptive_sensitivity",
            "powered_1024_role": "descriptive_sensitivity",
            "untouched_final_evaluation": False,
            "future_member_selection_or_learned_weights_requires": (
                "new sealed evaluation bank generated after freezing selection rules"
            ),
        },
        "tracks": track_reports,
        "all_tracks_theta_x_disjoint": all(
            track_reports[t]["exact_theta_x_overlap"] == 0 for t in track_reports
        ),
    }
    atomic_json(JSON_DIR / "evaluation_bank_independence.json", payload)
    return payload


def audit_standardizer_fit_source() -> dict[str, Any]:
    """Confirm x standardizer is fit on training-bank train_ids only."""

    _ensure_dirs()
    rows = []
    for track in ("8d", "7d"):
        for scale in (INTERMEDIATE_SCALE, FINAL_SCALE):
            data = load_training_data(track, scale)
            # Recompute expected location/scale from train_ids of the bank.
            bank_path = (
                RESULTS_ROOT
                / "matched_banks"
                / track
                / f"figure10_{track}_bank_{FINAL_SCALE}.npz"
            )
            with np.load(bank_path, allow_pickle=False) as bank:
                x = np.asarray(bank["x"], float)[:scale]
            loc = np.median(x[data.train_ids], axis=0)
            q25, q75 = np.percentile(x[data.train_ids], [25, 75], axis=0)
            scale_vec = q75 - q25
            fallback = np.std(x[data.train_ids], axis=0)
            scale_vec = np.where(scale_vec > 1e-12, scale_vec, fallback)
            scale_vec = np.where(scale_vec > 1e-12, scale_vec, 1.0)
            loc_ok = bool(np.allclose(loc, data.x_location, rtol=0, atol=1e-12))
            scale_ok = bool(np.allclose(scale_vec, data.x_scale, rtol=0, atol=1e-12))
            # Ensure validation ids were not mixed into fit by checking that
            # fitting on all ids would differ or train_ids exclude validation.
            overlap = int(np.intersect1d(data.train_ids, data.validation_ids).size)
            rows.append(
                {
                    "track": track,
                    "scale": scale,
                    "fit_source": "matched_training_bank_train_ids_only",
                    "train_n": int(len(data.train_ids)),
                    "validation_n": int(len(data.validation_ids)),
                    "train_validation_overlap": overlap,
                    "location_matches_train_fit": loc_ok,
                    "scale_matches_train_fit": scale_ok,
                    "evaluation_banks_used_in_fit": False,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES / "standardizer_fit_source_audit.csv", index=False)
    payload = {
        "created_utc": _utc_now(),
        "ok": bool(frame.train_validation_overlap.eq(0).all()
                   and frame.location_matches_train_fit.all()
                   and frame.scale_matches_train_fit.all()),
        "rows": rows,
        "conclusion": (
            "Observation standardizer (median/IQR) is fit on training-bank "
            "train_ids only; evaluation banks are not used in the fit."
        ),
    }
    atomic_json(JSON_DIR / "standardizer_fit_source_audit.json", payload)
    return payload


def mixture_variance_decomposition(
    means: np.ndarray, variances: np.ndarray, weights: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Weighted mixture variance decomposition (not sample unbiased 1/(K-1))."""

    means = np.asarray(means, float)
    variances = np.asarray(variances, float)
    if means.ndim == 1:
        means = means[:, None]
        variances = variances[:, None]
    k, d = means.shape
    if weights is None:
        weights = np.full(k, 1.0 / k)
    else:
        weights = np.asarray(weights, float)
        weights = weights / weights.sum()
    mu_mix = np.sum(weights[:, None] * means, axis=0)
    v_within = np.sum(weights[:, None] * variances, axis=0)
    v_between = np.sum(weights[:, None] * (means - mu_mix) ** 2, axis=0)
    v_mix = v_within + v_between
    with np.errstate(divide="ignore", invalid="ignore"):
        f_between = np.where(v_mix > 0, v_between / v_mix, np.nan)
    return {
        "mu_mix": mu_mix,
        "v_within": v_within,
        "v_between": v_between,
        "v_mix": v_mix,
        "f_between": f_between,
        "weights": weights,
    }


def mixture_covariance_decomposition(
    means: np.ndarray,
    covs: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    means = np.asarray(means, float)
    covs = np.asarray(covs, float)
    k, d = means.shape
    if weights is None:
        weights = np.full(k, 1.0 / k)
    else:
        weights = np.asarray(weights, float)
        weights = weights / weights.sum()
    mu_mix = np.sum(weights[:, None] * means, axis=0)
    sigma_within = np.sum(weights[:, None, None] * covs, axis=0)
    deltas = means - mu_mix
    sigma_between = np.zeros((d, d), float)
    for index in range(k):
        outer = np.outer(deltas[index], deltas[index])
        sigma_between += weights[index] * outer
    sigma_mix = sigma_within + sigma_between
    return {
        "mu_mix": mu_mix,
        "sigma_within": sigma_within,
        "sigma_between": sigma_between,
        "sigma_mix": sigma_mix,
        "weights": weights,
    }


def run_decomposition_unit_tests() -> dict[str, Any]:
    """Analytic identity + seeded empirical MC sanity tests."""

    _ensure_dirs()
    # --- Analytic identity: known component moments, no sampling noise ---
    means = np.array([[0.0, 0.0], [1.0, -1.0], [2.0, 0.5], [-1.0, 1.5], [0.5, 2.0]])
    covs = np.array(
        [
            [[0.20, 0.05], [0.05, 0.30]],
            [[0.40, -0.10], [-0.10, 0.25]],
            [[0.15, 0.00], [0.00, 0.15]],
            [[0.50, 0.12], [0.12, 0.35]],
            [[0.25, -0.05], [-0.05, 0.45]],
        ]
    )
    weights = np.full(5, 0.2)
    analytic = mixture_covariance_decomposition(means, covs, weights)
    var_analytic = mixture_variance_decomposition(
        means, np.array([np.diag(c) for c in covs]), weights
    )
    # Closed form mixture moments
    mu = analytic["mu_mix"]
    sigma_true = analytic["sigma_mix"]
    # Identity must be exact up to floating error
    recon = analytic["sigma_within"] + analytic["sigma_between"]
    fro = float(np.linalg.norm(recon - sigma_true, ord="fro"))
    diag_err = float(
        np.max(
            np.abs(
                var_analytic["v_mix"]
                - (var_analytic["v_within"] + var_analytic["v_between"])
            )
        )
    )
    analytic_tol = 1e-12
    analytic_ok = fro <= analytic_tol and diag_err <= analytic_tol

    # Wrong 1/(K-1) formula must disagree with weighted between for K=5
    wrong_between = np.var(means, axis=0, ddof=1)
    right_between = var_analytic["v_between"]
    inflation = float(np.mean(wrong_between / np.maximum(right_between, 1e-15)))

    # --- Empirical MC sanity: tolerance from Monte Carlo SE ---
    rng = np.random.default_rng(N33_SEEDS["decomp_mc_test"])
    n_draw = 200_000
    labels = rng.choice(5, size=n_draw, p=weights)
    samples = np.empty((n_draw, 2), float)
    for k in range(5):
        idx = np.flatnonzero(labels == k)
        samples[idx] = rng.multivariate_normal(means[k], covs[k], size=len(idx))
    # Estimate component moments from labeled samples (same as production path)
    est_means = np.stack([samples[labels == k].mean(axis=0) for k in range(5)])
    est_covs = np.stack(
        [np.cov(samples[labels == k], rowvar=False, ddof=1) for k in range(5)]
    )
    est = mixture_covariance_decomposition(est_means, est_covs, weights)
    emp_cov = np.cov(samples, rowvar=False, ddof=1)
    # Monte Carlo SE for covariance entries ~ O(sigma^2 / sqrt(n))
    # Use conservative absolute tolerance: 10 * max SE proxy
    se_proxy = float(np.max(np.abs(emp_cov)) / math.sqrt(n_draw))
    mc_tol = max(20.0 * se_proxy, 1e-4)
    fro_mc = float(np.linalg.norm(est["sigma_mix"] - emp_cov, ord="fro"))
    # Also check decomposition identity on estimates
    fro_id = float(
        np.linalg.norm(
            est["sigma_within"] + est["sigma_between"] - est["sigma_mix"], ord="fro"
        )
    )
    mc_ok = fro_mc <= mc_tol and fro_id <= 1e-12

    payload = {
        "created_utc": _utc_now(),
        "analytic_identity": {
            "frobenius_recon_error": fro,
            "diag_recon_error": diag_err,
            "tolerance": analytic_tol,
            "pass": analytic_ok,
            "wrong_ddof1_over_weighted_between_mean_ratio": inflation,
            "note": "ddof=1 between overstates weighted between by ~K/(K-1)=1.25 for K=5",
        },
        "empirical_mc_sanity": {
            "n_draw": n_draw,
            "seed": N33_SEEDS["decomp_mc_test"],
            "frobenius_est_vs_emp": fro_mc,
            "frobenius_identity_on_estimates": fro_id,
            "se_proxy": se_proxy,
            "tolerance": mc_tol,
            "pass": mc_ok,
        },
        "pass": bool(analytic_ok and mc_ok),
    }
    atomic_json(VALIDATION / "variance_decomposition_unit_test.json", payload)
    atomic_json(VALIDATION / "covariance_decomposition_unit_test.json", payload)
    return payload


def audit_checkpoints() -> dict[str, Any]:
    """Reload all 12 checkpoints (8192×2 + 32768×10)."""

    _ensure_dirs()
    require_neurolib_env()
    rows = []
    for spec in CHECKPOINT_SPECS:
        track = spec["track"]
        scale = int(spec["scale"])
        member_index = int(spec["member_index"])
        seeds = member_seeds(track, final=scale == FINAL_SCALE)
        seed = int(seeds[member_index - 1])
        ckpt = (
            RESULTS_ROOT
            / "training"
            / track
            / f"scale_{scale}"
            / f"member_{seed}"
            / "best_checkpoint.pt"
        )
        row: dict[str, Any] = {
            "track": track,
            "scale": scale,
            "member_index": member_index,
            "seed": seed,
            "checkpoint": ckpt.relative_to(RESULTS_ROOT).as_posix(),
            "exists": ckpt.exists(),
            "reload_ok": False,
            "sample_ok": False,
            "theta_dimension": None,
            "error": "",
        }
        try:
            if not ckpt.exists():
                raise FileNotFoundError(str(ckpt))
            data, members = load_members(track, scale)
            if len(members) < member_index:
                raise RuntimeError("member count mismatch")
            member = members[member_index - 1]
            # Dummy observation at training median scale
            x_scaled = np.zeros(14, dtype=np.float32)
            samples = sample_member(member, x_scaled, 8, seed=seed + 17)
            row["reload_ok"] = True
            row["sample_ok"] = bool(
                samples.shape == (8, data.theta_dimension)
                and np.isfinite(samples).all()
            )
            row["theta_dimension"] = int(data.theta_dimension)
            expected_dim = 8 if track == "8d" else 7
            if row["theta_dimension"] != expected_dim:
                raise RuntimeError(
                    f"theta dim {row['theta_dimension']} != {expected_dim}"
                )
            if not row["sample_ok"]:
                raise RuntimeError("invalid posterior samples")
        except Exception as exc:  # noqa: BLE001 - collect blockers
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES / "checkpoint_audit.csv", index=False)
    payload = {
        "created_utc": _utc_now(),
        "n_expected": 12,
        "n_ok": int((frame.reload_ok & frame.sample_ok).sum()),
        "pass": bool(len(frame) == 12 and frame.reload_ok.all() and frame.sample_ok.all()),
        "rows": rows,
    }
    atomic_json(JSON_DIR / "checkpoint_reload_smoke.json", payload)
    return payload


def freeze_matched_subset_case_ids() -> dict[str, Any]:
    """Pre-freeze case IDs without reference to joint KS diagnostics."""

    _ensure_dirs()
    # Evenly spaced indices over powered_1024 — independent of rank outcomes.
    case_ids = np.unique(
        np.linspace(0, 1023, MATCHED_SUBSET_N, dtype=np.int64)
    )
    if len(case_ids) < MATCHED_SUBSET_N:
        extra = [i for i in range(1024) if i not in set(case_ids.tolist())]
        case_ids = np.sort(
            np.concatenate(
                [case_ids, np.asarray(extra[: MATCHED_SUBSET_N - len(case_ids)])]
            )
        )
    case_ids = case_ids[:MATCHED_SUBSET_N]
    payload = {
        "created_utc": _utc_now(),
        "dataset_id": "powered_1024",
        "role": "descriptive_sensitivity_geometry_subset",
        "selection_rule": (
            "evenly spaced indices via numpy.linspace(0,1023,48); "
            "NOT selected by joint KS or other diagnostic outcomes"
        ),
        "n_cases_per_track": int(MATCHED_SUBSET_N),
        "posterior_samples_per_member": int(MATCHED_POSTERIOR_SAMPLES),
        "seed_policy": N33_SEEDS,
        "case_ids": case_ids.astype(int).tolist(),
        "tracks": ["8d", "7d"],
    }
    atomic_json(JSON_DIR / "matched_geometry_subset_lock.json", payload)
    pd.DataFrame({"case_id": case_ids}).to_csv(
        TABLES / "matched_geometry_subset_case_ids.csv", index=False
    )
    return payload


def run_phase0() -> dict[str, Any]:
    """Phase 0 gate: env, SHA256, whitelist, audits, unit tests."""

    _ensure_dirs()
    env = require_neurolib_env()
    whitelist = write_output_whitelist()
    pre = build_sha256_manifest("pre_run")
    independence = audit_evaluation_bank_independence()
    standardizer = audit_standardizer_fit_source()
    unit_tests = run_decomposition_unit_tests()
    checkpoints = audit_checkpoints()
    subset_lock = freeze_matched_subset_case_ids()
    protocol_lock = {
        "created_utc": _utc_now(),
        "preregistration_hash": verify_preregistration(),
        "conda_env": env,
        "execution_addendum": {
            "n_checkpoints_audited": 12,
            "official_powered_role": "descriptive_sensitivity_only",
            "canonical_hash": "normalized_theta_plus_x_only",
            "median_std_analysis_name": "location_scale_proxy",
            "formal_decomp_requires_posterior_samples": True,
            "primary_xo_geometry": "illustrative_only",
            "matched_subset_n": MATCHED_SUBSET_N,
            "notebook_presentation_only": True,
            "phase2_requires_benchmark_first": True,
        },
        "n33_seeds": N33_SEEDS,
        "matched_subset": subset_lock,
    }
    atomic_json(JSON_DIR / "evaluation_protocol_lock.json", protocol_lock)

    blockers = []
    if not checkpoints["pass"]:
        blockers.append("checkpoint_reload_failed")
    if not unit_tests["pass"]:
        blockers.append("decomposition_unit_tests_failed")
    if not standardizer["ok"]:
        blockers.append("standardizer_fit_source_failed")
    if not independence["all_tracks_theta_x_disjoint"]:
        blockers.append("evaluation_banks_not_theta_x_disjoint")
    # Overlap=0 is good, but role is still sensitivity-only by policy.
    gate = {
        "created_utc": _utc_now(),
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
        "conda_env": env,
        "pre_run_manifest_files": pre["n_files"],
        "checkpoints_ok": checkpoints["pass"],
        "checkpoints_n_ok": checkpoints["n_ok"],
        "unit_tests_ok": unit_tests["pass"],
        "standardizer_ok": standardizer["ok"],
        "theta_x_disjoint": independence["all_tracks_theta_x_disjoint"],
        "official_powered_allowed_use": "descriptive_sensitivity_only",
        "untouched_final_evaluation": False,
        "whitelist": whitelist["allowed_write_prefixes_under_results"],
        "phase1_allowed": not blockers,
    }
    atomic_json(JSON_DIR / "phase0_gate.json", gate)
    atomic_json(LOGS / "phase0_complete.json", {"created_utc": _utc_now(), "gate": gate})
    if blockers:
        raise RuntimeError(f"Phase 0 gate failed: {blockers}")
    return gate


def _load_global_summary(track: str, scale: int = FINAL_SCALE) -> pd.DataFrame:
    path = (
        RESULTS_ROOT
        / "diagnostics"
        / "global"
        / "powered_1024"
        / track
        / f"scale_{scale}"
        / "global_summary.csv"
    )
    return pd.read_csv(path)


def _load_sbc(track: str, scale: int = FINAL_SCALE) -> pd.DataFrame:
    path = (
        RESULTS_ROOT
        / "diagnostics"
        / "global"
        / "powered_1024"
        / track
        / f"scale_{scale}"
        / "marginal_sbc_summary.csv"
    )
    return pd.read_csv(path)


def _load_curve(track: str, scale: int = FINAL_SCALE) -> pd.DataFrame:
    path = (
        RESULTS_ROOT
        / "diagnostics"
        / "global"
        / "powered_1024"
        / track
        / f"scale_{scale}"
        / "joint_expected_coverage_curve.csv"
    )
    return pd.read_csv(path)


def build_joint_and_sbc_tables() -> None:
    frames = []
    sbc_frames = []
    for track in ("8d", "7d"):
        g = _load_global_summary(track)
        g = g.copy()
        g["dataset_role"] = "descriptive_sensitivity"
        frames.append(g)
        s = _load_sbc(track)
        s = s.copy()
        s["dataset_role"] = "descriptive_sensitivity"
        sbc_frames.append(s)
        # 8192 sensitivity (single member == ensemble)
        g8 = _load_global_summary(track, INTERMEDIATE_SCALE)
        g8 = g8.copy()
        g8["dataset_role"] = "descriptive_sensitivity_confounded_scale"
        g8["confound_note"] = "scale_8192_is_single_member_not_five_member_ensemble"
        frames.append(g8)
    joint = pd.concat(frames, ignore_index=True)
    joint.to_csv(TABLES / "joint_ks_summary.csv", index=False)
    sbc = pd.concat(sbc_frames, ignore_index=True)
    sbc.to_csv(TABLES / "marginal_sbc_member_matrix.csv", index=False)

    # Rank tables for heatmaps
    for track in ("8d", "7d"):
        s = _load_sbc(track)
        for value_col, name in (
            ("ks_pvalue", "sbc_pvalue"),
            ("holm_clear_issue", "sbc_holm_issue"),
            ("rank_mean", "sbc_rank_mean"),
            ("rank_variance", "sbc_rank_variance"),
            ("normalized_posterior_sd_median", "sbc_norm_sd"),
        ):
            pivot = s.pivot(index="parameter", columns="estimator", values=value_col)
            pivot.to_csv(TABLES / f"{track}_{name}_pivot.csv")


def plot_joint_coverage_overlay() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, track in zip(axes, ("8d", "7d")):
        curve = _load_curve(track)
        for estimator, sub in curve.groupby("estimator"):
            lw = 2.5 if estimator == "ensemble" else 1.0
            alpha = 1.0 if estimator == "ensemble" else 0.75
            ax.plot(
                sub["nominal"],
                sub["empirical"],
                label=estimator,
                lw=lw,
                alpha=alpha,
            )
        grid = np.linspace(0, 1, 101)
        ax.plot(grid, grid, "k--", lw=1, label="identity")
        # DKW band for n=1024, alpha=0.05: epsilon = sqrt(log(2/alpha)/(2n))
        eps = math.sqrt(math.log(2 / 0.05) / (2 * 1024))
        ax.fill_between(grid, np.clip(grid - eps, 0, 1), np.clip(grid + eps, 0, 1),
                         color="gray", alpha=0.15, label="95% DKW band")
        ax.set_title(f"{track} powered_1024 (descriptive/sensitivity)")
        ax.set_xlabel("Nominal credibility")
        ax.set_ylabel("Empirical coverage")
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "joint_coverage_overlay.png", dpi=150)
    fig.savefig(FIGURES / "joint_coverage_overlay.svg")
    plt.close(fig)

    # KS ranking
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, track in zip(axes, ("8d", "7d")):
        g = _load_global_summary(track).sort_values("joint_rank_ks_statistic")
        colors = ["#b22222" if e == "ensemble" else "#4c72b0" for e in g["estimator"]]
        ax.barh(g["estimator"], g["joint_rank_ks_statistic"], color=colors)
        ax.set_title(f"{track} joint KS")
        ax.set_xlabel("KS statistic")
    fig.tight_layout()
    fig.savefig(FIGURES / "joint_ks_ranking.png", dpi=150)
    fig.savefig(FIGURES / "joint_ks_ranking.svg")
    plt.close(fig)


def plot_sbc_heatmaps() -> None:
    for track in ("8d", "7d"):
        s = _load_sbc(track)
        issue = s.pivot(index="parameter", columns="estimator", values="holm_clear_issue")
        rank_mean = s.pivot(index="parameter", columns="estimator", values="rank_mean")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        im0 = axes[0].imshow(issue.astype(float).values, aspect="auto", cmap="Reds", vmin=0, vmax=1)
        axes[0].set_xticks(range(len(issue.columns)))
        axes[0].set_xticklabels(issue.columns, rotation=45, ha="right")
        axes[0].set_yticks(range(len(issue.index)))
        axes[0].set_yticklabels(issue.index)
        axes[0].set_title(f"{track} Holm clear issue")
        fig.colorbar(im0, ax=axes[0], fraction=0.046)
        im1 = axes[1].imshow(rank_mean.values, aspect="auto", cmap="coolwarm", vmin=0.2, vmax=0.8)
        axes[1].set_xticks(range(len(rank_mean.columns)))
        axes[1].set_xticklabels(rank_mean.columns, rotation=45, ha="right")
        axes[1].set_yticks(range(len(rank_mean.index)))
        axes[1].set_yticklabels(rank_mean.index)
        axes[1].set_title(f"{track} SBC rank mean")
        fig.colorbar(im1, ax=axes[1], fraction=0.046)
        fig.tight_layout()
        fig.savefig(FIGURES / f"marginal_sbc_heatmaps_{track}.png", dpi=150)
        fig.savefig(FIGURES / f"marginal_sbc_heatmaps_{track}.svg")
        plt.close(fig)


def build_focused_and_contraction_tables() -> None:
    rows = []
    contraction_rows = []
    for track in ("8d", "7d"):
        sbc = _load_sbc(track)
        npz_path = (
            RESULTS_ROOT
            / "diagnostics"
            / "global"
            / "powered_1024"
            / track
            / f"scale_{FINAL_SCALE}"
            / "global_ranks_and_recovery.npz"
        )
        with np.load(npz_path, allow_pickle=False) as data:
            medians = np.asarray(data["posterior_median"], float)
            stds = np.asarray(data["posterior_std"], float)
            theta = np.asarray(data["theta_true_unit"], float)
            estimators = [str(v) for v in data["estimator_names"]]
            params = [str(v) for v in data["parameter_names"]]
            n_samples = int(data["posterior_samples_per_case"].item())
            marginal = np.asarray(data["marginal_ranks"], float)
        prior_sd = 1.0 / math.sqrt(12.0)
        for est_i, est in enumerate(estimators):
            for p_i, param in enumerate(params):
                abs_err = np.abs(medians[:, est_i, p_i] - theta[:, p_i])
                width = 2 * 1.96 * stds[:, est_i, p_i]  # rough normal interval
                ranks = (marginal[:, est_i, p_i] + 0.5) / (n_samples + 1)
                # Quantile strata of true theta
                qs = np.quantile(theta[:, p_i], [0.0, 0.25, 0.5, 0.75, 1.0])
                strata_means = []
                for lo, hi in zip(qs[:-1], qs[1:]):
                    mask = (theta[:, p_i] >= lo) & (theta[:, p_i] <= hi)
                    strata_means.append(float(np.mean(ranks[mask])) if mask.any() else nan())
                # Boundary proximity
                boundary = np.minimum(theta[:, p_i], 1.0 - theta[:, p_i])
                slope, intercept = np.polyfit(theta[:, p_i], medians[:, est_i, p_i], 1)
                rows.append(
                    {
                        "track": track,
                        "estimator": est,
                        "parameter": param,
                        "rank_mean": float(np.mean(ranks)),
                        "rank_variance": float(np.var(ranks, ddof=1)),
                        "bias_mean": float(np.mean(medians[:, est_i, p_i] - theta[:, p_i])),
                        "mean_abs_error": float(np.mean(abs_err)),
                        "median_norm_sd": float(np.median(stds[:, est_i, p_i] / prior_sd)),
                        "median_interval_width": float(np.median(width)),
                        "corr_width_abs_error": float(
                            np.corrcoef(width, abs_err)[0, 1]
                        ),
                        "calibration_slope": float(slope),
                        "calibration_intercept": float(intercept),
                        "rank_mean_q1": strata_means[0],
                        "rank_mean_q2": strata_means[1],
                        "rank_mean_q3": strata_means[2],
                        "rank_mean_q4": strata_means[3],
                        "mean_boundary_proximity": float(np.mean(boundary)),
                        "fraction_true_near_boundary_0p05": float(np.mean(boundary < 0.05)),
                    }
                )
                contraction_rows.append(
                    {
                        "track": track,
                        "estimator": est,
                        "parameter": param,
                        "median_contraction": float(np.median(stds[:, est_i, p_i] / prior_sd)),
                        "mean_abs_recovery_error": float(np.mean(abs_err)),
                        "analysis_name": "location_scale_proxy_from_stored_median_std",
                    }
                )
        # Keep focused CSV of key params
        focus_params = ["mui", "mue", "tauA"] + (["c_ctx2th"] if track == "8d" else [])
        focus = sbc[sbc.parameter.isin(focus_params)].copy()
        focus.to_csv(TABLES / f"focused_params_{track}.csv", index=False)
    pd.DataFrame(rows).to_csv(TABLES / "focused_params_mui_mue_tauA.csv", index=False)
    pd.DataFrame(contraction_rows).to_csv(
        TABLES / "contraction_vs_recovery.csv", index=False
    )


def nan() -> float:
    return float("nan")


def build_location_scale_proxy() -> None:
    """Global median/std summaries — explicitly NOT mixture variance decomposition."""

    rows = []
    for track in ("8d", "7d"):
        npz_path = (
            RESULTS_ROOT
            / "diagnostics"
            / "global"
            / "powered_1024"
            / track
            / f"scale_{FINAL_SCALE}"
            / "global_ranks_and_recovery.npz"
        )
        with np.load(npz_path, allow_pickle=False) as data:
            medians = np.asarray(data["posterior_median"], float)
            stds = np.asarray(data["posterior_std"], float)
            estimators = [str(v) for v in data["estimator_names"]]
            params = [str(v) for v in data["parameter_names"]]
        # members only indices 1..
        member_idx = [i for i, e in enumerate(estimators) if e.startswith("member_")]
        ens_i = estimators.index("ensemble")
        for p_i, param in enumerate(params):
            member_med = medians[:, member_idx, p_i]
            member_std = stds[:, member_idx, p_i]
            # Spread of member medians / mean member std — proxy only
            between_med_var = np.var(member_med, axis=1, ddof=0)  # descriptive
            mean_member_var = np.mean(member_std**2, axis=1)
            rows.append(
                {
                    "track": track,
                    "parameter": param,
                    "analysis_name": "location_scale_proxy",
                    "not_mixture_variance_decomposition": True,
                    "median_across_cases_mean_member_var": float(np.median(mean_member_var)),
                    "median_across_cases_var_of_member_medians": float(
                        np.median(between_med_var)
                    ),
                    "median_ensemble_var_proxy": float(
                        np.median(stds[:, ens_i, p_i] ** 2)
                    ),
                    "note": (
                        "Uses stored posterior_median/std only; component means "
                        "are NOT estimated. Do not interpret as v_within/v_between."
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(TABLES / "location_scale_proxy_global.csv", index=False)


def _moments_from_samples(samples_by_member: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.stack([s.mean(axis=0) for s in samples_by_member])
    covs = np.stack([np.cov(s, rowvar=False, ddof=1) for s in samples_by_member])
    vars_ = np.stack([np.var(s, axis=0, ddof=1) for s in samples_by_member])
    return means, covs, vars_


def _geometry_metrics_from_moments(
    means: np.ndarray, covs: np.ndarray, vars_: np.ndarray, param_names: Sequence[str]
) -> dict[str, Any]:
    var_de = mixture_variance_decomposition(means, vars_)
    cov_de = mixture_covariance_decomposition(means, covs)
    tr_w = float(np.trace(cov_de["sigma_within"]))
    tr_b = float(np.trace(cov_de["sigma_between"]))
    tr_m = float(np.trace(cov_de["sigma_mix"]))
    eig_w = np.sort(np.linalg.eigvalsh(cov_de["sigma_within"]))[::-1]
    eig_b = np.sort(np.linalg.eigvalsh(cov_de["sigma_between"]))[::-1]
    eig_m = np.sort(np.linalg.eigvalsh(cov_de["sigma_mix"]))[::-1]
    # Pairwise mean distances
    k = means.shape[0]
    euclid = []
    # Pooled within cov for Mahalanobis
    pooled = cov_de["sigma_within"]
    # Ridge for stability
    pooled = pooled + np.eye(pooled.shape[0]) * 1e-8
    try:
        inv = np.linalg.inv(pooled)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(pooled)
    mahal = []
    for i in range(k):
        for j in range(i + 1, k):
            d = means[i] - means[j]
            euclid.append(float(np.linalg.norm(d)))
            mahal.append(float(np.sqrt(d @ inv @ d)))
    # Correlation matrices
    def corr_from_cov(cov: np.ndarray) -> np.ndarray:
        sd = np.sqrt(np.clip(np.diag(cov), 1e-15, None))
        return cov / np.outer(sd, sd)

    corr_w = corr_from_cov(cov_de["sigma_within"])
    corr_m = corr_from_cov(cov_de["sigma_mix"])
    focus = [p for p in ("mui", "mue", "tauA") if p in param_names]
    focus_corr = {}
    for a in focus:
        for b in focus:
            if a >= b:
                continue
            ia, ib = param_names.index(a), param_names.index(b)
            focus_corr[f"corr_mix_{a}_{b}"] = float(corr_m[ia, ib])
            focus_corr[f"corr_within_{a}_{b}"] = float(corr_w[ia, ib])
            focus_corr[f"corr_diff_mix_minus_within_{a}_{b}"] = float(
                corr_m[ia, ib] - corr_w[ia, ib]
            )
    # Principal directions: leading eigenvector angle member vs mixture
    # (illustrative / matched-subset)
    mix_eigvec = np.linalg.eigh(cov_de["sigma_mix"])[1][:, -1]
    angles = []
    for cov in covs:
        vec = np.linalg.eigh(cov)[1][:, -1]
        cos = float(np.abs(np.dot(vec, mix_eigvec)))
        cos = min(1.0, max(0.0, cos))
        angles.append(float(math.degrees(math.acos(cos))))
    return {
        "f_between_marginal": {
            param_names[i]: float(var_de["f_between"][i]) for i in range(len(param_names))
        },
        "trace_within": tr_w,
        "trace_between": tr_b,
        "trace_mix": tr_m,
        "trace_between_fraction": float(tr_b / tr_m) if tr_m > 0 else float("nan"),
        "eigenvalues_within": eig_w.tolist(),
        "eigenvalues_between": eig_b.tolist(),
        "eigenvalues_mix": eig_m.tolist(),
        "pairwise_euclid_mean": float(np.mean(euclid)) if euclid else float("nan"),
        "pairwise_mahal_mean": float(np.mean(mahal)) if mahal else float("nan"),
        "pairwise_euclid_max": float(np.max(euclid)) if euclid else float("nan"),
        "pairwise_mahal_max": float(np.max(mahal)) if mahal else float("nan"),
        "leading_pc_angle_deg_mean": float(np.mean(angles)) if angles else float("nan"),
        **focus_corr,
    }


def build_primary_xo_illustrative_geometry() -> None:
    rows = []
    for track in ("8d", "7d"):
        path = (
            RESULTS_ROOT
            / "diagnostics"
            / "posterior_structure"
            / track
            / "posterior_structure_samples.npz"
        )
        with np.load(path, allow_pickle=False) as data:
            samples = np.asarray(data["samples_unit"], float)
            labels = np.asarray(data["member_labels"], np.int64)
            params = [str(v) for v in data["parameter_names"]]
        k = int(labels.max()) + 1
        by_member = [samples[labels == i] for i in range(k)]
        means, covs, vars_ = _moments_from_samples(by_member)
        metrics = _geometry_metrics_from_moments(means, covs, vars_, params)
        for param, frac in metrics["f_between_marginal"].items():
            rows.append(
                {
                    "track": track,
                    "analysis": "primary_xo_illustrative_mc_decomp",
                    "parameter": param,
                    "f_between": frac,
                    "trace_between_fraction": metrics["trace_between_fraction"],
                    "trace_within": metrics["trace_within"],
                    "trace_between": metrics["trace_between"],
                    "trace_mix": metrics["trace_mix"],
                    "pairwise_mahal_mean": metrics["pairwise_mahal_mean"],
                    "n_samples_total": int(len(samples)),
                    "member_counts": json.dumps([int((labels == i).sum()) for i in range(k)]),
                }
            )
        atomic_json(
            JSON_DIR / f"primary_xo_geometry_illustrative_{track}.json",
            {
                "created_utc": _utc_now(),
                "track": track,
                "role": "illustrative_only",
                "metrics": {
                    k: v
                    for k, v in metrics.items()
                    if k != "f_between_marginal"
                },
                "f_between_marginal": metrics["f_between_marginal"],
            },
        )
    pd.DataFrame(rows).to_csv(
        TABLES / "variance_decomposition_primary_illustrative.csv", index=False
    )


def run_matched_subset_geometry(n_samples: int = MATCHED_POSTERIOR_SAMPLES) -> dict[str, Any]:
    """Lightweight posterior sampling on frozen case IDs; no simulator."""

    _ensure_dirs()
    require_neurolib_env()
    lock = json.loads(
        (JSON_DIR / "matched_geometry_subset_lock.json").read_text(encoding="utf-8")
    )
    case_ids = np.asarray(lock["case_ids"], np.int64)
    all_case_rows = []
    summary_rows = []
    for track in ("8d", "7d"):
        data, members = load_members(track, FINAL_SCALE)
        dataset_path = (
            RESULTS_ROOT
            / "diagnostics"
            / "global"
            / "powered_1024"
            / track
            / f"powered_1024_{track}.npz"
        )
        with np.load(dataset_path, allow_pickle=False) as dataset:
            theta = np.asarray(dataset["theta"], float)
            x = np.asarray(dataset["x"], float)
        params = list(PARAMETER_NAMES_8D if track == "8d" else PARAMETER_NAMES_7D)
        track_metrics = []
        for case_pos, case_id in enumerate(case_ids):
            x_scaled = scale_observation(data, x[int(case_id)])
            by_member = []
            for m_i, member in enumerate(members):
                seed = (
                    MATCHED_MEMBER_SAMPLE_SEED_BASE
                    + (0 if track == "8d" else 10_000)
                    + case_pos * 97
                    + (m_i + 1) * 104729
                )
                samples = sample_member(member, x_scaled, n_samples, seed)
                by_member.append(samples)
            means, covs, vars_ = _moments_from_samples(by_member)
            metrics = _geometry_metrics_from_moments(means, covs, vars_, params)
            track_metrics.append(metrics)
            for param, frac in metrics["f_between_marginal"].items():
                all_case_rows.append(
                    {
                        "track": track,
                        "case_id": int(case_id),
                        "parameter": param,
                        "f_between": frac,
                        "trace_between_fraction": metrics["trace_between_fraction"],
                        "pairwise_mahal_mean": metrics["pairwise_mahal_mean"],
                        "n_posterior_samples_per_member": n_samples,
                    }
                )
            summary_rows.append(
                {
                    "track": track,
                    "case_id": int(case_id),
                    "trace_within": metrics["trace_within"],
                    "trace_between": metrics["trace_between"],
                    "trace_mix": metrics["trace_mix"],
                    "trace_between_fraction": metrics["trace_between_fraction"],
                    "pairwise_euclid_mean": metrics["pairwise_euclid_mean"],
                    "pairwise_mahal_mean": metrics["pairwise_mahal_mean"],
                    "leading_pc_angle_deg_mean": metrics["leading_pc_angle_deg_mean"],
                }
            )
            if (case_pos + 1) % 8 == 0:
                print(
                    f"matched geometry {track}: {case_pos + 1}/{len(case_ids)}",
                    flush=True,
                )
        # Cross-case distribution
        tr_frac = np.array([m["trace_between_fraction"] for m in track_metrics])
        mahal = np.array([m["pairwise_mahal_mean"] for m in track_metrics])
        dist = {
            "track": track,
            "n_cases": len(case_ids),
            "trace_between_fraction_median": float(np.median(tr_frac)),
            "trace_between_fraction_q25": float(np.quantile(tr_frac, 0.25)),
            "trace_between_fraction_q75": float(np.quantile(tr_frac, 0.75)),
            "pairwise_mahal_mean_median": float(np.median(mahal)),
            "pairwise_mahal_mean_q25": float(np.quantile(mahal, 0.25)),
            "pairwise_mahal_mean_q75": float(np.quantile(mahal, 0.75)),
            "role": "global_geometry_attribution_subset",
            "not_selected_by_joint_ks": True,
        }
        atomic_json(JSON_DIR / f"matched_geometry_distribution_{track}.json", dist)
    pd.DataFrame(all_case_rows).to_csv(
        TABLES / "covariance_decomposition_matched_subset.csv", index=False
    )
    pd.DataFrame(summary_rows).to_csv(
        TABLES / "geometry_diagnostics_matched_subset.csv", index=False
    )

    # Plot distribution
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    geo = pd.DataFrame(summary_rows)
    for ax, track in zip(axes, ("8d", "7d")):
        sub = geo[geo.track == track]
        ax.hist(sub["trace_between_fraction"], bins=12, color="#4c72b0", alpha=0.85)
        ax.axvline(sub["trace_between_fraction"].median(), color="k", ls="--", label="median")
        ax.set_title(f"{track} trace between fraction (n={len(sub)})")
        ax.set_xlabel("trace(Σ_between)/trace(Σ_mix)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "geometry_trace_between_fraction.png", dpi=150)
    fig.savefig(FIGURES / "geometry_trace_between_fraction.svg")
    plt.close(fig)
    return {"n_case_rows": len(all_case_rows), "n_summary_rows": len(summary_rows)}


def build_lc2st_and_ppc_status() -> None:
    frames = []
    for track in ("8d", "7d"):
        path = RESULTS_ROOT / "diagnostics" / "lc2st" / track / "lc2st_summary.csv"
        df = pd.read_csv(path)
        df = df.copy()
        df["p_value_display"] = df["p_value"].apply(
            lambda p: "< permutation resolution" if float(p) == 0.0 else str(p)
        )
        frames.append(df)
    pd.concat(frames, ignore_index=True).to_csv(
        TABLES / "lc2st_member_comparison.csv", index=False
    )
    # PPC pending until Phase 2
    ens_rows = []
    for track in ("8d", "7d"):
        summary = json.loads(
            (
                RESULTS_ROOT / "diagnostics" / "ppc" / track / "ppc_summary.json"
            ).read_text(encoding="utf-8")
        )
        ens_rows.append(
            {
                "track": track,
                "estimator": "ensemble",
                "status": "available_from_notebook30",
                "features_posterior_better": summary["features_posterior_better"],
                "overall_improvement_fraction": summary["overall_improvement_fraction"],
                "note": "Observed aggregate ~60% is reference only, not a hard cutoff",
            }
        )
        for m in range(1, 6):
            ens_rows.append(
                {
                    "track": track,
                    "estimator": f"member_{m}",
                    "status": "phase2_pending",
                    "features_posterior_better": None,
                    "overall_improvement_fraction": None,
                    "note": "Member-wise PPC not run; presentation must show pending",
                }
            )
    pd.DataFrame(ens_rows).to_csv(TABLES / "ppc_by_estimator.csv", index=False)
    atomic_json(
        JSON_DIR / "ppc_phase2_status.json",
        {
            "created_utc": _utc_now(),
            "phase2_complete": False,
            "member_wise_status": "pending",
            "benchmark_required_before_full_ppc": True,
            "formal_budget_per_member": 256,
            "tracks": ["8d", "7d"],
        },
    )


def plot_lc2st() -> None:
    df = pd.read_csv(TABLES / "lc2st_member_comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, track in zip(axes, ("8d", "7d")):
        sub = df[df.track == track]
        x = np.arange(len(sub))
        ax.bar(x, sub["score_observed"], color="#4c72b0", label="score_observed")
        ax.plot(x, sub["null_q95"], "r--", label="null q95")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["estimator"], rotation=45, ha="right")
        ax.set_title(f"{track} L-C2ST (p=0 => < permutation resolution)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "lc2st_member_comparison.png", dpi=150)
    fig.savefig(FIGURES / "lc2st_member_comparison.svg")
    plt.close(fig)


def build_replication_and_attribution() -> None:
    joint = pd.read_csv(TABLES / "joint_ks_summary.csv")
    joint32 = joint[(joint.scale == FINAL_SCALE) & (joint.dataset_role == "descriptive_sensitivity")]
    rows = []
    for track in ("8d", "7d"):
        sub = joint32[joint32.track == track]
        ens = float(sub.loc[sub.estimator == "ensemble", "joint_rank_ks_statistic"].iloc[0])
        members = sub[sub.estimator.str.startswith("member_")]["joint_rank_ks_statistic"]
        rows.append(
            {
                "track": track,
                "ensemble_joint_ks": ens,
                "member_joint_ks_median": float(members.median()),
                "member_joint_ks_min": float(members.min()),
                "member_joint_ks_max": float(members.max()),
                "ensemble_over_member_median_ratio": float(ens / members.median()),
            }
        )
    pd.DataFrame(rows).to_csv(TABLES / "track_replication_summary.csv", index=False)

    # Geometry summary for matrix
    geo_notes = {}
    for track in ("8d", "7d"):
        path = JSON_DIR / f"matched_geometry_distribution_{track}.json"
        if path.exists():
            geo_notes[track] = json.loads(path.read_text(encoding="utf-8"))

    sbc = pd.read_csv(TABLES / "marginal_sbc_member_matrix.csv")
    focus = sbc[sbc.parameter.isin(["mui", "mue", "tauA"])]
    all_members_issue = bool(
        focus[focus.estimator.str.startswith("member_")].holm_clear_issue.all()
    )

    matrix = [
        {
            "hypothesis": "simulation_count_insufficient",
            "support": "8192 single-member KS worse than 32k members (confounded)",
            "oppose": "32k individual members already much better joint KS than ensemble",
            "missing": "fixed K=5 architecture aggregation scale sweep",
            "verdict": "evidence_insufficient_confounded",
        },
        {
            "hypothesis": "equal_weight_ensemble_changes_joint_geometry",
            "support": (
                f"ensemble KS >> members; matched-subset geometry available: {geo_notes}"
            ),
            "oppose": "",
            "missing": "aggregation interventions on sealed bank",
            "verdict": "supported_associational",
        },
        {
            "hypothesis": "marginal_variance_inflation_causes_joint_failure",
            "support": "",
            "oppose": (
                "location/scale proxy and primary illustrative f_between often small; "
                "ensemble vs member median_normalized_posterior_sd similar in global_summary"
            ),
            "missing": "none for rejection at marginal level",
            "verdict": "not_supported_as_sole_explanation",
        },
        {
            "hypothesis": "few_bad_members_drive_ensemble_degradation",
            "support": "8d mui member_5 rank_mean polarity opposite to members 1-4",
            "oppose": "mui/mue/tauA Holm issues on all members; all L-C2ST reject",
            "missing": "sealed-bank robust aggregation study",
            "verdict": "partial_for_local_polarity_not_global",
        },
        {
            "hypothesis": "all_estimators_share_calibration_failure",
            "support": f"focused SBC holm issues all members={all_members_issue}; L-C2ST all reject",
            "oppose": "PPC ensemble predictive improvement exists (calibration != prediction)",
            "missing": "architecture/summary/boundary probes",
            "verdict": "supported_for_sbc_lc2st",
        },
        {
            "hypothesis": "correlation_copula_mismatch",
            "support": "matched-subset corr_diff metrics recorded",
            "oppose": "",
            "missing": "targeted dependence diagnostics",
            "verdict": "candidate_unverified",
        },
        {
            "hypothesis": "posterior_mode_overlap_insufficient",
            "support": "pairwise Mahalanobis distances on matched subset",
            "oppose": "",
            "missing": "explicit mode-finding",
            "verdict": "candidate_unverified",
        },
        {
            "hypothesis": "c_ctx2th_is_primary_failure_cause",
            "support": "",
            "oppose": "7d replicates ensemble degradation and SBC issues without free c_ctx2th",
            "missing": "",
            "verdict": "not_supported",
        },
        {
            "hypothesis": "mui_boundary_or_recovery_bias",
            "support": "focused table: rank stratification and boundary proximity fields",
            "oppose": "high contraction alone does not establish classical overconfidence",
            "missing": "summary sensitivity ablation",
            "verdict": "open_needs_followup",
        },
        {
            "hypothesis": "mui_summary_sensitivity_insufficient",
            "support": "",
            "oppose": "",
            "missing": "summary ablation experiments",
            "verdict": "evidence_insufficient",
        },
        {
            "hypothesis": "log_prob_or_joint_rank_implementation_issue",
            "support": "",
            "oppose": "",
            "missing": "controlled log_prob / rank protocol audit",
            "verdict": "unverified",
        },
        {
            "hypothesis": "7d_vs_8d_dimension_is_primary_cause",
            "support": "",
            "oppose": "failure pattern replicates across tracks",
            "missing": "",
            "verdict": "not_supported",
        },
    ]
    pd.DataFrame(matrix).to_csv(TABLES / "attribution_matrix.csv", index=False)

    # Simple visual table render
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    text = "\n".join(
        f"{r['hypothesis']}: {r['verdict']}" for r in matrix
    )
    ax.text(0.01, 0.99, text, va="top", family="monospace", fontsize=8)
    ax.set_title("Attribution matrix verdicts (see CSV for full evidence)")
    fig.tight_layout()
    fig.savefig(FIGURES / "attribution_matrix.png", dpi=150)
    fig.savefig(FIGURES / "attribution_matrix.svg")
    plt.close(fig)


def build_decision_and_claim() -> None:
    decision = {
        "created_utc": _utc_now(),
        "provisional": True,
        "phase2_ppc_pending": True,
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
                    "failures; selecting one member is not a validated posterior."
                ),
            },
        ],
        "branches": [
            {
                "if": "joint geometry / dependence differences remain material",
                "then": "prioritize aggregation and joint-geometry research on a sealed bank",
            },
            {
                "if": "few members remain locally anomalous (e.g., mui polarity)",
                "then": "study robust aggregation with frozen selection rules + sealed eval",
            },
            {
                "if": "mui/mue/tauA anomalies persist on all members",
                "then": "prioritize parameter-specific recovery, boundary, summary sensitivity",
            },
            {
                "if": "members improve consistently with scale under fixed K=5 protocol",
                "then": "design confounded-free 8k/32k/131k sweep",
            },
            {
                "if": "all members fail severely without scale-improvement evidence",
                "then": "do not dump compute into 1M by default",
            },
        ],
        "forbidden_promises": [
            "131k or 1M will fix calibration",
            "picking best joint-KS member yields validated posterior",
            "logsumexp density dilution proven root cause",
        ],
    }
    atomic_json(JSON_DIR / "provisional_decision_tree.json", decision)
    claim = {
        "created_utc": _utc_now(),
        "supported": [
            "synthetic cortical-rate predictive improvement (ensemble PPC reference)",
            "individual vs ensemble diagnostic comparison on descriptive/sensitivity banks",
            "association between equal-weight ensemble and joint coverage degradation",
            "parameter-specific SBC abnormalities (mui/mue/tauA)",
            "7D/8D failure-pattern replication",
            "matched-subset joint geometry description",
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
        ],
        "dataset_roles": {
            "official_300": "descriptive_sensitivity",
            "powered_1024": "descriptive_sensitivity",
        },
    }
    atomic_json(JSON_DIR / "final_claim_card.json", claim)


def plot_contraction_heatmap() -> None:
    df = pd.read_csv(TABLES / "contraction_vs_recovery.csv")
    for track in ("8d", "7d"):
        sub = df[df.track == track]
        pivot = sub.pivot(
            index="parameter", columns="estimator", values="median_contraction"
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f"{track} posterior contraction (median sd / prior sd)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(FIGURES / f"posterior_contraction_heatmap_{track}.png", dpi=150)
        fig.savefig(FIGURES / f"posterior_contraction_heatmap_{track}.svg")
        plt.close(fig)


def build_offline_attribution() -> dict[str, Any]:
    gate_path = JSON_DIR / "phase0_gate.json"
    if not gate_path.exists():
        raise RuntimeError("Phase 0 gate missing; run run_phase0() first")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "pass":
        raise RuntimeError(f"Phase 0 gate not passed: {gate}")
    require_neurolib_env()
    build_joint_and_sbc_tables()
    plot_joint_coverage_overlay()
    plot_sbc_heatmaps()
    build_focused_and_contraction_tables()
    plot_contraction_heatmap()
    build_location_scale_proxy()
    build_primary_xo_illustrative_geometry()
    geo = run_matched_subset_geometry()
    build_lc2st_and_ppc_status()
    plot_lc2st()
    build_replication_and_attribution()
    build_decision_and_claim()
    # 7d vs 8d figure
    rep = pd.read_csv(TABLES / "track_replication_summary.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(rep))
    ax.bar(x - 0.15, rep["ensemble_joint_ks"], width=0.3, label="ensemble")
    ax.bar(x + 0.15, rep["member_joint_ks_median"], width=0.3, label="member median")
    ax.set_xticks(x)
    ax.set_xticklabels(rep["track"])
    ax.set_ylabel("Joint KS")
    ax.set_title("7D vs 8D ensemble degradation pattern")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "7d_vs_8d_attribution.png", dpi=150)
    fig.savefig(FIGURES / "7d_vs_8d_attribution.svg")
    plt.close(fig)

    payload = {
        "created_utc": _utc_now(),
        "status": "phase1_complete",
        "matched_geometry": geo,
        "ppc_member_wise": "pending",
    }
    atomic_json(LOGS / "phase1_complete.json", payload)
    atomic_json(JSON_DIR / "phase1_status.json", payload)
    return payload


def run_phase2_benchmark(n_benchmark: int = 24) -> dict[str, Any]:
    """Run 16–32 simulator calls for timing only; does not replace 256/member PPC."""

    from time import perf_counter

    from .figure10_bank import run_custom_dataset
    from .figure10_protocol import fixed_c_ctx2th
    from .figure10_training import unit_to_physical
    from .route3_global_robustness import deterministic_seed_schedule

    _ensure_dirs()
    require_neurolib_env()
    if not 16 <= int(n_benchmark) <= 32:
        raise ValueError("benchmark must use 16–32 simulations")
    n_benchmark = int(n_benchmark)
    primary_path = (
        RESULTS_ROOT
        / "diagnostics"
        / "ppc"
        / "primary_observation"
        / "8d"
        / "primary_observation_8d.npz"
    )
    with np.load(primary_path, allow_pickle=False) as primary:
        observed = np.asarray(primary["x"][0], float)

    track_reports = {}
    for track in ("8d", "7d"):
        data, members = load_members(track, FINAL_SCALE)
        mixture = EqualMixturePosterior(members)
        x_scaled = scale_observation(data, observed)
        # Draw a few posterior thetas from member_1 only for benchmark sims
        theta_unit = sample_member(
            members[0],
            x_scaled,
            n_benchmark,
            N33_SEEDS["ppc_benchmark_base"] + (0 if track == "8d" else 1000),
        )
        theta_phys = unit_to_physical(data, theta_unit)
        if track == "7d":
            theta_full = np.column_stack(
                [theta_phys, np.full(n_benchmark, fixed_c_ctx2th())]
            )
        else:
            theta_full = theta_phys
        seeds = deterministic_seed_schedule(
            n_benchmark, N33_SEEDS["ppc_benchmark_base"] + (10_000 if track == "8d" else 20_000)
        )
        out_dir = PPC_DIR / "benchmark" / track
        if out_dir.exists():
            shutil.rmtree(out_dir)
        started = perf_counter()
        path = run_custom_dataset(
            theta_full,
            seeds,
            out_dir,
            track,
            f"notebook33_ppc_benchmark_{track}",
        )
        wall = float(perf_counter() - started)
        with np.load(path, allow_pickle=False) as result:
            success = np.asarray(result["success"], bool)
            runtime_s = np.asarray(result["runtime_s"], float)
        fail_rate = float((~success).mean())
        per_sim_median = float(np.nanmedian(runtime_s[success])) if success.any() else float("nan")
        # Formal Phase-2: 2 tracks × 5 members × 256 new posterior sims; prior reused.
        formal_sims = 2 * 5 * 256
        # Throughput from this track wall time
        throughput = n_benchmark / wall if wall > 0 else float("nan")
        track_reports[track] = {
            "n_benchmark": n_benchmark,
            "wall_s": wall,
            "fail_rate": fail_rate,
            "median_worker_runtime_s": per_sim_median,
            "throughput_sims_per_s": throughput,
            "output": path.relative_to(RESULTS_ROOT).as_posix(),
        }

    # Conservative estimate uses slower track throughput
    throughputs = [
        track_reports[t]["throughput_sims_per_s"]
        for t in track_reports
        if np.isfinite(track_reports[t]["throughput_sims_per_s"])
    ]
    slow = min(throughputs) if throughputs else float("nan")
    formal_sims = 2560
    est_wall_s = formal_sims / slow if slow and slow > 0 else float("nan")
    payload = {
        "created_utc": _utc_now(),
        "benchmark_only": True,
        "does_not_replace_formal_256_per_member": True,
        "n_benchmark_per_track": n_benchmark,
        "tracks": track_reports,
        "formal_phase2_new_sims": formal_sims,
        "estimated_formal_wall_s": est_wall_s,
        "estimated_formal_wall_min": float(est_wall_s / 60.0) if np.isfinite(est_wall_s) else None,
        "estimated_formal_wall_hours": float(est_wall_s / 3600.0) if np.isfinite(est_wall_s) else None,
        "notes": (
            "Estimate assumes similar throughput to benchmark with 8 workers; "
            "excludes posterior sampling overhead and prior reuse savings."
        ),
    }
    atomic_json(JSON_DIR / "phase2_benchmark_estimate.json", payload)
    atomic_json(PPC_DIR / "benchmark" / "benchmark_summary.json", payload)
    return payload


__all__ = [
    "N33_ROOT",
    "require_neurolib_env",
    "run_phase0",
    "build_offline_attribution",
    "assert_old_results_unchanged",
    "build_sha256_manifest",
    "run_decomposition_unit_tests",
    "audit_checkpoints",
    "audit_evaluation_bank_independence",
    "run_phase2_benchmark",
]
