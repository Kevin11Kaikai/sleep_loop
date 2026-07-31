"""Global, local, predictive, and structural diagnostics for Figure-10 tracks."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import pickle
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kstest, qmc, spearmanr
import torch

from .figure10_bank import run_custom_dataset
from .figure10_protocol import (
    FINAL_SCALE,
    PARAMETER_NAMES_7D,
    PARAMETER_NAMES_8D,
    RESULTS_ROOT,
    SEEDS,
    atomic_json,
    fixed_c_ctx2th,
    prior_bounds_8d,
    rate_feature_names,
    read_preregistration,
    sha256_file,
    verify_preregistration,
)
from .figure10_training import (
    EqualMixturePosterior,
    load_members,
    physical_to_unit,
    sample_member,
    scale_observation,
    unit_to_physical,
)
from .route3_global_robustness import deterministic_seed_schedule
from .route3_synthetic_preflight import parameter_contract


DIAGNOSTICS_ROOT = RESULTS_ROOT / "diagnostics"
GLOBAL_ROOT = DIAGNOSTICS_ROOT / "global"
LC2ST_ROOT = DIAGNOSTICS_ROOT / "lc2st"
PPC_ROOT = DIAGNOSTICS_ROOT / "ppc"
STRUCTURE_ROOT = DIAGNOSTICS_ROOT / "posterior_structure"


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _sobol_full8(n: int, seed: int) -> np.ndarray:
    bounds = prior_bounds_8d()
    sampler = qmc.Sobol(d=8, scramble=True, seed=int(seed))
    if n > 0 and n & (n - 1) == 0:
        unit = sampler.random_base2(int(math.log2(n)))
    else:
        unit = sampler.random(n)
    return qmc.scale(unit, bounds[:, 0], bounds[:, 1])


def _paired_theta(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    theta8 = _sobol_full8(n, seed)
    theta7_full = theta8.copy()
    theta7_full[:, 7] = fixed_c_ctx2th()
    return theta8, theta7_full


def generate_global_datasets() -> dict[str, dict[str, str]]:
    """Generate disjoint official-300 and powered-1024 paired datasets."""

    verify_preregistration()
    specifications = {
        "official_300": (
            300,
            int(SEEDS["official_coverage_sobol"]),
            int(SEEDS["official_coverage_simulator_base"]),
        ),
        "powered_1024": (
            1024,
            int(SEEDS["powered_coverage_sobol"]),
            int(SEEDS["powered_coverage_simulator_base"]),
        ),
    }
    outputs: dict[str, dict[str, str]] = {}
    all_seeds = []
    for dataset_id, (n, sobol_seed, simulator_base) in specifications.items():
        theta8, theta7 = _paired_theta(n, sobol_seed)
        seeds = deterministic_seed_schedule(n, simulator_base)
        all_seeds.append(seeds)
        outputs[dataset_id] = {}
        for track, theta in (("8d", theta8), ("7d", theta7)):
            path = run_custom_dataset(
                theta,
                seeds,
                GLOBAL_ROOT / dataset_id / track,
                track,
                dataset_id,
            )
            outputs[dataset_id][track] = path.relative_to(RESULTS_ROOT).as_posix()
    if np.intersect1d(all_seeds[0], all_seeds[1]).size:
        raise RuntimeError("official and powered diagnostic simulator seeds overlap")
    atomic_json(
        GLOBAL_ROOT / "global_dataset_manifest.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "datasets": outputs,
            "official_cases": 300,
            "powered_cases": 1024,
            "seed_overlap": 0,
            "preregistration_hash": verify_preregistration(),
        },
    )
    return outputs


def _member_log_prob(member: Any, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    # norm_posterior=False skips leakage_correction rejection sampling, which can
    # crash nflows (discriminant < 0) on heavily leaking intermediate-scale NPEs.
    # Rank comparisons under a fixed x are invariant to the x-only normalizer.
    with torch.no_grad():
        values = member.log_prob(
            torch.as_tensor(theta, dtype=torch.float32),
            x=torch.as_tensor(x, dtype=torch.float32),
            norm_posterior=False,
        )
    result = values.detach().cpu().numpy()
    if not np.isfinite(result).all():
        raise RuntimeError("member log_prob is nonfinite")
    return result


def _randomized_rank(
    samples: np.ndarray, truth: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    less = (samples < truth).sum(axis=0)
    equal = (samples == truth).sum(axis=0)
    return less + rng.random(truth.shape) * equal


def _global_case(
    track: str,
    theta_physical: np.ndarray,
    x: np.ndarray,
    data: Any,
    members: Sequence[Any],
    case_seed: int,
    n_samples: int,
) -> dict[str, Any]:
    x_scaled = scale_observation(data, x)
    theta_unit = physical_to_unit(data, theta_physical)
    mixture = EqualMixturePosterior(members)
    estimator_names = ["ensemble"] + [
        f"member_{index + 1}" for index in range(len(members))
    ]
    joint_ranks = np.empty(len(estimator_names), float)
    marginal_ranks = np.empty((len(estimator_names), data.theta_dimension), float)
    posterior_median = np.empty_like(marginal_ranks)
    posterior_std = np.empty_like(marginal_ranks)
    rng = np.random.default_rng(case_seed + 7_000_001)
    for estimator_index, name in enumerate(estimator_names):
        if name == "ensemble":
            samples, _ = mixture.sample(x_scaled, n_samples, case_seed)
            sample_log_prob = mixture.log_prob(samples, x_scaled)
            true_log_prob = float(
                mixture.log_prob(theta_unit[None, :], x_scaled)[0]
            )
        else:
            member = members[estimator_index - 1]
            samples = sample_member(
                member,
                x_scaled,
                n_samples,
                case_seed + estimator_index * 104729,
            )
            sample_log_prob = _member_log_prob(member, samples, x_scaled)
            true_log_prob = float(
                _member_log_prob(member, theta_unit[None, :], x_scaled)[0]
            )
        higher = np.sum(sample_log_prob > true_log_prob)
        ties = np.sum(sample_log_prob == true_log_prob)
        joint_ranks[estimator_index] = higher + rng.random() * ties
        marginal_ranks[estimator_index] = _randomized_rank(
            samples, theta_unit, rng
        )
        posterior_median[estimator_index] = np.median(samples, axis=0)
        posterior_std[estimator_index] = np.std(samples, axis=0, ddof=1)
    return {
        "joint_ranks": joint_ranks,
        "marginal_ranks": marginal_ranks,
        "posterior_median": posterior_median,
        "posterior_std": posterior_std,
        "theta_true_unit": theta_unit,
        "x_scaled": x_scaled,
        "estimator_names": estimator_names,
    }


def run_global_diagnostic(
    track: str,
    dataset_id: str,
    scale: int = FINAL_SCALE,
) -> Path:
    """Compute joint expected-coverage and marginal SBC ranks."""

    verify_preregistration()
    if dataset_id not in {"official_300", "powered_1024"}:
        raise ValueError("unknown global diagnostic dataset")
    dataset_path = (
        GLOBAL_ROOT / dataset_id / track / f"{dataset_id}_{track}.npz"
    )
    with np.load(dataset_path, allow_pickle=False) as dataset:
        theta = np.asarray(dataset["theta"], float)
        x = np.asarray(dataset["x"], float)
        success = np.asarray(dataset["success"], bool)
    if not success.all():
        raise RuntimeError(f"{dataset_id}/{track} includes failed simulations")
    data, members = load_members(track, scale)
    n_cases = len(theta)
    n_samples = int(
        read_preregistration()["diagnostics"][
            "posterior_samples_per_global_case"
        ]
    )
    seed_key = (
        f"official_posterior_sampling_{track}"
        if dataset_id == "official_300"
        else f"powered_posterior_sampling_{track}"
    )
    base_seed = int(SEEDS[seed_key])
    output = GLOBAL_ROOT / dataset_id / track / f"scale_{scale}"
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    estimator_count = len(members) + 1
    dimension = data.theta_dimension
    joint = np.empty((n_cases, estimator_count), np.float32)
    marginal = np.empty((n_cases, estimator_count, dimension), np.float32)
    medians = np.empty_like(marginal)
    stds = np.empty_like(marginal)
    theta_unit = np.empty((n_cases, dimension), np.float32)
    runtimes = np.empty(n_cases, float)
    estimator_names = None
    for case in range(n_cases):
        path = checkpoint_dir / f"{case:05d}.npz"
        loaded = False
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as check:
                    loaded = (
                        str(check["preregistration_hash"].item())
                        == verify_preregistration()
                        and str(check["track"].item()) == track
                        and int(check["scale"].item()) == scale
                        and int(check["case"].item()) == case
                    )
                    if loaded:
                        joint[case] = check["joint_ranks"]
                        marginal[case] = check["marginal_ranks"]
                        medians[case] = check["posterior_median"]
                        stds[case] = check["posterior_std"]
                        theta_unit[case] = check["theta_true_unit"]
                        runtimes[case] = float(check["runtime_s"].item())
                        estimator_names = [
                            str(value) for value in check["estimator_names"]
                        ]
            except Exception:
                loaded = False
        if loaded:
            continue
        started = perf_counter()
        result = _global_case(
            track,
            theta[case],
            x[case],
            data,
            members,
            base_seed + case * 10_007,
            n_samples,
        )
        joint[case] = result["joint_ranks"]
        marginal[case] = result["marginal_ranks"]
        medians[case] = result["posterior_median"]
        stds[case] = result["posterior_std"]
        theta_unit[case] = result["theta_true_unit"]
        runtimes[case] = perf_counter() - started
        estimator_names = result["estimator_names"]
        _atomic_npz(
            path,
            case=np.asarray(case, np.int64),
            joint_ranks=joint[case],
            marginal_ranks=marginal[case],
            posterior_median=medians[case],
            posterior_std=stds[case],
            theta_true_unit=theta_unit[case],
            runtime_s=np.asarray(runtimes[case]),
            estimator_names=np.asarray(estimator_names, dtype="<U32"),
            track=np.asarray(track, dtype="<U8"),
            scale=np.asarray(scale, np.int64),
            preregistration_hash=np.asarray(
                verify_preregistration(), dtype="<U64"
            ),
        )
        if (case + 1) % 50 == 0:
            print(
                f"global {dataset_id}/{track}/scale{scale}: "
                f"{case + 1}/{n_cases}",
                flush=True,
            )
    consolidated = output / "global_ranks_and_recovery.npz"
    _atomic_npz(
        consolidated,
        joint_ranks=joint,
        marginal_ranks=marginal,
        posterior_median=medians,
        posterior_std=stds,
        theta_true_unit=theta_unit,
        runtime_s=runtimes,
        estimator_names=np.asarray(estimator_names, dtype="<U32"),
        parameter_names=np.asarray(
            PARAMETER_NAMES_8D if track == "8d" else PARAMETER_NAMES_7D,
            dtype="<U32",
        ),
        posterior_samples_per_case=np.asarray(n_samples, np.int64),
        dataset_id=np.asarray(dataset_id, dtype="<U32"),
        track=np.asarray(track, dtype="<U8"),
        scale=np.asarray(scale, np.int64),
        preregistration_hash=np.asarray(verify_preregistration(), dtype="<U64"),
    )
    return consolidated


def _holm_rejections(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    p_values = np.asarray(p_values, float)
    order = np.argsort(p_values)
    rejected = np.zeros(len(p_values), bool)
    for rank, index in enumerate(order):
        threshold = alpha / (len(p_values) - rank)
        if p_values[index] <= threshold:
            rejected[index] = True
        else:
            break
    return rejected


def analyze_global_ranks(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        joint = np.asarray(data["joint_ranks"], float)
        marginal = np.asarray(data["marginal_ranks"], float)
        medians = np.asarray(data["posterior_median"], float)
        stds = np.asarray(data["posterior_std"], float)
        theta = np.asarray(data["theta_true_unit"], float)
        estimator_names = [str(value) for value in data["estimator_names"]]
        parameter_names = [str(value) for value in data["parameter_names"]]
        n_samples = int(data["posterior_samples_per_case"].item())
        dataset_id = str(data["dataset_id"].item())
        track = str(data["track"].item())
        scale = int(data["scale"].item())
    rows = []
    sbc_rows = []
    levels = np.linspace(0, 1, 101)
    curve_rows = []
    for estimator_index, estimator in enumerate(estimator_names):
        normalized_joint = (joint[:, estimator_index] + 0.5) / (n_samples + 1)
        joint_test = kstest(normalized_joint, "uniform")
        recovery = np.mean(np.abs(medians[:, estimator_index] - theta))
        prior_recovery = np.mean(np.abs(0.5 - theta))
        contraction = np.median(
            stds[:, estimator_index] / (1 / np.sqrt(12)), axis=0
        )
        rows.append(
            {
                "dataset_id": dataset_id,
                "track": track,
                "scale": scale,
                "estimator": estimator,
                "cases": len(theta),
                "joint_rank_ks_statistic": float(joint_test.statistic),
                "joint_rank_ks_pvalue": float(joint_test.pvalue),
                "mean_posterior_median_abs_error": float(recovery),
                "prior_median_abs_error": float(prior_recovery),
                "recovery_improvement_fraction": float(
                    1 - recovery / prior_recovery
                ),
                "median_normalized_posterior_sd": float(
                    np.median(contraction)
                ),
            }
        )
        for level in levels:
            curve_rows.append(
                {
                    "dataset_id": dataset_id,
                    "track": track,
                    "scale": scale,
                    "estimator": estimator,
                    "nominal": float(level),
                    "empirical": float(np.mean(normalized_joint <= level)),
                }
            )
        p_values = []
        parameter_cache = []
        for parameter_index, parameter in enumerate(parameter_names):
            normalized = (
                marginal[:, estimator_index, parameter_index] + 0.5
            ) / (n_samples + 1)
            test = kstest(normalized, "uniform")
            p_values.append(float(test.pvalue))
            parameter_cache.append(
                {
                    "dataset_id": dataset_id,
                    "track": track,
                    "scale": scale,
                    "estimator": estimator,
                    "parameter": parameter,
                    "ks_statistic": float(test.statistic),
                    "ks_pvalue": float(test.pvalue),
                    "rank_mean": float(np.mean(normalized)),
                    "rank_variance": float(np.var(normalized, ddof=1)),
                    "bias_mean": float(
                        np.mean(
                            medians[:, estimator_index, parameter_index]
                            - theta[:, parameter_index]
                        )
                    ),
                    "normalized_posterior_sd_median": float(
                        np.median(
                            stds[:, estimator_index, parameter_index]
                            / (1 / np.sqrt(12))
                        )
                    ),
                }
            )
        rejections = _holm_rejections(np.asarray(p_values), alpha=0.05)
        for row, rejected in zip(parameter_cache, rejections):
            row["holm_clear_issue"] = bool(rejected)
            sbc_rows.append(row)
    output = path.parent
    summary_frame = pd.DataFrame(rows)
    sbc_frame = pd.DataFrame(sbc_rows)
    curve_frame = pd.DataFrame(curve_rows)
    summary_frame.to_csv(output / "global_summary.csv", index=False)
    sbc_frame.to_csv(output / "marginal_sbc_summary.csv", index=False)
    curve_frame.to_csv(output / "joint_expected_coverage_curve.csv", index=False)
    ensemble = summary_frame[summary_frame.estimator == "ensemble"].iloc[0]
    ensemble_sbc = sbc_frame[sbc_frame.estimator == "ensemble"]
    summary = {
        "dataset_id": dataset_id,
        "track": track,
        "scale": scale,
        "cases": len(theta),
        "posterior_samples_per_case": n_samples,
        "ensemble_joint_ks_statistic": float(
            ensemble.joint_rank_ks_statistic
        ),
        "ensemble_joint_ks_pvalue": float(ensemble.joint_rank_ks_pvalue),
        "ensemble_clear_sbc_issue_count": int(
            ensemble_sbc.holm_clear_issue.sum()
        ),
        "ensemble_recovery_improvement_fraction": float(
            ensemble.recovery_improvement_fraction
        ),
        "ensemble_median_normalized_posterior_sd": float(
            ensemble.median_normalized_posterior_sd
        ),
        "preregistration_hash": verify_preregistration(),
    }
    atomic_json(output / "global_analysis_summary.json", summary)
    return summary


def generate_primary_observation() -> Path:
    verify_preregistration()
    centers = {
        entry["identifier"]: entry
        for entry in parameter_contract()["centers"]
    }
    theta = np.asarray(centers["v8a_local_best"]["theta"], float)
    if not np.isclose(theta[7], fixed_c_ctx2th()):
        raise RuntimeError("primary V8a observation does not match frozen 7D value")
    seed = int(SEEDS["primary_observation_simulator"])
    path = run_custom_dataset(
        theta[None, :],
        np.asarray([seed], np.int64),
        PPC_ROOT / "primary_observation" / "8d",
        "8d",
        "primary_observation",
    )
    with np.load(path, allow_pickle=False) as data:
        if not bool(data["success"].item()):
            raise RuntimeError("frozen primary observation simulation failed")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "identifier": "canonical_v8a_local_best_primary",
        "theta_full_8d": theta.tolist(),
        "simulator_seed": seed,
        "same_observation_for_8d_and_7d": True,
        "c_ctx2th_equals_frozen_7d_value": True,
        "path": path.relative_to(RESULTS_ROOT).as_posix(),
        "sha256": sha256_file(path),
        "preregistration_hash": verify_preregistration(),
    }
    atomic_json(PPC_ROOT / "primary_observation_manifest.json", manifest)
    return path


def _mixture_sample_batched(
    members: Sequence[Any],
    x_scaled: np.ndarray,
    seed: int,
) -> np.ndarray:
    n = len(x_scaled)
    rng = np.random.default_rng(int(seed))
    labels = rng.integers(0, len(members), size=n)
    samples = np.empty((n, len(PARAMETER_NAMES_8D)), np.float32)
    actual_dimension = None
    for index, member in enumerate(members):
        positions = np.flatnonzero(labels == index)
        if not len(positions):
            continue
        torch.manual_seed(int(seed) + index * 104729)
        result = (
            member.sample_batched(
                (1,),
                x=torch.as_tensor(x_scaled[positions], dtype=torch.float32),
                max_sampling_batch_size=256,
                show_progress_bars=False,
                reject_outside_prior=False,
            )[0]
            .detach()
            .cpu()
            .numpy()
        )
        actual_dimension = result.shape[1]
        if samples.shape[1] != actual_dimension:
            samples = np.empty((n, actual_dimension), np.float32)
        samples[positions] = np.clip(result, 0.0, 1.0)
    if actual_dimension is None or not np.isfinite(samples).all():
        raise RuntimeError("batched mixture samples invalid")
    return samples


def generate_lc2st_datasets() -> dict[str, str]:
    n = int(
        read_preregistration()["diagnostics"][
            "lc2st_calibration_simulations_per_track"
        ]
    )
    theta8, theta7 = _paired_theta(n, int(SEEDS["lc2st_sobol"]))
    seeds = deterministic_seed_schedule(n, int(SEEDS["lc2st_simulator_base"]))
    outputs = {}
    for track, theta in (("8d", theta8), ("7d", theta7)):
        path = run_custom_dataset(
            theta,
            seeds,
            LC2ST_ROOT / "calibration" / track,
            track,
            "lc2st_calibration_20000",
        )
        outputs[track] = path.relative_to(RESULTS_ROOT).as_posix()
    atomic_json(
        LC2ST_ROOT / "lc2st_dataset_manifest.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "cases_per_track": n,
            "paths": outputs,
            "tracks_are_separate_posteriors": True,
            "same_shared_parameter_coordinates_and_seed_schedule": True,
            "preregistration_hash": verify_preregistration(),
        },
    )
    return outputs


def _posterior_samples_for_lc2st(
    track: str,
    estimator_index: int,
    x_scaled: np.ndarray,
    members: Sequence[Any],
) -> np.ndarray:
    seed = int(SEEDS[f"lc2st_posterior_{track}"]) + estimator_index * 104729
    if estimator_index == 0:
        return _mixture_sample_batched(members, x_scaled, seed)
    member = members[estimator_index - 1]
    torch.manual_seed(seed)
    values = (
        member.sample_batched(
            (1,),
            x=torch.as_tensor(x_scaled, dtype=torch.float32),
            max_sampling_batch_size=256,
            show_progress_bars=False,
            reject_outside_prior=False,
        )[0]
        .detach()
        .cpu()
        .numpy()
    )
    if not np.isfinite(values).all():
        raise RuntimeError("L-C2ST posterior samples are nonfinite")
    return np.clip(values, 0.0, 1.0).astype(np.float32, copy=False)


def run_lc2st(track: str) -> dict[str, Any]:
    """Train official-style LC2ST classifiers for ensemble and five members."""

    from sbi.diagnostics.lc2st import LC2ST

    verify_preregistration()
    dataset_path = (
        LC2ST_ROOT
        / "calibration"
        / track
        / f"lc2st_calibration_20000_{track}.npz"
    )
    primary_path = (
        PPC_ROOT
        / "primary_observation"
        / "8d"
        / "primary_observation_8d.npz"
    )
    with np.load(dataset_path, allow_pickle=False) as dataset:
        theta = np.asarray(dataset["theta"], float)
        x = np.asarray(dataset["x"], float)
        success = np.asarray(dataset["success"], bool)
    with np.load(primary_path, allow_pickle=False) as primary:
        x_o = np.asarray(primary["x"][0], float)
    if not success.all():
        raise RuntimeError("L-C2ST calibration dataset contains failed simulations")
    data, members = load_members(track)
    theta_unit = physical_to_unit(data, theta)
    x_scaled = scale_observation(data, x)
    x_o_scaled = scale_observation(data, x_o)
    estimator_names = ["ensemble"] + [
        f"member_{index + 1}" for index in range(len(members))
    ]
    rows = []
    for estimator_index, estimator in enumerate(estimator_names):
        output = LC2ST_ROOT / track / estimator
        output.mkdir(parents=True, exist_ok=True)
        result_path = output / "lc2st_result.json"
        classifier_path = output / "lc2st_classifier.pkl"
        if result_path.exists() and classifier_path.exists():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("preregistration_hash") == verify_preregistration():
                rows.append(existing)
                continue
        print(
            f"L-C2ST {track}/{estimator}: sampling posterior "
            f"({len(theta)} calibration rows)...",
            flush=True,
        )
        posterior_samples = _posterior_samples_for_lc2st(
            track, estimator_index, x_scaled, members
        )
        print(
            f"L-C2ST {track}/{estimator}: training classifier...",
            flush=True,
        )
        classifier_seed = int(SEEDS[f"lc2st_classifier_{track}"]) + (
            estimator_index * 1009
        )
        started = perf_counter()
        lc2st = LC2ST(
            thetas=torch.as_tensor(theta_unit, dtype=torch.float32),
            xs=torch.as_tensor(x_scaled, dtype=torch.float32),
            posterior_samples=torch.as_tensor(
                posterior_samples, dtype=torch.float32
            ),
            seed=classifier_seed,
            classifier="mlp",
            num_ensemble=1,
            num_folds=1,
            z_score=False,
            num_trials_null=100,
            device="cpu",
        )
        lc2st.train_under_null_hypothesis(verbosity=0)
        lc2st.train_on_observed_data(seed=classifier_seed + 1, verbosity=0)
        mixture = EqualMixturePosterior(members)
        n_local = int(
            read_preregistration()["diagnostics"][
                "lc2st_local_posterior_samples"
            ]
        )
        if estimator == "ensemble":
            local_samples, _ = mixture.sample(
                x_o_scaled, n_local, classifier_seed + 2
            )
        else:
            local_samples = sample_member(
                members[estimator_index - 1],
                x_o_scaled,
                n_local,
                classifier_seed + 2,
            )
        theta_o = torch.as_tensor(local_samples, dtype=torch.float32)
        x_o_tensor = torch.as_tensor(x_o_scaled, dtype=torch.float32)
        probabilities, score_data = lc2st.get_scores(
            theta_o=theta_o,
            x_o=x_o_tensor,
            return_probs=True,
            trained_clfs=lc2st.trained_clfs,
        )
        probabilities_null, scores_null = (
            lc2st.get_statistics_under_null_hypothesis(
                theta_o=theta_o,
                x_o=x_o_tensor,
                return_probs=True,
                verbosity=0,
            )
        )
        p_value = float(lc2st.p_value(theta_o, x_o_tensor.unsqueeze(0)))
        reject = bool(
            lc2st.reject_test(theta_o, x_o_tensor.unsqueeze(0), alpha=0.05)
        )
        with classifier_path.open("wb") as stream:
            pickle.dump(lc2st, stream)
        _atomic_npz(
            output / "lc2st_scores.npz",
            probabilities=np.asarray(probabilities, np.float32),
            score_data=np.asarray(score_data, np.float64),
            probabilities_null=np.asarray(probabilities_null, np.float32),
            scores_null=np.asarray(scores_null, np.float64),
            estimator=np.asarray(estimator, dtype="<U32"),
            track=np.asarray(track, dtype="<U8"),
            preregistration_hash=np.asarray(
                verify_preregistration(), dtype="<U64"
            ),
        )
        row = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "track": track,
            "estimator": estimator,
            "calibration_cases": len(theta),
            "local_posterior_samples": n_local,
            "score_observed": float(np.asarray(score_data).reshape(-1)[0]),
            "null_q95": float(np.quantile(scores_null, 0.95)),
            "p_value": p_value,
            "reject_alpha_0p05": reject,
            "classifier_seed": classifier_seed,
            "runtime_s": float(perf_counter() - started),
            "classifier_sha256": sha256_file(classifier_path),
            "preregistration_hash": verify_preregistration(),
        }
        atomic_json(result_path, row)
        rows.append(row)
        print(
            f"L-C2ST {track}/{estimator}: p={p_value:.4g} reject={reject}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(LC2ST_ROOT / track / "lc2st_summary.csv", index=False)
    ensemble = frame[frame.estimator == "ensemble"].iloc[0]
    summary = {
        "track": track,
        "ensemble_p_value": float(ensemble.p_value),
        "ensemble_reject": bool(ensemble.reject_alpha_0p05),
        "individual_rejections": int(
            frame[frame.estimator != "ensemble"].reject_alpha_0p05.sum()
        ),
        "preregistration_hash": verify_preregistration(),
    }
    atomic_json(LC2ST_ROOT / track / "lc2st_track_summary.json", summary)
    return summary


def run_ppc(track: str) -> dict[str, Any]:
    """Run new posterior and prior predictive simulations for primary x."""

    verify_preregistration()
    primary_path = (
        PPC_ROOT
        / "primary_observation"
        / "8d"
        / "primary_observation_8d.npz"
    )
    with np.load(primary_path, allow_pickle=False) as primary:
        observed = np.asarray(primary["x"][0], float)
    data, members = load_members(track)
    mixture = EqualMixturePosterior(members)
    x_scaled = scale_observation(data, observed)
    n_posterior = int(
        read_preregistration()["diagnostics"]["ppc_posterior_draws_per_track"]
    )
    posterior_unit, member_labels = mixture.sample(
        x_scaled,
        n_posterior,
        int(SEEDS[f"primary_posterior_sampling_{track}"]),
    )
    posterior_theta = unit_to_physical(data, posterior_unit)
    if track == "7d":
        posterior_full = np.column_stack(
            [posterior_theta, np.full(n_posterior, fixed_c_ctx2th())]
        )
    else:
        posterior_full = posterior_theta
    posterior_seeds = deterministic_seed_schedule(
        n_posterior, int(SEEDS[f"ppc_simulator_base_{track}"])
    )
    posterior_path = run_custom_dataset(
        posterior_full,
        posterior_seeds,
        PPC_ROOT / track / "posterior_predictive",
        track,
        "posterior_predictive",
    )
    n_prior = int(read_preregistration()["diagnostics"]["ppc_prior_draws"])
    prior8, prior7 = _paired_theta(n_prior, int(SEEDS["prior_ppc_sobol"]))
    prior_full = prior8 if track == "8d" else prior7
    prior_seeds = deterministic_seed_schedule(
        n_prior, int(SEEDS["prior_ppc_simulator_base"])
    )
    prior_path = run_custom_dataset(
        prior_full,
        prior_seeds,
        PPC_ROOT / track / "prior_predictive",
        track,
        "prior_predictive",
    )
    with np.load(posterior_path, allow_pickle=False) as posterior:
        posterior_x = np.asarray(posterior["x"], float)
        posterior_success = np.asarray(posterior["success"], bool)
    with np.load(prior_path, allow_pickle=False) as prior:
        prior_x = np.asarray(prior["x"], float)
        prior_success = np.asarray(prior["success"], bool)
    if not posterior_success.all() or not prior_success.all():
        raise RuntimeError("PPC contains failed simulations")
    posterior_error = np.median(
        np.abs((posterior_x - observed) / data.x_scale), axis=0
    )
    prior_error = np.median(
        np.abs((prior_x - observed) / data.x_scale), axis=0
    )
    improvement = 1 - posterior_error / np.maximum(prior_error, 1e-12)
    feature_rows = pd.DataFrame(
        {
            "feature": rate_feature_names(),
            "posterior_scaled_abs_error": posterior_error,
            "prior_scaled_abs_error": prior_error,
            "improvement_fraction": improvement,
            "posterior_better": posterior_error < prior_error,
        }
    )
    output = PPC_ROOT / track
    feature_rows.to_csv(output / "ppc_feature_metrics.csv", index=False)
    overall_posterior = float(np.mean(posterior_error))
    overall_prior = float(np.mean(prior_error))
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "track": track,
        "posterior_attempted": n_posterior,
        "posterior_failed": int((~posterior_success).sum()),
        "prior_attempted": n_prior,
        "prior_failed": int((~prior_success).sum()),
        "features_posterior_better": int(feature_rows.posterior_better.sum()),
        "overall_posterior_scaled_error": overall_posterior,
        "overall_prior_scaled_error": overall_prior,
        "overall_improvement_fraction": float(
            1 - overall_posterior / overall_prior
        ),
        "member_counts": np.bincount(
            member_labels, minlength=len(members)
        ).tolist(),
        "preregistration_hash": verify_preregistration(),
    }
    atomic_json(output / "ppc_summary.json", summary)
    return summary


def analyze_posterior_structure(track: str) -> dict[str, Any]:
    primary_path = (
        PPC_ROOT
        / "primary_observation"
        / "8d"
        / "primary_observation_8d.npz"
    )
    with np.load(primary_path, allow_pickle=False) as primary:
        observed = np.asarray(primary["x"][0], float)
    data, members = load_members(track)
    mixture = EqualMixturePosterior(members)
    x_scaled = scale_observation(data, observed)
    n = int(
        read_preregistration()["diagnostics"][
            "posterior_structure_samples_per_track"
        ]
    )
    samples_unit, labels = mixture.sample(
        x_scaled,
        n,
        int(SEEDS[f"primary_posterior_sampling_{track}"]) + 900_001,
    )
    samples = unit_to_physical(data, samples_unit)
    names = (
        list(PARAMETER_NAMES_8D) if track == "8d" else list(PARAMETER_NAMES_7D)
    )
    prior_sd = (data.theta_upper - data.theta_lower) / np.sqrt(12)
    posterior_sd = np.std(samples, axis=0, ddof=1)
    contraction = posterior_sd / prior_sd
    correlations = np.corrcoef(samples_unit, rowvar=False)
    rows = pd.DataFrame(
        {
            "parameter": names,
            "posterior_mean": np.mean(samples, axis=0),
            "posterior_median": np.median(samples, axis=0),
            "posterior_sd": posterior_sd,
            "prior_sd": prior_sd,
            "posterior_sd_over_prior_sd": contraction,
            "q05": np.quantile(samples, 0.05, axis=0),
            "q95": np.quantile(samples, 0.95, axis=0),
        }
    )
    output = STRUCTURE_ROOT / track
    output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / "posterior_marginals.csv", index=False)
    correlation_rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            correlation_rows.append(
                {
                    "parameter_a": names[i],
                    "parameter_b": names[j],
                    "correlation": float(correlations[i, j]),
                    "absolute_correlation": float(abs(correlations[i, j])),
                }
            )
    pd.DataFrame(correlation_rows).sort_values(
        "absolute_correlation", ascending=False
    ).to_csv(output / "posterior_correlations.csv", index=False)
    samples_path = output / "posterior_structure_samples.npz"
    _atomic_npz(
        samples_path,
        samples_physical=samples.astype(np.float32),
        samples_unit=samples_unit.astype(np.float32),
        member_labels=labels,
        parameter_names=np.asarray(names, dtype="<U32"),
        track=np.asarray(track, dtype="<U8"),
        preregistration_hash=np.asarray(verify_preregistration(), dtype="<U64"),
    )
    summary = {
        "track": track,
        "samples": n,
        "median_posterior_sd_over_prior_sd": float(np.median(contraction)),
        "least_contracted_parameter": names[int(np.argmax(contraction))],
        "least_contracted_ratio": float(np.max(contraction)),
        "most_contracted_parameter": names[int(np.argmin(contraction))],
        "most_contracted_ratio": float(np.min(contraction)),
        "strongest_pair": max(
            correlation_rows, key=lambda row: row["absolute_correlation"]
        ),
        "c_th2ctx_c_ctx2th_correlation": (
            float(
                correlations[
                    names.index("c_th2ctx"), names.index("c_ctx2th")
                ]
            )
            if track == "8d"
            else None
        ),
        "g_LK_g_h_correlation": float(
            correlations[names.index("g_LK"), names.index("g_h")]
        ),
        "preregistration_hash": verify_preregistration(),
    }
    atomic_json(output / "posterior_structure_summary.json", summary)
    return summary


def operational_verdict(track: str) -> dict[str, Any]:
    criteria = read_preregistration()["operational_verdict"]
    global_summary = json.loads(
        (
            GLOBAL_ROOT
            / "powered_1024"
            / track
            / f"scale_{FINAL_SCALE}"
            / "global_analysis_summary.json"
        ).read_text(encoding="utf-8")
    )
    lc2st = json.loads(
        (LC2ST_ROOT / track / "lc2st_track_summary.json").read_text(
            encoding="utf-8"
        )
    )
    ppc = json.loads(
        (PPC_ROOT / track / "ppc_summary.json").read_text(encoding="utf-8")
    )
    bank = json.loads(
        (RESULTS_ROOT / "matched_banks" / "matched_bank_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    failure_rate = float(bank[f"{track}_failure_rate"])
    technical = failure_rate <= float(
        criteria["technical_gates"]["bank_failure_rate_max"]
    )
    pass_checks = {
        "technical_gate": technical,
        "joint_coverage": global_summary["ensemble_joint_ks_pvalue"]
        >= criteria["pass"]["powered_joint_rank_ks_p_min"],
        "sbc": global_summary["ensemble_clear_sbc_issue_count"]
        <= criteria["pass"]["clear_sbc_issues_max"],
        "lc2st": not lc2st["ensemble_reject"],
        "ppc_features": ppc["features_posterior_better"]
        >= criteria["pass"]["ppc_features_better_than_prior_min"],
        "ppc_overall": ppc["overall_improvement_fraction"]
        >= criteria["pass"]["ppc_overall_improvement_min_fraction"],
    }
    qualified_checks = {
        "technical_gate": technical,
        "joint_coverage": global_summary["ensemble_joint_ks_pvalue"]
        >= criteria["qualified_pass"]["powered_joint_rank_ks_p_min"],
        "sbc": global_summary["ensemble_clear_sbc_issue_count"]
        <= criteria["qualified_pass"]["clear_sbc_issues_max"],
        "lc2st": not lc2st["ensemble_reject"],
        "ppc_features": ppc["features_posterior_better"]
        >= criteria["qualified_pass"]["ppc_features_better_than_prior_min"],
        "ppc_overall": ppc["overall_improvement_fraction"]
        >= criteria["qualified_pass"]["ppc_overall_improvement_min_fraction"],
    }
    if all(pass_checks.values()):
        verdict = f"{track.upper()} FIGURE-10-EQUIVALENT PASS"
    elif all(qualified_checks.values()):
        verdict = f"{track.upper()} FIGURE-10-EQUIVALENT QUALIFIED PASS"
    else:
        verdict = f"{track.upper()} FIGURE-10-EQUIVALENT FAIL"
    result = {
        "track": track,
        "verdict": verdict,
        "pass_checks": pass_checks,
        "qualified_checks": qualified_checks,
        "global": global_summary,
        "lc2st": lc2st,
        "ppc": ppc,
        "resource_scale_caveat": (
            f"{FINAL_SCALE} simulations per track, not official 3,000,000"
        ),
        "scope": "synthetic cortical-rate inference only",
        "preregistration_hash": verify_preregistration(),
    }
    atomic_json(DIAGNOSTICS_ROOT / f"{track}_operational_verdict.json", result)
    return result


__all__ = [
    "DIAGNOSTICS_ROOT",
    "GLOBAL_ROOT",
    "LC2ST_ROOT",
    "PPC_ROOT",
    "STRUCTURE_ROOT",
    "analyze_global_ranks",
    "analyze_posterior_structure",
    "generate_global_datasets",
    "generate_lc2st_datasets",
    "generate_primary_observation",
    "operational_verdict",
    "run_global_diagnostic",
    "run_lc2st",
    "run_ppc",
]
