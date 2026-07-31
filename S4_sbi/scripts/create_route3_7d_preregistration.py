"""Create or verify the immutable Route-3 7D preregistration artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "S4_sbi" / "src"
for value in (SRC, ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from sleep_sbi.route3_global_robustness import (
    deterministic_seed_schedule,
    rate_contract_hash,
    rate_feature_names,
)
from sleep_sbi.route3_pilot_snpe import ARCHITECTURE, TRAINING_POLICY
from sleep_sbi.route3_synthetic_preflight import parameter_contract


CONFIG = ROOT / "S4_sbi" / "configs" / "route3_7d_preregistered_v1.json"
LOCKED = ROOT / "S4_sbi" / "artifacts" / "route3_7d_preregistered_v1.locked.json"
DIGEST = ROOT / "S4_sbi" / "artifacts" / "route3_7d_preregistered_v1.sha256"
V8A_SOURCE = ROOT / "outputs" / "v8a_ultra_narrow_t6_t13_search" / "best_so_far.json"


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def old_simulator_seeds() -> set[int]:
    roots = [
        ROOT / "S4_sbi" / "results" / "route3_pilot_snpe" / "global_robustness"
        / "prior_multiseed" / "route3_prior_multiseed_128x3.npz",
        ROOT / "S4_sbi" / "results" / "route3_pilot_snpe" / "global_robustness"
        / "local_multiscale" / "route3_local_multiscale.npz",
        ROOT / "S4_sbi" / "results" / "route3_pilot_snpe" / "simulation_bank_2048"
        / "route3_cortex_rate_14d_bank_2048.npz",
        ROOT / "S4_sbi" / "results" / "route3_pilot_snpe" / "heldout_validation"
        / "dataset" / "route3_heldout_128.npz",
        ROOT / "S4_sbi" / "results" / "route3_pilot_snpe" / "heldout_validation"
        / "ppc" / "posterior_predictive_512.npz",
        ROOT / "S4_sbi" / "results" / "route3_pilot_snpe" / "heldout_validation"
        / "ppc" / "prior_predictive_512.npz",
    ]
    values: set[int] = set()
    for path in roots:
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            values.update(int(v) for v in data["simulator_seed"])
    return values


def build_payload() -> dict:
    old_contract = parameter_contract()
    bounds = old_contract["bounds"]
    source = json.loads(V8A_SOURCE.read_text(encoding="utf-8"))
    fixed = float(source["c_ctx2th"])
    prior_range = bounds[7]
    if not prior_range[0] <= fixed <= prior_range[1]:
        raise RuntimeError("V8a local-best c_ctx2th is outside the old 8D prior")
    seeds = {
        "preflight_sobol": 7315001,
        "preflight_simulator_replicates": [7315101, 7315201, 7315301],
        "local_simulator_replicates": [7315101, 7315201, 7315301],
        "training_sobol": 7320001,
        "training_simulator_base": 7321001,
        "train_validation_split": 7322001,
        "network_initialization": [7330001, 7330002, 7330003],
        "ensemble_validation_sampling": 7331001,
        "heldout_sobol": 7340001,
        "heldout_simulator_base": 7341001,
        "posterior_sampling": 7342001,
        "ppc_case_selection": 7343001,
        "ppc_posterior_simulator_base": 7344001,
        "ppc_prior_sobol": 7345001,
        "ppc_prior_simulator_base": 7346001,
    }
    new_simulator_seeds = set(seeds["preflight_simulator_replicates"])
    new_simulator_seeds.update(
        int(v) for v in deterministic_seed_schedule(4096, seeds["training_simulator_base"])
    )
    new_simulator_seeds.update(
        int(v) for v in deterministic_seed_schedule(256, seeds["heldout_simulator_base"])
    )
    new_simulator_seeds.update(
        int(v) for v in deterministic_seed_schedule(1024, seeds["ppc_posterior_simulator_base"])
    )
    new_simulator_seeds.update(
        int(v) for v in deterministic_seed_schedule(1024, seeds["ppc_prior_simulator_base"])
    )
    old_seeds = old_simulator_seeds()
    overlap = sorted(old_seeds & new_simulator_seeds)
    if overlap:
        raise RuntimeError(f"new 7D simulator seed schedule overlaps old 8D seeds: {overlap[:10]}")
    criteria = {
        "version": "route3-7d-formal-go-criteria-v1",
        "copied_from": "route3-heldout-go-criteria-v1",
        "adaptation": "8 to 7 free parameters; 4096 train; 256 held-out; 64 PPC cases",
        "formal_go": {
            "preflight_failure_rate_max": 0.01,
            "training_failure_rate_max": 0.01,
            "heldout_failure_rate_max": 0.01,
            "posterior_finite_and_in_prior_rate_min": 0.999,
            "coverage_80_or_90_wilson_contains_nominal_parameters_min": 6,
            "severe_undercoverage_definition": (
                "observed coverage < nominal-0.15 and Wilson upper bound < nominal"
            ),
            "severe_undercoverage_parameters_max": 0,
            "median_error_better_than_prior_parameters_min": 6,
            "overall_median_error_improvement_min_fraction": 0.10,
            "systematic_worsening_threshold": -0.05,
            "systematically_worse_parameters_max": 0,
            "prior_level_improvement_max": 0.05,
            "prior_level_rank_correlation_max": 0.10,
            "prior_level_unrecoverable_parameters_max": 0,
            "meaningful_contraction_definition": "median 90% CI width < 0.90 prior width",
            "meaningful_contraction_width_max": 0.90,
            "meaningful_contraction_with_reasonable_coverage_parameters_min": 6,
            "ppc_features_better_than_prior_min": 10,
            "overall_ppc_error_improvement_min_fraction": 0.10,
            "ensemble_member_median_disagreement_mean_max_prior_width": 0.10,
            "ensemble_member_disagreement_parameter_max_prior_width": 0.20,
            "global_collision_requirement": (
                "coverage and contraction must remain honest despite collisions; "
                "collision count is reported, not tuned after held-out results"
            ),
        },
        "conditional_go": {
            "engineering_failure_and_finite_gates_must_pass": True,
            "severe_undercoverage_parameters_max": 1,
            "median_error_better_than_prior_parameters_min": 5,
            "meaningful_contraction_with_reasonable_coverage_parameters_min": 5,
            "overall_ppc_must_improve": True,
            "ensemble_disagreement_gate_must_pass": True,
        },
        "no_go_if": [
            "conditional-go criteria fail",
            "one or more parameters remain prior-level unrecoverable",
            "multiple parameters show severe undercoverage or false contraction",
            "seed noise or global collisions dominate recovery",
            "ensemble scientific conclusions are unstable",
            "synthetic PPC does not improve over prior predictive",
        ],
    }
    return {
        "experiment_id": "route3_7d_fixed_c_ctx2th_formal_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "immutable_after_first_scientific_simulation": True,
        "parameter_order_8d": old_contract["parameter_names"],
        "parameter_order_7d": old_contract["parameter_names"][:7],
        "parameter_units_7d": old_contract["parameter_units"][:7],
        "prior_type": old_contract["prior_type"],
        "prior_bounds_7d": bounds[:7],
        "fixed_parameter": {
            "name": "c_ctx2th",
            "value": fixed,
            "unit": old_contract["parameter_units"][7],
            "selection_rule": (
                "Notebook 10 canonical V8a local best; no recovery/posterior result used"
            ),
            "source": "outputs/v8a_ultra_narrow_t6_t13_search/best_so_far.json",
            "source_sha256": file_hash(V8A_SOURCE),
            "prior_range": prior_range,
            "inside_old_8d_prior": True,
            "conditional_claim": (
                "All seven-parameter conclusions are conditional on this fixed value."
            ),
        },
        "observation_schema": {
            "name": "cortex-rate-only 14D",
            "schema_version": "route3-cortex-rate-14d-v1",
            "schema_hash": rate_contract_hash(),
            "feature_names": list(rate_feature_names()),
            "dimension": 14,
            "scope": "synthetic cortical population firing-rate observable; not EEG",
        },
        "simulator_contract": {
            "model_version": "v8a_t13",
            "full_parameter_insertion_order": old_contract["parameter_names"],
            "sampling_dt_ms": old_contract["sampling_dt_ms"],
            "model_integration_dt_ms": old_contract["model_integration_dt_ms"],
            "duration_s": old_contract["duration_s"],
            "warm_up_s": old_contract["warm_up_s"],
            "analyzed_window_s": old_contract["analyzed_window_s"],
            "seed_policy": "all four stochastic inputs plus numba RNG",
            "failed_policy": "retain theta; NaN x; validity false; explicit reason",
            "nonfinite_policy": "invalid; never fill with zero",
        },
        "training": {
            "scheduled_simulations": 4096,
            "split": "80/20 over valid rows with preregistered split seed",
            "feature_scaling": "median/IQR fitted only on training split",
            "architecture": ARCHITECTURE,
            "policy": TRAINING_POLICY,
            "members": 3,
            "proposal": "independent uniform 7D prior",
            "rounds": 1,
            "ensemble": "equal-count mixture; no learned weights",
        },
        "validation": {
            "preflight_prior_points": 128,
            "preflight_replicates": 3,
            "local_centers": ["v7_fitted", "v8_fitted", "v8a_local_best"],
            "local_scales": [0.01, 0.02, 0.05],
            "rank_rule": "relative singular value >= 1e-3 largest singular value",
            "heldout_cases": 256,
            "posterior_samples_per_case": 4096,
            "ppc_cases": 64,
            "ppc_random_cases": 32,
            "ppc_worst_cases": 32,
            "ppc_worst_rule": (
                "top mean normalized posterior-median absolute recovery error "
                "after excluding the 32 fixed-random cases"
            ),
            "ppc_draws_per_case": 16,
        },
        "seeds": seeds,
        "seed_nonoverlap_audit": {
            "old_simulator_seed_count": len(old_seeds),
            "new_planned_simulator_seed_count": len(new_simulator_seeds),
            "intersection_count": 0,
            "old_seed_artifact_scope": (
                "8D multiseed, local, 2048 bank, 128 held-out, posterior/prior PPC"
            ),
            "old_explicit_seeds": [
                1103, 2207, 3301, 710001, 910001, 920001, 930001,
                1301, 1302, 1303, 20260730, 20260731, 20260802,
                20260803, 20260804, 20260805,
            ],
        },
        "decision_criteria": criteria,
        "allowed_claims": [
            "recoverability of seven free parameters conditional on fixed c_ctx2th",
            "synthetic cortical-rate observable space only",
        ],
        "forbidden_claims": [
            "real EEG parameter inference",
            "simulated EEG",
            "validated Fpz-Cz measurement model",
            "mechanism truth from observable-level recovery",
        ],
        "package_versions": {
            name: metadata.version(name)
            for name in ("neurolib", "sbi", "torch", "numpy", "scipy", "pandas")
        },
    }


def main() -> None:
    if CONFIG.exists() or LOCKED.exists() or DIGEST.exists():
        if not (CONFIG.exists() and LOCKED.exists() and DIGEST.exists()):
            raise RuntimeError("partial preregistration artifact set exists")
        expected = DIGEST.read_text(encoding="ascii").strip()
        if file_hash(CONFIG) != expected or file_hash(LOCKED) != expected:
            raise RuntimeError("existing preregistration hash mismatch")
        print(json.dumps({"status": "verified_existing", "sha256": expected}, indent=2))
        return
    payload = build_payload()
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    LOCKED.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    shutil.copyfile(CONFIG, LOCKED)
    digest = file_hash(CONFIG)
    if file_hash(LOCKED) != digest:
        raise RuntimeError("locked preregistration copy mismatch")
    DIGEST.write_text(digest + "\n", encoding="ascii")
    print(json.dumps({
        "status": "created", "sha256": digest,
        "fixed_c_ctx2th": payload["fixed_parameter"]["value"],
        "seed_intersection_count": payload["seed_nonoverlap_audit"]["intersection_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
