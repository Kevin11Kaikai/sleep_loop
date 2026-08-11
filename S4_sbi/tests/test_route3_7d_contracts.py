from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "S4_sbi" / "src"))
sys.path.insert(0, str(ROOT))

from sleep_sbi.route3_7d_experiment import (
    PARAMETER_NAMES_7D,
    fixed_c_ctx2th,
    insert_fixed_parameter,
    prior_bounds_7d,
    rate_feature_names,
    read_preregistration,
    verify_preregistration,
)
from sleep_sbi.route3_7d_training import ARCHITECTURE, TRAINING_POLICY, model_seeds


def test_preregistration_hash_and_frozen_dimensions():
    digest = verify_preregistration()
    assert len(digest) == 64
    prereg = read_preregistration()
    assert tuple(prereg["parameter_order_7d"]) == PARAMETER_NAMES_7D
    assert len(rate_feature_names()) == 14
    assert len(set(rate_feature_names())) == 14
    assert prereg["seed_nonoverlap_audit"]["intersection_count"] == 0


def test_fixed_parameter_insertion_is_exact_and_last():
    bounds = prior_bounds_7d()
    theta = bounds.mean(axis=1)
    full = insert_fixed_parameter(theta)
    assert full.shape == (8,)
    assert np.array_equal(full[:7], theta)
    assert full[7] == fixed_c_ctx2th()
    matrix = insert_fixed_parameter(np.vstack([theta, theta]))
    assert matrix.shape == (2, 8)
    assert np.all(matrix[:, 7] == fixed_c_ctx2th())


def test_fixed_value_is_inside_old_prior_and_architecture_is_unchanged():
    prereg = read_preregistration()
    lower, upper = prereg["fixed_parameter"]["prior_range"]
    assert lower <= fixed_c_ctx2th() <= upper
    assert ARCHITECTURE == prereg["training"]["architecture"]
    assert TRAINING_POLICY == prereg["training"]["policy"]
    assert len(model_seeds()) == 3
    assert len(set(model_seeds())) == 3


def test_decision_counts_are_preregistered_for_seven_parameters():
    criteria = read_preregistration()["decision_criteria"]
    formal = criteria["formal_go"]
    assert formal["coverage_80_or_90_wilson_contains_nominal_parameters_min"] == 6
    assert formal["median_error_better_than_prior_parameters_min"] == 6
    assert formal["meaningful_contraction_with_reasonable_coverage_parameters_min"] == 6
    assert prereg_ppc_counts() == (64, 32, 32, 16)


def prereg_ppc_counts():
    validation = read_preregistration()["validation"]
    return (
        validation["ppc_cases"],
        validation["ppc_random_cases"],
        validation["ppc_worst_cases"],
        validation["ppc_draws_per_case"],
    )
