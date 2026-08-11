"""Fast contract tests for the Route-3 synthetic preflight."""

from __future__ import annotations

import numpy as np

from sleep_sbi.route3_synthetic_preflight import (
    AUGMENTED_FEATURES,
    PARAMETER_NAMES,
    RATE_ONLY_FEATURES,
    parameter_contract,
    schema_specs,
)


def test_parameter_contract_is_eight_dimensional_and_centers_are_inside() -> None:
    contract = parameter_contract()
    assert tuple(contract["parameter_names"]) == PARAMETER_NAMES
    assert np.asarray(contract["bounds"], dtype=float).shape == (8, 2)
    assert all(center["inside_v8a_prior"] for center in contract["centers"])


def test_route3_schemas_are_nested_and_fixed_length() -> None:
    rate_names = tuple(spec.name for spec in RATE_ONLY_FEATURES)
    augmented_names = tuple(spec.name for spec in AUGMENTED_FEATURES)
    assert len(rate_names) == 14
    assert len(augmented_names) == 24
    assert augmented_names[:14] == rate_names
    assert len(set(rate_names)) == 14
    assert len(set(augmented_names)) == 24
    assert schema_specs("cortex_rate_only_14d") == RATE_ONLY_FEATURES
    assert schema_specs("cortex_state_augmented_24d") == AUGMENTED_FEATURES


def test_route3_feature_contract_does_not_claim_eeg_units() -> None:
    for spec in AUGMENTED_FEATURES:
        assert "uV" not in spec.unit
        assert "microvolt" not in spec.unit.lower()
        assert "EEG" not in spec.scientific_interpretation
