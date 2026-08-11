from __future__ import annotations

import numpy as np
import pytest

from sleep_sbi.forward_model_feasibility import (
    MeasurementContractError,
    apply_measurement_model,
    calibration_leakage_matrix,
    leadfield_requirement_matrix,
    measurement_contract_components,
    route_decision_matrix,
    single_source_rank_audit,
    source_candidate_decisions,
    validate_measurement_declaration,
)


def test_decision_tables_have_stable_unique_entries() -> None:
    sources = source_candidate_decisions()
    contract = measurement_contract_components()
    routes = route_decision_matrix()

    assert len(sources) >= 10
    assert sources["candidate"].is_unique
    assert contract["stage_index"].tolist() == list(range(1, len(contract) + 1))
    assert contract["component"].is_unique
    assert set(routes.columns) == {
        "criterion",
        "route_2_measurement_model",
        "route_3_synthetic_observable",
        "current_assessment",
    }
    assert len(calibration_leakage_matrix()) >= 10
    assert len(leadfield_requirement_matrix()) >= 10


def test_empty_contract_is_explicit_no_go() -> None:
    gate = validate_measurement_declaration({})
    assert gate.status == "NO-GO"
    assert not gate.passed
    assert "source_variable" in gate.missing_fields
    assert "leadfield_id" in gate.missing_fields
    assert "calibration_independent_of_sc4001" in gate.leakage_fields


def test_single_source_is_rank_one_not_necessarily_zero() -> None:
    audit = single_source_rank_audit()
    assert audit["temporal_rank"] == 1
    assert audit["can_be_nonzero"] is True
    assert audit["adds_new_temporal_structure"] is False
    assert "L_Fpz-L_Cz" in audit["bipolar_model"]


def test_no_arbitrary_forward_fallback() -> None:
    with pytest.raises(MeasurementContractError):
        apply_measurement_model(np.ones(100), {})
