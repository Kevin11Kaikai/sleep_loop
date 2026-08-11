from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from sleep_sbi import load_observation_config
from sleep_sbi.pyloric_inspired_observation import (
    CORE_FIELDS,
    build_pyloric_inspired_observation,
    generate_and_save_figures,
    load_pyloric_inspired_config,
    write_publication_safe_artifacts,
)


@pytest.fixture(scope="session")
def pyloric_result(observation_bundle):
    return build_pyloric_inspired_observation(
        observation_bundle,
        load_observation_config(),
        load_pyloric_inspired_config(),
    )


def test_core_schema_is_exactly_18_unique_numeric_fields(pyloric_result):
    summary = pyloric_result.summary
    assert tuple(summary["field_name"]) == CORE_FIELDS
    assert len(summary) == 18
    assert summary["field_name"].is_unique
    assert "spindle_onset_phase_rad" not in set(summary["field_name"])
    assert not np.isinf(summary["value"].to_numpy(float)).any()
    assert set(summary["intended_role"]) == {
        "inference_summary_candidate",
        "held_out_ppc_candidate",
    }


def test_bounded_and_positive_metrics(pyloric_result):
    values = pyloric_result.summary.set_index("field_name")["value"]
    for field_name in (
        "so_up_proxy_duty_cycle",
        "spindle_occupancy",
        "spindle_onset_phase_concentration",
    ):
        assert 0.0 <= values[field_name] <= 1.0
    for field_name in (
        "so_up_proxy_duration_s",
        "so_down_proxy_duration_s",
        "so_trough_to_peak_time_s",
        "spindle_mean_duration_s",
    ):
        assert values[field_name] > 0.0


def test_epoch_ledger_and_metric_specific_support(
    observation_bundle, pyloric_result
):
    ledger = pyloric_result.epoch_ledger
    retained = ledger["global_qc_status"].eq("retained")
    rejected = ledger["global_qc_status"].eq("rejected")
    assert retained.sum() == len(observation_bundle.retained_epoch_indices)
    assert rejected.sum() == len(observation_bundle.rejected_epoch_indices)
    assert retained.sum() + rejected.sum() == len(
        observation_bundle.n3_epoch_indices
    )
    assert (
        ledger.loc[~ledger["is_n3"], "global_qc_status"]
        == "not_applicable_non_n3"
    ).all()
    assert ledger.loc[rejected, "rejection_reasons"].str.len().gt(0).all()
    assert ledger.loc[retained, "ibi_valid"].sum() < retained.sum()
    assert ledger.loc[retained, "psd_valid"].all()


def test_zero_events_and_undefined_duration_are_distinct(pyloric_result):
    per_epoch = pyloric_result.per_epoch_features
    zero_event = per_epoch["spindle_event_count"].eq(0)
    assert zero_event.any()
    assert (per_epoch.loc[zero_event, "spindle_density_per_min"] == 0.0).all()
    assert (per_epoch.loc[zero_event, "spindle_occupancy"] == 0.0).all()
    assert per_epoch.loc[zero_event, "spindle_mean_duration_s"].isna().all()
    assert (
        per_epoch.loc[zero_event, "spindle_invalid_reason"]
        == "no_spindle_events_mean_duration_undefined"
    ).all()


def test_cycles_and_pairing_never_cross_epoch_boundaries(
    observation_bundle, pyloric_result
):
    samples_per_epoch = int(
        observation_bundle.fs_hz * observation_bundle.epoch_duration_s
    )
    cycles = pyloric_result.diagnostics["so_proxy_cycles"]
    valid = cycles.loc[cycles["valid"]]
    assert (
        valid["negative_entry_sample"]
        < valid["negative_to_positive_sample"]
    ).all()
    assert (
        valid["negative_to_positive_sample"] < valid["positive_exit_sample"]
    ).all()
    assert (valid["negative_entry_sample"] >= 0).all()
    assert (valid["positive_exit_sample"] < samples_per_epoch).all()
    pairs = pyloric_result.diagnostics["paired_events"]
    assert (pairs["left_trough_sample"] <= pairs["spindle_onset_sample"]).all()
    assert (pairs["spindle_onset_sample"] < pairs["right_trough_sample"]).all()
    assert (pairs["phase_rad"] >= 0.0).all()
    assert (pairs["phase_rad"] < 2.0 * np.pi).all()


def test_summary_metadata_regression_and_validation(pyloric_result):
    required = {
        "field_name",
        "value",
        "unit",
        "frequency_band",
        "algorithm",
        "aggregation_method",
        "valid_epoch_count",
        "valid_event_count",
        "validity_status",
        "intended_role",
        "dependency_or_redundancy",
        "warnings",
    }
    assert required <= set(pyloric_result.summary.columns)
    assert pyloric_result.validation_checks["passed"].all()
    assert (
        pyloric_result.old_vs_new_regression["status"] == "pass"
    ).all()


def test_publication_safe_exports_and_figures(
    observation_bundle, pyloric_result
):
    config = load_pyloric_inspired_config()
    observation_config = load_observation_config()
    manifest = generate_and_save_figures(
        observation_bundle, pyloric_result, observation_config, config
    )
    output = write_publication_safe_artifacts(
        observation_bundle, pyloric_result, config, manifest
    )
    assert len(manifest) == 11
    for relative in manifest[["png", "pdf"]].to_numpy().flat:
        artifact = output / relative
        assert artifact.is_file()
        assert artifact.stat().st_size > 1000
    expected = {
        "pyloric_inspired_18d_summary.csv",
        "pyloric_inspired_18d_summary.json",
        "pyloric_inspired_per_epoch_features.csv",
        "epoch_ledger.csv",
        "feature_dictionary.csv",
        "old_vs_new_regression.csv",
        "validation_report.md",
    }
    assert expected <= {path.name for path in output.iterdir()}
    assert not any(path.suffix.lower() in {".edf", ".npz"} for path in output.rglob("*"))
    assert "D:\\\\" not in (
        output / "pyloric_inspired_18d_summary.json"
    ).read_text(encoding="utf-8")
    plt.close("all")
