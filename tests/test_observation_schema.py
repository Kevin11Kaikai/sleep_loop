import numpy as np

from sleep_sbi import metric_classification_rows


def test_observation_bundle_schema_and_summary_metadata(observation_bundle):
    bundle = observation_bundle
    bundle.validate()
    assert "T11_lag_ms" not in bundle.summaries
    assert "pac_up_down_ratio" in bundle.summaries
    assert "pac_preferred_phase_rad" in bundle.summaries
    assert "pac_preferred_phase_sin" in bundle.summaries
    assert "pac_preferred_phase_cos" in bundle.summaries
    required = {
        "unit",
        "band_hz",
        "algorithm",
        "aggregation",
        "source_signal",
        "category",
        "valid",
        "field_name",
        "frequency_band",
        "aggregation_method",
        "valid_epoch_count",
        "validity_status",
        "intended_role",
        "warnings",
    }
    allowed_roles = {
        "inference_summary_candidate",
        "mechanism_diagnostic",
        "held_out_ppc_candidate",
    }
    for row in bundle.summary_rows():
        assert required <= set(row)
        assert row["unit"]
        assert row["algorithm"]
        assert row["aggregation"]
        assert row["intended_role"] in allowed_roles
        assert row["valid_epoch_count"] >= 0
        if row["valid"]:
            assert np.isfinite(row["value"])
    assert bundle.provenance["software_versions"]["fooof"] == "1.1.1"


def test_psd_schema_freezes_hann_and_records_hamming_sensitivity(
    observation_bundle,
):
    psd = observation_bundle.psd
    assert psd.parameters["primary_window"] == "hann"
    assert psd.parameters["sensitivity_window"] == "hamming"
    assert psd.parameters["unit"] == "uV^2/Hz"
    assert psd.epoch_psd_hann_uv2_hz.shape[0] == 142
    assert psd.epoch_psd_hamming_uv2_hz.shape == (
        psd.epoch_psd_hann_uv2_hz.shape
    )
    assert np.isclose(
        np.trapezoid(
            psd.normalized_hann[
                (psd.frequencies_hz >= 0.5) & (psd.frequencies_hz <= 20.0)
            ],
            psd.frequencies_hz[
                (psd.frequencies_hz >= 0.5) & (psd.frequencies_hz <= 20.0)
            ],
        ),
        1.0,
    )


def test_metric_classification_keeps_mechanism_and_ppc_roles_separate():
    rows = metric_classification_rows()
    by_metric = {row["metric"]: row for row in rows}
    assert "simulator-side mechanism only" in by_metric[
        "T8 / T12 thalamic spindle diagnostics"
    ]["note"]
    assert "simulator-side mechanism only" in by_metric["V8a internal T13"][
        "note"
    ]
    assert "legacy misnomer" in by_metric["pac_up_down_ratio"]["note"]
