import numpy as np


def test_n3_qc_accounting_and_exact_epoch_lengths(observation_bundle):
    bundle = observation_bundle
    assert len(bundle.n3_epoch_indices) == 220
    assert len(bundle.retained_epoch_indices) == 142
    assert len(bundle.rejected_epoch_indices) == 78
    assert (
        len(bundle.retained_epoch_indices) + len(bundle.rejected_epoch_indices)
        == len(bundle.n3_epoch_indices)
    )
    assert bundle.segments.shape == (142, 30 * 100)
    assert set(bundle.retained_epoch_indices).isdisjoint(
        set(bundle.rejected_epoch_indices)
    )
    assert set(bundle.rejection_reasons) == set(bundle.rejected_epoch_indices)


def test_detectors_never_emit_cross_epoch_events(observation_bundle):
    bundle = observation_bundle
    samples_per_epoch = int(bundle.epoch_duration_s * bundle.fs_hz)
    retained = set(map(int, bundle.retained_epoch_indices))

    for event in bundle.diagnostics["slow_oscillation"]["events"]:
        assert event["epoch_index"] in retained
        assert 0 <= event["down_sample"] < samples_per_epoch
        assert 0 <= event["up_sample"] < samples_per_epoch
        assert event["down_sample"] < event["up_sample"]

    for event in bundle.diagnostics["spindle"]["events"]:
        assert event["epoch_index"] in retained
        assert 0 <= event["start_sample"] < event["stop_sample"]
        assert event["stop_sample"] <= samples_per_epoch

    assert "no concatenation before filtering" in (
        bundle.provenance["epoch_boundary_policy"]
    )


def test_detector_validity_masks_cover_retained_epochs(observation_bundle):
    n_retained = len(observation_bundle.retained_epoch_indices)
    for name in ("slow_oscillation", "spindle", "pac"):
        mask = observation_bundle.diagnostics[name]["validity_mask"]
        assert mask.dtype == np.bool_
        assert mask.shape == (n_retained,)
        failures = observation_bundle.diagnostics[name]["failure_reasons"]
        assert int((~mask).sum()) == len(failures)


def test_so_intervals_and_waveforms_respect_epoch_boundaries(observation_bundle):
    so = observation_bundle.diagnostics["slow_oscillation"]
    assert so["ibi_validity_mask"].shape == (
        len(observation_bundle.retained_epoch_indices),
    )
    assert so["waveform_validity_mask"].shape == (
        len(observation_bundle.retained_epoch_indices),
    )
    assert np.all(so["ibi_s"] > 0)
    assert np.all(so["ibi_s"] < observation_bundle.epoch_duration_s)
    for row in so["per_epoch"]:
        if row["ibi_valid"]:
            assert row["ibi_count"] >= 2
            assert np.isfinite(row["ibi_cv"])
        else:
            assert row["invalid_reason"] is not None
    for event in so["events"]:
        if not event["waveform_valid"]:
            assert event["waveform_invalid_reason"] is not None
    assert so["waveform_boundary_excluded_count"] == sum(
        event["waveform_invalid_reason"] == "incomplete_boundary_window"
        for event in so["events"]
    )
