"""Fast, simulation-free checks for the EEG mapping audit helper."""

from __future__ import annotations

import numpy as np

from sleep_sbi.eeg_observation_mapping_audit import (
    _mean_acf,
    _normalized_welch,
    audit_parameter_sets,
    mapping_option_decisions,
)


def test_audit_uses_v7_v8_v8a_only() -> None:
    parameter_sets = audit_parameter_sets()
    assert [item.identifier for item in parameter_sets] == [
        "v7_fitted",
        "v8_fitted",
        "v8a_local_best",
    ]


def test_mapping_options_cover_a_through_h_and_reject_route_1() -> None:
    decisions = mapping_option_decisions()
    assert len(decisions) == 8
    assert decisions.iloc[0]["option"].startswith("A.")
    assert decisions.iloc[-1]["option"].startswith("H.")
    assert "Reject" in decisions.iloc[0]["decision"]
    assert "Recommended" in decisions.loc[
        decisions["option"].str.startswith("F."), "decision"
    ].item()


def test_normalized_welch_has_unit_band_area() -> None:
    fs_hz = 100.0
    time_s = np.arange(3000) / fs_hz
    signal = np.sin(2 * np.pi * time_s) + 0.2 * np.sin(2 * np.pi * 12 * time_s)
    frequencies, density = _normalized_welch(signal, fs_hz)
    mask = (frequencies >= 0.5) & (frequencies <= 20.0)
    assert np.isclose(np.trapezoid(density[mask], frequencies[mask]), 1.0)
    assert frequencies[mask][np.argmax(density[mask])] == 1.0


def test_acf_is_epoch_local_and_normalized_at_zero_lag() -> None:
    fs_hz = 100.0
    segment_a = np.sin(2 * np.pi * np.arange(3000) / fs_hz)
    segment_b = -segment_a
    lags, acf = _mean_acf(np.vstack([segment_a, segment_b]), fs_hz)
    assert lags[0] == 0.0
    assert acf[0] == 1.0
    assert np.isfinite(acf).all()

