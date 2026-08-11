import json

import numpy as np

from sleep_sbi.observation import map_raw_stage


def test_sc4001_loads_with_real_annotation_labels(observation_bundle):
    bundle = observation_bundle
    assert bundle.subject_id == "SC4001"
    assert bundle.channel == "EEG Fpz-Cz"
    assert bundle.fs_hz == 100.0
    assert bundle.epoch_duration_s == 30.0
    assert bundle.raw_label_counts == {
        "Sleep stage 1": 24,
        "Sleep stage 2": 40,
        "Sleep stage 3": 48,
        "Sleep stage 4": 23,
        "Sleep stage ?": 1,
        "Sleep stage R": 6,
        "Sleep stage W": 12,
    }
    assert bundle.provenance["aligned_aasm_counts"]["N3"] == 220
    assert bundle.provenance["reference"].startswith("bipolar derivation Fpz-Cz")


def test_stage_3_and_stage_4_are_n3_including_short_label():
    assert map_raw_stage("Sleep stage 3") == "N3"
    assert map_raw_stage("Sleep stage 4") == "N3"
    assert map_raw_stage("3") == "N3"
    assert map_raw_stage("2") == "N2"


def test_publication_metadata_has_no_raw_arrays_or_absolute_paths(
    observation_bundle,
):
    payload = observation_bundle.publication_safe_dict()
    assert "segments" not in payload
    assert "psd" not in payload
    assert "diagnostics" not in payload
    encoded = json.dumps(payload, ensure_ascii=True)
    assert "D:\\\\" not in encoded
    assert "C:\\\\" not in encoded
    assert np.asarray(observation_bundle.segments).ndim == 2
    assert payload["provenance"]["full_raw_eeg_serialized"] is False
