"""Data-free synthetic adapter and execution-boundary tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import r4_empirical
from r4_adapter import (
    ANNOTATION_RELATIVE,
    PSG_RELATIVE,
    build_n3_epoch_metadata,
    choose_channel,
    comparison_features,
    guarded_relative_path,
    steward,
)
from r4_core import (
    canonical_bytes,
    canonical_sha256,
    load_protocol_claims,
    new_output_child,
    read_json,
    sha256_file,
    validate_frozen_inputs,
    write_canonical,
)


class FakeAnnotations:
    def __init__(self, onset, duration, description):
        self.onset = np.asarray(onset, dtype=float)
        self.duration = np.asarray(duration, dtype=float)
        self.description = np.asarray(description, dtype=object)


class FakeRaw:
    def __init__(self, seconds: float = 120.0, sfreq: float = 100.0, channels=None):
        self.info = {"sfreq": sfreq}
        self.ch_names = list(channels or ["EEG Pz-Oz", "EEG Fpz-Cz"])
        self.n_times = int(round(seconds * sfreq))
        time_s = np.arange(self.n_times, dtype=np.float64) / sfreq
        self._signal = (
            np.sin(2.0 * np.pi * 0.75 * time_s)
            + 0.8 * np.sin(2.0 * np.pi * 2.0 * time_s)
            + 0.5 * np.sin(2.0 * np.pi * 6.0 * time_s)
            + 0.3 * np.sin(2.0 * np.pi * 13.0 * time_s)
        )

    def get_data(self, *, picks, start, stop):
        if picks[0] not in self.ch_names:
            raise KeyError(picks[0])
        return self._signal[start:stop][None, :]


class FakeMNE:
    __version__ = "1.9.0"

    def __init__(self, annotations, raw):
        self.annotations = annotations
        self.raw = raw
        self.annotation_calls = []
        self.raw_calls = []
        self.io = self

    def read_annotations(self, *args, **kwargs):
        self.annotation_calls.append((args, kwargs))
        return self.annotations

    def read_raw_edf(self, *args, **kwargs):
        self.raw_calls.append((args, kwargs))
        return self.raw


class CanonicalAndFrozenTests(unittest.TestCase):
    def test_frozen_hashes_validate(self):
        self.assertEqual(len(validate_frozen_inputs()), 7)

    def test_protocol_and_claims_validate(self):
        protocol, claims, hashes = load_protocol_claims()
        self.assertEqual(protocol["protocol_id"], "COSTA_CLEANROOM_THALAMOCORTICAL_V2_PHASE1")
        self.assertEqual(claims["registry_id"], "COSTA_TC_V2_CLAIMS")
        self.assertEqual(len(hashes), 7)

    def test_canonical_order_and_digest(self):
        self.assertEqual(canonical_bytes({"b": 2, "a": 1}), b'{"a":1,"b":2}\n')
        self.assertEqual(canonical_sha256({"a": 1, "b": 2}), canonical_sha256({"b": 2, "a": 1}))

    def test_canonical_rejects_nan(self):
        with self.assertRaises(ValueError):
            canonical_bytes({"bad": float("nan")})

    def test_write_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "one.json"
            write_canonical(path, {"x": 1})
            with self.assertRaises(FileExistsError):
                write_canonical(path, {"x": 2})

    def test_output_must_be_new_simple_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            child = new_output_child(root, "RUN_A")
            self.assertEqual(child.parent, root.resolve())
            with self.assertRaises(FileExistsError):
                new_output_child(root, "RUN_A")
            with self.assertRaises(ValueError):
                new_output_child(root, "nested/RUN_B")


class PathAnnotationAndFeatureTests(unittest.TestCase):
    def test_only_exact_paths_are_accepted(self):
        self.assertEqual(guarded_relative_path(PSG_RELATIVE), PSG_RELATIVE)
        self.assertEqual(guarded_relative_path(ANNOTATION_RELATIVE), ANNOTATION_RELATIVE)
        for bad in ("SC4001E0-PSG.edf", "data/sleep-edfx-cassette/SC4002E0-PSG.edf", "/" + PSG_RELATIVE, PSG_RELATIVE.replace("/", "\\")):
            with self.assertRaises(ValueError):
                guarded_relative_path(bad)

    def test_stage_3_and_4_are_split_into_n3_epochs(self):
        annotations = FakeAnnotations([0, 60, 120], [60, 30, 30], ["Sleep stage 3", "Sleep stage 4", "Sleep stage 2"])
        rows = build_n3_epoch_metadata(annotations)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["stage"] == "N3" and row["night"] == "Night-1" for row in rows))
        self.assertEqual([row["onset_s"] for row in rows], [0.0, 30.0, 60.0])

    def test_partial_short_annotations_do_not_make_epochs(self):
        annotations = FakeAnnotations([0, 10], [29.999, 30], ["Sleep stage 3", "Sleep stage 2"])
        self.assertEqual(build_n3_epoch_metadata(annotations), [])

    def test_duplicate_epoch_ids_fail_closed(self):
        annotations = FakeAnnotations([0, 0], [30, 30], ["Sleep stage 3", "Sleep stage 4"])
        with self.assertRaises(ValueError):
            build_n3_epoch_metadata(annotations)

    def test_channel_priority_is_exact(self):
        self.assertEqual(choose_channel(["EEG Pz-Oz", "EEG Fpz-Cz"]), "EEG Fpz-Cz")
        self.assertEqual(choose_channel(["EEG Pz-Oz"]), "EEG Pz-Oz")
        with self.assertRaises(ValueError):
            choose_channel(["Fpz-Cz", "EEG Fpz-Cz-Ref"])

    def test_feature_operator_is_dimensionless_and_deterministic(self):
        sfreq = 100.0
        time_s = np.arange(3000) / sfreq
        signal = (
            np.sin(2 * np.pi * 0.75 * time_s)
            + np.sin(2 * np.pi * 2.0 * time_s)
            + np.sin(2 * np.pi * 6.0 * time_s)
            + np.sin(2 * np.pi * 13.0 * time_s)
        )
        first = comparison_features(signal, sfreq)
        second = comparison_features(signal + 3.0 + 0.02 * time_s, sfreq)
        self.assertEqual(list(first), ["SO", "delta", "theta", "spindle"])
        np.testing.assert_allclose(list(first.values()), list(second.values()), atol=1e-10, rtol=1e-10)

    def test_feature_operator_rejects_failure_cases(self):
        with self.assertRaises(ValueError):
            comparison_features(np.zeros(3000), 100.0)
        with self.assertRaises(ValueError):
            comparison_features(np.ones(3000) * np.nan, 100.0)
        with self.assertRaises(ValueError):
            comparison_features(np.ones(100), 100.0)
        with self.assertRaises(ValueError):
            comparison_features(np.ones(3000), 40.0)


class StewardIsolationTests(unittest.TestCase):
    def _run_steward(self, temporary: str, seconds: float = 120.0):
        annotations = FakeAnnotations([0], [120], ["Sleep stage 4"])
        fake_mne = FakeMNE(annotations, FakeRaw(seconds=seconds))
        directory = steward(Path(temporary), Path(temporary) / "outputs", "STEWARD", fake_mne)
        return directory, fake_mne

    def test_exact_mne_call_signatures(self):
        with tempfile.TemporaryDirectory() as temporary:
            _directory, fake_mne = self._run_steward(temporary)
            self.assertEqual(len(fake_mne.annotation_calls), 1)
            self.assertEqual(fake_mne.annotation_calls[0][1], {})
            self.assertEqual(fake_mne.annotation_calls[0][0], (Path(temporary).resolve() / Path(ANNOTATION_RELATIVE),))
            self.assertEqual(len(fake_mne.raw_calls), 1)
            self.assertEqual(fake_mne.raw_calls[0][0], (Path(temporary).resolve() / Path(PSG_RELATIVE),))
            self.assertEqual(fake_mne.raw_calls[0][1], {"preload": True, "verbose": "ERROR"})

    def test_role_payloads_are_separate_and_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, _fake_mne = self._run_steward(temporary)
            fit = read_json(directory / "FIT_PAYLOAD.json")
            heldout = read_json(directory / "HELDOUT_PAYLOAD.json")
            fit_ids = {row["epoch_id"] for row in fit["valid_epochs"] + fit["excluded_epochs"]}
            heldout_ids = {row["epoch_id"] for row in heldout["valid_epochs"] + heldout["excluded_epochs"]}
            self.assertEqual(fit["role"], "FIT")
            self.assertEqual(heldout["role"], "HELDOUT")
            self.assertTrue(fit_ids.isdisjoint(heldout_ids))
            self.assertEqual(len(fit_ids | heldout_ids), 4)
            receipt = read_json(directory / "STEWARD_RECEIPT.json")
            self.assertTrue(receipt["partition_before_missingness"])
            self.assertEqual(sha256_file(directory / "FIT_PAYLOAD.json"), receipt["role_payload_hashes"]["FIT"])

    def test_out_of_range_is_excluded_without_reassignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, _fake_mne = self._run_steward(temporary, seconds=90.0)
            fit = read_json(directory / "FIT_PAYLOAD.json")
            heldout = read_json(directory / "HELDOUT_PAYLOAD.json")
            excluded = fit["excluded_epochs"] + heldout["excluded_epochs"]
            self.assertEqual(len(excluded), 1)
            self.assertEqual(excluded[0]["reason"], "OUT_OF_RANGE")
            self.assertEqual(fit["assigned_epoch_count"] + heldout["assigned_epoch_count"], 4)

    def test_fewer_than_four_eligible_epochs_fails_before_raw_open(self):
        with tempfile.TemporaryDirectory() as temporary:
            fake_mne = FakeMNE(FakeAnnotations([0], [90], ["Sleep stage 3"]), FakeRaw(seconds=90))
            with self.assertRaises(RuntimeError):
                steward(Path(temporary), Path(temporary) / "outputs", "FAIL", fake_mne)
            self.assertEqual(fake_mne.raw_calls, [])

    def test_non_1_9_mne_fails_before_any_open(self):
        with tempfile.TemporaryDirectory() as temporary:
            fake_mne = FakeMNE(FakeAnnotations([0], [120], ["Sleep stage 3"]), FakeRaw())
            fake_mne.__version__ = "1.10.0"
            with self.assertRaisesRegex(RuntimeError, "MNE 1.9"):
                steward(Path(temporary), Path(temporary) / "outputs", "FAIL", fake_mne)
            self.assertEqual(fake_mne.annotation_calls, [])
            self.assertEqual(fake_mne.raw_calls, [])


class ImportedPartitionAndModelInterfaceTests(unittest.TestCase):
    def test_imported_split_is_order_blind_and_two_by_two(self):
        protocol, _claims, _hashes = load_protocol_claims()
        from r4_core import activate_mechanistic_imports
        activate_mechanistic_imports()
        from partitions import build_single_subject_n3_split, n3_split_digest
        rows = [
            {"subject_pseudonym": "S", "epoch_id": f"E{index}", "stage": "N3", "night": "Night-1"}
            for index in range(4)
        ]
        first = build_single_subject_n3_split(rows, protocol["seeds"]["partition_seed"], protocol["partitions"]["fit_fraction"])
        second = build_single_subject_n3_split(list(reversed(rows)), protocol["seeds"]["partition_seed"], protocol["partitions"]["fit_fraction"])
        self.assertEqual(n3_split_digest(first), n3_split_digest(second))
        self.assertEqual(sum(row.role == "FIT" for row in first), 2)
        self.assertEqual(sum(row.role == "HELDOUT" for row in first), 2)

    def test_model_proxy_uses_identical_feature_operator(self):
        protocol, _claims, _hashes = load_protocol_claims()
        bank = r4_empirical._model_feature_bank(protocol)
        self.assertEqual(set(bank), {f"C{index:03d}" for index in range(16)})
        self.assertTrue(all(vector.shape == (4,) and np.all(np.isfinite(vector)) for vector in bank.values()))


def role_payload(role: str, features: list[list[float]], partition_digest: str, frozen_hashes: dict[str, str]) -> dict[str, object]:
    return {
        "schema": "COSTA_R4_ROLE_PAYLOAD_V1",
        "role": role,
        "subject_pseudonym": "SYNTHETIC",
        "night": "Night-1",
        "channel": "EEG Fpz-Cz",
        "feature_order": ["SO", "delta", "theta", "spindle"],
        "feature_units": "dimensionless log10 relative power",
        "total_eligible_epochs": 4,
        "assigned_epoch_count": len(features),
        "valid_epochs": [
            {"epoch_id": f"{role}_{index}", "stable_rank": index, "features": dict(zip(("SO", "delta", "theta", "spindle"), row))}
            for index, row in enumerate(features)
        ],
        "excluded_epochs": [],
        "partition_digest": partition_digest,
        "frozen_input_hashes": frozen_hashes,
    }


class FitHeldoutContractTests(unittest.TestCase):
    def test_fit_selection_reads_fit_and_freezes_calibrations(self):
        _protocol, _claims, hashes = load_protocol_claims()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload_path = root / "FIT.json"
            write_canonical(payload_path, role_payload("FIT", [[1, 2, 4, 8], [1, 2, 4, 8]], "D", hashes))
            bank = {"C000": np.array([1, 2, 3, 4.0]), "C001": np.array([1, 2, 4, 8.0])}
            with patch.object(r4_empirical, "_model_feature_bank", return_value=bank):
                directory = r4_empirical.select_fit(payload_path, root / "outputs", "FIT_RUN")
            freeze = read_json(directory / "SELECTION_FREEZE.json")
            self.assertEqual(freeze["selected_candidate_id"], "C001")
            self.assertEqual(freeze["population_candidate_id"], "C000")
            self.assertEqual(freeze["selection_source_role"], "FIT")

    def test_fit_rejects_heldout_payload(self):
        _protocol, _claims, hashes = load_protocol_claims()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "HELDOUT.json"
            write_canonical(path, role_payload("HELDOUT", [[1, 2, 3, 4], [1, 2, 3, 4]], "D", hashes))
            with self.assertRaises(ValueError):
                r4_empirical.select_fit(path, Path(temporary) / "outputs", "FAIL")

    def test_heldout_verifies_freeze_hash_before_payload_open(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            freeze = root / "freeze.json"
            write_canonical(freeze, {"schema": "COSTA_R4_SELECTION_FREEZE_V1"})
            missing_payload = root / "missing.json"
            with self.assertRaisesRegex(ValueError, "freeze hash mismatch"):
                r4_empirical.evaluate_heldout(missing_payload, freeze, "0" * 64, root / "outputs", "FAIL")

    def test_distinct_heldout_stage_preserves_candidate_and_runs_1000_draws(self):
        _protocol, _claims, hashes = load_protocol_claims()
        bank = {"C000": np.array([1.0, 2.0, 3.0, 4.0]), "C001": np.array([1.0, 2.0, 4.0, 8.0])}
        for index in range(2, 16):
            scale = float(index + 2)
            bank[f"C{index:03d}"] = np.array([1.0, scale, scale**2, scale**3])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fit_path = root / "FIT.json"
            heldout_path = root / "HELDOUT.json"
            write_canonical(fit_path, role_payload("FIT", [[1, 2, 4, 8], [1, 2, 4, 8]], "SPLIT", hashes))
            write_canonical(heldout_path, role_payload("HELDOUT", [[1, 2, 4, 8], [1.01, 2.01, 4.01, 8.01]], "SPLIT", hashes))
            with patch.object(r4_empirical, "_model_feature_bank", return_value=bank):
                fit_directory = r4_empirical.select_fit(fit_path, root / "outputs", "FIT_RUN")
            freeze_path = fit_directory / "SELECTION_FREEZE.json"
            with patch.object(r4_empirical, "_model_feature_bank", return_value=bank):
                heldout_directory = r4_empirical.evaluate_heldout(
                    heldout_path, freeze_path, sha256_file(freeze_path), root / "outputs", "HELDOUT_RUN"
                )
            result = read_json(heldout_directory / "HELDOUT_G6_RESULT.json")
            self.assertEqual(result["selected_candidate_id"], "C001")
            self.assertEqual(result["paired_epoch_bootstrap"]["repetitions"], 1000)
            self.assertTrue(result["checks"]["candidate_unchanged_after_fit_freeze"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
