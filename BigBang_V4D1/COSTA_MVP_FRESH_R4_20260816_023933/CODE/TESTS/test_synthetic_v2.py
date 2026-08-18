"""Deterministic fixture/unit tests for the frozen v2 implementation."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


PREFLIGHT_DIR = Path(__file__).resolve().parent
MECHANISTIC_DIR = PREFLIGHT_DIR.parent / "MECHANISTIC_V2"
sys.path.insert(0, str(MECHANISTIC_DIR))

from model import candidates_from_protocol, simulate
from partitions import build_single_subject_n3_split, n3_split_digest


with (PREFLIGHT_DIR / "protocol_v2.json").open("r", encoding="utf-8") as handle:
    PROTOCOL = json.load(handle)


class CandidateBankTests(unittest.TestCase):
    def test_bank_is_bounded_unique_and_bidirectional(self):
        candidates = candidates_from_protocol(PROTOCOL)
        self.assertEqual(len(candidates), 16)
        self.assertEqual(len({candidate.candidate_id for candidate in candidates}), 16)
        self.assertTrue(all(candidate.cortex_to_thalamus > 0.0 for candidate in candidates))
        self.assertTrue(all(candidate.thalamus_to_cortex > 0.0 for candidate in candidates))

    def test_explicit_four_states_and_repeatability(self):
        candidate = candidates_from_protocol(PROTOCOL)[0]
        samples = 256
        drive_e = np.linspace(0.2, 0.5, samples, dtype=np.float64)
        drive_t = np.linspace(0.3, 0.1, samples, dtype=np.float64)
        first = simulate(candidate, drive_e, drive_t, PROTOCOL["model"], PROTOCOL["observation_mapping"])
        second = simulate(candidate, drive_e, drive_t, PROTOCOL["model"], PROTOCOL["observation_mapping"])
        self.assertEqual(first.states.shape, (samples, 4))
        self.assertTrue(np.array_equal(first.states, second.states))
        self.assertTrue(np.array_equal(first.proxy, second.proxy))
        self.assertTrue(np.all((first.states >= 0.0) & (first.states <= 1.0)))


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.epochs = [
            {"subject_pseudonym": "ONE_SYNTHETIC_SUBJECT", "epoch_id": f"E{index:03d}", "stage": "N3", "night": "Night-1"}
            for index in range(10)
        ]

    def test_single_subject_n3_split_is_order_invariant_and_disjoint(self):
        first = build_single_subject_n3_split(self.epochs, PROTOCOL["seeds"]["partition_seed"], PROTOCOL["partitions"]["fit_fraction"])
        second = build_single_subject_n3_split(list(reversed(self.epochs)), PROTOCOL["seeds"]["partition_seed"], PROTOCOL["partitions"]["fit_fraction"])
        self.assertEqual(n3_split_digest(first), n3_split_digest(second))
        fit = {row.epoch_id for row in first if row.role == "FIT"}
        heldout = {row.epoch_id for row in first if row.role == "HELDOUT"}
        self.assertTrue(fit)
        self.assertTrue(heldout)
        self.assertTrue(fit.isdisjoint(heldout))
        self.assertEqual(fit | heldout, {epoch["epoch_id"] for epoch in self.epochs})

    def test_split_rejects_second_subject_and_non_n3(self):
        two_subjects = list(self.epochs)
        two_subjects[-1] = dict(two_subjects[-1], subject_pseudonym="SECOND")
        with self.assertRaises(ValueError):
            build_single_subject_n3_split(two_subjects, PROTOCOL["seeds"]["partition_seed"], 0.5)
        non_n3 = list(self.epochs)
        non_n3[-1] = dict(non_n3[-1], stage="N2")
        with self.assertRaises(ValueError):
            build_single_subject_n3_split(non_n3, PROTOCOL["seeds"]["partition_seed"], 0.5)


class MappingLimitTests(unittest.TestCase):
    def test_proxy_units_and_limits_are_explicit(self):
        mapping = PROTOCOL["observation_mapping"]
        joined = " ".join(mapping["limits"])
        self.assertIn("arbitrary model unit", mapping["proxy_unit"])
        self.assertIn("not volts or microvolts", joined)
        self.assertIn("scalp EEG causality", joined)


if __name__ == "__main__":
    unittest.main(verbosity=2)
