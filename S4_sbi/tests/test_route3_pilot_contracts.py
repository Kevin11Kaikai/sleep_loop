"""Fast contract checks for the Route-3 exploratory pilot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sleep_sbi.route3_global_robustness import (
    deterministic_seed_schedule,
    rate_feature_names,
    sobol_theta,
)
from sleep_sbi.route3_pilot_snpe import MODEL_SEEDS, load_training_data, write_go_criteria


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "S4_sbi" / "results" / "route3_pilot_snpe"


def test_seed_schedule_and_sobol_design_are_reproducible() -> None:
    first = deterministic_seed_schedule(2048, 710001)
    second = deterministic_seed_schedule(2048, 710001)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 2048
    assert np.array_equal(sobol_theta(128, 20260802), sobol_theta(128, 20260802))


def test_bank_split_and_scaling_contract() -> None:
    bank_root = RESULTS / "simulation_bank_2048"
    data = load_training_data(
        bank_root / "route3_cortex_rate_14d_bank_2048.npz",
        bank_root / "split_and_scaling.npz",
    )
    assert data.theta_train.shape == (1638, 8)
    assert data.x_train.shape == (1638, 14)
    assert data.theta_validation.shape == (410, 8)
    assert data.x_validation.shape == (410, 14)
    assert np.isfinite(data.x_train).all()
    assert len(rate_feature_names()) == 14


def test_go_criteria_are_immutable_on_reload(tmp_path: Path) -> None:
    path = tmp_path / "criteria.json"
    first = write_go_criteria(path)
    original_text = path.read_text(encoding="utf-8")
    second = write_go_criteria(path)
    assert first == second
    assert path.read_text(encoding="utf-8") == original_text
    assert first["immutable_after_heldout_open"]


def test_final_decision_and_posterior_artifact() -> None:
    decision = json.loads(
        (RESULTS / "heldout_validation" / "formal_route3_decision.json").read_text(
            encoding="utf-8"
        )
    )
    assert decision["decision"] == "NO-GO"
    assert not decision["real_eeg_inference_unlocked"]
    posterior_path = (
        RESULTS
        / "heldout_validation"
        / "posterior_samples"
        / "heldout_ensemble_posterior_samples.npz"
    )
    with np.load(posterior_path, allow_pickle=False) as data:
        assert data["samples"].shape == (128, 4096, 8)
        assert np.isfinite(data["samples"]).all()
        assert not [name for name in data.files if data[name].dtype == object]
        assert list(data["member_seeds"]) == list(MODEL_SEEDS)
