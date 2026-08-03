"""Automatic tests for Notebook 34 calibration root-cause audit."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "S4_sbi" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sleep_sbi.figure10_diagnostics import _randomized_rank
from sleep_sbi.figure10_notebook34_audit import (
    N34_ROOT,
    PHASE2_N_DRAWS,
    apply_strategy,
    equal_tailed_central_coverage,
    load_matched_case_ids,
    normalize_rank_legacy,
    parameter_names,
    randomized_rank_protocol,
    ranks_npz_path,
)
from sleep_sbi.figure10_protocol import PARAMETER_NAMES_8D, rate_feature_names


def test_parameter_order_contract():
    assert parameter_names("8d") == PARAMETER_NAMES_8D
    assert parameter_names("7d") == PARAMETER_NAMES_8D[:7]
    assert len(rate_feature_names()) == 14


def test_rank_matches_legacy_and_is_not_midpoint():
    samples = np.array([[0.1], [0.5], [0.5], [0.9]])
    truth = np.array([0.5])
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    legacy = _randomized_rank(samples, truth, rng_a)
    protocol = randomized_rank_protocol(samples, truth, rng_b)
    assert np.allclose(legacy, protocol["rank_randomized"])
    # With ties, randomized ranks should vary across seeds (not fixed midpoint)
    ranks = [
        float(
            randomized_rank_protocol(
                samples, truth, np.random.default_rng(s)
            )["rank_randomized"][0]
        )
        for s in range(20)
    ]
    assert len(set(np.round(ranks, 6))) > 1


def test_normalize_and_central_coverage_definitions():
    ranks = normalize_rank_legacy(np.array([0.0, 500.0, 1000.0]), 1000)
    assert np.allclose(ranks, np.array([0.5, 500.5, 1000.5]) / 1001.0)
    # Exact Uniform(0,1) sample → equal-tailed 90% coverage ≈ 0.90
    u = np.linspace(0.0, 1.0, 1001)
    cov = equal_tailed_central_coverage(u, 0.90)
    assert 0.88 <= cov <= 0.92


def test_support_leakage_and_boundary_point_mass():
    raw = np.array([[-0.1, 0.5], [0.2, 1.2], [0.0, 1.0], [0.4, 0.6]])
    clipped = apply_strategy(raw, "clipped")
    rejection = apply_strategy(raw, "rejection_in_support")
    assert clipped["lower_leak_frac"][0] > 0
    assert clipped["upper_leak_frac"][1] > 0
    assert rejection["accept_rate"] < 1.0
    assert np.all((clipped["samples"] >= 0) & (clipped["samples"] <= 1))


def test_draw_budget_arithmetic():
    case_ids = load_matched_case_ids()
    member = 2 * len(case_ids) * 5 * PHASE2_N_DRAWS
    ens = 2 * len(case_ids) * PHASE2_N_DRAWS
    assert member + ens == 294_912


def test_protocol_lock_and_decision_gate_schema_if_present():
    lock_path = N34_ROOT / "json" / "protocol_lock.json"
    gate_path = N34_ROOT / "json" / "decision_gate.json"
    if not lock_path.is_file():
        return
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for key in (
        "parameter_order",
        "conditioning_summary",
        "rank_definition",
        "decision_priority",
        "sampling",
        "clipping_audit",
        "software_provenance",
    ):
        assert key in lock
    assert lock["sampling"]["budget_arithmetic"]["total_draws"] == 294_912
    if gate_path.is_file():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        required = {
            "primary_action",
            "secondary_supported_mechanisms",
            "unresolved_mechanisms",
            "evidence_strength",
            "failed_integrity_tests",
            "scale_up_status",
            "sealed_bank_required_for_final_confirmation",
        }
        assert required <= set(gate)
        assert gate["primary_action"] in {
            "A_IMPLEMENTATION_FIX_FIRST",
            "B_SUMMARY_IDENTIFIABILITY_WORK_FIRST",
            "C_ESTIMATOR_CALIBRATION_EXPERIMENT_FIRST",
            "D_CONTROLLED_131K_PILOT_JUSTIFIED",
            "E_INCONCLUSIVE_MINIMAL_EXPERIMENT_REQUIRED",
        }
        assert gate["sealed_bank_required_for_final_confirmation"] is True


def test_oof_only_sensitivity_guard_if_present():
    path = N34_ROOT / "csv" / "phase3_sensitivity_oof.csv"
    if not path.is_file():
        return
    df = pd.read_csv(path)
    assert "oof_r2_extratrees" in df.columns
    assert (df["metrics_source"] == "OOF_only").all()


def test_ranks_npz_schema_readable():
    path = ranks_npz_path("8d", 32768)
    assert path.is_file()
    z = np.load(path)
    assert z["marginal_ranks"].shape[0] == 1024
    assert list(z["parameter_names"]) == list(PARAMETER_NAMES_8D)


def test_prohibited_language_in_notebook_and_artifacts():
    notebook = (
        PROJECT_ROOT
        / "S4_sbi"
        / "notebooks"
        / "34_posterior_calibration_root_cause_audit.ipynb"
    )
    texts = []
    if notebook.is_file():
        texts.append(notebook.read_text(encoding="utf-8"))
    # Scan narrative outputs; skip protocol_lock prohibited_claims inventory itself.
    scan_names = {
        "decision_gate.json",
        "phase1_summary.json",
        "phase2_clip_audit_report.json",
        "phase3_summary.json",
        "phase4_summary.json",
        "run_all_summary.json",
        "attribution_matrix.csv",
        "final_action_go_nogo.csv",
    }
    for path in N34_ROOT.rglob("*"):
        if path.name in scan_names and path.is_file():
            try:
                texts.append(path.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                continue
    blob = "\n".join(texts).lower()
    # Affirmative prohibited claims only (required framing uses explicit negations).
    banned_patterns = [
        r"is untouched confirmatory",
        r"classic overconfidence",
        r"proves structural non-identifiability",
        r"proof of structural non-identifiability",
        r"is general joint coverage",
        r"trusted posterior",
        r"131k.*will (solve|fix|resolve)",
        r"1m.*will (solve|fix|resolve)",
        r"ranks give(?:s)? (?:hpd|hdi)",
        r"pit ecdf is credible-interval coverage",
    ]
    for pattern in banned_patterns:
        assert re.search(pattern, blob) is None, pattern
    assert re.search(r"not\*{0,2}\s+untouched confirmatory", blob)


def test_cholesky_gaussian_moment_smoke():
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(512, 3))
    cov = np.cov(samples.T, ddof=1) + 1e-3 * np.eye(3)
    L = np.linalg.cholesky(cov)
    z = np.linalg.solve(L, np.zeros(3) - samples.mean(axis=0))
    assert z.shape == (3,)
    assert np.isfinite(z).all()
