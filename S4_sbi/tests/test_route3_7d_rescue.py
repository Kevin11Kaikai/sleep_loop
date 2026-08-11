from __future__ import annotations

import numpy as np

from sleep_sbi.route3_7d_rescue import coverage_table, verify_rescue_preregistration
from sleep_sbi.route3_7d_rescue_training import (
    apply_rank_calibrator,
    fit_rank_calibrator,
)


def test_rescue_preregistration_lock_is_valid() -> None:
    assert len(verify_rescue_preregistration()) == 64


def test_coverage_uses_central_intervals() -> None:
    theta = np.full((4, 7), 0.5)
    samples = np.linspace(0, 1, 1001)[None, :, None]
    samples = np.broadcast_to(samples, (4, 1001, 7)).copy()
    table = coverage_table(theta, samples, "test", (0.8, 0.9))
    assert table.covered.eq(4).all()
    assert table.n.eq(4).all()


def test_rank_calibrator_is_finite_bounded_and_monotone() -> None:
    rng = np.random.default_rng(1234)
    theta = rng.uniform(size=(64, 7))
    raw = np.clip(
        theta[:, None, :] + rng.normal(0, 0.08, size=(64, 512, 7)),
        0,
        1,
    )
    calibrator = fit_rank_calibrator(theta, raw)
    calibrated = apply_rank_calibrator(raw, calibrator)
    assert calibrated.shape == raw.shape
    assert np.isfinite(calibrated).all()
    assert ((calibrated >= 0) & (calibrated <= 1)).all()
    for case in (0, 31, 63):
        for parameter in range(7):
            order = np.argsort(raw[case, :, parameter], kind="mergesort")
            assert np.all(np.diff(calibrated[case, order, parameter]) >= 0)


def test_rank_calibrator_does_not_invert_sample_copula_ranks() -> None:
    rng = np.random.default_rng(91)
    theta = rng.uniform(size=(32, 7))
    raw = rng.uniform(size=(32, 256, 7))
    calibrated = apply_rank_calibrator(raw, fit_rank_calibrator(theta, raw))
    for case in range(4):
        for parameter in range(7):
            order = np.argsort(raw[case, :, parameter], kind="mergesort")
            assert np.all(np.diff(calibrated[case, order, parameter]) >= 0)
