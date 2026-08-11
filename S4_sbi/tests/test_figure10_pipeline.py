"""Deterministic correctness tests for the Figure-10-equivalent pipeline."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import kstest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "S4_sbi" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sleep_sbi.figure10_bank import paired_theta
from sleep_sbi.figure10_diagnostics import _holm_rejections, _randomized_rank
from sleep_sbi.figure10_protocol import (
    PARAMETER_NAMES_7D,
    PARAMETER_NAMES_8D,
    prior_bounds_7d,
    prior_bounds_8d,
    rate_feature_names,
)
from sleep_sbi.figure10_training import EqualMixturePosterior


class _NormalToyPosterior:
    """Small posterior object implementing the DirectPosterior surface used."""

    def __init__(self, mean: float, scale: float):
        self.mean = float(mean)
        self.scale = float(scale)
        self.prior = torch.distributions.Independent(
            torch.distributions.Uniform(torch.zeros(1), torch.ones(1)), 1
        )

    def sample(self, shape, x, **_kwargs):
        del x
        n = int(shape[0])
        return torch.normal(
            mean=torch.full((n, 1), self.mean),
            std=torch.full((n, 1), self.scale),
        )

    def log_prob(self, theta, x):
        del x
        distribution = torch.distributions.Normal(self.mean, self.scale)
        return distribution.log_prob(theta[:, 0])


def test_parameter_and_feature_contracts_are_ordered_and_unique():
    assert PARAMETER_NAMES_8D == (
        "mue",
        "mui",
        "b",
        "tauA",
        "g_LK",
        "g_h",
        "c_th2ctx",
        "c_ctx2th",
    )
    assert PARAMETER_NAMES_7D == PARAMETER_NAMES_8D[:7]
    assert prior_bounds_8d().shape == (8, 2)
    assert np.array_equal(prior_bounds_7d(), prior_bounds_8d()[:7])
    assert np.all(prior_bounds_8d()[:, 1] > prior_bounds_8d()[:, 0])
    assert len(rate_feature_names()) == 14
    assert len(set(rate_feature_names())) == 14


def test_paired_design_changes_only_c_ctx2th():
    theta8, theta7_full = paired_theta(16)
    assert theta8.shape == theta7_full.shape == (16, 8)
    assert np.array_equal(theta8[:, :7], theta7_full[:, :7])
    assert np.unique(theta7_full[:, 7]).size == 1
    assert np.unique(theta8[:, 7]).size > 1


def test_equal_mixture_log_prob_is_logsumexp_not_product_or_average():
    members = [_NormalToyPosterior(0.25, 0.08), _NormalToyPosterior(0.75, 0.08)]
    mixture = EqualMixturePosterior(members)
    theta = np.asarray([[0.25], [0.50], [0.75]], float)
    observed = np.zeros(1)
    actual = mixture.log_prob(theta, observed)
    member_logs = np.stack(
        [
            member.log_prob(
                torch.tensor(theta, dtype=torch.float32),
                torch.zeros(1),
            )
            .detach()
            .numpy()
            for member in members
        ]
    )
    expected = np.logaddexp(member_logs[0], member_logs[1]) - math.log(2)
    assert np.allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert not np.allclose(actual, member_logs.sum(axis=0))


def test_randomized_sbc_rank_detects_bias_and_underdispersion():
    rng = np.random.default_rng(1234)
    truth = rng.uniform(size=(2000, 1))
    calibrated_samples = rng.uniform(size=(2000, 1000, 1))
    calibrated_ranks = np.asarray(
        [
            _randomized_rank(calibrated_samples[i], truth[i], rng)[0]
            for i in range(len(truth))
        ]
    )
    calibrated = (calibrated_ranks + 0.5) / 1001
    assert kstest(calibrated, "uniform").pvalue > 0.01
    biased_samples = np.clip(
        truth[:, None, :] + 0.20 + rng.normal(0, 0.02, (2000, 1000, 1)),
        0,
        1,
    )
    biased_ranks = np.asarray(
        [
            _randomized_rank(biased_samples[i], truth[i], rng)[0]
            for i in range(len(truth))
        ]
    )
    biased = (biased_ranks + 0.5) / 1001
    assert kstest(biased, "uniform").pvalue < 1e-20


def test_holm_correction_is_familywise_and_monotone():
    rejected = _holm_rejections(np.asarray([0.001, 0.010, 0.2, 0.9]), 0.05)
    assert rejected.tolist() == [True, True, False, False]


def test_lc2st_accepts_calibrated_toy_and_rejects_distortion():
    from sbi.diagnostics.lc2st import LC2ST

    rng = np.random.default_rng(42)
    n = 2500
    theta = rng.normal(size=(n, 1))
    x = theta + rng.normal(size=(n, 1))
    exact = x / 2 + rng.normal(scale=np.sqrt(0.5), size=(n, 1))
    distorted = x / 2 + 1.0 + rng.normal(scale=0.2, size=(n, 1))
    decisions = {}
    for label, posterior_samples in (
        ("exact", exact),
        ("distorted", distorted),
    ):
        test = LC2ST(
            thetas=torch.tensor(theta, dtype=torch.float32),
            xs=torch.tensor(x, dtype=torch.float32),
            posterior_samples=torch.tensor(
                posterior_samples, dtype=torch.float32
            ),
            seed=42,
            classifier="random_forest",
            classifier_kwargs={
                "n_estimators": 100,
                "max_depth": 8,
                "n_jobs": 1,
            },
            num_trials_null=30,
            permutation=True,
        )
        test.train_under_null_hypothesis(verbosity=0)
        test.train_on_observed_data(seed=43, verbosity=0)
        x_o = torch.tensor([[0.0]], dtype=torch.float32)
        local = (
            rng.normal(0, np.sqrt(0.5), size=(5000, 1))
            if label == "exact"
            else rng.normal(1, 0.2, size=(5000, 1))
        )
        theta_o = torch.tensor(local, dtype=torch.float32)
        decisions[label] = {
            "p": test.p_value(theta_o, x_o),
            "reject": test.reject_test(theta_o, x_o, alpha=0.05),
        }
    assert not decisions["exact"]["reject"]
    assert decisions["exact"]["p"] >= 0.05
    assert decisions["distorted"]["reject"]
    assert decisions["distorted"]["p"] < 0.05
