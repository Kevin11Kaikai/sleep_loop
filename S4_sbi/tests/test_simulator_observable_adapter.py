"""Fast contract tests for the cortical-rate proxy adapter.

These tests deliberately do not run neurolib simulations.  The notebook performs
the integration run; this file protects ordering and undefined-value handling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "S4_sbi" / "src"))

from sleep_sbi.simulator_observable_adapter import (  # noqa: E402
    BASELINE14_FEATURES,
    PAC_FEATURES,
    SCHEMA_FEATURES,
    SO_MORPHOLOGY_FEATURES,
    SPECTRAL_SO_PROXY_FEATURES,
    SPINDLE_FEATURES,
    _to_float_parameters,
)


def test_schema_lengths_and_baseline_order_are_stable() -> None:
    assert len(SPECTRAL_SO_PROXY_FEATURES) == 4
    assert len(SO_MORPHOLOGY_FEATURES) == 3
    assert len(SPINDLE_FEATURES) == 2
    assert len(PAC_FEATURES) == 5
    assert len(BASELINE14_FEATURES) == 14
    assert SCHEMA_FEATURES["baseline14"] == BASELINE14_FEATURES
    assert SCHEMA_FEATURES["minimal_spectral_so_proxy4"] == BASELINE14_FEATURES[:4]


def test_parameter_normalization_accepts_legacy_lowercase_g_lk() -> None:
    payload = {
        "mue": 3.0,
        "mui": 3.1,
        "b": 30.0,
        "tauA": 1200.0,
        "g_lk": 0.05,
        "g_h": 0.06,
        "c_th2ctx": 0.02,
        "c_ctx2th": 0.10,
    }
    normalized = _to_float_parameters(payload)
    assert normalized["g_LK"] == 0.05
    assert np.isfinite(list(normalized.values())).all()


def test_missing_parameters_are_not_silently_defaulted() -> None:
    payload = {"mue": 3.0}
    try:
        _to_float_parameters(payload)
    except KeyError as error:
        assert "missing required model parameter" in str(error)
    else:
        raise AssertionError("missing model parameters must fail explicitly")
