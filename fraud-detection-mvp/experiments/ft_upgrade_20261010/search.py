"""Deterministic, auditable trial queue, without test-dependent choices."""
from __future__ import annotations

import numpy as np


def defaults():
    return {"features": "original52", "seed": 42, "embedding": "linear",
            "width": 128, "depth": 3, "heads": 4, "dropout": .1,
            "residual_dropout": 0., "ffn_multiplier": 4/3, "batch_size": 512,
            "lr": .0003, "weight_decay": .00001, "loss": "bce", "positive_weight": 5.237777208312512,
            "max_epochs": 120, "min_epochs": 20, "patience": 15, "max_seconds": 2700,
            "sdpa": True, "compression": None, "feature_dropout": 0.,
            "recency_ratio": 1., "positive_fraction": None}


def pilot_configs():
    d = defaults()
    specifications = [
        {"width": 64, "depth": 2, "dropout": .15, "residual_dropout": .1},
        {"positive_weight": 1.},
        {"embedding": "ple", "n_bins": 32},
        {"embedding": "periodic", "frequency_scale": .05},
        {"features": "quantile52", "embedding": "ple", "n_bins": 32},
        {"features": "quantile52", "embedding": "periodic", "frequency_scale": .05},
        {"features": "quantile52", "embedding": "ple", "n_bins": 32, "positive_weight": 1.},
        {"features": "quantile52", "embedding": "ple", "n_bins": 32, "loss": "focal", "focal_gamma": 1.},
        {"features": "rich_pruned", "embedding": "linear", "batch_size": 256},
        {"features": "rich_pruned", "embedding": "ple", "n_bins": 32, "batch_size": 256},
        {"features": "rich_pruned", "embedding": "periodic", "frequency_scale": .05, "batch_size": 256},
        {"features": "rich_pruned", "embedding": "ple", "n_bins": 32, "batch_size": 256,
         "loss": "pairwise", "ranking_weight": .15},
        {"features": "rich_pruned", "embedding": "ple", "n_bins": 32, "batch_size": 256,
         "positive_weight": 1., "positive_fraction": .10},
        {"features": "rich_pruned", "embedding": "ple", "n_bins": 32, "batch_size": 256,
         "recency_ratio": 4.},
        {"features": "rich_all", "embedding": "ple", "n_bins": 32, "batch_size": 128, "compression": .25},
        {"features": "rich_all", "embedding": "periodic", "frequency_scale": .05,
         "batch_size": 128, "compression": .25}]
    return [d | spec for spec in specifications]


def proposal(index, results):
    """Validation-only exploitation plus controlled fresh exploration.

    Every fourth trial retests one top configuration with a new seed. Others
    change a declared representation/architecture/loss/training dimension.
    """
    rng = np.random.RandomState(42+index)
    top = sorted(results, key=lambda r: (r["validation_scores"]["average_precision"],
                                         r["validation_scores"]["recall_top_1pct"]), reverse=True)[:5]
    if top and index % 4 == 0:
        cfg = dict(top[(index//4) % len(top)]["config"])
        cfg["seed"] = 1000+index
        return cfg
    if top and index % 3 != 0:
        cfg = dict(top[int(rng.randint(len(top)))]["config"])
    else:
        cfg = defaults()
        cfg["features"] = str(rng.choice(["quantile52", "rich_pruned", "rich_all"]))
        cfg["embedding"] = str(rng.choice(["ple", "periodic", "linear"]))
    cfg["seed"] = 42
    cfg["width"] = int(rng.choice([96, 128, 192, 256]))
    cfg["depth"] = int(rng.choice([2, 3, 4]))
    cfg["heads"] = 4 if cfg["width"] == 96 else int(rng.choice([4, 8]))
    cfg["dropout"] = float(rng.choice([.05, .1, .2]))
    cfg["ffn_multiplier"] = float(rng.choice([4/3, 2.]))
    cfg["lr"] = float(rng.choice([.0001, .0003, .0006, .001]))
    cfg["weight_decay"] = float(rng.choice([1e-6, 1e-5, 1e-4]))
    cfg["loss"] = str(rng.choice(["bce", "bce", "focal", "pairwise"]))
    cfg["focal_gamma"] = float(rng.choice([1., 2.]))
    cfg["ranking_weight"] = float(rng.choice([.05, .15, .3]))
    cfg["positive_weight"] = float(rng.choice([1., 3., 5.237777208312512]))
    cfg["n_bins"] = int(rng.choice([16, 32, 64]))
    cfg["frequency_scale"] = float(rng.choice([.01, .05, .1]))
    cfg["feature_dropout"] = float(rng.choice([0., 0., .025]))
    cfg["recency_ratio"] = float(rng.choice([1., 1., 4.]))
    cfg["positive_fraction"] = None
    cfg["compression"] = .25 if cfg["features"] == "rich_all" else None
    cfg["batch_size"] = 128 if cfg["features"] == "rich_all" else (256 if cfg["features"] == "rich_pruned" else 512)
    return cfg
