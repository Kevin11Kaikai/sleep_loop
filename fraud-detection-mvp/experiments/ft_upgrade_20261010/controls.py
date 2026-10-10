"""Matched, validation-selected stronger LightGBM controls."""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .data import load_data
from .utils import HERE, digest, fingerprints, now, read, scores, write


def frame(arrays, meta):
    out = pd.DataFrame(arrays["x_lgb"], columns=meta["numeric_output_names"])
    for j, (name, n) in enumerate(zip(meta["categorical"], meta["cat_cardinalities"])):
        out[name] = pd.Categorical(arrays["x_cat"][:, j], categories=range(n))
    return out


def fit_controls():
    records = []
    for feature_set in ["original52", "quantile52", "rich_pruned", "rich_all"]:
        arrays, meta = load_data(feature_set)
        train, valid = arrays["train"], arrays["validation"]
        x_train, x_valid = frame(train, meta), frame(valid, meta)
        positive_weight = float(np.sqrt((len(train["y"])-train["y"].sum())/train["y"].sum()))
        for variant, leaves, depth, min_child, weight in [
                ("moderate", 31, 8, 100, positive_weight),
                ("stronger", 63, 12, 50, positive_weight),
                ("unweighted", 63, 12, 50, 1.)]:
            trial_id = f"lgb_{feature_set}_{variant}"
            path = HERE / "runs" / trial_id
            path.mkdir(parents=True, exist_ok=True)
            if (path / "result.json").is_file():
                record = read(path / "result.json")
                if digest(path / "model.txt") != record["checkpoint_sha256"]:
                    raise RuntimeError("Tree control fingerprint mismatch")
                records.append(record)
                continue
            cfg = dict(objective="binary", n_estimators=4000, learning_rate=.04,
                       num_leaves=leaves, max_depth=depth, min_child_samples=min_child,
                       subsample=.8, subsample_freq=1, colsample_bytree=.9, reg_lambda=5.,
                       scale_pos_weight=weight, random_state=42, n_jobs=8,
                       deterministic=True, force_col_wise=True, verbosity=-1, metric="None")
            model = lgb.LGBMClassifier(**cfg)
            def ap(y, p):
                return "average_precision", float(average_precision_score(y, p)), True
            history = {}
            start = time.perf_counter()
            print("TREE START", trial_id, flush=True)
            model.fit(x_train, train["y"], eval_set=[(x_valid, valid["y"])], eval_metric=ap,
                      callbacks=[lgb.early_stopping(150, first_metric_only=True, verbose=False),
                                 lgb.record_evaluation(history)])
            training_seconds = time.perf_counter()-start
            p = model.predict_proba(x_valid)[:, 1]
            model.booster_.save_model(str(path / "model.txt"))
            np.savez_compressed(path / "predictions_validation.npz", p=p, ids=valid["ids"], y=valid["y"])
            pd.DataFrame({"iteration": np.arange(1, len(history["valid_0"]["average_precision"])+1),
                          "validation_ap": history["valid_0"]["average_precision"]}).to_csv(path / "history.csv", index=False)
            record = {"status": "COMPLETED", "trial_id": trial_id, "model": "LightGBM control",
                      "features": feature_set, "config": cfg, "validation_scores": scores(valid["y"], p, valid["ids"]),
                      "training_seconds": training_seconds, "device": "CPU, 8 threads",
                      "best_iteration": model.best_iteration_, "finished_utc": now(),
                      "checkpoint_sha256": digest(path / "model.txt"), "code_fingerprints": fingerprints(),
                      "representation_sha256": digest(HERE / f"cache/{feature_set}/manifest.json")}
            write(path / "result.json", record)
            records.append(record)
            print("TREE DONE", trial_id, record["validation_scores"], flush=True)
        del x_train, x_valid, arrays
    write(HERE / "tree_controls.json", records)
    return records
