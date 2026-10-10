"""Fit all follow-up representations on training covariates only."""
from __future__ import annotations

import argparse
import gc
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from .utils import HERE, PROJECT, digest, now, read, write

CATS = ["ProductCD", *[f"card{i}" for i in range(1, 7)], "addr1", "addr2",
        "P_emaildomain", "R_emaildomain", *[f"M{i}" for i in range(1, 10)]]
BASE_NUM = ["TransactionAmt", "dist1", "dist2", *[f"C{i}" for i in range(1, 15)],
            *[f"D{i}" for i in range(1, 16)]]
SOURCE = PROJECT / "data/raw/train_transaction.csv"


def add_engineering(frame, frequencies=None):
    """Non-target transforms; maps are frozen after fitting on train only."""
    frame = frame.copy()
    for name, columns in {"card1_addr1": ["card1", "addr1"],
                          "card1_card2_addr1": ["card1", "card2", "addr1"]}.items():
        frame[name] = frame[columns].astype("string").fillna("<MISSING>").agg("|".join, axis=1)
    extra_cats = ["card1_addr1", "card1_card2_addr1"]
    frequency_fields = ["card1", "card2", "addr1", "P_emaildomain", "R_emaildomain", *extra_cats]
    if frequencies is None:
        frequencies = {name: frame[name].astype("string").value_counts().to_dict()
                       for name in frequency_fields}
    extra_num = []
    for name in frequency_fields:
        target = f"{name}__log_train_count"
        frame[target] = np.log1p(frame[name].astype("string").map(frequencies[name]).fillna(0).astype(float))
        extra_num.append(target)
    amt = frame["TransactionAmt"].to_numpy(dtype=float)
    frame["amount_fraction"] = (np.rint(amt * 100) % 100) / 100
    extra_num.append("amount_fraction")
    dt = frame["TransactionDT"].to_numpy(dtype=float)
    for period, label in [(86400, "daily"), (7 * 86400, "weekly")]:
        for fun, tag in [(np.sin, "sin"), (np.cos, "cos")]:
            name = f"relative_{label}_{tag}"
            frame[name] = fun(2 * np.pi * dt / period)
            extra_num.append(name)
    return frame, frequencies, extra_num, extra_cats


def choose_v(train, threshold=.98, max_features=128):
    """Remove redundant V fields using only train covariates, no label access."""
    candidates = [f"V{i}" for i in range(1, 340)
                  if train[f"V{i}"].nunique(dropna=True) > 1]
    candidates.sort(key=lambda name: (float(train[name].isna().mean()), int(name[1:])))
    values = train[candidates].to_numpy(dtype=np.float32)
    medians = np.nanmedian(values, axis=0)
    values = np.where(np.isnan(values), medians, values)
    correlations = np.nan_to_num(np.corrcoef(values, rowvar=False), nan=0)
    chosen = []
    for j in range(len(candidates)):
        if all(abs(correlations[j, i]) < threshold for i in chosen):
            chosen.append(j)
        if len(chosen) >= max_features:
            break
    return [candidates[j] for j in chosen]


def fit_representation(train, name, selected_v=None):
    cat_names, numeric_names = list(CATS), list(BASE_NUM)
    state = {"name": name, "fit_rows": len(train), "fit_partition": "training only",
             "cat_max_values": 2048 if name == "quantile52" else 50000}
    if name.startswith("rich_"):
        train, frequencies, extra_num, extra_cats = add_engineering(train)
        v_names = selected_v if name == "rich_pruned" else [f"V{i}" for i in range(1, 340)]
        numeric_names += v_names + extra_num
        cat_names += extra_cats
        state["frequencies"] = frequencies
        state["v_selection"] = {"rule": "train Pearson abs<.98; lower missingness first; max128"
                                if name == "rich_pruned" else "all nonconstant V fields",
                                "features": v_names}
    numeric_names = [n for n in numeric_names if train[n].nunique(dropna=True) > 1]
    cat_names = [n for n in cat_names if train[n].nunique(dropna=True) > 1]
    values = train[numeric_names].to_numpy(dtype=np.float32)
    values[~np.isfinite(values)] = np.nan
    missing = np.isnan(values)
    medians = np.nanmedian(values, axis=0)
    imputed = np.where(missing, medians, values).astype(np.float32)
    # One flag per distinct training missingness pattern avoids hundreds of
    # duplicated V missingness tokens. The same chosen columns are used later.
    indicator_indices, seen = [], set()
    for j in range(values.shape[1]):
        rate = missing[:, j].mean()
        if .01 <= rate <= .99:
            pattern = np.packbits(missing[:, j]).tobytes()
            if pattern not in seen:
                indicator_indices.append(j)
                seen.add(pattern)
    quantile = QuantileTransformer(n_quantiles=1024, output_distribution="normal",
                                  subsample=min(100000, len(train)), random_state=42,
                                  copy=True)
    quantile.fit(imputed)
    vocabs = {}
    for n in cat_names:
        counts = train[n].astype("string").value_counts(dropna=True)
        eligible = [(str(v), int(c)) for v, c in counts.items() if c >= 2]
        eligible.sort(key=lambda item: (-item[1], item[0]))
        vocabs[n] = {v: j + 2 for j, (v, _) in enumerate(eligible[:state["cat_max_values"]])}
    state.update(numeric=numeric_names, categorical=cat_names, medians=medians,
                 quantile=quantile, indicator_indices=indicator_indices, vocabularies=vocabs,
                 numeric_output_names=numeric_names + [f"{numeric_names[j]}__missing" for j in indicator_indices],
                 cat_cardinalities=[len(vocabs[n])+2 for n in cat_names])
    return state


def transform(frame, state):
    if state["name"].startswith("rich_"):
        frame, _, _, _ = add_engineering(frame, state["frequencies"])
    values = frame[state["numeric"]].to_numpy(dtype=np.float32)
    values[~np.isfinite(values)] = np.nan
    missing = np.isnan(values)
    imputed = np.where(missing, state["medians"], values).astype(np.float32)
    indicators = missing[:, state["indicator_indices"]].astype(np.float32)
    x_num = np.column_stack([state["quantile"].transform(imputed), indicators]).astype(np.float32)
    x_lgb = np.column_stack([imputed, indicators]).astype(np.float32)
    codes = []
    for name in state["categorical"]:
        series = frame[name].astype("string")
        code = series.map(state["vocabularies"][name]).fillna(0).to_numpy(dtype=np.int64)
        code[series.isna().to_numpy()] = 1
        codes.append(code)
    x_cat = np.column_stack(codes)
    if not np.isfinite(x_num).all() or not np.isfinite(x_lgb).all():
        raise ValueError("Nonfinite representation")
    return {"x_num": x_num, "x_lgb": x_lgb, "x_cat": x_cat,
            "ids": frame["TransactionID"].to_numpy(dtype=np.int64),
            "dt": frame["TransactionDT"].to_numpy(dtype=np.int64)}


def load_data(name):
    cache = HERE / "cache" / name
    meta = read(cache / "manifest.json")
    arrays = {}
    for split in ["train", "validation"]:
        with np.load(cache / f"{split}.npz", allow_pickle=False) as file:
            arrays[split] = {key: file[key] for key in file.files}
    return arrays, meta


def prepare():
    master = HERE / "cache/preparation.json"
    if master.is_file():
        info = read(master)
        for name in ["original52", "quantile52", "rich_pruned", "rich_all"]:
            meta = read(HERE / f"cache/{name}/manifest.json")
            for filename, expected in meta["files_sha256"].items():
                if digest(HERE / f"cache/{name}" / filename) != expected:
                    raise RuntimeError("Prepared representation fingerprint mismatch")
        return info
    original_manifest = read(PROJECT / "results/prepared/full/manifest.json")
    if digest(SOURCE) != original_manifest["source_sha256"]:
        raise RuntimeError("Dataset fingerprint mismatch")
    # Verify file order using IDs/timestamps without reading the test target.
    meta = pd.read_csv(SOURCE, usecols=["TransactionID", "TransactionDT"])
    if not meta["TransactionDT"].is_monotonic_increasing:
        raise RuntimeError("This cache builder requires the verified chronological CSV order")
    n_train, n_end = int(len(meta)*.70), int(len(meta)*.85)
    header = pd.read_csv(SOURCE, nrows=0).columns
    dtypes = {c: "string" if c in CATS else "float32" for c in header
              if c not in {"TransactionID", "TransactionDT", "isFraud"}}
    frame = pd.read_csv(SOURCE, nrows=n_end, dtype=dtypes)
    y = frame.pop("isFraud").to_numpy(np.int8)
    train, valid = frame.iloc[:n_train], frame.iloc[n_train:]
    selected_v = choose_v(train)
    summary = {"created_utc": now(), "data_sha256": original_manifest["source_sha256"],
               "train_rows": n_train, "validation_rows": n_end-n_train,
               "original_test_target_not_read": True, "v_train_selected": selected_v,
               "representations": {}}
    for name in ["original52", "quantile52", "rich_pruned", "rich_all"]:
        cache = HERE / "cache" / name
        cache.mkdir(parents=True, exist_ok=True)
        print("PREPARING", name, flush=True)
        if name == "original52":
            old_state = read(PROJECT / "results/prepared/full/preprocessor.json")
            state = {"name": name, "numeric_output_names": old_state["numeric_output_names"],
                     "categorical": old_state["categorical"], "cat_cardinalities": old_state["cat_cardinalities"],
                     "fit_rows": n_train}
            for split in ["train", "validation"]:
                with np.load(PROJECT / f"results/prepared/full/{split}.npz", allow_pickle=False) as file:
                    arrays = {"x_num": file["x_num"], "x_lgb": file["x_lgb_num"],
                              "x_cat": file["x_cat"], "ids": file["transaction_id"],
                              "dt": file["transaction_dt"], "y": file["y"]}
                np.savez_compressed(cache / f"{split}.npz", **arrays)
        else:
            state = fit_representation(train, name, selected_v)
            joblib.dump(state, cache / "preprocessor.joblib")
            for split, part, labels in [("train", train, y[:n_train]), ("validation", valid, y[n_train:])]:
                arrays = transform(part, state)
                arrays["y"] = labels
                np.savez_compressed(cache / f"{split}.npz", **arrays)
        info = {"name": name, "created_utc": now(), "source_sha256": summary["data_sha256"],
                "fit_rows": n_train, "train_rows": n_train, "validation_rows": n_end-n_train,
                "numeric_output_names": state["numeric_output_names"],
                "categorical": state["categorical"], "cat_cardinalities": state["cat_cardinalities"],
                "files_sha256": {p.name: digest(p) for p in cache.iterdir() if p.is_file() and p.name != "manifest.json"}}
        write(cache / "manifest.json", info)
        summary["representations"][name] = {"numeric_tokens": len(info["numeric_output_names"]),
                                            "categorical_tokens": len(info["categorical"]),
                                            "manifest_sha256": digest(cache / "manifest.json")}
        del state, arrays
        gc.collect()
    write(master, summary)
    write(HERE / "feature_preparation_summary.json", summary)
    return summary


def test_features(name):
    """Only called after selection is locked; no target read here."""
    if name == "original52":
        with np.load(PROJECT / "results/prepared/full/test_features.npz", allow_pickle=False) as f:
            return {"x_num": f["x_num"], "x_lgb": f["x_lgb_num"], "x_cat": f["x_cat"],
                    "ids": f["transaction_id"], "dt": f["transaction_dt"]}
    state = joblib.load(HERE / f"cache/{name}/preprocessor.joblib")
    header = list(pd.read_csv(SOURCE, nrows=0).columns)
    columns = [c for c in header if c != "isFraud"]
    dtypes = {c: "string" if c in CATS else "float32" for c in columns
              if c not in {"TransactionID", "TransactionDT"}}
    frame = pd.read_csv(SOURCE, usecols=columns, dtype=dtypes)
    end = int(len(frame)*.85)
    return transform(frame.iloc[end:], state)


if __name__ == "__main__":
    prepare()
