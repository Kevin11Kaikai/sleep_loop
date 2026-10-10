"""Freeze validation-only choices, then evaluate the previously viewed test once."""
from __future__ import annotations

import gc
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from .controls import frame
from .data import test_features
from .train import build, cuda, predict
from .utils import HERE, PROJECT, digest, fingerprints, now, read, scores, verify_baseline, write


def freeze_selection(neural_records, tree_records):
    # Ignore smoke: callers only provide finished campaign trials.
    ranked = sorted(neural_records, key=lambda r: (r["validation_scores"]["average_precision"],
                                                  r["validation_scores"]["recall_top_1pct"]), reverse=True)
    if not ranked:
        raise RuntimeError("No neural model available to select")
    predictions, ids, y = {}, None, None
    for record in ranked[:8]:
        with np.load(HERE / "runs" / record["trial_id"] / "predictions_validation.npz", allow_pickle=False) as f:
            if ids is None:
                ids, y = f["ids"], f["y"]
            if not np.array_equal(ids, f["ids"]) or not np.array_equal(y, f["y"]):
                raise RuntimeError("Validation ensemble row alignment mismatch")
            predictions[record["trial_id"]] = f["p"]
    candidates = [[ranked[0]["trial_id"]]]
    for n in [3, 5]:
        if len(predictions) >= n:
            candidates.append([r["trial_id"] for r in ranked[:n]])
    options = []
    for members in candidates:
        p = np.mean([predictions[m] for m in members], axis=0)
        options.append({"members": members, "validation_scores": scores(y, p, ids),
                        "rule": "equal probability average; neural models only"})
    chosen = max(options, key=lambda x: (x["validation_scores"]["average_precision"],
                                        x["validation_scores"]["recall_top_1pct"]))
    tree = max(tree_records, key=lambda x: (x["validation_scores"]["average_precision"],
                                          x["validation_scores"]["recall_top_1pct"]))
    selection = {"status": "LOCKED_BEFORE_TEST_ACCESS", "locked_utc": now(),
                 "neural": chosen, "best_single_neural": ranked[0]["trial_id"],
                 "tree": tree["trial_id"], "validation_candidates": options,
                 "neural_checkpoint_sha256": {r["trial_id"]: r["checkpoint_sha256"] for r in ranked
                                               if r["trial_id"] in chosen["members"]+[ranked[0]["trial_id"]]},
                 "tree_checkpoint_sha256": tree["checkpoint_sha256"],
                 "code_fingerprints": fingerprints(),
                 "test_status": "previously viewed original test; exploratory follow-up"}
    write(HERE / "selection_lock.json", selection)
    return selection


def finalize(neural_records, tree_records, state):
    target = HERE / "final"
    if (target / "completion.json").is_file():
        return read(target / "completion.json")
    target.mkdir(parents=True, exist_ok=True)
    verify_baseline()
    lock = read(HERE / "selection_lock.json") if (HERE / "selection_lock.json").exists() else freeze_selection(neural_records, tree_records)
    all_records = {r["trial_id"]: r for r in neural_records+tree_records}
    neural_members = list(dict.fromkeys(lock["neural"]["members"]+[lock["best_single_neural"]]))
    for trial_id in neural_members:
        if digest(HERE / "runs" / trial_id / "best.pt") != lock["neural_checkpoint_sha256"][trial_id]:
            raise RuntimeError("Selected neural checkpoint fingerprint changed")
    if digest(HERE / "runs" / lock["tree"] / "model.txt") != lock["tree_checkpoint_sha256"]:
        raise RuntimeError("Selected tree checkpoint fingerprint changed")
    device = cuda(42)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    probabilities, ids, inference_seconds = {}, None, {}
    # Generate all selected test probabilities before reading test labels.
    for trial_id in neural_members:
        cfg = all_records[trial_id]["config"]
        meta = read(HERE / f"cache/{cfg['features']}/manifest.json")
        arrays = test_features(cfg["features"])
        if ids is None:
            ids = arrays["ids"].copy()
        if not np.array_equal(ids, arrays["ids"]):
            raise RuntimeError("Test representation ID mismatch")
        checkpoint = torch.load(HERE / "runs" / trial_id / "best.pt", map_location="cpu", weights_only=False)
        model = build(meta, cfg, checkpoint["bins"]).to(device)
        model.load_state_dict(checkpoint["model"])
        x_num = torch.as_tensor(arrays["x_num"], device=device)
        x_cat = torch.as_tensor(arrays["x_cat"], device=device)
        start = time.perf_counter()
        probabilities[trial_id] = predict(model, x_num, x_cat, cfg["batch_size"], dtype)
        inference_seconds[trial_id] = time.perf_counter()-start
        del model, arrays, x_num, x_cat, checkpoint
        gc.collect()
        torch.cuda.empty_cache()
    tree_record = all_records[lock["tree"]]
    meta = read(HERE / f"cache/{tree_record['features']}/manifest.json")
    arrays = test_features(tree_record["features"])
    if not np.array_equal(ids, arrays["ids"]):
        raise RuntimeError("Tree/neural test IDs mismatch")
    model = lgb.Booster(model_file=str(HERE / "runs" / lock["tree"] / "model.txt"))
    start = time.perf_counter()
    tree_probability = model.predict(frame(arrays, meta))
    tree_seconds = time.perf_counter()-start
    write(target / "test_access.json", {"access_utc": now(), "selection_sha256": digest(HERE / "selection_lock.json"),
                                        "status": "exploratory: original test previously viewed",
                                        "selection_used_test_labels": False})
    with np.load(PROJECT / "results/prepared/full/test_labels.npz", allow_pickle=False) as file:
        if not np.array_equal(ids, file["transaction_id"]):
            raise RuntimeError("Test truth alignment mismatch")
        y = file["y"]
    original = pd.read_csv(PROJECT / "results/metrics.csv")
    table, plot_probabilities = [], {}
    for name, filename in [("Original LightGBM", "lightgbm"), ("Original FT", "ft_transformer")]:
        saved = pd.read_csv(PROJECT / f"results/full/predictions_{filename}_test.csv")
        if not np.array_equal(ids, saved.TransactionID.to_numpy()) or not np.array_equal(y, saved.isFraud.to_numpy()):
            raise RuntimeError("Original test predictions no longer align")
        p = saved.probability.to_numpy()
        old = original.iloc[0 if filename == "lightgbm" else 1]
        table.append({"model": name, **scores(y, p, ids), "training_seconds": float(old.training_seconds)})
        plot_probabilities[name] = p
    single = lock["best_single_neural"]
    selected = np.mean([probabilities[m] for m in lock["neural"]["members"]], axis=0)
    for name, p, seconds, members in [
            ("Strong matched LightGBM", tree_probability, tree_record["training_seconds"], [lock["tree"]]),
            ("Selected single FT", probabilities[single], all_records[single]["active_neural_seconds"], [single]),
            ("Selected FT (validation choice)", selected,
             sum(all_records[m]["active_neural_seconds"] for m in lock["neural"]["members"]), lock["neural"]["members"])]:
        row = {"model": name, **scores(y, p, ids), "training_seconds": seconds, "members": ";".join(members)}
        table.append(row)
        plot_probabilities[name] = p
        pd.DataFrame({"TransactionID": ids, "isFraud": y, "probability": p}).to_csv(
            target / f"predictions_{'tree' if name.startswith('Strong') else 'single' if 'single' in name else 'selected'}_test.csv", index=False)
    table = pd.DataFrame(table)
    table.to_csv(target / "metrics.csv", index=False)
    draw(y, ids, plot_probabilities, table, target)
    comparison = {"status": "COMPLETED_EXPLORATORY", "finished_utc": now(),
                  "selection": lock, "metrics": table.replace({np.nan: None}).to_dict("records"),
                  "active_neural_seconds": state["active_neural_seconds"], "wall_seconds": state["wall_seconds"],
                  "minimum_12h_achieved": state["active_neural_seconds"] >= 43200,
                  "inference_seconds": inference_seconds | {"strong_tree": tree_seconds},
                  "test_independent_of_search": True, "test_was_previously_viewed": True,
                  "baseline_preservation": verify_baseline()}
    write(target / "completion.json", comparison)
    summary(table, comparison, neural_records, target)
    return comparison


def draw(y, ids, probabilities, table, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    for kind in ["precision_recall", "roc", "recall_budget"]:
        fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
        for name, p in probabilities.items():
            if kind == "precision_recall":
                precision, recall, _ = precision_recall_curve(y, p)
                ax.plot(recall, precision, label=name)
                ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
            elif kind == "roc":
                fpr, tpr, _ = roc_curve(y, p)
                ax.plot(fpr, tpr, label=name)
                ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
            else:
                budget = np.linspace(.001, .2, 200)
                ordered = np.lexsort((ids, -p))
                cumulative = np.cumsum(y[ordered])/y.sum()
                ax.plot(budget*100, cumulative[np.ceil(len(y)*budget).astype(int)-1], label=name)
                ax.set(xlabel="Investigation budget (%)", ylabel="Fraud recall", xlim=(0, 20), ylim=(0, 1))
        ax.set_title("Exploratory follow-up: previously viewed chronological test")
        ax.legend(fontsize=8)
        fig.savefig(destination / f"{kind}.png", dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, col in zip(axes, ["average_precision", "recall_top_1pct"]):
        bars = ax.barh(table.model, table[col])
        ax.bar_label(bars, fmt="%.4f")
        ax.set(xlim=(0, 1), title=col)
    fig.savefig(destination / "model_comparison.png", dpi=180)
    plt.close(fig)


def summary(table, completion, neural, destination):
    single = table[table.model == "Selected single FT"].iloc[0]
    selected = table[table.model == "Selected FT (validation choice)"].iloc[0]
    tree = table[table.model == "Strong matched LightGBM"].iloc[0]
    cols = ["model", "roc_auc", "average_precision", "recall_top_1pct", "precision_top_1pct", "recall_top_5pct", "brier_score"]
    lines = ["|"+"|".join(cols)+"|", "|"+"|".join(["---"]*len(cols))+"|"]
    for _, r in table.iterrows():
        lines.append("|"+"|".join(str(r[c]) if c == "model" else f"{r[c]:.6f}" for c in cols)+"|")
    text = f"""# FT-Transformer 改进实验实际报告

状态：完成探索性后续比较。原测试集已在 MVP 中查看，本报告不是新的独立测试。

训练/验证按原 70%/15% 时间窗；搜索不访问最后 15% 标签或预测。
训练字段、词表、频率、quantile 和 embedding bins 均只拟合训练集。
验证 AP 固定选择单模型或等权神经 ensemble，之后一次性复评原测试。
扩展字段也提供了匹配且更充分训练的 LightGBM；原基线全部保持原样。

{chr(10).join(lines)}

FT 选择方案相对强匹配树模型 test AP 差：{selected.average_precision-tree.average_precision:+.6f}；
单个 FT 差：{single.average_precision-tree.average_precision:+.6f}。
1% 调查预算下分别捕获 {int(selected.captured_top_1pct)} / {int(tree.captured_top_1pct)} 个欺诈。
这些是结果，不构成事先承诺的架构优越性。

已完成 {len(neural)} 个神经训练 trial，累计同步训练/验证时间
{completion['active_neural_seconds']/3600:.3f} 小时，campaign wall time
{completion['wall_seconds']/3600:.3f} 小时。12 小时目标实际达到：{completion['minimum_12h_achieved']}。
全部候选、失败、每 epoch 曲线、显存、配置、checkpoint 和指纹位于 runs/ 与 trials.csv。
计算时间包含训练、验证与 checkpoint I/O，不宣称 GPU 始终满负载。

限制：已看过的测试、单个时间切分、选择了多个验证候选、匿名字段时点未知、
不同类别权重/重采样/focal 导致未校准概率不能直接解释为风险、没有金融损失验证。
ensemble 的结果必须标明 ensemble，不能写成单个 FT 的表现。不能承诺 production ready。

下一步：在另一个未接触时间段或新数据上，冻结全部改进后进行含标签延迟的时间外验证。
"""
    (destination / "experiment_summary.md").write_text(text, encoding="utf-8")
