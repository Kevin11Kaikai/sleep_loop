"""
s1_all_stages.py — Session 2-A：全睡眠分期 PSD 提取
--------------------------------------------------
从 Sleep-EDF cassette 数据中为每个分期 (Wake/N1/N2/N3/REM) 计算平均功率谱密度 (PSD)。
读取 manifest.csv 中前 MAX_SUBJECTS 名受试者；N3 与 N4 标签合并为 n3。

全流程 1→6（执行顺序；正文中搜索「# ── 步骤」可跳到对应代码块）
────────────────────────────────────────────────────────────────────────
  步骤 1 — 入口
    manifest 前 MAX_SUBJECTS 行；初始化 stage_psds / stage_records / freqs_out；
    外层 for 按行取 subject_id、psg_path、hypnogram_path，缺文件则 skip。

  步骤 2 — 每名受试者（在外层循环体内，每个 subject 执行一次）
    read_raw_edf（仅 EEG Fpz-Cz / Pz-Oz）→ sfreq；
    read_annotations(hyp) + set_annotations 把睡眠分期标到 raw 上。

  步骤 3 — 按分期（内层 for stage_name in STAGE_MAP，与其它分期独立）
    用标注字符串建 event_id → events_from_annotations → mne.Epochs 切 30 s
    （tmin=0 对齐分期事件起点）。

  步骤 4 — 每个 epoch（再内层 for ep_idx，跨所有被试的 PSD 都 append 到同一列表）
    get_data → 峰峰值伪迹剔除 → 双通道时间维平均 → welch → 0.5–30 Hz 掩码 →
    stage_psds[stage].append；stage_records 记 subject_id 与 epoch_idx。

  步骤 5 — 全部受试者循环结束后
    各分期将 (n_epochs, n_freqs) 在 epoch 维 mean → stage_mean_psds。

  步骤 6 — 输出与校验
    保存 target_freqs.npy、psd_<stage>.npy、target_psd.npy(N3)、epochs_*.csv；
    画 psd_all_stages.png；打印 N3 delta 占比及与 Wake 的对比。

数据流（细粒度）：manifest → EDF + hyp → events → 30s Epochs → 伪迹 → 平均 → Welch
  → 0.5–30 Hz → 累加 → 分期平均 → 落盘 / 图 / 校验

输出文件：
  - data/psd_<stage>.npy、data/target_freqs.npy
  - data/target_psd.npy（= N3 平均 PSD，供 Session 2-B）
  - data/epochs_<stage>.csv（通过质检的 epoch 所属 subject/索引）
  - outputs/psd_all_stages.png
"""

import os
import numpy as np
import pandas as pd
import mne
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mne.set_log_level("WARNING")

# ── config ────────────────────────────────────────────────────────────────────
MANIFEST_PATH   = "data/manifest.csv"
EEG_CHANNELS    = ["EEG Fpz-Cz", "EEG Pz-Oz"]  # 与 Sleep-EDF 通道名一致；两路平均后做 Welch
EPOCH_DURATION  = 30.0        # 与标准睡眠分期 epoch 一致（秒）
ARTIFACT_THRESH = 200e-6      # 峰峰值超过 200 µV 的 epoch 丢弃（伏）
MAX_SUBJECTS    = 10          # 只处理 manifest 前 N 行，便于快速跑通

# 逻辑分期名 → hypnogram 里可能出现的字符串（N3/N4 合并到 n3）
STAGE_MAP = {
    "wake": ["Sleep stage W"],
    "n1":   ["Sleep stage 1"],
    "n2":   ["Sleep stage 2"],
    "n3":   ["Sleep stage 3", "Sleep stage 4"],
    "rem":  ["Sleep stage R"],
}

STAGE_COLORS = {
    "wake": "#E24B4A",
    "n1":   "#EF9F27",
    "n2":   "#1D9E75",
    "n3":   "#534AB7",
    "rem":  "#D4537E",
}

# ── 步骤 1：入口 — manifest 前 MAX_SUBJECTS 行（列含 subject_id, psg_path, hypnogram_path） ──
manifest = pd.read_csv(MANIFEST_PATH).head(MAX_SUBJECTS)
print(f"Processing {len(manifest)} subjects, {len(STAGE_MAP)} stages...")
print(f"Stages: {list(STAGE_MAP.keys())}")

# 步骤 1（续）：跨所有受试者共用的累加器（步骤 4 往里 append，步骤 5 再求平均）
#   stage_psds: 每分期「通过质检的 epoch」各一条 PSD（长度同 freqs_out）
#   stage_records: 对应 epoch 的 subject_id 与在该分期内的 epoch_idx
stage_psds    = {stage: [] for stage in STAGE_MAP}
stage_records = {stage: [] for stage in STAGE_MAP}
freqs_out     = None  # 首次有效 Welch 后锁定 0.5–30 Hz 频率轴

# ── 主循环：每名受试者依次执行「步骤 2 → 步骤 3 → 步骤 4」────────────────────
for _, row in manifest.iterrows():
    subject_id = row["subject_id"]
    psg_path   = row["psg_path"]
    hyp_path   = row["hypnogram_path"]

    if not os.path.exists(psg_path) or not os.path.exists(hyp_path):
        print(f"  [{subject_id}] SKIP — file not found")
        continue

    try:
        # ── 步骤 2：读 PSG（双通道 EEG）与 hypnogram，并挂到 raw ─────────────────
        raw = mne.io.read_raw_edf(
            psg_path,
            include=EEG_CHANNELS,
            preload=True,
            verbose=False,
        )
        fs = raw.info["sfreq"]

        annotations = mne.read_annotations(hyp_path)
        raw.set_annotations(annotations)

        stage_counts = {}  # 每分期：(通过质检数, 该分期总 epoch 数)

        # ── 步骤 3：按 STAGE_MAP 各分期单独匹配标注 → events → 30 s Epochs ───────
        for stage_name, labels in STAGE_MAP.items():
            event_id = {lbl: idx + 1 for idx, lbl in enumerate(labels)}
            try:
                events, event_dict = mne.events_from_annotations(
                    raw, event_id=event_id, verbose=False
                )
            except Exception:
                # 标注里没有这些字符串时 mne 会报错
                stage_counts[stage_name] = (0, 0)
                continue

            if len(events) == 0:
                stage_counts[stage_name] = (0, 0)
                continue

            # 自每个分期事件 onset 起截 30 s（tmin=0）
            epochs = mne.Epochs(
                raw, events,
                event_id=event_dict,
                tmin=0.0, tmax=EPOCH_DURATION,
                baseline=None,
                preload=True,
                verbose=False,
            )

            n_total  = len(epochs)
            n_passed = 0

            # ── 步骤 4：逐 epoch — 伪迹 → 双通道平均 → Welch → 频段 → 写入累加器 ───
            for ep_idx in range(n_total):
                data = epochs[ep_idx].get_data()[0]   # (n_channels, n_times)

                # 4a 伪迹：任一路峰峰值 > 200 µV 则丢弃本 epoch
                pp = data.max(axis=1) - data.min(axis=1)
                if np.any(pp > ARTIFACT_THRESH):
                    continue

                # 4b 双通道时间维平均 → 单通道；再 4c Welch PSD
                mean_signal = data.mean(axis=0)

                nperseg = min(int(10.0 * fs), len(mean_signal))  # ~10 s 窗，50% 重叠
                f_ep, p_ep = welch(
                    mean_signal, fs=fs,
                    nperseg=nperseg,
                    noverlap=nperseg // 2,
                    window="hann",
                )

                # 4d 0.5–30 Hz；首次确定 freqs_out，之后各 epoch 须同形状
                freq_mask = (f_ep >= 0.5) & (f_ep <= 30.0)
                if freqs_out is None:
                    freqs_out = f_ep[freq_mask]

                stage_psds[stage_name].append(p_ep[freq_mask])
                stage_records[stage_name].append({
                    "subject_id": subject_id,
                    "epoch_idx":  ep_idx,
                })
                n_passed += 1

            stage_counts[stage_name] = (n_passed, n_total)

        parts = [f"{s}:{v[0]}/{v[1]}" for s, v in stage_counts.items()]
        print(f"  [{subject_id}] " + "  ".join(parts))

    except Exception as e:
        print(f"  [{subject_id}] ERROR — {e}")
        continue

# ── 步骤 5：先打印各分期已收集的 epoch 条数，再计算 stage_mean_psds ───────────
print("\n=== Stage Summary ===")
print(f"{'Stage':<6} {'Epochs':>8}  Note")
print("-" * 38)
for stage in STAGE_MAP:
    n    = len(stage_psds[stage])
    note = "← target for Session 2-B" if stage == "n3" else ""
    print(f"{stage:<6} {n:>8}  {note}")

# 步骤 5（续）：(n_epochs, n_freqs) → axis=0 平均
stage_mean_psds = {}
for stage in STAGE_MAP:
    if len(stage_psds[stage]) == 0:
        print(f"  WARNING: no epochs for stage {stage}")
        stage_mean_psds[stage] = np.zeros_like(freqs_out)
    else:
        arr = np.array(stage_psds[stage])        # (n_epochs, n_freqs)
        stage_mean_psds[stage] = arr.mean(axis=0)

# ── 步骤 6：保存 npy / csv ────────────────────────────────────────────────────
os.makedirs("data",    exist_ok=True)
os.makedirs("outputs", exist_ok=True)

np.save("data/target_freqs.npy", freqs_out)
print(f"\nSaved: data/target_freqs.npy  shape={freqs_out.shape}")

for stage, mean_psd in stage_mean_psds.items():
    fname = f"data/psd_{stage}.npy"
    np.save(fname, mean_psd)
    print(f"Saved: {fname}  shape={mean_psd.shape}  n_epochs={len(stage_psds[stage])}")

np.save("data/target_psd.npy", stage_mean_psds["n3"])
print("Saved: data/target_psd.npy  (= psd_n3.npy, for Session 2-B)")

for stage, records in stage_records.items():
    if records:
        pd.DataFrame(records).to_csv(f"data/epochs_{stage}.csv", index=False)

# ── 步骤 6（续）：对比图 — 左绝对 PSD，右归一化；N3 加粗实线 ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for stage in STAGE_MAP:
    psd = stage_mean_psds[stage]
    n   = len(stage_psds[stage])
    if n == 0:
        continue
    kw = dict(color=STAGE_COLORS[stage],
              lw=2.0 if stage == "n3" else 1.2,
              ls="-"  if stage == "n3" else "--",
              label=f"{stage.upper()} (n={n})")
    axes[0].semilogy(freqs_out, psd, **kw)
    axes[1].plot(freqs_out, psd / (psd.sum() + 1e-30),
                 color=STAGE_COLORS[stage],
                 lw=kw["lw"], ls=kw["ls"],
                 label=stage.upper())

for ax in axes:
    ax.axvspan(0.5,  4.0,  alpha=0.10, color="blue",  label="Delta" if ax is axes[0] else None)
    ax.axvspan(8.0,  13.0, alpha=0.10, color="green", label="Alpha/Sigma" if ax is axes[0] else None)
    ax.set_xlabel("Frequency [Hz]", fontsize=11)
    ax.set_xlim(0.5, 30)
    ax.legend(fontsize=9)

axes[0].set_ylabel("Power [V²/Hz]",   fontsize=11)
axes[0].set_title("All stages — absolute PSD", fontsize=12)
axes[1].set_ylabel("Normalised power", fontsize=11)
axes[1].set_title("All stages — normalised PSD", fontsize=12)

plt.suptitle(
    f"Sleep-EDF: Mean PSD per stage  ({MAX_SUBJECTS} subjects)",
    fontsize=13,
)
plt.tight_layout()
plt.savefig("outputs/psd_all_stages.png", dpi=150, bbox_inches="tight")
print("\nSaved: outputs/psd_all_stages.png")

# ── 步骤 6（续）：校验 — N3 delta 占比、与 Wake 的 delta 总功率比较 ───────────
print("\n=== Validation ===")
n3_psd = stage_mean_psds["n3"]

if len(stage_psds["n3"]) == 0:
    print("✗ FAIL: no N3 epochs found")
else:
    delta_mask = (freqs_out >= 0.5) & (freqs_out <= 4.0)
    total_mask = (freqs_out >= 0.5) & (freqs_out <= 30.0)
    delta_ratio = n3_psd[delta_mask].sum() / (n3_psd[total_mask].sum() + 1e-10)

    if delta_ratio > 0.4:
        print(f"✓ N3 delta dominance confirmed  (delta ratio = {delta_ratio:.2f})")
    else:
        print(f"✗ N3 delta ratio low  ({delta_ratio:.2f} < 0.4)")

    if len(stage_psds["wake"]) > 0:
        wake_psd   = stage_mean_psds["wake"]
        n3_delta   = n3_psd[delta_mask].sum()
        wake_delta = wake_psd[delta_mask].sum()
        if n3_delta > wake_delta:
            print("✓ N3 has more delta power than Wake  (physiologically correct)")
        else:
            print("✗ N3 delta power unexpectedly lower than Wake")

print("\nSession 2-A ALL STAGES complete.")
print("data/target_psd.npy  is ready for Session 2-B.")
