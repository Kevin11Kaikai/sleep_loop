# `load_target_psd()` 说明（`s4_personalize_fig7_v3.py`）

本文档汇总对 **`load_target_psd`** 的讨论：在个体化拟合流程里它如何构造 **经验 N3 EEG 目标功率谱**，以及 **MNE**、**Welch**、**输出数组形状** 的含义。

**代码位置**：`models/s4_personalize_fig7_v3.py` 中 **Part 1**（约第 272–307 行）。

---

## 1. 函数在整体流程中的角色

- 在 **差分进化开始前调用一次**（与 `main()` 中顺序一致）。  
- 输出 **`target_psd`**、**`target_freqs`**，供后续 **`compute_target_periodic`**（FOOOF）和 **`compute_fitness_v3`** 使用。  
- 含义：**当前受试者（默认 `SUBJECT_ID = SC4001`）在 N3 睡眠阶段、多 epoch 平均后的「目标 PSD vs 频率」曲线**——在优化里当作 **ground truth / 拟合目标**（严格说是**估计谱**，不是理论真值）。

---

## 2. 逐步在做什么？

| 步骤 | 代码意图 |
|------|-----------|
| 读 manifest | 从 `data/manifest.csv` 取该受试者的 **PSG** 与 **睡眠分期文件**路径。 |
| **`mne.io.read_raw_edf`** | 读 EDF 连续记录为 **`Raw`**；**`include=EEG_CHANNELS`** 只载入 **`EEG Fpz-Cz`、`EEG Pz-Oz`**；**`preload=True`** 整段进内存，便于按 epoch 切片。 |
| **`fs_eeg`** | 采样率（Hz），常见 **100 Hz**（Sleep-EDF cassette）。 |
| **分期对齐** | `set_annotations` + `events_from_annotations`：把 **hypnogram** 与 `Raw` 对齐。 |
| **`Epochs`** | 按 **N3 / N4 标签**（`N3_LABELS`）切 **30 s** 段（`EPOCH_DURATION`）。 |
| **逐 epoch 循环** | 取双通道数据 → **伪迹剔除**（任一路峰峰值 > `ARTIFACT_THRESH` 则丢弃）→ **通道平均** → **Welch** → 只保留 **`[F_LO, F_HI]`**（默认 **0.5–20 Hz**）→ 将 **`p_ep[freq_mask]`** 加入列表 **`psds`**。 |
| **聚合** | **`target_psd = np.mean(psds, axis=0)`**：在 **epoch 维**上平均；**`target_freqs = f_ep[freq_mask]`**（与最后一个合格 epoch 的频率轴一致，所有 epoch 共用同一 `freq_mask` 逻辑故网格一致）。 |

**MNE 在本函数中的角色**：主要是 **读 PSG、选 EEG 通道、与分期对齐、切 N3 epoch**；**谱估计**用 **SciPy `welch`**，不是 MNE 的谱函数。

---

## 3. 为什么用 `np.mean(psds, axis=0)`？

- **`psds`**：每个元素是一条 **某一 N3 epoch** 在 **相同频率掩码下** 的 1D PSD。  
- **`axis=0`**：沿 **epoch 维**平均，得到 **每个频率点一个标量** → **`target_psd.shape == (n_freqs,)`**。

**原因简述**：

1. **降方差**：单次 epoch 的 Welch 谱波动大，多段平均后更平滑、更稳。  
2. **代表「该受试者的典型 N3 谱」**：不是拟合单段，而是拟合 **跨夜/跨 epoch 的平均形态**。  
3. **优化接口简单**：进化每次只与 **一条固定目标曲线** 比较，无需 per-epoch 多目标。  

---

## 4. `target_psd` 与 `target_freqs` 的关系

- **一一对应**：**`target_freqs[i]`**（Hz）与 **`target_psd[i]`**（Welch 在默认 `density` 标度下约为 **V²/Hz** 量级）描述同一条 **经验 PSD 曲线**。  
- 在后续脚本中可记为 **「ground truth EEG：PSD vs frequency」**。

---

## 5. 频率轴长度与间隔（例如 196 点、0.5～20 Hz、步长约 0.1 Hz）

Welch 的频率分辨率近似为：

\[
\Delta f \approx \frac{f_s}{n_{\text{perseg}}}
\]

本函数中 **`nperseg = min(10 * fs_eeg, len(mean_sig))`**：常用 **30 s × 100 Hz = 3000** 点 → **`nperseg = 1000`**（10 s 窗）→ **`Δf = 100/1000 = 0.1` Hz**。

在 **`f_ep`** 上取 **`F_LO ≤ f ≤ F_HI`**（0.5～20 Hz）后，频点形如 **0.5, 0.6, …, 20.0**，个数为：

\[
\frac{20 - 0.5}{0.1} + 1 = 196
\]

**注意**：**196 不是写死的**；若 **`fs_eeg`** 或 **epoch 长度** 变化导致 **`nperseg`** 变化，**\(\Delta f\)** 与 **掩码后点数** 都会变。

---

## 6. Welch（与 FFT 的关系，极简）

- **目的**：从 **有限长** 时间序列估计 **功率谱密度**。  
- **做法**：将序列分成 **多段**（每段长度 **`nperseg`**），每段 **加窗（Hann）** 做 **FFT/周期图**，再 **平均** → 比 **单次整段 FFT** 方差更小。  
- **`noverlap=nperseg//2`**：50% 重叠，增加可平均段数，进一步平滑。  

详见信号处理教材或 `scipy.signal.welch` 文档。

---

## 7. 相关配置常量（同文件 Config）

| 常量 | 典型值 | 含义 |
|------|--------|------|
| `SUBJECT_ID` | `SC4001` | 受试者 ID |
| `EEG_CHANNELS` | Fpz-Cz, Pz-Oz | 载入的头皮通道 |
| `N3_LABELS` | stage 3 / 4 | N3 epoch 来源 |
| `ARTIFACT_THRESH` | 200 µV | 峰峰值伪迹阈值 |
| `EPOCH_DURATION` | 30 s | 每段时长 |
| `F_LO`, `F_HI` | 0.5, 20 Hz | 保留的频段 |

---

## 8. 下游衔接（便于对照）

- **`compute_target_periodic(target_psd, target_freqs)`**：FOOOF → **`target_periodic`**, **`fooof_freqs`**。  
- **`compute_fitness_v3(..., target_psd, target_freqs, ...)`**：仿真谱与目标比较；**`shape_r`** 主路径在 **`fooof_freqs`** 网格上对齐（见 `docs/s4_personalize_fig7_v3_compute_fitness_v3.md`）。

---

*文档根据对话整理；若 `load_target_psd` 实现变更，请以 `models/s4_personalize_fig7_v3.py` 为准。*
