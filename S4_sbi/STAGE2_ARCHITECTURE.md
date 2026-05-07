# SBI Stage2 三文件架构（教学向）

本文档对应 `S4_sbi/` 下三脚本的**数据流、职责分工与三处关键接口**，便于对照代码阅读 `sbi` 的 SNPE 流程。

---

## 数据流（EEG → x_obs → 训练 → 后验）

```text
[data/manifest.csv + Sleep-EDF]
            |
            v
+------------------------------------------+
| compute_xobs_from_eeg.py                 |
| 读 N3 段 -> r_proxy -> 汇总统计 -> 校验  |
+------------------------------------------+
            |
            v
     S4_sbi/x_obs.npz
  keys: values, keys, extraction_metadata
            |
            v
+------------------------------------------+
| run_sbi.py                               |
| 读 x_obs -> BoxUniform 先验 -> 多轮 SNPE |
| proposal.sample -> simulator(theta)->x   |
+--------+---------------------------------+
         |                    ^
    theta| (4,)              | x 与 SUMMARY_KEYS 同序
         v                    |
+------------------------------------------+
| simulator_wrapper.py                     |
| importlib 加载 V7 -> build_model -> run  |
| r_ctx,r_thal -> _extract_summaries -> x  |
+------------------------------------------+
            |
            v
   S4_sbi/sbi_outputs/round{r}_posterior.pkl
   （每轮 train 后 build_posterior，非 dry-run 时 pickle）
```

---

## 各文件一句话

- **`compute_xobs_from_eeg.py`**：从 SC4001 的 N3 EEG 构建 firing-rate 代理信号，按与 V7 对齐的规则计算汇总统计量，做硬性 sanity check 后写入 `x_obs.npz`。
- **`simulator_wrapper.py`**：将 4 维自由参数与 Seed B 固定参数拼成 V7 模型，单次仿真后提取与 `SUMMARY_KEYS` 同序的汇总向量（失败则全 `NaN`）。
- **`run_sbi.py`**：加载 `x_obs` 与先验，多轮 `append_simulations` + `train` + `build_posterior`，checkpoint 后验与 `(theta,x)`，并跑边际/配对图、PPC、SBC 等诊断。

---

## 三处关键交接

1. **x_obs**  
   由 `np.savez` 写入 `S4_sbi/x_obs.npz`，含：
   - `values`：`float32` 一维数组，顺序与 `SUMMARY_KEYS` 一致；
   - `keys`：统计量名称列表；
   - `extraction_metadata`：JSON 字符串（被试、通道、代理方法等）。  
   `run_sbi.py` 用 `np.load` 读 `values`，转为 `torch.float32` 的 `x_obs_t`；凡对后验做条件化（训练后采样、作图、PPC 等）均传 `x=x_obs_t`。

2. **simulator 返回值**  
   `simulator(theta)` 接受 `array-like` 或可 `.detach().cpu().numpy()` 的 `Tensor`，内部展平为 `[g_h, g_LK, c_ctx2th, b]`（与先验维序一致）。  
   返回 **`np.ndarray`，形状 `(len(SUMMARY_KEYS),)`**，dtype 一般为 `float64`；各分量多为**标量统计**（如 `T4_freq` 单位为 Hz，`T8_n_sp_events` 为按 60 s 归一化的事件计数等，详见各脚本注释）。失败时返回**同形状**的全 `NaN` 向量。批次在 `run_sbi.py` 中堆叠为 `torch.float32` 的 `(N, d_x)`。

3. **后验对象**  
   每轮 `posterior = inference.build_posterior(density_est)`；完整运行时用 `pickle.dump` 写入 `sbi_outputs/round{r}_posterior.pkl`。  
   - **采样**：`posterior.sample((n,), x=x_obs_t)`  
   - **对数概率**：`posterior.log_prob(theta_t, x=x_obs_t)`（例如 Pareto 种子 overlay）  
   仓库脚本**未**提供加载函数；需自行 `pickle.load` 读回，并继续传入与训练时一致的 `x_obs_t`。

---

## 实现与注释差异（学习时务必核对）

部分顶层 docstring 仍写「8 维汇总」或「8 个 SUMMARY」，但 **`compute_xobs_from_eeg.py` 与 `simulator_wrapper.py` 中当前的 `SUMMARY_KEYS` 列表为 7 项**（未包含 `T12_n_verified`）。以两文件内 `SUMMARY_KEYS` 与 `x_obs.npz` 里 `values.shape` 为准。

---

## 五条要点（速查）

- **流程顺序**：先运行 `compute_xobs_from_eeg.py` 生成 `x_obs.npz`，再运行 `run_sbi.py`；`simulator_wrapper` 由 `run_sbi` 导入并在仿真循环中调用。
- **观测与模拟必须在同一坐标系**：`x_obs` 与 `simulator` 输出按相同 `SUMMARY_KEYS` 顺序对齐，NSF 的 `z_score_x` 在 `x` 各维上独立标准化，维数或顺序错位会直接破坏推断。
- **theta 只有 4 维**：`mue, mui, tauA, c_th2ctx` 固定在 Seed B；SBI 只推断 `g_h, g_LK, c_ctx2th, b`，与 `BoxUniform` 边界一致。
- **NaN 即无效仿真**：`run_sbi.py` 会丢弃含 `NaN` 的 `x` 行，并在 NaN 率过高或有效样本过少时中止；Windows 上须保持 `num_workers=1`、顺序仿真（见各文件注释）。
- **后验使用方式**：推理目标固定在 `x_obs_t`；读入已保存的 `posterior.pkl` 后，仍需在 `sample` / `log_prob` 中传入**同一次提取**的 `x_obs_t`（同形状、dtype、数值），否则条件分布不对应。
