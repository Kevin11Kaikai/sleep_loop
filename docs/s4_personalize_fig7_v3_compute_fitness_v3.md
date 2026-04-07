# `compute_fitness_v3` 逐段说明

本文档解释 `models/s4_personalize_fig7_v3.py` 中的 **`compute_fitness_v3`**：差分进化（Differential Evolution, DE）每尝试一组参数时调用一次；成功时返回 **`-fitness`**（便于 `scipy` **最小化** = 最大化 `fitness`），失败时返回 **`0.0`**。

**阅读前只需知道：**

- **功率谱密度（PSD）**：把时间序列拆成「不同频率的正弦成分各占多少能量」，横轴是频率（Hz），纵轴是功率。粗略理解：**哪个频段波动强**。
- **Welch 方法**：把长时序切成多段、加窗、做傅里叶变换再平均，得到一条**更稳**的 PSD 估计（比直接对整段做 FFT 噪声小）。
- **FOOOF**：在 log 域把 PSD 近似成 **1/f 背景（非周期）+ 若干鼓包（周期峰）**。本脚本只用「减掉 1/f 后剩下的形状」来比较 EEG 与仿真。
- **Pearson 相关系数 \(r\)**：两条曲线「一起高、一起低」的程度，在 \([-1,1]\)，**不比较绝对大小**，主要比较**形状**。

---

## 函数签名与输入

```python
def compute_fitness_v3(params_vec,
                       target_psd, target_freqs,
                       target_periodic, fooof_freqs):
```

| 参数 | 含义 |
|------|------|
| `params_vec` | 长度 8 的向量：`mue, mui, b, tauA, g_LK, g_h, c_th2ctx, c_ctx2th`（皮层/丘脑/耦合参数）。 |
| `target_psd`, `target_freqs` | 受试者 N3 EEG 的平均 PSD 及其频率轴（脚本启动时算好，**每次适应度评估共用**）。 |
| `target_periodic` | 目标 EEG 在 log 域里 **去掉 FOOOF 拟合的 1/f 后** 的序列（与 `fooof_freqs` 对齐）。 |
| `fooof_freqs` | 目标 EEG 做 FOOOF 时使用的**频率采样点**（仿真侧也要插值到这条轴上再拟合，才能和 `target_periodic` 逐点比较）。 |

---

## 第 1 段：全局计数与解包参数

```python
global _eval_count, _best_score, _best_params, _records
_eval_count += 1
mue, mui, b, tauA, g_lk, g_h, c_th2ctx, c_ctx2th = params_vec
```

- **`_eval_count`**：第几次评估（用于日志、CSV）。
- 把向量拆成 8 个标量，传给 `build_model`。

---

## 第 2 段：运行仿真

```python
try:
    m = build_model(...)
    m.run()
except Exception:
    return 0.0
```

- 用当前参数建 **丘脑–皮层 MultiModel** 并积分 **30 s**（`SIM_DUR_MS`）。
- 仿真失败时 **`return 0.0`**；成功时在函数末尾 **`return -fitness`**。`differential_evolution` **最小化**目标函数：好的解 `fitness` 大（如 0.8），对应目标值 **-0.8**（更小）；失败返回 **0.0** 大于 **-0.8**，优化器更倾向保留好解。

---

## 第 3 段：取出皮层与丘脑发放率

```python
r_exc = m[f"r_mean_{EXC}"]
if r_exc.ndim == 2 and r_exc.shape[0] >= 2:
    r_ctx  = r_exc[0, :] * 1000.0
    r_thal = r_exc[1, :] * 1000.0
else:
    ...
```

- **`r_mean_EXC`**：兴奋性群体的平均发放率；双节点网络时为 **2×T** 矩阵，**第 0 行皮层、第 1 行丘脑**。
- 库内单位常为 kHz，**×1000** 转成 **Hz**（每秒发放次数的量级）。

---

## 第 4 段：去掉前 5 秒（burn-in）

```python
n_drop = int(5.0 * FS_SIM)
r_ctx  = r_ctx[n_drop:]
r_thal = r_thal[n_drop:]
```

- 仿真开始常有瞬态，**丢掉前 5 s** 再算谱、算动力学，与论文式设定一致。

---

## 第 5 段：皮层太弱则判失败

```python
if r_ctx.max() < 0.1:
    return 0.0
```

- burn-in 后若皮层几乎不放电，没有可比的谱，直接最差分。

---

## 第 6 段：皮层 Welch PSD

```python
nperseg = min(int(10.0 * FS_SIM), len(r_ctx))
f_ctx, p_ctx = welch(r_ctx, fs=FS_SIM, nperseg=nperseg, ...)
mask = (f_ctx >= F_LO) & (f_ctx <= F_HI)
f_ctx, p_ctx = f_ctx[mask], p_ctx[mask]
```

- **`f_ctx`**：频率（Hz）。**`p_ctx`**：**线性**功率谱密度（发放率波动的谱）。
- 只保留 **0.5–20 Hz**，与 EEG 目标频段一致。

---

## 第 7 段：`shape_r`（FOOOF 路径）

**目标**：比较 **仿真皮层谱** 与 **真实 N3 EEG 谱** 在「**去掉 1/f 背景后**」的**形状**有多像。

1. **`p_interp = interp1d(f_ctx, p_ctx)(fooof_freqs)`**  
   把 Welch 得到的功率 **线性插值** 到 **`fooof_freqs`**（与目标 EEG FOOOF 相同的频率轴）。  
   **注意**：必须先插**功率**，再在 FOOOF 网格上拟合；与「先 FOOOF 再插残差」不等价。

2. **`fm_sim = FOOOF(...)`**  
   新建模型，超参与目标 EEG 侧一致（峰宽、最多峰数等）。

3. **`fm_sim.fit(fooof_freqs, p_interp, [F_LO, F_HI])`**  
   在 log 域拟合 **1/f + 峰**；得到 **`fm_sim._ap_fit`**（非周期线）。

4. **`sim_log = log10(p_interp[:len(_ap_fit)] + 1e-30)`**  
   与 `_ap_fit` **等长**的 log 功率；`1e-30` 防止 log(0)。

5. **`sim_periodic = sim_log - fm_sim._ap_fit`**  
   **周期残差**（log 谱 − 只含 1/f 的线）。与 `target_periodic` 定义方式相同。

6. **`n_r = min(len(sim_periodic), len(target_periodic))`**  
   防长度差 1。

7. **`shape_r, _ = pearsonr(sim_periodic[:n_r], target_periodic[:n_r])`**  
   形状相似度，写入 JSON 的 **`shape_r`**。

8. **`shape_r = max(shape_r, 0.0)`**  
   负相关截成 0，避免 **`0.35 * shape_r`** 拉负分。

若 FOOOF 不可用或异常，走 **fallback**：在 `target_freqs` 上插值后与 `target_psd` 比 log 谱的加权平方差，再映射成 0–1 的标量代替 `shape_r`。

---

## 第 8 段：`so_power` 与 `spindle_power`

- 再次 **`p_interp2`** 到 `fooof_freqs`，再 **`fm2.fit`**（与 `shape_r` 同一数据，但单独拟合一次读峰参数——实现上略重复，便于读峰表）。
- 遍历 **`fm2.peak_params_`**：若峰中心落在 **慢波带 [0.3, 1.5] Hz** 或 **纺锤带 [8, 16] Hz**，取峰幅度中的最大值作为 **`so_power` / `spindle_power`**（FOOOF 内部 log 域峰高）。
- 这两个量经后续 **`0.15 * …`** 进入总适应度（与 v2 思想一致；v3 文档中若权重写法不同以代码为准——代码为 **0.15 与 0.15**）。

---

## 第 9 段：动力学分数

```python
dynamics_score, dyn_details = compute_dynamics_score_v3(r_ctx, r_thal)
```

- 用 **时域** 检查是否像真 N3：**DOWN、UP、UP 持续、皮层 SO 峰频、丘脑纺锤宽度** 等（详见另一篇 Part 文档中的 **V3 dynamics** 节）。
- 返回 **[0, 1]** 的加权分及诊断字典，供 CSV 记录。

---

## 第 10 段：加权总分

```python
fitness = (0.35 * shape_r
           + 0.15 * so_power
           + 0.15 * spindle_power
           + 0.35 * dynamics_score)
```

| 项 | 权重 | 通俗含义 |
|----|------|----------|
| `shape_r` | 0.35 | 频谱形状（去 1/f 后）像不像 EEG |
| `so_power` | 0.15 | 慢波相关峰在 FOOOF 里够不够强 |
| `spindle_power` | 0.15 | 纺锤相关峰够不够强 |
| `dynamics_score` | 0.35 | 时域是否像真慢波/纺锤动力学 |

---

## 第 11 段：记录与返回

- 把参数、各分项、**T1–T5** 布尔结果等写入 **`record`**，追加到 **`_records`**。
- 若 **`fitness > _best_score`**，更新全局最优 **`_best_params`**。
- **`return -fitness`**：交给 DE **最小化** → 等价于 **最大化 fitness**。

---

## 与绘图脚本对照（避免混淆）

若用 **`plot_fig7_v2_fast.py`** 重算 **`shape_r`**，须 **重复本节第 7 段的顺序**（插值功率 → FOOOF → `sim_periodic`），并使用 **相同的仿真时长与 burn-in**。  
仅在原生 Welch 频率上做 FOOOF、再插值 **`sim_periodic`**，得到的 **Pearson r 与 JSON 中 `shape_r` 一般不一致**。

---

*文档对应仓库脚本版本以 `models/s4_personalize_fig7_v3.py` 为准。*
