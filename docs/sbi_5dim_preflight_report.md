# SBI 5 维重跑 Pre-flight Report

**生成时间**: 2026-05-22
**目标**: 在 5 维 x_obs 上重跑 SBI 之前的只读 sanity check
**作用**: 不改代码、不跑训练，只验证配置一致性

---

## 检查结果概览

| # | 项目 | 状态 |
|---|------|------|
| A1 | simulator SUMMARY_KEYS 与字段顺序 | ✅ PASS |
| A2 | compute_xobs_from_eeg_v3 输出字段与顺序 | ✅ PASS |
| A3 | simulator 与 x_obs 字段一致性 | ✅ PASS |
| A4 | x_obs_v3.npz 实际值与 Cursor 报告吻合 | ✅ PASS |
| B1 | run_sbi.py 默认 x_obs / 输出目录 / prior 数值 | ⚠️ 有问题 |
| B2 | b 上界 ≤ 42.6 是否需放宽 | ⚠️ 需修复 |
| B3 | S4_sbi/ 未提交修改梳理 | ✅ PASS(已转 5 维) |
| C1+C2 | 锚定参数与 Seed B 一致性 | ✅ PASS(完全一致) |
| D1 | smoke test 入口存在性 | ✅ PASS |
| D2 | multiprocessing 与墙钟估算 | ✅ PASS(估算 ~10h) |

---

## A. 维度一致性(致命级)

### A1. `S4_sbi/simulator_wrapper.py`

- **SUMMARY_KEYS**(file_path:88-91):
  1. `shape_r`
  2. `T4_q`
  3. `T4_freq`
  4. `T8_n_sp_events`
  5. `T11_lag_ms`
- **`_extract_summaries` 返回 dict 字段**(file_path:157-163): 同上 5 个 key,顺序一致
- **`simulator(theta)` 最终输出**(file_path:223): `np.array([stats[k] for k in SUMMARY_KEYS])`,逐字段按 SUMMARY_KEYS 顺序拼装
- **失败回退**(file_path:227): `np.full(len(SUMMARY_KEYS), np.nan)` — 维度跟 SUMMARY_KEYS 锁死

### A2. `S4_sbi/compute_xobs_from_eeg_v3.py`

- **SUMMARY_KEYS**(file_path:101-107): 完全相同的 5 个 key,顺序一致
- **npz 写入字段**(file_path:344-349): `values`(数值数组)、`keys`(字符串数组)、`extraction_metadata`(JSON)
- **x_obs 写入顺序**(file_path:319): `np.array([d[k] for k in SUMMARY_KEYS])` — 同样按 SUMMARY_KEYS 顺序

### A3. 一致性比对

simulator_wrapper.SUMMARY_KEYS == compute_xobs_from_eeg_v3.SUMMARY_KEYS

```
index 0: shape_r          ←→ shape_r        ✅
index 1: T4_q             ←→ T4_q           ✅
index 2: T4_freq          ←→ T4_freq        ✅
index 3: T8_n_sp_events   ←→ T8_n_sp_events ✅
index 4: T11_lag_ms       ←→ T11_lag_ms     ✅
```

**结论:simulator 输出与 x_obs 字段名 + 顺序完全一致。**

### A4. `x_obs_v3.npz` 实际加载验证

```
values      = [1.0, 2.645, 0.75, 15.31, 1.28]
shape       = (5,)
dtype       = float32
keys        = ['shape_r', 'T4_q', 'T4_freq', 'T8_n_sp_events', 'T11_lag_ms']
metadata    = {version: 'v3', n_dims: 5, dropped: ['T6_ibi_cv', 'MI']}
```

数值与 Cursor 报告的 `[1.0, 2.645, 0.75, 15.31, 1.28]` **完全吻合**。

---

## B. 入口配置(严重级)

### B1. `run_sbi.py` 默认配置

- **默认 x_obs 路径**(file_path:592): `_SCRIPT_DIR / "x_obs_v3.npz"` — ✅ 正确指向 5 维文件
- **默认输出目录**(file_path:137): `S4_sbi/sbi_outputs/`
  - 该目录**已存在**,内含上一次 7 维(MI/T6 未删完版)旧产物:
    ```
    all_simulations.npz   (May 7, 旧的 7 维数据)
    round1_posterior.pkl  round2_posterior.pkl  round3_posterior.pkl
    fig_marginals.png  fig_pairplot.png  fig_ppc.png  fig_sbc.png  fig_pareto_overlay.png
    ```
  - **⚠️ 关键风险**:`append_to_simulations`(file_path:225-234)使用 `np.concatenate([prev["x"], x_np], axis=0)` 追加写入。
    - 旧 `all_simulations.npz` 的 `x` 是 (N, 7);新一轮的 `x` 是 (N, 5)。
    - **第一轮就会因 axis=1 维度不匹配抛 ValueError 崩溃。**
  - 旧的 `round*_posterior.pkl` 和 PNG 会被静默覆盖(影响小,但旧 7 维结果会丢失备份)。
- **PRIOR_LOW / PRIOR_HIGH**(file_path:131-132):

  | 参数 | 下界 | 上界 |
  |------|------|------|
  | g_h | 0.035 | 0.095 |
  | g_LK | 0.020 | 0.070 |
  | c_ctx2th | 0.05 | 0.22 |
  | b | 28.4 | **42.6** |

### B2. b 上界判定

- 当前 `PRIOR_HIGH[3] = 42.6`,**尚未放宽**。
- Seed B 的 b = 41.84,距离上界 42.6 仅 1.8% 余量。Seed A b=34.4,Seed C b=36.5,V7 BOUNDS 原始上界也是 42.6。
- **⚠️ 待修**:5 维重跑前建议放宽到 60 或 80,否则后验会在 42.6 处贴边截断。

### B3. S4_sbi/ 下未提交修改

`git diff --stat S4_sbi/` 显示 3 个 modified 文件:

| 文件 | 改动方向 |
|------|----------|
| `simulator_wrapper.py` | **8 维 → 5 维**:删 T6_ibi_cv 和 MI(`T9_mi`),SUMMARY_KEYS 由 7 项收为 5 项,docstring 同步更新,smoke test 提示 NaN count 由 8 改 5 |
| `run_sbi.py` | **8 维 → 5 维 + i18n**:默认 x_obs 路径由 `x_obs.npz` 改 `x_obs_v3.npz`,新增 `--x-obs` CLI 参数,大量英文注释换中文。算法/PRIOR/ROUND_SIMS 数值未动 |
| `compute_xobs_from_eeg.py` | 旧的 8 维(实际生成 7 维)抽取脚本,改动方向只有注释/编码层面,**不再被 run_sbi.py 引用**,可视为废弃路径 |

新增未跟踪文件:`compute_xobs_from_eeg_v3.py`(新主线)、`compute_xobs_from_eeg_v2.py`、`compute_xobs_from_eeg_v1_buggy.py`、`x_obs_v3.npz`、`scan_xobs_params.py`、`plot_scan_diagnostics.py`、`scan_diagnostics/`。

---

## C. 锚定参数验证(严重级)

### C1. simulator_wrapper.py 固定参数(file_path:94-99)

```python
_FIXED = {
    "mue":      3.3406859406304865,
    "mui":      3.2758268081375705,
    "tauA":     1257.4091819444602,
    "c_th2ctx": 0.0329531573906836,
}
```

### C2. Pareto Seed B 对比(`S4_v7_repair/pareto_seeds_fresh_DE.json`,seeds[1])

| 参数 | simulator_wrapper | Seed B | Δ |
|------|-------------------|--------|---|
| mue | 3.3406859406304865 | 3.3406859406304865 | 0 |
| mui | 3.2758268081375705 | 3.2758268081375705 | 0 |
| tauA | 1257.4091819444602 | 1257.4091819444602 | 0 |
| c_th2ctx | 0.0329531573906836 | 0.0329531573906836 | 0 |

**四个值与 Seed B 逐位完全一致(不仅小数点后 3 位)。** ✅

---

## D. Smoke test 与吞吐量(健康检查)

### D1. smoke test 入口

- `simulator_wrapper.py` 末尾(file_path:231-259)有 `if __name__ == "__main__"` 入口,使用 Seed B 的 4 维自由参数:
  ```python
  theta_b = [0.0550, 0.0524, 0.0998, 41.839]  # [g_h, g_LK, c_ctx2th, b]
  ```
- 跑一次预期耗时:V7 的 `SIM_DUR_MS = 60000 ms`,实际墙钟 ≈ 仿真 60 s ÷ neurolib 加速比(经验 ~6-10×)+ 模块导入和 PSD/FOOOF 计算开销。
- **粗估单次 smoke ≈ 70-100 秒**(模块导入 ~15s + 仿真 ~50-80s + FOOOF/PAC 后处理 ~10s)。

### D2. multiprocessing 与 4000 sims 墙钟估算

- **`run_sbi.py` 不使用 multiprocessing**。docstring 明确说明"全程 num_workers=1(fork 下 numba 易挂起)"(file_path:21-22),`run_batch`(file_path:170-222)就是单进程 for 循环。
- 实际预算:
  ```
  ROUND_SIMS = [2000, 1000, 1000, 1000]   → 主循环 5000 次
  + 200 次 SBC
  + 100 次 PPC
  + 4 × 500 次后验自检采样(轻量,基本忽略)
  = ≈ 5300 次仿真
  ```
- 参考上次 9.66h 实际墙钟,等效平均 ~6.6 s/sim(neurolib 在 60 s 仿真上跑约 ~9× realtime + FOOOF/PAC 后处理)。
- **5 维重跑预期墙钟:9-11 小时**(simulator 单次成本不变,仅维度更窄;NSF 训练略快但占比小)。

---

## ⚠️ 5 维重跑前必修清单(按严重程度排序)

| 优先级 | 问题 | 触发后果 | 修复方法 |
|--------|------|---------|---------|
| **P0(必修)** | `sbi_outputs/all_simulations.npz` 是上次 7 维残留;`append_to_simulations` 会在 R1 收尾时 `np.concatenate` 7-dim 旧数据与 5-dim 新数据,axis=1 维度不匹配 → **ValueError 崩溃** | R1 训练通过但 checkpoint 写入失败;若加 try/except 则旧数据会污染新数据集 | 跑前删除/重命名:`mv S4_sbi/sbi_outputs S4_sbi/sbi_outputs_8dim_archive` 或单独删除 `all_simulations.npz` |
| **P1(强烈建议)** | b 的 prior 上界 42.6,Seed B 已经贴边(b=41.84,1.8% 余量);后验若想往上推一点会被截断 | 后验在 b=42.6 处贴边、看不到真实形状 | 在 `run_sbi.py:132` 把 `PRIOR_HIGH = tensor([..., 42.6])` 改为 `..., 60.0` 或 `..., 80.0` |
| **P2(建议)** | 旧 round{1,2,3}_posterior.pkl 与 fig_*.png 会被覆盖 | 失去 8 维旧实验的图与 checkpoint 备份 | 同 P0 一起归档目录即可 |
| **P3(可选)** | `S4_sbi/compute_xobs_from_eeg.py` 已是废弃路径但仍在 git 跟踪中,CLAUDE.md 还写它作为入口 | 后续接手的人可能跑错 | 把它从 CLAUDE.md 入口指令换成 `compute_xobs_from_eeg_v3.py`,或直接 deprecate 旧文件(本次重跑不影响) |

---

## Verdict(中文 200 字以内)

**目前不能直接跑。**

5 维核心一致性(SUMMARY_KEYS、x_obs 数值、Seed B 锚定参数、默认 x_obs 路径)全部 PASS,代码层 ready。但有两个非代码级问题必须先处理:

1. **P0 致命**:`sbi_outputs/all_simulations.npz` 是上次 7 维残留,第一轮 append 时 axis=1 维度不匹配会直接崩。跑前归档/删除该目录。
2. **P1 建议**:`b` 上界 42.6 离 Seed B 的 41.84 只剩 1.8% 余量,后验会被截断。建议放宽到 60 或 80。

处理完这两条再开跑,预计墙钟 9-11 小时。

---

## E. May 7 历史 run 维度确认 + sbi_report_0511.md §3.5 准确性

### E1. `sbi_outputs/` 实际内容

| 项 | 实际值 | 含义 |
|----|-------|------|
| `all_simulations.npz` `theta.shape` | (4000, 4) | 4 个自由参数 ✓ |
| `all_simulations.npz` `x.shape` | **(4000, 7)** | **7 维** x |
| `summary_keys` | `[shape_r, T4_q, T4_freq, T6_ibi_cv, T8_n_sp_events, T11_lag_ms, MI]` | 含 T6 和 MI |
| `round{1,2,3}_posterior.pkl` | 存在,均 2026-05-07 | 训练时使用 7 维 x → 密度估计器输入维度=7 |

### E2. May 7 run 的 7 维身份铁证(`S4_sbi/sbi_log.txt`)

- **第 3 行** `x_obs loaded:` 是 7 个数:`[1.0, 2.645, 0.75, 0.77, 14.352, 1.319, 0.00023]`
- **第 101–108 行** PPC 百分位逐行打印 7 个维度(含 T6_ibi_cv 和 MI),无歧义
- **第 111 行** `SBC skipped (sbi.analysis.run_sbc not available)` — 主 run **SBC 根本没跑**;`fig_sbc.png` 的 mtime 10:46 在主 run 09:03 结束 1.5h 后,意味着它来自后续独立运行的 `run_sbc_standalone.py`

### E3. 时间线

```
2026-05-07 23:24 → 05-08 09:03   7 维 SBI 主 run(9.66h,4000 sims)
2026-05-07 10:46                  fig_sbc.png 单独补出(run_sbc_standalone.py)
2026-05-10 (某时)                 诊断扫描确认 T6/MI 不可修复(scan_xobs_params.py)
2026-05-10 20:32                  x_obs_v3.npz 生成(5 维)
2026-05-10 后                     simulator_wrapper.py / run_sbi.py 改为 5 维(尚未 commit)
2026-05-11                        sbi_report_0511.md 写就
```

**5 维 simulator 与 5 维 x_obs 都是 May 10 之后才存在的产物;sbi_outputs/ 内任何文件都是 7 维 run 的产出。**

### E4. sbi_report_0511.md §3.5 数据准确性核对

报告 §3.5 称"本次运行 5 维均通过(各维度精确百分位待 `sbi_results.md` 中查询)"。

直接查 `S4_sbi/sbi_results.md` PPC 表(May 7 主 run 的真实输出):

| 维度 | 实际百分位 | 状态 | 是否在 §3.5 声称的 5 维子集中 |
|------|-----------|------|-----------------------------|
| shape_r | 100% | **FAIL** | ✓(设计哨兵,可豁免) |
| T4_q | 98% | **FAIL** | ✓(无豁免理由) |
| T4_freq | 30% | PASS | ✓ |
| T6_ibi_cv | 100% | **FAIL** | ✗(被删) |
| T8_n_sp_events | 1% | **FAIL** | ✓(无豁免理由) |
| T11_lag_ms | 6% | PASS(贴边) | ✓ |
| MI | 0% | **FAIL** | ✗(被删) |

**结论**:即便把 T6 和 MI 投影掉,5 维子集中也有 **2 个硬 FAIL(T4_q 98%,T8 1%)+ 1 个豁免 FAIL(shape_r 100%)**。

- §3.5 的"5 维均通过" **与它自己引用的 `sbi_results.md` 数据直接冲突**。
- 报告附录索引表把 §3.5 的来源列为"用户 prompt 提供" — 这等于承认该段落不是从磁盘文件查询出来的。
- §3.4 列举的 SBC KS 数值(`g_h p=0.017` 等)同样不是 May 7 主 run 的输出(主 run SBC skipped),最可能来自后续 `run_sbc_standalone.py` 的某次独立运行,但该独立 run 的日志未在主目录里找到。

### E5. 对下周 5 维重跑的实务影响

1. 下周这次将是**第一次真正意义上的 5 维 SBI 训练**,也是 5 维 PPC 第一次真正出数。
2. **任何 May 7 产物都不可复用**:
   - `round{1,2,3}_posterior.pkl`(密度估计器输入维度=7,与 5 维 x_obs_v3 不兼容)
   - `all_simulations.npz`(x 维度=7,P0 已涵盖)
   - 所有 fig_*.png(数据来源是 7 维 run)
3. **sbi_report_0511.md 须重大修订**(不是修订,是重写以下小节):
   - §3.3 MAP 与 95% CI → 用 5 维新 round3 后验重抽
   - §3.4 SBC KS p-values → 用 5 维新后验重跑 `run_sbc_standalone.py`
   - §3.5 PPC 通过情况 → 用 5 维新 run 的 PPC 输出替换(并诚实记录每维实际百分位,不再隐藏 FAIL)
   - §3.5 Pareto log_prob → 用 5 维新后验重算(注意 Seed B b=41.84 若上界放宽到 60+ 后 log_prob 会变)
4. **历史数据并非全无用**:7 维 run 的 PPC 5/7 FAIL 本身就是有意义的实验证据 — 它支持 §4.1 §4.2 关于 T6 和 MI 不可修复的诊断结论。可在 Stage 3 报告中作为"为什么必须降到 5 维"的佐证保留,但不能伪装成 5 维 SBI 的结果。

### E6. 推荐归档动作(跑 5 维前的清理)

```bash
# 归档 7 维历史产物(保留可追溯性,避免污染新 run)
mv S4_sbi/sbi_outputs S4_sbi/sbi_outputs_7dim_archive_20260507
mv S4_sbi/sbi_log.txt S4_sbi/sbi_log_7dim_archive_20260507.txt
mv S4_sbi/sbi_results.md S4_sbi/sbi_results_7dim_archive_20260507.md
mv S4_sbi/x_obs.npz S4_sbi/x_obs_7dim_archive.npz
```

这样下周 5 维新 run 起始目录干净,旧的 7 维证据仍可追溯(用于 Stage 3 §4.1–4.2 引用)。
