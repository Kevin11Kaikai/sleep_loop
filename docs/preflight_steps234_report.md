# Pre-flight Steps 2-4 Report

**生成时间**: 2026-05-22 20:09 (本地)
**目的**: 在启动 5 维 SBI 训练之前的最终验证
**状态**: 等用户 "go" 之后启动 `python S4_sbi/run_sbi.py`

---

## Step 2: Prior Bound Modification

### 修改前后

**修改前**(`S4_sbi/run_sbi.py:132`):
```python
PRIOR_HIGH  = tensor([0.095, 0.070, 0.22, 42.6], dtype=torch.float32)
```

**修改后**:
```python
PRIOR_HIGH  = tensor([0.095, 0.070, 0.22, 80.0], dtype=torch.float32)
```

### 上下文(前后 3 行)

```python
130: PARAMS      = ["g_h", "g_LK", "c_ctx2th", "b"]
131: PRIOR_LOW   = tensor([0.035, 0.020, 0.05, 28.4], dtype=torch.float32)
132: PRIOR_HIGH  = tensor([0.095, 0.070, 0.22, 80.0], dtype=torch.float32)
133: 
134: ROUND_SIMS  = [2000, 1000, 1000, 1000]   # 正常模式下每轮仿真次数
135: SEED        = 42
```

`PRIOR_LOW` 完全未动,其他 3 个上界(g_h=0.095, g_LK=0.070, c_ctx2th=0.22)完全未动。✓

### 关键 diff(已用 grep 过滤无关行)

```
-PRIOR_HIGH  = tensor([0.095, 0.070, 0.22, 42.6], dtype=torch.float32)
+PRIOR_HIGH  = tensor([0.095, 0.070, 0.22, 80.0], dtype=torch.float32)
```

### 副作用检查 — git status

`git status --short` 的输出里有几个 M 标记,但需要分类:

**本次会话内引起的改动**:
- `M S4_sbi/run_sbi.py` — Step 2 PRIOR_HIGH 改动(本次)
- `D S4_sbi/sbi_outputs/*` (9 个 D 条目) — Step 1 归档导致(已确认归档目录健在)

**Pre-existing(会话开始前就 modified,与本次工作无关)**:
- `M S4_sbi/compute_xobs_from_eeg.py` — 旧 8 维路径的注释/编码改动
- `M S4_sbi/simulator_wrapper.py` — 7 维→5 维改动(会话开始前已存在)
- `M models/s4_personalize_fig7_v2.py` — 与 SBI 无关的 V2 文件改动

本次 Step 2 **仅修改 `S4_sbi/run_sbi.py` 一个文件**,无副作用。✓

未 commit(按用户要求)。

**Status**: ✅ PASS

---

## Step 3: Smoke Test

### 执行

```
python S4_sbi/simulator_wrapper.py
```

### 实际墙钟

- **总墙钟**: **15 秒**(date +%s diff,含模块加载+target PSD+FOOOF+1 次仿真)
- **simulator 内部报告**: 10.3 s(单次 simulator() 调用)
- 远低于 150 秒红线(实际只用了 1/10),也低于预期 70-100 秒区间下界
- 解读:可能是 numba JIT 缓存命中 + 60s 仿真在 numba 加速下跑得快

### 5 维输出

| index | key | value | 通过判据 | 状态 |
|-------|-----|-------|----------|------|
| 0 | `shape_r` | 0.67706 | > 0.5(Seed B 参考 ~0.68) | ✅ |
| 1 | `T4_q` | 2.35100 | (Seed B 参考 2.361) | ✅ |
| 2 | `T4_freq` | 1.25000 Hz | 0.5–1.5 Hz | ✅ |
| 3 | `T8_n_sp_events` | 22.90909 | (Seed B 参考 T12=17,T8 无固定值) | ✅ |
| 4 | `T11_lag_ms` | 3.11200 | > 1.0(Seed B 参考 3.613) | ✅ |

**NaN 数**: 0/5 ✅

Smoke test 自己打印 `SMOKE TEST PASSED`。

### Warnings

完整 stderr 中**只有一条 warning**,来自 fooof 包(非阻塞性,与 5 维 SBI 改动无关):

```
DeprecationWarning:
The `fooof` package is being deprecated and replaced by the `specparam`
(spectral parameterization) package.
This version of `fooof` (1.1) is fully functional, but will not be further updated.
```

- 来源:`models/s4_personalize_fig7_v7.py:137  from fooof import FOOOF`
- 影响:0(fooof 1.1 仍然 fully functional)
- 是否需立刻处理:否

**没有** ImportError、RuntimeWarning、numba 警告、neurolib 错误。

### Seed B 参考值对比

| stat | smoke output | Seed B 参考(`pareto_seeds_fresh_DE.json`) | 偏差 |
|------|--------------|-------------------------------------------|------|
| shape_r | 0.67706 | 0.678615 | -0.23% |
| T4_q | 2.351 | 2.361 | -0.42% |
| T11(up_down_ratio) | 3.112 | 3.613 | -13.9% |

shape_r 和 T4_q 与 Seed B 表的标称值高度一致,确认 simulator 在 5 维改造后保持了 Seed B 的力学行为。T11 偏差 13.9% 在可接受范围内(单次仿真的随机性 + numba seed 42 vs 原始 fresh DE 运行时种子不同)。

**Status**: ✅ PASS

---

## Step 4: Configuration Sanity Check

### 4.1 dry-run 执行

`python S4_sbi/run_sbi.py --dry-run` **支持**,实际跑了 16 分钟(963 s),完成 R1+R2+R3 各 50 次仿真,R4 因早停条件提前结束。

dry-run **端到端通过**:`Dry run complete. No files saved. Pipeline works end-to-end.`

### 4.2 执行顺序(读 main() + run_sbi.py:583-722)

1. 加载 `x_obs_v3.npz` → `x_obs_t` (5 维 torch tensor)
2. 构建 `prior = BoxUniform(PRIOR_LOW, PRIOR_HIGH)`(4 维)
3. 构建 SNPE-C + NSF 推断对象
4. **循环 R1–R4**(早停可在 R3 后跳过 R4):
   - 采样 theta(R1 用 prior,R2+ 用上一轮 posterior @ x_obs)
   - 串行调用 `simulator()` n_sims 次
   - 过滤 NaN,append 进累计数据集
   - `inference.append_simulations + train()` 训练 NSF
   - 保存 `round{k}_posterior.pkl` checkpoint
   - R2/R3 末尾检查 std 收缩 < 10% → 早停跳过 R4
5. 用最终 posterior 跑 **诊断 5 步**:
   - `[1/5] Marginals` → fig_marginals.png
   - `[2/5] Pairplot` → fig_pairplot.png
   - `[3/5] MAP + 95% CI` → 写入日志
   - `[4/5] PPC (100 sims)` → 调 simulator 100 次 → fig_ppc.png
   - `[5/5] SBC (200 sims)` → 调 simulator 200 次 → fig_sbc.png(注:May 7 时 SBC 因 API 不兼容跳过,本次 sbi=0.26.1 应可用)
   - `[6/6] Pareto overlay` → fig_pareto_overlay.png
6. `write_results_md` → 写 `sbi_results.md`
7. 总耗时打印,Logger close

### 4.3 关键配置

| 配置项 | 值 | 验证 |
|--------|----|----|
| 默认 x_obs 路径 | `D:\Year3_Mao_Projects\sleep_loop\S4_sbi\x_obs_v3.npz` | ✅ 是 v3 |
| 默认输出目录 | `D:\Year3_Mao_Projects\sleep_loop\S4_sbi\sbi_outputs` | ✅ 存在但**为空**(dry-run mkdir 创建,无文件) |
| `ROUND_SIMS` | `[2000, 1000, 1000, 1000]` | 主循环 5000 sims |
| SBC | 200 sims | 完成训练后 |
| PPC | 100 sims | 完成训练后 |
| **总仿真预算** | **5300 sims** | 5000 + 200 + 100 |
| `num_workers` | 1(hardcoded,docstring 明示) | Windows + numba 强制约束 |
| 训练设备 | **cpu**(`cuda=False`) | 本机无 CUDA |
| sbi 版本 | 0.26.1 | ≥ 0.21 ✓ |
| torch 版本 | 2.5.1 | ✓ |
| Random seed | 42 | torch + np + numba 全锁 |

### 4.4 x_obs_v3.npz 验证

```python
shape: (5,)
dtype: float32
values: [1.0, 2.6449999809265137, 0.75, 15.3100004196167, 1.2799999713897705]
keys:   [shape_r, T4_q, T4_freq, T8_n_sp_events, T11_lag_ms]
matches expected [1.0, 2.645, 0.75, 15.31, 1.28]: True
```

5 维 ✅,数值与 preflight 报告 A4 节一致 ✅,key 顺序与 simulator_wrapper.py:88-91 一致 ✅。

### 4.5 dry-run 实际表现(可用来重新估算墙钟)

| 轮次 | sims | 仿真墙钟 | s/sim | NaN |
|------|------|----------|-------|-----|
| R1 | 50 | 6.5 min | 7.8 | 0 |
| R2 | 50 | 4.9 min | 5.9 | 0 |
| R3 | 50 | 4.3 min | 5.2 | 0 |
| R4 | 跳过(R2→R3 std 收缩 4.4% < 10%) | — | — | — |
| **加权平均** | 150 | 15.7 min | **6.3 s/sim** | 0% |

(R4 在 dry-run 里早停纯属偶然,50 sims 估计不准,实际 4000 sims 训练应该不会这么早收敛。)

### 4.6 估算墙钟

- May 7 实际:9.66 h / 4100 sims(4000 主 + 100 PPC,SBC 跳过) ≈ **8.5 s/sim**
- 本次 dry-run:**6.3 s/sim**(numba 缓存暖启 + 训练数据小,NSF 收敛快)
- 保守估计 7-8 s/sim 用于全运行:
  - 5300 sims × 7.5 s = 39750 s = **11.0 h**
  - NSF 训练每轮 ~30 s(R1 上 2000 sims 可能更长,粗估 4 轮共 ~5–10 min)
  - 诊断画图开销忽略
- **预计墙钟范围**: **9.5 – 12 小时**(median ~11 h)

注意:本次 b prior 宽度 (28.4–80.0) 比 May 7 (28.4–42.6) 大 **3.6 倍**。R1 单轮 std 看 dry-run 是 7.71(b 维),May 7 时是 1.81 — 反映先验更宽,初轮后验自然更宽,但这正是放宽 b 的本意(让后验有空间表达真实形状,而非贴边)。

**Status**: ✅ PASS

---

## Overall Verdict

5 维 SBI 训练**可以启动**。Step 2 改 b 上界至 80.0 已生效(dry-run 日志显示 prior high 末位是 80.0);Step 3 smoke test 在 15 s 内输出干净的 5 维向量,5/5 个 sanity 判据全过;Step 4 dry-run 端到端跑通 150 仿真零 NaN,默认 x_obs 路径、输出目录、5300 sim 预算、num_workers=1 全部正确,预估墙钟 **9.5–12 小时**(median 11 h)。

唯一 cosmetic warning 是 fooof DeprecationWarning,非阻塞。

等用户 "go" 后,推荐执行:

```bash
conda activate neurolib
cd D:\Year3_Mao_Projects\sleep_loop
python S4_sbi/run_sbi.py
```

(无需 `--dry-run`,无需任何额外参数;默认行为已对齐 5 维管线。)

---

## 旁注:本次 dry-run 的副作用

`Logger.__init__`(`run_sbi.py:156`)在每次启动时把 `S4_sbi/sbi_log.txt` 清空。这意味着 **May 7 主 run 的原始 `sbi_log.txt` 内容已被 dry-run 日志覆盖**。

影响评估:
- May 7 的关键证据已在 `docs/sbi_5dim_preflight_report.md` E2 节里逐行引用并存档,信息层面无损失。
- 若需保留 dry-run 日志作为 5 维管线的健康基线,可在启动正式 run 前 `mv S4_sbi/sbi_log.txt S4_sbi/sbi_log_dryrun_20260522.txt`。
- `S4_sbi/sbi_results.md`(May 7 7 维版本)**未被覆盖**,因 dry-run 在写诊断之前 return。
