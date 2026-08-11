# Route-3 7D Formal Validation Handoff

创建时间：2026-07-28  
实验：`route3_7d_fixed_c_ctx2th_formal_v1`  
最终裁决：**NO-GO**

## 科学边界

本实验只检验：在固定 `c_ctx2th` 后，其余七个参数能否在 **synthetic cortical population firing-rate observable space** 中被恢复。模拟输出不是 EEG。本实验没有解决 cortical source 到 Fpz-Cz 的 measurement/forward-model blocker，因此真实 EEG inference 仍为 NO-GO。

## 冻结合同

- 七参数顺序：`mue, mui, b, tauA, g_LK, g_h, c_th2ctx`
- 固定参数：`c_ctx2th = 0.1253491302153237`
- 固定值规则：Notebook 10 明确运行的 canonical `V8a local best`
- 来源：`outputs/v8a_ultra_narrow_t6_t13_search/best_so_far.json`
- 来源 SHA-256：`BF18212EBAC0EBEC39D4F13AECD5B7A9DA5618FFE78ABB5F8694A67F569AF76A`
- 旧 8D prior 中 `c_ctx2th` 范围：`[0.05, 0.22]`
- 14D schema hash：`78d0da704422a03917ef4c68207f5e8e8213b0c6f6279deaf26f34c9958d64a8`
- preregistration SHA-256：`2cedb6a68de307ab2f9233b4009a8f58854316259b6c185aeb61af5efdcbc363`
- held-out criteria SHA-256：`49a4f9590466a6d9ba83f83ee5b6783e268ba996f8e8fc7328e261fb0405edee`

Source JSON 与只读 locked copy 全程 hash 一致。新旧 simulator seed 交集为 0。新训练/held-out 与旧 8D 训练/held-out 的完整 8D exact duplicates 和忽略固定维后的 7D exact duplicates 均为 0；最近 7D prior-normalized RMS 距离最小值为 0.0343，没有样本因近邻审计被删除。

14D inference features 保持 Notebook 10–14 原定义和顺序：

```text
r_exc_mean_hz
r_exc_std_hz
r_exc_relative_so_power
r_exc_relative_sigma_power
r_exc_so_peak_frequency_hz
r_exc_sigma_peak_frequency_hz
r_exc_spectral_entropy_0p5_20
r_inh_mean_hz
r_inh_std_hz
r_inh_relative_so_power
r_inh_relative_sigma_power
r_inh_so_peak_frequency_hz
r_inh_sigma_peak_frequency_hz
r_inh_spectral_entropy_0p5_20
```

## Notebook 15：独立稳健性

```text
prior-wide = 128 theta x 3 seeds = 384
local multiscale = 387
total = 771
valid = 771
failed = 0
sum simulator runtime = 4372.91 s
observed stage wall time = 747.3 s
```

1%、2%、5% 尺度下的 27 个 Jacobian 全部 effective rank 7。Condition number：

| Scale | Median | Max |
|---:|---:|---:|
| 1% | 19.75 | 44.73 |
| 2% | 21.93 | 32.54 |
| 5% | 18.95 | 56.01 |

工程 hard gate 通过，但科学警告明显：

- 三 seed nearest-neighbor 全一致率仅 25.8%，pairwise 均值 41.7%；
- EXC/INH sigma-peak SNR 仅 1.66/2.31；
- SO peak 每个 seed 仅有 5 个离散值，sigma peak 仅 12/14 个；
- 128 点审计发现 2,106 个 collision 和 3,869 个 near collision；
- `g_LK/g_h` Jacobian cosine 随 center、seed、scale 变化很大，5% 尺度 median 0.711、范围 `[-0.889, 0.953]`。

局部 full rank 没有证明 global injectivity。

## Notebook 16：独立 4,096 Bank

```text
scheduled = 4096
valid = 4096
failed = 0
train = 3276
validation = 820
sum simulator runtime = 22935.91 s
observed wall time = 3879.1 s
bank SHA-256 = 9862dc986314be3a7c940548f79cba821f1b5bdb19c055a243c7186e3f8585d3
split SHA-256 = 897f20ebd296f056a033a2fb0c64b1398a6f8852355039dbbdd92590706de7a2
```

全部 4,096 个 simulator seeds 唯一。Scaler 只在 3,276 个 training rows 上拟合。

高分辨率 collision audit：

```text
all pairs = 8,386,560
far-theta pairs = 7,878,727
inside independent seed-noise floor = 2,192,916
near collisions = 3,759,596
collision fraction among far pairs = 27.83%
```

这是最终 NO-GO 解释中的关键 global-identifiability warning。

## Notebook 17：三成员 NPE Ensemble

三个模型沿用 Notebook 13 的 MAF architecture 和 training policy，只把 theta dimension 从 8 改为 7。

| Seed | Best epoch | Best validation loss | Runtime |
|---:|---:|---:|---:|
| 7330001 | 290 | -10.1513 | 94.44 s |
| 7330002 | 287 | -10.2580 | 90.62 s |
| 7330003 | 299 | -9.7485 | 88.51 s |

设备为 CPU。Validation-only posterior samples 的 finite/in-prior rate 均为 100%。Ensemble 使用等数量 mixture sampling，不学习权重。

## Notebook 18：256-case Held-out

```text
held-out attempted = 256
valid = 256
failed = 0
training theta exact duplicates = 0
training simulator-seed overlaps = 0
posterior samples = (256, 4096, 7)
posterior finite/in-prior = 100%
```

### Parameter-wise Results

`Contraction` 定义为 `1 - median 90% CI width / prior width`。

| Parameter | Prior error | Posterior error | Improvement | Rank correlation | 80% coverage | 90% coverage | Contraction | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| mue | 0.2500 | 0.1467 | 41.3% | 0.774 | 0.672 | 0.812 | 49.4% | recovery improves; calibration fails |
| mui | 0.2500 | 0.0049 | 98.0% | 0.999 | 0.902 | 0.973 | 98.3% | strong recovery; overcoverage |
| b | 0.2500 | 0.0879 | 64.8% | 0.908 | 0.746 | 0.871 | 68.8% | strong recovery; 80% calibration fails |
| tauA | 0.2500 | 0.1863 | 25.5% | 0.593 | 0.676 | 0.824 | 34.0% | weakest recovery; calibration fails |
| g_LK | 0.2500 | 0.1512 | 39.5% | 0.693 | 0.703 | 0.820 | 50.1% | recovery improves; calibration fails |
| g_h | 0.2500 | 0.1256 | 49.8% | 0.822 | 0.727 | 0.836 | 58.4% | recovery improves; calibration fails |
| c_th2ctx | 0.2500 | 0.1453 | 41.9% | 0.768 | 0.691 | 0.832 | 49.6% | recovery improves; calibration fails |

七个参数的 posterior-median MAE 均优于 prior-median baseline，整体 MAE 改善 51.6%。但 **0/7** 参数同时满足 80% 和 90% Wilson coverage compatibility，因而 contraction-with-coverage 也是 **0/7**。`b` 只有 90% level compatible；`mui` 明显 overcovered；其余多为 undercoverage。按预注册 severe-undercoverage 定义没有参数越过 15 percentage-point 门槛，但整体 calibration 仍系统性不合格。

所有 ensemble SBC 10-bin descriptive p-values 均小于 0.05。该结果与 coverage、bias、contraction 联合解释，不使用单一 p-value 作裁决。

Ensemble disagreement gate 通过：

```text
mean member-median range = 0.0427 prior width
largest parameter-level mean range = 0.0597 prior width
```

`g_LK/g_h` held-out posterior Spearman ridge 的 median 为 0.090，范围为 `[-0.511, 0.665]`。它不是所有 cases 中固定方向的 ridge，但与局部方向不稳定和高 global-collision rate 一致，仍是重要非线性混淆风险。

## Synthetic PPC

64 个 preregistered cases 包括 32 个固定随机 cases 和 32 个从其余 cases 中按 normalized recovery score 选择的 worst cases；两组互斥。

```text
posterior predictive = 1024, failed = 0
prior predictive = 1024, failed = 0
features better than prior = 14/14
posterior scaled error = 0.0996
prior scaled error = 0.5951
overall PPC improvement = 83.3%
```

PPC 很强，但不能覆盖 calibration 失败和 27.83% global collision warning。

## 唯一裁决

```text
FINAL DECISION = NO-GO
```

通过：

- preflight/training/held-out failure gates；
- posterior finite/prior-support gate；
- 7/7 point-recovery improvement；
- overall recovery improvement；
- no prior-level-unrecoverable parameter gate；
- 14/14 PPC feature gate；
- overall PPC gate；
- ensemble disagreement gate。

失败：

- coverage-compatible parameters：`0/7`，要求至少 `6/7`；
- contraction-with-reasonable-coverage：`0/7`，要求至少 `6/7`；
- 4,096-bank far-pair collision fraction 为 27.83%；
- SBC 对所有参数给出分布不一致警告。

因此不能声称七参数已获得 adequately calibrated uncertainty，也不能给出 Formal GO 或 Conditional GO。

## Claim Boundaries

允许：

- 七个点估计在固定 `c_ctx2th` 条件下均优于 prior-median baseline；
- 14D synthetic PPC 明显优于 prior predictive；
- 工程 pipeline、checkpoint、posterior sampling 和 artifact contract 成功。

禁止：

- 七参数已正式 validated；
- posterior uncertainty 已校准；
- cortical firing rate 是 simulated EEG；
- 真实 Fpz-Cz EEG parameter inference；
- measurement/forward-model blocker 已解决；
- 自动扩大 bank 或根据 held-out 结果调参。

## 文件与验证

Notebooks：

```text
S4_sbi/notebooks/15_Route3_7D_Preregistration_and_Robustness.ipynb
S4_sbi/notebooks/16_Route3_7D_4096_Simulation_Bank.ipynb
S4_sbi/notebooks/17_Route3_7D_SNPE_Ensemble.ipynb
S4_sbi/notebooks/18_Route3_7D_Heldout_Recovery_Coverage_PPC.ipynb
```

核心模块：

```text
S4_sbi/src/sleep_sbi/route3_7d_experiment.py
S4_sbi/src/sleep_sbi/route3_7d_training.py
S4_sbi/src/sleep_sbi/route3_7d_validation.py
```

配置与锁：

```text
S4_sbi/configs/route3_7d_preregistered_v1.json
S4_sbi/artifacts/route3_7d_preregistered_v1.locked.json
S4_sbi/artifacts/route3_7d_preregistered_v1.sha256
```

Results：`S4_sbi/results/route3_7d_formal_validation/`

验证：

```text
neurolib Python = C:\Users\YUS190\AppData\Local\anaconda3\envs\neurolib\python.exe
Python = 3.10.20
sbi = 0.26.1
torch = 2.5.1
Notebooks executed = 19/19 code cells
Notebook error outputs = 0
tests = 8 passed
JSON reload = 17
CSV reload = 20
NPZ reload without pickle = 7435
PyTorch checkpoints reload = 6
unexpected object arrays = 0
HTML exports = 4
visual checks = passed
```

Notebook 00–14 的 hashes 保持与本轮开始时记录一致。本轮未 commit、未 push。
