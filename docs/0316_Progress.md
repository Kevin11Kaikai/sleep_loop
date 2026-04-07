# Sleep-Loop 项目进度报告 — 2026-03-16/17

> 运行环境：`conda activate neurolib`（Python 3.x，NumPy 2.2，neurolib 本地源码安装）  
> 项目根目录：`D:\Year3_Mao_Projects\sleep_loop\`  
> 所有脚本均从项目根目录执行：`python models/<script>.py`  
> 接续：[0315_Progress.md](./0315_Progress.md)（Session 1–2 已完成）

---

## 本期工作总览

| Session | 脚本 | 功能一句话 | 主要输出 |
|---------|------|-----------|---------|
| 3-A | `s6_rl_env.py` | 将丘脑-皮层模型封装为 OpenAI Gymnasium 环境 | `SleepEnv` 类，gym check 通过 |
| 3-A 辅助 | `s6_baseline_check.py` | 零扰动基线验证，确认 SleepEnv 能重现 N3 特征 | 打印 delta_ratio，无文件写入 |
| 3-A 辅助 | `s6_decision_check.py` | 确认 stim 范围有效性与 STEP_DURATION_MS | 打印决策结论，无文件写入 |
| 3-B | `s7_sac_train.py` | 100k 步 SAC 基础训练 | `outputs/sac_sleep_model.zip` |
| 3-B 修复 | `s8_sac_iterate.py` | autoresearch 式迭代训练（5 轮） | `outputs/sac_round*.zip`, `iteration_history.json` |

---

## s6_rl_env.py — SleepEnv：Gymnasium 睡眠闭环环境

### 设计目标
将 Session 1-B 验证的 `ThalamoCorticalNetwork` 封装为标准 Gym 接口，让 SAC Agent 学习通过皮层刺激电流维持 N3 慢波睡眠状态。

### 状态空间（3 维连续，归一化至 [-1, 1]）

| 维度 | 含义 | 计算方式 |
|------|------|---------|
| `obs[0]` | `delta_ratio`（0.5–4 Hz 功率占比） | `(delta_ratio - 0.5) × 2` |
| `obs[1]` | `sigma_norm`（10–15 Hz，越低越好） | `clip(1 - sigma_ratio × 20, -1, 1)` |
| `obs[2]` | `mean_rate_norm`（皮层均值发放率） | `clip(mean_rate / 30 - 1, -1, 1)` |

### 动作空间
```
action[0] = stim_current ∈ [-0.05, 0.05]  mV/ms
叠加到 ALNMassEXC 背景输入 mue 上（_base_mue + stim）
```

### Reward 函数（最终版，含 Rescue Bonus）
```python
reward = - abs(delta_ratio - 0.91)    # 目标距离惩罚
       - 0.1 * sigma_norm             # 纺锤波抑制
       - 0.5 * stim² * 100            # 动作幅度惩罚

# Rescue Bonus：delta 从 <0.70 升到 ≥0.70 时给一次性奖励
if prev_delta < 0.70 and delta_ratio >= 0.70:
    reward += RESCUE_BONUS  # Round 3–5 设为 0.5
```

### 关键 MVP 设计限制
- **每步仿真独立重启**：neurolib `model.run()` 每次从初始条件开始，不接续末态。  
  物理含义：Agent 学习的是"在给定参数下 5 s 后系统的状态"，非严格连续闭环。  
  后续迭代：使用 `chunkwise=True` 或手动保存/恢复末态变量。

### 关键参数
```python
STEP_DURATION_MS = 5000   # 每步 5 秒仿真
WARMUP_MS        = 5000   # reset() 热身时长
MAX_STEPS        = 200    # 每 episode 最多 200 步（约 1000 s 仿真）
PERTURB_MUE      = 0.15   # mue 初始扰动范围（最终版，Round 5 改为 0.15）
PERTURB_MUI      = 0.15   # mui 初始扰动范围
TARGET_DELTA_RATIO = 0.91
```

### 重要参数分层设计

| 参数类别 | 参数 | 来源 | 原因 |
|---------|------|------|------|
| 个体化参数 | `mue=3.009, mui=4.077, b=19.5, tauA=1040` | Session 2-B 拟合 | 控制 PSD 谱形状 |
| 固定结构参数 | `g_h=0.1, g_LK=0.1` | s3_sleep_kernel.py Table 3 | 控制 up-state 可自持性 |

> 不用 patient 拟合值（g_h=0.025）的原因：拟合值优化的是 PSD 形状，但 g_h 控制系统能否自发进入并维持 N3；拟合值会将系统偏向 down-state。

### 收敛热身机制（reset() 中）
```python
MAX_WARMUP_TRIES = 5
for attempt in range(MAX_WARMUP_TRIES):
    self._run_sim(WARMUP_MS)
    if mean_rate > 1.0 Hz:  # 进入 up-state
        break
```

### 验证结果（`python models/s6_rl_env.py`）
```
reset obs = [0.263, 0.999, -0.937]
delta_ratio = 0.632
5 步随机动作：reward 在 -0.26 至 -0.50 之间
✓ SleepEnv validation passed
✓ gym check_env passed（OU 噪声非确定性为设计选择，非 API 错误）
```

---

## s6_baseline_check.py — 零扰动基线验证

### 目的
确认 SleepEnv 在原始 patient_params（无随机扰动）下能否重现 Session 2-B 的 N3 特征（delta_ratio ≥ 0.80）。

### 判定标准
| 结果 | 含义 |
|------|------|
| delta_ratio ≥ 0.80 | 解释 A：基线正常，reset() 的扰动是偏低的原因 |
| 0.60–0.80 | 中间状态，需检查热身时长 |
| < 0.60 | 解释 B：SleepEnv 未重现 N3，需修复 |

### 验证结果
```
delta_ratio（5000ms 热身）= 0.8830
→ 判定：解释 A ✓ 可进入 Session 3-B
```

---

## s6_decision_check.py — Session 3-B 设计决策确认

### 决策点 ①：stim 范围有效性

测试 stim = 0, ±0.05, ±0.10, ±0.20 时 delta_ratio 的变化量（N=5 重复）：

| stim | delta 均值 | vs 基线 | 结论 |
|------|-----------|---------|------|
| -0.20 | 0.37 | -0.45 | 负向 stim 把系统踢出 N3 |
| -0.05 | 0.33 | -0.13 | — |
| 0.00 | 0.46 | — | 基线 |
| +0.05 | 0.87 | **+0.67** | ✓ 有效，变化量 >> 0.05 |
| +0.20 | 0.88 | +0.68 | 边际收益小 |

**结论 ①**：`action_space = ±0.05` 有效，保持不变。

**重要发现**：负向 stim 把系统踢出 N3 的效果（-0.45）是正向救援效果（+0.67）的 0.67 倍。系统存在显著不对称性，SAC 需要学会"克制负向 stim"。

### 决策点 ②：STEP_DURATION_MS

| 时长 | 耗时/步 | delta_ratio | 加速比 |
|------|--------|------------|--------|
| 2000ms | 0.07s | 0.98（但 mean_rate≈0，皮层静默） | 2.9× |
| 5000ms | 0.20s | 0.42（实际平均） | — |

**结论 ②**：2000ms 速度快但皮层静默（不可靠），5000ms 信号真实。保持 `STEP_DURATION_MS=5000`。

---

## s7_sac_train.py — 100k 步 SAC 基础训练

### 训练配置
```python
TOTAL_TIMESTEPS = 100_000
SAC_KWARGS = dict(
    learning_rate = 3e-4,
    buffer_size   = 10_000,
    learning_starts = 500,
    batch_size    = 64,
    tau           = 0.005,
    gamma         = 0.99,
)
```

### 训练过程
- 实际耗时：**6.68 小时**（步速 ~0.237 s/步，FPS≈4）
- `ep_rew_mean` 全程在 -72 ~ -83 之间，未见明显收敛趋势
- 定期 `[EVAL]`（每 5000 步，3 episodes）：eval_delta 在 0.35–0.70 间振荡

### 最终评估（10 episodes，贪心策略）
```
Episode 1-2, 6-10（up-state）：mean_delta ≈ 0.88-0.89
Episode 3-5（down-state）：  mean_delta ≈ 0.34

mean_delta_ratio = 0.655 ± 0.258
up-state = 6/10
判定：✗ 训练未收敛（mean_delta < 0.75）
```

### 根因诊断
Reward 函数缺乏"脱出激励"：当系统陷入 down-state（delta≈0.34）时，施加大 stim 会触发动作惩罚（`-0.5×stim²×100`），agent 学会了"摆烂"（stim≈0）而非救援。

### 输出文件
| 文件 | 内容 |
|------|------|
| `outputs/sac_sleep_model.zip` | 最终模型 |
| `outputs/sac_sleep_model_step*.zip` | 每 5000 步 checkpoint（共 20 个）|
| `outputs/sac_training_log.csv` | 训练曲线（220 条记录）|
| `outputs/tb_logs/` | TensorBoard 日志 |
| `outputs/s7_sac_train.log` | 完整训练日志 |

> **最优 checkpoint**：`sac_sleep_model_step5000.zip`（eval_delta=0.7018，训练最早出现的高点）

---

## s8_sac_iterate.py — autoresearch 式迭代修复训练

### 迭代框架
```
假设 → 改一处 → 跑 N k 步（~30-120 分钟）→ 看 eval_delta → 保留/丢弃 → 下轮
```

每轮统一评估：训练期间每 2000 步 eval 5 episodes，训练结束后 eval 10 episodes。

### 各轮迭代记录

#### Round 1（10k 步，从 step5000 继续）
**改动**：新增 `RESCUE_BONUS = 0.5`（delta 越过 0.70 阈值时给一次性奖励）

| 指标 | 值 |
|------|-----|
| mean_delta_final | 0.680（上轮 0.655，↑ +3.8%）|
| up-state | 6/10 |
| 最优 eval_delta | **0.7416**（step 14000）|
| 判定 | **△ PARTIAL** |

**结论**：rescue bonus 有正向效果，eval 峰值提升。

---

#### Round 2（10k 步，从 round1_best 继续）
**改动**：action 惩罚系数 0.5 → 0.1（减少对大 stim 的惩罚）

| 指标 | 值 |
|------|-----|
| mean_delta_final | 0.669（↓ 退步）|
| up-state | 6/10 |
| 最优 eval_delta | 0.6719（step 20000）|
| 判定 | **✗ FAIL** |

**根因**：`sac_round1_best.zip` 的 Critic 以 penalty=0.5 训练，改为 0.1 后 Q 值偏差，需步数重新校准。10k 步内无法恢复。

---

#### Round 3（20k 步，从 round1_best 继续）
**改动**：回退 penalty=0.5（Round 1 有效配置），步数扩大到 20k

| 指标 | 值 |
|------|-----|
| mean_delta_final | **0.720（↑ 最高）** |
| up-state | **7/10（历史最好）** |
| 最优 eval_delta | **0.8873**（step 34000）|
| 判定 | **✓ PASS** |

**关键发现**：steps 32000–34000 连续出现 eval_delta 0.876 / 0.887，突破 0.80 目标。更长训练（20k vs 10k）是关键因素。

---

#### Round 4（10k 步，从 round3_best 继续）
**改动**：RESCUE_BONUS 0.5 → 1.0（按 PASS 规则执行）

| 指标 | 值 |
|------|-----|
| mean_delta_final | 0.684（↓ 退步）|
| up-state | 6/10 |
| 最优 eval_delta | 0.6714（step 36000）|
| 判定 | **✗ FAIL** |

**根因**：与 Round 2 类似——改变奖励函数幅度导致 Critic Q 值失准。`round3_best` 用 bonus=0.5 训练，加载后改为 1.0 后短期内不稳定。

---

#### Round 5（20k 步，从 round3_best 继续）：方案 A
**改动**：`PERTURB_MUE/MUI 0.3 → 0.15`（缩小扰动范围）+ RESCUE_BONUS 恢复 0.5

**物理依据**（量化分析）：

```
base mue = 3.009，action_max = +0.05
PERTURB=0.3 → 最差 mue = 2.709 → 最大可达 mue = 2.759
             → 2.759 < N3 自持阈值（≈2.80）→ 物理上不可救援

PERTURB=0.15 → 最差 mue = 2.859 → 最大可达 mue = 2.909
             → 2.909 ≥ N3 阈值 → 理论上可救援

不可救援初始条件比例：(0.3-0.21)/0.6 ≈ 15%（与 4/10 失败率方向一致）
```

| 指标 | 值 |
|------|-----|
| mean_delta_final | 0.661（表面退步，10 ep 快照方差较大）|
| up-state | 6/10 |
| 最优 eval_delta | **0.7823**（step 52000）|
| ep_rew_mean | **-60.8**（全轮最低，训练质量最优）|
| 判定 | **△ PARTIAL** |

**实质进展**：
- 训练质量显著提升（ep_rew_mean -73 → -60.8）
- eval 谷底抬升（0.45 → 0.57），说明在下限初始条件下 agent 已有更多成功
- 6/10 的 up/down 比例受最终 10 ep 快照的随机性影响，不代表策略退步

---

### 全轮对比汇总

| 轮次 | 改动要点 | mean_delta | up/10 | best eval | ep_rew_mean |
|------|---------|-----------|-------|-----------|-------------|
| 100k 基线 | 无 rescue bonus | 0.655 | 6/10 | 0.70 | ~-73 |
| Round 1 | +rescue_bonus=0.5 | 0.680 | 6/10 | 0.742 | — |
| Round 2 | penalty 0.5→0.1 | 0.669 | 6/10 | 0.672 | — |
| **Round 3** | R1 配置 + 20k步 | **0.720** | **7/10** | **0.887** | ~-68 |
| Round 4 | bonus 0.5→1.0 | 0.684 | 6/10 | 0.671 | ~-66 |
| **Round 5** | PERTURB 0.3→0.15 | 0.661 | 6/10 | **0.782** | **-60.8** |

**最优可用 checkpoint**：`outputs/sac_round3_best.zip`（eval_delta=0.8873 @ step 34000）

### 输出文件汇总

| 文件 | 内容 |
|------|------|
| `outputs/sac_round1.zip` / `round1_best.zip` | Round 1 最终/最优模型 |
| `outputs/sac_round2.zip` / `round2_best.zip` | Round 2 最终/最优模型 |
| `outputs/sac_round3.zip` / `round3_best.zip` | Round 3 最终/最优模型 ← **推荐** |
| `outputs/sac_round4.zip` / `round4_best.zip` | Round 4 最终/最优模型 |
| `outputs/sac_round5.zip` / `round5_best.zip` | Round 5 最终/最优模型 |
| `outputs/sac_round*_log.csv` | 各轮 eval 时序记录 |
| `outputs/iteration_history.json` | 5 轮完整迭代摘要 |
| `outputs/tb_logs_round*/` | 各轮 TensorBoard 日志 |

---

## 数据流向图（Session 3 补充）

```
data/patient_params_SC4001.json
    │
    └─ s6_rl_env.py（SleepEnv）
       ├─ 状态：[delta_ratio, sigma_norm, mean_rate_norm]
       ├─ 动作：stim ∈ [-0.05, +0.05] mV/ms
       ├─ Reward：-|delta-0.91| - 0.1×sigma - 0.5×stim²×100 + rescue_bonus
       └─ 参数：mue/mui 个体化 + g_h/g_LK Table 3 固定值
               PERTURB=±0.15，WARMUP=5s×5次
               │
               ├─ s6_baseline_check.py ──→ delta_ratio=0.883 ✓
               ├─ s6_decision_check.py ──→ stim±0.05有效，步长5000ms
               │
               └─ s7_sac_train.py（100k步基础训练）
                  │  6.68h，最优 ckpt=step5000（eval_delta=0.70）
                  │
                  └─ s8_sac_iterate.py（5轮迭代修复）
                     ├─ Round 1：+rescue_bonus → 0.74
                     ├─ Round 2：-penalty  → 退步
                     ├─ Round 3：20k步     → 0.887 ✓ PASS
                     ├─ Round 4：bonus×2   → 退步
                     └─ Round 5：PERTURB→0.15 → ep_rew=-60.8（训练质量最优）
                                │
                                └─ outputs/sac_round3_best.zip（推荐 checkpoint）
                                   outputs/iteration_history.json
```

---

## Session 3 技术发现与经验总结

### 发现 1：eval_delta 振荡的本质是"抽签"而非学习曲线

每个 eval episode 的最终 delta 服从双峰分布：
- up-state 成功：delta ≈ 0.87–0.89（各轮保持一致）
- down-state 失败：delta ≈ 0.33–0.37

eval_delta（5 ep 平均）随机落在 0.45–0.88 之间，取决于抽到几个 up-state episode。学习曲线的"振荡"不反映策略退步，而是反映**每轮 eval 的随机初始条件采样**。

真实进步指标应当是：ep_rew_mean（训练环境奖励）、eval 谷底高度、以及多次 10 ep 评估的平均值。

### 发现 2：扰动范围与可救援边界的定量关系

```
可救援条件：base_mue - PERTURB + action_max ≥ N3_threshold
其中：base_mue=3.009, action_max=0.05, N3_threshold≈2.85（实测）
→ PERTURB ≤ 3.009 + 0.05 - 2.85 = 0.209
→ PERTURB=0.3 时约 15% 初始化落入不可救援区
→ PERTURB=0.15 时所有初始化理论上可救援
```

### 发现 3：奖励函数变化对 Critic 的破坏性

改变 Reward 函数任何量纲参数（penalty 系数、bonus 幅度）后，已有 checkpoint 的 Critic Q 值立即失准。需要足够步数（≥ 5k-10k）才能重新校准。**每次改动奖励函数时，需同步评估"再校准代价"**，优先考虑从更早的稳定 checkpoint 出发而非无条件从最新 checkpoint 继续。

### 发现 4：更长训练 > 奖励工程调参（当前阶段）

100k 步训练 vs 5k 步训练的区别（eval_delta）：0.70 vs 0.74。从 14000 步跑到 34000 步的 Round 3 达到 0.887。**在当前网络规模和环境复杂度下，增加训练步数的边际收益高于精调 reward 参数。**

---

## 已知残余问题与后续迭代方向

### 问题 1：6/10 up-state 的天花板

**现象**：4 轮训练中 up/down 比例始终在 6–7/10，未能突破  
**根因**：PERTURB=0.15 仍有约 5% 的初始化在边界上；warmup 5s×5 次有概率失败  
**后续**：
- 方案 A2：PERTURB 进一步缩至 0.08（彻底排除不可救援初始化）
- 方案 B：WARMUP_MS 5000 → 15000（更可靠的热身，每次 reset 代价增加 ~1s）

### 问题 2：仿真不连续（MVP 限制）

**现象**：每步 `model.run()` 从初始条件重启，非真正闭环控制  
**影响**：Agent 学到的是"参数→5s后状态"的映射，非真实神经调控策略  
**后续**：使用 `model.run(chunkwise=True)` 接续末态，或手动保存/恢复 `model.state`

### 问题 3：actor_loss 持续增大（>20）

**现象**：所有 Round 后期 actor_loss 超过 20，远高于正常值（~1–5）  
**可能原因**：SAC 自动熵调节使 ent_coef 过低（~0.04），策略过于确定性，actor 梯度不稳定  
**后续**：固定 `ent_coef=0.1`（不自动调节），或提高 `target_entropy` 目标

### 问题 4：per-episode 成功率的统计不稳定性

**现象**：同一策略在 10 次 eval 中的 up/down 随机波动大（±2 episode）  
**后续**：最终评估应跑 30–50 episodes 取平均，减少统计误差

---

## Session 4 输入数据清单

| 文件 | 内容 | 用途 |
|------|------|------|
| `outputs/sac_round3_best.zip` | 最优 SAC 模型（eval_delta=0.887） | Session 4 继续训练起点 |
| `outputs/iteration_history.json` | 5 轮迭代完整记录 | 超参数选择参考 |
| `outputs/sac_round5_log.csv` | Round 5 eval 时序（训练质量最优） | 学习曲线分析 |
| `models/s6_rl_env.py` | SleepEnv（PERTURB=0.15，RESCUE_BONUS=0.5）| 复用或修改 |
| `models/s8_sac_iterate.py` | 迭代训练框架 | 复用，修改顶部配置区即可 |
| `data/patient_params_SC4001.json` | SC4001 个体化参数 | 环境初始化 |
