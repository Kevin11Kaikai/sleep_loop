# Compute_fitness_v3 设计解析（结合 N3 睡眠拟合背景）

## 1. 目标函数在做什么
在 `compute_fitness_v3` 里，核心目标是：

$$
\text{fitness}
= W_{\text{SHAPE}}\cdot \text{shape\_r}
+ W_{\text{SO}}\cdot \text{so\_power}
+ W_{\text{SP}}\cdot \text{spindle\_power\_eff}
+ W_{\text{DYN}}\cdot \text{dynamics\_score}
- \text{so\_overshoot\_penalty}
- \text{narrow\_spindle\_penalty}
$$

对应实现是：

```python
fitness = (W_SHAPE * shape_r
+ W_SO * so_power
+ W_SP * spindle_power_eff
+ W_DYN * dynamics_score
- so_overshoot_penalty- narrow_spindle_penalty)
```

这不是单纯“频谱拟合”，而是“频谱 + 动力学 + 反投机惩罚”的联合设计，目标是逼近真实 N3 生理机制，而不是仅仅拟合一条看起来相似的 PSD 曲线。

---

## 2. 为什么 N3 需要这种多项组合目标

N3 的生理特征不是单一峰值，而是多维现象：
- 慢振荡（SO）显著
- 皮层有明显 UP/DOWN 状态交替
- 纺锤活动存在且形态合理（waxing-waning，而非窄带极限环）

因此如果只拟合 PSD，很容易出现“谱像但机制不对”的解。
这也是 v3 引入更严格动力学测试的原因。

---

## 3. 分项设计动机与代码对应

### 3.1 数值稳定与无效解屏蔽
先做仿真可行性筛选，失败直接返回极差目标值：

```python
try:
    m = build_model(...)
    m.run()
except Exception:
    return BAD_OBJECTIVE
```

还做了两层过滤：
- burn-in 去除（前 5 秒），避免初始瞬态污染评估
- 近乎静默信号直接判坏，避免无意义候选继续参与比较

这保证优化器把预算花在“有生理意义、可比较”的候选上。

---

### 3.2 shape_r：为何用 FOOOF 残差相关而不是原始功率相关
核心流程：
1. 把模型 PSD 插值到与目标一致的 FOOOF 频率网格
2. 用 FOOOF 分离 aperiodic 成分
3. 对周期残差做 Pearson 相关

```python
shape_r, _ = pearsonr(sim_periodic[:n_r], target_periodic[:n_r])
shape_r = max(shape_r, 0.0)
```

这样做的意义：
- 更关注节律峰形状（SO/纺锤）是否匹配
- 降低 1/f 背景差异对评分的干扰
- 避免只靠整体功率缩放拿高分

当没有 FOOOF 时，才回退到加权 chi2 方案。

---

### 3.3 so_power 与 spindle_power：为何单独奖励峰强
在 FOOOF 峰参数里分别提取 SO 与 spindle 频段的峰功率：

```python
if SO_FREQ_LO <= freq <= SO_FREQ_HI:
    so_power = max(so_power, float(power))
if SPINDLE_LO <= freq <= SPINDLE_HI:
    spindle_power = max(spindle_power, float(power))
```

原因是 shape_r 偏向“形状一致性”，但生理上仍关心关键峰是否“够明显”，所以额外给低权重峰强奖励。

---

### 3.4 dynamics_score：防止机制性假阳性
`compute_dynamics_score_v3` 通过 T1-T5 子测试做“生理真实性守门”：
- T1：有 DOWN
- T2：有 UP
- T3：UP 持续时间足够
- T4：SO 主峰频率在合理区间
- T5：纺锤峰宽足够（排除窄带极限环）

这部分专门防止“频谱看起来不错，但实际动力学不符合 N3”的投机解。

---

### 3.5 spindle_gate：为什么纺锤奖励要门控
实现：

```python
spindle_gate = SPINDLE_GATE_ALPHA + (1.0 - SPINDLE_GATE_ALPHA) * dyn_details["T5_spindle_score_cont"]
spindle_power_eff = spindle_power * spindle_gate
```

其含义是：
- 峰宽质量越好，纺锤奖励越接近全额
- 峰宽不足时，仍保留少量奖励（由 `SPINDLE_GATE_ALPHA` 决定），避免优化梯度完全断掉

这是将硬约束转成可优化“软引导”的关键技巧。

---

### 3.6 双惩罚：为什么还要再扣分

#### SO overshoot penalty

```python
so_overshoot = max(so_power - SO_TARGET_MAX, 0.0)
so_overshoot_penalty = SO_OVERSHOOT_LAMBDA * so_overshoot
```

作用：防止模型把 SO 峰无限抬高来刷分，抑制非生理超强慢波。

#### narrow spindle penalty

```python
narrow_spindle_penalty = LAMBDA_NARROW * (
    NARROW_PENALTY_FLOOR + (1.0 - NARROW_PENALTY_FLOOR) * (1.0 - T5_score_cont)
) * spindle_power
```

作用：对“纺锤很强但很窄”的投机解额外扣分，与 spindle_gate 形成组合约束：
- 门控减少奖励
- 惩罚进一步扣分

两者一起把优化方向拉回真实 waxing-waning spindle。

---

### 3.7 权重结构背后的策略
当前权重：
- `W_SHAPE = 0.45`
- `W_DYN = 0.35`
- `W_SO = 0.10`
- `W_SP = 0.10`

可解读为：
- 主目标：频谱形状 + 动力学真实性
- 次目标：SO 与纺锤峰强微调
- 再叠加惩罚项，避免“强峰刷分”和“窄峰投机”

这与 N3 拟合需求一致：先保证“像 N3 状态”，再追求“峰值漂亮”。

---

## 4. 为什么最后 return -fitness
因为 `scipy.optimize.differential_evolution` 默认做最小化，所以用：

```python
return -fitness
```

将“最大化 fitness”转成“最小化 -fitness”。

---

## 5. 一句话总结
`compute_fitness_v3` 的本质是“生理约束的多目标标量化”：
它不仅拟合 N3 的谱，还强制出现真实的 UP/DOWN 与纺锤形态，并通过门控与惩罚系统性抑制优化器投机路径，从而提高参数解的机制可信度。
