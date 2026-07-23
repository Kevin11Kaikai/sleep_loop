# V8 实施计划

## 范围

V8 是在 V7 基础上的受控审计补丁，不应被表述为“最终模型正确”。它要回答的问题是：同一个个体化丘脑-皮层神经质量模型，能否同时保留以下四件事：

- 皮层慢振荡结构
- 丘脑纺锤波真实性
- 慢振荡-纺锤波协调
- 皮层/类 EEG 纺锤波可观测性

在代码路径通过便宜的 smoke test 之前，不运行完整 differential evolution，不跑长时间仿真，也不跑 300s 验证。

## 1. 创建 V8 主脚本

1. 复制 `models/s4_personalize_fig7_v7.py` 为 `models/s4_personalize_fig7_v8.py`。
2. 重命名版本相关函数：
   - `compute_constraints_v7` 改为 `compute_constraints_v8`
   - `compute_fitness_v7` 改为 `compute_fitness_v8`
   - 所有打印标签从 `V7` 改为 `V8`
   - 顶部 docstring、Usage、Outputs 同步改为 V8
3. 更新输出路径：
   - `data/patient_params_fig7_v8_SC4001.json`
   - `outputs/evolution_fig7_v8_records.csv`
4. 除非下文明确说明，保留 V7 的模型构建、随机种子、60s 仿真时长和 T1-T12 生理检查。

## 2. 新增 T13：皮层纺锤波可观测性

新增硬约束 T13，必须在皮层信号 `r_ctx` 上测量。T13 的目标不是拟合真实 EEG 纺锤密度，而是先堵住 V7 的“丘脑有纺锤、皮层完全没有纺锤”的输出层漏洞。

建议检测流程：

1. 对 `r_ctx` 做 10-14 Hz sigma bandpass，先与 V7 的 T8/T12 保持一致。
2. 计算 Hilbert envelope。
3. 使用 V7 T8 相同的 Gaussian smoothing。
4. 使用 V7 T8 相同的 percentile threshold 作为训练约束的事件阈值。
5. 提取超过阈值的连续事件。
6. 过滤事件持续时间。
7. 对每个皮层事件窗口做 Welch PSD。
8. 要求事件内 sigma peak 大于 4-10 Hz non-sigma peak 的指定倍数。

批准后的初始常量：

```python
T13_CTX_EVT_MIN = 1
T13_CTX_VERIFIED_MIN = 1
T13_PEAK_INSIDE_RATIO = 1.5
T13_CTX_DUR_LO_S = 0.5
T13_CTX_DUR_HI_S = 3.0
```

建议 T13 pass 条件：

```python
t13 = (
    n_ctx_events >= T13_CTX_EVT_MIN
    and n_ctx_verified >= T13_CTX_VERIFIED_MIN
)
```

皮层纺锤密度必须记录，但不要作为硬约束接近真实 EEG 密度。density 只进入 reward 和后续审计解释。

需要记录的新字段：

- `T13`
- `T13_n_ctx_events`
- `T13_n_ctx_verified`
- `T13_ctx_density_per_min`
- `T13_ctx_verified_density_per_min`
- `T13_mean_ctx_dur`

## 3. 更新可行性逻辑

1. `n_passed` 从 T1-T12 改为 T1-T13。
2. `feasible` 从 `n_passed == 12` 改为 `n_passed == 13`。
3. `compute_feasibility_score` 增加第 13 项。
4. infeasible scoring 的 denominator 从 12 改为 13。
5. 所有 `/12` 显示改为 `/13`。
6. callback 输出增加 T13 状态和皮层 verified density。

建议 T13 soft score：

```python
if con["T13"]:
    scores.append(1.0)
else:
    scores.append(np.clip(
        min(
            con.get("T13_n_ctx_events", 0) / T13_CTX_EVT_MIN,
            con.get("T13_n_ctx_verified", 0) / T13_CTX_VERIFIED_MIN,
        ),
        0,
        0.99,
    ))
```

## 4. 更新 fitness reward

保留 V7 的 feasible-only reward 结构，但新增皮层可观测性 reward。

批准后的第一版权重：

```python
W_SHAPE = 0.40
W_SO = 0.20
W_THAL_SPINDLE = 0.15
W_CTX_SPINDLE = 0.25
```

建议 reward 定义：

```python
so_power = clip((T4_q - 1) / 4, 0, 1)
thal_spindle_power = clip(T12_n_verified / 15, 0, 1)
ctx_spindle_power = clip(T13_ctx_verified_density_per_min / 2.0, 0, 1)
```

这样 V8 仍然奖励皮层 SO 结构和丘脑纺锤真实性，同时避免 V7 中皮层纺锤完全不可见的漏洞。

## 5. 更新参数边界

第一版只建议改 `c_th2ctx`。

建议 V8 边界：

```python
c_th2ctx: [0.00, 0.075]
```

不要一开始就恢复 V6 的 `[0.05, 0.25]`。V7 诊断已经提示 `c_th2ctx >= 0.075` 可能破坏皮层 SO，因此 V8 应该先小幅扩展传播强度，而不是直接打开到高耦合区。

如果 V8a 找不到可行候选，后续再单独测试 V8b：`c_th2ctx = [0.005, 0.085]`。不要打开到 V6 的 `[0.05, 0.25]`。

其他参数边界第一版保持 V7 不变。

## 6. 更新记录和最终总结

V8 JSON/CSV 保持 V7 的 flat schema，不引入 nested `"params"` 结构。

CSV 记录应保留所有 V7 字段，并新增 T13 字段和重命名后的 reward 字段：

- `thal_spindle_power`
- `ctx_spindle_power`
- 可选兼容字段：`spindle_power = thal_spindle_power`

最终 validation summary 应报告：

1. `score`
2. `feasible`
3. `n_passed/13`
4. `shape_r`
5. `so_power`
6. `thal_spindle_power`
7. `ctx_spindle_power`
8. T1-T13 pass/fail 明细
9. 最优参数

summary 里要明确说明：T8/T12 检查的是丘脑纺锤真实性，T13 检查的是皮层/类 EEG 纺锤可观测性。

## 7. 验证脚本

等 `s4_personalize_fig7_v8.py` 能正常 import 后，再创建或更新验证脚本。

必须做：

1. 复制或泛化 `valid_scripts/audit_v1_v3_v7_t12.py`。
   - 导入 `s4_personalize_fig7_v8.py`
   - 使用 T1-T13
   - 如果已有 V8 JSON，则同时比较 V7 和 V8 selected points
2. 复制或更新 `valid_scripts/validate_spindle_density_confirm.py`。
   - 加入 V8
   - 保留 300s simulation
   - 保留 RMS detector variant
   - 保留 10-14 Hz 和 11-15 Hz 双 band
   - 输出标签应写成 post-hoc robustness，而不是完全 held-out validation
3. 复制或更新 `valid_scripts/validate_so_waveform_heldout.py`。
   - 加入 V8
   - 检查 V8 是否为了皮层纺锤可见性牺牲了 SO morphology

可选：

- 新增一个 V7 vs V8 皮层可观测性图：
  - cortical `r_ctx` time series
  - cortical sigma RMS 或 envelope
  - 检测到的 cortical spindle events
  - 对照显示 thalamic T8/T12 events

## 8. 绘图依赖

需要检查 `plot_scripts/plot_fig7_compare_v7_vs_v8.py` 后再决定是否复用。当前这个脚本主要用于 V7 和 warm-start PAC 结果对比，而且 spindle event detection 仍在 `r_thal` 上。

V8 审计补丁最好不要直接复用这个命名，以免把“旧的 V8/PAC warm-start”与“新的 V8 cortical observability audit patch”混淆。

建议二选一：

1. 把现有绘图明确标注为 PAC/warm-start 专用。
2. 新增独立的 V7 vs V8 cortical observability plot。

## 9. Full run 前的 smoke test

只做便宜检查：

1. 激活 neurolib 环境：`conda activate neurolib`。
2. 运行 `python -m py_compile models/s4_personalize_fig7_v8.py`。
3. 静态搜索：
   - 非预期的 V7 输出路径
   - 应改为 `/13` 但仍残留的 `/12` 显示
   - records 缺失 T13 字段
   - callback 缺失 T13
   - final validation 缺失 T13

在以上检查通过前，不运行：

- full DE
- expensive sweeps
- 300s validation

## 10. Full run 后的解释规则

如果 V8 找到 13/13 feasible candidate：

- 检查 SO morphology 是否仍然合理。
- 检查 300s RMS detector 下皮层 spindle density 是否非零。
- 检查 T13 是否只是训练 detector 上的过拟合。

如果 V8 失败：

- 不能马上说模型容量不足。
- 可能原因包括 T13 太严格、detector 不匹配、search budget 不够、`c_th2ctx` bounds 太窄、或 objective 仍不完整。

如果 V8 靠提高 `c_th2ctx` 恢复了皮层纺锤，但破坏 T4/T6：

- 这说明 SO 稳定性和皮层纺锤可观测性之间可能存在真实 trade-off。

如果 V8 通过 T13，但 300s RMS detector 仍失败：

- 应解释为 detector-transfer failure，并重新审视训练检测器和结果措辞。

## 待决定问题

1. T13 应该只是要求非零皮层 verified spindle，还是要求接近真实 EEG 密度？
2. T13 训练约束应使用 V7-style percentile Hilbert detector、RMS detector，还是两者都用？
3. `c_th2ctx` 上界应设为 0.075，还是更保守？
4. V8 应做完整 DE，还是从 V7 feasible solutions warm-start？
5. V8 JSON 应保持 V7 的 flat format，还是采用 warm-start 脚本里的 nested `"params"` format？
6. 引入 T13 后，300s RMS 检查应如何在论文中区分 post-hoc robustness 和 held-out validation？
