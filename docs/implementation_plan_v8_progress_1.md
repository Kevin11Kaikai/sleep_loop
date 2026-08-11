# V8 实施进度 1

## 当前状态

V8a 已完成第一阶段实现。

当前定位保持不变：

> V8a = V7 + cortical spindle observability audit patch。

V8a 不是最终模型正确性的证明，而是针对 V7 残余漏洞的受控审计补丁。它保留 V7 的 T1-T12 生理检查，并新增 T13，用于确认皮层/类 EEG 信号 `r_ctx` 上至少存在可观测的纺锤波事件。

## 已完成文件

1. `models/s4_personalize_fig7_v8.py`
   - 从 V7 主脚本复制并修改为 V8a。
   - 输出路径更新为：
     - `data/patient_params_fig7_v8_SC4001.json`
     - `outputs/evolution_fig7_v8_records.csv`

2. `docs/implementation_plan_v8.md`
   - 已按批准后的修订同步。
   - 明确 T13 是 permissive nonzero observability gate。

## 已实现的 V8a 逻辑

### 1. T13 皮层纺锤波可观测性

新增 T13，检测对象是 cortical `r_ctx`，不是 thalamic `r_thal`。

T13 当前硬条件：

```python
t13 = (
    n_ctx_events >= T13_CTX_EVT_MIN
    and n_ctx_verified >= T13_CTX_VERIFIED_MIN
)
```

当前常量：

```python
T13_CTX_EVT_MIN = 1
T13_CTX_VERIFIED_MIN = 1
T13_CTX_DUR_LO_S = 0.5
T13_CTX_DUR_HI_S = 3.0
T13_PEAK_INSIDE_RATIO = 1.5
```

T13 的检测流程：

1. 对 `r_ctx` 做 10-14 Hz bandpass。
2. 计算 Hilbert envelope。
3. 使用 V7 T8 同源的 Gaussian smoothing。
4. 使用 percentile threshold 检测事件。
5. 使用 0.5-3.0s 皮层 spindle-style duration bounds。
6. 对每个 cortical event 做 Welch PSD。
7. 要求事件内 sigma peak / non-sigma peak 大于 1.5。

### 2. Density 计算

皮层事件密度使用 post-burn-in 后的真实信号长度：

```python
duration_min = len(r_ctx) / fs / 60.0
```

这避免了错误使用 nominal 60s duration，因为 fitness path 已经丢弃前 5s burn-in。

T13 density 只记录并用于 reward，不作为 hard constraint。

### 3. Feasibility

V8a 从 12 约束改为 13 约束：

```python
details["feasible"] = (n_passed == 13)
```

`compute_feasibility_score` 已新增 T13 soft score。T13 soft score 只反映是否接近非零事件和非零 verified 事件，不要求接近真实 EEG 密度。

infeasible scoring denominator 已从 12 改为 13。

### 4. Reward

V8a 使用批准后的 reward 权重：

```python
W_SHAPE = 0.40
W_SO = 0.20
W_THAL_SPINDLE = 0.15
W_CTX_SPINDLE = 0.25
```

当前 reward 定义：

```python
so_power = clip((T4_q - 1) / 4, 0, 1)
thal_spindle_power = clip(T12_n_verified / 15, 0, 1)
ctx_spindle_power = clip(T13_ctx_verified_density_per_min / 2.0, 0, 1)
```

保留了 `spindle_power = thal_spindle_power` 作为 V7-era 兼容别名。

### 5. Bounds

V8a 使用保守 `c_th2ctx` 扩展：

```python
c_th2ctx = [0.00, 0.075]
```

未打开到 V6 的 `[0.05, 0.25]`。

## 已更新的输出字段

CSV/JSON 仍保持 V7 flat schema，不引入 nested `"params"`。

新增或保留的关键字段：

- `thal_spindle_power`
- `ctx_spindle_power`
- `spindle_power`，作为兼容别名
- `T13`
- `T13_n_ctx_events`
- `T13_n_ctx_verified`
- `T13_ctx_density_per_min`
- `T13_ctx_verified_density_per_min`
- `T13_mean_ctx_dur`

## 已更新的显示逻辑

已更新：

- callback 中的 `/13`
- callback 中的 `T13`
- callback 中的 `ctxsp`
- callback 中的 `ctxver`
- callback 中的 `ctxdens`
- final validation summary 中的 `/13`
- final validation summary 中的 T13 pass/fail 明细
- final validation summary 中的 `thal_spindle_power`
- final validation summary 中的 `ctx_spindle_power`

## 已执行验证

### 1. py_compile

已在显式 conda `neurolib` 激活路径下通过：

```bash
conda.bat activate neurolib && python -m py_compile models\s4_personalize_fig7_v8.py
```

说明：PowerShell 中裸 `conda activate neurolib` 不可用，因此使用了本机显式路径：

```text
C:\Users\YUS190\AppData\Local\anaconda3\condabin\conda.bat
```

### 2. 静态搜索

已检查并通过：

- 无非预期 V7 输出路径残留。
- 无应改为 `/13` 但仍残留的 `/12`。
- T13 已出现在 constraint details。
- T13 已出现在 records 字段。
- T13 已出现在 callback。
- T13 已出现在 final validation summary。

## 未执行内容

按要求，当前没有运行：

- full differential evolution
- expensive sweeps
- 300s spindle-density validation
- post-hoc long simulation

## 当前注意事项

1. `py_compile` 产生了 V8 的 `__pycache__` 缓存文件，这是编译检查副产物。
2. 当前 V8a 只实现主脚本和文档计划，尚未创建 V8 专用 validation scripts。
3. T13 使用 V7-style percentile/Hilbert detector 作为训练约束；后续 300s RMS detector 仍应作为 post-hoc robustness 检查，而不是完全 held-out validation。
4. 如果 V8a 后续找不到 13/13 feasible candidate，再考虑 V8b 的 `c_th2ctx = [0.005, 0.085]`。
5. 不应把 V8a 结果表述为最终模型正确性，只能表述为对 V7 输出层漏洞的审计补丁。

## 下一步建议

1. 人工 review `models/s4_personalize_fig7_v8.py` 的 T13 检测逻辑。
2. 确认是否接受 V7-style percentile detector 作为训练约束。
3. 在批准后再决定是否运行完整 V8a DE。
4. DE 结束后再创建或更新：
   - V8 T1-T13 audit script
   - V8 300s RMS spindle-density confirm script
   - V7 vs V8 cortical observability plot

---

## 第二部分：V8a 代码审查与便宜 sanity test

本部分追加记录一次 full DE 之前的 targeted code review 和便宜测试。没有运行 full DE，没有运行 300s validation。

## 代码审查位置

关键实现位置如下：

- T13 常量：`models/s4_personalize_fig7_v8.py` 第 247-255 行。
- `compute_constraints_v8`：第 333 行开始。
- T13 cortical detector block：第 617-680 行。
- full feasibility：第 683-686 行，`n_passed` 包含 T1-T13，`feasible = (n_passed == 13)`。
- `compute_feasibility_score` 的 T13 soft score：第 790-796 行。
- `compute_fitness_v8` reward 说明和实现：第 876-905 行、第 986-1017 行。
- records 字段：第 1020-1065 行。
- callback T13 显示：第 1097-1119 行。
- final validation summary：第 1230-1275 行。

当前没有单独抽出 T13 helper function；T13 检测逻辑是直接内联在 `compute_constraints_v8` 中。后续如果 validation script 需要复用同一检测器，可以再提取 helper，但本轮没有做结构重构。

## 审查结论

1. T13 使用的是 `r_ctx`：
   - T13 block 中 bandpass、Hilbert envelope、事件窗口和 Welch PSD 都使用 `r_ctx`。
   - 关键位置：第 626-629 行、第 645-646 行。

2. T8/T12 仍保持丘脑定义：
   - T7/T8 的 envelope 来自 `r_thal`。
   - T12 事件内 Welch 验证使用 `event = r_thal[s:e]`。
   - 本轮没有改变既有 thalamic T8/T12 定义。

3. T13 density 使用 post-burn-in 后的实际长度：
   - 当前实现为：
     ```python
     duration_min = len(r_ctx) / fs / 60.0
     ```
   - 因为进入 `compute_constraints_v8` 的 `r_ctx` 已经在 fitness path 中丢弃 5s burn-in，所以这里不会误用 nominal 60s。

4. T13 不把接近真实 EEG 密度作为 hard constraint：
   - hard pass 条件仅为：
     ```python
     n_ctx_events >= T13_CTX_EVT_MIN
     n_ctx_verified >= T13_CTX_VERIFIED_MIN
     ```
   - 当前 `T13_CTX_EVT_MIN = 1`，`T13_CTX_VERIFIED_MIN = 1`。
   - density 只记录，并作为 `ctx_spindle_power` reward 的输入。

5. full feasibility 已改为 13 约束：
   - `n_passed` 包含 `t13`。
   - `details["feasible"] = (n_passed == 13)`。

## Detector-only synthetic sanity test

该测试不导入 neurolib，只复制 T13-like detector 逻辑，用合成信号检查检测器基本方向是否正确。

### Synthetic A：慢振荡 + 噪声，无 12 Hz burst

期望：没有 verified cortical spindle，或者至少 peak verification fail。

实际结果：

```text
T13 = False
n_ctx_events = 9
n_ctx_verified = 0
ctx_density_per_min = 9.818
ctx_verified_density_per_min = 0.0
mean_ctx_dur = 0.847 s
max sigma/non-sigma ratio = 0.135
```

解释：percentile envelope threshold 在噪声中仍可能产生候选事件，这是预期风险；但事件内 peak verification 全部失败，`n_ctx_verified = 0`，因此 T13 fail。这个结果符合设计意图。

### Synthetic B：慢振荡 + 两个约 1s 的 12 Hz bursts

期望：检测到 cortical sigma events，并有 verified events。

实际结果：

```text
T13 = True
n_ctx_events = 8
n_ctx_verified = 2
ctx_density_per_min = 8.727
ctx_verified_density_per_min = 2.182
mean_ctx_dur = 1.046 s
max sigma/non-sigma ratio = 4.613
```

解释：两个注入的 12 Hz burst 均通过事件内 sigma peak verification。检测器方向正确。

## 单次 V7 best 参数的 V8 constraint evaluation

已发现 V7 best JSON：

```text
data/patient_params_fig7_v7_SC4001.json
```

只运行了一次 V8 constraint evaluation，没有运行 DE。运行环境使用 `conda.bat activate neurolib`，并导入本地 patched neurolib 路径。

结果：

```text
n_passed = 12
feasible = False
failed = [T13]
T1_T12_passed = 12
T8_n_sp_events = 25
T12_n_verified = 20
T13 = False
T13_n_ctx_events = 1
T13_n_ctx_verified = 0
T13_ctx_density_per_min = 1.091
T13_ctx_verified_density_per_min = 0.0
T13_mean_ctx_dur = 0.505 s
T4_q = 4.433
T6_ibi_cv = 0.197
T9_mi = 0.01902
T10_phase = 0.524
T11_lag_ms = 1.493
```

科学解释：

- V7 best 在 V8a 中仍保留 T1-T12 的 V7 行为，12/12 通过。
- 新增 T13 失败，因为 cortical candidate event 虽然有 1 个，但 verified cortical spindle 为 0。
- 这正符合 V8a 的目标：V7 的 thalamic spindle audit 通过，但 cortical/EEG-like spindle observability 不足。

## 本轮结论

1. T13 的信号来源、hard pass 条件、density 计算和 feasibility 逻辑与批准方案一致。
2. 合成测试显示：无 12 Hz burst 不会通过 peak verification；有 12 Hz burst 可以产生 verified cortical spindle。
3. V7 best 单点评估显示：T1-T12 保持通过，T13 单独失败，符合预期科学行为。
4. 没有发现需要立即修改阈值的 clear bug。
5. 当前仍不建议运行 full DE，除非先人工 review 并确认接受当前 V7-style percentile/Hilbert detector 作为 V8a 训练约束。

---

## 第三部分：V8a full DE 启动尝试记录

本部分记录一次按批准方案启动 V8a full DE 的尝试。

## 执行命令

按要求只运行 V8a 主脚本，并将 terminal output 写入指定 log：

```text
cmd /c "C:\Users\YUS190\AppData\Local\anaconda3\condabin\conda.bat activate neurolib && python -u models\s4_personalize_fig7_v8.py > outputs\v8a_full_de_run_log.txt 2>&1"
```

没有运行：

- 300s validation
- SO morphology validation
- V8b
- 任何阈值、边界、reward 或科学 framing 修改

## 运行结果

运行没有进入 full DE。脚本在启动阶段打印 T10 说明时因 Windows terminal 编码失败而退出。

状态：

```text
run completed: no
exit code: 1
wall time: about 7 seconds
log file: outputs/v8a_full_de_run_log.txt
V8 JSON created: no
V8 CSV created: no
```

## Traceback

log 中记录的错误为：

```text
Traceback (most recent call last):
  File "D:\Year3_Mao_Projects\sleep_loop\models\s4_personalize_fig7_v8.py", line 1284, in <module>
    main()
  File "D:\Year3_Mao_Projects\sleep_loop\models\s4_personalize_fig7_v8.py", line 1154, in main
    print(f"  T10 preferred phase near 0 OR ±π within ±50° "
UnicodeEncodeError: 'gbk' codec can't encode character '\xf6' in position 71: illegal multibyte sequence
```

## 当前解释

这是运行环境/终端编码错误，不是 DE 搜索结果，也不是模型可行性结果。脚本在打印启动说明时失败，尚未加载目标 EEG、尚未开始 differential evolution，因此没有 best score、feasible flag、best parameters 或 CSV 汇总可报告。

## 输出文件状态

已创建：

```text
outputs/v8a_full_de_run_log.txt
```

未创建：

```text
data/patient_params_fig7_v8_SC4001.json
outputs/evolution_fig7_v8_records.csv
```

## 后续

按照用户指令，遇到 runtime error 后停止并报告 traceback。本轮没有自动修改代码，也没有自动更换运行方式继续执行。是否通过设置 UTF-8 环境变量、调整 shell 编码，或修改打印字符串来继续运行，需要用户明确批准。

---

## 第四部分：V8a full DE UTF-8 重跑结果

本次按批准方案只修正运行环境编码问题，没有修改代码、阈值、边界、reward 权重、T13 逻辑或科学 framing。运行命令使用 `PYTHONIOENCODING=utf-8`，输出写入：

```text
outputs\v8a_full_de_run_log.txt
```

未运行：

- 300s validation
- SO morphology validation
- V8b
- 任何阈值、边界、reward 或科学逻辑调整

## 运行完成状态

V8a full DE 已完成并写出日志中的最终验证摘要：

```text
Evolution complete in 12.20 h
Saved: data/patient_params_fig7_v8_SC4001.json
Saved: outputs/evolution_fig7_v8_records.csv
```

产物状态：

```text
V8 JSON created: yes
data\patient_params_fig7_v8_SC4001.json

V8 CSV created: yes
outputs\evolution_fig7_v8_records.csv
```

注意：日志和产物已完成写出后，原 Python PID 34580 仍显示存活并继续占用 CPU。这里不将其解释为新的 DE 结果，也未手动终止进程；需要后续人工决定是否清理该进程。

## Best 结果

```text
best score = -0.421538
feasible = 0
n_passed = 12/13
failed constraints = T6
wall time = 12.20 h
```

Best parameters：

```text
mue = 4.151310357840672
mui = 2.719275889607868
b = 32.472799040782654
tauA = 1418.5542703195433
g_LK = 0.05121903777890511
g_h = 0.05304253169385108
c_th2ctx = 0.02933169668775031
c_ctx2th = 0.13183054727641988
```

Reward / summary metrics：

```text
shape_r = 0.0
so_power = 0.0
thal_spindle_power = 0.0
ctx_spindle_power = 0.0
T4_q = 3.438
T6_ibi_cv = 0.548
T8_n_sp_events = 14
T12_n_verified = 10
T13_n_ctx_events = 2
T13_n_ctx_verified = 1
T13_ctx_verified_density_per_min = 1.091
```

科学解释：本次 V8a best 已满足 T13 cortical spindle observability 的 permissive nonzero gate，并保留 T8/T12 thalamic spindle reality，但没有同时满足 T6 SO regularity。因此它不是 13/13 feasible 解。

## CSV 汇总

```text
total evaluations = 4960
13/13 feasible candidates = 0
12/13 near-feasible candidates = 105
```

`n_passed` 分布：

```text
3: 2
4: 6
5: 37
6: 269
7: 597
8: 719
9: 636
10: 1116
11: 1473
12: 105
13: 0
```

T1-T13 pass rate：

```text
T1 = 100.00%
T2 = 100.00%
T3 = 99.90%
T4 = 84.68%
T5 = 95.56%
T6 = 8.17%
T7 = 56.81%
T8 = 99.03%
T9 = 77.48%
T10 = 57.24%
T11 = 67.08%
T12 = 82.36%
T13 = 1.27%
```

T13/T4/T6 交叉统计：

```text
T13=1 candidates = 63
among T13=1, fail T4 or T6 = 63
T4=1 and T6=1 candidates = 375
among T4=1 and T6=1, pass T13 = 0
```

解释：在本次 V8a 搜索中，T13 cortical observability 可以出现，但与 T6 SO regularity 没有在同一候选中同时成立；所有 T13=1 的候选至少失败 T4 或 T6，而所有 T4=1 且 T6=1 的候选都没有通过 T13。这支持将 V8a 结果记录为“受控 audit patch 下未找到 full feasible candidate”，而不是解释为模型最终不正确。

---

## 第四部分补充：V8a evolution records 事后 trade-off 诊断

本次新增并运行了只读诊断脚本：

```text
valid_scripts/analyze_v8a_tradeoff.py
```

该脚本只读取既有 CSV/JSON/log，没有运行 neurolib simulation，没有运行新的优化，没有运行 300s validation，也没有运行 SO morphology validation。

## 生成文件

```text
outputs/v8a_diagnostics/v8a_tradeoff_summary.txt
outputs/v8a_diagnostics/v8a_12of13_candidates.csv
outputs/v8a_diagnostics/v8a_group_stats.csv
outputs/v8a_diagnostics/v8a_correlations.csv
outputs/v8a_diagnostics/fig_v8a_t6_vs_t13_density.png
outputs/v8a_diagnostics/fig_v8a_cth2ctx_vs_t13_density.png
outputs/v8a_diagnostics/fig_v8a_cth2ctx_vs_t6.png
outputs/v8a_diagnostics/fig_v8a_npassed_distribution.png
outputs/v8a_diagnostics/fig_v8a_failed_constraint_12of13.png
```

## 基本结果

```text
total evaluations = 4960
13/13 candidates = 0
12/13 candidates = 105
best score = -0.421538
best n_passed = 12/13
best failed constraint = T6
```

105 个 12/13 near-feasible 候选中，单一失败约束分布为：

```text
T13 failed = 104
T6 failed = 1
```

唯一一个失败 T6 的 12/13 候选就是本次 best：

```text
c_th2ctx = 0.02933169668775031
c_ctx2th = 0.13183054727641988
T4_q = 3.438
T6_ibi_cv = 0.548
T8_n_sp_events = 14
T12_n_verified = 10
T13_n_ctx_events = 2
T13_n_ctx_verified = 1
T13_ctx_verified_density_per_min = 1.091
```

## T4/T6/T13 trade-off

核心交叉统计：

```text
T13=1 candidates = 63
among T13=1, T4=1 and T6=1 = 0
among T13=1, T4=1 and T6=0 = 7
among T13=1, T4=0 and T6=1 = 0
among T13=1, T4=0 and T6=0 = 56

T4=1 and T6=1 candidates = 375
among T4=1 and T6=1, T13=1 = 0
```

这支持一个实用层面的 V8a trade-off：在本次搜索记录中，T13 cortical spindle observability 与 T6 SO regularity 没有共同出现；T13=1 的候选全部失败 T4 或 T6，而 T4=1/T6=1 的候选全部失败 T13。

## 参数区域和边界诊断

分组统计显示：

```text
All candidates median c_th2ctx = 0.0291
T13=1 median c_th2ctx = 0.0552
T4=1 and T6=1 median c_th2ctx = 0.0195
12/13 median c_th2ctx = 0.0220
best 12/13 T6-only c_th2ctx = 0.0293
```

T13=1 候选更偏向较高 `c_th2ctx`，但并不只集中在 V8a 上边界：

```text
T13=1 near c_th2ctx >= 0.070: 8/63
T6=1 near c_th2ctx >= 0.070: 4/405
top T13=1 c_th2ctx range = 0.022600 to 0.074615
top T13=1 median c_th2ctx = 0.048280
```

当前记录不支持“必须打开到 V8b 上边界才能得到 T13”的简单解释。扩大 `c_th2ctx` 到 V8b `[0.005, 0.085]` 可能增加 cortical observability，但也可能进一步损害 T4/T6；更稳妥的是先围绕 V8a near-feasible 区域做 targeted sweep。

## Spearman 趋势

主要 Spearman 相关：

```text
c_th2ctx vs T13 density: rho = 0.148
c_th2ctx vs T6_ibi_cv: rho = 0.264
c_th2ctx vs T4_q: rho = -0.355
c_ctx2th vs T13 density: rho = -0.028
c_ctx2th vs T6_ibi_cv: rho = 0.017
g_LK vs T13 density: rho = -0.146
g_LK vs T6_ibi_cv: rho = -0.064
g_h vs T13 density: rho = -0.107
g_h vs T6_ibi_cv: rho = -0.103
```

解释：`c_th2ctx` 与 T13 density 有弱正相关，同时与 T6_ibi_cv 有更明显的正相关、与 T4_q 有负相关。这与“增加 thalamus-to-cortex drive 有助于 cortical spindle observability，但可能扰动 SO regularity/structure”的假设方向一致，不过相关性本身不能证明因果。

## 保守结论和下一步

本次 records 支持 V8a objective/search 下存在 T13-vs-T6 的实际张力，但不能据此声称模型容量不足，也不能声称 V8a 科学失败。V8a 的价值是暴露了 cortical spindle observability audit patch 与 SO regularity 之间的冲突区域。

建议下一步优先选择：

```text
B) targeted c_th2ctx/c_ctx2th sweep around V8a near-feasible candidates
```

理由是：当前 records 已经显示 T13 不是单纯靠触碰 V8a 上边界产生；直接创建 V8b 并扩展 `c_th2ctx` 可能有用，但也可能恶化 T4/T6。更合理的顺序是先围绕 best 12/13 T6-only 点和 12/13 T13-only 点做局部 targeted sweep，确认 trade-off 的局部形状，再决定是否需要 V8b、300s T6 stability validation、detector redesign 或 model-structure revision。

---

## 第五部分：V8a coupling trade-off 长程定向 sweep

本次新增并运行了长程定向诊断脚本：

```text
valid_scripts/sweep_v8a_coupling_tradeoff_long.py
```

该脚本复用 `models/s4_personalize_fig7_v8.py` 中的 `build_model` 和 `compute_constraints_v8`，按 V8a 相同 5s burn-in 约定计算约束。没有修改主 V8a 脚本，没有运行 full DE，没有创建 V8b，没有改阈值、边界、reward 权重或 T13 逻辑，也没有运行 300s validation 或 SO morphology validation。

## 输出文件

输出目录：

```text
outputs/v8a_coupling_sweep_long/
```

主要生成文件：

```text
outputs/v8a_coupling_sweep_long/run_log.txt
outputs/v8a_coupling_sweep_long/selected_seed_candidates.csv
outputs/v8a_coupling_sweep_long/phase1_cth2ctx_sweep.csv
outputs/v8a_coupling_sweep_long/phase2_coupling_2d_sweep.csv
outputs/v8a_coupling_sweep_long/phase2_selected_bases.csv
outputs/v8a_coupling_sweep_long/v8a_coupling_sweep_long_summary.txt
```

生成图：

```text
outputs/v8a_coupling_sweep_long/fig_phase1_cth2ctx_vs_t13_density.png
outputs/v8a_coupling_sweep_long/fig_phase1_cth2ctx_vs_t6.png
outputs/v8a_coupling_sweep_long/fig_phase1_cth2ctx_vs_t4q.png
outputs/v8a_coupling_sweep_long/fig_phase1_t6_vs_t13_density.png
outputs/v8a_coupling_sweep_long/fig_phase2_heatmap_npassed.png
outputs/v8a_coupling_sweep_long/fig_phase2_heatmap_t6.png
outputs/v8a_coupling_sweep_long/fig_phase2_heatmap_t13_density.png
```

## 运行规模

脚本按限制在总 simulation 数达到 2500 时优雅停止：

```text
total simulations completed = 2500
phase1 simulations = 2070
phase2 simulations = 430
phase3 simulations = 0
phase4 120s simulations = 0
```

Phase 3/4 没有运行的原因不是条件完全没有触发，而是 simulation budget 已在 Phase 2 达到 2500。实际 sweep 中出现了 4 个 `T4=1, T13=1, T6_ibi_cv<0.60` 的候选，本来符合 Phase 3 的触发条件，但脚本按“超过 2500 停止”的限制优先停止。

## 主要结果

```text
13/13 candidates = 0
T4=1, T6=1, T13=1 candidates = 0
best n_passed = 12
best T4/T6/T13 combination = T4=1, T6=1, T13=0
```

全部 sweep 结果的 `n_passed` 分布：

```text
4: 2
5: 4
6: 42
7: 192
8: 287
9: 293
10: 542
11: 969
12: 169
13: 0
```

其他交叉统计：

```text
T13=1 candidates = 135
T4=1 and T6=1 candidates = 293
T4=1 and T6=1 and T13=1 candidates = 0
T4=1 and T13=1 and T6_ibi_cv<0.60 candidates = 4
```

Best sweep point：

```text
phase = phase1_cth2ctx
seed_id = seed_000
n_passed = 12
T4 = 1
T6 = 1
T13 = 0
c_th2ctx = 0.015000
c_ctx2th = 0.131831
T4_q = 4.544
T6_ibi_cv = 0.376
T13_ctx_verified_density_per_min = 0.000
```

最接近 T4/T13 共存、但仍失败 T6 的点包括：

```text
n_passed = 12
T4 = 1
T6 = 0
T13 = 1
c_th2ctx = 0.095000
c_ctx2th = 0.129055
T4_q = 3.181
T6_ibi_cv = 0.524
T13_ctx_verified_density_per_min = 1.091
```

以及：

```text
n_passed = 12
T4 = 1
T6 = 0
T13 = 1
c_th2ctx = 0.082500
c_ctx2th = 0.123069
T4_q = 2.717
T6_ibi_cv = 0.577
T13_ctx_verified_density_per_min = 2.182
```

这些点说明 cortical spindle observability 可以被较高 `c_th2ctx` 恢复到 T13 通过，但 T6 regularity 仍未恢复。

## c_th2ctx 趋势

Phase 1 Spearman 趋势：

```text
c_th2ctx vs T13 density: rho = 0.3205
c_th2ctx vs T6_ibi_cv: rho = 0.5114
c_th2ctx vs T4_q: rho = -0.3873
```

解释：在这个局部 sweep 中，增加 `c_th2ctx` 与 T13 cortical spindle density 增加相关，但同时更强地关联到 T6_ibi_cv 增加，并与 T4_q 降低相关。这支持“更强 thalamus-to-cortex coupling 有助于 cortical spindle observability，但会损害 SO regularity/structure”的 trade-off 解释。

## V8b 边界判断

对 V8b 相关区间 `[0.075, 0.085]` 的观察：

```text
points in 0.075 to 0.085 band = 226
best n_passed in 0.075 to 0.085 band = 12
T4=1, T6=1, T13=1 points in 0.075 to 0.085 band = 0
```

这不支持“V8a 失败主要是因为 `c_th2ctx` 上界太小”的简单解释。扩大到 V8b `[0.005, 0.085]` 可能作为诊断有价值，但当前 sweep 显示高 `c_th2ctx` 区域仍未解决 T6/T13 共存问题，直接跑 V8b full DE 的优先级不高。

## 保守结论

本次 sweep 没有找到 13/13，也没有找到 `T4=1, T6=1, T13=1` 的候选。结果支持 V8a 下存在一个 coupling trade-off：提高 `c_th2ctx` 倾向于提升 cortical spindle observability，但也倾向于提高 T6_ibi_cv、降低 T4_q，从而破坏 cortical SO stability/regularity。

这仍然不能被解释为模型容量不足，也不能说明 V8a 科学失败；它只是说明在当前 V8a objective、detector、参数区域和 sweep 策略下，`c_th2ctx` 不是单独的主阻塞因素。

## 唯一推荐下一步

```text
B) Run narrower local DE around best sweep region
```

理由：当前已有少数 `T4=1, T13=1, T6_ibi_cv<0.60` 的近邻点，但还没有恢复 T6 到阈值内。相比直接创建 V8b full DE，更合理的是围绕这些 near-feasible 区域做更窄的 local DE 或更小范围局部搜索，专门测试是否能把 T6_ibi_cv 从约 0.52-0.58 拉回到 0.40 以下，同时保持 T13。

---

## 第六部分：V8a local DE T6 rescue 长程诊断

本次新增并运行了本地诊断脚本：

```text
valid_scripts/local_de_v8a_t6_rescue_long.py
```

该脚本是围绕 V8a near-feasible 区域的窄范围 local DE / local search diagnostic，不是 V8b，也不是 full DE。它复用 `models/s4_personalize_fig7_v8.py` 中的 `build_model` 和 `compute_constraints_v8`，使用与 V8a 相同的 5s burn-in 约定。没有修改主 V8a 脚本，没有修改 T13 逻辑、阈值、reward 权重或 V8a bounds，没有运行 300s validation、SO morphology validation 或任何 post-hoc long validation。

## 输出文件

输出目录：

```text
outputs/v8a_local_de_t6_rescue_long/
```

生成文件：

```text
outputs/v8a_local_de_t6_rescue_long/run_log.txt
outputs/v8a_local_de_t6_rescue_long/local_de_records.csv
outputs/v8a_local_de_t6_rescue_long/best_so_far.json
outputs/v8a_local_de_t6_rescue_long/summary.txt
outputs/v8a_local_de_t6_rescue_long/selected_base_candidates.csv
```

## 运行规模

```text
total simulations = 6000
wall time = 8.880 h
13/13 candidates = 0
T4=1, T6=1, T13=1 candidates = 0
```

`n_passed` 分布：

```text
0: 9
4: 1
5: 28
6: 350
7: 1292
8: 1754
9: 564
10: 901
11: 1054
12: 47
13: 0
```

## Best 结果

Best candidate 仍是 12/13，并且只失败 T6：

```text
n_passed = 12
failed_constraints = T6
T4 = 1
T6 = 0
T13 = 1
T4_q = 3.099
T6_ibi_cv = 0.482
T8_n_sp_events = 21
T12_n_verified = 19
T13_n_ctx_events = 9
T13_n_ctx_verified = 1
T13_ctx_verified_density_per_min = 1.091
```

Best parameters：

```text
mue = 3.5828800501419704
mui = 2.860547170573283
b = 34.6023732743635
tauA = 1234.5109372050006
g_LK = 0.052488452709857754
g_h = 0.05128426102128965
c_th2ctx = 0.07876430111018883
c_ctx2th = 0.13718717944472264
```

## T6 rescue 与 T13 保持的关系

关键交叉统计：

```text
T13 preserved but T6 failed count = 650
T6 rescued but T13 lost count = 39
lowest T6_ibi_cv while T13 preserved = 0.439
best T6_ibi_cv among 12/13 with T13 preserved = 0.458
highest T13 verified density when T6 passed = 0.000
```

解释：

- Local DE 能把保持 T13 的候选从原先约 `T6_ibi_cv = 0.52-0.58` 推近到 `0.439`，但仍没有低于 V8a 的 T6 hard threshold `0.40`。
- 当 T6 被 rescue 到通过时，T13 verified density 没有保留下来，最高仍为 `0.000`。
- 因此，在当前 V8a detector/objective 和这些局部参数邻域内，T6 rescue 仍然倾向于破坏 T13；保持 T13 则仍然让 T6 停留在阈值外。

## 保守结论

本次 local DE 没有找到 13/13 candidate，也没有找到 `T4=1, T6=1, T13=1` candidate。这个结果不能被解释为模型失败或模型容量不足；它是当前 V8a cortical spindle detector/objective 与局部参数邻域下，T6-vs-T13 trade-off 的进一步证据。

相比前一轮 coupling sweep，本次 local DE 已经显示：在近可行区域内，确实可以把 T6_ibi_cv 往 0.40 靠近，同时保留 T13，但目前最好的 T13-preserved 点仍停在 `0.439`，没有跨过 T6 阈值。

## 唯一推荐下一步

```text
B) Run narrower local DE around best sweep region
```

理由：当前最好的局部点已经接近 T6 threshold，但仍未越过；直接创建 V8b 或修改 detector/objective 还太早。更合理的是围绕 `c_th2ctx ≈ 0.079`、`c_ctx2th ≈ 0.137`、`T6_ibi_cv ≈ 0.482`，以及最低 `T13-preserved T6_ibi_cv ≈ 0.439` 的邻域，做更窄、更定向的 local search，测试是否存在小范围参数组合能同时满足 T4、T6 和 T13。

---

## 第七部分：V8a T6-vs-T13 coupling sweep 诊断脚本准备

本次按要求只实现并编译检查 coupling sweep 诊断脚本，没有运行 full sweep。

新增脚本：

```text
valid_scripts/diagnose_v8a_t6_t13_coupling_sweep.py
```

该脚本用于在代表性参数组附近扫描 `c_th2ctx` 和 `c_ctx2th`，映射 T6-vs-T13 trade-off。它不是 DE，不是 V8b，也不会修改 V8a 主脚本。

## 安全边界

本次没有做以下事情：

```text
没有运行 coupling sweep
没有运行 full DE
没有创建 V8b
没有修改 models/s4_personalize_fig7_v8.py
没有改变 V8a hard constraints
没有放松 T6
没有放松 T13
没有改变 T13 detector logic
没有改变 reward weights 或 scientific framing
```

## 脚本设计

脚本会在用户批准后读取代表性 base：

```text
1. data/patient_params_fig7_v7_SC4001.json
2. data/patient_params_fig7_v8_SC4001.json
3. outputs/v8a_local_de_t6_rescue_narrow/best_so_far.json
4. outputs/v8a_local_de_t6_rescue_narrow/narrow_local_de_records.csv 中 top 3 n_passed=12 候选
```

每个 base 会固定非 coupling 参数，只扫描：

```text
c_th2ctx: 0.00 to 0.10
c_ctx2th: 0.08 to 0.20
```

每个 grid point 使用 V8a 的：

```text
build_model
compute_constraints_v8
60s simulation
5s burn-in
```

记录字段包括：

```text
base_id
c_th2ctx
c_ctx2th
n_passed
T4, T4_q
T6, T6_ibi_cv
T8, T8_n_sp_events
T12, T12_n_verified
T13, T13_n_ctx_events, T13_n_ctx_verified
T13_ctx_verified_density_per_min
failed_constraints
```

计划输出：

```text
outputs/v8a_t6_t13_coupling_sweep.csv
outputs/v8a_t6_t13_coupling_sweep_summary.txt
outputs/v8a_t6_t13_coupling_sweep_bases.csv
```

## 编译检查

已运行：

```text
python -m py_compile valid_scripts\diagnose_v8a_t6_t13_coupling_sweep.py
```

结果：

```text
py_compile passed
```

静态检查显示脚本没有 `differential_evolution` 调用，也没有 300s validation 逻辑。脚本只导入 V8a 的 `build_model`、`compute_constraints_v8` 和 PSD helper。

## 当前状态

脚本已经准备好，但尚未运行 sweep。下一步需要用户明确批准后，才能启动该 coupling grid scan。

---

## 第八部分：V8a T6-vs-T13 coupling sweep 诊断运行结果

本次按批准运行了诊断 grid scan：

```text
valid_scripts/diagnose_v8a_t6_t13_coupling_sweep.py
```

运行前确认：

```text
py_compile passed
使用 PYTHONIOENCODING=utf-8
输出文件为 outputs/v8a_t6_t13_coupling_sweep*
不会覆盖 outputs/v8a_coupling_sweep_long/ 或 local DE 输出目录
```

本次没有修改 V8a 主脚本，没有改变 V8a hard constraints，没有放松 T6 或 T13，没有改变 T13 detector logic，没有运行 full DE、V8b、300s validation 或 SO morphology validation。

## 运行中修复

启动时发现两个只属于新诊断脚本 base selection 的问题，均发生在任何 simulation 开始前：

```text
1. 输入 narrow records 已有 base_id 列，脚本再次 insert base_id 导致列冲突。
2. 某些代表性 JSON/base 缺少部分 T 列，failed_constraints 对 NaN 转 int 失败。
```

已做最小修复：

```text
若输入已有 base_id，则先 drop 再生成本次 sweep 的 base_id。
failed_constraints 对 NaN/缺失 T 列做 safe handling，缺失时标记 unknown。
```

这些修复只影响新诊断脚本的输入表整理，不影响 V8a 模型、约束、T13 detector 或科学逻辑。

## 输出文件

生成文件：

```text
outputs/v8a_t6_t13_coupling_sweep.csv
outputs/v8a_t6_t13_coupling_sweep_summary.txt
outputs/v8a_t6_t13_coupling_sweep_bases.csv
```

## 基本结果

```text
total simulations = 3875
base count = 5
13/13 candidates = 0
T4=1, T6=1, T13=1 points = 0
best n_passed = 12
```

`n_passed` 分布：

```text
6: 3
7: 14
8: 205
9: 600
10: 1136
11: 1813
12: 104
13: 0
```

## 代表性 best points

Best 13/13：

```text
none
```

Best `T4=1, T6=1, T13=1`：

```text
none
```

Best overall 按 `n_passed` 和核心约束优先排序的代表点来自 V7 best base，仍是 12/13，但失败 T13：

```text
base_id = base_00
base_source = v7_best
c_th2ctx = 0.010
c_ctx2th = 0.190
n_passed = 12
failed_constraints = T13
T4 = 1
T4_q = 4.207
T6 = 1
T6_ibi_cv = 0.256
T13 = 0
T13_n_ctx_verified = 0
T13_ctx_verified_density_per_min = 0.000
T8_n_sp_events = 25
T12_n_verified = 23
```

Lowest T6_ibi_cv while preserving `T4=1` and `T13=1`：

```text
base_id = base_02
base_source = narrow_local_de_best
c_th2ctx = 0.06666666666666667
c_ctx2th = 0.120
n_passed = 12
failed_constraints = T6
T4 = 1
T4_q = 2.214
T6 = 0
T6_ibi_cv = 0.417
T13 = 1
T13_n_ctx_verified = 1
T13_ctx_verified_density_per_min = 1.091
T8_n_sp_events = 6
T12_n_verified = 6
```

Highest T13 verified density while preserving `T4=1` and `T6=1`：

```text
T13_ctx_verified_density_per_min = 0.000
```

Representative point in this class:

```text
base_id = base_00
base_source = v7_best
c_th2ctx = 0.000
c_ctx2th = 0.080
n_passed = 10
failed_constraints = T9,T10,T13
T4 = 1
T4_q = 3.886
T6 = 1
T6_ibi_cv = 0.295
T13 = 0
T13_n_ctx_verified = 0
T13_ctx_verified_density_per_min = 0.000
T8_n_sp_events = 23
T12_n_verified = 20
```

## Coupling trend

Summary 文件给出的 `c_th2ctx` 趋势：

```text
mean Spearman rho c_th2ctx -> T13 density = 0.809
mean Spearman rho c_th2ctx -> T6 IBI CV = 0.956
mean Spearman rho c_th2ctx -> T4_q = -0.597
```

解释：在这些 representative bases 的 coupling grid 上，增加 `c_th2ctx` 强烈增加 T13 cortical spindle density，但也更强地增加 T6_ibi_cv，并降低 T4_q。这与 V8a full DE、long coupling sweep、local DE T6 rescue 看到的 T6-vs-T13 trade-off 一致。

## 保守结论

本次 3875 点 coupling grid scan 没有找到 13/13，也没有找到 `T4=1, T6=1, T13=1` 的点。

结果进一步支持：在当前 V8a hard constraints、T13 detector 和这些代表性参数邻域下，`c_th2ctx` 增大可以恢复 cortical spindle observability，但同时会损害 SO regularity/structure，表现为 T6_ibi_cv 上升和 T4_q 下降。

这不是模型失败证据，也不是模型容量不足结论；它是当前 V8a audit/objective 下的局部 trade-off 证据。

## 推荐下一步

不建议立即放松 T6 或 T13 阈值。

由于 grid scan 找到了一个非常接近 T6 阈值的 T13-preserved 点：

```text
c_th2ctx = 0.06666666666666667
c_ctx2th = 0.120
T4 = 1
T13 = 1
T6_ibi_cv = 0.417
```

推荐下一步：

```text
one final ultra-narrow local search around the best T13-preserved T6_ibi_cv point
```

如果这个 ultra-narrow search 仍不能找到 `T4=1, T6=1, T13=1`，再停止继续追 13/13，转向论文 framing 和 post-hoc diagnostic figures。

---

## 第九部分：最终 ultra-narrow T6-vs-T13 local search

本次按批准运行了最终 ultra-narrow diagnostic：

```text
valid_scripts/local_search_v8a_ultra_narrow_t6_t13.py
```

该脚本从 coupling sweep 中自动选择 `T4=1, T13=1` 且 `T6_ibi_cv` 最低的点作为 base：

```text
c_th2ctx = 0.06666666666666667
c_ctx2th = 0.120
T4 = 1
T13 = 1
T6_ibi_cv = 0.417
T13_ctx_verified_density_per_min = 1.091
```

本次仍然没有修改 V8a 主脚本，没有改变 hard constraints，没有放松 T6 或 T13，没有改变 T13 detector logic，没有改变 reward weights，没有创建 V8b，没有运行 300s validation，也没有运行 SO morphology validation。

## 输出文件

输出目录：

```text
outputs/v8a_ultra_narrow_t6_t13_search/
```

生成文件：

```text
outputs/v8a_ultra_narrow_t6_t13_search/run_log.txt
outputs/v8a_ultra_narrow_t6_t13_search/selected_base_point.json
outputs/v8a_ultra_narrow_t6_t13_search/ultra_narrow_records.csv
outputs/v8a_ultra_narrow_t6_t13_search/best_so_far.json
outputs/v8a_ultra_narrow_t6_t13_search/summary.txt
```

## 运行规模

```text
total simulations = 1536
wall time = 2.153 h
13/13 candidates = 0
T4=1, T6=1, T13=1 candidates = 0
```

`n_passed` 分布：

```text
5: 4
6: 26
7: 81
8: 95
9: 146
10: 170
11: 976
12: 38
13: 0
```

## Best candidate

Best candidate 仍为 12/13，只失败 T6：

```text
n_passed = 12
failed_constraints = T6
T4 = 1
T4_q = 2.016
T6 = 0
T6_ibi_cv = 0.504
T8_n_sp_events = 13
T12_n_verified = 13
T13 = 1
T13_n_ctx_verified = 1
T13_ctx_verified_density_per_min = 1.091
c_th2ctx = 0.06737994007146982
c_ctx2th = 0.1253491302153237
```

Best parameters：

```text
mue = 3.5213182430642145
mui = 2.8850644250475654
b = 35.8512589767372
tauA = 1220.0298990584624
g_LK = 0.05024591467710125
g_h = 0.05428717809722123
c_th2ctx = 0.06737994007146982
c_ctx2th = 0.1253491302153237
```

## T13-preserved vs T6-rescued points

本次 ultra-narrow search 中：

```text
T13=1 candidates = 64
T6=1 candidates = 34
T4=1, T6=1, T13=1 candidates = 0
```

Lowest `T6_ibi_cv` while `T13=1`：

```text
T6_ibi_cv = 0.430
T13_ctx_verified_density_per_min = 1.091
n_passed = 8
failed_constraints = T4,T6,T7,T10,T11
T4_q = 1.829
c_th2ctx = 0.06800218893746436
c_ctx2th = 0.11200146552928054
```

注意：这个最低 T6-CV 的 T13-preserved 点已经丢失 T4，因此不能作为 SO-structure-preserved 解。

Highest T13 density while `T6=1`：

```text
T13_ctx_verified_density_per_min = 0.000
n_passed = 12
failed_constraints = T13
T4 = 1
T4_q = 2.149
T6 = 1
T6_ibi_cv = 0.390
c_th2ctx = 0.06372198043712286
c_ctx2th = 0.1181704231737158
```

这说明在本次 ultra-narrow neighborhood 中，一旦 T6 被 rescue 到阈值内，T13 verified cortical spindle observability 仍然消失。

## 结论

本次最终 ultra-narrow search 没有找到 13/13，也没有找到 `T4=1, T6=1, T13=1` 的点。

它没有改变此前结论：在当前 V8a hard constraints、T13 detector、objective framing 和这个局部参数邻域下，T6-vs-T13 trade-off 仍然存在。保持 T13 时 T6 仍高于阈值；T6 被救回时 T13 verified density 仍为 0。

这不是模型失败证据，也不是模型容量不足结论；它是当前 audit patch 下的负结果，应作为受控诊断结果报告。

## 推荐下一步

```text
stop searching for 13/13 and move to paper framing plus diagnostic figures
```

不建议继续 blind search，也不建议立即放松 T6 或 T13。下一步应把 V8a/V8a-diagnostics 的结果整理成论文 framing：V8a 是 cortical spindle observability audit patch，不是最终模型正确性证明；现有结果支持一个局部 SO-regularity vs cortical-spindle-observability trade-off。

---

## 第十部分：V8a relaxed-T6 sensitivity post-hoc 分析

本次新增并运行了只读 post-hoc sensitivity 脚本：

```text
valid_scripts/analyze_v8a_relaxed_t6_sensitivity.py
```

该脚本没有运行任何 neurolib simulation，没有运行 DE，没有创建 V8b，没有修改 `models/s4_personalize_fig7_v8.py`，没有改变 T13 detector logic，也没有改写任何已有 CSV。它只读取已有记录，检查如果只 post-hoc 放宽 T6 阈值，当前 near-miss candidates 会如何变化。

## 输入文件

全部指定输入文件均存在并被使用：

```text
outputs/evolution_fig7_v8_records.csv
outputs/v8a_t6_t13_coupling_sweep.csv
outputs/v8a_ultra_narrow_t6_t13_search/ultra_narrow_records.csv
outputs/v8a_local_de_t6_rescue_long/local_de_records.csv
outputs/v8a_local_de_t6_rescue_narrow/narrow_local_de_records.csv
outputs/v8a_coupling_sweep_long/phase1_cth2ctx_sweep.csv
outputs/v8a_coupling_sweep_long/phase2_coupling_2d_sweep.csv
```

缺失文件：

```text
none
```

## 输出文件

输出目录：

```text
outputs/v8a_relaxed_t6_sensitivity/
```

生成文件：

```text
outputs/v8a_relaxed_t6_sensitivity/relaxed_t6_summary.csv
outputs/v8a_relaxed_t6_sensitivity/relaxed_t6_best_candidates.csv
outputs/v8a_relaxed_t6_sensitivity/relaxed_t6_interpretation.md
```

## 分析定义

严格 V8a 使用：

```text
T6_ibi_cv < 0.40
```

本次只做 post-hoc sensitivity，测试：

```text
T6_ibi_cv < 0.40
T6_ibi_cv < 0.42
T6_ibi_cv < 0.45
T6_ibi_cv < 0.50
```

Relaxed-feasible 的定义是：

```text
T4 = 1
T13 = 1
所有可用 T constraints 除 T6 外均通过
relaxed_T6 = T6_ibi_cv < threshold
```

只放宽 T6，不放宽 T13、T4、T8 或 T12。

## Summary table

```text
threshold = 0.40
relaxed_13of13_count = 0
T4_T13_relaxedT6_count = 0
best_status = closest_near_miss
best T6_ibi_cv = 0.417
threshold_margin = -0.017
```

```text
threshold = 0.42
relaxed_13of13_count = 1
T4_T13_relaxedT6_count = 1
best_status = relaxed_feasible
best T6_ibi_cv = 0.417
threshold_margin = 0.003
```

```text
threshold = 0.45
relaxed_13of13_count = 1
T4_T13_relaxedT6_count = 2
best_status = relaxed_feasible
best T6_ibi_cv = 0.417
threshold_margin = 0.033
```

```text
threshold = 0.50
relaxed_13of13_count = 9
T4_T13_relaxedT6_count = 19
best_status = relaxed_feasible
best T6_ibi_cv = 0.417
threshold_margin = 0.083
```

## Selected candidate

所有 relaxed thresholds 下选出的 best candidate 都是同一个 coupling-sweep near-miss：

```text
source_file = outputs/v8a_t6_t13_coupling_sweep.csv
source_row_index = 2058
original_n_passed = 12
original_failed_constraints = T6
T4 = 1
T4_q = 2.214
original_T6 = 0
T6_ibi_cv = 0.417
T13 = 1
T13_n_ctx_verified = 1
T13_ctx_verified_density_per_min = 1.091
T8_n_sp_events = 6
T12_n_verified = 6
c_th2ctx = 0.0666666666666666
c_ctx2th = 0.120
mue = 3.5518400881697088
mui = 2.8161439514884195
b = 35.913119447152965
tauA = 1269.884859611847
g_LK = 0.0505396660372011
g_h = 0.0534014722851463
```

## 保守解释

严格 V8a 阈值 `T6_ibi_cv < 0.40` 下，仍然没有 relaxed 13/13 candidate；最佳 near-miss 距离阈值还差 `0.017`。

当 post-hoc 阈值放宽到 `0.42` 时，出现 1 个 relaxed-feasible candidate。这应表述为“5% relaxed-T6 sensitivity result”，不能表述为严格 V8a 成功。

本分析不证明模型已完全验证，也不证明 V8a 在原始 strict criteria 下成功。它只说明当前 near-miss 与 T6 阈值非常接近：`0.417` 落在 `0.40` 与 `0.42` 之间。

不建议基于这张表直接修改 V8a 主阈值。任何未来阈值调整都需要单独科学论证和明确批准。

# 第十一部分：V8a 0712 held-out cortical spindle-density 可视化

## 目的

本部分参考现有 `validation_outputs/fig_spindle_density_heldout.png` 的图形结构，新增一张 V8a 0712 best0417 的 held-out spindle-density 对比图，用于展示：

```text
Real SC4001 N3 EEG
V1 canonical
V7 event-constrained
V8a 0712 best0417
```

该图使用独立 sigma-RMS detector 测量 cortical / EEG-like spindle density。它是 post-hoc visualization / diagnostic，不是 V8a strict success，也不是新的模型版本。

## 执行限制

本步骤没有修改：

```text
models/s4_personalize_fig7_v8.py
T6 hard threshold
T13 detector logic
V8a reward weights
V8a scientific framing
```

本步骤没有运行：

```text
DE optimization
300s validation
SO morphology validation
V8b
```

本步骤只运行了用于绘图的 120s cortical `r_ctx` density visualization，与原 held-out spindle-density 图的设置保持一致。

## 新增脚本

```text
valid_scripts/plot_v8a_0712_heldout_spindle_density.py
```

脚本复用了原 held-out spindle-density detector 的核心设置：

```text
sigma band = 11-15 Hz
RMS window = 0.20 s
threshold = mean(RMS) + 1.5 * SD(RMS)
event duration = 0.5-3.0 s
merge gap = 0.10 s
model duration = 120 s
burn-in = 5 s
signal = cortical r_ctx
```

## 输出文件

输出目录：

```text
outputs/v8a_0712_best0417_figures/
```

新增文件：

```text
outputs/v8a_0712_best0417_figures/fig_v8a_0712_heldout_spindle_density.png
outputs/v8a_0712_best0417_figures/fig_v8a_0712_heldout_spindle_density.csv
outputs/v8a_0712_best0417_figures/fig_v8a_0712_heldout_spindle_density_summary.txt
```

## 关键数值结果

```text
Real SC4001 N3 EEG          210 events / 71.00 min = 2.958 /min
V1 canonical                 2 events / 1.917 min = 1.043 /min
V7 event-constrained         0 events / 1.917 min = 0.000 /min
V8a 0712 best0417            0 events / 1.917 min = 0.000 /min
```

相对真实 EEG density 的绝对距离：

```text
|V1 - real|  = 1.915 events/min
|V7 - real|  = 2.958 events/min
|V8a - real| = 2.958 events/min
```

## 保守解释

这张 held-out sigma-RMS density 图不能说明 V8a 0712 best0417 相比 V7 在外部 RMS detector 下已经改善。

更准确的解释是：

```text
V8a best0417 在 V8a 内部 T13 audit 中有 cortical spindle observability：
T13_n_ctx_verified = 1
T13_ctx_verified_density_per_min = 1.091

但在独立 held-out sigma-RMS detector、120s cortical r_ctx 检查下：
V8a best0417 = 0.000 /min
V7 = 0.000 /min
```

因此，V8a 的 T13 permissive nonzero gate 确实修补了 V7 的“完全没有 cortical spindle observability”这一内部审计漏洞，但它还没有通过更严格、独立的 held-out RMS spindle-density 检查。

与 V1/V7 的比较中，V1 canonical 在该 held-out density 指标上更接近真实 EEG：

```text
Real = 2.958 /min
V1   = 1.043 /min
V7   = 0.000 /min
V8a  = 0.000 /min
```

因此，这张图适合用于论文/汇报中的保守表述：

```text
V8a is an audit patch, not final proof of model correctness.
It restores an internal cortical observability check, but the stricter held-out RMS density diagnostic still exposes residual mismatch.
```

不建议基于这张图继续盲目追 13/13，也不建议直接放宽 T13 或 T6。更合理的下一步是把该结果纳入 paper framing 和 diagnostic figures，说明 V8a 的贡献与局限。

# 第十二部分：relaxed-T6 sensitivity 分析复核

## 目的

本部分复核此前用于找到 `V8a best0417` 的 relaxed-T6 sensitivity 分析，确认该 candidate 是在只放宽 T6 阈值的 post-hoc sensitivity 检查中被选出的 near-miss，而不是严格 V8a 13/13 成功。

本步骤没有运行任何新的 neurolib simulation，也没有运行 DE。

## 脚本状态

脚本已存在并通过语法检查：

```text
valid_scripts/analyze_v8a_relaxed_t6_sensitivity.py
python -m py_compile valid_scripts/analyze_v8a_relaxed_t6_sensitivity.py
```

由于已有输出文件已经存在，且本任务要求不改变现有 CSV，本次只做只读复核，没有重跑会覆盖输出的分析脚本。

## 已有输出文件

```text
outputs/v8a_relaxed_t6_sensitivity/relaxed_t6_summary.csv
outputs/v8a_relaxed_t6_sensitivity/relaxed_t6_best_candidates.csv
outputs/v8a_relaxed_t6_sensitivity/relaxed_t6_interpretation.md
```

## 输入文件状态

解释文件显示本分析使用了全部 7 个既有 CSV，且没有缺失文件：

```text
outputs/evolution_fig7_v8_records.csv
outputs/v8a_t6_t13_coupling_sweep.csv
outputs/v8a_ultra_narrow_t6_t13_search/ultra_narrow_records.csv
outputs/v8a_local_de_t6_rescue_long/local_de_records.csv
outputs/v8a_local_de_t6_rescue_narrow/narrow_local_de_records.csv
outputs/v8a_coupling_sweep_long/phase1_cth2ctx_sweep.csv
outputs/v8a_coupling_sweep_long/phase2_coupling_2d_sweep.csv
```

缺失文件：

```text
none
```

## 复核结果

relaxed-T6 sensitivity 表中的核心结果如下：

```text
T6 threshold 0.40:
relaxed_13of13_count = 0
best_status = closest_near_miss
best T6_ibi_cv = 0.417
threshold_margin = -0.017

T6 threshold 0.42:
relaxed_13of13_count = 1
best_status = relaxed_feasible
best T6_ibi_cv = 0.417
threshold_margin = 0.003

T6 threshold 0.45:
relaxed_13of13_count = 1
best_status = relaxed_feasible
best T6_ibi_cv = 0.417
threshold_margin = 0.033

T6 threshold 0.50:
relaxed_13of13_count = 9
best_status = relaxed_feasible
best T6_ibi_cv = 0.417
threshold_margin = 0.083
```

## best0417 candidate 来源

所有 relaxed thresholds 下选出的 best candidate 都是同一个 coupling-sweep near-miss：

```text
source_file = outputs/v8a_t6_t13_coupling_sweep.csv
source_row_index = 2058
base_id = base_02
base_source = narrow_local_de_best
n_passed = 12
failed_constraints = T6
T4 = 1
T4_q = 2.214
T6 = 0
T6_ibi_cv = 0.417
T13 = 1
T13_n_ctx_verified = 1
T13_ctx_verified_density_per_min = 1.091
T8_n_sp_events = 6
T12_n_verified = 6
c_th2ctx = 0.0666666666666666
c_ctx2th = 0.120
```

参数为：

```text
mue = 3.5518400881697088
mui = 2.8161439514884195
b = 35.913119447152965
tauA = 1269.884859611847
g_LK = 0.0505396660372011
g_h = 0.0534014722851463
```

## 保守解释

严格 V8a 仍然使用：

```text
T6_ibi_cv < 0.40
```

在该 strict threshold 下，`best0417` 不是 13/13 feasible candidate，而是 12/13 near-miss，唯一失败项是 T6。

当 post-hoc threshold 放宽到 `0.42` 时，`best0417` 成为唯一 relaxed-feasible candidate。这只能表述为：

```text
5% relaxed-T6 sensitivity result
```

不能表述为：

```text
strict V8a success
```

该复核进一步确认：`best0417` 的价值在于展示 T6-vs-T13 trade-off 的边界位置，而不是证明模型已经满足所有严格 V8a 审计标准。

# 第十三部分：Figure-10-inspired validation panel

## 目的

本部分创建一个受 SBI Figure 10 启发的多面板 validation / audit 图，用于当前 8 参数 cortical-thalamic sleep neural mass fitting 项目。

重要科学限定：

```text
当前项目还没有 NPE/NLE/NRE posterior estimator。
因此，本图不使用 posterior samples / expected coverage / SBC / TARP / L-C2ST 等表述。
```

本图使用的术语是：

```text
fitted-candidate ensemble
candidate archive
PPC-like predictive diagnostics
synthetic parameter-recovery diagnostic
future SBI posterior diagnostics
```

## 新增脚本

```text
valid_scripts/plot_figure10_inspired_validation_panel.py
```

脚本已通过：

```text
python -m py_compile valid_scripts/plot_figure10_inspired_validation_panel.py
```

脚本没有修改任何模型、阈值、objective 或已有 CSV/JSON 输出。

## 运行命令

```text
PYTHONIOENCODING=utf-8 python valid_scripts/plot_figure10_inspired_validation_panel.py --max-candidates 50 --n-reps 3
```

实际在 `neurolib` conda 环境中运行。

第一次完整运行完成了 50 个候选、每个 3 个 replicate 的短仿真，共 150 次 60s simulation。随后发现 synthetic recovery 距离函数返回类型有 bug，修复后第二次运行复用了已生成的 predictive summaries，只重算 synthetic recovery 和 figure。

## 输出目录

```text
outputs/figure10_inspired_validation_panel/
```

生成文件：

```text
outputs/figure10_inspired_validation_panel/candidate_archive.csv
outputs/figure10_inspired_validation_panel/candidate_predictive_summaries.csv
outputs/figure10_inspired_validation_panel/real_observation_summaries.csv
outputs/figure10_inspired_validation_panel/synthetic_recovery_results.csv
outputs/figure10_inspired_validation_panel/figure10_inspired_validation_panel.png
outputs/figure10_inspired_validation_panel/figure10_inspired_validation_panel.pdf
outputs/figure10_inspired_validation_panel/figure10_inspired_validation_panel_summary.md
```

## Candidate archive

最终 archive：

```text
candidate count = 50
```

覆盖版本：

```text
V1
V2
V3
V4
V5
V6
V7
V8a
V8a-coupling
V8a-relaxed
V8a-ultra
```

各版本数量：

```text
V1               2
V2               1
V3               2
V4               3
V5               3
V6               5
V7              10
V8a              7
V8a-coupling    10
V8a-relaxed      1
V8a-ultra        6
```

已使用文件包括 V1-V8 JSON、V3-V8 evolution records，以及 V8a relaxed/coupling/ultra/local diagnostic CSV。

缺失但已跳过的文件：

```text
data/patient_params_fig7_v1_SC4001.json
```

实际使用了 V1 canonical alias：

```text
data/patient_params_fig7_v1_0418_2_SC4001.json
```

## Real observation summaries

真实 SC4001 N3 EEG summary 已计算：

```text
subject_id = SC4001
channel = EEG Fpz-Cz
n_clean_n3_epochs = 142
heldout_rms_spindle_density = 2.9577464788732395 /min
heldout_rms_spindle_events = 210
psd_delta_mean = 210.78446887149425
psd_sigma_mean = 1.2982878152505395
```

PAC / SO morphology 的真实 EEG summary 如果当前 utility 不直接提供，则保留为 NaN，不让脚本失败。

## PPC-like predictive diagnostics

完成短仿真：

```text
candidate count = 50
replicates per candidate = 3
completed simulations = 150
SIM_DUR_MS = 60,000
burn-in = 5 s
```

所有 150 次 simulation 均成功：

```text
simulation_ok = 1: 150
```

预测诊断摘要包括：

```text
shape_r
T4_q
T6_ibi_cv
T8_n_sp_events
T12_n_verified
T13_n_ctx_verified
T13_ctx_verified_density_per_min
heldout_rms_spindle_density
T9_mi
T10_phase
T11_lag_ms
n_passed
failed_constraints
```

注意：这些是 fitted-candidate ensemble 上的 PPC-like predictive diagnostics，不是 calibrated posterior predictive checks。

## Synthetic parameter-recovery diagnostic

本部分实现的是 archive-based nearest-neighbor recovery，不是 SBC。

最终 synthetic recovery 结果：

```text
synth_00: true V1          -> recovered V3            same_version = 0, mean_relative_error = 0.9538
synth_01: true V7          -> recovered V7            same_version = 1, mean_relative_error = 0.0682
synth_02: true V8a         -> recovered V8a-coupling  same_version = 0, mean_relative_error = 0.2370
synth_03: true V8a-relaxed -> recovered V8a-coupling  same_version = 0, mean_relative_error = 0.0682
synth_04: true V3          -> recovered V3            same_version = 1, mean_relative_error = 0.0000
```

解释：

```text
该结果只说明在当前 candidate archive 和 summary feature 空间下，nearest-neighbor recovery 对某些候选能找回相近版本或同版本，对 V1 则恢复较差。
它不是 SBC，不提供 posterior calibration 或 coverage 结论。
```

## Figure panels

主图保存为：

```text
outputs/figure10_inspired_validation_panel/figure10_inspired_validation_panel.png
outputs/figure10_inspired_validation_panel/figure10_inspired_validation_panel.pdf
```

Panel A：

```text
8-parameter cortical-thalamic neural mass model schematic / text schematic
```

Panel B：

```text
Real SC4001 N3 EEG summaries
```

Panel C：

```text
fitted-candidate ensemble scatter
x = c_th2ctx
y = c_ctx2th
color = version / source
```

Panel D：

```text
PPC-like predictive diagnostics over fitted-candidate ensemble
T6_ibi_cv
T13 cortical density
held-out RMS spindle density
T4_q
```

Panel E：

```text
synthetic parameter-recovery diagnostic
archive-based nearest-neighbor recovery, not SBC
```

Panel F：

```text
future SBI diagnostics box
true expected coverage / SBC / TARP / L-C2ST require q_phi(theta|x)
```

## 保守解释

这张图适合放在论文或汇报中作为 validation / audit framing：

```text
Figure inspired by SBI Figure 10, but not an SBI posterior diagnostic.
```

它展示的是：

```text
1. 当前已有 fitted-candidate ensemble 覆盖的参数区域；
2. 这些候选在短仿真下的 PPC-like predictive summary 分布；
3. archive-based synthetic recovery 的粗略可恢复性；
4. 未来真正 SBI posterior diagnostics 需要 q_phi(theta|x)。
```

不能声称：

```text
posterior samples
expected coverage
SBC
TARP
L-C2ST
calibrated posterior predictive check
```

本图的合理结论是：当前 V1-V8/V8a candidate archive 可以支持一张 Figure-10-inspired audit panel，但真正的 SBI posterior validation 仍是未来工作。

# 第十四部分：2026-07-22 S4_sbi 第一阶段只读审计

## 1. 审计边界与执行状态

本节是追加式审计记录，没有改写前十三部分。除追加本节外，本轮没有修改、覆盖或移动任何项目文件，没有启动大规模 simulation，也没有训练 NPE。

运行时核验全部通过 `neurolib` conda 环境完成：

```text
conda env: C:\Users\YUS190\AppData\Local\anaconda3\envs\neurolib
Python: 3.10.20
neurolib: 0.6.1
sbi: 0.26.1
torch: 2.5.1 (CUDA unavailable)
numpy: 2.2.6
scipy: 1.15.2
mne: 1.9.0
pandas: 2.3.3
matplotlib: 3.10.8
fooof: 1.1.1
numba: 0.64.0
jupyterlab: 4.5.6
ipykernel: 7.2.0
```

`neurolib.__file__` 实际解析到 `D:\Year3_Mao_Projects\neurolib\neurolib\__init__.py`，即本地源码优先，而不是环境内 site-packages；这与 `CLAUDE.md:96-102` 记录的 NumPy compatibility shim 和本地 patched neurolib 要求一致。`requirements.txt:1-11` 没有锁定 `sbi`、`torch`、Jupyter 等关键版本，因此当前环境清单必须作为后续可复现元数据保存，但本轮没有安装、升级或降级任何包。

Git 审计结果：当前分支 `main`，HEAD 与 `origin/main` 均为 `82aea49`（2026-07-04，`Add held-out validation scripts and outputs`）；另一个可见本地分支 `publish-main` 为 `441a2bf`。工作树已有大量用户未提交变更和未跟踪文件。`S4_sbi` 根目录被 Git 标为删除的旧文件，在 `S4_sbi/legacy/` 中存在 Git blob 完全相同的未跟踪副本；本轮未对这些状态做清理或回滚。

项目中没有找到 `AGENTS.md`；仓库级操作说明来自 `CLAUDE.md`。README 仍把 v3 称为当前主线（`README.md:30-79`），而较新的项目说明明确 V7 才是当前主线（`CLAUDE.md:71-80`），所以 README 不能单独作为当前状态依据。

## 2. 项目研究路线图与依赖关系

当前研究链可整理为：

```text
Sleep-EDF SC4001 PSG + hypnogram
  -> N3 分期、Fpz-Cz 提取、artifact/QC、epoch PSD
  -> V1-V7 的 8 参数 Fig.7 式拟合
  -> V7 PAC cycle-by-cycle repair 与 Pareto/warm-start 复核
  -> 独立 SO waveform / spindle-density held-out validation
  -> V8a/T13 cortical observability audit patch 与局部 trade-off 搜索
  -> 旧 S4_sbi 的 4 参数、5-summary、4 轮 sequential SNPE-C
  -> 新 S4_sbi：统一 observation/summary contract + V7/V8a adapters
  -> prior predictive support check
  -> 每个机制版本各自的 5 个独立 NPE ensemble
  -> global validation、local validation/L-C2ST、PPC、生理解释
```

具体证据：

- SC4001 数据入口在 `data/manifest.csv`；预处理默认 Fpz-Cz、30 s epoch、4 s Welch window、1 s overlap，并把电压从 V 转为 µV（`utils/02_preprocess_psd.py:22-27,116-138`）。
- V1 已经建立完整 8 参数模型和 EEG PSD 对皮层 firing-rate PSD 的拟合框架（`models/s4_personalize_fig7.py:130-155,272-334,388-398`）。V2-V6 是历史演进，V7 是成熟 baseline（`CLAUDE.md:71-80`；`CODEBASE_INVENTORY.md:32-35,58-64`）。
- V7 的 PAC 修复是 cycle-by-cycle phase，并在 `S4_v7_repair/compute_pac_metrics_fixed.py:79-120,228-260` 输出 MI、preferred phase、phase concentration 和 up/down ratio；`valid_scripts/compute_pac_metrics_fixed.py` 是逐字节镜像（`CLAUDE.md:82-86`）。
- EEG-native T6/PAC 是另一套 observation-side 算法，文件自己明确 simulator 仍保留 V7 的跨信号定义（`S4_v7_repair/compute_pac_metrics_eeg_native.py:4-20,31-147`）。
- held-out spindle density 对真实 EEG 和模拟 cortical `r_ctx` 使用同一 11-15 Hz RMS detector（`valid_scripts/validate_spindle_density_heldout.py:56-65,147-161,211-249`）；SO held-out 使用 0.3-1.5 Hz waveform template（`valid_scripts/validate_so_waveform_heldout.py:56-59,186-218,443-444`）。
- V8a/T13 是在 V7 上增加 cortex-visible spindle 审计，而不是替换整个动力学模型（`models/s4_personalize_fig7_v8.py:101-136,333-348,617-680`）。
- 旧 SBI 则缩为 4 个自由参数、5 个 summary，并固定另外 4 个参数（`S4_sbi/STAGE2_ARCHITECTURE.md:62-83`；`S4_sbi/simulator_wrapper.py:4-22,88-99`）。这不是新项目目标要求的统一 8 参数推断。

## 3. 关键文件清单、当前用途与归类

| 文件/目录 | 当前作用 | 审计归类 |
|---|---|---|
| `utils/02_preprocess_psd.py:22-94,116-178` | SC4001/全队列 EEG 分期、Fpz-Cz、PSD 基础函数 | 在用，但需修正/锁定分期与 window contract |
| `models/s4_personalize_fig7.py:130-155,388-398` | 最早完整 8 参数 V1 Fig.7 拟合 | 历史基线，可读参考 |
| `models/s4_personalize_fig7_v7.py:153-214,230-278,399-689,869-1039` | V7 模型、8 参数 bounds、T1-T12、fitness、simulation | 成熟 baseline；adapter 的主要来源 |
| `models/s4_personalize_fig7_v8.py:76-137,145-202,333-688,876-1064` | V8a 模型、T1-T13、cortical spindle observability | 机制候选；当前仅本地未跟踪 |
| `S4_v7_repair/compute_pac_metrics_fixed.py:79-120,158-174,228-260` | simulator 的 cycle-by-cycle PAC repair | 可复用算法，但字段命名需修正 |
| `S4_v7_repair/compute_pac_metrics_eeg_native.py:4-20,31-147` | EEG-native T6 与单通道 PAC | observation 参考；不能直接宣称与 simulator 同构 |
| `valid_scripts/validate_spindle_density_heldout.py:51-65,101-177,199-249` | 独立 sigma-RMS spindle-density 检查 | 可重构为共享 validation 函数 |
| `valid_scripts/validate_so_waveform_heldout.py:56-68,115-218,293-444` | 独立 SO waveform 检查 | 可重构为共享 validation 函数 |
| `S4_sbi/compute_xobs_from_eeg_v3.py:84-109,115-255,319-349` | 当前已用 5-summary observation | 旧 SBI 基线；算法片段可复用，contract 不应继承 |
| `S4_sbi/compute_xobs_from_eeg_v4.py:91-119,125-177,225-324,422-460` | 计划中的 7-summary EEG-native observation | 实验代码；脚本文档与实际 wrapper 不一致 |
| `S4_sbi/simulator_wrapper.py:78-99,111-227` | 4D theta -> V7 -> 5 summaries | 需要重构，不能作为统一 adapter |
| `S4_sbi/run_sbi.py:130-135,592-660` | 4 轮 sequential SNPE-C，共 5000 simulations | 训练/诊断片段可参考，推断设计不继承 |
| `S4_sbi/sbi_outputs/` | theta `(5000,4)`、x `(5000,5)` 与四轮 checkpoint | 结果归档，不是新 8D ensemble 训练数据 |
| `S4_sbi/sbi_outputs_7dim_archive_20260507/` | theta `(4000,4)`、x `(4000,7)` 的旧 run | 结果归档，不进入新 contract |
| `outputs/evolution_fig7_v*_records.csv`、`data/patient_params_fig7_v*.json` | DE 拟合记录与单个最优/近优候选 | fitting candidates；不能称为 posterior samples |
| `S4_v7_repair/`、`warm_start_de/`、`reevaluate_v7/` | V7 PAC 修复、warm-start 和重评估 | 仍有研究价值，但不是 SBI posterior |
| `models/s4_personalize_fig7_v1-v6.py`、旧 plot/scan 脚本 | 历史实验与对比 | 保留归档，避免复制进新 pipeline |

`CODEBASE_INVENTORY.md:15-26,32-69,92-168` 对大部分目录已有分类，但它生成于 V8a 之前；其中 `README.md`/`CLAUDE.md` 对 summary 维度也已经过时。例如 `CLAUDE.md:142-144` 仍称 `compute_xobs_from_eeg.py` 和 wrapper 为 8 维，而当前实际 wrapper 明确只有 5 维（`S4_sbi/simulator_wrapper.py:4-22,88-99`）。后续应以可执行代码和有 schema 的 artifact 为事实源。

## 4. V8a/T13 全范围搜索结果

### 4.1 搜索范围

本轮搜索了：

1. 当前工作树全部可读文本，包括 tracked、modified、deleted 状态和 untracked 文件；宽模式为 `V8|V8a|T13|cortex-visible spindle|c_th2ctx|th2ctx`。
2. 所有当前可见 refs：`main`、`origin/main`、`publish-main`，使用 Git object 搜索，不假设 GitHub main 最新。
3. 全部本地 Git log 与文件名历史。
4. 相关 `.py/.md/.txt/.json/.csv/.yaml/.ipynb` 文本；宽搜索命中 145 个当前工作树文件，其中大量只是早期 `c_th2ctx` 记录和输出。

结论：可见 Git refs 中没有 V8a 实现。`main` 与 `origin/main` 对精确 `V8a|T13|cortex-visible spindle` 只有 `S4_v7_repair/diagnose_v7_phase.py:547` 一句“未来可加 T13”的建议；`publish-main` 为 0 条。当前实际 V8a/T13 代码、文档、参数和结果全部是本地未跟踪文件，所以本地工作树明显比可见 `origin/main` 更新。没有执行远程 fetch，因此结论严格限定为“当前本地可见 refs”。

### 4.2 实现与分析候选文件

下表时间为本地文件修改时间（America/New_York）；全部显示为 Git `??`：

| 候选文件 | 修改时间 | 相对 V7 / 输入输出 | 可直接运行性判断 |
|---|---:|---|---|
| `models/s4_personalize_fig7_v8.py` | 2026-07-05 17:56 | V7 + T13；`c_th2ctx` 上限 0.075；读 manifest/EDF，写 `data/patient_params_fig7_v8_SC4001.json` 与 `outputs/evolution_fig7_v8_records.csv`（`101-136,333-348,617-680,1191-1223`） | 语法可解析；直接 main 会启动约 4960 次完整 DE，无 smoke CLI，不适合直接运行 |
| `data/patient_params_fig7_v8_SC4001.json` | 2026-07-06 08:48 | V8a full-DE 的 8 参数与 12/13 记录 | 数据，不可执行；是 fitting result，不是 posterior sample |
| `docs/implementation_plan_v8.md` | 2026-07-05 17:47 | V8a/T13 设计、唯一 prior 扩展、validation 计划（`14-134,136-217`） | 文档，不可执行 |
| `docs/implementation_plan_v8_progress_1.md` | 2026-07-12 12:22（本节追加前） | 全 DE、trade-off、held-out、Figure-10-inspired 审计记录（`440-505,1470-1688,1826-2102`） | 文档，不可执行 |
| `docs/resume_0711.md` | 2026-07-11 10:35 | 恢复运行注意事项；记录 local-DE rerun 可能重复 block | 文档，不可执行 |
| `valid_scripts/analyze_v8a_tradeoff.py` | 2026-07-06 09:44 | 读 V8 CSV/JSON/log，写 `outputs/v8a_diagnostics/` 表和图（`25-28,57-83,332-426`） | 可直接做轻量 post-hoc，但会覆盖/更新诊断输出 |
| `valid_scripts/sweep_v8a_coupling_tradeoff_long.py` | 2026-07-06 10:13 | 读 full-DE 与 near-12 records；最多 2500 次模拟，写 4 phase CSV/图/summary（`46-56,98,764-809`） | 技术上可运行；重型、自动续 phase、会写输出，本阶段禁止 |
| `valid_scripts/local_de_v8a_t6_rescue_long.py` | 2026-07-06 18:47 | 读 full-DE/coupling sweep；最多 6000 次、11.5 h，持续 checkpoint（`40-48,83-85,494-548`） | 技术上可运行；重型且会追加记录，本阶段禁止 |
| `valid_scripts/local_de_v8a_t6_rescue_narrow.py` | 2026-07-07 13:52 | 读 long local-DE/coupling；最多 4500 次、10.5 h（`42-51,86-88,486-539`） | 技术上可运行；重型且 resume 语义不足，本阶段禁止 |
| `valid_scripts/diagnose_v8a_t6_t13_coupling_sweep.py` | 2026-07-11 16:54 | 读 V7/V8a/narrow 候选；默认 31x25 grid，写 coupling CSV/summary（`42-49,78-87,270-398`） | 有 `--max-sims/--resume`；不带 `--resume` 会删除旧 CSV（`275-277`），不能随意直接跑 |
| `valid_scripts/local_search_v8a_ultra_narrow_t6_t13.py` | 2026-07-11 17:03 | 读 coupling sweep；最多 1600 次、8 h，写 local-DE checkpoints（`38-44,83-96,440-496`） | 技术上可运行；重型并会继续/追加，本阶段禁止 |
| `valid_scripts/analyze_v8a_relaxed_t6_sensitivity.py` | 2026-07-11 21:51 | 只读 7 组现有 CSV，写 relaxed sensitivity 表与解释（`3,18-30,287-325`） | 可轻量直接运行，但会覆盖 post-hoc 输出；不改变 strict V8a |
| `valid_scripts/plot_v8a_0712_best0417_figures.py` | 2026-07-12 00:41 | 读 coupling/relaxed CSV，选择 0.42 candidate，生成图、metrics、manifest，并含短候选模拟（`62-106,257-531`） | 可运行但会模拟并覆盖图；不是 validation pipeline |
| `valid_scripts/plot_v8a_0712_heldout_spindle_density.py` | 2026-07-12 00:48 | 读 EEG、V1/V7/V8a 参数；各做 120 s simulation，写 held-out 图/CSV/summary（`39-53,83-128,303-339`） | 可运行但无 dry-run，且会做多次 120 s simulation；本阶段未运行 |
| `valid_scripts/plot_figure10_inspired_validation_panel.py` | 2026-07-12 12:20 | 汇总 V1-V8/V8a fitting candidate archive；默认最多 50x3 次短模拟，写 archive/predictive/recovery/图（`67-89,115-279,756-812`） | 可运行且能复用缓存，但不是 NPE、SBC 或 L-C2ST；本阶段未运行 |

上述 11 个 Python 候选均已在 `neurolib` 环境中做 AST 语法解析并通过；没有 import 它们的 main、没有运行 simulation。`outputs/v8a_*`、`outputs/evolution_fig7_v8_records.csv` 和 `outputs/figure10_inspired_validation_panel/` 是这些脚本的结果产物，不是新的实现候选。

### 4.3 当前最新结果的科学状态

- full DE 共 4960 evaluations，严格 13/13 为 0，12/13 为 105；原始 best 只失败 T6（`docs/implementation_plan_v8_progress_1.md:449-495`）。
- 后续 `best0417` 来自 `outputs/v8a_t6_t13_coupling_sweep.csv` row 2058：T6 IBI CV=0.417，T13 verified density=1.091/min，仍是严格 12/13；只有把 T6 阈值从 `<0.40` post-hoc 放宽到 `<0.42` 才成为唯一 relaxed-feasible candidate（`docs/implementation_plan_v8_progress_1.md:1470-1562`）。
- 独立 sigma-RMS held-out：真实 EEG=2.958/min，V1=1.043/min，V7=0，V8a best0417=0；因此 T13 的内部 permissive gate 没有转移到外部 detector（`docs/implementation_plan_v8_progress_1.md:1637-1688`）。
- 当前 Figure-10-inspired panel 使用的是 fitted-candidate ensemble，项目明确还没有 posterior estimator，不能把这些候选称为 posterior samples（`docs/implementation_plan_v8_progress_1.md:1826-1836,2091-2102`）。

## 5. V7 与 V8a/T13 代码差异

| 项目 | V7 baseline | V8a/T13 candidate | 审计结论 |
|---|---|---|---|
| 参数顺序 | 8D：`mue,mui,b,tauA,g_LK,g_h,c_th2ctx,c_ctx2th`（`models/s4_personalize_fig7_v7.py:178`） | 完全相同（`models/s4_personalize_fig7_v8.py:101`） | 可以共享统一 theta schema |
| 动力学构建 | ALN cortex + thalamus；connectivity `[[0,c_th2ctx],[c_ctx2th,0]]`（V7 `230-278`） | 构建逻辑相同（V8 `145-202`） | adapter 不应复制模型 pipeline |
| prior/support | `c_th2ctx=[0,0.05]`，其余见 `205-214` | 仅 `c_th2ctx` 扩到 `[0,0.075]`，其余相同（`109-137`） | 唯一正式 support 差异 |
| 约束 | T1-T12（`399-689`） | T1-T12 + T13 cortical event/verified/density（`333-348,617-688`） | T13 应保留为 mechanism diagnostic |
| T13 detector | 无 | cortex `r_ctx` 10-14 Hz、Hilbert envelope、200 ms smoothing、75th percentile、0.5-3 s、event PSD ratio>1.5，至少 1 verified event（`617-680`） | 与 held-out 11-15 Hz RMS detector 不同 |
| reward | `0.50 shape + 0.25 SO + 0.25 thalamic spindle`（`379-381,942-1002`） | `0.40 shape + 0.20 SO + 0.15 thalamic + 0.25 cortical`（`312-315,951-1017`） | 只影响 DE；不应进入 SBI likelihood contract |
| infeasible scaling | `sum(scores)/12`（`1000-1002`） | `/13`（`1015-1017`） | 版本诊断差异 |
| 输出 | V7 JSON/CSV | V8 JSON/CSV + T13 fields（`1050-1064,1191-1223`） | 都是 fitting result，不是 posterior |

额外发现一个跨 V7/V8a 的高优先级语义 bug：注释和常量把 T11 描述为 `SO leads spindle` 且阈值为 20 ms（V7 `371-376`；V8 `304-309`），但实际实现比较的是 `up_down_ratio >= 1.20`，随后把该无量纲比值写入 `T11_lag_ms`（V7 `638-641`；V8 `572-575`）。`compute_pac_metrics_fixed` 也明确定义 `up_down_ratio` 是 UP/DOWN amplitude ratio（`S4_v7_repair/compute_pac_metrics_fixed.py:53-63,246-260`），并没有返回 lag。V7/V8 的 soft-score 代码又把这个 ratio 当成 ms 与 20 比较（V7 `771-781`；V8 `770-780`）。新 summary contract 必须重命名为 `pac_up_down_ratio`，不能继续声称它是 lag_ms；若确实需要 directionality lag，必须另写并验证真正的时滞估计。

V8a 还有一个结果解释陷阱：shape/SO/spindle rewards 只在全部约束通过后计算（`models/s4_personalize_fig7_v8.py:951-1017`），所以当前 infeasible JSON 中四个 reward 都是 0（`docs/implementation_plan_v8_progress_1.md:472-488`）并不表示真实 spectral correlation 或功率恰好为 0。

## 6. 旧 S4_sbi：可复用、需重构、不应继承

### 可以复用

- `BoxUniform`、SNPE/NSF 初始化、simulation batch、NaN guard、checkpoint、PPC/plot 的基本调用方式（`S4_sbi/run_sbi.py:100-135,233-254,592-668`）。
- EEG manifest/channel/artifact 读取片段，以及将 extraction metadata 写入 artifact 的做法（`S4_sbi/compute_xobs_from_eeg_v3.py:115-180,319-349`）。
- V7 `build_model`、seed helper、raw `r_ctx/r_thal` 提取和 burn-in 约定（`models/s4_personalize_fig7_v7.py:230-278,899-939`）。
- held-out detector 的纯计算部分，重构后供 observation 与两个 adapter 共用（`valid_scripts/validate_spindle_density_heldout.py:154-177`；`valid_scripts/validate_so_waveform_heldout.py:217-293`）。

### 必须重构

- 从 4D theta 改为严格 8D schema。旧 wrapper 固定 Seed B 四参（`docs/sbi_obs_vs_sim_comparison.md:130-150`），不满足项目目标。
- observation loader、simulation、summary、validation 分层。旧 wrapper import 时就读取真实 EEG/FOOOF target（`S4_sbi/simulator_wrapper.py:78-85`），有隐藏 I/O 和全局状态。
- summary 必须由同一个纯函数作用于显式 signal object；不能 observation 和 simulator 各写一套同名不同义实现。
- seed 必须作为调用参数。旧 wrapper 每次固定 seed 42（`docs/sbi_obs_vs_sim_comparison.md:139-147`），无法表达 simulator stochasticity。
- circular phase 应使用 `sin/cos` 或 circular distance；不能把角度当普通实数直接 z-score。
- simulation artifacts 需要记录 version、theta schema hash、summary contract version、seed、采样率、duration、burn-in、代码 commit/worktree provenance。

### 不应继承

- `shape_r=1.0` observation sentinel 对 simulator 的 EEG-target correlation。这一维在 observation 与 simulation 上不是同一个统计量（`docs/sbi_obs_vs_sim_comparison.md:89-101,152-176`）。
- `T11_lag_ms` 的错误名称和错误物理解释。
- v3 的 5-summary contract 或 v4 尚未落盘的 7-summary contract。v4 自称与 wrapper 一致（`S4_sbi/compute_xobs_from_eeg_v4.py:6-18`），但当前 wrapper 仍是 5 维（`S4_sbi/simulator_wrapper.py:4-22,88-99`），磁盘也没有 `x_obs_v4.npz`（`CODEBASE_INVENTORY.md:201`）。
- 把 4 个 round checkpoints 当作 4 个或 5 个独立 NPE。旧代码在同一个 `inference` 对象上逐轮 `append_simulations/train`，上一轮 posterior 作为下一轮 proposal（`S4_sbi/run_sbi.py:582-607`）；它是一个 sequential SNPE-C 过程的 checkpoint 序列，不是独立 ensemble。
- 旧 prior 的 `b` 上限 80。旧 4D prior 为 `[g_h,g_LK,c_ctx2th,b]`，其中 b=`[28.4,80]`（`S4_sbi/run_sbi.py:130-134`），与 V7/V8 模型正式 bounds `[28.4,42.6]` 不一致（V7 `205-214`；V8 `128-137`）。
- 已归档的 7D simulation bank。它的 observation/simulator coordinate 问题没有通过改维数解决（`docs/sbi_obs_vs_sim_comparison.md:178-184,232-240`）。

## 7. 八个推断参数与 prior/support

统一参数顺序必须固定为：

```text
[mue, mui, b, tauA, g_LK, g_h, c_th2ctx, c_ctx2th]
```

代码事实源是 V7 `models/s4_personalize_fig7_v7.py:178,205-214` 和 V8a `models/s4_personalize_fig7_v8.py:101,128-137`：

| 参数 | 生理/模型含义 | 项目内单位 | V7 prior/support | V8a/T13 prior/support |
|---|---|---|---:|---:|
| `mue` | cortex excitatory background input | mV/ms（`models/s5_bifurcation.py:40-47`） | [3.31075, 4.47925] | 同 V7 |
| `mui` | cortex inhibitory background input | mV/ms（同上） | [2.57295, 3.48105] | 同 V7 |
| `b` | excitatory adaptation current amplitude | pA（`docs/0315_Progress.md:129-132`） | [28.4, 42.6] | 同 V7 |
| `tauA` | adaptation time constant | ms（`docs/0315_Progress.md:129-132`） | [998.2, 1853.8] | 同 V7 |
| `g_LK` | TCR potassium leak conductance | 项目语义为 conductance；建议在 contract 中锁定 neurolib 的 mS/cm^2 | [0.020, 0.070] | 同 V7 |
| `g_h` | TCR h-current conductance | 同上 | [0.035, 0.095] | 同 V7 |
| `c_th2ctx` | thalamus -> cortex coupling coefficient | 项目未声明物理单位，按模型耦合系数记录 | [0.000, 0.050] | [0.000, 0.075] |
| `c_ctx2th` | cortex -> thalamus coupling coefficient | 同上 | [0.050, 0.220] | 同 V7 |

建议的比较策略：主比较使用两版共同 support，即 `c_th2ctx=[0,0.05]`，使 V7 与 V8a 的 prior predictive/global validation 可直接比较；另开明确标记的 V8a mechanism-extension 实验使用 `[0,0.075]`。如果直接让两个版本使用不同 prior，posterior 差异会混合 mechanism 与 prior-support 差异。最终选择需要用户确认。

## 8. 当前 EEG 与 simulator summary 定义、单位和来源

### 8.1 Observation 侧

当前真正落盘并被旧 `run_sbi.py` 默认读取的是 v3 5D：`x_obs_v3.npz=[1.0,2.645,0.75,15.31,1.28]`，代码定义在 `S4_sbi/compute_xobs_from_eeg_v3.py:101-109,194-251`，默认加载在 `S4_sbi/run_sbi.py:592`。

| 字段 | 单位 | observation 信号/算法 |
|---|---|---|
| `shape_r` | 无量纲 | 固定 1.0 sentinel，不是从同一 summary 函数计算 |
| `T4_q` | 无量纲 | `r_proxy=abs(EEG)->50 ms smooth->p95 rescale` 的 SO PSD Q factor |
| `T4_freq` | Hz | 同一 `r_proxy` SO peak frequency |
| `T8_n_sp_events` | events/60 s | signed raw EEG 的 10-14 Hz envelope events |
| `T11_lag_ms` | 实际无量纲 | `r_proxy` phase x raw EEG spindle amplitude 的 `up_down_ratio` |

v4 计划为 7D，增加 EEG-native `T6_ibi_cv` 和 `MI`，并把 T11 改成单通道 EEG-native up/down ratio（`S4_sbi/compute_xobs_from_eeg_v4.py:225-324`）：

| 新字段 | 单位 | observation 信号/算法 |
|---|---|---|
| `T6_ibi_cv` | 无量纲 | raw EEG 上 0.2-4 Hz AASM-style half-wave/UP events 的 IBI CV |
| `MI` | 无量纲 | raw EEG 单通道 0.5-1.5 Hz phase x 10-14 Hz amplitude 的 Tort MI |

两版 observation 都先拼接通过 200 µV peak-to-peak QC 的 N3 epochs，再把约 100 Hz EEG 上采样到 1000 Hz（`S4_sbi/compute_xobs_from_eeg_v3.py:115-180`；v4 `125-221`）。上采样只统一离散时间网格，不增加观测信息。

### 8.2 Simulator 侧

当前 wrapper 只输出 5D（`S4_sbi/simulator_wrapper.py:88-99,111-163`）：

| 字段 | 单位 | simulator 信号/算法 |
|---|---|---|
| `shape_r` | Pearson r，无量纲 | cortical `r_ctx` PSD periodic component 与真实 EEG average-PSD target 的相关 |
| `T4_q` | 无量纲 | cortical `r_ctx` PSD 的 SO Q factor |
| `T4_freq` | Hz | cortical `r_ctx` SO peak frequency |
| `T8_n_sp_events` | events/60 s | thalamic `r_thal` event detector |
| `T11_lag_ms` | 实际无量纲 | cross-signal `r_ctx` phase x `r_thal` amplitude 的 up/down ratio |

V7/V8a constraint functions内部还产生 T6、T9 MI、T10 phase/concentration、T12 verified thalamic spindle；V8a 再产生 T13 cortical event counts/density（V7 `399-689`；V8 `333-688`）。这些是 mechanism/feasibility diagnostics，目前不是与 EEG 同坐标的 inference summary。

## 9. 现有 coordinate mismatch

1. **物理信号不一致**：真实 observation 是 Fpz-Cz scalp voltage [µV]，simulator 是 cortex/thalamus population firing rate [Hz]。项目没有显式 EEG forward/observation model；只能对 scale-invariant 特征做有限比较，不能把 `r_ctx` 直接称为 simulated EEG（`models/s4_personalize_fig7.py:272-285,552-557`）。
2. **同名字段来源不同**：T4 是 `r_proxy` vs `r_ctx`；T8 是 scalp EEG vs thalamus；T11 是混合单/双信号；T6 和 MI 的 observation/simulator detectors 也不同（`docs/sbi_obs_vs_sim_comparison.md:167-184`）。
3. **shape_r 不是同一统计量**：observation 固定为 1，simulation 则拿每个 r_ctx 与真实 EEG target 比较，导致 PPC 的该维没有 exchangeability（`docs/sbi_obs_vs_sim_comparison.md:89-101,152-176`）。
4. **T11 坐标和单位错误**：字段叫 lag_ms，数据却是 amplitude ratio；这是 schema bug，不只是命名偏好（V7 `638-641`；`compute_pac_metrics_fixed.py:246-260`）。
5. **epoch/边界处理不同**：observation 把非连续 N3 epochs 拼成长信号后做 event/filter/PAC；这会在拼接点制造滤波和事件边界。V7 target PSD 则先逐 epoch 求 PSD 后平均（`docs/sbi_obs_vs_sim_comparison.md:117-128,187-203`）。新 contract 应保持 segment-aware，不跨 epoch 过滤或配对事件。
6. **detector transfer 失败**：V8a T13 内部 10-14 Hz percentile/Hilbert detector 检出 1.091/min，但独立 11-15 Hz RMS detector 对同候选为 0/min（`docs/implementation_plan_v8_progress_1.md:1637-1670`）。T13 不宜直接作为 observation summary。
7. **时长与随机性不一致**：observation 约 4260 s，旧 simulator 使用 55 s post-burn-in 且固定 seed 42（`docs/sbi_obs_vs_sim_comparison.md:54-65,139-150`）。event rate、CV、PAC 的估计方差不可比。
8. **分期映射可能错误**：完整标签把 `Sleep stage 3/4` 都映射 N3，但短标签表把 `"3"` 映射为 N2、`"4"` 映射 N3（`utils/02_preprocess_psd.py:31-45`）。必须先查 SC4001 annotation 实际走哪条路径。
9. **Welch window 文档/实现不一致**：模块声明 paper 使用 Hamming，代码实际 `window="hann"`，并在注释里承认差异（`utils/02_preprocess_psd.py:5-27,85-92`）。Observation Notebook 前需锁定复现标准。
10. **prior coordinate 不一致**：旧 SBI 是 4D 且 b 上限 80，新目标是 8D 且正式 model bound 上限 42.6（`S4_sbi/run_sbi.py:130-134`；V7 `205-214`）。旧 simulation bank 不能直接混用。
11. **FOOOF fallback 不等价**：项目说明 FOOOF 是否可用会改变结果（`CLAUDE.md:88-92`）；当前环境有 fooof 1.1.1，后续应强制记录并禁止 silent algorithm fallback。

## 10. 统一 SimulatorAdapter 方案

本阶段只设计，不实现。核心原则是一个 pipeline、两个薄 adapter、一个 summary contract：

```python
Theta8 = [mue, mui, b, tauA, g_LK, g_h, c_th2ctx, c_ctx2th]

SimulatorAdapter(version="v7" | "v8a_t13")
    .validate_theta(theta8)
    .simulate(theta8, seed, duration_s, burn_in_s) -> SimulationRecord

SimulationRecord = {
    "version": str,
    "theta8": float[8],
    "seed": int,
    "fs_hz": float,
    "duration_s": float,
    "burn_in_s": float,
    "cortex_rate_hz": array,
    "thalamus_rate_hz": array,
    "valid": bool,
    "diagnostics": dict,
}

compute_summary(SignalBatch, SummaryContractV1) -> SummaryVector
```

设计约束：

- registry 只保存 `build_model`、version support、diagnostic function：V7 调 `compute_constraints_v7`，V8a 调 `compute_constraints_v8`。共同 output 不随版本变化。
- `diagnostics` 可以分别含 T1-T12/T1-T13，但 inference summary 不从版本专用约束函数取同名字段。
- observation loader 产出 segment-aware `SignalBatch`；simulator adapter 也按与 EEG 对齐的 segment 长度切块。共享 `compute_summary` 对两者调用完全相同的滤波、事件、PSD、PAC 实现。
- 在没有 forward model 前，`cortex_rate_hz` 必须按原名保留；可配置 observation operator 生成 `observable`，但不能默认为“EEG”。第一版 summary 应优先选择频率、相对/标准化谱、event rate、duration、CV、MI、phase sin/cos 等 scale-invariant 特征。
- 建议 SummaryContractV1 至少显式包含：SO peak Hz、SO relative power/Q、SO event rate/min、IBI CV、observable-channel spindle density/min、median duration s、sigma/non-sigma relative power、PAC MI、preferred-phase sin/cos。每个字段要固定 unit、band、detector、normalization、aggregation 与 invalid policy。
- T8 thalamic event、T12 thalamic verified spindle、T13 cortical audit 作为 latent/mechanism diagnostics 保留，不冒充 scalp-observable summaries。
- adapter 必须接收 seed，不能硬编码 42；失败返回结构化 error/mask，而不是无上下文的全 NaN。
- V7 与 V8a 建议分别训练 5 个独立 NPE（独立初始化和数据 split，可共享同一合格 simulation bank），再构成等权或 validation-weighted ensemble。它们不与旧 4 轮 sequential checkpoints混淆。

## 11. 推荐的新目录结构

```text
S4_sbi/
  README.md
  configs/
    v7.yaml
    v8a_t13.yaml
    summary_v1.yaml
  src/sleep_sbi/
    schemas.py
    parameters.py
    observation.py
    simulator.py
    adapters/
      v7.py
      v8a_t13.py
    summaries/
      contract.py
      spectral.py
      events.py
      pac.py
    inference/
      prior.py
      simulate.py
      train_ensemble.py
    validation/
      support.py
      global_validation.py
      local_c2st.py
      ppc.py
    interpretation.py
  notebooks/
    01_s4_sbi_figure10_workflow.ipynb
  tests/
    test_parameter_schema.py
    test_adapter_contract.py
    test_summary_parity.py
    test_epoch_boundaries.py
    test_seed_reproducibility.py
  artifacts/
    observations/
    simulations/
    npe/
    validation/
  legacy/
```

后续不应复制 V7/V8a 两套 pipeline；`adapters/` 只封装版本差异，observation、summaries、inference 和 validation 都是共享模块。现有 `S4_sbi/legacy/` 和 outputs 在得到确认前不移动。

## 12. 教学型 Notebook 逐章节实施计划

1. **Observation 可视化**：环境/seed/provenance；SC4001 hypnogram、N3 epoch QC、Fpz-Cz 原始片段、PSD、SO/sigma bands；展示保留/剔除数量，不跨 epoch 过滤。
2. **共享 summary contract**：表格化字段、单位、信号来源、算法；在同一 cell 对 EEG `SignalBatch` 和两个 adapter 输出调用同一个 `compute_summary`；显示 invalid mask 与 parity tests。
3. **Prior predictive support check**：先做批准后的小规模 pilot，分别报告 V7 common support、V8a common support 和 V8a extended support 的 failure rate、marginal/joint summary coverage、x_obs tail position 与 classifier support diagnostic；支持不足时停止，不训练 NPE。
4. **Ensemble of five NPEs**：在 support check 通过并获批后，每个版本训练 5 个独立 NPE；明确 train/validation split、network seed、simulation provenance 和 ensemble aggregation。旧四轮 sequential posterior 不计入这 5 个网络。
5. **Global validation**：prior-drawn synthetic truths 上做 expected coverage、SBC rank/ECDF、TARP 或等价全局校准，并按参数/版本分层。
6. **Local validation / L-C2ST**：在真实 `x_obs` 邻域评估局部 posterior correctness；说明 classifier split、null calibration、p-value/score 和重复随机种子。
7. **Posterior predictive checks**：从真正的 ensemble posterior 抽 theta，经 adapter 和共享 summary/observable pipeline生成 replicated data；同时展示 summary-level 和 waveform/event-level PPC。
8. **Physiological interpretation**：8 参数 marginal/joint、不确定性和可辨识性；比较 V7 baseline 与 V8a/T13 mechanism candidate；将 T8/T12/T13 latent diagnostics 与 scalp-observable evidence 分栏解释。

Notebook 首页必须明确区分：Stage-1 DE fitting candidates、NPE posterior samples、posterior predictive simulations。当前 V1-V8/V8a archive 只能用于审计和初始化范围讨论，不能冒充 posterior。

## 13. 阻塞项与需要确认的问题

在开始 Observation Notebook 前，需要确认以下事项：

1. **Observation operator**：第一版是否接受“对 EEG 与 cortical rate 分别做 robust normalization 后计算 scale-invariant summary”，还是必须先建立显式 cortex-rate -> EEG forward model？这是最大的 coordinate blocker。
2. **共同 prior**：V7/V8a 主比较是否采用交集 `c_th2ctx=[0,0.05]`，并把 V8a `[0.05,0.075]` 作为独立 mechanism-extension study？
3. **分期规则**：确认短标签 `"3"` 应映射 N3，并核对 SC4001 annotation 实际标签；当前 `utils/02_preprocess_psd.py:31-45` 有冲突。
4. **Welch window**：严格按论文改用 Hamming，还是冻结当前 Hann 作为项目 contract？
5. **summary 字段**：同意把 `T11_lag_ms` 废弃并改为 `pac_up_down_ratio`，且把 T8/T12/T13 留在 mechanism diagnostics，而不是 inference summary 吗？
6. **spindle detector**：共同 observable summary 是采用 held-out 11-15 Hz RMS detector，还是重新校准一个对 EEG 与 normalized cortical observable 都适用的 detector？现有 T13 不能直接承担该角色。
7. **segment/duration**：建议保持 30 s epoch 边界、按 epoch 求 summary 后做稳健聚合，并让 simulator 生成相同数量/长度的 segments；需要确认 simulation 预算和是否允许随机重复。
8. **五网络定义**：建议每个机制版本各自 5 个独立 NPE；共享合格 simulation bank，但独立 initialization/data split。请确认不是训练一个把 `version` 当离散参数的混合模型。
9. **严格 V8a 状态**：是否接受“V8a 是 mechanism candidate/审计补丁，严格 13/13 仍未成功，held-out RMS 仍失败”作为 Notebook 的固定表述？
10. **历史文件处理**：新实现开始后，是否允许只在文档中标记 legacy，而暂不移动现有 untracked/modified 文件？当前工作树很脏，先建立 provenance snapshot 比整理目录更安全。
11. **FOOOF 与单位**：是否锁死 fooof 1.1.1、禁止 fallback，并从本地 neurolib 参数定义最终确认 `g_LK/g_h` 的单位后再发布 contract？

## 14. 第一阶段结论

第一阶段审计已完成。V7 可作为成熟 baseline；V8a/T13 的本地实现和完整实验链已找到，但它是未提交的机制候选，严格 13/13 和独立 cortical spindle-density held-out 都没有通过。旧 S4_sbi 证明了 SNPE-C 工程链能运行，但它是 4 参数、5-summary、4 轮 sequential 推断，存在 observation/simulator coordinate mismatch，不能直接扩写成目标 notebook。

下一步应在用户确认上述决策后，从 Observation Notebook 与共享 schema/summary contract 开始；在 prior predictive support check 通过之前不训练任何 NPE。

# 第十五部分：2026-07-23 Pre-Observation 本地成果保护检查

## 本阶段目标

在开始 Observation Notebook 前，为当前 V7、V8a/T13、PAC repair、held-out validation 与 S4_sbi 状态建立可追溯 checkpoint。本阶段没有实现 Notebook、没有重构科研代码、没有运行 simulation、没有训练 NPE，也没有安装或修改依赖。

## 已完成的只读检查

1. 仓库根目录、README、`.gitignore`、`CLAUDE.md`、Git refs、remote 和 dirty worktree 已检查；仓库内没有 `AGENTS.md`。
2. 当前原始状态为：

```text
branch = main
HEAD = 82aea491f757b101e702c919dc98aa95d95e11ec
origin/main = 82aea491f757b101e702c919dc98aa95d95e11ec
visible branches = main, publish-main, origin/main
git conflicts = none
```

3. 所有 Python 检查均使用 `neurolib` conda 环境。版本为：

```text
Python 3.10.20
neurolib 0.6.1
sbi 0.26.1
torch 2.5.1
numpy 2.2.6
scipy 1.15.2
mne 1.9.0
fooof 1.1.1
```

4. ignored 文件共 492 个。其中 `data/` 约 7.60 GB，包含 153 个 EDF、合计约 7.24 GiB；这些原始 EEG 明确排除。`outputs/` 有 146 个 ignored 文件、约 37.2 MB，V8a/T13 的重要结果已按大小、用途和可复现性进一步分类。
5. 15 个拟保护的 V8a/S4_sbi/validation Python 文件已在 `neurolib` 环境中通过 AST parse；没有 import main，也没有执行 simulation。
6. 没有发现 private key、GitHub/OpenAI token、AWS key 或显式 credential assignment。

## 保护性 manifest

已创建：

```text
docs/checkpoints/pre_observation_checkpoint_20260723.md
```

manifest 包含：

- 原 branch、HEAD、origin/main、remote 与 dirty worktree 摘要；
- V8a/T13 源代码、分析脚本和小型关键 artifact 的用途、字节数与 SHA-256；
- 7 个重要 full-DE/coupling/local-search CSV 的 SHA-256；
- S4_sbi legacy 原路径与当前副本的 Git blob 一致性；
- ignored 原始 EEG、cache、图像和大结果的排除理由；
- neurolib 环境与关键包版本；
- V7/V8a/T13 当前科学状态；
- checkpoint commit 前的安全阻塞项。

## A-D 分类结论

### A 类：安全检查通过，拟纳入 checkpoint

主要包括：

```text
models/s4_personalize_fig7_v8.py
data/patient_params_fig7_v8_SC4001.json
V8a/T13 trade-off、local-search、relaxed-sensitivity 与 Figure10-inspired 分析脚本
docs/implementation_plan_v8.md
docs/sbi_obs_vs_sim_comparison.md
CODEBASE_INVENTORY.md
必要的小型 best JSON、summary TXT/MD 和 aggregate CSV
本次 checkpoint manifest
```

这些文件目前尚未暂存。

### B 类：保留但不直接提交

包括：

```text
outputs/evolution_fig7_v8_records.csv
outputs/v8a_t6_t13_coupling_sweep.csv
各 long/narrow/ultra local-DE records
可重新生成的 PNG/PDF
NPZ 与 Python cache
```

关键 CSV 和 NPZ 已在 manifest 中逐项记录 SHA-256。V8/Figure10 可重生成 media 共 28 个、约 4.73 MB，并记录了排序清单的 aggregate inventory SHA-256。

### C 类：敏感或禁止提交

包括：

```text
原始 EDF EEG
data/manifest.csv
本地工具配置
包含用户目录或机器绝对路径的 progress/resume/log
含硬编码本地 neurolib 绝对路径的未确认代码修改
```

### D 类：需要用户确认

包括：

```text
S4_sbi 根目录到 legacy/ 的现有重排
held-out validation 的大量注释式修改
S4_sbi/compute_xobs_from_eeg_v4.py 的注释式修改
manuscript DOCX/MD/TEX/PDF/PPTX 与 build script
包含 pseudonymous subject/channel aggregate 的 real_observation_summaries.csv
```

其中 S4_sbi legacy 下的 6 个 Python 文件和 1 个 NPZ 与原 tracked deleted paths 的 Git blob 完全一致；没有内容丢失，但是否把该重排作为 rename commit 仍需确认。

## 安全闸门结果

本轮发现了：

1. 重要但未提交的文件包含机器绝对路径；
2. 多组用户修改的纳入范围不明确；
3. 单一 subject/channel aggregate 的公开边界未确认。

因此按要求停在 commit 之前：

```text
checkpoint branch created = no
files staged = none
checkpoint commit created = no
push executed = no
```

没有执行 `git reset`、`git clean`、`git restore`、`git checkout --`、`git stash`、删除、覆盖、移动、rebase 或 force push。

## 当前科学状态保持不变

```text
V7 = mature baseline
V8a/T13 = mechanism/audit candidate
strict 13/13 = not stably achieved
T13 internal detector != independent RMS detector
DE fitting candidates != posterior samples
old four sequential posterior checkpoints != five independent NPEs
```

## 等待确认的 checkpoint 范围

需要用户确认：

1. 是否只提交 manifest 第 5、6 节的 A 类文件，并排除所有 C、D 类；
2. 是否把 `S4_sbi/legacy/` 当前重排作为 rename 纳入；
3. 是否明确排除三份包含机器路径或注释大改的 tracked modifications；
4. 是否允许提交 `real_observation_summaries.csv`；
5. 是否允许对 ignored `outputs/` 中 manifest 第 6 节的小型关键摘要逐路径使用 `git add -f`。

确认前不会创建 branch、staging、commit 或 push，也不会开始 Observation Notebook。

# 第十六部分：2026-07-23 Checkpoint commit 前第二次安全闸门

## 用户确认后的执行

用户确认只提交 checkpoint manifest 第 5、6 节 A 类，排除 C/D、legacy 重排、held-out 注释式修改、S4_sbi v4 注释式修改、manuscript artifacts 和 `real_observation_summaries.csv`。

已完成：

```text
branch created = checkpoint/pre-observation-20260723
source HEAD = 82aea491f757b101e702c919dc98aa95d95e11ec
explicit staged paths = 51
staged/manifest set mismatch = 0
staged total bytes = 476372
largest staged file = 69967 bytes
credential hits = 0
disallowed binary/raw-data path hits = 0
```

暂存严格使用显式路径；ignored `outputs/` 只使用 manifest 第 6 节列出的 JSON、TXT、MD 和 aggregate CSV。没有使用 `git add .`、`git add -A` 或 wildcard。

## 第二次 staged-content 检查发现

机器绝对路径扫描命中 3 个此前误列为 A 类的 helper：

```text
valid_scripts/audit_v1_v3_v7_t12.py
valid_scripts/run_so_waveform_canonicalV1.py
valid_scripts/t5_deconfound.py
```

命中内容分别是 hard-coded repository root、local neurolib path 或 `os.chdir` path。绝对路径值没有写入本节或 checkpoint manifest。

因此这 3 个文件已重新分类为 C 类，必须从 commit 排除。文件 worktree 内容未被修改、移动或删除。

## 当前停止状态

```text
checkpoint branch = checkpoint/pre-observation-20260723
checkpoint commit = not created
push = not executed
Observation Notebook = not started
simulation = not run
NPE training = not run
```

由于原保护规则明确禁止 `git reset`、`git restore` 和 `rm`，本轮没有自行使用这些命令从 index 移除上述 3 个新文件，并按安全闸门停在 commit 前。

继续前需要用户明确授权执行仅影响 index、不影响 worktree 的 unstage 操作，例如：

```text
git restore --staged -- \
  valid_scripts/audit_v1_v3_v7_t12.py \
  valid_scripts/run_so_waveform_canonicalV1.py \
  valid_scripts/t5_deconfound.py
```

授权后需要重新验证 staged set、敏感路径、credentials、文件大小和 diff summary；全部通过后才能创建本地 checkpoint commit。

# 第十七部分：2026-07-23 Pre-Observation 本地 checkpoint 完成

## Checkpoint 结果

用户再次确认排除全部机器绝对路径文件后，已仅对 3 个命中文件执行 index-only unstage；worktree 文件保持原样。最终安全检查通过并创建本地 commit。

```text
branch = checkpoint/pre-observation-20260723
parent = 82aea491f757b101e702c919dc98aa95d95e11ec
checkpoint commit = fb3c46cab10a68e34d4ed59f1a864b2b447c2503
commit message = checkpoint: preserve pre-observation V8a T13 work
committed files = 48
insertions = 9156
push = not executed
```

## 最终安全检查

```text
expected safe files = 48
actual staged files before commit = 48
set mismatch = 0
staged total bytes = 464730
largest staged file = 69967 bytes
machine absolute path hits = 0
credential hits = 0
disallowed extension/path hits = 0
full-DE/local-search record hits = 0
```

`git diff --cached --check` 仅报告已有研究文件中的 trailing whitespace 和 EOF blank-line 警告。没有为了消除这些非安全性警告而格式化科研代码。

## 已提交内容

提交内容包括：

```text
V8a/T13 model source
V8a trade-off / coupling / local-search / relaxed-sensitivity scripts
Figure10-inspired 与 V8a held-out plotting/audit scripts
S4_sbi concat-vs-mean PSD diagnostic source
V8a parameter JSON
implementation、coordinate-audit 与 checkpoint Markdown
manifest 第 6 节批准的小型 JSON/TXT/MD/aggregate CSV
validation aggregate summaries
```

没有提交：

```text
raw EDF 或 data/manifest.csv
full-DE/local-search per-evaluation records
NPZ/PNG/PDF/PPTX/cache/log
real_observation_summaries.csv
含机器绝对路径的 3 个 helper
S4_sbi legacy 重排
held-out validation 注释式修改
S4_sbi/compute_xobs_from_eeg_v4.py 注释式修改
manuscript 与 build script 修改
本 progress 文档
```

## Commit 后 worktree

Git index 已清空。原有 dirty worktree 被保留，包括 S4_sbi 根目录的 tracked deletions 与 `legacy/` 未跟踪副本、明确排除的代码修改、manuscript artifacts、可重生成图像和 3 个含机器路径的 helper。没有回滚、移动、删除或覆盖这些成果。

## 远程备份建议

建议在用户明确确认后把 checkpoint branch 推送到 origin。拟执行命令：

```text
git push -u origin checkpoint/pre-observation-20260723
```

本轮没有执行 push。Observation Notebook 尚未开始，也没有运行 simulation 或训练 NPE。

# 第十八部分：2026-07-23 Observation Notebook 第一阶段实现

## Checkpoint 推送与开发分支

已按用户确认执行 checkpoint 远程备份，并通过只读 `git ls-remote` 验证远程 branch 精确指向原 commit：

```text
remote branch = origin/checkpoint/pre-observation-20260723
remote commit = fb3c46cab10a68e34d4ed59f1a864b2b447c2503
force push = 未执行
rebase = 未执行
```

随后从该 commit 创建：

```text
feature/observation-notebook-20260723
```

分支创建前已记录 15 个 tracked dirty path 和 26 个未跟踪条目。新分支与 checkpoint 的 tree diff 为 0，因此切换没有覆盖现有修改。`S4_sbi/legacy/` 重排及所有原 dirty 文件保持原状。

## 新增 Observation 实现

本阶段新增：

```text
S4_sbi/notebooks/01_observation.ipynb
S4_sbi/src/sleep_sbi/__init__.py
S4_sbi/src/sleep_sbi/schemas.py
S4_sbi/src/sleep_sbi/observation.py
S4_sbi/configs/observation_sc4001.yaml
tests/conftest.py
tests/test_observation_loading.py
tests/test_epoch_boundaries.py
tests/test_observation_schema.py
```

实现了 observation-side `ObservationBundle`，保留二维 epoch 结构；publication-safe 导出明确排除 `segments`、epoch PSD 和 detector 中间数组。FOOOF 被硬锁为 1.1.1，版本不一致时直接失败，不允许 fallback。

## SC4001 标签与坐标核验

真实 annotation EDF 中出现的原始标签事件数：

```text
Sleep stage W = 12
Sleep stage 1 = 24
Sleep stage 2 = 40
Sleep stage 3 = 48
Sleep stage 4 = 23
Sleep stage R = 6
Sleep stage ? = 1
```

按 annotation duration 展开并与 PSG 原生 30 秒边界对齐后，Stage 3 为 101 epochs，Stage 4 为 119 epochs，共 220 个 N3 epochs。新映射明确规定 `Sleep stage 3`、`Sleep stage 4` 和短标签 `3` 均映射为 N3，绕开旧代码的 `"3" -> N2` 错误。

PSG 为 `EEG Fpz-Cz`、100 Hz、微伏分析单位、记录时长 79500 秒（22.083 小时），采用 EDF 中已记录的 Fpz-Cz 双极导联，不额外重参考。Hypnogram 比完整 PSG epochs 多 6900 秒，已明确告警并在 PSG 末端裁切，没有移动 epoch 坐标。

## Epoch QC 与摘要结果

```text
all complete PSG epochs = 2650
N3 epochs = 220
retained N3 epochs = 142
rejected N3 epochs = 78
rejection reason = peak_to_peak_above_threshold (78)
segment shape = (142, 3000)
```

所有 filter、SO detector、spindle detector 和 PAC 均逐 epoch 运行；三个 validity mask 均为 142/142 有效，没有跨 epoch filter 或事件。主要 observation 摘要：

```text
SO peak frequency = 0.500 Hz
relative SO power = 0.777428
SO Q = 4.869293
SO event rate = 13.112676 events/min
IBI_CV = 1.019784
observable spindle density = 2.859155 events/min
mean spindle duration = 0.676798 s
PAC MI = 0.000203706
preferred phase = 1.22173 rad
preferred phase sin/cos = 0.939693 / 0.342020
pac_up_down_ratio = 1.034337
FOOOF aperiodic exponent = 2.558807
```

新 schema 不产生 `T11_lag_ms`；该字段只在读取旧结果时被视为 legacy misnomer，新字段统一为 `pac_up_down_ratio`。

## Hann 与 Hamming sensitivity

Hann 保持为主结果，以匹配当前 fitting target 的实际实现；Hamming 仅作为 sensitivity：

```text
normalized PSD correlation = 0.999594
SO peak frequency: Hann 0.500 Hz, Hamming 0.500 Hz
relative SO power: Hann 0.777428, Hamming 0.780901
SO Q: Hann 4.869293, Hamming 5.041093
```

Notebook 已明确说明旧文档写 Hamming、实际 target 代码使用 Hann，没有静默改变 target。

## Notebook 与测试验证

Notebook 在 `neurolib` 环境中从头执行成功：

```text
code cells = 11
unexecuted code cells = 0
error outputs = 0
image outputs = 7
```

生成并目视检查了 hypnogram、epoch QC、代表性 N3 EEG、epoch/aggregate PSD、SO waveform 与 IBI、spindle event/envelope、PAC phase-amplitude 七类图。验证副本和 PNG 位于 gitignored 的 `outputs/observation_notebook_validation/`，没有写回源码 Notebook，也没有保存 EDF 或完整 EEG 数组。

定向测试结果：

```text
9 passed
```

测试覆盖真实 SC4001 加载、Stage 3/4/短标签映射、30 秒边界、QC 数量守恒、事件不跨 epoch、detector validity、summary 单位与字段、Hann/Hamming 定义以及 publication-safe 序列化。

## 当前科学限制与停止点

11-15 Hz RMS detector 采用逐 epoch threshold，和旧 held-out 脚本先拼接后求全局 threshold 的数值不完全同义。PAC MI 很弱，preferred phase 与 `pac_up_down_ratio` 对 detector band、Hilbert 边缘裁切及 phase convention 敏感；这些指标继续保留为 held-out/mechanism diagnostics，不应直接升级为稳定 inference summaries。

本阶段没有运行 V7 或 V8a simulation，没有 prior predictive，没有训练 NPE，也没有把任何 DE fitting candidate 称为 posterior sample。下一阶段应先定义版本化 `SummaryContractV1`，再让 Observation 与统一 SimulatorAdapter 对同一组名称、单位、band、aggregation、validity 和 source semantics 进行契约校验。

# 第十九部分：2026-07-23 Observation 分支远程备份

## 上传结果

用户明确确认上传 Observation Notebook 和本进度文档后，已将第一阶段实现提交到独立开发分支：

```text
branch = feature/observation-notebook-20260723
implementation commit = 918baba2b7352878eb1a6f4b72a7303745f1af4c
commit message = feat: add SC4001 observation notebook
remote branch = origin/feature/observation-notebook-20260723
```

`git ls-remote` 已只读验证远程 branch 指向上述 implementation commit。该 commit 共包含 10 个明确审查的文件：Observation Notebook、配置、`sleep_sbi` observation/schema 源代码、3 个目标测试、测试 fixture，以及截至第十八部分的本进度文档。

## 上传边界

本次没有纳入原有 dirty worktree、`S4_sbi/legacy/` 重排、manuscript、机器路径 helper、原始 EDF、`data/manifest.csv`、NPZ、PNG、执行后 Notebook 或其他 ignored outputs。Notebook 源文件没有 cell output 或嵌入 EEG 数组。

由于本进度文档的历史 section 含 6 行机器路径引用，此前 checkpoint 阶段曾将其排除；本次是在用户明确要求上传最新版文档后保持历史 section 原样纳入，没有静默改写过去记录。重新扫描未发现 credentials、密钥或 EDF 内容。

本 section 将通过后续纯文档 commit 上传，使远程进度文档能够完整记录 Observation implementation commit 与远程验证结果。

# 第二十部分：2026-07-24 Observation 独立面板实现与验证

## 本轮边界与组织方式

本轮在 `feature/observation-notebook-20260723` 分支继续实现真实 SC4001 EEG Observation。Notebook 只学习 SBI Practical Guide Figure 10 assembly notebook 的组织方式：每个 panel 独立说明、准备数据、调用绘图函数、立即显示，并分别导出 SVG 与 PNG。没有复制 pyloric network 的科学内容、panel 字母含义或视觉风格，也没有把 Observation panels 称为 Figure 10(a)-(g)。

正式内容统一命名为 `Obs-a` 至 `Obs-h`。Notebook 开头只用路线表把 simulator、5 个独立 NPE、global diagnostics、L-C2ST、posterior predictive 和 posterior marginals 标记为 `Future notebook`，没有生成空白、占位或虚假 scientific panel。本轮没有运行 V7/V8a simulator、prior predictive、SBC、expected coverage、L-C2ST、NPE 或 posterior predictive，也没有把 fitting candidates 称为 posterior samples。

## 新增与完善的实现

本轮最小增量修改为：

```text
S4_sbi/notebooks/01_observation.ipynb
S4_sbi/configs/observation_sc4001.yaml
S4_sbi/src/sleep_sbi/__init__.py
S4_sbi/src/sleep_sbi/schemas.py
S4_sbi/src/sleep_sbi/observation.py
S4_sbi/src/sleep_sbi/observation_plots.py
tests/test_epoch_boundaries.py
tests/test_observation_schema.py
tests/test_observation_plots.py
docs/implementation_plan_v8_progress_1.md
```

没有移动或重构 `S4_sbi/legacy/`，没有修改原始 EDF、`data/manifest.csv`、旧 held-out validation、manuscript 或其他既有 dirty files。

`observation_plots.py` 提供八个独立绘图函数，以及统一的 `save_panel()` 和 `write_observation_artifacts()`。每个绘图函数返回 `fig` 与 publication-safe `panel_data`；完整 EEG samples 只在内存中的 `ObservationBundle.segments` 保持二维 epoch 结构，不写入 CSV/JSON 或 Git artifact。

## Obs-a 至 Obs-h

```text
Obs-a  full-night hypnogram、N3位置、retained/rejected位置、真实annotation标签计数
Obs-b  按固定peak-to-peak分位数选择的4个30秒retained N3 EEG epochs
Obs-c  单epoch PSD、跨epoch区间、aggregate Hann PSD、SO band/peak、Hamming sensitivity
Obs-d  逐epoch SO检测示例、trough-centered individual/aggregate waveform
Obs-e  严格同epoch IBI分布、SO rate、mean/median IBI、IBI_CV及invalid reasons
Obs-f  真实EEG observable-channel 11-15 Hz、RMS threshold、事件、density和duration
Obs-g  SO phase、10-14 Hz amplitude、phase-amplitude分布、PAC MI和preferred phase
Obs-h  epoch/QC守恒、rejection reasons、跨epoch分布和最终summary table
```

所有 panels 使用同一字体、字号层级、线宽、Observation/retained/rejected颜色、频带阴影、DPI 和单位格式。Obs-f 的显示事件按“距 epoch 边界至少 2 秒的事件中取 peak envelope 中位数”确定，避免把边界附近极值作为代表图，但没有改变 detector 或汇总结果。Obs-g 显式阴影标出两端各 2 秒的 PAC edge trim。Obs-h 对不同量纲的跨 epoch 分布使用 `(value - median) / IQR` 做仅用于绘图的稳健标准化，原始数值保留在 summary table。

## 标签、epoch 与 QC 核验

真实 annotation 事件计数保持为：

```text
Sleep stage W = 12
Sleep stage 1 = 24
Sleep stage 2 = 40
Sleep stage 3 = 48
Sleep stage 4 = 23
Sleep stage R = 6
Sleep stage ? = 1
```

Stage 3 展开为 101 个 30 秒 epochs，Stage 4 展开为 119 个 30 秒 epochs，二者均映射为 N3，总计 220 个 N3 epochs。`Sleep stage 3`、`Sleep stage 4` 和短标签 `"3"` 均有回归测试确认映射为 N3；短标签 `"3"` 不再落入 N2。

```text
complete PSG epochs = 2650
N3 epochs = 220
retained N3 epochs = 142
rejected N3 epochs = 78
peak_to_peak_above_threshold = 78
segment shape = (142, 3000)
sampling rate = 100 Hz
epoch duration = 30 s
channel = EEG Fpz-Cz
unit = uV
recording duration = 22.083 h
```

所有 filtering、SO/spindle event detection、PAC 和 IBI 均逐 30 秒 epoch 运行。SO waveform 中 40 个无法在本 epoch 内取得完整 ±1.25 秒窗口的边界事件以 `incomplete_boundary_window` 排除；891 个完整事件来自 141 个有效 waveform epochs。IBI 只在同一 epoch 内计算，124 个 epochs 满足至少 3 个 SO events 的要求，共提供 779 个有效 IBIs；其余 18 个 epochs 以 `fewer_than_three_so_events` 标记 invalid。

## Observation summaries

严格 validity 聚合后的主要结果为：

```text
FOOOF aperiodic exponent = 2.558807
SO peak frequency = 0.500000 Hz
relative SO power = 0.777428
SO Q = 4.869293
SO event rate = 13.112676 events/min
mean within-epoch IBI = 2.931861 s
median within-epoch IBI = 1.830000 s
IBI_CV = 1.011715
observable spindle density = 2.859155 events/min
mean spindle duration = 0.676798 s
PAC MI = 0.000203706
PAC preferred phase = 1.221730 rad
PAC preferred phase sin/cos = 0.939693 / 0.342020
pac_up_down_ratio = 1.034337
SO waveform peak-to-peak = 1.993695 z
```

`IBI_CV` 与第十八部分的 1.019784 不同，是因为本轮修正为只聚合满足每 epoch 至少两个 IBIs 的有效 epochs；没有把单个 interval 的 epoch 静默混入 aggregate。该修改不涉及 simulator 或 fitting target。

Summary schema 现在明确包含：

```text
field_name, value, unit, frequency_band, algorithm,
aggregation_method, valid_epoch_count, validity_status,
intended_role, warnings
```

`intended_role` 只允许 `inference_summary_candidate`、`mechanism_diagnostic` 或 `held_out_ppc_candidate`。PAC 新字段统一为 `pac_mi`、`pac_preferred_phase_rad`、`pac_preferred_phase_sin`、`pac_preferred_phase_cos` 和 `pac_up_down_ratio`。`T11_lag_ms` 只在说明中标记为 `legacy misnomer; do not use as lag`，不作为 summary 字段或真实延迟解释。

## Hann/Hamming sensitivity

Hann 继续作为主结果，Hamming 只作为 legacy 文档差异的 sensitivity：

```text
Welch segment = 4.0 s
overlap = 1.0 s
frequency resolution = 0.25 Hz
aggregation = arithmetic mean across retained epoch PSDs
Hann/Hamming normalized PSD correlation = 0.999570
SO peak = 0.500/0.500 Hz
relative SO power = 0.777428/0.780901
SO Q = 4.869293/5.041093
```

Hamming 没有改变 SO peak frequency 或主要频谱结论。Notebook 明确记录旧文档写 Hamming、当前 target 实现使用 Hann，没有静默替换主定义。

## Spindle 与 PAC 限制

Obs-f 检测到 203 个 observable-channel events，142/142 epochs 的 detector 计算有效。该 detector 仍标记为 `provisional_held_out_ppc_candidate`：它不是 V8a 内部 T13，也不能代替 thalamic T8/T12；逐 epoch threshold、边缘行为和跨信号校准尚未冻结，不能直接升级为 inference summary。

PAC 有 142/142 有效 epochs、369200 个 edge-trim 后支持 samples，但 aggregate MI 仅 0.000203706，效应很弱。其 preferred phase、sin/cos 和 `pac_up_down_ratio` 仍对 band、phase convention、Hilbert edge trim 与 aggregation 选择敏感，因此 PAC 保留为 held-out PPC candidate，`pac_up_down_ratio` 保留为 mechanism diagnostic。

## 自动验证与 artifacts

所有 Python 命令均在 `neurolib` conda 环境中运行。Observation 定向测试结果：

```text
19 passed, 1 fooof deprecation warning
```

Notebook 使用非交互 nbconvert 从头执行成功：

```text
executed code cells = 12
error outputs = 0
image outputs = 8
```

八个 panels 均独立生成 SVG 和 PNG；测试逐个解析 SVG、读取 PNG 并检查有限像素与非空图像。人工目视检查确认没有空白 panel、NaN 扩散、错误坐标轴、单位缺失或明显文字重叠。publication-safe 输出位于 gitignored 的：

```text
outputs/observation_sc4001/panels/
outputs/observation_sc4001/observation_summary.csv
outputs/observation_sc4001/qc_summary.json
outputs/observation_sc4001/panel_manifest.json
```

`panel_manifest.json` 包含 panel ID/title、subject/channel 安全标识、config SHA-256、算法参数、输入/有效 epoch 数、相对 artifact 路径、UTC 时间和 warnings。安全扫描未发现机器绝对路径、EDF 副本或完整 EEG arrays。没有对 ignored outputs 使用 `git add -f`。

全仓库 `pytest -q` 在 collection 阶段被旧 `tests/test_spindles.py` 阻塞，因为该测试直接要求不存在的 `outputs/r_cortex.npy`。本轮禁止运行 simulator，因此没有为该旧测试生成文件；这不是 Observation 测试失败。

## 下一阶段建议与停止点

下一阶段 `SummaryContractV1` 建议首先纳入 scale-invariant normalized PSD、严格 `fooof==1.1.1` 的 aperiodic exponent、`so_peak_frequency_hz`、`relative_so_power`/`so_q`、`so_event_rate_per_min` 和带 validity 规则的 `ibi_cv`。observable spindle density/duration、PAC MI、preferred-phase sin/cos 和 SO waveform morphology 继续作为 held-out PPC candidates；T8/T12、V8a internal T13 和 `pac_up_down_ratio` 保持 mechanism diagnostics，并明确 simulator/observation source semantics。

本轮没有 push Observation 分支，没有创建 commit，也没有开始 SummaryContract、simulator、prior predictive 或 NPE 阶段；等待用户审阅。

# 第二十一部分：2026-07-24 Observation 独立面板上传记录

## 实现提交

经用户明确要求上传最新进展后，本轮已将第二十部分记录的 Observation 实现整理为独立提交：

```text
branch = feature/observation-notebook-20260723
implementation commit = 054d02b
commit message = feat: complete SC4001 observation panels
```

该提交只包含 10 个经过安全检查的文件：

```text
S4_sbi/configs/observation_sc4001.yaml
S4_sbi/notebooks/01_observation.ipynb
S4_sbi/src/sleep_sbi/__init__.py
S4_sbi/src/sleep_sbi/observation.py
S4_sbi/src/sleep_sbi/observation_plots.py
S4_sbi/src/sleep_sbi/schemas.py
tests/test_epoch_boundaries.py
tests/test_observation_plots.py
tests/test_observation_schema.py
docs/implementation_plan_v8_progress_1.md
```

## Notebook 上传形态

上传版本的 `S4_sbi/notebooks/01_observation.ipynb` 是最新的完整执行版本，包含 `Obs-a` 至 `Obs-h` 八个 publication-safe panel 输出，使 GitHub notebook 预览可以直接检查图和表。安全检查结果：

```text
notebook cells = 25
embedded panel images = 8
error outputs = 0
machine absolute paths = 0
credentials or key patterns = 0
raw EEG arrays serialized = 0
notebook size = 1.61 MB
```

唯一包含本地路径的 FOOOF deprecation warning 已改写为仓库相对路径 `S4_sbi/src/sleep_sbi/observation.py:117`。Notebook 中保留的是 publication-safe 图片、aggregate tables 和 QC metadata，不包含 EDF、完整 EEG samples 或原始数据副本。

## 上传边界

本次仍明确排除原有 `S4_sbi/legacy/` 重排、S4_sbi 根目录 tracked deletions、`compute_xobs_from_eeg_v4.py` 注释式修改、manuscript、held-out validation 注释式修改、NPZ、PNG/PDF 原始输出、simulation records、logs、cache、原始 EDF 和 `data/manifest.csv`。没有使用 `git add .`、`git add -A` 或 `git add -f`，也没有回滚、移动或删除这些既有成果。

Observation 定向测试仍为 `19 passed`；完整仓库测试仍受旧 `tests/test_spindles.py` 依赖缺失的 `outputs/r_cortex.npy` 阻塞。本轮未运行 simulator 来生成该文件。

# 第二十二部分：2026-07-25 Pyloric-inspired Observation 18D 实现与验证

## 本轮范围与文件

本轮保留现有 `S4_sbi/notebooks/01_observation.ipynb` 完全不变，并新增独立的真实 EEG observation engineering 实现：

```text
S4_sbi/notebooks/01_Pyloric_Inspired_Observation.ipynb
S4_sbi/src/sleep_sbi/pyloric_inspired_observation.py
S4_sbi/configs/pyloric_inspired_observation_sc4001.yaml
tests/test_pyloric_inspired_observation.py
```

原 notebook 在本轮开始和结束时的 SHA-256 均为：

```text
6ee079f548cb871996f10dc394f742ad4d2b8859c7f917f53ca4c14c34fbe1e5
```

新增 notebook 只运行 SC4001 真实 EEG，不运行 V7/V8a simulator、prior predictive、NPE/NLE、SBC、L-C2ST 或 posterior predictive，也不产生或声称任何 posterior。

## neurolib 运行证据

所有 Python 检查、测试和 notebook 执行均使用既有 `neurolib` 环境：

```text
Python executable = <neurolib-env>\python.exe（公开上传前已隐藏机器绝对路径）
Conda environment = neurolib
Jupyter kernelspec = python3
kernelspec location = ...\envs\neurolib\share\jupyter\kernels\python3
notebook display name = Python (neurolib)
Python = 3.10.20
nbformat = 5.10.4
nbclient = 0.10.4
jupyter_core = 5.9.1
ipykernel = 7.2.0
numpy = 2.2.6
scipy = 1.15.2
pandas = 2.3.3
matplotlib = 3.10.8
mne = 1.9.0
fooof = 1.1.1
neurolib = 0.6.1
```

Notebook metadata 使用 `display_name = Python (neurolib)`、`name = python3`；运行时 cell 同时断言 `CONDA_DEFAULT_ENV == "neurolib"` 和 `fooof == "1.1.1"`，禁止 silent fallback。

## 实际 SC4001 epoch accounting

完整数据重算得到：

```text
aligned 30-second epochs = 2650
all N3 epochs = 220
retained N3 epochs = 142
rejected N3 epochs = 78
rejection reason = peak_to_peak_above_threshold: 78
channel = EEG Fpz-Cz
sampling rate = 100 Hz
```

Stage 3、Stage 4 和短标签 `"3"` 均显式回归验证为 N3。`retained + rejected = all N3`、两集合互斥、每个 rejected epoch 有原因、non-N3 不进入 QC rejection 均通过。所有 filtering、event detection、IBI、完整 SO cycle 和 SO-spindle pairing 都限制在原生 30 秒 epoch 内。

## Pyloric-inspired 18D 结果

18 个数值字段按稳定顺序定义，其中 9 个为 `inference_summary_candidate`，9 个为 `held_out_ppc_candidate`：

```text
fooof_aperiodic_exponent                 2.558807   inference
relative_so_power                       0.777428   inference
so_peak_frequency_hz                    0.500000   inference
so_q                                    4.869293   inference
so_event_rate_per_min                  13.112676   inference
so_median_ibi_s                         2.452500   inference
ibi_cv                                  1.011715   inference
so_up_proxy_duration_s                  0.606530   inference candidate
so_down_proxy_duration_s                0.611617   inference candidate
so_up_proxy_duty_cycle                  0.487025   held out
waveform_peak_to_peak_z                 1.993695   held out
so_trough_to_peak_time_s                0.672500   held out
spindle_density_per_min                 2.859155   held out, provisional
spindle_mean_duration_s                 0.676798   held out, provisional
spindle_occupancy                       0.032251   held out, provisional
spindle_onset_phase_cos                -0.361593   held out, provisional
spindle_onset_phase_sin                 0.932336   held out, provisional
spindle_onset_phase_concentration       0.198199   held out, provisional
```

IBI 的新增 median 使用“每个 epoch 内 median，再跨有效 epoch 取 median”的 hierarchical robust aggregation。`ibi_cv` 继续复用旧版“只汇集通过最少事件支持的 within-epoch intervals”的定义，从而保持回归可比性。

## 新增时间结构的科学边界

SO temporal proxy 使用 as-recorded Fpz-Cz polarity：负相位称为 DOWN-proxy，正相位称为 UP-proxy。它只表示 scalp EEG observable-level temporal proxy，不等于神经元膜电位或 simulator 内部 cortical UP/DOWN state。

视觉审计发现第一版 cycle 终点曾从 detector 的后续 peak 之后搜索，可能把多个中间振荡合并为一个正相位。实现已修正为：

1. 从 trough 前的 positive-to-negative crossing 开始；
2. 使用 trough 后的第一个 negative-to-positive crossing；
3. 使用紧邻的 positive-to-negative crossing 结束；
4. detector peak 必须落在该完整正相位内；
5. 边界截断、polarity 不符或 peak 不在紧邻正相位内均返回明确 invalid reason。

修正后支持为 `322` 个完整 SO proxy cycles、`116` 个有效 epochs；其中 `580` 个 upstream events 因 `detected_up_peak_outside_immediate_positive_phase` 被保守排除。该修正不改变原 notebook 的 detector 或任何重叠 aggregate。

Spindle occupancy 使用每个 epoch 内 accepted intervals 的 union duration 除以 detector-valid observation duration，防止重叠区间重复计数。有效但 0-event epoch 的 density 和 occupancy 为 0，mean duration 保持 NaN 并记录 `no_spindle_events_mean_duration_undefined`。

Event-conditioned coordination 只配对同一 epoch 内、位于两个支持范围内 SO trough 之间的 spindle onset。当前 trough 为 0 rad、下一 trough 为 2*pi；共有 `52` 个 paired events、`42` 个有配对支持的 epochs。核心 18D 只保留 circular mean 的 cos/sin 和 resultant length `R`，raw angle 只进入 auxiliary diagnostic，避免重复计维。`R = 0.198199` 表明方向集中度偏弱，暂不应作为稳定生理靶点。

## 冗余、角色与旧版回归

稳定性审计显示：

```text
spindle density vs occupancy: Spearman rho = 0.92, n = 142
event-conditioned phase per-epoch missing fraction = 0.929577
SO proxy per-epoch missing fraction = 0.183099
IBI median/CV per-epoch missing fraction = 0.126761
near-constant core features = 0
```

因此 18D 是 feature library，不是 18 个独立证据。UP-proxy duty cycle 是两个 proxy durations 的确定性派生量；SO rate 与 IBI 具有反比结构；spindle occupancy 与 density、duration 具有派生关系。Spindle 三项继续 held out 并要求 detector validation；event-conditioned phase 三项因 per-epoch 支持稀疏而继续 held out。

旧版重叠指标回归包含 aperiodic exponent、relative SO power、SO peak、SO Q、SO event rate、IBI_CV、waveform peak-to-peak、spindle density/duration，以及 continuous PAC MI、preferred phase 和 `pac_up_down_ratio`。结果为：

```text
12 pass
0 warning
0 fail
maximum absolute difference = 0
```

Continuous PAC 和 `pac_up_down_ratio` 只在 auxiliary diagnostics 中保留。`T11_lag_ms` 继续标记为 `legacy misnomer; do not use as lag`。V7/V8 的 T9-T13 均标记为 model-side mechanism diagnostic，不为单通道真实 EEG 伪造数值。

## Notebook、测试、图形与输出

新 notebook 共 `31` 个 cells、`15` 个 code cells。使用 `neurolib` 的 `nbconvert --execute --inplace` 从头执行成功：

```text
execution counts = 1..15
error outputs = 0
display/execute outputs = 36
```

结构和执行验证包括 nbformat JSON、所有 code cells 静态编译、18D 恰好 18 个唯一字段、raw angle 不重复计维、bounded metrics 位于 `[0, 1]`、duration/frequency/rate 单位、无 inf、epoch ledger 恒等式、metric-specific support、旧版回归和 artifact existence。

相关测试结果：

```text
26 passed
0 failed
1 expected fooof deprecation warning
```

11 组独立方法检查图均在 notebook 中立即显示并保存 PNG/PDF：concept mapping、epoch QC、18D distributions、robust variability、Spearman/valid-pair heatmap、SO proxy morphology、spindle detection、event pairing、circular onset phase、support/missingness 和 role map。人工检查确认图非空、无 NaN 扩散、单位和坐标轴可读，SO proxy 修正后的相位阴影不再跨越中间零点。

所有运行输出位于 gitignored 的：

```text
S4_sbi/outputs/pyloric_inspired_observation/
```

主要导出包括 `pyloric_inspired_18d_summary.csv/json`、`pyloric_inspired_per_epoch_features.csv`、`epoch_ledger.csv`、`feature_dictionary.csv`、`old_vs_new_regression.csv`、`feature_stability_audit.csv`、`validation_report.md` 及 `figures/*.png/pdf`。这些 artifact 不包含 EDF、完整原始或 filtered EEG arrays、simulation records 或 posterior；本轮未使用 `git add -f`。

## 停止点

本轮没有 commit、push、创建 PR、训练 NPE、运行 simulator 或修改 legacy 重排。当前没有阻止 notebook 在本机完整 Run All 的 blocker。进入 fitting 前应先审查 notebook 的 `SO temporal morphology`、`Observable spindle activity and occupancy`、`Event-conditioned SO-spindle coordination`、`Redundancy and stability audit` 及 `Regression against the existing Observation notebook` 五节。

## 第二十三部分：2026-07-27 Simulator Observable Adapter 审计与验证

本轮目标是解除 `04_Real_Simulation_Extractor_Parity.ipynb` 发现的实现层 blocker，而不是训练 SNPE。未改写 `03`--`06` notebooks，未修改 V7/V8/V8a simulator 核心、真实 EEG extractor、科学阈值、filters 或现有 fitting 结果；未 commit、push 或创建 mini simulation bank。

新增文件：

```text
S4_sbi/src/sleep_sbi/simulator_observable_adapter.py
S4_sbi/tests/test_simulator_observable_adapter.py
S4_sbi/notebooks/07_Simulator_Observable_Adapter_Validation.ipynb
S4_sbi/results/simulator_observable_adapter_validation/
```

### 审计结论

既有 real-vs-model / fitting 路径实际使用的是 `r_mean_EXC` 的 cortex index 0：这是 ALN cortical excitatory population firing rate state observable。模型记录值按既有实现由 kHz 乘以 `1000` 转为 Hz；它不是 simulated EEG，也不存在项目已验证的 EEG forward proxy。V7/V8/V8a 使用 `sampling_dt=1 ms`（`1000 Hz`）、总时长 `60 s`，并丢弃前 `5 s` warm-up，留下 `55 s` active signal。为不改变已有模拟时长，adapter 只取第一个完整 post-warm-up `30 s` epoch，并显式丢弃余下 `25 s`，不拼接边界。

真实 observation 是 `EEG Fpz-Cz` scalp EEG（uV）。PSD peak frequency、relative SO power、SO Q、FOOOF exponent 以及归一化 PAC 统计对正比例幅度缩放较不敏感，可在 cortical-rate proxy 上调用同一底层 Python 函数作 algorithm-level audit；但这不建立 channel/reference/forward-model 的语义等价。SO half-wave detector 使用真实 EEG 的 `75 uV` threshold，因此 adapter 没有将该数值错误应用于 Hz rate，而是将 SO event rate、IBI CV 与 waveform morphology 标为明确的 unit-mismatch blocker。Spindle detector 和 PAC 函数可运行，但输出仍标记为 cortical-rate proxy diagnostic，不能作为 scalp EEG inference feature。

### 代表性运行与结果

`07` 已在 `neurolib` / `python3` kernel 完整 Run All。验证了：V7 fitted、V8 fitted、V8a local best、以及 3 个来自 `outputs/v8a_coupling_sweep_long/selected_seed_candidates.csv` 的 V8a near-feasible candidates。6/6 simulation 成功；实际单次 runtime 为约 `4.67--9.09 s`，均为 `1000 Hz`、`55 s` active signal、`25 s` excluded tail。未保存完整 firing-rate arrays。

```text
minimal_spectral_so_proxy4: 6/6 finite, algorithm-level available, semantic parity NO-GO
minimal_plus_pac9:         6/6 finite, algorithm-level available, semantic parity NO-GO
minimal_plus_spindle6:     0/6 finite, spindle mean duration undefined because 0 detected events
minimal_plus_so_morphology7: 0/6 finite, blocked by uV-dependent SO detector
baseline14:                0/6 finite, contains the blocked/undefined features
```

所有 schema 的 `semantic_parity_pass` 都是 `False`。没有 NaN/Inf 被静默填为 0：有效但零 spindle-event 的 density 为 0，而 spindle mean duration 保持 undefined/NaN。`adapter_feature_matrix.csv`、`adapter_schema_status.csv` 和 `feature_level_blocker_matrix.csv` 分别保存每个 candidate、feature validity/support/failure reason 与 schema-level hard-gate 证据。

### 结论和最小下一步

当前不存在可用于 pilot SNPE 的 shared observation schema；因此没有重新运行或修改 `04`、没有创建 mini bank、没有训练 SNPE、没有 posterior、coverage、SBC 或 PPC 结果。`parity_reassessment.json` 明确记录 `NO-GO`。

解除 blocker 的最小研究步骤不是将 rate 数值线性改成 uV，而是定义并验证 model cortical observable 到声明的 `Fpz-Cz` scalp EEG observable 的 forward/reference mapping；随后必须冻结单位、30 s epoch policy、aggregation、validity/missingness 和 scaling，并重新运行 parity。只有该 semantic contract 通过后，才可建立小型 checkpointed simulation bank 并考虑 pilot SNPE。

验证完成：`07` 的 `8` 个 code cells 均有 execution count，`nbformat.validate()` 与静态 compile 通过，无 error/traceback；adapter 的 3 个快速 contract tests 通过；CSV/JSON/SVG/PNG/HTML artifacts 均已重新读取或人工检查。HTML 位于 `S4_sbi/results/simulator_observable_adapter_validation/html/07_Simulator_Observable_Adapter_Validation.html`。


# 第二十四部分：2026-07-27 EEG Observation Mapping Audit 与正式路线决策

## 本轮范围与文件

本轮只审计 neurolib thalamocortical simulator 输出到真实 `EEG Fpz-Cz` scalp EEG 的 measurement mapping，不训练 SNPE、不生成 simulation bank、不修改 simulator 动力学、真实 EEG extractor、科学阈值、filter、QC 或既有 `03`--`07` notebooks。新增：

```text
S4_sbi/src/sleep_sbi/eeg_observation_mapping_audit.py
S4_sbi/tests/test_eeg_observation_mapping_audit.py
S4_sbi/tests/validate_eeg_observation_mapping_audit.py
S4_sbi/notebooks/08_EEG_Observation_Mapping_Audit.ipynb
S4_sbi/results/eeg_observation_mapping_audit/
S4_sbi/results/overnight_observation_ablation/html/08_EEG_Observation_Mapping_Audit.html
```

## 既有 fitting / Figure 5/7 信号的代码证据

Figure 5/7 的真实侧目标是 `EEG Fpz-Cz`：MNE 内部 V 乘以 `1e6` 转为 `uV`，保留原生 30 秒 N3 epoch，按 `200 uV` peak-to-peak 做 QC，分别计算 Hann-Welch PSD 后取 arithmetic mean。Figure-5-era `s3_sleep_kernel.py` / `s3_band_power_bars.py` 同样把 `r_mean_EXC` 从 kHz 转为 Hz 后计算 cortical/thalamic firing-rate Welch PSD；其中 `EEGlike` 只出现在输出文件名和频带对齐说明中，不是 measurement function。Figure 7 模拟侧使用 `r_mean_EXC` 的 cortex index 0，60 秒模拟丢弃前 5 秒 warm-up，再计算 firing-rate PSD 与真实 EEG PSD 的 log-space loss。

因此既有结果支持的是跨 modality 的 spectral-shape fitting，不是 source-to-sensor measurement parity。`r_mean_EXC` 和 `r_mean_INH` 分别是 cortical excitatory / inhibitory population firing rate；不得称作 membrane potential 或 simulated scalp EEG。项目搜索没有找到 lead field、bipolar reference operator、cortical LFP、source geometry、volume conduction、sensor noise 或经验证的 Fpz-Cz forward model。

neurolib 的 ALN cortex 还包含 `I_mu`、`I_A`、excitatory/inhibitory synaptic mean/variance states；平均 membrane voltage 只作为 transfer-function lookup 的中间量，不是项目记录的 cortical state。Thalamic TCR/TRN 包含 `V_EXC` / `V_INH` membrane voltages，但它们是丘脑内部状态，不能替代单通道 scalp EEG。

## 三组参数的小规模实证 screening

`08` 只运行 V7 fitted、V8 fitted、V8a local best 三组现有参数。每组为 60 秒、1000 Hz，去除 5 秒 warm-up 后保留 55 秒；EEG extractor screening 只使用第一个完整 30 秒窗口，不拼接剩余 25 秒。实际 runtime 分别约为：

```text
V7 fitted       8.71 s
V8 fitted       4.70 s
V8a local best  4.79 s
```

审计额外记录变量但不改变微分方程，共检查 19 种 native/derived signals：cortical EXC/INH rates、等权 E-I 与 E+I rate contrasts、EXC/INH `I_mu`、adaptation `I_A`、`I_mu - I_A/C` effective drive、8 个 individual synaptic mean/variance states、synaptic-state balance，以及 TCR/TRN voltages。三组参数的 57 条 signal records 全部 finite；没有保存完整时间序列。

真实 SC4001 observation 仍为 `220 N3 = 142 retained + 78 rejected`，`EEG Fpz-Cz`、100 Hz、30 秒 epoch。标准化比较仅用于 screening：PSD 在 0.5--20 Hz 内归一为 unit area，autocorrelation 在每个 30 秒 epoch 内 z-score 后计算。核心结果包括：

```text
log-PSD Spearman versus real Fpz-Cz
V7 cortical r_EXC / I_mu_EXC / effective drive: 0.932 / 0.945 / 0.941
V8 cortical r_EXC / I_mu_EXC / effective drive: 0.955 / 0.956 / 0.955
V8a cortical r_EXC / I_mu_EXC / effective drive: 0.953 / 0.956 / 0.954
TCR voltage: 0.242--0.309
```

这些值只说明 unit-area spectral rank shape，不能证明 measurement model。线性缩放或 z-score 只能改变 offset/amplitude，不能补足 source mixture、lead field、volume conduction、Fpz-Cz bipolar reference 或 sensor noise。

## EEG extractor 迁移的 blockers

- PSD peak frequency、relative band power、SO Q 和某些 normalized spectral statistics 对正比例幅度缩放较不敏感，可作 shape-level screening；但仍会受 measurement operator 和 source mixture 影响。
- SO detector 的 `75 uV` half-wave threshold 不能应用于 Hz、pA、mV/ms、dimensionless synaptic states 或内部 mV。数值代码即使返回事件，也没有维度意义。
- 核心 cortical-rate/current candidates 的 observable spindle detector 在三组参数中均为 0 event；这只表示该 detector 在该 model-state signal 上未检出事件，不代表真实 scalp spindle 为零。`mean duration` 不能由零事件伪造为 0。
- 单通道 PAC 的 filter/Hilbert/MI 可以数值运行，但 phase、amplitude 与 preferred phase 属于 measurement-dependent quantities；cortical-rate PAC、thalamic-amplitude PAC 和 Fpz-Cz single-channel PAC 不是同一个 observable。
- 单通道 Fpz-Cz 不能直接观测 thalamic voltage、thalamic spindle onset、传播 latency 或 directional thalamocortical coupling。

## Mapping options 与路线决策

`mapping_option_decisions.csv` 比较 A--H：直接 rate、rate 标准化/线性缩放、EXC/INH 组合、internal voltage/current、LFP-like proxy、显式 EEG forward model、synthetic-only SBI、以及暂停 real-EEG inference。直接 rate、任意线性 `Hz -> uV`、未校准 E/I 组合均不具备 real-EEG inference 的科学可辩护性；internal current/voltage 可作为未来 source-model research candidates，但仍缺 scalp projection。

正式推荐为 **Route 2：先实现并独立验证 measurement/forward model，再考虑 real-EEG SNPE**。Route 1 被拒绝，因为没有找到可信 EEG/LFP proxy。Route 3 仅可作为明确限定的 synthetic cortical-observable parameter recovery；真实 EEG 在此期间只能作为带 modality caveat 的 Route-4 external shape validation。

当前 hard gate：

```text
shared EEG schema parity = NO-GO
simulation bank authorized = false
pilot SNPE authorized = false
```

进入下一次 parity 前必须冻结并验证：具有生物物理依据的 cortical source variable、source geometry/lead field、Fpz-Cz bipolar reference、physical gain/unit、独立于 inference target 的 calibration protocol、sensor/noise model，以及 SO/spindle/PAC 的 held-out validation。

## 执行、验证与人工复核

Notebook 在实际 `neurolib` 环境执行：

```text
Python executable = C:\Users\YUS190\AppData\Local\anaconda3\envs\neurolib\python.exe
Python = 3.10.20
kernel ID = python3
display name = Python 3 (ipykernel)
code cells = 10
execution counts = 1..10
error outputs = 0
```

`nbformat.validate()`、所有 code-cell static compile、4 个 simulation-free unit tests、8 个 CSV 与 JSON artifact reload、4 张 PNG 像素/尺寸检查、HTML export 和 protected notebook SHA-256 复核均通过。HTML 顶部、native-state traces、standardized PSD、z-scored autocorrelation 与 extractor-support 图已人工检查：公式、表格、单位、图例和坐标轴可读，无空白 panel、NaN 扩散或 traceback。`01`--`07` 的 notebook hashes 与本轮开始时一致。

## neurolib 通用 lead-field 工具的补充核对

本机 neurolib 源码另含 `neurolib/utils/leadfield.py`：它可调用 MNE/BEM 构建一般 EEG lead field，并将 sensor-by-dipole 矩阵下采样到 atlas regions。该工具没有在 `sleep_loop` 中被调用，也没有定义两节点 thalamocortical model 的哪个 state 是具有方向和位置的 cortical dipole source，更没有提供 Fpz-Cz bipolar reference、gain/unit 或 sensor-noise contract。因此它是 Route 2 可评估复用的几何基础设施，不是项目已有的 EEG proxy 或已完成的 forward model；本轮 Route 2 / NO-GO 结论不变。

## 明天需要研究者决定的问题

1. 项目的近期科学目标是 Route 2 的真实 EEG parameter inference，还是先做 Route 3 的 synthetic recovery？
2. 若选择 Route 2，应把 reconstructed ALN mean voltage、transmembrane/synaptic-current proxy 或其他 neural-mass observable 中的哪一种作为 cortical source，并由哪类文献/实验校准支持？
3. Fpz-Cz measurement contract 应采用何种 source geometry、lead field、bipolar reference、gain/unit 与 sensor-noise model？
4. 哪套独立数据用于 measurement-model calibration 与 validation，避免使用 SC4001 inference target 造成 leakage？
5. forward model 通过后，哪些 SO/spindle/PAC summaries 进入 inference，哪些继续保留为真正 held-out evidence？

# 第二十五部分：2026-07-27 Forward-Model Feasibility 与 Route 2/3 正式决策

## 本轮边界与新增交付

本轮只审计当前单个 ALN cortical node 加单个 thalamic node 是否足以定义真实 `EEG Fpz-Cz` measurement contract；没有训练 SNPE、没有生成 simulation bank、没有再次运行 V7/V8/V8a，也没有执行任意 `Hz -> uV` 缩放。`01`--`08` notebooks、simulator 核心、真实 EEG extractor、filter、threshold、QC 和既有 artifacts 均未修改。

新增：

```text
S4_sbi/src/sleep_sbi/forward_model_feasibility.py
S4_sbi/tests/test_forward_model_feasibility.py
S4_sbi/tests/build_forward_route_notebooks.py
S4_sbi/tests/validate_forward_route_notebooks.py
S4_sbi/notebooks/09_Forward_Model_Feasibility_and_Route_Decision.ipynb
S4_sbi/notebooks/00_Observation_SBI_Reader_Guide.ipynb
S4_sbi/results/forward_model_feasibility_route_decision/
S4_sbi/results/observation_sbi_reader_guide/
S4_sbi/results/overnight_observation_ablation/html/09_Forward_Model_Feasibility_and_Route_Decision.html
S4_sbi/results/overnight_observation_ablation/html/00_Observation_SBI_Reader_Guide.html
```

## ALN states、内部中间量与 source candidates

neurolib ALN 的 excitatory mass 真正积分保存 `I_mu`、`I_A`、两类 synaptic mean、两类 synaptic variance 和 `r_mean`；inhibitory mass 保存除 `I_A` 外的对应 states。当前项目网络历史上只要求记录 `r_mean_EXC` 与 `r_mean_INH`。`voltage_lookup(I_mu - I_A/C, I_sigma)` 由预计算的 `V_mean_ss` transfer-function table 提供 population mean voltage，并在 excitatory derivative 内用于 adaptation，但 voltage 本身不是保存的 state。

因此 reconstructed population mean voltage 可以在未来通过完整重建 `I_sigma` 与 coupling inputs 后得到，且是当前代码中物理含义最清楚的 voltage-level research candidate；但 membrane voltage 仍不是 primary-current dipole moment，不能直接输入 EEG lead field。更接近 EEG forward physics 的 transmembrane/synaptic-current dipole proxy 当前并不存在：缺少可追踪的 dipole-moment 单位、population size/patch area、laminar geometry、方向和独立校准。

`source_candidate_decisions.csv` 共比较 12 类候选：

- `r_mean_EXC` / `r_mean_INH`：合法 model observable，只适合 Route 3；
- 等权 E-I / E+I rate：权重没有科学校准，Route 2 排除；
- `I_mu`、`I_A`、effective drive、individual synaptic states：内部机制或未来 source-derivation candidates，不是现成 dipole；
- reconstructed population mean voltage：优先保留为 Route-2 source-model research candidate，但未获 forward approval；
- transmembrane/synaptic-current dipole：最相关的目标定义，但当前完全缺失；
- TCR/TRN voltage：thalamic internal mechanism state，不能直接进入 cortical surface lead field 或替代 scalp EEG。

## 单 cortical source 与 Fpz-Cz 的 rank-1 限制

若当前唯一 cortical node 被声明为一个 scalar source `q(t)`，线性 instantaneous lead field 给出：

```text
V_Fpz(t) = L_Fpz * q(t)
V_Cz(t)  = L_Cz  * q(t)
V_Fpz-Cz(t) = (L_Fpz - L_Cz) * q(t)
```

当 `L_Fpz != L_Cz` 时 bipolar signal 可以非零，因此不能简单称为数学上的零信号；但它的 temporal rank 仍为 1，只是相同 `q(t)` 的固定 gain/sign copy，不增加 distributed-source mixing 或 channel-specific dynamics。当前两节点模型没有 source coordinates、patch extent、cortical-normal orientation 或 dipole moment，因此连该固定 gain 也无法由模型本身确定。结论是：当前单 cortical node **不能仅凭现有 states 支持 spatially non-degenerate、measurement-valid 的 Fpz-Cz forward model**；Route 2 必须引入新的、外部约束的 source/spatial assumptions。

Thalamic node 不应直接进入当前 scalp forward path。它是抽象深部结构，缺少深部 source geometry、orientation 和经过验证的 volume-conductor mapping；把 TCR/TRN voltage 或 spindle onset 直接投影到 Fpz-Cz 会把内部机制状态伪装为可观测 scalp source。

## neurolib leadfield.py 能做什么、不能解决什么

本机 `neurolib/utils/leadfield.py` 能复用的部分包括：MNE `Raw/Info` 与 montage、head-to-MRI transform、cortical surface source space、三层 EEG BEM/conductivity、`mne.make_forward_solution`、固定 surface-normal orientation，以及将 sensor-by-dipole matrix 下采样到 AAL2 cortical regions。

它不能解决：

- ALN 哪个 state 是 primary-current/dipole source；
- 单 cortical node 对应哪个 cortical region、位置、面积与方向；
- state 到 A m / nA m 等 dipole-moment unit 的转换；
- Fpz 与 Cz 的独立 sensor potentials 及随后 `[1, -1]` bipolar reference；
- physical gain、reference/sensor noise；
- measurement-model calibration 和 held-out validation。

因此通用 leadfield utility 是 Route 2 的几何基础设施，不是当前项目已有的 EEG forward model。

## 严格 measurement contract gate

`measurement_contract_components.csv` 将完整链条冻结为：

```text
theta
-> neurolib dynamics
-> declared cortical source
-> deterministic source reconstruction
-> source geometry/orientation
-> head volume conductor
-> lead field
-> Fpz and Cz sensor potentials
-> Fpz-Cz bipolar reference
-> physical gain/unit
-> sensor/reference noise
-> 100 Hz resampling
-> native 30-second epochs
-> unchanged EEG feature extractor
-> independent validation
```

当前只有 theta、dynamics、100 Hz resampling 和 30-second epoch policy 为 PASS；EEG extractor 已存在但被 sensor-voltage contract 阻塞；source、geometry、BEM instance、lead field、sensor potentials、gain/unit、noise 和 independent validation 均为 NO-GO。空 declaration 不会获得默认值，`apply_measurement_model()` 会明确拒绝并抛出 `MeasurementContractError`；即使 declaration 字段完整，在没有实际 validated implementation 时仍会抛出 `NotImplementedError`。本轮没有实现任意常数 forward model。

source choice、source weights、coordinates/orientation、source-to-dipole gain 和 noise 参数若使用 SC4001 inference target 按匹配程度校准，会产生 measurement-target leakage。这些量必须由文献、解剖、独立 EEG/MEG/LFP 数据或预注册的外部 calibration protocol 冻结。lead field 可在 geometry 固定后确定性计算；Fpz-Cz reference 本身可固定为 `[1, -1]`，但前提是先有两个物理 sensor potentials。

## Route 2 与 Route 3 推荐

**Route 2** 在物理上并非永远不可实现，但不能从当前两节点 states 单独推出。它是一个新的 source/measurement research program，需要：source derivation、空间位置/方向/面积、BEM/lead field、dipole unit/gain、noise/reference model，以及独立于 SC4001 的 SO/sigma/spindle/PAC measurement validation。当前状态为：

```text
Route 2 real-EEG SBI = NO-GO_PENDING_EXTERNAL_SOURCE_AND_SPATIAL_ASSUMPTIONS
```

**Route 3** 将 SBI 严格限定于 declared cortical-observable space，可研究 synthetic theta recovery、calibration 和 identifiability；真实 EEG 只能作为带 modality caveat 的 external shape-level evidence。Route 3 不能声称真实 Fpz-Cz parameter inference、patient posterior 或内部机制正确。近期 conference 最稳妥的范围是：报告模型动力学、adapter 工程、shape-level external comparison 和明确的 measurement limitation；如需增加 inference 内容，优先考虑经过研究者批准并预注册的 Route-3 synthetic recovery。

本 notebook 只推荐近期待讨论 Route 3，并未授权：

```text
simulation bank authorized = false
pilot SNPE authorized = false
```

## Reader Guide

`00_Observation_SBI_Reader_Guide.ipynb` 建立 02--09 的 90 分钟阅读路线并逐本列出 scientific question、input、output、当前结论、最值得看的 1--3 个 sections/figures 与不可声称内容。它明确记录：

- `02` 的实际 baseline 为 14D，candidate augmented 为 23D；
- `03`/`04` 的 schema/extractor parity 为 NO-GO；
- `05` 没有训练，simulation count 与 seeds 均为 0；
- `06` 没有 posterior recovery、coverage 或 PPC 结果；
- `07` engineering adapter 成功但 semantic parity 失败；
- `08` 仅证明 normalized shape screening 可执行；
- `09` 证明当前单源模型仍缺 source/spatial measurement assumptions。

## 执行与验证

两个 notebooks 均在实际 `neurolib` 环境执行：

```text
Python executable = C:\Users\YUS190\AppData\Local\anaconda3\envs\neurolib\python.exe
Python = 3.10.20
kernel ID = python3
display name = Python 3 (ipykernel)

09: 7/7 code cells executed, execution counts 1..7, errors 0
00: 8/8 code cells executed, execution counts 1..8, errors 0
```

`nbformat.validate()`、全部 code-cell static compile、8 项相关 unit tests、12 个 CSV reload、全部 JSON reload、4 张 PNG 像素/尺寸检查、2 个 HTML export 和视觉检查均通过。HTML 的中文、公式、表格、图例和坐标轴可读，无空白 panel、NaN 扩散或 traceback。新增输出目录不含 NPZ/NPY/PT/PKL/EDF、raw EEG、simulator arrays、simulation bank 或 posterior。`01`--`08` 的 SHA-256 与本轮开始时一致。

## 研究者回来后必须决定的三个问题

1. 近期目标选择 Route 3 synthetic cortical-observable recovery，还是投入 Route 2 source/forward-model research program？
2. 若选择 Route 2，批准哪一种生物物理 source definition，以及使用哪些独立 calibration/validation 数据冻结 source geometry、dipole unit/gain 和 noise？
3. 若选择 Route 3，在生成任何 bank 前冻结哪一个 cortical observable、prior、extractor、simulation budget、seed protocol 和 synthetic diagnostics？

# 第二十六部分：2026-07-27 Route-3 Synthetic Recovery Preflight

## 本轮授权边界与结论

本轮只执行了研究者明确授权的 Route-3 synthetic cortical-observable preflight、三个中心的局部敏感性simulation，以及最多64点的prior-wide diagnostic simulations。没有训练SNPE/NPE，没有做真实EEG参数推断，没有执行任意`Hz -> uV`缩放，也没有把cortical firing rate或内部state称为simulated EEG。`00`--`09` notebooks、simulator核心机制、真实EEG extractor与科学threshold均未修改。

本轮总体结论为：

```text
Route-3 local numerical preflight = PASS
64-point prior-wide numerical diagnostic = PASS
local full-rank evidence = PASS at V7, V8, V8a for both schemas
global recoverability = NOT ESTABLISHED
next staged diagnostic-bank expansion = CONDITIONAL GO
SNPE/NPE training = NO-GO / NOT AUTHORIZED
real-EEG inference = NO-GO
```

## 冻结的8参数contract

精确顺序为：

```text
mue, mui, b, tauA, g_LK, g_h, c_th2ctx, c_ctx2th
```

prior采用当前V8a的独立uniform box，以容纳V7、V8和V8a三个中心：

| 参数 | 下界 | 上界 | 单位 |
|---|---:|---:|---|
| `mue` | 3.31075 | 4.47925 | mV/ms |
| `mui` | 2.57295 | 3.48105 | mV/ms |
| `b` | 28.4 | 42.6 | pA |
| `tauA` | 998.2 | 1853.8 | ms |
| `g_LK` | 0.02 | 0.07 | mS/cm^2 |
| `g_h` | 0.035 | 0.095 | mS/cm^2 |
| `c_th2ctx` | 0.0 | 0.075 | dimensionless coupling |
| `c_ctx2th` | 0.05 | 0.22 | dimensionless coupling |

V7历史上的`c_th2ctx`上界为0.05；V8a将其扩到0.075。V7 fitted、V8 fitted和V8a local best三个实际中心均位于V8a prior内。模型积分步长为0.1 ms，recording sampling interval为1 ms（1000 Hz），总时长60 s，去除5 s warm-up，只分析之后第一个完整30 s窗口，余下25 s不进入summary。simulator固定seed为42；局部差分使用common random numbers。failed simulation保留theta、以NaN和false validity记录x并保存明确reason，绝不填0。

旧5D/7D banks分别为`theta.shape=(5000,4), x.shape=(5000,5)`和`theta.shape=(4000,4), x.shape=(4000,7)`，只有4个参数，只作为历史证据，禁止改名、补列或复用为8参数训练bank。

## 两套嵌套Route-3 schemas

### A. `cortex_rate_only_14d`

只使用`r_mean_EXC`与`r_mean_INH`，每个信号各7项：

1. mean firing rate；
2. firing-rate standard deviation；
3. relative 0.5--1.5 Hz power；
4. relative 11--15 Hz power；
5. 0.5--1.5 Hz peak frequency；
6. 11--15 Hz peak frequency；
7. normalized spectral entropy over 0.5--20 Hz。

Welch固定为Hann、4 s window、50% overlap、0.25 Hz resolution。所有字段是synthetic cortical-rate observables，不使用75 uV threshold、scalp channel semantics、真实EEG calibration或任意单位缩放。

### B. `cortex_state_augmented_24d`

前14项与rate-only逐值、逐顺序完全相同，再加入10项：

```text
I_mu_exc mean/std
I_mu_inh mean/std
I_A mean/std
effective_drive mean/std
syn_mu_exc_on_exc mean
syn_mu_inh_on_exc mean
```

该schema明确是`privileged internal-state upper-bound experiment`。即使它提高synthetic recovery，也不能据此声称真实Fpz-Cz EEG可以恢复参数。

保留在inference schema之外的synthetic held-out diagnostics包括：EXC/INH zero-lag correlation、rate waveform quantiles、其他synaptic-state variability以及thalamic spindle mechanism metrics。后者只属于mechanism diagnostic，不是cortical observable或scalp EEG。

## 局部Jacobian结果

每个中心对每个参数采用prior width的对称2%扰动，共执行：

```text
3 centers * (1 center + 8 parameters * 2 directions) = 51 simulations
success = 51
failed = 0
NaN/Inf vectors = 0
artifacted runtime = 250.90 s
```

标准化Jacobian使用固定unit-aware feature scale。effective-rank规则在查看结果前冻结为：

```text
relative singular value >= 1e-3 * largest singular value
```

| 中心 | schema | effective rank | condition number |
|---|---|---:|---:|
| V7 fitted | rate-only 14D | 8 | 26.85 |
| V7 fitted | state-augmented 24D | 8 | 30.82 |
| V8 fitted | rate-only 14D | 8 | 33.92 |
| V8 fitted | state-augmented 24D | 8 | 13.02 |
| V8a local best | rate-only 14D | 8 | 17.45 |
| V8a local best | state-augmented 24D | 8 | 14.85 |

两套schema在三个中心均达到local full rank 8。privileged states没有增加rank，因为rate-only已经是8；它在V8和V8a改善了condition number，但在V7略微变差，因此不能简单声称内部state普遍改善可辨识性。

三个中心的最弱参数并不稳定：V7 rate-only最弱为`c_th2ctx`、`g_LK`和`mue`；V8 rate-only最弱为`b`、`mui`和`mue`；V8a rate-only最弱为`c_th2ctx`、`b`和`c_ctx2th`。`g_LK/g_h`在V8 rate-only的Jacobian-column cosine为0.985，在V8a为0.894，提示明显局部混淆；加入privileged states后仍分别为0.934和0.811。`c_th2ctx/c_ctx2th`在三个中心没有出现同等级的局部平行性，但部分参数的sensitivity direction在中心之间发生负cosine或反转。由此可见，局部full rank不等于跨prior全局injectivity。

## 64点diagnostic micro-bank

因为两套schema均满足固定order、fixed-length、local finite和明确failure policy，使用scrambled Sobol（seed 20260727）在完整V8a 8D prior内生成了：

```text
bank role = diagnostic_microbank_not_for_SNPE_training
theta shape = (64, 8)
rate-only x shape = (64, 14)
state-augmented x shape = (64, 24)
completed = 64
success = 64
failed = 0
NaN/Inf rows = 0
artifacted runtime = 316.45 s
median runtime = approximately 4.9 s/simulation
NPZ object arrays = none
```

每个simulation完成后立即原子checkpoint。24D前14列与14D矩阵严格相同。micro-bank中rate-only的主要高相关结构包括：EXC/INH SO peak frequency约0.996、EXC/INH relative SO power约0.981、EXC/INH spectral entropy约0.969。privileged schema还出现`r_inh_mean`与`I_mu_inh_mean`约0.999、`I_mu_exc_std`与`effective_drive_std`约0.988等强相关。高相关性是冗余风险筛查，不等于确定性重复，也没有在本轮自动删除feature。

## 科学判断与下一步门槛

`cortex_rate_only_14d`已提供三个中心的local full-rank证据和64点prior-wide numerical stability，因此允许下一轮按阶段扩大纯诊断bank，并设计held-out theta synthetic recovery。它仍不足以支持正式SNPE训练，原因包括：

1. 仅使用一个固定simulator seed，尚未验证stochastic-seed robustness；
2. sensitivity direction跨中心不一致，存在nonlinearity与全局collision风险；
3. 0.25 Hz离散peak-frequency summary可能产生量化不连续；
4. 64点不足以建立global injectivity、coverage或calibration；
5. scaling、train/validation/test split、failed-run policy、simulation budget和multi-seed protocol尚未正式冻结；
6. privileged internal-state结果不能外推到真实EEG。

因此推荐：

```text
next action:
  staged diagnostic bank expansion + multi-seed robustness
  + explicit global collision search
  + held-out synthetic theta recovery protocol

do not start:
  SNPE/NPE training
  real EEG inference
  posterior claims
```

## 新增交付与验证

新增：

```text
S4_sbi/src/sleep_sbi/route3_synthetic_preflight.py
S4_sbi/tests/test_route3_synthetic_preflight.py
S4_sbi/scripts/build_route3_preflight_notebook.py
S4_sbi/notebooks/10_Route3_Synthetic_Recovery_Preflight.ipynb
S4_sbi/results/route3_synthetic_preflight/
```

主要artifacts：

```text
route3_parameter_contract.json
route3_feature_dictionary.csv
route3_held_out_diagnostics.csv
local_sensitivity/checkpoints/*.json
local_sensitivity/jacobian.csv
local_sensitivity/rank.csv
local_sensitivity/sensitivity.csv
local_sensitivity/confounding.csv
local_sensitivity/direction_consistency.csv
diagnostic_microbank/diagnostic_microbank_64.npz
diagnostic_microbank/diagnostic_microbank_status.csv
diagnostic_microbank/*spearman.csv
validation_report.json
environment_report.json
html/10_Route3_Synthetic_Recovery_Preflight.html
```

执行证据：

```text
conda environment = neurolib
sys.executable = C:\Users\YUS190\AppData\Local\anaconda3\envs\neurolib\python.exe
sys.prefix = C:\Users\YUS190\AppData\Local\anaconda3\envs\neurolib
Python = 3.10.20
kernel ID = neurolib
display name = Python (neurolib)
Notebook code cells = 15/15 executed
Notebook error outputs = 0
unit tests = 3 passed
```

`nbformat.validate()`、全部code-cell静态compile、JSON/CSV/NPZ reload、feature order、shape、finite mask、schema version和no-object-array检查均通过。HTML完整导出并以Chrome headless检查全文：公式、表格、Jacobian、singular-value、sensitivity、Sobol coverage、runtime和redundancy图均正常，无空白panel、NaN扩散或traceback。本轮实际执行115个有artifact的simulation，加上1个独立端到端smoke run，共116次；Notebook后续Run All只复用经过contract-hash验证的checkpoint。

# 第二十七部分：2026-07-27 Route-3 Pilot SNPE、Held-out Recovery与正式NO-GO决策

## 授权范围与不可越过的科学边界

本轮完成研究者授权的Route-3 multi-seed/global robustness、2048点bank、三个独立NPE、等权ensemble、128-case held-out recovery、coverage、SBC与synthetic PPC。没有使用真实SC4001 EEG选择、校准或缩放inference features，没有执行`Hz -> uV`映射，没有把cortical firing rate称为simulated EEG，没有运行sequential rounds，也没有修改`00`--`10` notebooks。

所有结论只涉及：

```text
8 parameters
-> neurolib thalamocortical dynamics
-> cortical EXC/INH population firing rates
-> frozen cortex-rate-only 14D summaries
-> exploratory synthetic posterior
```

它不解决cortical source到Fpz-Cz的measurement-model blocker，也不允许真实EEG-SNPE。

## Notebook 11：Multi-seed与Global Robustness

实际执行：

```text
prior-wide: 128 theta x 3 seeds = 384
local multi-scale: 3 centers x 3 seeds x 49 runs = 441
total = 825
failed = 0
non-finite = 0
simulation runtime sum = 4860.31 s
wall time = 831.2 s
```

same-theta/same-seed重复逐值完全一致；不同seed改变12/14 summaries，证明新wrapper真实控制了ALN EXC、ALN INH、TCR、TRN stochastic input seeds和numba RNG。

1%、2%、5%三种扰动尺度下，V7/V8/V8a三个中心与三个seed的27个Jacobian全部effective rank 8。condition number范围18.85--80.93；1%尺度最不稳定，median condition约44.43。该结果只支持local full-rank，不支持global injectivity。

seed/global警告：

- all-three-seed nearest-neighbor agreement仅28.9%；
- pairwise nearest-neighbor agreement为39.1%--46.9%；
- EXC/INH sigma peak frequency SNR仅1.53/3.14；
- SO peak每个seed只有5个离散值，sigma peak只有12--13个；
- 多个参数的Jacobian方向随seed、center或scale反转；
- 128点audit检测到1469个collision和3670个near collision。

工程gate通过，因此依授权继续exploratory pilot，但global identifiability没有成立。

## Notebook 12：2048点Simulation Bank

```text
attempted = 2048
valid = 2048
failed = 0
training = 1638
validation = 410
unique simulator seeds = 2048
simulation runtime sum = 11446.81 s
wall time = 1936.4 s
```

bank使用独立scrambled Sobol sequence和逐sample seed schedule。每个simulation均有原子NPZ checkpoint；split固定后，14D median/IQR scaling只用1638个training rows拟合。

高分辨率collision audit：

```text
all pairs = 2,096,128
far-theta pairs = 1,999,861
inside seed-noise floor = 327,248
within two noise floors = 939,831
collision fraction among far pairs = 16.36%
```

这构成严重global-identifiability warning，但不违反预注册的training工程gate。

## Notebook 13：三个Exploratory NPE与等权Ensemble

三个single-round MAF均使用相同bank、split、architecture、batch policy和early-stopping规则，只改变初始化/training seed：

| Seed | Best epoch | Best validation loss | 初次训练runtime |
|---:|---:|---:|---:|
| 1301 | 294 | -8.4982 | 48.02 s |
| 1302 | 267 | -8.5618 | 46.75 s |
| 1303 | 295 | -8.2378 | 47.02 s |

初次ensemble training总wall time约144.02 s，设备为CPU。工程screening中的posterior samples均finite并位于prior support。ensemble严格使用三成员等数量采样，不学习held-out-dependent权重。

GO criteria在独立held-out theta生成前冻结：

```text
frozen_utc = 2026-07-27T20:58:09.411016+00:00
version = route3-heldout-go-criteria-v1
immutable_after_heldout_open = true
```

## Notebook 14：128-case Held-out Recovery

```text
held-out cases = 128
training-theta exact duplicates = 0
held-out failures = 0
posterior samples per case = 4096
posterior tensor = (128, 4096, 8)
finite/in-prior = 100%
```

参数恢复：

| 参数 | Posterior median MAE | Prior median MAE | 改善 | rank correlation | Median 90% width |
|---|---:|---:|---:|---:|---:|
| mue | 0.1545 | 0.2500 | 38.2% | 0.754 | 0.516 |
| mui | 0.0050 | 0.2500 | 98.0% | 0.999 | 0.021 |
| b | 0.0952 | 0.2500 | 61.9% | 0.894 | 0.389 |
| tauA | 0.1912 | 0.2500 | 23.5% | 0.550 | 0.651 |
| g_LK | 0.1579 | 0.2500 | 36.9% | 0.682 | 0.504 |
| g_h | 0.1338 | 0.2500 | 46.5% | 0.793 | 0.457 |
| c_th2ctx | 0.1505 | 0.2500 | 39.8% | 0.741 | 0.501 |
| c_ctx2th | 0.2515 | 0.2500 | -0.6% | -0.039 | 0.780 |

7/8参数优于prior-median baseline；`c_ctx2th`完全没有恢复。

ensemble empirical coverage：

| 参数 | 50% | 80% | 90% |
|---|---:|---:|---:|
| mue | 0.383 | 0.703 | 0.820 |
| mui | 0.742 | 0.914 | 0.961 |
| b | 0.531 | 0.781 | 0.898 |
| tauA | 0.391 | 0.664 | 0.820 |
| g_LK | 0.383 | 0.703 | 0.828 |
| g_h | 0.414 | 0.727 | 0.852 |
| c_th2ctx | 0.375 | 0.719 | 0.805 |
| c_ctx2th | 0.352 | 0.633 | 0.773 |

仅`b`的80%与90% Wilson intervals同时包含nominal coverage。`c_ctx2th`在80% level满足预冻结的severe-undercoverage定义。ensemble SBC的10-bin descriptive warnings包括`mui`、`tauA`与`c_ctx2th`；没有使用单个p-value决定GO。

ensemble disagreement gate通过：

```text
mean member-median range = 0.051 prior width
largest parameter-level mean range = 0.072 prior width
```

`g_LK/g_h`在不同case中呈可变ridge，median posterior Spearman rho约0.154，最大约0.618；两者均改善恢复但coverage多数低于nominal。双向coupling posterior平均相关接近0不代表两个方向均可辨识：`c_th2ctx`部分可恢复但undercovered，`c_ctx2th`不可恢复且严重undercovered。

## Synthetic PPC

固定随机选择32个held-out cases，每个case使用16个posterior theta和16个prior theta，全部用新的simulator seeds：

```text
posterior predictive simulations = 512, failed = 0
prior predictive simulations = 512, failed = 0
features better than prior = 14/14
posterior scaled error = 0.0964
prior scaled error = 0.6433
overall improvement = 85.0%
```

强PPC不能覆盖calibration失败，因为global theta collisions仍可产生相似14D summaries。

## 预冻结规则下的正式决定

```text
FINAL DECISION = NO-GO
```

通过的标准：

- training与held-out failure gates；
- posterior finite/prior-support gate；
- 7/8 recovery优于prior baseline；
- overall recovery改善43.0%；
- 14/14 PPC features改善；
- overall PPC改善85.0%；
- ensemble disagreement gate。

失败的标准：

- 只有1/8参数同时满足80%与90% nominal coverage compatibility；
- 1个参数出现severe undercoverage；
- 只有1/8参数同时满足contraction与coverage要求；
- `c_ctx2th`不优于prior；
- global collisions占far pairs约16.36%。

因此不能声称全部8参数在Route-3 synthetic cortical-rate空间中validated。

## 下一步建议

不自动生成4K/8K bank，不启动真实EEG inference。优先开展新的、重新预注册并使用全新held-out set的synthetic实验：

1. 固定`c_ctx2th`，测试7参数contract；
2. 或为双向coupling引入有科学依据的reparameterization/constraint；
3. 独立审计calibration-aware training与density-estimator capacity；
4. 若修改量化peak-frequency features，必须建立新schema版本，不得在当前held-out结果上调参；
5. Route 2 measurement model仍是任何Fpz-Cz inference的独立前置条件。

## 交付与验证

新增Notebook：

```text
11_Route3_Global_Robustness.ipynb
12_Route3_2048_Simulation_Bank.ipynb
13_Route3_Exploratory_SNPE_Ensemble.ipynb
14_Route3_Heldout_Recovery_and_Coverage.ipynb
```

新增模块：

```text
route3_global_robustness.py
route3_pilot_snpe.py
route3_heldout_validation.py
```

完整handoff位于：

```text
docs/route3_pilot_snpe_handoff.md
```

验证结果：

```text
Notebook code cells executed = 35/35
Notebook error outputs = 0
unit tests = 7 passed
JSON reload = 17
CSV reload = 22
NPZ reload without pickle = 4161
PyTorch checkpoints reload = 6
unexpected object arrays = 0
HTML exports = 4
visual checks = passed
```

本轮未commit、未push，也未修改`00`--`10` notebooks。

## 二十八、Route-3 七参数独立正式验证（2026-07-28）

本轮按照预注册规则固定 `c_ctx2th = 0.1253491302153237`，只推断：

```text
mue, mui, b, tauA, g_LK, g_h, c_th2ctx
```

固定值来自 Notebook 10 明确运行的 canonical `V8a local best`：

```text
source = outputs/v8a_ultra_narrow_t6_t13_search/best_so_far.json
source SHA-256 = BF18212EBAC0EBEC39D4F13AECD5B7A9DA5618FFE78ABB5F8694A67F569AF76A
```

预注册与 held-out criteria 在新科学模拟和 held-out 结果前分别锁定：

```text
preregistration SHA-256 = 2cedb6a68de307ab2f9233b4009a8f58854316259b6c185aeb61af5efdcbc363
held-out criteria SHA-256 = 49a4f9590466a6d9ba83f83ee5b6783e268ba996f8e8fc7328e261fb0405edee
```

14D cortex-rate-only schema、旧 8D prior 的七个 marginal ranges、simulator duration/warm-up/sampling、feature order 和 validity policy 均未改变。新旧 simulator seeds 无交集；新 train/held-out 与旧 8D train/held-out 的完整 8D 和自由 7D exact duplicates 均为 0。

### Notebook 15：7D robustness

```text
prior-wide = 384
local multi-scale = 387
total simulations = 771
valid = 771
failed = 0
sum simulator runtime = 4372.91 s
```

1%、2%、5% 下 27 个 Jacobian 全部 rank 7，工程 hard gate 通过。但三 seed nearest-neighbor 全一致率仅 25.8%；128 点 audit 有 2,106 collisions 和 3,869 near collisions；sigma peak SNR 仅 1.66/2.31，SO/sigma peak 仍有明显频率格点量化。

### Notebook 16：4,096 点独立 bank

```text
scheduled/valid/failed = 4096 / 4096 / 0
train/validation = 3276 / 820
sum simulator runtime = 22935.91 s
collision fraction among far pairs = 27.83%
```

Scaling 只用 training split 拟合。高 collision fraction 是严重 global-identifiability warning。

### Notebook 17：三成员 NPE

| Seed | Best epoch | Best validation loss | Runtime |
|---:|---:|---:|---:|
| 7330001 | 290 | -10.1513 | 94.44 s |
| 7330002 | 287 | -10.2580 | 90.62 s |
| 7330003 | 299 | -9.7485 | 88.51 s |

三个模型均为 single-round MAF，使用相同 bank/split/policy、不同初始化 seed 和等权 mixture。Validation-only finite/in-prior rate 均为 100%。

### Notebook 18：256-case held-out

```text
held-out valid/failed = 256 / 0
posterior tensor = (256, 4096, 7)
posterior finite/in-prior = 100%
```

| 参数 | Prior MAE | Posterior MAE | 改善 | Rank correlation | 80% coverage | 90% coverage |
|---|---:|---:|---:|---:|---:|---:|
| mue | 0.2500 | 0.1467 | 41.3% | 0.774 | 0.672 | 0.812 |
| mui | 0.2500 | 0.0049 | 98.0% | 0.999 | 0.902 | 0.973 |
| b | 0.2500 | 0.0879 | 64.8% | 0.908 | 0.746 | 0.871 |
| tauA | 0.2500 | 0.1863 | 25.5% | 0.593 | 0.676 | 0.824 |
| g_LK | 0.2500 | 0.1512 | 39.5% | 0.693 | 0.703 | 0.820 |
| g_h | 0.2500 | 0.1256 | 49.8% | 0.822 | 0.727 | 0.836 |
| c_th2ctx | 0.2500 | 0.1453 | 41.9% | 0.768 | 0.691 | 0.832 |

7/7 点估计优于 prior median，整体 MAE 改善 51.6%。但 0/7 参数同时满足 80% 和 90% Wilson coverage compatibility，contraction-with-coverage 同样为 0/7。所有 ensemble SBC 10-bin descriptive p-values 均小于 0.05；该结果只与 coverage、bias 和 contraction 联合解释。

64-case PPC 使用 32 个固定随机和 32 个互斥的 worst-recovery cases：

```text
posterior predictive = 1024, failed = 0
prior predictive = 1024, failed = 0
features better than prior = 14/14
overall PPC improvement = 83.3%
```

### 正式裁决

```text
FINAL DECISION = NO-GO
```

失败原因不是工程故障，而是 uncertainty calibration：

- coverage-compatible parameters 为 0/7，预注册要求至少 6/7；
- contraction-with-reasonable-coverage 为 0/7，预注册要求至少 6/7；
- 4,096 bank 的 far-theta collision fraction 为 27.83%；
- SBC 对全部参数给出分布不一致警告。

因此不能声称七参数在 synthetic cortical-rate observable space 中已正式 validated。强 recovery 和 PPC 不能覆盖 calibration 失败。

新增：

```text
15_Route3_7D_Preregistration_and_Robustness.ipynb
16_Route3_7D_4096_Simulation_Bank.ipynb
17_Route3_7D_SNPE_Ensemble.ipynb
18_Route3_7D_Heldout_Recovery_Coverage_PPC.ipynb
route3_7d_experiment.py
route3_7d_training.py
route3_7d_validation.py
docs/route3_7d_formal_validation_handoff.md
```

验证：

```text
Notebook code cells executed = 19/19
Notebook errors = 0
tests = 8 passed
JSON/CSV/NPZ/PT reload = passed
unexpected object arrays = 0
HTML exports = 4
visual checks = passed
```

真实 EEG inference 继续 NO-GO，直到存在经过独立验证的 cortical-source 到 Fpz-Cz measurement/forward model。本轮未 commit、未 push，也未修改 Notebook 00–14。

## 二十九、Route-3 七参数 coverage rescue 与全新独立终检（2026-07-28）

本轮在不修改 Notebook 15–18、不放宽任何原判决门槛的前提下，完成了 coverage failure 诊断、救援预注册、额外 stochastic training bank、两套五成员 NPE ensemble、development-only calibration，以及一次性打开的 1,024-case 全新终检。

### 失败诊断与救援预注册

对原 Notebook 18 的 256-case posterior 进行了独立复算。训练/验证 split 无泄漏，feature scaler 与只使用 training rows 重算的 median/IQR 完全一致，参数归一化与 prior support 正确，三个网络按 `1366/1365/1365` 等数混合，4096 posterior samples 的前后两半 coverage 结果稳定。没有发现 parameter transform、inverse transform、credible interval、ensemble pooling、NaN、clipping 或 Monte Carlo sample-count bug。

原 undercoverage 在 prior 边界样本最严重，并在每个单网络中同时出现，因此定位为真实 calibration failure，而不是展示或实现错误。

在任何新科学 simulation 前冻结：

```text
rescue preregistration SHA-256 =
01e0b0cb2277340872d95fef5fb5f2c0dd15bdf7104d363bd67fb41f1acd4a02
```

七参数、prior bounds、固定 `c_ctx2th`、14D cortex-rate-only schema、simulator contract、coverage/SBC/PPC 定义和原 numerical GO logic 均未改变。新增 training、development 和 final 的 Sobol sequences、simulator seeds 与旧实验及彼此的 exact intersection 均为 0。

### 新 training bank 与 NPE rescue

新增 2,048 个 theta，每个 theta 使用两个独立 simulator seeds：

```text
additional rows = 4096
valid / failed = 4096 / 0
sum simulator runtime = 26825.54 s
median simulator runtime = 5.94 s
```

与原 4,096 rows 合并后：

```text
combined rows = 8192
unique theta groups = 6144
train / validation rows = 6583 / 1609
train / validation theta groups = 4915 / 1229
same-theta cross-split leakage = false
scaling fit on training only = true
```

按预注册的有限候选集训练：

```text
maf64_t5:  5 independent initializations
maf128_t8: 5 independent initializations
single-round prior proposal; no sequential SNPE
```

全部 10 个模型训练成功。`maf64_t5` best epochs 为 284–298，单模型 runtime 168.5–174.0 s；`maf128_t8` best epochs 为 176–212，单模型 runtime 252.3–303.8 s。所有 ensemble 均为透明的等权、等数 mixture。

独立 development set 为 512/512 valid、0 failure。两套 raw ensemble 在 development 上都只有 1/7 参数同时满足 80%/90% coverage compatibility。预注册的单一 whole-pipeline marginal empirical-rank calibrator 在 development 上达到 7/7，但原 preregistration 没有授权 calibration，因此该方法从一开始就不具备 Formal GO 资格。冻结 primary 为：

```text
architecture = maf128_t8
method = empirical_rank_calibrated
primary pipeline SHA-256 =
6c46a981b81112005fe09f1be4436426d88b48bf96a93643ec0272ce1d1c75b8
```

### 1,024-case 全新独立终检

终检在 primary pipeline 和 calibrator 冻结后只打开一次：

```text
fresh final scheduled / valid / failed = 1024 / 1024 / 0
posterior samples per case = 4096
posterior finite and inside prior = 100%
```

未校准 raw ensemble 在终检上仍为 0/7 coverage-compatible parameters，整体 point-recovery improvement 为 53.3%。冻结 calibrated primary 的结果为：

| 参数 | Prior MAE | Posterior MAE | 改善 | Rank correlation | 80% coverage | 90% coverage | 两级均兼容 |
|---|---:|---:|---:|---:|---:|---:|---|
| mue | 0.2500 | 0.1452 | 41.9% | 0.777 | 0.772 | 0.888 | 否 |
| mui | 0.2500 | 0.0034 | 98.6% | 1.000 | 0.824 | 0.914 | 是 |
| b | 0.2500 | 0.0792 | 68.3% | 0.927 | 0.789 | 0.870 | 否 |
| tauA | 0.2500 | 0.1836 | 26.5% | 0.601 | 0.812 | 0.898 | 是 |
| g_LK | 0.2500 | 0.1430 | 42.8% | 0.717 | 0.807 | 0.900 | 是 |
| g_h | 0.2500 | 0.1214 | 51.4% | 0.814 | 0.831 | 0.900 | 否 |
| c_th2ctx | 0.2500 | 0.1401 | 44.0% | 0.764 | 0.801 | 0.902 | 是 |

calibrated primary 的 4/7 参数同时满足 80% 和 90% Wilson compatibility，且 4/7 同时满足 coverage 加 contraction；原 Formal GO 要求至少 6/7。`mue` 的 80%、`b` 的 90% 和 `g_h` 的 80% compatibility 失败。SBC 仍对 `mui`、`g_LK` 和 `g_h` 给出显著非均匀警告，需与 coverage、bias 和 contraction 联合解释。

64-case synthetic PPC 仍使用 32 个固定随机和 32 个预定义 worst-recovery cases：

```text
posterior predictive = 1024, failed = 0
prior predictive = 1024, failed = 0
features better than prior = 14/14
overall PPC improvement = 87.7%
ensemble disagreement mean = 0.0474 prior width
ensemble disagreement parameter maximum = 0.0664 prior width
```

### 唯一最终裁决

```text
FINAL DECISION = NO-GO
```

冻结 primary 虽然整体 recovery 改善 53.4%、PPC 改善 87.7%、7/7 点估计优于 prior median，且没有 severe undercoverage、simulation failure 或 ensemble instability，但只有 4/7 参数通过原始 coverage gate，也只有 4/7 通过 coverage-plus-contraction gate。因此同时未达到 Formal GO 和 Conditional GO，不能用 point recovery 或 PPC 覆盖 uncertainty calibration 的失败。

这说明当前结果仍不能声称七参数在 synthetic cortical-rate observable space 中已正式 validated。真实 EEG inference 继续 NO-GO；本轮没有解决 cortical-source 到 Fpz-Cz measurement/forward model blocker。

### 新增文件与验证

```text
S4_sbi/notebooks/19_Route3_7D_Coverage_Failure_Diagnosis.ipynb
S4_sbi/notebooks/20_Route3_7D_Rescue_Preregistration.ipynb
S4_sbi/notebooks/21_Route3_7D_Rescue_Training.ipynb
S4_sbi/notebooks/22_Route3_7D_Fresh_Heldout_Validation.ipynb
S4_sbi/notebooks/23_Route3_7D_Rescue_Handoff.ipynb
S4_sbi/src/sleep_sbi/route3_7d_rescue.py
S4_sbi/src/sleep_sbi/route3_7d_rescue_training.py
S4_sbi/src/sleep_sbi/route3_7d_rescue_validation.py
S4_sbi/src/sleep_sbi/route3_7d_rescue_reporting.py
S4_sbi/scripts/run_route3_7d_rescue.py
S4_sbi/scripts/orchestrate_route3_7d_rescue.py
S4_sbi/scripts/build_route3_7d_rescue_notebooks.py
S4_sbi/tests/test_route3_7d_rescue.py
S4_sbi/configs/route3_7d_rescue_preregistered_v1.json
S4_sbi/artifacts/route3_7d_rescue_preregistered_v1.locked.json
ROUTE3_7D_RESCUE_FINAL_DECISION.md
ROUTE3_7D_RESCUE_FINAL_DECISION.json
```

验证结果：

```text
Notebook code cells executed = 23/23
Notebook error outputs = 0
kernel ID / display name = neurolib / neurolib
sys.executable = C:\Users\YUS190\AppData\Local\anaconda3\envs\neurolib\python.exe
tests = 8 passed
JSON / CSV / NPZ / PT reload = 30 / 40 / 9739 / 20
unexpected object arrays = 0
HTML exports = 5
visual checks = passed
Notebook 15–18 SHA-256 unchanged = true
```

完整人类可读报告位于 `ROUTE3_7D_RESCUE_FINAL_DECISION.md`，机器可读裁决位于 `ROUTE3_7D_RESCUE_FINAL_DECISION.json`。本轮未 commit、未 push，也未修改 Notebook 15–18。
