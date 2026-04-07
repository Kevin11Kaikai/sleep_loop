# Sleep-Loop 项目进度报告 — 2025-03-15

> 运行环境：`conda activate neurolib`（Python 3.x，NumPy 2.2，neurolib 本地源码安装）  
> 项目根目录：`D:\Year3_Mao_Projects\sleep_loop\`  
> 所有脚本均从项目根目录执行：`python models/<script>.py`

---

## 脚本总览

| 脚本 | 功能一句话 | 主要输出 |
|------|-----------|---------|
| `s1_trusted_segments.py` | 探查 Sleep-EDF 数据目录，确认文件配对 | 打印信息，无文件写入 |
| `s1_all_stages.py` | 提取五阶段（W/N1/N2/N3/REM）平均 PSD | `data/psd_*.npy`, `data/target_psd.npy` |
| `s2_skeleton.py` | 加载 HCP 脑连接组结构矩阵 | `data/cmat.npy`, `data/dmat.npy` |
| `s3_sleep_kernel.py` | 运行 60 s 丘脑-皮层 MultiModel 仿真 | `outputs/r_cortex.npy`, `outputs/r_thalamus.npy` |
| `s4_personalize.py` | SC4001 个体化参数拟合（归一化 PSD + 三步适应度函数） | `data/patient_params_SC4001.json`，score=0.881 |
| `s5_bifurcation.py` | 分岔图分析（含 MultiModel→ALN 坐标映射修正） | `outputs/bifurcation_map.png`，距边界 0.793 mV/ms |

---

## s1_trusted_segments.py — 数据探查

### 输入
- `data/manifest.csv`（153 行，含 `subject_id / psg_path / hypnogram_path`）
- `data/sleep-edfx-cassette/*.edf`（磁盘文件）

### 运行过程
1. 读取 manifest，统计磁盘上有效 PSG/Hypnogram 配对数
2. 用 `glob` 独立扫描 EDF 文件做交叉核验
3. 用 MNE 打开第一个受试者（SC4001）的 PSG 文件
4. 打印通道列表、采样率、时长
5. 读取 Hypnogram annotations，打印所有唯一描述标签

### 关键输出（打印，无文件写入）
```
Found 306 EDF files  |  PSG: 153  |  Hypnogram: 153  |  Valid pairs: 153/153
Channels: ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', ...]
Sfreq: 100.0 Hz  |  Duration: 79500.0 s (22.08 h)
Unique descriptions: ['Sleep stage 1', '2', '3', '4', 'R', 'W', '?']
```

### 验证结论
- EEG 目标通道 `EEG Fpz-Cz` ✓ 和 `EEG Pz-Oz` ✓ 均存在
- 慢波睡眠：`Sleep stage 3` / `Sleep stage 4` ✓
- REM：`Sleep stage R` ✓

---

## s1_all_stages.py — 全睡眠阶段 PSD 提取

### 输入
- `data/manifest.csv`（取前 10 名受试者）
- `data/sleep-edfx-cassette/<subject>-PSG.edf`（EEG 通道：Fpz-Cz + Pz-Oz）
- `data/sleep-edfx-cassette/<subject>-Hypnogram.edf`

### 运行过程
1. 按阶段标签（`Sleep stage W/1/2/3/4/R`）从 Hypnogram 中提取事件
2. 对每个 30 s epoch 进行伪迹检测（峰峰值 > 200 µV 则丢弃）
3. 两通道均值信号 → Welch PSD（10 s 窗，50% 重叠，0.5–30 Hz）
4. 聚合所有受试者所有通过 QC 的 epoch，计算各阶段平均 PSD

### epoch 统计（10 名受试者，QC 后）
| 阶段 | Epoch 数 | 备注 |
|------|---------|------|
| wake | 75 | 清醒期伪迹多，通过率偏低属正常 |
| n1 | 296 | — |
| n2 | 429 | — |
| **n3** | **280** | **← Session 2-B 目标** |
| rem | 148 | — |

### 输出文件
| 文件 | 内容 | 形状 |
|------|------|------|
| `data/target_freqs.npy` | 统一频率轴 (0.5–30 Hz) | (296,) |
| `data/psd_wake/n1/n2/n3/rem.npy` | 各阶段平均 PSD | (296,) |
| `data/target_psd.npy` | N3 平均 PSD（供 s4 使用） | (296,) |
| `data/epochs_{stage}.csv` | 每阶段 epoch 索引记录 | — |
| `outputs/psd_all_stages.png` | 五阶段绝对 + 归一化 PSD 对比图 | — |

### 验证结论
- N3 delta 主导率 = **0.86**（delta 0.5–4 Hz 占总功率 86%，符合慢波睡眠生理）
- N3 delta 功率 > Wake delta 功率 ✓

---

## s2_skeleton.py — HCP 脑连接组加载

### 输入
- neurolib 内置数据集：`Dataset("hcp")`（自动下载，约 3 MB）

### 运行过程
1. 加载 HCP 平均连接组：`Cmat`（连接强度，max 归一化）和 `Dmat`（纤维长度 mm）
2. 打印矩阵形状、非零元素数、最大/最小值
3. 保存为 numpy 文件

### 输出文件
| 文件 | 内容 | 形状 |
|------|------|------|
| `data/cmat.npy` | 连接强度矩阵（max 归一化） | (80, 80) |
| `data/dmat.npy` | 纤维长度矩阵（mm） | (80, 80) |

### 关键统计
```
Cmat: non-zero=6320, min=0.0, max=1.0, mean=0.0225
Dmat: non-zero=6320, min=0.0, max=248.35 mm, mean=128.48 mm
N = 80 brain regions (AAL2 atlas)
```

---

## s3_sleep_kernel.py — 丘脑-皮层 MultiModel 仿真

### 输入
- 无外部数据文件
- 参数来自 neurolib 论文 Table 3（Cakan et al. 2021）+ 本地校正

### 模型结构
```
ALNNode (皮层，节点 0) ←──── cTh→ALN=0.15 ────→ ThalamicNode (节点 1)
                       ──── cALN→Th=0.02 ────→
```
- `ALNNode`：自适应 Leaky Integrate-and-Fire 平均场（兴奋 + 抑制两个 mass）
- `ThalamicNode`：丘脑皮层中继核（TCR）+ 丘脑网状核（TRN）

### 关键参数（经过 Session 1-B 调试验证）
| 参数 | 值 | 说明 |
|------|----|------|
| `mue` | 3.20 | 皮层兴奋性背景输入（原 Table 3=2.30，MultiModel 有效阈值更高）|
| `mui` | 3.50 | 皮层抑制性背景输入 |
| `b` | 19.5 pA | 适应电流幅度 |
| `tauA` | 1040 ms | 适应电流时间常数 |
| 时长 | 60 s | `duration=60000` ms |
| `backend` | numba | JIT 编译，仅首次慢 |

### 重要 API 修正（实测发现）
- `Network.__init__` 参数：`nodes=`, `connectivity_matrix=`, `delay_matrix=`（非文档写法）
- `ALNNode(exc_seed=42, inh_seed=42)`（无 `seed` 参数）
- `sync_variables` 需手动覆写 `_sync()`，覆盖所有 mass 的 `required_couplings`
- `noise_input_idx` 必须在 `super().__init__()` 前设置
- `tc_model["t"]` 返回秒（非毫秒）

### 输出文件
| 文件 | 内容 | 形状 |
|------|------|------|
| `outputs/t_ms.npy` | 时间轴（秒） | (60000,) |
| `outputs/r_cortex.npy` | 皮层兴奋性发放率（Hz） | (60000,) |
| `outputs/r_thalamus.npy` | 丘脑 TCR 发放率（Hz） | (60000,) |

### 仿真结果
```
Cortex  r_E    : 0.00 – 66.29 Hz  （up/down state 切换）
Thalamus r_TCR : 0.04 – 12.00 Hz  （稳定振荡）
耗时：~13 秒（含 numba JIT 编译）
```

### tests/test_spindles.py 验证
| 断言 | 结果 |
|------|------|
| 丘脑纺锤波峰 (10–15 Hz)，ratio ≥ 2 | ✓ ratio = 3.7，峰在 10.50 Hz |
| 皮层慢波能量 (0.2–1.5 Hz)，ratio ≥ 0.5 | ✓ ratio = 55.4，峰在 0.50 Hz |
| 皮层不全为零，max > 0.01 Hz | ✓ max = 66.29 Hz |

---

## s4_personalize.py — 进化算法参数拟合（个体化版本）

### 输入
- SC4001 个人 N3 EEG（从 PSG 文件实时提取，57 个 epoch 通过 QC）
- `data/target_freqs.npy`（频率轴，0.5–30 Hz）

### 重要修正历程

#### 问题一：目标 PSD 来源错误（已修复）
原始版本使用 10 个受试者的 N3 群体平均 PSD 作为拟合目标，导致拟合的是"群体平均人"而非 SC4001 个体。
修复方案：在脚本内部直接加载 SC4001 的 PSG 文件，提取 57 个 N3 epoch，计算个人平均 PSD 作为目标。

   #### 问题二：适应度函数量纲差异（已修复）
   原始适应度函数（log₁₀ 域 Pearson r）存在根本性问题：
   - 仿真 PSD 单位：Hz²/Hz，量级约 1e-3
   - 目标 PSD 单位：V²/Hz，量级约 1e-11
   - 两者相差 8 个数量级，但 Pearson r = 0.982（虚高）
   - 原因：Pearson r 自动去均值，纵向偏移被完全消除，只捕捉共同的 1/f 下降趋势

   修复方案：三步适应度函数
   ```
   Step 1: 归一化（各除以总功率）→ 消除量纲差异
   Step 2: log 域 Pearson r → 比较归一化后的谱形状（0.6 权重）
   Step 3a: delta 占比奖励 → 惩罚 delta 偏离目标（0.3 权重）
   Step 3b: sigma 惩罚项 → 基于诊断动态设置
            SC4001 N3 的 sigma 占比仅 1.1%（81x delta/sigma 比值）
            → 走惩罚路径：仿真纺锤波越强越扣分（0.1 权重）
   综合得分 = 0.6 × r_shape + 0.3 × delta_bonus + 0.1 × sigma_term
   ```

### 目标 PSD 诊断（SC4001 N3）
| 频段 | 占比 | 说明 |
|------|------|------|
| delta (0.5–4 Hz) | **90.9%** | 极强慢波主导，N3 典型 |
| sigma (10–15 Hz) | 1.1% | 纺锤波极弱（delta/sigma = 81x）|
| beta (15–30 Hz) | 0.3% | 几乎无 |

TARGET_DELTA_RATIO = 0.909，TARGET_SIGMA_RATIO = 0.011

### 运行过程
1. 加载 SC4001 的 PSG + Hypnogram，提取 N3 epoch（57/71 通过 QC）
2. 计算 SC4001 个人 N3 平均 PSD 作为拟合目标
3. 构建 ThalamoCorticalNetwork（同 s3_sleep_kernel.py）
4. 用 `scipy.optimize.differential_evolution` 做 4 维参数搜索

| 参数 | 搜索范围 | 物理含义 |
|------|---------|---------|
| `mue` | [2.5, 4.5] | 皮层兴奋性背景输入 |
| `mui` | [2.5, 5.0] | 皮层抑制性背景输入 |
| `g_LK` | [0.02, 0.20] | TCR K-漏电导 |
| `g_h` | [0.02, 0.20] | TCR h-整流电流电导 |

MVP 规模：popsize=5×4=20 个/代，maxiter=10 代，共 220 次评估

> **API 修正**：`neurolib.optimize.evolution.Evolution` 依赖 `pypet`，后者与 NumPy ≥ 2.0 不兼容（`np.string_` 已删除）。用 `scipy.optimize.differential_evolution` 替代。

### 输出文件
| 文件 | 内容 |
|------|------|
| `data/patient_params.json` | 最优参数（SC4001 个性化） |
| `data/patient_params_SC4001.json` | 同上，含 subject_id 标识 |
| `data/target_psd_SC4001.npy` | SC4001 个人 N3 目标 PSD |
| `outputs/evolution_records.csv` | 220 个个体的参数 + 适应度记录 |
| `outputs/evolution_result.png` | 进化曲线 + 适应度分布 + 归一化 Sim vs Target PSD |

### 最优参数（data/patient_params_SC4001.json）
```json
{
  "mue":      3.009,
  "mui":      4.077,
  "g_LK":     0.128,
  "g_h":      0.025,
  "b":        19.5,
  "tauA":     1040.0,
  "psd_corr": 0.881,
  "subject_id": "SC4001",
  "n3_epochs_used": 57
}
```

**综合 score = 0.881（真实可信，不再虚高）**
- sim delta=0.88 vs tgt delta=0.91，差距仅 3% ✓
- g_h=0.025 在搜索范围低端，进化算法在压制丘脑纺锤波 ✓
耗时：~207 秒（220 次 × ~0.94 s/次）

### 残余问题（已知，待迭代）
仿真 PSD 在 10–30 Hz 存在多个谐波峰（约 12、16、22、27 Hz），来源于
ThalamicNode TCR-TRN 回路的非线性振荡。sigma 惩罚项压制了主峰，但
谐波仍存在。对 Session 3 RL 控制不构成阻塞。

---

## s5_bifurcation.py — 分岔图分析（含坐标映射修正）

### 输入
- `data/patient_params.json`（来自 s4_personalize.py）

### 重要修正：坐标映射
MultiModel ALNNode 的 `input_0.mu` 和核心 ALNModel 的 `mue_ext_mean` 物理含义相同，
但有效数值尺度不同（Session 1-B 实测）：
- MultiModel mue=3.20 → 产生慢波
- 核心 ALN mue_ext_mean=2.30（Table 3）→ 产生同样慢波
- 经验映射比例：SCALE = 0.76

修正后的病人坐标：
```
MultiModel  : mue=3.009, mui=4.077  （patient_params.json 原始值）
ALN (×0.76) : mue=2.287, mui=3.098  （分岔图上的实际标注位置）
```

修正前（错误）：红星在图的右上角高参数区 (3.01, 4.08)，脱离 ALN 物理范围  
修正后（正确）：红星在图中部偏左 (2.29, 3.10)，位于 up-state 主导区，物理位置合理

### 运行过程
1. 用 `ALNModel`（单节点核心模型）做 **21×21 二维网格扫描**
2. 扫描轴：`mue_ext_mean ∈ [0, 4]` vs `mui_ext_mean ∈ [0, 5]`
3. 每个格点：2 s 无噪声仿真（`sigma_ou=0`），记录最后 1 s 的 `max_r / min_r`
4. 双稳态判据：`max_r > 5 Hz`（up-state 存在）且 `min_r < 2 Hz`（down-state 存在）
5. 运行两套扫描：
   - **Default**：`b=0, tauA=200`（复现论文 Fig.3g）
   - **Patient**：`b=19.5, tauA=1040`（个性化分析）
6. 计算 ALN 坐标下病人参数到双稳态边界的欧氏距离

> **API 修正**：`BoxSearch` 和 `ParameterSpace` 均依赖 `pypet`（同 s4），用手动嵌套循环替代。

### 诊断结果（坐标映射修正后）
| 配置 | 双稳态格点数 | 病人到边界距离 | 风险评级 |
|------|------------|-------------|---------|
| Default (b=0) | 21 | **0.818 mV/ms** | ✓ 低风险 |
| **Patient (b=19.5)** | **91** | **0.793 mV/ms** | **● 中等风险** |

**关键发现**：
- 适应电流使双稳态区域扩张 4.3 倍（21→91 格点），说明 SFA 显著拓宽了皮层 down-state 的参数范围
- 两个距离差值仅 0.025 mV/ms（3%），说明适应电流主要扩张双稳态面积，而非缩短红星到边界的距离
- 双稳态区域向 mue 增大方向扩张，红星处于"接近但尚未进入"的状态
- 与 Session 1-B 观察到的稀疏放电模式（偶发 up-state）物理一致

### 输出文件
| 文件 | 内容 |
|------|------|
| `outputs/bifurcation_map.png` | 2×2 子图（default/patient 热图 + 振荡幅度图） |
| `outputs/bifurcation_default.csv` | Default 扫描 441 格点数据 |
| `outputs/bifurcation_patient.csv` | Patient 扫描 441 格点数据 |
| `data/bifurcation_summary.json` | 距离、格点数等摘要（供 Session 3-A 使用） |

`data/bifurcation_summary.json` 关键字段：
```json
{
  "patient_mue": 2.287,
  "patient_mui": 3.098,
  "bistable_dist_default": 0.818,
  "bistable_dist_patient": 0.793,
  "bistable_n_default": 21,
  "bistable_n_patient": 91
}
```

耗时：~9 秒（882 次仿真，JIT 缓存后极快）

---

## 数据流向图

```
Sleep-EDF EDF 文件
    │
    ├─ s1_trusted_segments.py  ─→ [仅打印，确认数据可用]
    │
    ├─ s1_all_stages.py  ──────→ data/psd_{stage}.npy
    │                            (群体平均 PSD，供后续 RL 状态对比用)
    │
    └─ SC4001 PSG + Hypnogram ─→ s4_personalize.py  ─────────→ data/patient_params_SC4001.json
       (57 N3 epochs, QC'd)       (SC4001 个体化,                (mue=3.009, mui=4.077,
                                   归一化 PSD 拟合,               g_LK=0.128, g_h=0.025,
                                   score=0.881)                   score=0.881)
                                                                         │
HCP 连接组                                                               ├─→ s5_bifurcation.py
    │                                                                    │   (×0.76 坐标映射)
    └─ s2_skeleton.py  ────────→ data/cmat.npy                          │       └─→ outputs/bifurcation_map.png
                                 data/dmat.npy                           │           距边界 0.793 mV/ms
                                                                         │
neurolib MultiModel                                                      │
    │                                                                    │
    └─ s3_sleep_kernel.py  ────→ outputs/r_cortex.npy                   │
       (Table 3 + mue=3.2)       outputs/r_thalamus.npy                 │
                                      │                                  │
                                      └─ tests/test_spindles.py ←───────┘
                                         (PSD 验证，ALL PASSED ✓)
```

---

## 已知的 neurolib API 兼容性问题（NumPy 2.0）

以下 neurolib 模块依赖 `pypet`，后者使用了已在 NumPy 2.0 删除的 `np.string_`，**全部不可用**：

| 模块 | 预期用途 | 替代方案 |
|------|---------|---------|
| `neurolib.optimize.evolution.Evolution` | NSGA-II 进化优化 | `scipy.optimize.differential_evolution` |
| `neurolib.optimize.exploration.BoxSearch` | 参数网格扫描 | 手动双重循环 + `ALNModel` |
| `neurolib.utils.parameterSpace.ParameterSpace` | 参数空间定义 | Python dict / numpy linspace |

---

## 已知残余问题与迭代计划

### 问题 1：ThalamicNode 高频谐波（10–30 Hz）

**现象**：仿真 PSD 在 10–30 Hz 存在多个振荡峰（~12, 16, 22, 27 Hz）  
**根源**：TCR-TRN 非线性振荡回路的谐波结构，sigma 惩罚项只压制了主峰  
**影响**：不阻塞 Session 3，但会影响 PSD 拟合的视觉质量  
**迭代方案**：
- 缩小 g_h 搜索范围至 [0.01, 0.04]，在更低区间精细搜索
- 在适应度函数加入高频惩罚项（15–30 Hz 超出目标的部分扣分）

### 问题 2：适应度函数仍可进一步改进

**现象**：score=0.881 是真实的，但仍依赖 log 域形状相关，未直接比较峰值  
**根源**：真正应该比较的是 FOOOF 提取后的 delta 峰位置和高度  
**迭代方案**（MVP 后）：
```python
# 用 FOOOF 提取周期性峰值后比较
fm_sim.fit(f_s, p_s, freq_range=[0.5, 30])
fm_tgt.fit(target_freqs, target_psd, freq_range=[0.5, 30])
# 比较 delta 峰的中心频率（CF）和功率（PW）的差距
```

### 问题 3：坐标映射比例 SCALE=0.76 是经验值

**现象**：MultiModel → ALN 的映射比例基于单点校准（mue 3.20→2.30）  
**影响**：红星位置存在约 ±10% 的不确定性  
**迭代方案**：跑系统性的双模型对比实验，建立完整的参数映射曲线

### 问题 4：per-epoch 拟合尚未实现

**现象**：当前用 57 个 epoch 的平均 PSD 拟合单一参数，不知道 N3 参数的变异范围  
**迭代方案**：
1. 对 57 个 epoch 各自拟合 → 57 个参数点（云团）
2. 云团中心 = N3 代表参数，云团范围 = RL 目标区域大小
3. 对 Wake/N1/N2/N3/REM 各自建立云团 → 睡眠状态轨迹
4. 云团特征 → M_F（论文 Stage 2）的新特征维度

---

## Session 3-A 输入数据清单

| 文件 | 内容 | 用途 |
|------|------|------|
| `data/patient_params_SC4001.json` | SC4001 个性化参数 | RL 环境初始化 |
| `data/bifurcation_summary.json` | 双稳态距离 0.793 mV/ms | reward 函数设计 |
| `data/psd_wake.npy` | Wake 阶段 PSD | RL 初始状态分布 |
| `data/psd_n1.npy` | N1 阶段 PSD | RL 初始状态分布 |
| `data/psd_n2.npy` | N2 阶段 PSD | RL 初始状态分布 |
| `data/psd_n3.npy` | N3 阶段 PSD（目标） | RL 目标状态定义 |
| `data/cmat.npy` | HCP 连接组 | 后续全脑网络扩展 |
