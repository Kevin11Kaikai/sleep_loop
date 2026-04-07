# `sleep_loop`

丘脑-皮层睡眠建模与个体化拟合项目。该仓库围绕 **Sleep-EDF** 睡眠脑电、**neurolib** 神经质量模型、**Welch + FOOOF** 频谱分析，以及 **差分进化** / 强化学习等方法，探索 **N3 慢波睡眠** 的频谱与动力学特征。

当前最完整、最建议阅读的主线是：

- `models/s4_personalize_fig7_v3.py`
- `tests/test_compute_fitness_v3.py`
- `docs/s4_personalize_fig7_v3_overview.md`

---

## 项目在做什么

以受试者 `SC4001` 为例，项目会：

1. 从真实睡眠 EEG 中提取 N3 阶段，计算目标功率谱 `target_psd`
2. 构建丘脑-皮层平均场模型，模拟皮层/丘脑发放率
3. 用频谱形状、慢波/纺锤峰，以及时域动力学指标定义 `fitness`
4. 用差分进化搜索 8 个关键参数，使模拟结果更接近真实 N3 EEG

简化理解：这是一个“**用真实睡眠脑电来反推模型参数**”的研究型代码仓库。

---

## 主要内容

- `models/`
  - 各阶段主脚本与实验脚本
  - `s4_personalize_fig7_v3.py` 是当前 v3 个体化主线
  - 还保留了 `v1/v2`、bifurcation、RL 等探索代码
- `tests/`
  - 面向脚本的功能检查
  - `test_compute_fitness_v3.py` 可单次运行 `compute_fitness_v3`，并可继续出图
- `plot_scripts/`
  - 画 Fig.7 风格图、残差图等
- `docs/`
  - 过程文档、分函数说明、阶段总结
- `data/`
  - 参数 JSON、manifest、阶段中间数据
- `outputs/`
  - 进化记录、日志、PNG 图、缓存结果

---

## 推荐阅读顺序

1. `docs/s4_personalize_fig7_v3_overview.md`
2. `models/s4_personalize_fig7_v3.py`
3. `docs/load_target_psd_notes.md`
4. `docs/compute_target_periodic_notes.md`
5. `docs/compute_fitness_v3_notes.md`
6. `docs/s4_personalize_fig7_v3_compute_fitness_v3.md`

---

## 核心脚本

### 1. 个体化主流程

`models/s4_personalize_fig7_v3.py`

功能：

- 读取真实 N3 EEG
- 构造目标 PSD 与 FOOOF 残差
- 定义 `compute_fitness_v3`
- 运行差分进化
- 输出最优参数 JSON 与完整 CSV 记录

运行：

```bash
python models/s4_personalize_fig7_v3.py
```

主要输出：

- `data/patient_params_fig7_v3_SC4001.json`
- `outputs/evolution_fig7_v3_records.csv`

### 2. 单次适应度检查与出图

`tests/test_compute_fitness_v3.py`

功能：

- 只跑一次 `compute_fitness_v3`
- 打印 `shape_r`、`so_power`、`spindle_power`、`dynamics_score`
- 检查 T1-T5 动力学子测试
- 可额外生成 Fig.7 风格图片

运行：

```bash
python tests/test_compute_fitness_v3.py
python tests/test_compute_fitness_v3.py --plots
python tests/test_compute_fitness_v3.py --plots --out-v2-names
```

默认绘图输出：

- `outputs/fig7_v3_test_timeseries.png`
- `outputs/fig7_v3_test_spectra.png`
- `outputs/fig7_v3_test_residuals.png`

---

## 数据要求

仓库本身依赖外部睡眠数据文件。默认流程要求：

- `data/manifest.csv` 存在
- `manifest.csv` 中能找到目标受试者（默认 `SC4001`）
- 其中记录的 `psg_path` / `hypnogram_path` 指向本机可访问的 Sleep-EDF 文件

也就是说，**仅上传代码到 GitHub 后，别人通常还不能直接运行主流程**，还需要：

1. 自己准备 Sleep-EDF 数据
2. 修改或重建 `data/manifest.csv`
3. 保证路径与本地环境一致

如果准备公开仓库，建议在上传前再次确认：

- `manifest.csv` 是否包含本机绝对路径
- `outputs/` 中是否有不需要上传的大文件、缓存或日志
- `data/` 中是否有仅限本地使用的中间文件

---

## 依赖

从代码可见，主流程依赖这些 Python 包：

- `numpy`
- `pandas`
- `scipy`
- `mne`
- `matplotlib`
- `neurolib`
- `fooof`（推荐；未安装时部分路径会退化为 fallback）

如果你准备自己补环境，建议优先根据这些脚本检查：

- `models/s4_personalize_fig7_v3.py`
- `tests/test_compute_fitness_v3.py`
- `plot_scripts/plot_fig7_v2_fast.py`

当前仓库里还没有统一整理好的 `requirements.txt` / `environment.yml`。

---

## 版本线索

仓库里保留了多个阶段脚本，大致可按名字理解：

- `s1_*`：早期数据整理 / 全睡眠阶段分析
- `s3_*`：睡眠核与谱相关实验
- `s4_*`：Fig.7 风格个体化拟合主线
- `s5_*`：bifurcation
- `s6_*`：baseline / RL 环境
- `s7_*`, `s8_*`：SAC 训练与迭代

如果只是想看“目前主结果怎么来的”，优先关注 `s4` 系列即可。

---

## 与 v2 / v3 的关系

`s4_personalize_fig7_v3.py` 是在 v2 基础上的改进版，重点变化包括：

- 动力学评分从较松的规则改为更严格的 `T1-T5`
- 显式要求皮层存在真实 UP state，避免“持续 DOWN”也拿高分
- 输出文件统一使用 `_v3` 命名

相关说明见：

- `docs/s4_personalize_fig7_v3_overview.md`
- `docs/compute_fitness_v3_notes.md`

---

## 已有图与可视化

仓库里已有多种图像输出与脚本，例如：

- `plot_scripts/plot_fig7_v2_fast.py`
- `plot_scripts/plot_fig7_residuals.py`
- `tests/test_compute_fitness_v3.py --plots`

常见图包括：

- 皮层/丘脑时间序列
- EEG vs simulation 功率谱
- 去掉 1/f 背景后的 FOOOF residual 对比

---

## 说明

这是一个**研究型、迭代中的代码仓库**，特点是：

- 有较多实验脚本和历史版本
- 文档是逐步补充的
- 某些路径和参数目前仍默认写死为 `SC4001`

如果你准备继续整理并公开到 GitHub，下一步通常值得做的是：

1. 补一个 `requirements.txt` 或 `environment.yml`
2. 说明 `manifest.csv` 的格式
3. 在 `.gitignore` 里过滤不需要上传的 `outputs/` 大文件和日志
4. 选定一个最小可复现实验作为仓库首页示例

---

## 相关文档

- `docs/s4_personalize_fig7_v3_overview.md`
- `docs/load_target_psd_notes.md`
- `docs/compute_target_periodic_notes.md`
- `docs/compute_fitness_v3_notes.md`
- `docs/s4_personalize_fig7_v3_compute_fitness_v3.md`
- `docs/0315_Progress.md`
- `docs/0404_Progress.md`

---

## License

仓库当前未见明确许可证文件。若准备公开到 GitHub，建议补充 `LICENSE`。
