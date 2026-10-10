# IEEE-CIS Fraud Detection：LightGBM 与 FT-Transformer

**当前是进行中的实验快照，2026-10-10T08:10:44-04:00。** 长训练累计 7.05 小时，尚未完成。单 FT 验证 AP=0.5784，强匹配 LightGBM AP=0.6339。

先阅读 **[中文进度与拟合分析](PROGRESS_ZH.md)**：模型如何输出调查队列、实际指标、过拟合曲线、时间漂移、局限及下一步。所有本轮对比使用同一验证集。

## 本项目范围

交易表 `isFraud` 二分类；70/15/15 chronological split；训练数据拟合全部预处理；无目标编码、`TransactionID` 预测或官方无标签 test 本地评价。复用官方 RTDL FT backbone、官方 PLE/periodic numeric embeddings，保留强匹配树对照。单 FT 与神经 ensemble 分开报告；目前页面是单 FT。

原 MVP 已完成，历史测试汇总单独位于 [results/original_mvp/metrics.csv](results/original_mvp/metrics.csv)。旧测试已被查看，本轮搜索只看训练/验证；最终锁定后的旧测试复评属于探索性证据。PR-AUC 采用 Average Precision；Top-K 固定调查预算；Brier 衡量概率质量，但类别权重可能影响校准。离线指标不代表实际金融损失降低，也不代表生产就绪。

## 本地复现

在本目录创建 Python 3.12 环境，安装 CUDA PyTorch 后安装依赖；从 Kaggle 按规则下载 `train_transaction.csv` 到 `data/raw/`。`train_identity.csv` 未使用。数据、模型、凭据和逐行衍生物必须留在本地。

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m src.check_environment
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
& .\.venv\Scripts\python.exe -u -m src.run --stage all
```

原 MVP 入口先跑真实数据 smoke，再跑固定小规模全量比较。长 campaign 使用 `experiments/ft_upgrade_20261010` 内的代码与固定协议；它会核验已有本地 MVP 冻结资产并生成缓存。**公开包省略私有冻结资产，不可将其冒充完整 checkpoint 恢复包。** 在新的机器上需先重建本地 MVP 资产，并按照原代码的 fingerprint 检查建立自己的新实验目录/manifest；不要篡改原有资产以绕过保护。当前训练机上已验证的继续命令如下，进程已运行时不要重复启动：

```powershell
& .\.venv\Scripts\python.exe -m pip install -r experiments\ft_upgrade_20261010\requirements-extra.txt
& .\.venv\Scripts\python.exe -u -m experiments.ft_upgrade_20261010.smoke
& .\.venv\Scripts\python.exe -u -m experiments.ft_upgrade_20261010.campaign
```

## 快照证据

[状态与指纹](experiments/ft_upgrade_20261010/progress/20261010T121044Z/status.json) · [验证指标](experiments/ft_upgrade_20261010/progress/20261010T121044Z/validation_metrics.csv) · [试验汇总](experiments/ft_upgrade_20261010/progress/20261010T121044Z/neural_trials.csv) · [训练曲线记录](experiments/ft_upgrade_20261010/progress/20261010T121044Z/training_history.csv) · [固定协议](experiments/ft_upgrade_20261010/protocol.json) · [参考文献](experiments/ft_upgrade_20261010/research_notes.md)

这是用户授权上传的公开阅读快照。没有上传 Kaggle 数据、预测明细、权重、凭据或本地进程日志；未修改其他 sleep_loop 项目文件，也没有停止 GPU 作业。快照非实时仪表盘，后续结果需要新的明确上传动作。
