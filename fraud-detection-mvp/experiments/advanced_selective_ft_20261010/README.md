# Selective FT — IEEE-CIS

状态：COMPLETED。本目录独立保存全部代码与结果，不修改另一组。

- [最终中文结论、实验归因和旧test](FINAL_SUMMARY_ZH.md)
- [完整计划](PLAN_ZH.md)
- [实际结果和解释](report.md)
- [实验总表](results/experiment_comparison.csv)
- [拟合诊断](results/train_vs_validation_diagnostics.csv)
- [时间/OOV群体](results/validation_groups.csv)
- [资源用量](results/resource_usage.csv)
- [当前状态](campaign_state.json)
- [固定协议](protocol.json)

从 fraud-detection-mvp 运行：
```powershell
.\.venv\Scripts\python.exe -m experiments.advanced_selective_ft_20261010.campaign --prepare
.\.venv\Scripts\python.exe -m experiments.advanced_selective_ft_20261010.campaign --run
.\.venv\Scripts\python.exe -m experiments.advanced_selective_ft_20261010.campaign --resume
.\.venv\Scripts\python.exe -m experiments.advanced_selective_ft_20261010.campaign --report
```

CUDA必需；依赖复用项目环境，额外包固定libauc==2.0.1（--no-deps）。源数据、row-level predictions和checkpoint仅本地保存。

公开包的运行先决条件与固定历史预算限制见最终报告；不是独立checkpoint恢复包。完成时间2026-10-10 16:17 America/New_York。
