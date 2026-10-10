> 最终状态：实验矩阵已完成，未达到超越LightGBM目标。下文保留执行计划；最终事实以[最终报告](FINAL_SUMMARY_ZH.md)为准。

# Selective FT 实验计划：匹配消融与资源协调

状态：实现及 CPU 正确性检查完成；完整 best/last 诊断及三个 CUDA smoke 已通过，M0_compressed 正在训练。旧 ft_upgrade 已按用户要求停止。

调度更新：根据协调聊天中最新直接用户指令，当前 trial 完成后切换到共同独占 GPU lease；不丢弃有效结果，不重置预算。当前仍采用旧40%显存的 trial 将自然结束，后续资源接口由协调聊天统一提供。

## 目标与证据边界

在固定 chronological validation、seed=42 上争取 AP > 0.6338591016，且 Recall@Top1% >= 0.2705456936。保留强匹配 LightGBM；不削弱对照、不用旧 test 调参、不做 ensemble。用户选择不做 fresh seeds 或 rolling folds，最终声明限定为单 seed 开发集结果。达到目标是实验目标，不是承诺。

沿用原始预算起点 2026-10-10T04:50:59.137769Z。绝对截止为 2026-10-11 00:50:59 America/New_York；最后45分钟用于核验和报告。用户已取消必须累计训练满12小时的要求，不为凑时长延长实验。等待不算神经训练，但会消耗绝对截止前的时间。

## 两组实验与资源

本组输出全部位于本目录 advanced_selective_ft_20261010。另一组已存在于相邻 advanced_ft_transformer_goal_20261010；不修改其代码、状态、结果或进程。

默认两组并行准备和分析，GPU 训练分时进行，禁止本组内部多个 CUDA 训练作业。没有双方明确采用共同锁时，不声称已实现跨组互斥。

执行前需双方采用项目级共享 GPU lease：锁文件记录 owner、PID、进程创建时间、命令和到期/更新时间；原子获取，只在核验原持有进程已退出后清理过期锁。lease 只协调资源，不共享或覆盖实验状态。每个完整 trial 后释放，若另一组已排队则让出下一轮；一个 trial 最多3600秒，预算在安全 epoch 边界执行。未知训练进程存在时，本组等待，不结束其他进程。

开始 CUDA smoke/推理/训练前检查真实进程、可用显存和内存；共同 lease 也覆盖较重诊断推理。双方未协作时本组采取保守检查，不以一个采样时刻的空闲视为未来保证。CPU 准备限制最多4个计算线程和一个重预处理作业；优先只读复用已核验缓存。可用 RAM 低于6 GiB 时不启动新重任务。禁止为两组分别保留不必要的完整数据副本。

本组新增神经训练最多6小时，且始终服从原绝对截止。优先五组主消融与配对 loss；可选实验仅在剩余时间足以保证 SOAP 和报告时执行。记录排队时间、实际训练时间、GPU allocated/reserved memory 及设备占用，不把等待算训练。

若两组完成了配置、代码语义、数据指纹和停止规则均匹配的实验，可引用其冻结产物，记录来源、哈希和差异；不将仅名称相同或未核验结果视为可复用。最终比较在本组目录生成来源明确的汇总，不复制大 checkpoint。

## 诊断与主实验

对 ft_023 的 best/last checkpoint，在完整固定 train 和 validation 上以 eval 模式计算 AP、普通 BCE、pos_weight=3 BCE、Brier 和类别平均概率。验证按六个连续时间段及 card/entity known、unknown/rare、missing 群体分解；分别标明组内 Top1% 与全局队列贡献。已有另一组诊断须核验其推理来源、样本覆盖和 checkpoint，不能把存盘预测重算冒充新的训练集推理。

五组架构：M0 compression=.25；M1 full；M2 full+SPA；M3 full+IAI；M4 full+SPA+IAI。共同 rich_all、periodic、width128、depth2、heads8、dropout.2、ReGLU、FFN4/3、AdamW lr3e-4/wd1e-6、pos_weight3、recency bootstrap4、frequency_scale.1、seed42。保持匹配初始化和采样序列。

统一 max_epochs40、min_epochs8、patience8、AP improvement1e-6、ReduceLROnPlateau factor.5/patience4/min_lr1e-6、单组3600秒。BF16、clip1、effective batch128；OOM 时 microbatch128->64->32，以梯度累积维持有效 batch，其他组匹配。先用真实数据 smoke；不静默回退CPU。

SPA 是项目适配：仅用训练行最多50k估计 mixed-feature MI，减三次固定标签置换平均 MI 后截断为非负。feature query 读取分数不低于自身的 key、自身始终可读；CLS 读取全部 feature，feature query 不读 CLS，避免旁路。missing indicator 自己评分。IAI 仅将 attention Q/K/V 初始权重乘.1、bias置零，其他初始化不变。不声称完整复现 ExcelFormer。

若主实验未达目标且预算充足，最多三项跟进：最佳BCE架构的完整覆盖+归一化时间 loss weight1->4；完整覆盖+uniform weight；若普通SPA较匹配基线AP提升>=.005且R1下降<=.005，则测试四个训练内部窗口的 stable_MI=median(MI)-.5*IQR(MI)，每窗最多25k。无随机搜索。

## 配对 fine-tuning、正确性与恢复

从已完成新候选及 ft_023 中按验证 AP 选择最佳 BCE checkpoint，复制为继续BCE和官方LibAUC2.0.1 APLoss+SOAP两组。共享官方DualSampler序列、正例比例.25、batch128、SOAP Adam lr1e-5/wd1e-6。APLoss gamma.9/margin1/squared_hinge；loss FP32。最多三个 sampler epochs、每组1800秒；SOAP AP下降超过起点.005时，两组在相同验证边界停止，保留起点候选。保存 u_all/u_pos、sampler、optimizer、RNG及稳定训练行索引。

预检覆盖官方attention与SDPA的压缩/非压缩 FP32 forward和无dropout梯度一致性、有dropout有限梯度、SPA方向及CLS旁路、checkpoint重放、AP与sklearn一致、TopK ceil及TransactionID同分排序。所有目标相关统计只用训练数据。

提供 --prepare / --run / --resume / --report。各阶段保存命令、配置、代码/缓存/checkpoint哈希、状态、失败和资源。连续三次相同基础设施错误停止相关任务并报告。原64冻结文件不得变化。最终优先选满足R1约束的最高AP候选，并公开不受约束的最高AP与全部负结果。只有锁定选型后才可一次探索性复评已经查看过的旧test，不称独立测试、不据此再调参。

## 完整查看入口与交付

实现后本目录 README.md 是总入口：状态、最佳结果、基线差距、运行命令和全部产物链接。protocol.json 固定规则；campaign_state.json 保存当前阶段和资源等待状态；runs/ 每个实验独立保存；results/ 保存总表、诊断表和图；report.md 给中文解释与实际结论。原始数据和大 checkpoint 不上传GitHub。

每组结果标明完成/失败/未执行、配置、AP/ROC/R1/P1/R5/Brier、best epoch、训练/推理时间、参数量、显存与来源。诊断和假设不能标为确认结果。结束前复算指标、核验ID/标签、查看图、验证原始64文件指纹；更新已有Draft PR #3和真实英文面试介绍。若截止或方法无效，仍生成完整负结果报告，不继续追分或美化作品集。
