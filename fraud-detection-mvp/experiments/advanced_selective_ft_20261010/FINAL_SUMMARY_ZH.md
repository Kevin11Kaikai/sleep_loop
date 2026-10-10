# Selective FT：最终结果与技术解释

**完整实验完成，但未击败强 LightGBM。** 这是一组 seed=42 的 chronological development experiments，所有负结果公开保留。

## 如何从交易到调查队列

IEEE-CIS 的 train_transaction.csv 按 TransactionDT 排序，前70%训练、中15%验证、后15%为已在早期查看的旧test。训练413,378条、验证88,581条；rich_all表示包含407个数值字段和22个类别字段。全部词表、分位数、频率、缺失与MI统计只在训练段拟合。数值periodic tokens与类别embeddings经FT attention产生风险分数，再排序调查前1%；ID仅用于对齐与并列排序。

## 实测比较

| 模型 | 验证AP | 验证ROC-AUC | Recall@1% | Precision@1% | Recall@5% | Brier |
|---|---:|---:|---:|---:|---:|---:|
| 强LightGBM | .633859 | .925485 | 27.0546% | 92.8894% | 66.5023% | .019143 |
| 原FT ft_023 | .578391 | .889786 | 26.1012% | 89.6163% | 63.4780% | .025095 |
| 最佳FT：继续BCE | .581126 | .881181 | 26.5943% | 91.3093% | 63.6752% | .046224 |

调查886条时，继续BCE捕获809个欺诈，原FT为794，LightGBM为823。AP增加.002735、仅缩小原AP差距4.93%，没有达到AP>.633859且Recall@1%≥27.0546%的双目标。ROC-AUC与Brier也未改善：ranking的小收益不能被描述为整体优势。

## 每项假设得到了什么证据

| 干预 | 验证AP | 对照与解释 |
|---|---:|---|
| M0 压缩attention | .562347 | 新匹配初始化与停止规则下的基准 |
| M1 full attention | .546023 | 相对M0下降.016324，未支持压缩是主要瓶颈 |
| M2 full＋SPA | .525791 | 相对M1下降.020232，未支持本次MI硬mask |
| M3 full＋IAI | .532915 | 相对M1下降.013109，QKV保守初始化未改善 |
| M4 SPA＋IAI | .522027 | 组合未改善；不等同于完整ExcelFormer |
| 完整覆盖＋时间loss weight | .563174 | 增益很小，R1下降 |
| 完整覆盖＋uniform weight | .564197 | 仍低于原FT；未解决差距 |
| C0 继续BCE | .581126 | 从原FT最优checkpoint额外训练一个sampler epoch |
| C1 官方SOAP更新后 | .559188 | 与C0同序列、4154步、531712次采样；触发AP下降>.005的停止规则 |

SOAP最终保留的是epoch0起始checkpoint（AP=.578391），不是SOAP学出了这个分数。原始更新后.559188与继续BCE .581126相差−.021938；两臂相同更新次数、sampler序列和优化器设置，实际耗时分别157.5秒与153.3秒。这里的相同预算是相同训练更新边界和时间上限，并非逐秒相同。stable-MI跟进因SPA未达到预设增益门槛而未执行，不算失败或缺失的必需实验。

M0–M4使用共同初始化和相同显式抽样序列；原ft_023使用历史初始化及停止规则，不能把它与M0的差异单独归因于某个结构。M0最初与另一组同时训练（CPU2、40%allocator上限），后续在trial边界迁移为FIFO独占（CPU4、80%上限）；因此运行时间不宜直接作为架构吞吐消融结论。预算未重置。

## 欠拟合还是过拟合

原ft_023固定完整train/validation、eval模式fresh CUDA推理：best训练AP=.890699、验证AP=.578391；last训练AP=.990067、验证AP=.522443。训练普通BCE从.047054降至.008646，验证BCE从.111723升至.185418。**存在明显late overfitting；最优checkpoint已能很好拟合训练数据，不能简单诊断为整体容量不足。** 这并不证明全部LightGBM差距由过拟合造成，也不排除特定稀有模式、表示或时间外泛化不足。

验证时间段及known/unknown群体结果见[分层表](results/validation_groups.csv)。unknown/rare/missing在本表示中合并编码为0，不能据此单独声称未见实体或缺失字段的因果贡献。分层只用于诊断，没有根据旧test重新选型。

## 锁定后的旧test探索性复评

| 模型 | AP | ROC-AUC | Recall@1% | Precision@1% | Recall@5% | Brier |
|---|---:|---:|---:|---:|---:|---:|
| 继续BCE FT | .451846 | .824756 | 23.1917% | 80.6998% | 54.4924% | .053275 |
| 匹配LightGBM | .530784 | .891270 | 25.1379% | 87.4718% | 58.0603% | .022836 |

这一窗口已在早期MVP被查看，**不是新的独立测试**。本次在验证选择锁定之后只评价一次，不再据此调参。FT捕获715个欺诈，LightGBM捕获775个，进一步没有支持Transformer胜出。

## 实际成本、复现与限制

新增神经训练（含epoch内validation）8479.95秒，约2小时21分20秒；从本组CUDA诊断启动到完成约4小时14分钟，包含GPU排队、诊断、复核和报告。RTX4080 SUPER 16GB，Python3.12.7、PyTorch2.5.1+cu121、BF16，官方LibAUC2.0.1；没有CPU Transformer fallback。GPU阶段资源见[资源表](results/resource_usage.csv)。C0/C1保存result时重置过峰值计数，表内这两行反映finalization阶段，不能当作整个微调训练的peak；单轮训练记录曾显示SOAP reserved=2262MiB。因此资源表不是完整device-level峰值审计。

所有保存预测ID/标签和指标已自行复算，checkpoint和原MVP64文件哈希通过；这不是独立验证。公开文件不含raw数据、row-level预测、checkpoint、凭据、进程日志或聊天内容。历史本地缓存、原MVP artifact manifest和ft_023权重是运行先决条件，公开包不是可脱离这些资产直接恢复的checkpoint包。原绝对预算已固定；未来复跑必须新建隔离实验协议并声明新的日期预算，不能偷偷重置本轮历史预算。

不做fresh seeds/rolling folds是本轮用户选择，结论限定为单seed开发比较。AP经过反复模型选择；匿名V/D字段线上可得性未验证；weighted BCE与balanced sampling会改变概率校准，未经独立calibration。不声称production readiness、金融损失减少或JPMorgan认证。

## 求职作品表达与唯一下一步

这组结果可用于技术讨论：展示强基线、训练拟合诊断、受控negative ablations与固定调查预算，而不能写成“Transformer超过GBDT”。最有价值的下一步是冻结方法后在新的未用于开发的时间数据上检验泛化，尤其解释FT为何对时间漂移更脆弱；本轮不为追分增加随机搜索。

English introduction: I built a chronological IEEE-CIS fraud-ranking benchmark with training-only preprocessing and a strong matched LightGBM. Controlled attention and initialization ablations did not close the neural gap. A matched BCE continuation improved development AP from 0.5784 to 0.5811 and captured 15 more fraud cases at a 1% investigation budget, while LightGBM remained stronger. Fresh best/last inference exposed substantial late overfitting, and previously viewed test evaluation was explicitly exploratory.

Resume bullet: Built and audited a temporal fraud-detection benchmark on 590K+ IEEE-CIS transactions, comparing strong LightGBM with FT-Transformer using Average Precision and fixed-budget investigation metrics; preserved all negative results and training-only preprocessing provenance.

Resume bullet: Diagnosed neural overfitting through complete train/validation checkpoint inference and controlled full-attention, selective-mask, initialization, and official AP-loss ablations; improved development AP by .0027 through matched BCE continuation without claiming independent-test superiority.
