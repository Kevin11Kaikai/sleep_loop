# Selective FT: 当前实验报告

状态：COMPLETED；更新：2026-10-10T20:17:38.320169+00:00。所有主结果是固定验证集、seed=42 的开发结果。

                    model  average_precision  recall_top_1pct  training_seconds
                 LightGBM           0.633859         0.270546        172.366544
              Original FT           0.578391         0.261012       1519.004750
          C0_continue_bce           0.581126         0.265943        153.278660
         C1_official_soap           0.578391         0.261012        157.511118
E1_complete_temporal_loss           0.563174         0.255424       1050.578212
      E2_complete_uniform           0.564197         0.252465       1050.168550
            M0_compressed           0.562347         0.260355       1858.658611
                  M1_full           0.546023         0.253780       1300.190091
              M2_full_spa           0.525791         0.248521       1037.350135
              M3_full_iai           0.532915         0.248849       1129.548135
          M4_full_spa_iai           0.522027         0.256739        735.373720

## 解释与证据边界
原test已查看；本轮训练/选型不读取test标签。不做fresh seeds或rolling folds，不能声称跨seed稳定或独立测试胜出。
诊断是观察；压缩、SPA、IAI的匹配差异用于检验假设；SOAP必须与同预算继续BCE比较。负结果与未执行项目保留。

不受约束最高AP：C0_continue_bce；AP=0.581126。无候选达到R1约束，回退选择最高AP：C0_continue_bce，AP=0.581126、R1=0.265943；双目标达到：False。
最高AP候选相对原FT改变+0.002735；原AP差距缩小比例4.93%。该比例不是独立泛化证明。

## 受控实验逐项回答
- 去掉压缩是否有帮助：实测AP差-0.016324，R1差-0.006575；这是单seed开发观察，不是机制因果证明。
- SPA是否有帮助：实测AP差-0.020232，R1差-0.005260；这是单seed开发观察，不是机制因果证明。
- IAI是否有帮助：实测AP差-0.013109，R1差-0.004931；这是单seed开发观察，不是机制因果证明。
- SPA之上加入IAI是否有帮助：实测AP差-0.003764，R1差+0.008218；这是单seed开发观察，不是机制因果证明。
- SOAP是否优于继续BCE：实测AP差-0.002735，R1差-0.004931；这是单seed开发观察，不是机制因果证明。

## 拟合与后期退化观察
     model      split  average_precision      bce  weighted_bce
ft023_best      train           0.890699 0.047054      0.076202
ft023_best validation           0.578391 0.111723      0.234236
ft023_last      train           0.990067 0.008646      0.016955
ft023_last validation           0.522443 0.185418      0.430150
训练与验证差距、时间/OOV分层提示可能机制，但不能证明attention是唯一瓶颈。

## 局限与下一步
无fresh-seed确认；开发集已被多次选择；原test已查看；训练统计不保证匿名V/D字段在线可得；weighted BCE的概率可能失准；离线指标不等于减少损失，也不代表production readiness。
若出现可信增益，最有价值的下一步是冻结方法后用新的独立时间数据验证；本轮不额外增加这项验收门槛。

English: Controlled, seed-42 development benchmark on IEEE-CIS with a preserved strong LightGBM control. Current status: COMPLETED. Results above are measured; no independent-test superiority or production readiness is claimed.

最终解释、旧test与资源局限见[最终中文报告](FINAL_SUMMARY_ZH.md)。SOAP更新后的AP=.559188；结果表中.578391来自保留epoch0起始checkpoint。
