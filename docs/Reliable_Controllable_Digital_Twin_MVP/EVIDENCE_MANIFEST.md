# Evidence Manifest

## Manifest policy

Every major dissertation claim below is tied to one fixed M1 or C1 artifact. “Copied” means a lightweight byte-for-byte copy is included in this package; “Referenced” means the authoritative local artifact remains in place and is identified by absolute path and SHA-256. No scientific result was recomputed.

## Claim-to-artifact traceability

| Claim | Authoritative artifact and absolute source path | Key result / number | Evidence role | Archive status | SHA-256 |
|---|---|---|---|---|---|
| M1 closed with C006 selected | `FINAL_M1_JUDGMENT.json` - `D:\Year3_Mao_Projects\sleep_loop\BigBang_V4D1\COSTA_MVP_FRESH_R4_20260816_023933\TOURNAMENT\FINAL_M1_JUDGMENT.json` | C006; G1-G7 PASS; no material disagreement | Final M1 decision | Copied as `M1_RELIABLE/M1_FINAL_DECISION.json` | `A46C6CB46449105762CA4F8489A416B36FB0D4DE184D02249E9EC03835FBBF48` |
| C006 won the frozen FIT comparison | `SELECTION_FREEZE.json` - `D:\Year3_Mao_Projects\sleep_loop\BigBang_V4D1\COSTA_MVP_FRESH_R4_20260816_023933\RUNS\FIT_WORKER\SELECTION_FREEZE.json` | 16 candidates; C006 FIT NRMSE 0.2546119; 110 FIT epochs | Candidate comparison, calibration, selection freeze | Copied as `M1_RELIABLE/M1_SELECTION_FREEZE.json` | `3D8C3D035F074CED2D608132864996D4AF9920C52889A11FE7C8E7DDA5DD8BBF` |
| C006 mechanistic identity | `protocol_v2.json` - `D:\Year3_Mao_Projects\sleep_loop\BigBang_V4D1\COSTA_MVP_FRESH_R4_20260816_023933\CODE\FROZEN_V2\protocol_v2.json` | gain 0.82; `g_CT` 0.95; `g_TC` 1.10; `w_TR` 1.25 | Frozen candidate bank and model protocol | Copied as `M1_RELIABLE/M1_FROZEN_C006_PROTOCOL.json` | `338366593A2A4C8626349D70AA38306C5931F0FF4D22FD96DB546DD41505E3C3` |
| C006 was frozen before held-out access | `CANDIDATE_FREEZE_CHECKPOINT.json` - `D:\Year3_Mao_Projects\sleep_loop\BigBang_V4D1\COSTA_MVP_FRESH_R4_20260816_023933\STATE\CANDIDATE_FREEZE_CHECKPOINT.json` | C006; selection SHA `3D8C...8BBF`; heldout open count 0 at checkpoint | Freeze provenance | Referenced | `C0BEB4CBC618252C75438D91B958DF0FF863BC06865D2A48F10D4FBA5F378BAF` |
| Held-out evidence supports bounded reliability | `HELDOUT_G6_RESULT.json` - `D:\Year3_Mao_Projects\sleep_loop\BigBang_V4D1\COSTA_MVP_FRESH_R4_20260816_023933\RUNS\HELDOUT_WORKER\HELDOUT_G6_RESULT.json` | 110 HELDOUT epochs; C006 0.2478453 vs C000 0.4637754; 46.5592% improvement; CI [0.0798002, 0.1014646] | Held-out credibility gate | Copied as `M1_RELIABLE/M1_HELDOUT_RESULT.json` | `E0AD4B1855E6BD570D068D4AC3B962397BAAEF5F21AC917E8C35E238B7004E95` |
| M1 claim boundary | `FINAL_REPORT.md` - `D:\Year3_Mao_Projects\sleep_loop\BigBang_V4D1\COSTA_MVP_FRESH_R4_20260816_023933\HUMAN_REVIEW\FINAL_REPORT.md` | One subject, Night-1, four-band, dimensionless proxy; not population/clinical/causal | M1 human-readable interpretation | Referenced | `093E0644DA54445D65F0332A4147352FD35B791E24C53396840F46F1EE0C5E22` |
| C1 target, direction, seeds, durations, and gates were frozen prospectively | `C1_PRIMARY_TARGET_V1_T2.json` - `D:\Year3_Mao_Projects\sleep_loop\BigBang_Phase2\COSTA_QSOLVER_C1_HIERARCHICAL_TARGET_20260820_204034\PROTOCOL\C1_PRIMARY_TARGET_V1_T2.json` | T2; `dT2 > 0`; seeds 101003/101019/101033; 220 s/910 s | Prospective primary-target protocol | Copied as `C1_CONTROLLABLE/C1_PRIMARY_TARGET_PROTOCOL.json` | `34729FE59A4B0E0AD811EE013F3A378A568F798913ADF9C582CD9E49712F3491` |
| F01 was frozen, causal, and bounded | `FROZEN_CONTROLLER.json` - `D:\Year3_Mao_Projects\sleep_loop\BigBang_Phase2\COSTA_QSOLVER_C1_HIERARCHICAL_TARGET_20260820_204034\STATE\FROZEN_CONTROLLER.json` | amplitude 0.045; width 0.10 s; refractory 1.0 s; legal range [0, 0.05] | Frozen controller definition | Copied as `C1_CONTROLLABLE/FROZEN_F01_CONTROLLER.json` | `64B5C154BB02681650E5B730C5C9FD6F05E2D79EC911512E5400FF317AA71A79` |
| 220 s primary response replicated | `C1_PRIMARY_TARGET_AGGREGATE.json` - `D:\Year3_Mao_Projects\sleep_loop\BigBang_Phase2\COSTA_QSOLVER_C1_HIERARCHICAL_TARGET_20260820_204034\RUNS\C1_PRIMARY_TARGET_AGGREGATE.json` | median dT2 +0.1913%; positive 3/3; I_SO supportive 3/3; integrity 3/3 | Intermediate confirmation aggregate | Copied as `C1_CONTROLLABLE/C1_PRIMARY_TARGET_AGGREGATE.json` | `21C817621A0D63B1B6F134E2A1A93156F1279CE27650F520ED317B31D7E5CFA8` |
| 910 s primary response survived | same aggregate path as above | median dT2 +0.1784%; positive 3/3; I_SO supportive 3/3; integrity 3/3 | Long-duration confirmation aggregate | Copied | `21C817621A0D63B1B6F134E2A1A93156F1279CE27650F520ED317B31D7E5CFA8` |
| Frozen C006 remained intact in C1 | six per-seed result JSON files listed below | M1 protocol/model/selection hashes identical in 6/6 | Cross-stage provenance continuity | Referenced | individual hashes below |
| F01 integrity passed in every run | six per-seed results plus aggregate | finite, causal, legal bounds, duty < 0.10, safety/integrity PASS 6/6 | Controller integrity | Referenced / summarized in copied aggregate | individual hashes below |
| Secondary trade-offs remain | aggregate and `C1_PRIMARY_TARGET_METRICS.csv` - `D:\Year3_Mao_Projects\sleep_loop\BigBang_Phase2\COSTA_QSOLVER_C1_HIERARCHICAL_TARGET_20260820_204034\RUNS\C1_PRIMARY_TARGET_METRICS.csv` | I_SP 0/3 positive at both durations; T6 1/3 positive with negative medians | Mandatory limitation evidence | Metrics copied as `C1_CONTROLLABLE/C1_PRIMARY_TARGET_METRICS.csv` | `2B19A5B823AE4D84ACC088B5546B30DC7DE41F90A74DACC6ABD14EBFF417098E` |
| Terminal combined claim | `FINAL_C1_DECISION.json` - `D:\Year3_Mao_Projects\sleep_loop\BigBang_Phase2\COSTA_QSOLVER_C1_HIERARCHICAL_TARGET_20260820_204034\DECISIONS\FINAL_C1_DECISION.json` | `C1_CONTROLLABILITY_MVP_PASS_T2_PRIMARY`; `RELIABLE_AND_CONTROLLABLE_DIGITAL_TWIN_MVP` | Final C1/combined decision | Copied as `C1_CONTROLLABLE/C1_FINAL_DECISION.json` | `48E8E66C59C47DEAD9FD6C2FB0970F6887D7D192E29531E3DDB656F1D8756220` |
| Six result files are checkpoint-consistent | `LATEST_CHECKPOINT.json` - `D:\Year3_Mao_Projects\sleep_loop\BigBang_Phase2\COSTA_QSOLVER_C1_HIERARCHICAL_TARGET_20260820_204034\CHECKPOINTS\LATEST_CHECKPOINT.json` | all six SHA-256 entries match actual files | Terminal integrity checkpoint | Referenced | `49B96098CFC6CA4E7F76349D3F07829C997210750E932C498E93499F57DF8DEA` |
| Human-readable C1 interpretation | `C1_HIERARCHICAL_PRIMARY_TARGET_FINAL_SUMMARY.md` - `D:\Year3_Mao_Projects\sleep_loop\BigBang_Phase2\COSTA_QSOLVER_C1_HIERARCHICAL_TARGET_20260820_204034\HUMAN_REVIEW\C1_HIERARCHICAL_PRIMARY_TARGET_FINAL_SUMMARY.md` | Primary controllability with secondary trade-offs | C1 human review | Referenced | `60A92EB175F4FBF95A5344F8C5B5986FF790FB8E683AF1430D25BA35682AE787` |
| C1 quicklook is authoritative | `C1_HIERARCHICAL_PRIMARY_TARGET_QUICKLOOK.png` - `D:\Year3_Mao_Projects\sleep_loop\BigBang_Phase2\COSTA_QSOLVER_C1_HIERARCHICAL_TARGET_20260820_204034\FIGURES\C1_HIERARCHICAL_PRIMARY_TARGET_QUICKLOOK.png` | Seedwise T2, I_SO, and direction-count heatmap | Final C1 figure | Copied byte-for-byte | `A54D369EA585D6E103116935BC8BCDCA112B8E8947A08D7858EB6CE52F60179B` |

## Large local-only per-seed artifacts

These files are not copied because they contain detailed trigger-level arrays and are redundant with the lightweight aggregate/metrics for dissertation review. Their absolute paths and hashes preserve traceability.

| File | Absolute local source path | Bytes | SHA-256 | Why not archived |
|---|---|---:|---|---|
| `V1_S101003_220S_F01/result.json` | `D:\Year3_Mao_Projects\sleep_loop\BigBang_Phase2\COSTA_QSOLVER_C1_HIERARCHICAL_TARGET_20260820_204034\RUNS\V1_S101003_220S_F01\result.json` | 38,336 | `A95034FFF3BE29CABAB431CE738BA2F2E038D797A571394FE8F9D978547541F3` | Detailed trigger payload; aggregate and metrics copied |
| `V1_S101019_220S_F01/result.json` | `D:\Year3_Mao_Projects\sleep_loop\BigBang_Phase2\COSTA_QSOLVER_C1_HIERARCHICAL_TARGET_20260820_204034\RUNS\V1_S101019_220S_F01\result.json` | 33,290 | `C1BD7DDF8355892CDAC731E3B7D16806076D482C8887ED1CAB248734F041B241` | Detailed trigger payload; aggregate and metrics copied |
| `V1_S101033_220S_F01/result.json` | `D:\Year3_Mao_Projects\sleep_loop\BigBang_Phase2\COSTA_QSOLVER_C1_HIERARCHICAL_TARGET_20260820_204034\RUNS\V1_S101033_220S_F01\result.json` | 22,736 | `CA6C1A15CBC8AFA20EAE5B36383EA04F26363C2C9DA0420EF061914AD0B85B2A` | Detailed trigger payload; aggregate and metrics copied |
| `V1_S101003_910S_F01/result.json` | `D:\Year3_Mao_Projects\sleep_loop\BigBang_Phase2\COSTA_QSOLVER_C1_HIERARCHICAL_TARGET_20260820_204034\RUNS\V1_S101003_910S_F01\result.json` | 147,178 | `8428B976C1142284D44F1D1626D2732073D046E1AFE5656BCC07DC9EFED70AFF` | Detailed trigger payload; aggregate and metrics copied |
| `V1_S101019_910S_F01/result.json` | `D:\Year3_Mao_Projects\sleep_loop\BigBang_Phase2\COSTA_QSOLVER_C1_HIERARCHICAL_TARGET_20260820_204034\RUNS\V1_S101019_910S_F01\result.json` | 129,258 | `DD6AED65D5A76DB2B5B005E6F818D23B57F7CC9900593CF2A8D6751817D94D9C` | Detailed trigger payload; aggregate and metrics copied |
| `V1_S101033_910S_F01/result.json` | `D:\Year3_Mao_Projects\sleep_loop\BigBang_Phase2\COSTA_QSOLVER_C1_HIERARCHICAL_TARGET_20260820_204034\RUNS\V1_S101033_910S_F01\result.json` | 76,528 | `F9287005AEE311CC5E2AE876781D45318E646CB7314C65BEA6BE8A68F33D10E4` | Detailed trigger payload; aggregate and metrics copied |

## Additional lightweight authority

- `D:\Year3_Mao_Projects\sleep_loop\BigBang_Phase2\COSTA_QSOLVER_C1_HIERARCHICAL_TARGET_20260820_204034\RUNS\EXPERIMENT_LEDGER.csv` - SHA-256 `85544A9C5B9AE604AD68E56967232219E0A466F53548533F0E5BFB1EE33AAD16`; referenced rather than copied because the aggregate and metrics already preserve the final evidence table.

## Synthesis artifacts

- `M1_RELIABLE/figures/M1_WHY_C006_SYNTHESIS_FROM_EXISTING_M1_EVIDENCE.png` is generated solely from the M1 selection-freeze and held-out JSON values, with the label `SYNTHESIS_FROM_EXISTING_M1_EVIDENCE`. It is a communication artifact, not an additional scientific result.
- `slides/RELIABLE_CONTROLLABLE_DIGITAL_TWIN_MVP.pptx` and its PDF are synthesis artifacts whose visible claims are mapped to the sources above. Each slide's detailed source block is recorded in `slides/SPEAKER_NOTES.md`.

## Deliverable integrity

| Deliverable | Bytes | SHA-256 |
|---|---:|---|
| `M1_RELIABLE/figures/M1_WHY_C006_SYNTHESIS_FROM_EXISTING_M1_EVIDENCE.png` | 92,504 | `5C65447D82419CF0BC2864D87929A7E599B54239D479CADBE1497169C39B428D` |
| `C1_CONTROLLABLE/figures/C1_HIERARCHICAL_PRIMARY_TARGET_QUICKLOOK.png` | 164,731 | `A54D369EA585D6E103116935BC8BCDCA112B8E8947A08D7858EB6CE52F60179B` |
| `slides/RELIABLE_CONTROLLABLE_DIGITAL_TWIN_MVP.pptx` | 201,075 | `86C542767B0801DEF1CE346D87CFC6AAB7163600CEC2FBEB317538C874E45DFE` |
| `slides/RELIABLE_CONTROLLABLE_DIGITAL_TWIN_MVP.pdf` | 635,596 | `F3BB38B789265BA0D002F719BF56359457352E223A695AC9E45F09EE999AFA14` |
