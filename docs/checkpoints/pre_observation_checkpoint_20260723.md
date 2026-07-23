# Pre-Observation Checkpoint Manifest - 2026-07-23

## 1. 状态与目的

- 创建时间：2026-07-23 15:06:43 -04:00（America/New_York）
- 仓库根目录：当前 Git worktree 根目录（manifest 内不记录机器绝对路径）
- 原 branch：`main`
- 原 HEAD：`82aea491f757b101e702c919dc98aa95d95e11ec`
- `origin/main`：`82aea491f757b101e702c919dc98aa95d95e11ec`
- remote：`origin = https://github.com/Kevin11Kaikai/sleep_loop.git`
- 可见 refs：
  - `main`：`82aea491f757b101e702c919dc98aa95d95e11ec`
  - `publish-main`：`441a2bfbc1a52cf1b842bcea1fd2a371680d6276`
  - `origin/main`：`82aea491f757b101e702c919dc98aa95d95e11ec`
- Git conflict：未发现。
- checkpoint 状态：**安全闸门阻塞，尚未创建 branch、尚未暂存、尚未 commit、尚未 push。**

本 manifest 用于保护 Observation Notebook 开始前的 V7、V8a/T13、PAC repair、held-out validation 与 S4_sbi 本地研究状态。本轮没有运行 simulation、没有训练 NPE、没有实现 Observation Notebook，也没有安装或修改依赖。

## 2. 仓库规则

- 未找到 `AGENTS.md`。
- 已阅读 `README.md`、`.gitignore` 和 `CLAUDE.md`。
- `.gitignore` 明确排除：
  - `data/manifest.csv`
  - `data/sleep-edfx-cassette/`
  - `*.edf`、`*.hyp`
  - `outputs/`
  - `*.log`
  - `.claude/`
  - Python/Jupyter cache、IDE 配置和临时 archive
- `CLAUDE.md` 要求从仓库根目录使用 `neurolib` 环境，并说明 V7、PAC repair、本地 patched neurolib 与 S4_sbi 的历史结构。
- README 仍把 v3 写成主线；当前代码与 `CLAUDE.md` 显示 V7 才是成熟 baseline。

## 3. neurolib 环境

所有 Python 检查均通过：

```text
conda run -n neurolib python ...
```

环境结果：

| 组件 | 版本/状态 |
|---|---|
| Python | 3.10.20 |
| neurolib | 0.6.1 |
| sbi | 0.26.1 |
| torch | 2.5.1 |
| CUDA available | false |
| numpy | 2.2.6 |
| scipy | 1.15.2 |
| mne | 1.9.0 |
| pandas | 2.3.3 |
| matplotlib | 3.10.8 |
| fooof | 1.1.1 |
| numba | 0.64.0 |
| jupyterlab | 4.5.6 |
| ipykernel | 7.2.0 |

`neurolib` 实际从仓库外的本地 patched source tree 导入；为避免写入机器隐私，本 manifest 不记录该绝对路径。

## 4. dirty worktree 摘要

初始 `git status --short` 包含：

### Tracked deletions

```text
D S4_sbi/compute_xobs_from_eeg.py
D S4_sbi/compute_xobs_from_eeg_v1_buggy.py
D S4_sbi/compute_xobs_from_eeg_v2.py
D S4_sbi/plot_scan_diagnostics.py
D S4_sbi/run_sbc_standalone.py
D S4_sbi/scan_xobs_params.py
D S4_sbi/x_obs.npz
```

这些路径在 `S4_sbi/legacy/` 下有 byte-identical 未跟踪副本；7 组 Git blob hash 均一致。因为这是未明确授权的目录重排，本 manifest 只记录，不暂存 rename/deletion。

### Tracked modifications

```text
M S4_sbi/compute_xobs_from_eeg_v4.py
M docs/Claude_Writing/BMEiCON_STAGE1_MANUSCRIPT.docx
M docs/Claude_Writing/BMEiCON_STAGE1_MANUSCRIPT.md
M docs/Claude_Writing/BMEiCON_STAGE1_MANUSCRIPT.tex
M docs/Claude_Writing/build_manuscript_docx.py
M models/s4_personalize_fig7_v2.py
M valid_scripts/validate_so_waveform_heldout.py
M valid_scripts/validate_spindle_density_heldout.py
```

其中 `models/s4_personalize_fig7_v2.py` 的 worktree Git blob 与 index blob 相同，属于状态/换行元数据噪声；没有实际 content diff。其余修改未经本轮创建或授权，不自动纳入 checkpoint。

### 主要 untracked groups

```text
CODEBASE_INVENTORY.md
S4_sbi/legacy/
S4_sbi/plot_concat_vs_mean_psd.py
S4_sbi/scan_diagnostics/*.png
data/patient_params_fig7_v8_SC4001.json
docs/implementation_plan_v8*.md
docs/resume_0711.md
docs/sbi_obs_vs_sim_comparison.{md,html}
models/s4_personalize_fig7_v8.py
valid_scripts/*v8a*.py
valid_scripts/plot_figure10_inspired_validation_panel.py
valid_scripts/plot_v8a_*.py
validation_outputs/*
manuscript/PDF/PPTX/HTML artifacts
```

完整 `git status --short` 仍保留在工作树中；本轮没有清理、移动、覆盖或格式化任何现有成果。

### Ignored files

`git ls-files --others --ignored --exclude-standard` 共识别 492 个 ignored files：

| Top-level group | 文件数 | 总字节 |
|---|---:|---:|
| `data/` | 307 | 7,596,580,410 |
| `outputs/` | 146 | 37,218,484 |
| `docs/` | 2 | 792,581 |
| `valid_scripts/` caches | 13 | 318,733 |
| `models/` caches | 5 | 183,397 |
| `S4_v7_repair/` caches | 4 | 48,372 |
| other caches/config/logs | 15 | 125,729 |

其中 153 个 EDF 文件合计约 7.24 GiB，属于原始 EEG，严禁进入 Git。ignored `outputs/` 中包含需要 manifest 保护的 V8a/T13 结果；其 SHA-256 见下文。

## 5. A 类：拟纳入 checkpoint 的源代码与文档

以下文件通过了内容级 credential scan；15 个 Python 候选均在 `neurolib` 环境中通过 AST parse。它们当前**尚未暂存**，需用户确认安全闸门后才可逐项 `git add`。

| 文件 | 字节 | SHA-256 | 用途 |
|---|---:|---|---|
| `CODEBASE_INVENTORY.md` | 16815 | `36f13682b989e29257c2194c45144a52552c66c6a71e5e2eaa0be8374ee5ac99` | 项目结构与结果资产清单 |
| `data/patient_params_fig7_v8_SC4001.json` | 1146 | `575b56b5265cda3515cb80777799a49a6872dd264ba83ae85706339ed721a493` | V8a full-DE 关键 8 参数与 T1-T13 摘要；不含原始 EEG |
| `docs/implementation_plan_v8.md` | 8055 | `49559b5d1755e825ddb606f66e83c3d02a0a62e5f8607e7202bc1dfbbf650a72` | V8a/T13 受控机制补丁计划 |
| `docs/sbi_obs_vs_sim_comparison.md` | 10868 | `bf664a75e73d93bf87abef8189a0901719856498f3dbd14e522d408d5bfb5609` | observation/simulator summary coordinate 审计 |
| `models/s4_personalize_fig7_v8.py` | 71033 | `9e4f525bee282a7c3fff2ea3a6557a41d8fdf4b6620b32dff97058c58dd598b1` | V8a/T13 模型、约束与 DE 入口 |
| `S4_sbi/plot_concat_vs_mean_psd.py` | 11774 | `2555056689a8c41470eb08553a491758fc0fc7d4e618f498757e9fc53ce1bec1` | observation PSD 聚合方式诊断 |
| `valid_scripts/analyze_v8a_relaxed_t6_sensitivity.py` | 10777 | `def7fe1a53bfeeede61676bd7d211190fb4665fe1d7650a3daf6ee736fcbf2ee` | T6 threshold sensitivity 的 post-hoc 分析 |
| `valid_scripts/analyze_v8a_tradeoff.py` | 15004 | `5d04109635d683addc384308f0e5072d6fed49c258b88e2351ec8d69305e9438` | full-DE T6/T13 trade-off 分析 |
| `valid_scripts/audit_v1_v3_v7_t12.py` | 8410 | `768d413deebe307d3436827d390344201f1c8aee0e583378762c2d2da1ddfe3b` | V1/V3/V7 T12 对照审计 |
| `valid_scripts/diagnose_v8a_t6_t13_coupling_sweep.py` | 13211 | `a7e0a1bf3000bd8e92109eb379685afb7b1ba4423eda953fc50f10c97b99d9f2` | T6/T13 coupling grid 诊断 |
| `valid_scripts/local_de_v8a_t6_rescue_long.py` | 18972 | `8abfbf25ef40bd4cb969312d17f5de991fa7276edf77eaff606be00046cddf62` | long local-DE T6 rescue |
| `valid_scripts/local_de_v8a_t6_rescue_narrow.py` | 18460 | `86ea6f68cb129b2d268e24f35bf09b229965098a825a8a06f731033ebf49234d` | narrow local-DE T6 rescue |
| `valid_scripts/local_search_v8a_ultra_narrow_t6_t13.py` | 16141 | `c20987e072f2f3c7e6cdcbeb63c893612a88d993f90530b130928b12c0952126` | ultra-narrow T6/T13 search |
| `valid_scripts/plot_figure10_inspired_validation_panel.py` | 33736 | `4a41e9341c3837c9d0b927570abbc9d13a6d36d9e0d0f3fc73115366e22f71df` | fitted-candidate audit panel；不是 posterior validation |
| `valid_scripts/plot_v8a_0712_best0417_figures.py` | 20940 | `34d4e7758641adc4a775adbbd7a4e9eb6cc9f367b3f4c9993b8577dd4c12caed` | best0417 图与复核 |
| `valid_scripts/plot_v8a_0712_heldout_spindle_density.py` | 12592 | `f32b5c7a7dc2e6c4008ebcbb72242e8eb1c37f8513e851b030534772e0c0841d` | V8a 独立 RMS held-out density 图 |
| `valid_scripts/run_so_waveform_canonicalV1.py` | 1823 | `9ba98868a2b83e39d8cd9a31d8c04603c7068452b6bed8a07a3d8992cd30196d` | canonical V1 SO waveform runner |
| `valid_scripts/sweep_v8a_coupling_tradeoff_long.py` | 29586 | `b02f73dfa65d6aea33d10ee6c26686980a736229757daf76d422e136560a5f1c` | V8a multi-phase coupling sweep |
| `valid_scripts/t5_deconfound.py` | 3911 | `40161a2b9628d78a47e4d90488a04c2eff729d89e23dd4f9b731e3163a4714ad` | T5 deconfounding 审计 |

## 6. A 类：拟纳入 checkpoint 的小型关键 artifact

这些是 aggregate summary、best-candidate JSON 或必要小型 CSV，不含 raw EEG、模型权重、PKL/PT 或大数组。它们位于 ignored `outputs/` 时，后续需逐路径使用 `git add -f -- <path>`，不能使用 `git add .`。

| 文件 | 字节 | SHA-256 |
|---|---:|---|
| `outputs/v8a_diagnostics/v8a_12of13_candidates.csv` | 10636 | `da8c3534c2d40efcf0e8f12e08b5a7355fdb0786a58ae7009215b6d88a9f249d` |
| `outputs/v8a_diagnostics/v8a_correlations.csv` | 706 | `3fc56d634f7d72ec94a3cfff194783401e6b2ffe56c87ade6534bf21b505ec4a` |
| `outputs/v8a_diagnostics/v8a_group_stats.csv` | 5083 | `6084949fe1347c6abb8e01de0bc54799854d3fa3fd4b2119315aa7b1b157a504` |
| `outputs/v8a_diagnostics/v8a_tradeoff_summary.txt` | 7620 | `c3bfeb58e1cabe0c655f73266aacc9c43fdacd210441b5bbae6211de740df74f` |
| `outputs/v8a_relaxed_t6_sensitivity/relaxed_t6_best_candidates.csv` | 2384 | `78febb6b90fbc83d46fdbf9e72b3b8d723647186b49c2967898fdeffe2df14da` |
| `outputs/v8a_relaxed_t6_sensitivity/relaxed_t6_interpretation.md` | 2020 | `a6c4052f55cc0ab0a3ad0c59a23d2a2506a916a8c79c9468419e183c195357ca` |
| `outputs/v8a_relaxed_t6_sensitivity/relaxed_t6_summary.csv` | 1406 | `354c35f8f16470b86ab6db19eefc6290ed6d6a9d2382c42656ccf48ef96664e7` |
| `outputs/v8a_t6_t13_coupling_sweep_summary.txt` | 610 | `fab9a58344c0ddfd31e16e06bac5eca581ceb7affbeedb53c5da53d1a034b5b3` |
| `outputs/v8a_ultra_narrow_t6_t13_search/best_so_far.json` | 877 | `bf18212ebac0ebec39d4f13aecd5b7a9da5618ffe78abb5f8694a67f569af76a` |
| `outputs/v8a_ultra_narrow_t6_t13_search/selected_base_point.json` | 732 | `d1fa91bc47efb1b882a9c400fcde62c91fa75f21f60f4d48ccf6e6f2e1a4d768` |
| `outputs/v8a_ultra_narrow_t6_t13_search/summary.txt` | 906 | `935629c3862b9ad5352e9f9ea4e8d3017730a340c48ef8c55a20d8f419951fbf` |
| `outputs/v8a_local_de_t6_rescue_long/best_so_far.json` | 1029 | `428cc830179c0cf35aff4fff3ea3ca4ef0819d916669a1e2c929281373349ed0` |
| `outputs/v8a_local_de_t6_rescue_long/summary.txt` | 1016 | `b607523ca83664bec7d61b136497359dfc2fde0ef06e6f399f4d30d1ca95501f` |
| `outputs/v8a_local_de_t6_rescue_narrow/best_so_far.json` | 1039 | `4bc2ec24fddbe361e49fa155e9f184723e0db602914d4d26f2fd158005b83936` |
| `outputs/v8a_local_de_t6_rescue_narrow/summary.txt` | 44 | `c108494c61c91277f14212f40d3d2e6b6ad19aa6d9bf2a11abd3b1323cda9d5c` |
| `outputs/v8a_coupling_sweep_long/selected_seed_candidates.csv` | 11064 | `ffeaa399ac1f56e1dcba0dd9d5f75fb06b7d0d8662e525d0a224df9252677b41` |
| `outputs/v8a_coupling_sweep_long/v8a_coupling_sweep_long_summary.txt` | 1897 | `9cbf4f258f75d70ce4ff848e97812b938ad885360deaf7e47419d5930775b578` |
| `outputs/v8a_0712_best0417_figures/fig_v8a_0712_heldout_spindle_density.csv` | 317 | `13606acfe051ebdb0e8eebedd10d09ad08f98bc45d4ec44104153961b3668ccb` |
| `outputs/v8a_0712_best0417_figures/fig_v8a_0712_heldout_spindle_density_summary.txt` | 937 | `eb1b0f14f5dd20c4350c682326a33e061acef3b67c5c98bfa2417f1d3e3424e0` |
| `outputs/v8a_0712_best0417_figures/fig7_v8a_0712_best0417_residual_metrics.json` | 214 | `1982907346ef23c710cb2c6f4bb6e2eb10b1f898ebfcea780c0e3affed833412` |
| `outputs/v8a_0712_best0417_figures/manifest.json` | 1452 | `fc854cd7468d2f5204702df4e67ae780eb688386346895236af6d9f13668262b` |
| `outputs/v8a_0712_best0417_figures/v8a_0712_best0417_candidate_verification_metrics.json` | 542 | `d6524954eb819fb043adc9c494d104e15ea4f50dacbb393411fa920a6f1b9263` |
| `outputs/figure10_inspired_validation_panel/candidate_archive.csv` | 13852 | `2352511f13ef89542143ad2966aefb8704b0c1a1c27c1b103f78e3315e2fc4ca` |
| `outputs/figure10_inspired_validation_panel/candidate_predictive_summaries.csv` | 52007 | `d084206ad4c7c95b47afff53b35c6616d56ef402a0e620dc93216ac936bfba25` |
| `outputs/figure10_inspired_validation_panel/figure10_inspired_validation_panel_summary.md` | 2999 | `f2d45b2c32fefa77f387033b3f3a41294577d8c8f19ee3214d5353e4d5b50c1c` |
| `outputs/figure10_inspired_validation_panel/synthetic_recovery_results.csv` | 3822 | `b5447ade9a62599034bf9881aefcc4ac978a0a92101f2e81ca8d6b698ac9d11e` |
| `validation_outputs/rescore_all_versions_142.csv` | 711 | `5fc2cc63eccc65350c38f174dacbc5f27fe657a3d5e2c6ed6cbde65d1bed0ddf` |
| `validation_outputs/rescore_all_versions_142.txt` | 1112 | `2415e8e3deb57de02c839ae463b70915f16c5f6dc46d14a9f797f85aab526154` |
| `validation_outputs/t5_deconfound_summary.txt` | 1282 | `26805eabf1d2720316de444ff1d0b4feb247aa7401d730ddb8e6b61f2c6193ce` |
| `validation_outputs/v1_v3_v7_t12_audit.csv` | 1345 | `61f4d0d93dba257e3c076cd05c2d60c75230fbb2e5c7b27522fcb4a651e6815c` |
| `validation_outputs/v1_v3_v7_t12_audit_summary.txt` | 2680 | `f4a5f2eb2fac9977d88b9f4b985a127d074c9a32a1c4b6c9599f6bced912b286` |

`outputs/figure10_inspired_validation_panel/real_observation_summaries.csv` 未列入 A 类，因为它包含单一 subject/channel 标识与 observation-derived aggregate；在用户确认可提交 pseudonymous subject aggregate 前按 D 类处理。

## 7. B 类：保留但不直接提交

以下逐点评估记录是重要研究 artifact，已计算 SHA-256，但属于大 CSV、simulation records 或可重新生成中间结果：

| 文件 | 字节 | SHA-256 | 排除原因 |
|---|---:|---|---|
| `outputs/evolution_fig7_v8_records.csv` | 1609712 | `8a3f069cf2f073890f6c68b541e706dfd6ac2e4f8162722f2c3af0218f8cec71` | 4960-point full-DE record |
| `outputs/v8a_coupling_sweep_long/phase1_cth2ctx_sweep.csv` | 616584 | `c240e96c44da0e78849bcb7f7bbdd4cdf8098663de6072fd2c2d85526e47b77d` | 可重生成 grid record |
| `outputs/v8a_coupling_sweep_long/phase2_coupling_2d_sweep.csv` | 129585 | `01e3f38949ca429b5c4b52c3ef7cb249aa3070e1348751d466e6d045e682b6c9` | 可重生成 grid record |
| `outputs/v8a_local_de_t6_rescue_long/local_de_records.csv` | 1993578 | `2b8667e290ade7ed4032354cb8eb9e5a8f398da7b628a2663bd151603c800432` | 最大 local-DE record |
| `outputs/v8a_local_de_t6_rescue_narrow/narrow_local_de_records.csv` | 205063 | `05ed4086ff89aba7138052e55ca97a715ccc106a557707d7c32695f96264fefa` | 可重生成 local record |
| `outputs/v8a_t6_t13_coupling_sweep.csv` | 926402 | `1170abeb31e784c24e9664cb772e94b3e254bb07c8baf3c88527549c3feb7e77` | 3875-point coupling record |
| `outputs/v8a_ultra_narrow_t6_t13_search/ultra_narrow_records.csv` | 422144 | `aa5e340c009685601e93d67f96afadbcabb91093444de74b154ce5d8b06bcc66` | 可重生成 local-DE record |
| `S4_sbi/legacy/x_obs.npz` | 2324 | `ef669b21437d5edf32374e80d5f972a10b431650a81c6fa0c680ac5169162678` | binary observation artifact；与 tracked deleted blob 相同 |
| `S4_sbi/scan_diagnostics/fig_concat_vs_mean_psd_overview.png` | 129787 | `7d7f5ab2aff5789c520bcc4655ac96ac99133e204f47b42e91cfedb4f272da79` | 可重生成图像 |
| `S4_sbi/scan_diagnostics/fig_concat_vs_mean_psd_so_zoom.png` | 91850 | `b5f1694f6ef550826f1ab15a0bda3136902cd12734a4035311bc698457ad1e25` | 可重生成图像 |
| `S4_sbi/scan_diagnostics/fig_concat_vs_mean_psd_timeline.png` | 151807 | `da2e32100b6e2997fca7a5c7e398c82da05d2fd2863a34362742c3790840a2fb` | 可重生成图像 |

V8/V8a/T13/Figure10 相关可重生成 PNG/PDF 共 28 个、4,725,632 bytes。按排序后的 `path|size|sha256` 清单计算的 inventory SHA-256 为：

```text
a74e5e553493a385a67bbe374410f739ea34a4bec38920e21b49c42d47f9f29e
```

相关 `.pyc` cache 共 13 个、344,619 bytes，全部排除。

## 8. C 类：敏感或禁止提交

| 路径/组 | 字节/SHA-256 | 原因 |
|---|---|---|
| `data/sleep-edfx-cassette/**/*.edf` | 153 files，约 7.24 GiB | 原始 EEG；严禁提交 |
| `data/manifest.csv` | 不读取内容、不提交 | 含本地 PSG/hypnogram 绝对路径与 subject mapping |
| `.claude/settings.local.json` | ignored | 本地机器/工具配置 |
| `docs/implementation_plan_v8_progress_1.md` | 追加前：89280 bytes，`1fd5a654ff3a5e4aedf13382d67a31be266365e0abf2b86df273c5a23ac0c89c`；追加后：94403 bytes，`31ebbe0e222e34f810056df96391dded809ce5fd0d13cbf19068105e8e81a93a` | 含多处用户目录和机器绝对路径；已按要求追加本次进度，但不拟提交 |
| `docs/resume_0711.md` | 3975 bytes；`cc0cbcbdb4783c56d7bd5d3e846c05bac83db1c842531d56b5f5ef064de8bb0a` | 含用户目录绝对路径 |
| `outputs/v8a_full_de_run_log.txt` | 12470 bytes；`955d7c7a2271bdc577fc77bdd81350345fa7ddf6d6e8218c2e294bd065f444d7` | 含本地解释器绝对路径；结果由小型 summary 代替 |
| `S4_sbi/compute_xobs_from_eeg_v4.py` | 21240 bytes；`5cfb99fb306a27b16dcf90eb96f0924464701bcd8a6aa78cb884a5ea82c75a79` | 含硬编码本地 neurolib 绝对路径，且本地改动主要为大规模注释翻译 |
| `valid_scripts/validate_spindle_density_heldout.py` | 14430 bytes；`c4c748da25fff2b8a0bae4c01984481e9fbc370281b063aa13f42cf245740f36` | 含硬编码本地 neurolib 绝对路径，且为未明确用户修改 |
| `validation_outputs/psd_overlay_sigma_guard_metrics.txt` | 未拟提交 | 内容扫描发现机器绝对路径 |

credential scan 未在拟纳入 A 类文件中发现 private key、GitHub/OpenAI token、AWS key 或 credential assignment。该结论仅针对当前扫描模式，不能替代提交前 staged-content 再检查。

## 9. D 类：暂时无法判断

### S4_sbi legacy 重排

以下 untracked 文件与原 tracked deleted 文件 Git blob 完全一致，因此内容可从原 HEAD 恢复；是否将目录重排作为 rename commit 需要用户确认：

| 原路径 -> 当前路径 | Git blob |
|---|---|
| `S4_sbi/compute_xobs_from_eeg.py` -> `S4_sbi/legacy/compute_xobs_from_eeg.py` | `b632d580053cb2a7a2697fddabba85b7d3b715a6` |
| `S4_sbi/compute_xobs_from_eeg_v1_buggy.py` -> `S4_sbi/legacy/compute_xobs_from_eeg_v1_buggy.py` | `b632d580053cb2a7a2697fddabba85b7d3b715a6` |
| `S4_sbi/compute_xobs_from_eeg_v2.py` -> `S4_sbi/legacy/compute_xobs_from_eeg_v2.py` | `68bbe920ed0d30b2229f09375341117b10ab2c77` |
| `S4_sbi/plot_scan_diagnostics.py` -> `S4_sbi/legacy/plot_scan_diagnostics.py` | `427ab033fbae827350f6175df3677fb9cd8d8fc5` |
| `S4_sbi/run_sbc_standalone.py` -> `S4_sbi/legacy/run_sbc_standalone.py` | `5097cb11056a07c428806adbfdd8844572439a63` |
| `S4_sbi/scan_xobs_params.py` -> `S4_sbi/legacy/scan_xobs_params.py` | `a5512baa09ca05dc322ea2064747878c48d40dcc` |
| `S4_sbi/x_obs.npz` -> `S4_sbi/legacy/x_obs.npz` | `87f5d96329e499aad8716f4167c04c58b222a5b9` |

### 未明确用户修改

| 文件 | 字节 | SHA-256 | 疑点 |
|---|---:|---|---|
| `valid_scripts/validate_so_waveform_heldout.py` | 24660 | `0ed0fcbcd561ac8be47bd27232872e8412d7245d192bcb945ac4ad68cd119121` | 大量注释式修改，未确认是否要纳入研究 checkpoint |
| `valid_scripts/validate_spindle_density_heldout.py` | 14430 | `c4c748da25fff2b8a0bae4c01984481e9fbc370281b063aa13f42cf245740f36` | 注释式修改 + 机器路径 |
| `S4_sbi/compute_xobs_from_eeg_v4.py` | 21240 | `5cfb99fb306a27b16dcf90eb96f0924464701bcd8a6aa78cb884a5ea82c75a79` | 注释式修改 + 机器路径 |
| `models/s4_personalize_fig7_v2.py` | 27820 | `fe20ec077eee195c4f0415519c3e236844af662c3313d46bc4ed789ea051dbda` | Git status 为 M，但 worktree/index blob 相同 |
| `docs/Claude_Writing/BMEiCON_STAGE1_MANUSCRIPT.docx` | 766991 | `74c02a1c34ca04ff851f8e6b26e70123ca83b575d156900786b7174530ce996b` | 用户 manuscript binary 修改 |
| `docs/Claude_Writing/BMEiCON_STAGE1_MANUSCRIPT.md` | 41475 | `ab7be3b87ed18db9414924b531996b283a39a59633ae6a2fc21482382258fb0c` | 用户 manuscript 修改 |
| `docs/Claude_Writing/BMEiCON_STAGE1_MANUSCRIPT.tex` | 47651 | `e26bfe31d4bafd4bb5e02824f495ce6182a616fbc355bf3a8a335d1c34c83a4b` | 用户 manuscript 修改 |
| `docs/Claude_Writing/build_manuscript_docx.py` | 54071 | `489b44482633cf08cddd3e389a180da68e14132dab09390ca5a1303f178e50a8` | 用户 manuscript build 修改 |
| `docs/Claude_Writing/BMEiCON2026_Writing_Plan.pptx` | 250462 | `59e396f14942c9f1df97f26e5bd6717299624d748566f0ac06e18b4d83ce3fe1` | 未跟踪 binary presentation |
| `docs/Claude_Writing/BMEiCON_STAGE1_MANUSCRIPT.pdf` | 983424 | `1ccbb7894f06f1309a302f68a5db5413ad425d54c4516371332ede98614c0f82` | 可重生成 manuscript PDF |
| `docs/数字孪生策略报告.pdf` | 4289926 | `3f9b40e5c68924169e1b5ceaf5c8fd4e4ca2332a382fe553979ec2c79f093a65` | 最大非 ignored untracked binary；与本 checkpoint 范围关系不明 |
| `outputs/figure10_inspired_validation_panel/real_observation_summaries.csv` | 254 | `cb37d15a73a5515bbdf8e2efbf0f06bf32199db9b83e846ec3515351ad7d21a2` | 含 pseudonymous subject/channel aggregate；需确认公开边界 |

HTML、PNG、PDF、PPTX 和重复 manuscript export 默认不进入代码 checkpoint，除非用户明确指定。

## 10. 已知科学状态

1. V7 是成熟 baseline。
2. V8a/T13 是 mechanism/audit candidate，不是最终验证成功的模型。
3. 严格 13/13 尚未稳定实现；当前 best0417 在 strict T6 阈值下仍是 12/13 near-miss。
4. T13 内部 cortical detector 与独立 RMS spindle detector 仍不一致；V8a best0417 的独立 cortical RMS density 仍为 0/min。
5. V1-V8/V8a DE fitting candidates 不能称为 posterior samples。
6. 旧 S4_sbi 的四轮 posterior files 是一个 sequential SNPE-C 过程的 checkpoints，不是五个独立 NPE。
7. 当前项目没有 Observation Notebook；本轮也没有创建 Notebook。

## 11. 安全闸门与下一步

本轮在 commit 前发现：

1. 重要 S4_sbi/held-out 文件包含机器绝对路径。
2. S4_sbi legacy 重排、held-out 注释大改和 manuscript 修改属于未明确用户修改。
3. `real_observation_summaries.csv` 是否可作为 pseudonymous aggregate 提交尚未确认。

因此严格按用户要求停止在 branch/staging/commit 之前。当前没有 checkpoint commit hash。

建议用户确认以下范围后再继续：

- 是否只提交第 5、6 节的 A 类文件，并排除全部 C、D 类；
- 是否将 `S4_sbi/legacy/` 重排作为 rename 纳入；
- 是否排除三份含机器路径或注释大改的 tracked modifications；
- 是否允许提交 `real_observation_summaries.csv`；
- 是否将 ignored `outputs/` 中第 6 节的小型摘要使用逐路径 `git add -f` 纳入。

确认后拟创建 branch：

```text
checkpoint/pre-observation-20260723
```

拟用 commit message：

```text
checkpoint: preserve pre-observation V8a T13 work
```

本轮不 push。只有 checkpoint commit 完成并再次通过 staged-content 安全检查后，才会提出准确的 push 命令。

## 12. 用户确认与实际执行范围

用户已在 2026-07-23 明确确认继续本地 checkpoint。本节取代第 1、11 节中“等待确认”的临时状态描述；前述分类和哈希记录保持有效。

实际执行范围：

```text
checkpoint branch = checkpoint/pre-observation-20260723
source branch = main
source HEAD = 82aea491f757b101e702c919dc98aa95d95e11ec
staged scope = 第 5、6 节 A 类文件 + 本 manifest
legacy reorganization = excluded
C class = excluded
D class = excluded
real_observation_summaries.csv = excluded
full-DE/local-search records = excluded
NPZ/PNG/PDF/cache/log/raw EDF/data manifest = excluded
push = not authorized in this stage
```

ignored `outputs/` 仅暂存第 6 节逐项列出的 JSON、TXT、MD 和 aggregate summary CSV。没有使用 `git add .`、`git add -A` 或 wildcard。

checkpoint commit message：

```text
checkpoint: preserve pre-observation V8a T13 work
```

manifest 无法在自身所处 commit 内预写最终 commit hash；最终 hash 由 commit 后的 `git rev-parse HEAD` 和本轮报告记录。

## 13. Commit 前第二次安全闸门

逐路径暂存后，对 Git index 中的实际内容执行了第二次安全检查：

```text
expected staged files = 51
actual staged files = 51
manifest/staged set mismatch = 0
staged total bytes = 476372
largest staged file = 69967 bytes
credential hits = 0
disallowed extension/path hits = 0
machine absolute path hits = 3 files
```

命中的 3 个文件为：

```text
valid_scripts/audit_v1_v3_v7_t12.py
  line 19: hard-coded repository root
  line 23: hard-coded local neurolib path

valid_scripts/run_so_waveform_canonicalV1.py
  line 15: hard-coded repository root
  line 19: hard-coded local neurolib path

valid_scripts/t5_deconfound.py
  line 18: hard-coded os.chdir path
```

本 manifest 不记录绝对路径值。以上文件此前被误列入第 5 节 A 类；第二次 staged-content 检查将其重新分类为 C 类，必须从 checkpoint 排除。其 worktree 内容没有被修改、移动或删除。

`git diff --cached --check` 另报告了若干 trailing whitespace 和 EOF blank-line 警告；这些来自现有研究文件，不涉及敏感信息。本轮不为获得 clean diff 而自动格式化科研文件。

由于用户此前明确禁止 `git reset`、`git restore` 和 `rm`，本轮没有自行使用这些命令清理 index，并严格停在 commit 前。继续 checkpoint 前，需要用户明确授权仅对上述 3 个新文件执行 index-only unstage；该操作不得修改 worktree 内容。

## 14. 授权 unstage 后的最终安全集合

用户再次确认完成 checkpoint，并明确排除所有含机器绝对路径的文件。随后仅对第 13 节列出的 3 个新文件执行 index-only unstage；worktree 文件仍存在，大小和 SHA-256 未变化。

最终 commit 候选集合：

```text
expected safe files = 48
actual staged files = 48
set mismatch = 0
staged total bytes = 463708
largest staged file = 69967 bytes
credential hits = 0
machine absolute path hits = 0
disallowed extension/path hits = 0
full-DE/local-search record hits = 0
```

最终集合等于：

```text
第 5、6 节原 A 类
- valid_scripts/audit_v1_v3_v7_t12.py
- valid_scripts/run_so_waveform_canonicalV1.py
- valid_scripts/t5_deconfound.py
+ 本 checkpoint manifest
```

3 个排除文件仍为 untracked worktree files，未修改、未移动、未删除。`git diff --cached --check` 的非零结果仅来自已记录的 trailing-whitespace/EOF 警告；不涉及敏感信息、原始数据或意外二进制文件。
