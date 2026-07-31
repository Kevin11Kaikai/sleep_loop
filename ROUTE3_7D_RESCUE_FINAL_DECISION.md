# NO-GO

1. Verdict applies to: **empirical_rank_calibrated**.
2. Fresh independent final validation: **1024 cases**.
3. Parameters passing every 80% and 90% coverage requirement: **4/7**.
4. Parameters passing simultaneous coverage plus contraction: **4/7**.
5. Posterior-median recovery improvement over prior median: **53.4%**.
6. Synthetic PPC improvement over prior predictive: **87.7%**.
7. Every original numerical hard gate passed: **False**.
8. Exact blocker if not Formal GO: **coverage_compatible_gate, contraction_with_coverage_gate**.

## Original, development, and final comparison

| stage                         |   cases | method                              |   coverage_compatible_parameters |   recovery_improvement_fraction |   ppc_improvement_fraction | verdict                                |
|:------------------------------|--------:|:------------------------------------|---------------------------------:|--------------------------------:|---------------------------:|:---------------------------------------|
| Original Route-3              |     256 | raw 3-member MAF64x5                |                                0 |                          0.5155 |                     0.8326 | NO-GO                                  |
| Best development rescue       |     512 | maf128_t8:empirical_rank_calibrated |                                7 |                          0.5293 |                   nan      | selection only; not final              |
| Fresh independent final (raw) |    1024 | maf128_t8:raw                       |                                0 |                          0.5330 |                   nan      | diagnostic raw result; no separate PPC |
| Fresh independent final       |    1024 | empirical_rank_calibrated           |                                4 |                          0.5337 |                     0.8774 | NO-GO                                  |

## Fresh parameter evidence

| parameter   |   prior_median_mae |   posterior_median_mae |   mae_improvement_fraction |   rank_correlation |   median_90_ci_width |   coverage_80 |   coverage_90 | coverage_80_90_compatible   | contracted   |
|:------------|-------------------:|-----------------------:|---------------------------:|-------------------:|---------------------:|--------------:|--------------:|:----------------------------|:-------------|
| mue         |             0.2500 |                 0.1452 |                     0.4193 |             0.7769 |               0.5645 |        0.7725 |        0.8877 | False                       | True         |
| mui         |             0.2500 |                 0.0034 |                     0.9862 |             0.9998 |               0.0112 |        0.8242 |        0.9141 | True                        | True         |
| b           |             0.2500 |                 0.0792 |                     0.6830 |             0.9266 |               0.2676 |        0.7891 |        0.8701 | False                       | True         |
| tauA        |             0.2500 |                 0.1836 |                     0.2655 |             0.6005 |               0.7350 |        0.8125 |        0.8984 | True                        | True         |
| g_LK        |             0.2500 |                 0.1430 |                     0.4278 |             0.7172 |               0.5065 |        0.8066 |        0.9004 | True                        | True         |
| g_h         |             0.2500 |                 0.1214 |                     0.5143 |             0.8140 |               0.4719 |        0.8311 |        0.9004 | False                       | True         |
| c_th2ctx    |             0.2500 |                 0.1401 |                     0.4397 |             0.7635 |               0.6050 |        0.8008 |        0.9023 | True                        | True         |

## Raw versus calibrated final evidence

The selected raw ensemble passed both required coverage levels for **0/7** parameters and improved median recovery by **53.3%**. The primary calibrated result passed **4/7** coverage requirements. Raw and calibrated coverage/SBC tables are stored separately; PPC was run only for the preregistered frozen primary.

## Scientific interpretation

The experiment concerns recoverability in a synthetic cortical population firing-rate observable space. It does not validate a cortical-source-to-Fpz-Cz measurement model and does not authorize real-EEG inference.

Scientifically established: the complete statement is limited to the frozen 7D prior, fixed c_ctx2th, 14D cortical-rate schema, selected pipeline, and fresh independent simulator seeds reported here.

Unsupported: real-subject parameters, a real EEG digital twin, thalamic-state observability from one scalp channel, or mechanism truth.

Readiness: synthetic-only inference remains blocked by calibration.

Single most important next action: resolve the remaining frozen coverage/calibration blocker with a new preregistered method and another untouched final set.
