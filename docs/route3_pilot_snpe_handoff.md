# Route-3 Pilot SNPE Handoff

Date: 2026-07-27  
Scope: synthetic cortical-rate observable recovery only  
Formal decision: **NO-GO**

## Scientific boundary

This experiment uses cortical excitatory and inhibitory population firing rates and the frozen 14D model-observable summaries. These signals are not simulated EEG. No real SC4001 observation, EEG threshold, EEG scaling, Hz-to-uV mapping, or real-EEG parameter inference was used.

Even a successful Route-3 result would not resolve the cortical-source-to-Fpz-Cz measurement-model blocker. The actual result does not validate all eight parameters in synthetic space.

## Frozen contract

Parameter order:

```text
mue, mui, b, tauA, g_LK, g_h, c_th2ctx, c_ctx2th
```

Inference schema:

```text
cortex_rate_only_14d
```

The privileged 24D internal-state schema was not used for training.

## Notebook 11: multi-seed and global robustness

Actual simulations:

```text
128 theta x 3 seeds = 384 prior-wide simulations
3 centers x 3 seeds x (1 center + 3 scales x 8 parameters x 2 directions)
    = 441 local simulations
total = 825
failed = 0
non-finite = 0
simulation runtime sum = 4,860.31 s
stage wall time = 831.2 s
```

All 27 center/scale/seed Jacobians reached effective rank 8 using the pre-frozen threshold `relative singular value >= 1e-3`. Condition numbers ranged from 18.85 to 80.93:

| Perturbation | Rank range | Median condition | Maximum condition |
|---|---:|---:|---:|
| 1% | 8--8 | 44.43 | 80.93 |
| 2% | 8--8 | 38.32 | 73.37 |
| 5% | 8--8 | 34.28 | 51.52 |

Seed robustness was not uniformly strong:

- all-three-seed nearest-neighbor identity agreement: 28.9%;
- pairwise agreement: 39.1%--46.9%;
- sigma peak-frequency SNR: 1.53 for EXC and 3.14 for INH;
- SO peak features had only 5 distinct grid values per seed;
- sigma peak features had 12--13 distinct values;
- multiple Jacobian directions reversed across seeds, centers, or perturbation scales.

The 128-point audit found 1,469 far-theta collisions inside the 95th-percentile same-theta seed-noise floor and 3,670 near collisions.

## Notebook 12: 2,048-point bank

```text
attempted = 2,048
valid = 2,048
failed = 0
training rows = 1,638
validation rows = 410
unique simulator seeds = 2,048
simulation runtime sum = 11,446.81 s
stage wall time = 1,936.4 s
```

The bank and split artifacts contain no object arrays. Feature order and contract hashes match Notebook 10. Scaling was fit only on the 1,638 training rows.

The higher-resolution collision audit found:

```text
all pairs = 2,096,128
far-theta pairs = 1,999,861
collisions inside seed-noise floor = 327,248
near collisions within two noise floors = 939,831
collision fraction among far pairs = 16.36%
```

This is a major global-identifiability warning.

## Notebook 13: exploratory NPE ensemble

Three independently initialized single-round MAF estimators used the same split and training policy:

| Seed | Best epoch | Best validation loss | Initial training runtime |
|---:|---:|---:|---:|
| 1301 | 294 | -8.4982 | 48.02 s |
| 1302 | 267 | -8.5618 | 46.75 s |
| 1303 | 295 | -8.2378 | 47.02 s |

Initial ensemble training stage runtime was 144.02 s on CPU. All engineering samples were finite and inside normalized prior support. The equal ensemble samples approximately one third from each member with no learned or held-out-dependent weights.

The GO criteria were frozen at `2026-07-27T20:58:09.411016+00:00`, before the independent held-out dataset was generated.

## Notebook 14: held-out recovery

```text
held-out theta = 128
training theta duplicates = 0
held-out failures = 0
ensemble samples per case = 4,096
posterior sample tensor = (128, 4096, 8)
finite and in-prior sample rate = 100%
```

Parameter-wise normalized recovery:

| Parameter | Posterior median MAE | Prior median MAE | Improvement | Rank correlation | Median 90% width |
|---|---:|---:|---:|---:|---:|
| mue | 0.1545 | 0.2500 | 38.2% | 0.754 | 0.516 |
| mui | 0.0050 | 0.2500 | 98.0% | 0.999 | 0.021 |
| b | 0.0952 | 0.2500 | 61.9% | 0.894 | 0.389 |
| tauA | 0.1912 | 0.2500 | 23.5% | 0.550 | 0.651 |
| g_LK | 0.1579 | 0.2500 | 36.9% | 0.682 | 0.504 |
| g_h | 0.1338 | 0.2500 | 46.5% | 0.793 | 0.457 |
| c_th2ctx | 0.1505 | 0.2500 | 39.8% | 0.741 | 0.501 |
| c_ctx2th | 0.2515 | 0.2500 | -0.6% | -0.039 | 0.780 |

Seven parameters beat the prior-median baseline. `c_ctx2th` did not.

## Coverage and SBC

Ensemble empirical coverage:

| Parameter | 50% | 80% | 90% |
|---|---:|---:|---:|
| mue | 0.383 | 0.703 | 0.820 |
| mui | 0.742 | 0.914 | 0.961 |
| b | 0.531 | 0.781 | 0.898 |
| tauA | 0.391 | 0.664 | 0.820 |
| g_LK | 0.383 | 0.703 | 0.828 |
| g_h | 0.414 | 0.727 | 0.852 |
| c_th2ctx | 0.375 | 0.719 | 0.805 |
| c_ctx2th | 0.352 | 0.633 | 0.773 |

Only `b` had both 80% and 90% Wilson intervals containing nominal coverage. `c_ctx2th` met the pre-frozen severe-undercoverage definition at the 80% level.

Overall mean coverage across parameters was 0.446, 0.730, and 0.845 for nominal 50%, 80%, and 90%. Ensemble SBC ten-bin warnings included `mui`, `tauA`, and `c_ctx2th`; these p-values are descriptive and were not used alone.

## Ensemble disagreement and ridges

The ensemble disagreement gate passed:

```text
mean member-median range across parameters = 0.051 prior width
largest parameter-level mean range = 0.072 prior width
```

`g_LK/g_h` posterior correlation varied by case, with median Spearman rho 0.154 and maximum 0.618. Both parameters improved over prior baseline, but their coverage was generally below nominal.

The bidirectional-coupling posterior correlation was near zero on average, but this does not imply both directions were identifiable: `c_ctx2th` showed no recovery and severe 80% undercoverage, whereas `c_th2ctx` improved but remained undercovered.

## Synthetic PPC

PPC used 32 fixed-random held-out cases with 16 posterior theta and 16 prior theta per case:

```text
posterior-predictive simulations = 512, failures = 0
prior-predictive simulations = 512, failures = 0
features better than prior predictive = 14/14
median posterior-predictive scaled error = 0.0964
median prior-predictive scaled error = 0.6433
overall improvement = 85.0%
```

Strong PPC does not override calibration failure because predictive summaries can match under global theta collisions.

## Frozen decision

**NO-GO** for claiming validated eight-parameter Route-3 recovery.

Passed:

- training and held-out simulation reliability;
- finite and prior-supported posterior samples;
- seven parameters better than prior baseline;
- overall recovery improvement of 43.0%;
- 14/14 PPC features better than prior predictive;
- ensemble disagreement gate.

Failed:

- only 1/8 parameters met the joint 80%/90% coverage compatibility requirement;
- one parameter had severe undercoverage;
- only 1/8 parameters simultaneously met contraction and coverage requirements;
- `c_ctx2th` was not recovered;
- global collisions were common.

## Recommended next experiment

Do not automatically expand to 4K or 8K and do not unlock real-EEG inference.

The most defensible next step is a new pre-registered synthetic experiment that first resolves the weak direction:

1. test a seven-parameter contract with `c_ctx2th` fixed using independent held-out data;
2. alternatively reparameterize the two coupling directions with a scientifically justified constraint;
3. audit calibration-aware training and estimator capacity without reusing the current held-out set for tuning;
4. decide whether quantized peak-frequency features should be replaced only through a new schema version and new held-out dataset;
5. retain Route 2 measurement-model work as a separate prerequisite for any real Fpz-Cz inference.

## Delivered files

Notebooks:

```text
S4_sbi/notebooks/11_Route3_Global_Robustness.ipynb
S4_sbi/notebooks/12_Route3_2048_Simulation_Bank.ipynb
S4_sbi/notebooks/13_Route3_Exploratory_SNPE_Ensemble.ipynb
S4_sbi/notebooks/14_Route3_Heldout_Recovery_and_Coverage.ipynb
```

Modules and tests:

```text
S4_sbi/src/sleep_sbi/route3_global_robustness.py
S4_sbi/src/sleep_sbi/route3_pilot_snpe.py
S4_sbi/src/sleep_sbi/route3_heldout_validation.py
S4_sbi/scripts/run_route3_pilot_stage.py
S4_sbi/scripts/build_route3_pilot_notebooks.py
S4_sbi/tests/test_route3_pilot_contracts.py
```

Artifacts and HTML:

```text
S4_sbi/results/route3_pilot_snpe/
S4_sbi/results/route3_pilot_snpe/html/11_Route3_Global_Robustness.html
S4_sbi/results/route3_pilot_snpe/html/12_Route3_2048_Simulation_Bank.html
S4_sbi/results/route3_pilot_snpe/html/13_Route3_Exploratory_SNPE_Ensemble.html
S4_sbi/results/route3_pilot_snpe/html/14_Route3_Heldout_Recovery_and_Coverage.html
```

Validation:

```text
Notebook code cells executed: 35/35
Notebook error outputs: 0
Unit tests: 7 passed
JSON reloaded: 17
CSV reloaded: 22
NPZ reloaded without pickle: 4,161
PyTorch checkpoints reloaded: 6
Unexpected object arrays: 0
HTML exports: 4
```

No commit or push was performed. Notebooks `00`--`10` were not modified.
