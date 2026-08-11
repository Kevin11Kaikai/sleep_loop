# 8D FIGURE-10-EQUIVALENT: FAIL
# 7D FIGURE-10-EQUIVALENT: FAIL
# ORIGINAL ROUTE-3 STRICT: NO-GO

- Valid 8D training simulations: **32,768**
- Valid matched 7D training simulations: **32,768**
- Networks per final ensemble: **5**
- Expected-coverage/SBC cases: **300 official-equivalent + 1,024 powered per track**
- Posterior samples per global case: **1,000**
- L-C2ST calibration simulations: **20,000 per track**
- 8D powered joint-rank KS: `D=0.2250`, `p=4.915e-46`
- 7D powered joint-rank KS: `D=0.2118`, `p=8.893e-41`
- 8D clear SBC issues: **3/8**
- 7D clear SBC issues: **3/7**
- 8D L-C2ST: `p=0`, reject=`True`
- 7D L-C2ST: `p=0`, reject=`True`
- 8D PPC improvement: **60.0%**, 14/14 features
- 7D PPC improvement: **59.3%**, 14/14 features
- Original Route-3 hard-gate outcome: **NO-GO**
- Applicability: **synthetic cortical-rate inference only**

## Method comparison

| method                            | dimension    | training_simulations   | density_estimator   | ensemble   |   recovery | joint_coverage                 | sbc                          | lc2st   |        ppc | contraction        | verdict                      |
|:----------------------------------|:-------------|:-----------------------|:--------------------|:-----------|-----------:|:-------------------------------|:-----------------------------|:--------|-----------:|:-------------------|:-----------------------------|
| Historical raw SNPE               | 7D/8D stored | stored pilot           | historical MAF      | 3 stored   |   0.430218 | not Figure-10 joint diagnostic | historical marginal coverage | not run |   0.850178 | strict gate failed | NO-GO                        |
| Historical rank-calibrated rescue | 7D           | 8192 rows              | MAF                 | 5          | nan        | not Figure-10 joint diagnostic |                              | not run | nan        |                    | NO-GO                        |
| New large-scale raw 8D ensemble   | 8D           | 32768                  | NSF 10x100          | 5          |   0.430475 | 4.914556785984683e-46          | 3                            | 0.0     |   0.60049  | 0.6595531769290588 | 8D FIGURE-10-EQUIVALENT FAIL |
| New matched raw 7D ensemble       | 7D           | 32768                  | NSF 10x100          | 5          |   0.50285  | 8.893435253963037e-41          | 3                            | 0.0     |   0.592818 | 0.6245432999231879 | 7D FIGURE-10-EQUIVALENT FAIL |

## Direct answers

1. **Was 8,192 an important cause of earlier miscalibration?** See the frozen
   8,192-versus-32,768 scale table. Improvement is reported only when the
   measured calibration metrics support it; unrun 131k–3M scales are not
   extrapolated as scientific results.
2. **Did five-network ensembling improve calibration?** The individual and
   ensemble rank curves are reported side by side in the global diagnostics.
3. **Did releasing `c_ctx2th` harm the other seven parameters?** Shared
   contraction and calibration are compared under paired simulation draws and
   seeds; any widening is distinguished from miscalibration.
4. **Is `c_ctx2th` identifiable?** Its contraction and the
   `c_th2ctx`–`c_ctx2th` ridge are reported. Broad-but-calibrated uncertainty is
   accepted; narrow biased uncertainty is not.
5. **Does 8D give a scientifically more complete posterior?** It represents
   uncertainty in both coupling directions, but completeness does not override
   failed diagnostics.
6. **Which result is Figure-10-style valid?** Only the verdicts on the first
   two lines, under the resource-limited 32,768-scale preregistration.
7. **Which result is valid under original Route-3 strict rules?** The preserved
   historical `NO-GO`.
8. **What remains unsupported for real EEG?** All real Fpz–Cz parameter
   inference, because no independently validated source-to-sensor measurement
   model exists.

## Figures

- [Expected coverage and SBC](S4_sbi/results/figure10_8d_7d/figures/figure10_expected_coverage_sbc_8d_7d.png)
- [L-C2ST](S4_sbi/results/figure10_8d_7d/figures/figure10_lc2st_8d_7d.png)
- [Posterior predictive checks](S4_sbi/results/figure10_8d_7d/figures/figure10_ppc_8d_7d.png)
- [Posterior-predictive cortical-rate traces](S4_sbi/results/figure10_8d_7d/figures/figure10_ppc_cortical_rate_traces.png)
- [Posterior structure](S4_sbi/results/figure10_8d_7d/figures/figure10_posterior_structure_identifiability.png)
- [7D-vs-8D ablation](S4_sbi/results/figure10_8d_7d/figures/figure10_7d_vs_8d_ablation.png)

1. What has been established: the reported verdicts characterize raw NSF posterior quality within the frozen synthetic cortical-rate proxy space at the measured 32,768-simulation scale.
2. What remains unsupported: real-EEG inference, a validated Fpz–Cz measurement model, subject-specific physiology, and exact 3-million-simulation Practical Guide reproduction.
3. The single highest-priority next action: acquire GPU/HPC simulation capacity and repeat the frozen scale ladder through at least one million matched valid simulations without changing priors, summaries, or final diagnostics.
