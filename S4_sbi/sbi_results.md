# SBI Stage 2 Results

**Date**: 2026-05-07 09:03
**Subject**: SC4001

## x_obs values

| Summary | Value | Extraction method |
|---------|-------|-------------------|
| shape_r | 1.00000 | hardcoded 1.0 (EEG = reference) |
| T4_q | 2.64500 | SO peak Q-factor |
| T4_freq | 0.75000 | SO peak freq [Hz] |
| T6_ibi_cv | 0.77000 | UP-burst IBI CV |
| T8_n_sp_events | 14.35200 | spindle events per 60 s (normalized) |
| T11_lag_ms | 1.31900 | up_down_ratio (PAC) |
| MI | 0.00023 | PAC Modulation Index |

## Posterior MAP + 95% CI

| Parameter | MAP | CI_lo | CI_hi |
|-----------|-----|-------|-------|
| g_h | 0.05919 | 0.05885 | 0.08857 |
| g_LK | 0.05137 | 0.04849 | 0.06074 |
| c_ctx2th | 0.06917 | 0.06997 | 0.21479 |
| b | 42.36661 | 36.73719 | 42.54034 |

## SBC Results

SBC not run or failed.

## PPC Results

| Summary | x_obs percentile | Pass (5–95%) |
|---------|-----------------|-------------|
| shape_r | 100% | FAIL |
| T4_q | 98% | FAIL |
| T4_freq | 30% | PASS |
| T6_ibi_cv | 100% | FAIL |
| T8_n_sp_events | 1% | FAIL |
| T11_lag_ms | 6% | PASS |
| MI | 0% | FAIL |

## Wall-clock breakdown

- Round 1: 273.7 min
- Round 2: 144.3 min
- Round 3: 146.5 min