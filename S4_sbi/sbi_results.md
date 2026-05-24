# SBI Stage 2 Results

**Date**: 2026-05-23 04:34
**Subject**: SC4001

## x_obs values

| Summary | Value | Extraction method |
|---------|-------|-------------------|
| shape_r | 1.00000 | fixed 1.0 (EEG reference) |
| T4_q | 2.64500 | SO peak Q-factor |
| T4_freq | 0.75000 | SO peak frequency [Hz] |
| T8_n_sp_events | 15.31000 | spindle events per 60 s (from eeg_raw) |
| T11_lag_ms | 1.28000 | up_down_ratio (PAC) |

## Posterior MAP + 95% CI

| Parameter | MAP | CI_lo | CI_hi |
|-----------|-----|-------|-------|
| g_h | 0.07641 | 0.04939 | 0.07973 |
| g_LK | 0.04914 | 0.04220 | 0.05687 |
| c_ctx2th | 0.20636 | 0.05193 | 0.21358 |
| b | 50.52520 | 49.48069 | 50.76123 |

## SBC Results

SBC not run or failed.

## PPC Results

| Summary | x_obs percentile | Pass (5–95%) |
|---------|-----------------|-------------|
| shape_r | 100% | FAIL |
| T4_q | 100% | FAIL |
| T4_freq | 77% | PASS |
| T8_n_sp_events | 11% | PASS |
| T11_lag_ms | 37% | PASS |

## Wall-clock breakdown

- Round 1: 185.6 min
- Round 2: 88.3 min
- Round 3: 90.6 min
- Round 4: 90.9 min