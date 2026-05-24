# Summary Stat Computation Consistency Check
**Date**: 2026-05-10  
**Scope**: V7 simulation side (`models/s4_personalize_fig7_v7.py`) vs SBI x_obs side (`S4_sbi/compute_xobs_from_eeg.py`)  
**7 dimensions**: shape_r · T4_q · T4_freq · T6_ibi_cv · T8_n_sp_events · T11_lag_ms · MI

---

## 1. Summary table

| Stat | V7 input signal | x_obs input signal | Formula match | Verdict |
|------|----------------|-------------------|--------------|---------|
| shape_r | r_ctx (FOOOF Pearson-r vs target) | hardcoded 1.0 | N/A | ⚠️ Design |
| T4_q | r_ctx Welch PSD | r_proxy Welch PSD | ✅ Identical | ⚠️ Signal differs |
| T4_freq | r_ctx Welch PSD | r_proxy Welch PSD | ✅ Identical | ⚠️ Signal differs |
| T6_ibi_cv | r_ctx, hard threshold 15 Hz | r_proxy 500ms-smoothed, 50th pctile | ❌ Different threshold | ❌ Mismatch |
| T8_n_sp_events | r_thal (10–14 Hz content real) | r_proxy (10–14 Hz content ≈ 0) | ✅ Identical | ❌ Mismatch |
| T11_lag_ms | r_ctx + r_thal PAC | r_proxy + r_proxy PAC | ✅ Identical | ⚠️ Known design |
| MI | r_ctx + r_thal PAC | r_proxy + r_proxy PAC | ✅ Identical | ❌ Severe mismatch |

---

## 2. Per-stat detailed analysis

### shape_r

**V7 side** (`compute_fitness_v7`, not in `compute_constraints_v7`):
```
FOOOF → periodic residual of simulated PSD → Pearson-r vs target EEG periodic component
Range: [0, 1]
```

**x_obs side** (`compute_summaries` line 237):
```python
d["shape_r"] = 1.0   # hardcoded sentinel
```

**Why**: EEG's own periodic component correlates perfectly with itself (by definition). Setting x_obs[shape_r]=1.0 anchors the SBI posterior toward parameters that maximize spectral shape fidelity.

**Is this a bug?** No. It is an intentional design choice. The SBI density estimator sees (simulator shape_r ∈ [0,1], x_obs shape_r = 1.0) pairs during training and learns to assign higher posterior probability to parameters that produce high shape_r.

**SBI PPC**: shape_r at 100th percentile → FAIL. This is expected: no simulator run achieves shape_r = 1.0 exactly. The posterior cannot literally satisfy this; it just pushes shape_r as high as possible.

---

### T4_q and T4_freq

**V7 side** (`compute_constraints_v7` lines 444–475):
```
SO band: SO_FREQ_LO=0.2 Hz, SO_FREQ_HI=1.5 Hz
so_width = 1.3 Hz
neighbor_lo: [max(0.1, 0.2-1.3), 0.2) = [0.1, 0.2) Hz
neighbor_hi: (1.5, 2.8] Hz
T4_q    = p_peak / mean(p_neighbor_lo + p_neighbor_hi)
T4_freq = freq of argmax(p_c[so_mask])
```

**x_obs side** (`compute_summaries` lines 241–258):
```python
# Exact same formula, constants, and neighbor logic
so_mask     = (f_c >= 0.2) & (f_c <= 1.5)
so_width    = 1.5 - 0.2  # = 1.3 Hz
neigh_lo    = (f_c >= max(0.1, 0.2-1.3)) & (f_c < 0.2)
neigh_hi    = (f_c > 1.5) & (f_c <= 2.8)
so_q        = so_peak_val / nbrs.mean()
so_peak_freq = f_c[so_mask][argmax(p_c[so_mask])]
```

**Formula verdict**: ✅ Byte-for-byte identical logic.

**Signal difference**: r_ctx is a zero-baseline spiking signal (~0 Hz in DOWN, ~30–60 Hz in UP peaks). r_proxy = abs(EEG) smoothed, rescaled to [0, 60]. Both are non-negative and have SO-scale oscillations, but their PSD shapes differ:
- r_ctx has a sharp SO peak due to periodic UP pulses
- r_proxy has a broader envelope due to the 50ms Gaussian smoothing

This explains why T4_q shows PPC failure at 98th percentile: the simulator SO peak is typically sharper (higher Q) than the EEG proxy's smoothed envelope.

---

### T6_ibi_cv (IBI coefficient of variation)

**V7 side** (`compute_constraints_v7` lines 495–514):
```python
# UP detection: hard absolute threshold on raw r_ctx
above = (r_ctx > UP_THRESH_HZ)    # UP_THRESH_HZ = 15.0 Hz
starts = where(diff(above) == 1)   # start of each UP run
intervals = diff(starts) / fs
ibi_cv = intervals.std() / intervals.mean()
```

**x_obs side** (`compute_summaries` lines 264–277):
```python
# UP detection: adaptive percentile threshold on ADDITIONALLY smoothed signal
r_for_ibi = gaussian_filter1d(r_proxy, sigma=500.0)   # extra 500ms smooth
up_thresh  = np.percentile(r_for_ibi, 50)              # 50th percentile
above = (r_for_ibi > up_thresh)
starts = where(diff(above) == 1)
intervals = diff(starts) / fs
ibi_cv = intervals.std() / intervals.mean()
```

**Differences** (two separate issues):

| Issue | V7 | x_obs |
|-------|-----|-------|
| Smoothing before threshold | None (raw r_ctx) | Extra 500ms Gaussian (σ=500 samples) |
| Threshold type | Absolute: 15 Hz | Adaptive: 50th percentile of smoothed signal |
| Detected UP rate | Sharp excursions above 15 Hz | Slow envelope half-crossings |

**Why the extra smoothing in x_obs?** r_proxy has sub-second amplitude fluctuations from spindles and other transients. Without extra smoothing, threshold crossings would split each SO UP cycle into many micro-bursts, destroying the IBI structure. The 500ms smooth extracts only the SO-scale envelope, giving one burst per SO cycle.

**Is this a bug?** The extra smoothing is justified. The threshold choice is not — using 50th percentile on r_proxy ≈ using median(envelope) as UP threshold. This gives ~50% duty cycle by construction, which may not match V7's UP/DOWN duty cycle (determined by physiological parameters, not percentile).

**Consequence**: IBI event detection will differ: V7 detects rapid r_ctx excursions above 15 Hz; x_obs detects slow envelope excursions above the median. The detected IBI intervals will have similar temporal structure but different counts.

**SBI PPC**: T6_ibi_cv at 100th percentile → FAIL. The simulator always produces IBI_CV > x_obs IBI_CV (0.77). This suggests the simulator IBI pattern is less regular than what x_obs detects, OR the detection method gives different CV values.

---

### T8_n_sp_events (spindle events per 60 s)

**Algorithm** (identical in both sides):
```
input_signal → butter([10, 14], band) → abs(hilbert) → gaussian(σ=200ms) →
percentile(75) threshold → events with duration [0.3, 2.0] s → count × 60/duration_s
```

**V7 side**: `input_signal = r_thal` (thalamic excitatory firing rate)  
**x_obs side**: `input_signal = r_proxy` (EEG amplitude envelope)

**Critical issue — r_proxy contains no 10–14 Hz spindle content.**

r_proxy = abs(EEG) after 50ms Gaussian smoothing. The Gaussian with σ=50 samples at 1000 Hz has frequency response:

```
H(f) = exp(-2π²σ²f²/fs²) = exp(-2π²×50²×f²/1000²)
H(10 Hz) ≈ 0.0014    (−57 dB)
H(12 Hz) ≈ 0.0008    (−62 dB)
```

Spindle-band power is attenuated by >99.9% before T8 detection begins. Applying `butter([10, 14])` to r_proxy yields essentially band-limited noise.

**What T8 actually detects in x_obs**: The 10–14 Hz bandpass output of r_proxy is noise. The 75th percentile threshold is very small. The duration filter [0.3, 2.0 s] admits correlated noise bursts that happen to last 0.3–2 s. The "spindle count" is a noise statistic, not physiological spindles.

**What T8 detects in V7**: r_thal has real 10–14 Hz T-current resonance oscillations (physiological spindles from thalamic relay cells). The count is the number of genuine spindle bursts.

**Scale mismatch**: x_obs T8 = 14.35 events/60 s. Simulator T8 for Seed B = ~17–21 events/60 s. SBI PPC T8 at 1st percentile → FAIL: the simulator almost always exceeds x_obs T8.

**Is this a bug?** Yes, in the methodological sense. The proxy signal construction (50ms smooth) makes it incompatible with spindle-band detection. **The x_obs T8 value does not measure the same physical quantity as the simulator T8.**

---

### T11_lag_ms (up_down_ratio) and MI

Both use `compute_pac_metrics_fixed` (same function on both sides).

**V7 side**:
```
r_ctx → find_peaks → cycle-by-cycle phase
r_thal → butter([10, 14]) → abs(hilbert) → spindle amplitude
```

**x_obs side**:
```
r_proxy → find_peaks → cycle-by-cycle phase   (r_ctx role)
r_proxy → butter([10, 14]) → abs(hilbert) → spindle amplitude   (r_thal role)
```

**Phase extraction**: Uses `find_peaks(r_proxy, distance=0.7s×fs, prominence=0.3×range)`. r_proxy has clear slow-wave peaks corresponding to N3 SO UP states. Phase extraction here is sound.

**Spindle amplitude extraction**: Same issue as T8 — `butter([10, 14])` on r_proxy extracts noise, not real spindles.

**T11_lag_ms (up_down_ratio)**: Numerically, the coupling between r_proxy UP peaks (phase) and r_proxy bandpass noise (amplitude) will show some positive up_down_ratio by construction: the SO peaks in r_proxy will drive transient amplitude fluctuations in the (noise-dominated) bandpass signal. x_obs T11 = 1.319. SBI PPC at 6th percentile → marginal PASS.

**MI**: x_obs MI = 0.00023. This is extremely small. Simulator MI values: Seed A ≈ 0.10, Seed B ≈ 0.06. SBI PPC MI at 0th percentile → FAIL. The simulator never produces MI as small as x_obs MI. This is direct evidence that x_obs MI measures noise coupling, not physiological PAC.

**Is this a bug?** Partial. The x_obs docstring documents this limitation:
> "因此，MI和up_down_ratio的尺度将与仿真得到的值不同；SBI的密度估计器会学习这种映射关系。"

But if x_obs MI = 0.00023 is OUTSIDE the simulator's distributional support, SBI cannot learn any mapping — it would extrapolate. The PPC MI = 0% confirms extrapolation.

---

## 3. Known differences (by design)

These are documented or philosophically justified:

1. **shape_r = 1.0 sentinel**: EEG against itself = perfect match. Anchors posterior toward high shape_r. Not a bug.

2. **r_proxy as both ctx and thal for PAC**: "丘脑活动在头皮EEG不直接可见，这是一种技术性简化。" Acknowledged in docstring.

3. **Different input signals overall**: The entire proxy pipeline (detrend → abs → 50ms smooth → rescale) is an intentional bridge between EEG (bipolar µV, signed) and r_ctx (firing rate Hz, non-negative). The mapping is approximate by design.

---

## 4. Genuine mismatches (may need fixing)

### ❌ Critical: T8 spindle detection in x_obs

**Problem**: The 50ms Gaussian in `build_rate_proxy` eliminates all 10–14 Hz content. T8 detection on r_proxy returns a noise-based count, not physiological spindles.

**Evidence**: SBI PPC T8 at 1st percentile — the simulator almost always exceeds x_obs T8.

**Possible fixes** (not implemented, flagged for discussion):
- Option A: Extract T8 directly from raw EEG (before abs/smooth) using the same butterworth-hilbert-threshold pipeline. This gives genuine EEG spindle counts.
- Option B: Replace x_obs T8 with a count from a spindle detector applied to the raw EEG voltage (e.g., YASA or a simple 10–14 Hz detector on the original EEG signal before rectification).
- Option C: Accept the mismatch and exclude T8 from x_obs, using only 6 dimensions.

### ❌ Critical: MI nearly zero in x_obs

**Problem**: x_obs MI = 0.00023 is outside the simulator's distributional support (minimum simulated MI ≈ 0.02–0.05). SBI cannot learn a mapping because there is no overlap.

**Evidence**: SBI PPC MI at 0th percentile.

**Root cause**: Same as T8 — r_proxy has no 10–14 Hz content, so the "spindle amplitude" in PAC computation is noise. A noise amplitude uncorrelated with SO phase yields MI ≈ 0.

**Possible fix**: Compute x_obs MI using raw EEG spindle envelope (before rectification) instead of r_proxy bandpass.

### ❌ Moderate: T6 threshold mismatch

**Problem**: V7 uses a hard 15 Hz threshold for UP detection; x_obs uses the 50th percentile of a 500ms-smoothed signal. These detect different numbers of UP events and produce IBI intervals with different distributions.

**Evidence**: SBI PPC T6 at 100th percentile — simulator IBI CV always exceeds x_obs IBI CV.

**Possible fix**: Align detection threshold. Since r_proxy is always in [0, 60], a hard threshold at 30 Hz (50% of max range) would more closely mirror the V7 15 Hz threshold's role as a midpoint between DOWN (~0) and UP (~60) states. The 500ms extra smooth is still needed and appropriate.

---

## 5. Code changes needed?

| Fix | Priority | Effort | Expected impact |
|-----|----------|--------|-----------------|
| T8: detect from raw EEG (before abs/smooth) | High | Medium | PPC T8 likely to pass |
| MI/T11: compute PAC from raw EEG spindle envelope | High | Medium | PPC MI likely to pass; T11 may improve |
| T6: align UP detection threshold | Medium | Low | PPC T6 may improve |
| shape_r: leave as 1.0 | None | — | By design, no change needed |

**Recommendation**: Before modifying code, verify the PPC failures are indeed caused by these mismatches (not SBI underfitting or prior misspecification) by running a diagnostic: compute the same T8 and MI on the raw EEG signal before the proxy transformation, and compare with x_obs values.
