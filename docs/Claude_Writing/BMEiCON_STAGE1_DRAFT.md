# Spectrally Similar, Dynamically Invalid: A Single-Subject Diagnosis of Loophole Exploitation in Neural Mass Model Fitting

**Draft status:** BMEiCON 2026 Stage 1 manuscript draft — **Revision Round 1** (responds to internal 5-reviewer panel).  
**Scope:** Single-subject diagnostic study; Stage 2 SBI is reserved for a future journal paper.  
**Citation status:** All references verified (author / venue / year / DOI confirmed against the publisher record, June 2026); inline `[REF: key]` tags map to the verified reference list. Two entries carry an author-decision note (Robinson model choice; neurolib citation year).  
**Revision note:** Title set to "Dynamically **Invalid**" (from "Wrong" → briefly "Unverified" → "Invalid"). This reflects the V1 audit (Sec. V-A): the canonical V1 point — `data/patient_params_fig7_v1_0418_2_SC4001.json`, the `shape_r=0.8586` solution the paper cites — passes only 5/12 event-level criteria, with zero SO–spindle coupling (MI = 0) and no sustained UP states. "Invalid (under explicit audit)" is the evidence-backed term; "Wrong" is avoided as a conservative choice, since the held-out evidence is *mixed* (V7 is closer than canonical V1 on SO waveform morphology, but V1 is closer on cortical spindle density), so we do not claim either model is the more faithful one. All held-out numbers use the canonical V1 point. Items still needing new analysis are marked `[TODO-DATA: ...]`; references are now verified (see the References section). Audit artifacts: `validation_outputs/v1_v3_v7_t12_audit.{csv,_summary.txt}`, `validation_outputs/fig_v1_v3_v7_t12_audit_timeseries.png`; held-out artifacts: `validation_outputs/so_waveform_heldout_canonicalV1_summary.txt`, `validation_outputs/spindle_density_confirm_summary.txt`.

---

## Abstract

Personalized neural mass models for sleep EEG are commonly fitted with spectral objectives. We show that such objectives can select solutions that are spectrally similar to a target recording while leaving the intended sleep dynamics *unverified* — and, when those dynamics are explicitly audited, invalid — a failure mode we call *shape-alike, spirit-unalike* (形似神不似). Using one Sleep-EDF subject (SC4001, N3) and a neurolib thalamocortical model combining an ALN cortical node with a thalamic node, we diagnose this failure through an iterative V1-V7 fitting history. A spectral-only baseline achieved high residual spectral similarity (`shape_r=0.8586`) but provided no event-level guarantees; auditing that same V1 solution against the 12 event-level criteria used later shows it passes only 5/12, failing sustained UP-state (T3), SO sharpness and regularity (T4, T6), spindle burstiness (T7), and the entire SO-spindle coupling triad (T9-T11, modulation index = 0), making the "spectrally similar yet dynamically invalid" claim concrete on a single solution rather than inferred from the version history. Subsequent versions exposed optimizer loopholes including persistent-DOWN solutions, SO-spindle decoupling, fake spindle transients, and unreachable FOOOF-based spindle rewards. V7 replaced those failure-prone rewards with event-level constraints and rewards, increasing feasible solutions from 1/4960 in V6 to 364/4960 and producing a selected point that passed all 12 event-level constraints. Because reward targets, parameter bounds, simulation duration, and one spindle-width threshold changed together across versions, we report this feasibility trajectory as confounded evidence and quantify the threshold contribution separately: only 156 of V7's 364 feasible points survive V6's stricter spindle-width rule, so roughly half of the feasibility recovery reflects threshold relaxation rather than loophole closure. Two held-out checks, both computed on the canonical V1 point, give mixed rather than uniform evidence: on slow-oscillation average-cycle waveform morphology V7 is closer to real EEG than V1 (RMSE 0.39 vs 0.57), but on cortical spindle density V1 is closer — V7 produces essentially zero spindles in the EEG-analog cortical signal across both 10-14 and 11-15 Hz bands, because it passes the spindle audit on the thalamic node while its small thalamocortical coupling leaves no spindles in cortex. We do not claim V7 is better than V1: V1 is invalid under the audit (5/12) and V7 fails the cortical spindle check, so neither solution is correct and neither held-out check is decisive. We therefore frame the result as a single-subject diagnostic existence claim: spectral objectives can leave personalized models dynamically unverified, an explicit audit can expose invalidity that the spectrum hides, and even a closed audit can retain residual loopholes — treating the optimizer as a loophole-searcher reveals all three.

**Keywords:** sleep EEG, neural mass model, thalamocortical model, spectral fitting, slow oscillation, sleep spindle, optimization, specification gaming

---

## I. Introduction

Personalized neural mass models are attractive tools for building mechanistic "digital twins" of brain rhythms. In sleep modeling, the target signals are often summarized spectrally: N3 sleep is characterized by slow oscillations, delta activity, and sleep spindles, and a fitted model is commonly judged by whether its simulated power spectrum resembles the EEG target [REF: Cakan2022-or-2023] [REF: Jajcay2022] [REF: SleepEDF]. Spectral matching is useful because it compresses noisy electrophysiological time series into stable rhythm-level summaries. However, it can also hide mechanistic errors. Two signals may share similar spectral peaks while differing in whether they contain real UP/DOWN alternation, waxing-waning spindle events, or coordinated slow-oscillation-spindle coupling.

This paper diagnoses that failure mode in one subject: Sleep-EDF SC4001 during N3 sleep. We call the failure *shape-alike, spirit-unalike* (形似神不似): the simulated spectrum resembles the target, but the underlying dynamics do not implement the intended sleep physiology. The study is deliberately scoped as a single-subject diagnostic and existence claim. It does not establish a population-level fitting method, nor does it claim that the final model is globally more EEG-like on every metric. Instead, it shows that a standard spectral objective can select a dynamically invalid solution — one that fails an explicit event-level audit — and that the iterative history of failed optimizers is itself evidence of objective loopholes.

The key framing is to treat the optimizer as an adversarial loophole-searcher. Differential evolution does not know the intended physiology; it only searches for parameter settings that satisfy the objective it is given. If the objective rewards spectral similarity without event-level safeguards, or if a constraint admits a loophole, the optimizer can exploit it. Our V1-V7 development history repeatedly exposed such failures: spectral-only fitting, weak dynamics constraints, PAC-free spindle solutions, transient events masquerading as spindles, and FOOOF-based rewards that were unreachable in the SO-stable regime. V7 is the result of closing those specific loopholes with event-level constraints and redesigned rewards.

In estimation terms, this is a statement about objective misspecification and non-identifiability: a spectral summary does not uniquely determine the underlying dynamics, so an objective built on it is satisfiable by dynamically distinct solutions. Our contribution is not that observation, which is well known, but its *iterative operationalization* — treating differential evolution as an adversary and using each feasibility collapse to localize a specific objective loophole. The principle generalizes to any simulator fitted to summary statistics; we keep the evidence single-subject by design.

The contributions are:

1. We demonstrate, in a single SC4001 N3 case study, that high spectral similarity can coexist with dynamically invalid fitted solutions, and we make this concrete by auditing the V1 spectral-only solution against the same event-level criteria (Sec. V-A).
2. We present the V1-V7 fitting history as an empirical loophole audit of the objective — an adversarial red-teaming protocol — rather than as an implementation footnote.
3. We report both the positive event-level result and two independent held-out checks that give mixed evidence (V7 closer on SO waveform morphology, V1 closer on cortical spindle density), narrowing the claim to event-level loophole closure rather than global superiority — and we surface a residual loophole that survives inside the closed V7 audit.

---

## II. Related Work

### A. Neural mass fitting for sleep rhythms

Neural mass and mean-field models have been used to reproduce macroscopic sleep rhythms, including slow oscillations and spindle-band activity [REF: Cakan2022-or-2023] [REF: Jajcay2022] [REF: RobinsonSleepModel]. These models provide mechanistic structure beyond descriptive EEG statistics: parameters can represent cortical excitability, adaptation, thalamic conductances, and thalamocortical coupling. Prior work has often emphasized reproduction of rhythm-level or spectral phenomena. Our work is complementary but more diagnostic: we ask whether spectral matching is sufficient as an objective for personalization, and we show that it can select solutions that satisfy spectral appearance while failing event-level dynamics.

### B. Slow oscillation-spindle physiology

N3 sleep is not defined by a spectrum alone. Slow oscillations involve alternating DOWN and UP states in cortical activity [REF: SteriadeSO] [REF: MassiminiSO], while spindles appear as transient waxing-waning sigma-band events associated with thalamocortical circuitry [REF: Molle2011] [REF: Helfrich2018]. Coupling between slow-oscillation phase and spindle amplitude has been studied as a marker of memory consolidation and sleep physiology [REF: Tort2010] [REF: Mander2015]. These phenomena motivate event-level constraints: a model should not merely have power in the correct bands; it should produce plausible events with temporal structure. The event metrics used here are deliberately aligned with established detection conventions rather than invented: our spindle event count and duration follow amplitude/duration criteria in the spirit of standard detectors [REF: Molle2011] [REF: Ferrarelli2007] and open implementations such as YASA [REF: Vallat2021-YASA], and our SO criteria follow zero-crossing/amplitude conventions for slow oscillations [REF: SteriadeSO] [REF: MassiminiSO]. The thresholds attached to these metrics are empirical and subject-specific (Sec. III-D); the metric *forms* are not.

### C. Optimizer loopholes and specification gaming

The failure studied here resembles specification gaming in machine learning: an optimizer exploits the literal objective rather than the designer's intended goal [REF: SpecificationGaming] [REF: RewardHacking]. In neural model fitting, a candidate can score well by exploiting spectral artifacts, overly broad rewards, or incomplete constraints. This paper imports that adversarial-optimizer framing into personalized sleep modeling. The claim is not that the optimizer is malicious, but that objective functions define incentives; if the incentive misses event-level physiology, the search can find shape-alike but dynamically invalid solutions. We are aware that, statically, this is the familiar problem of model/objective misspecification and parameter non-identifiability. The framing earns its place by converting a static property into a *procedure*: each time a patched objective is re-attacked and feasibility collapses, the collapse localizes the next loophole. The reusable output is therefore an audit workflow, not the (already accepted) conclusion that constraints help.

---

## III. Methods

### A. Data and target spectrum

We used Sleep-EDF subject SC4001 and selected N3 sleep epochs. The target EEG preprocessing follows the project pipeline: EEG epochs are aligned to the hypnogram, 30 s N3 epochs are extracted, epochs with peak-to-peak amplitude above 200 uV are rejected, and Welch PSD estimates are averaged across accepted epochs (channel `EEG Fpz-Cz`; 57 accepted N3 epochs enter the target PSD, per the stored fit records). The spectral fitting range is 0.5-20 Hz. (The held-out SO-waveform check of Sec. III-E uses a separate, larger pool of 142 clean N3 epochs under its own QC, and is independent of this target PSD.) Throughout, we use "slow oscillation" (SO) for the < ~1.25 Hz band that carries the UP/DOWN alternation and "delta" for the ~1-4 Hz band; the V7 selected point reports `T4_freq = 1.25 Hz`, which sits at this SO/delta boundary and is treated as SO here.

For the spectral similarity term, the target PSD is decomposed using FOOOF/specparam-style aperiodic fitting [REF: FOOOF]. The comparison is not raw PSD correlation. Instead, both target and simulated PSDs are converted to log-domain residual spectra after subtracting the fitted aperiodic background. The `shape_r` score is the Pearson correlation between those residual spectra after aligning the simulated PSD to the target FOOOF frequency grid. This makes the spectral objective more rhythm-specific than raw power matching, but the V1 result shows that even this residual spectral objective is not sufficient to guarantee correct dynamics.

### B. Thalamocortical model and free parameters

The simulator is a two-node thalamocortical mean-field model implemented with neurolib's `MultiModel`: an ALN cortical node coupled to a thalamic node. The fitting procedure searches over eight parameters:

`mue`, `mui`, `b`, `tauA`, `g_LK`, `g_h`, `c_th2ctx`, and `c_ctx2th`.

These include cortical background inputs and adaptation parameters, thalamic TCR conductances, and directed thalamocortical / corticothalamic coupling strengths. Each candidate is simulated for 60 s at 1 kHz; the first 5 s are discarded as burn-in before any metric is computed. The search uses `scipy.optimize.differential_evolution` (`strategy='best1bin'`, population size 20, 30 generations), and all stochastic components (Ornstein-Uhlenbeck noise and node initialization) are seeded deterministically (seed 42) so that metrics are reproducible across calls with identical parameters. Each reported version corresponds to a single DE run under one seed; we therefore treat per-version feasible *counts* as point estimates without run-to-run variance, and we caution against over-reading small differences (Sec. V-B).

### C. V7 fitness and feasibility

The V7 feasible-solution reward is:

```text
fitness = 0.50 * shape_r + 0.25 * so_power + 0.25 * spindle_power
```

where `shape_r` is the FOOOF residual spectral correlation described above. Unlike earlier versions, V7 computes `so_power` and `spindle_power` from event-level quantities:

```text
so_power      = clip((T4_q - 1) / 4, 0, 1)
spindle_power = clip(T12_n_verified / 15, 0, 1)
```

The denominators in these expressions (4 for `so_power`, 15 for `spindle_power`) are empirical reward-saturation scales, not literature-derived normative targets. In particular, 15 is simply the spindle-event count at which `spindle_power` saturates to 1 over the 60 s window; it was chosen so the reward keeps discriminating among candidates across the range of counts the optimizer actually produces, and it should be read purely as a normalization factor internal to the objective. It is **not** a claim that physiological N3 contains 15 spindles per 60 s window: the held-out empirical density measured here (≈ 2.4-3.0 events/min for real SC4001 EEG, Sec. III-E) is far lower, so the constant is deliberately set well above realistic event counts to keep the gradient informative rather than to encode a target rate.

This creates partial overlap between the reward and constraints. We therefore do not treat the V7 fitness score as held-out validation. Instead, the paper uses the version history and feasibility audit as diagnostic evidence, and separately reports a held-out SO waveform check as a limitation.

Infeasible points are ranked by continuous constraint satisfaction rather than by a single cliff penalty. This preserves a search gradient for differential evolution: candidates that nearly satisfy the event-level audit can be distinguished from candidates that fail all checks, while all feasible solutions remain ranked above infeasible ones.

### D. Event-level constraints

V7 uses 12 binary constraints grouped into four physiological categories:

1. **Cortical bistability:** DOWN state, UP state, and sustained UP duration (T1-T3).
2. **Slow-oscillation structure:** SO peak sharpness/frequency and inter-burst interval regularity (T4, T6).
3. **Spindle reality:** spindle spectral width, envelope burstiness, event count/duration, and event-internal sigma verification (T5, T7, T8, T12).
4. **SO-spindle coordination:** PAC strength, phase preference, and up/down ratio / lag-like coupling metric (T9-T11).

The metric forms are physiologically motivated, but the thresholds are empirical and subject-specific. This distinction is important. The paper does not claim that every threshold is literature-derived. Rather, the iterative threshold history is part of the evidence: the optimizer repeatedly exposed how apparently reasonable objectives admitted unintended solutions.

Two points guard against the reading that constraints were simply tuned until V7 passed. First, each constraint is motivated by a failure observed in an *earlier* version, independently of whether the final V7 point happens to satisfy it: the UP-state tests (T1-T3) respond to persistent-DOWN solutions seen in V2-V3; the PAC three-pack (T9-T11) responds to SO-spindle decoupling seen in V5; and the event-internal sigma verification (T12) responds to fake spindle transients seen in V6. The motivating diagnostic for each precedes the V7 reward redesign and does not depend on its outcome. Second, the PAC metrics (T9-T11) are computed with a convention-free, cycle-by-cycle estimator (`compute_pac_metrics_fixed`) rather than the bandpass-plus-Hilbert phase used in early V7 prototypes, which systematically biased the preferred SO phase for spike-like firing-rate signals; we therefore report PAC strength as a constraint but avoid preferred-phase point claims in this manuscript.

### E. Held-out SO waveform check

To test whether event-level feasibility also improved an independent waveform statistic, we implemented a held-out slow-oscillation average-cycle morphology check. Real SC4001 N3 EEG and simulated cortical activity were bandpassed to the SO range, aligned to SO cycle anchors, normalized, averaged into templates, and compared using RMSE and correlation. This check does not call the V7 constraint function and does not reuse T1-T12 labels, `shape_r`, `so_power`, or `spindle_power`.

Because the SO average-cycle template primarily probes low-frequency cycle shape and can look unremarkable even for degenerate event dynamics, we add a second held-out statistic that is event-level yet still disjoint from the T1-T12 audit: empirical spindle density (events per minute), measured with a standard sigma-band RMS detector (Moelle/Ferrarelli convention: band-pass to sigma, moving-RMS envelope, threshold = mean + 1.5·SD, accept events of 0.5-3.0 s). This detector is deliberately disjoint from the fitness machinery: it operates on the *cortical* (EEG-analog) signal, whereas T8/T12 operate on the *thalamic* signal; it uses a moving-RMS envelope and mean+SD threshold, whereas T8 uses a 75th-percentile Gaussian-smoothed Hilbert envelope and T12 adds Welch peak-inside verification; and it uses 0.5-3.0 s duration bounds rather than T8's 0.3-2.0 s. We apply it identically to real EEG and to the simulated cortical firing rate of V1 and V7, over 300 s simulations and across two sigma bands (10-14 Hz, the model's T-current resonance, and 11-15 Hz, the EEG sigma band), so the result is robust to duration and band choice. Using a held-out statistic that is *able* to favor V7 (rather than only one that favors V1) is what lets the event-level result be read in either direction; as shown below, the two held-out statistics disagree — V7 is closer on SO morphology, V1 is closer on spindle density — so neither is decisive and the manuscript's claim remains the narrower "invalid under explicit audit" rather than "more EEG-like."

---

## IV. Experiments

### A. V1-V7 as an experimental design

The experiment is not a single final optimization run. The evidence lies in the V1-V7 sequence, where each version exposed a new objective loophole.

**V1: spectral-only fitting.** The initial Fig. 7-style baseline used FOOOF residual spectral similarity and SO/spindle spectral rewards. It reached high spectral similarity (`shape_r=0.8586`) but imposed no event-level guarantees. This is the cleanest exhibit for the claim that spectral fit can look good without proving mechanistic validity.

**V2-V3: persistent-DOWN loophole.** Weak dynamics checks could be satisfied by solutions with inadequate UP-state behavior. V3 added explicit UP-state existence and sustained-UP tests, closing a persistent-DOWN loophole.

**V4: constraints separated from rewards.** V4 separated feasibility constraints from reward ranking and introduced SO sharpness, SO regularity, and spindle envelope burstiness checks. It produced feasible solutions, but only rarely: 22/4703 evaluations.

**V5: PAC constraints.** V5 added a three-part PAC audit to prevent SO-spindle decoupling. However, the best solution still had `spindle_power=0`, showing that PAC constraints alone did not guarantee robust spindle rewards.

**V6: fake-spindle exposure.** V6 fixed T8 event detection and added T12 to verify that detected events contain sigma-dominant oscillations. This made the search stricter but exposed another failure: only 1/4960 evaluations were feasible.

**V7: reward and bound redesign.** Diagnostics showed that earlier FOOOF spindle rewards were effectively unreachable in SO-stable regimes, that 30 s simulations made T6 noisy, that T5's original FWHM threshold rejected mechanistically valid narrow spindles, and that large `c_th2ctx` destabilized cortical SO. V7 doubled simulation duration to 60 s, replaced FOOOF SO/spindle rewards with T4/T12-derived event rewards, tightened `c_th2ctx`, and relaxed T5 FWHM to admit narrow T-current resonance peaks.

### B. Evaluation measures

We report:

1. Best score and `shape_r` for each version.
2. Feasible rate for versions with explicit feasibility rules.
3. V7 final T1-T12 metrics.
4. Held-out SO waveform morphology distance.

The held-out SO waveform result is not used to select V7. It is included to test and constrain the interpretation of V7's event-level improvements.

---

## V. Results

### A. Spectral fit alone can look successful

The V1 spectral-only baseline achieved a high best `shape_r=0.8586`. This confirms that the optimizer can find a solution whose FOOOF residual spectrum resembles the target EEG. However, V1 did not enforce UP/DOWN alternation, spindle event structure, event-internal sigma verification, or SO-spindle coupling.

To convert this from "unchecked" to "demonstrably invalid," we evaluated the canonical V1 selected parameter vector (`data/patient_params_fig7_v1_0418_2_SC4001.json`, the `shape_r=0.8586` point) through the *same* event-level audit later formalized as V7's T1-T12, using the identical 60 s simulation, 5 s burn-in, and seed (42). The V1 point passes only **5/12** constraints. It retains a DOWN state (T1) and a momentary UP excursion (T2, max cortical rate 38.6 Hz), but fails: **T3** — the longest sustained UP state is 86 ms, below the 100 ms threshold, so the UP/DOWN alternation is spiky rather than bistable; **T4** — the SO spectral peak has Q = 1.68 (< 2.0), i.e. no sharp slow-oscillation peak; **T6** — inter-burst-interval CV = 1.03 (≫ 0.4), an irregular slow rhythm; **T7** — spindle-envelope CV = 0.52 (< 0.7), not waxing-waning; and **T9-T11** — the modulation index is exactly 0, i.e. a *complete absence* of slow-oscillation-spindle coupling. The corresponding cortical time series is shown in Fig. 3 (source: `validation_outputs/fig_v1_v3_v7_t12_audit_timeseries.png`). This is the paper's existence object: a single solution that is spectrally similar to the target (`shape_r=0.8586`) yet absent two core N3 phenomena — sustained UP states and SO-spindle coupling. (Re-running V1 under V7's audit is a re-scoring of a stored solution, not a re-fit; it adds no degrees of freedom; the harness reproduces V7's stored 12/12 exactly, confirming fidelity.) Thus V1 is not merely unvalidated — under an explicit audit it is invalid on named criteria — while remaining spectrally convincing. For reference, the V3 dynamics point likewise passes 5/12 (failing T4, T6, T7, T9-T11, T12), so the spectral-versus-event gap is not unique to V1.

### B. Feasibility changed non-monotonically across versions

The feasible-rate trajectory supports the loophole-closure interpretation:

| Version | Best score | Best `shape_r` | Feasible rate | Interpretation |
|---|---:|---:|---:|---|
| V1 | 0.8586 | 0.8586 | N/A | Spectral objective can look good without event guarantees |
| V3 | 0.7722 | 0.8334 | N/A | UP-state tests close persistent-DOWN loophole |
| V4 | 0.4481 | 0.5449 | 22/4703 | Event feasibility introduced but rare |
| V5 | 0.5296 | 0.5867 | 5/4960 | PAC constraints added; spindle reward still problematic |
| V6 | 0.3080 | 0.6159 | 1/4960 | Fake-spindle loophole exposed |
| V7 | 0.7759 | 0.6226 | 364/4960 | Event-level loopholes closed much more often |

The non-monotonic pattern matters. V5 and V6 did not simply improve over earlier versions; they became stricter and revealed new failures. V7 recovered feasibility only after the reward targets and bounds were redesigned according to those diagnostics.

Two caveats keep this trajectory honest. First, the feasible *rate* is not a clean measure of loophole closure, because between V6 and V7 four things changed at once: simulation duration (30 -> 60 s, which reduces T6 sampling noise), the `c_th2ctx` bound (tightened), the SO/spindle reward definitions (FOOOF-based -> reachable event-based), and the T5 spindle-width threshold (relaxed from > 2.0 Hz to > 0.2 Hz, which mechanically admits narrow T-current resonance peaks previously rejected). The last change in particular loosens a gate, so part of the V6 -> V7 recovery is threshold relaxation rather than the optimizer finding genuinely better physiology. We therefore re-score the V7 evaluation population under V6's original T5 threshold to isolate this contribution: of V7's 364 feasible points, only 156/4960 remain feasible under V6's strict `FWHM > 2.0 Hz` rule, so 208 (57%) of the feasibility recovery is attributable to the T5 relaxation alone, leaving 156 attributable to the duration/reward/bound redesign. The point is sharpened by the V7 selected solution itself, whose spindle FWHM is exactly 2.0 Hz: under V6's strict threshold it would be infeasible, confirming that T5 is load-bearing rather than incidental. (This re-scoring uses the per-evaluation `T5_fwhm` logged in `outputs/evolution_fig7_v7_records.csv`; the 25 points at exactly 2.0 Hz, including the selected point, are counted as lost under the code's strict `>` and would survive only under a non-strict `>=`.) Second, because each version is a single seeded DE run, the small absolute counts are within plausible run-to-run variation; in particular the V5-vs-V6 difference (5 vs 1) should not be interpreted as a meaningful change, and we draw conclusions only from the order-of-magnitude V6 -> V7 shift after de-confounding.

### C. V7 selected point passed the event-level audit

The selected V7 point passed all 12 constraints (`n_passed=12`) with:

- `score=0.775878`
- `shape_r=0.62263`
- `T4_q=4.433`
- `T4_freq=1.25 Hz`
- `T6_ibi_cv=0.197`
- `T8_n_sp_events=25`
- `T12_n_verified=20`

This result should be interpreted carefully. V7's `shape_r` is lower than V1's spectral-only score, so V7 is not a better spectral fit. The comparison is also not on identical ground: V1 searched broad bidirectional coupling bounds, whereas V7 tightened `c_th2ctx` for SO stability, so part of the `shape_r` gap reflects a smaller, more physiologically constrained search space rather than a pure spectral/event trade-off. With that caveat, V7's advantage is event-level feasibility: the final point satisfies the explicit audit for cortical bistability, SO regularity, spindle event structure, and SO-spindle coordination.

### D. Held-out checks give mixed evidence

We evaluated two held-out statistics, both on the canonical V1 point (`shape_r=0.8586`), neither of which calls the V7 constraint function or reuses T1-T12 / `shape_r` fields.

**(1) SO average-cycle waveform morphology.** Using 142 clean SC4001 N3 epochs from `EEG Fpz-Cz`, the real template comprised 3881 SO snippets. Distances to the real template were:

| Version | RMSE | Correlation |
|---|---:|---:|
| V1 canonical (`shape_r=0.8586`) | 0.5685 | 0.8390 |
| V3 dynamics | 0.3424 | 0.9414 |
| V7 event-constrained | 0.3909 | 0.9234 |

On this metric the canonical V1 point is the *worst* of the three, and V7 is closer to real EEG than V1 (RMSE 0.39 vs 0.57). (An earlier internal run had used a different, non-canonical spectral-only point that happened to be closest to the real template; we report the canonical V1 that the rest of the paper cites, for consistency. V3 and V7 reproduce across both runs.)

**(2) Cortical spindle density (Sec. III-E).** Real SC4001 N3 EEG has a cortical spindle density of 2.41/min (10-14 Hz) to 2.96/min (11-15 Hz); the canonical V1 point reaches 1.22-1.42/min — about half the real value — while V7 produces **zero** cortical spindles in both bands over a 300 s simulation. On this metric V1 is closer to real EEG than V7.

The two held-out checks therefore disagree: V7 is closer on SO morphology, V1 is closer on spindle density. Neither is decisive, and the evidence is mixed rather than uniformly pro-V1 or pro-V7. We read this conservatively. The SO average-cycle metric is narrow — it normalizes amplitude and probes only low-frequency cycle shape — so V7's edge there does not establish global fidelity; and V7's complete absence of cortical spindles is a real deficiency the audit did not prevent (Sec. VI-F). We therefore retain only the audited-invalidity claim (Sec. V-A) and disclaim any global ranking. This is why the title settles on "Invalid" (under explicit audit) rather than "Wrong": the audit demonstrates that the spectrally convincing V1 point violates named event-level criteria, but the held-out evidence does not support the stronger claim that either model is the more faithful one. V1 is not thereby validated — it passes only 5/12 audit checks and under-produces cortical spindles by about half — and V7, though it passes all 12, fails the cortical spindle check. Neither solution is correct; the spectral objective and the event audit each select a differently-deficient model.

---

## VI. Discussion and Limitations

### A. Main interpretation

The central result is not that V7 is globally superior to V1. The central result is that V1 demonstrates the danger of spectral fitting: a high residual spectral score can coexist with missing event-level guarantees. The V1-V7 sequence then shows how the optimizer exploited each objective or constraint loophole until a more explicit event-level audit was introduced.

This framing changes how the iterative history should be read. The post-hoc nature of several constraints is not something to hide. Under the adversarial-optimizer framing, it is evidence: each constraint was introduced because an apparently reasonable previous objective admitted an unintended solution. The result is a diagnostic methodology for discovering objective failures in personalized neural mass fitting.

### B. Single-subject scope

This is a single-subject case study. It demonstrates that the failure mode exists for SC4001 N3, not that the same thresholds or final parameters generalize across subjects. A population-level analysis would require repeating the procedure across subjects and assessing whether the same loopholes and constraint responses recur.

### C. Empirical thresholds

The constraints combine physiologically interpretable metrics with empirical thresholds. UP/DOWN states, spindle event duration, spindle envelope burstiness, and PAC are grounded in sleep physiology, but the exact values used here are subject- and model-specific. The manuscript should therefore be read as a diagnostic study of objective failure, not as a universal definition of N3 physiology.

### D. Partial circularity

V7's feasible-solution fitness includes a T4-derived SO reward and a T12-derived spindle reward. This creates overlap between the fitness and constraints. We therefore do not use the V7 score itself as held-out validation. Instead, we report the overlap explicitly and include two independent held-out checks (Sec. V-D), which give mixed evidence and do not let us rank V1 against V7.

### E. Mixed held-out results

Two held-out checks narrow the conclusion, and they disagree. On SO average-cycle waveform morphology, V7 is closer to real EEG than the canonical V1 (RMSE 0.39 vs 0.57); on cortical spindle density, V1 is closer because V7 produces no cortical spindles at all. Together they prevent both the claim that V7 is simply "more EEG-like" and the opposite claim that V1 is more EEG-like — the held-out evidence does not rank the two. V7 is more event-feasible under the explicit T1-T12 audit, but that feasibility does not translate into uniform held-out superiority. This distinction is important: a model can pass the event audit while still being deficient on an independent held-out statistic (here, cortical spindle expression), and a spectrally selected model can be closer on one waveform statistic yet invalid under the audit. Critically, neither model is correct — V1 fails 5/12 audit checks and under-produces cortical spindles, and V7, though it passes all 12, produces zero cortical spindles. The contribution is diagnostic, not a ranking.

### F. Residual loophole: thalamic spindles without cortical expression

The spindle-density check exposes a residual loophole inside the "closed" V7 audit, and it is the sharpest illustration of this paper's thesis. V7 passes T8 (25 detected spindle events) and T12 (20 sigma-verified events) — but those constraints are evaluated on the *thalamic* firing rate. When we measure spindle density on the *cortical* signal, the EEG-analog quantity, V7 yields exactly **zero** events over a 300 s simulation, in both the 10-14 Hz and 11-15 Hz bands. The mechanism is direct: V7's selected `c_th2ctx = 0.0127` is roughly eight times smaller than V1's `0.099`, because V7 deliberately tightened thalamocortical coupling toward zero for slow-oscillation stability. At that coupling, thalamic spindles barely reach cortex, so the events that satisfy the audit would not appear in EEG.

This is specification gaming one level deeper than the failures V7 was designed to close. The V7 audit verified spindle *reality* (an event must contain a genuine sigma oscillation, T12) but not spindle *observability* in the signal the EEG actually records. The optimizer, rewarded for thalamic spindle events, found a regime that produces them on the unobserved node while the SO-stability bound suppressed their cortical expression. The fix is not difficult to state — verify spindles on the cortical signal, or constrain `c_th2ctx` from below — but the point of the paper is that the loophole was invisible until an independent, signal-matched held-out check revealed it. It also reinforces that V1 is no remedy: V1's larger coupling yields some cortical spindles (1.2-1.4/min) but still only about half the real density (2.4-3.0/min), and V1 fails 5/12 audit checks. Both models are deficient in different ways; the audit and the spectrum each hide a different failure.

### G. Stage 2 SBI

Simulation-based inference is reserved for a future journal extension focused on posterior identifiability. The current Stage 2 SBI results are not used here to prove V7 superiority. They address a different question: which parameters are identifiable under selected summary statistics. This BMEiCON paper remains focused on Stage 1 objective diagnosis and event-level loophole closure.

---

## VII. Conclusion

This single-subject diagnostic study shows that spectral objectives can select thalamocortical model parameters that are shape-alike yet leave the intended dynamics unverified — and that, when those dynamics are explicitly audited, the spectrally convincing V1 solution is invalid on named event-level criteria (it passes only 5 of 12 checks, with no sustained UP states and zero SO-spindle coupling). Treating the optimizer as a loophole-searcher revealed a sequence of failure modes across V1-V6, including persistent-DOWN behavior, SO-spindle decoupling, fake spindle events, and unreachable spectral spindle rewards. V7 closed these event-level loopholes and increased feasible solutions from 1/4960 in V6 to 364/4960 (in part through threshold relaxation we quantify separately), yielding a final point that passed all 12 event-level constraints. At the same time, two independent held-out checks gave mixed evidence — V7 was closer to real EEG on SO waveform morphology, while V1 was closer on cortical spindle density — and a residual loophole emerged: V7 passes its thalamic spindle audit yet expresses zero spindles in the EEG-analog cortical signal at its small thalamocortical coupling. We therefore make no claim that V7 is the better model; V1 is itself invalid under the audit (5/12) and under-produces cortical spindles by roughly half. Both solutions are deficient in different ways. The contribution is therefore not a universal fitting method, nor a ranking of V1 against V7, but a transferable diagnostic workflow: anchor a spectral objective, treat the optimizer as an adversary, and use each feasibility collapse — and each independent held-out check — to localize the next loophole, including loopholes that survive inside an apparently closed audit. We report this for one subject; whether the same loopholes recur across subjects is the natural next study.

---

## Draft Figure and Table Plan

**Table I:** Loophole-to-constraint map across V1-V7.  
**Table II:** Version-level scores, `shape_r`, and feasible rates (with a footnote giving the feasible count under V6's T5 threshold to de-confound the trajectory).  
**Figure 1:** Schematic of spectral-fit loophole: PSD similarity vs event-level dynamics.  
**Figure 2:** Feasible-rate recovery from V4-V7.  
**Figure 3:** Existence object — V1 (spectral-only) vs V7 cortical time series side by side, with the V1 point's failed T1-T12 checks annotated (instantiates the Sec. V-A claim).  
**Figure 4:** Held-out checks giving mixed evidence (canonical V1 throughout) — (a) SO waveform template distances (V1 0.57, V3 0.34, V7 0.39; V7 beats V1) and (b) cortical spindle density (real ≈ 2.4-3.0/min, V1 ≈ 1.2-1.4/min, V7 = 0/min in both 10-14 and 11-15 Hz bands; V1 beats V7). Source: `validation_outputs/fig_so_waveform_canonicalV1_distances.png`, `validation_outputs/fig_spindle_density_confirm.png`.  
**Figure 5 (or Discussion inset):** Residual loophole schematic — thalamic spindles (audited) vs absent cortical spindles at V7's small `c_th2ctx`.

---

## References

Author, venue, year, and DOI were verified against the publisher / DOI record (June 2026). Two items carry an inline author-decision note; all others are confirmed.

- **[REF: SleepEDF]** B. Kemp, A. H. Zwinderman, B. Tuk, H. A. C. Kamphuisen, J. J. L. Oberye, "Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG," *IEEE Trans. Biomed. Eng.*, 47(9):1185–1194, 2000, doi:10.1109/10.867928. Database resource: A. L. Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals," *Circulation*, 101(23):e215–e220, 2000, doi:10.1161/01.cir.101.23.e215.
- **[REF: Cakan2022-or-2023]** C. Cakan, N. Jajcay, K. Obermayer, "neurolib: A Simulation Framework for Whole-Brain Neural Mass Modeling," *Cognitive Computation*, 15:1132–1152, 2023 (online-first 2021), doi:10.1007/s12559-021-09931-9. *(Cite as 2023; the in-text key may be renamed from "Cakan2022-or-2023".)*
- **[REF: Jajcay2022]** N. Jajcay, C. Cakan, K. Obermayer, "Cross-Frequency Slow Oscillation–Spindle Coupling in a Biophysically Realistic Thalamocortical Neural Mass Model," *Front. Comput. Neurosci.*, 16:769860, 2022, doi:10.3389/fncom.2022.769860. *(Directly the thalamocortical SO–spindle model lineage of this work; cite prominently in Related Work.)*
- **[REF: RobinsonSleepModel]** P. A. Robinson, C. J. Rennie, D. L. Rowe, "Dynamics of large-scale brain activity in normal arousal states and epileptic seizures," *Phys. Rev. E*, 65(4):041924, 2002, doi:10.1103/PhysRevE.65.041924. *(AUTHOR DECISION: this is the canonical corticothalamic mean-field EEG model; if a sleep-specific corticothalamic reference is preferred, cite Costa et al., "Slow-wave oscillations in a corticothalamic model of sleep and wake," J. Theor. Biol., 2016 instead/additionally.)*
- **[REF: SteriadeSO]** M. Steriade, A. Nuñez, F. Amzica, "A novel slow (<1 Hz) oscillation of neocortical neurons in vivo: depolarizing and hyperpolarizing components," *J. Neurosci.*, 13(8):3252–3265, 1993, doi:10.1523/JNEUROSCI.13-08-03252.1993.
- **[REF: MassiminiSO]** M. Massimini, R. Huber, F. Ferrarelli, S. Hill, G. Tononi, "The sleep slow oscillation as a traveling wave," *J. Neurosci.*, 24(31):6862–6870, 2004, doi:10.1523/JNEUROSCI.1318-04.2004.
- **[REF: Molle2011]** M. Mölle, T. O. Bergmann, L. Marshall, J. Born, "Fast and slow spindles during the sleep slow oscillation: disparate coalescence and engagement in memory processing," *Sleep*, 34(10):1411–1421, 2011, doi:10.5665/SLEEP.1290.
- **[REF: Ferrarelli2007]** F. Ferrarelli, R. Huber, M. J. Peterson, M. Massimini, M. Murphy, B. A. Riedner, A. Watson, P. Bria, G. Tononi, "Reduced sleep spindle activity in schizophrenia patients," *Am. J. Psychiatry*, 164(3):483–492, 2007, doi:10.1176/ajp.2007.164.3.483. Cited here only for its sigma-band spindle-detection convention (Sec. III-E), not for any normative spindle rate. *(CAVEAT: this is a schizophrenia spindle-deficit study; it does not establish a "~15–20 spindles/min in N3" normative rate. The manuscript does not rely on such a rate — the `/15` denominator in `spindle_power` is treated as an empirical saturation scale, not a normative target (Sec. II-C) — but a residual normative claim survives in the project code comments and should be removed there; cf. the held-out density ≈ 2.4–3.0/min measured here.)*
- **[REF: Helfrich2018]** R. F. Helfrich, B. A. Mander, W. J. Jagust, R. T. Knight, M. P. Walker, "Old Brains Come Uncoupled in Sleep: Slow Wave-Spindle Synchrony, Brain Atrophy, and Forgetting," *Neuron*, 97(1):221–230.e4, 2018, doi:10.1016/j.neuron.2017.11.020.
- **[REF: Tort2010]** A. B. L. Tort, R. Komorowski, H. Eichenbaum, N. Kopell, "Measuring phase-amplitude coupling between neuronal oscillations of different frequencies," *J. Neurophysiol.*, 104(2):1195–1210, 2010, doi:10.1152/jn.00106.2010.
- **[REF: Canolty2006]** R. T. Canolty, E. Edwards, S. S. Dalal, M. Soltani, S. S. Nagarajan, H. E. Kirsch, M. S. Berger, N. M. Barbaro, R. T. Knight, "High gamma power is phase-locked to theta oscillations in human neocortex," *Science*, 313(5793):1626–1628, 2006, doi:10.1126/science.1128115.
- **[REF: Mander2015]** B. A. Mander et al., "β-amyloid disrupts human NREM slow waves and related hippocampus-dependent memory consolidation," *Nat. Neurosci.*, 18(7):1051–1057, 2015, doi:10.1038/nn.4035.
- **[REF: Vallat2021-YASA]** R. Vallat, M. P. Walker, "An open-source, high-performance tool for automated sleep staging," *eLife*, 10:e70092, 2021, doi:10.7554/eLife.70092.
- **[REF: FOOOF]** T. Donoghue, M. Haller, E. J. Peterson, P. Varma, P. Sebastian, R. Gao, T. Noto, A. H. Lara, J. D. Wallis, R. T. Knight, A. Shestyuk, B. Voytek, "Parameterizing neural power spectra into periodic and aperiodic components," *Nat. Neurosci.*, 23:1655–1665, 2020, doi:10.1038/s41593-020-00744-x.
- **[REF: SpecificationGaming]** V. Krakovna, J. Uesato, V. Mikulik, M. Rahtz, T. Everitt, R. Kumar, Z. Kenton, J. Leike, S. Legg, "Specification gaming: the flip side of AI ingenuity," DeepMind Safety Research (blog), 2020.
- **[REF: RewardHacking]** D. Amodei, C. Olah, J. Steinhardt, P. Christiano, J. Schulman, D. Mané, "Concrete Problems in AI Safety," arXiv:1606.06565, 2016. J. Skalse, N. H. R. Howe, D. Krasheninnikov, D. Krueger, "Defining and Characterizing Reward Hacking," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2022, arXiv:2209.13085.

