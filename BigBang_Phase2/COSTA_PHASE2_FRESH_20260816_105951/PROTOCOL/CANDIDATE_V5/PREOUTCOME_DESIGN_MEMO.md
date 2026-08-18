# COSTA Phase 2 prospective design memo — Candidate V5

Campaign: `COSTA_PHASE2_FRESH_20260816_105951`  
Protocol candidate: `COSTA_PHASE2_PROTOCOL_CANDIDATE_V5`  
Diagnostic candidate identity: `PHASE2_CONTROL_BENCH_V1`  
Status: `FREEZE_CANDIDATE_PREOUTCOME`  
Scientific outcomes visible while designing V5: `NONE`

## Immutable lineage and repair scope

Candidate V5 is a prospective, outcome-blind repair of exactly `PV4-EDGE-B001` in `GOVERNANCE/V4_EDGE_SEMANTICS_ADJUDICATION.json` (SHA-256 `227E0CF0E8865A3101A3E293758483B5ADF450AB6486D955E6BE334DAC50277A`). Its exact parent is `PROTOCOL/PROTOCOL_FREEZE_V4.json` (SHA-256 `F002EE65EA450757E85C854FF9A6DB93518F39F2D1188CEEFF253D5AC2BEA996`) and the eight Candidate V4 artifacts bound by that freeze. The parent’s independent re-audit is `GOVERNANCE/PROTOCOL_FREEZE_REAUDIT_V4.json` (SHA-256 `D30035223430075381048426BE8E2C9AAD53E6165F4CB15CD2242E48189EF496`).

V5 preserves the V4 repair of `PV3-B001`, every V3 repair of `PV2-B001` through `PV2-B004`, and every earlier V2 repair of `PF-B001` through `PF-B005`: the complete 16-row T4 attribution domain, executable parameterized Level-B schedules, mechanically enumerable policies and scalars, exact T7/circular inference, the bound deterministic T4 fixture, the causal phase scheduler, six exact 25 s T8 windows and compatible Welch estimator, the unique phase grids, and exact simultaneous-band construction. It also preserves target priorities (primary T1/T4/T6; secondary T2/T3/T5/T7; exploratory T8), model equations/defaults/uncertainty draws, all scientific thresholds and grids, run identities/counts and compute partitions, Level-A margins, negative-result retention, all other stability/off-target vetoes, and the at-most-three advancement rule.

The exact Candidate V4 artifact parents are: `control_target_registry.json` SHA-256 `E300F2B2184B1430428FD1E2DB0CF76D07DCD5B89FF78E287C06F5213629B058`; `mechanistic_parameter_map.json` `EB811129C358B7A25F78FE8B9EB9E253F79D155F548EDDF9D8919A5366FCD013`; `phase2_protocol.json` `AB4B5226DF13529BED15D910DF12401FF192110FC152489E153FF9E2641B3BE7`; `advancement_rule.json` `0F0EDE0523A40D59D598CEA18672F53D0780607999253B88CF93F201E028D086`; `c2_gate_protocol.json` `DB7A36F243BBB08885B122AC4353A3ACE94D066A677F27FA04A912B64942135D`; `claim_registry.json` `E792BEF42B7A2A279CF2F25658C3165FA1F3D3090D01D5A22A573394B393268F`; `PREOUTCOME_DESIGN_MEMO.md` `83EB7ADE9ADCFBA3DF371C3192EB904C6194B2AC2A79EE55B47237D42227735B`; and `t4_attribution_fixture_v3.json` `30402053094C0C1BCDEF87989D1D7DE3DD4628D2C8D815757785BBA75D9DB4A9`.

No V1–V4 artifact, freeze, audit/adjudication receipt, shared ledger, STATE file, repository code/configuration/data, R4 material, implementation bytes, preflight candidate manifest, protected output, or scientific outcome was modified or used to choose this repair. V5 is a candidate only; it does not freeze or authorize implementation, preflight, simulation, or scientific execution.

## V4-to-V5 change log

### PV4-EDGE-B001 — executable paired-sham mean/variance-rescale event veto

Candidate V5 replaces the single underdefined veto sentence with one canonical machine-readable contract, `RESCALE_ONLY_EVENT_VETO_V5`, in `phase2_protocol.json`. It runs on every launched Level-A, Level-B, and verifier condition for every T1–T8 target and off-target use, and separately diagnoses SO and spindle events. Reused trajectories, including the T6/T7 overlap, produce one diagnostic record referenced by all consumers. The diagnostic adds no scientific run, writes to no intervention channel, changes no model parameter or schedule, and remains posthoc validity analysis only.

For each exact paired draw-seed block, the only inputs are the uncentered 200 Hz observation arrays `x_raw[n]=(r_E_stored[n]+0.25r_T_stored[n])/10` for the intervention and its exact all-zero-input sham, over all 36,000 stored samples of the 180 s analysis record. Both arrays must already be contiguous float64 and finite. Their exact little-endian bytes are hashed. The mean is an ascending sequential float64 sum divided by 36,000. Population variance is a second ascending sequential sum of squared deviations divided by 36,000 (`ddof=0`). Reassociation, compensated or parallel reduction, fused multiplication-add, floors, and alternate normalizers are prohibited.

When both variances are strictly positive and finite, V5 computes `a=sqrt(v_sham/v_intervention)` and then, in the frozen operation order, `x_rescaled[n]=mu_sham+a*(x_intervention[n]-mu_intervention)`. Exact zero variance in either member is `NOT_ESTIMABLE_ZERO_VARIANCE`; negative/nonfinite arithmetic or nonfinite inputs are `INVALID` and veto-equivalent. No epsilon repair, false default, or fabricated empty event table is allowed.

Sham, intervention, and rescaled intervention are each centered over their own full record and passed through the unchanged V4 Butterworth filters and exact SO/spindle detectors. The official sham/intervention reruns must deep-equal their official event tables. SO reuses the fixed duration, `>=0.10` m.u. peak-to-peak, and `<=-0.04` m.u. trough criteria. Spindles reuse the exact high/low constants derived once from the paired sham; recalibration on either non-sham trajectory is prohibited. A canonical provenance object, including float64-hex thresholds and SOS coefficients, is serialized deterministically and must have the same hash for all three reruns. Equality and strict-edge rules are now explicit: SO upward crossing is `y[c-1]<0 && y[c]>=0`; threshold/duration equality passes; spindle high/low equality and 20-sample gap equality pass; retained event boundaries must lie strictly inside samples 2,000 and 34,000.

Within each event type, V5 performs three independent one-to-one matches: official intervention to sham, rescaled intervention to sham, then original ADDED to rescaled ADDED. Candidate anchors may differ by at most 50 stored samples, with equality accepted. Pairs are sorted by absolute difference, left anchor, right anchor, left ID, and right ID; one greedy traversal is final. SO anchors are earliest trough samples and spindle anchors are earliest maximum-RMS samples.

Per block, V5 records sham/intervention/rescaled counts, original and rescaled ADDED counts, persistent matches, and `N_rescale_only=N_added_original-N_persistent`. If original additions exist, `F_rescale_only=N_rescale_only/N_added_original`; otherwise the fraction is null with `NO_ADDED_EVENTS`. Pooling is only within `(execution_context,target_id,condition_id,event_type)` across complete paired blocks, with unchanged context minima: Level A at least 30/36, five seeds and five draws; Level B at least 10/12, three seeds and all three draws; verifier at least 20/24, three seeds and five draws.

The exact hard-veto predicate is the all-added-events-vanish rule: pooled original ADDED count is positive, pooled rescaled ADDED count is zero, pooled persistent count is zero, and the exact integer ratio `F_rescale_only` equals one. No confidence interval, tolerance, or majority vote applies. No original additions yields `NO_VETO_NO_ADDED_EVENTS`; partial or nonzero survival yields `NO_VETO_PERSISTENT_OR_MIXED` with exact sensitivity counts. `INVALID` is veto-equivalent. `NOT_ESTIMABLE` is never coerced to false and prevents Grade C, Level-B/verifier passage, advancement, and C2-G7/G8.

This veto is deterministic and outside `F_TARGET_DECISIONS`, `F_LEVEL_B_SURFACES`, `F_CROSS_TARGET_8X8`, `F_FORMAL_C2_INTERACTIONS`, and Holm. It changes no family size, seed, resampling, test, or quantile and cannot be overridden by a favorable interval or p value. Candidate V5 also restates without changing the V4 all-missing-bootstrap rule: no redraw, skip, imputation, or shorter family; an all-missing component/family is `NOT_ESTIMABLE` with no numeric band; finite constant replicates use the scale floor; `T4 ZERO_ADDED_BOOTSTRAP` remains the sole exception after its point denominator passes.

The required block and condition artifacts have exact schemas for identities, raw hashes, float64-hex moments/scale, threshold provenance, all three event and match tables, counts, densities, fraction, denominators, statuses, reasons, downstream consumers, deterministic JSON serialization, and record hashes. Exact T6/T7 trajectory reuse produces one canonical T7-source block record at the matching phase, with T6 named as a downstream consumer; Level-B cells remain distinct under the frozen no-deduplication rule. Every missing/failed block has one enumerated reason; silent exclusion is prohibited.

V5 gives the already frozen Level-A runs exact, mechanically enumerable identities without adding a run: `6+7+7+7+7+9+7=50` condition identities in target order, crossed with the unchanged six draws and six seeds for exactly 1,800 runs. The nine shared T6/T7 identities consist of eight canonical T7 phase runs plus its amplitude-zero sham; T6 references the matching T7 physical ID. Each Level-A group has one exact sham condition, and its explicit zero/sham identity is allowed to self-pair only when listed. This naming layer does not change a grid value, waveform, execution order, or scientific comparator.

`PAIRED_SHAM_IDENTITY_REGISTRY_V5` maps each intervention run ID directly to one physical sham run ID; waveform search is prohibited. Every Level-B cell, including retained duplicate and zero cells, maps to `LEVEL_B_SHARED_SHAM::{draw_id}::S{seed}`, which resolves to the already completed all-zero Level-A T4 run for the same draw and seed; the Level-B draw/seed universe is a subset of Level A, so the cache adds zero runs. Every verifier intervention maps to its exact `X3__SHAM` run. The existing T4 `LEVEL_A_SHARED_SHAM` and `LEVEL_B_SHARED_SHAM` aliases resolve through this same registry. Thus multiple all-zero trajectories can never make pairing ambiguous.

Each expected block emits an exact `RV5B` block ID even on early failure, and each pooled condition emits an `RV5C` condition-record ID. Uncomputed downstream scalars, objects, and arrays are JSON null—not fabricated zeros or empty containers—while a valid detector run with genuinely zero events emits a real empty event array and calculated zero counts. Valid blocks use a null reason; failed blocks use exactly one enumerated reason. An invalid condition has a null hard-veto Boolean and remains veto-equivalent through its exact `INVALID_*` status. Every record’s downstream consumers are exactly T1–T8 in frozen order, and the six unmatched-event array members and their within-array ordering are exact. Block and condition payload hashes exclude only their own hash field, eliminating recursion while binding every other required field.

The new data-free oracle is `PROTOCOL/CANDIDATE_V5/rescale_event_veto_fixture_v5.json`, SHA-256 `DDB61398DC401BB8AF1CC7E7AE9594696E2DD1DAE38F6F40FA046F9425355DDC`. Its hand-calculated primitives cover mean shift, variance scaling, nonzero event survival, complete added-event disappearance, no added events, mixed survival, exact zero variance, nonfinite input, SO/spindle threshold equality, strict retained edges, and greedy matching ties at the exact 50-sample boundary. Primitive-mode length and table inputs are explicitly segregated from the production 36,000-sample shape and can never enter science. Exact fixture-only projection rules bind the reduced comparator/event rows and the local `B001`–`B030` pooling universe while bypassing only their corresponding production schema gates; reuse in a production or scientific artifact is a preflight failure. Any deep inequality, order difference, unconsumed row, exception, or status difference emits `PF_RESCALE_EVENT_VETO_FAIL` and stops before science. The pre-existing T4 fixture is copied byte-for-byte to the V5 package and retains SHA-256 `30402053094C0C1BCDEF87989D1D7DE3DD4628D2C8D815757785BBA75D9DB4A9`.

No target priority, intervention, dose/carrier/phase/frequency/exposure grid, estimator, practical margin, decision threshold, comparator, uncertainty seed, multiplicity family, run identity/count, compute partition, claim ceiling, or Phase2B denominator/status changed. This is the sole substantive V4-to-V5 repair.

## V3-to-V4 change log

### PV3-B001 — context-indexed T4 attribution domain

Candidate V4 replaces the under-scoped T4 attribution denominator with one canonical machine-readable table, `T4_ATTRIBUTION_DOMAIN_V4`, in `phase2_protocol.json`. Its exact join key is `(execution_context,dose_hz_equivalent,carrier_hz,repetition_hz)`. Every positive T4 condition used by a worker `G4`, Level-B `C4/G4`, or the selected verifier row must resolve to exactly one row. A missing match, duplicate match, candidate/table set difference, or duplicate join key is `PF_T4_ATTRIBUTION_DOMAIN_FAIL` and stops before science.

The table has exactly 16 rows:

- Three Level-A worker rows at `d=[0.5,1.0,1.5]`, carrier 13 Hz and repetition 0.85 Hz: `T4_LA_WORKER_D0P50_FC13`, `T4_LA_WORKER_D1P00_FC13`, and `T4_LA_WORKER_D1P50_FC13`.
- Ten Level-B rows at `d=[0.75,1.5]` crossed with carriers `[11,12,13,14,15]` Hz and repetition 0.85 Hz: the five `T4_LB_D0P75_FCxx` rows followed by the five `T4_LB_D1P50_FCxx` rows.
- Three possible verifier rows corresponding one-to-one to the three Level-A worker candidates: `T4_XV_D0P50_FC13`, `T4_XV_D1P00_FC13`, and `T4_XV_D1P50_FC13`. Exactly the row corresponding to the frozen worker winner is executed; the other two are logged `NOT_LAUNCHED_BY_POLICY`, not treated as missing outcomes.

Each row freezes dose, carrier, repetition, condition and policy candidate identity, exact paired sham key, block key and block universe, context-specific complete-pair/seed/draw minima, `N_added>=30`, at least ten row-specific contributing blocks, `F_auto=N_autonomous/N_added`, `F_replica=N_packet_replica/N_added`, bootstrap family, zero-added convention, and formal-versus-diagnostic use. The exact packet schedule and waveform are shared only by reference to one frozen common contract; the emitted packet table remains the sole association input.

Pooling is strictly within one `domain_row_id` after blockwise matching and classification. Level-A worker rows require at least 30 of 36 complete pairs, five seeds and five draws. Level-B rows require at least 10 of 12 complete pairs, three seeds and all three draws. The selected verifier row requires at least 20 of 24 complete pairs, three verifier seeds and five draws. Every context also requires `N_added>=30` from at least ten blocks containing an added event. Failure makes all `F_auto`/`F_replica`-dependent scores in that row `NOT_ESTIMABLE`; another dose, carrier, context, or formal pool cannot substitute.

After, and only after, the point-estimate denominator passes, a registered hierarchical bootstrap replicate with zero added events assigns `F_auto=0` and `F_replica=1` and logs `ZERO_ADDED_BOOTSTRAP` with row, family and replicate. Required logs include the domain/context/condition/policy IDs, draw/seed/block and paired-sham IDs, dose/carrier/repetition, complete packet/event/match/crossing/class tables, all numerators and denominators, status, and zero-added records.

`T4_LA_WORKER_D1P50_FC13` remains the sole formal `F_TARGET_DECISIONS` attribution comparison. The other two Level-A worker rows affect only prospective worker policy selection. The ten Level-B rows affect only the descriptive surface and Level-B policy. The one selected verifier row affects only the diagnostic `F_CROSS_TARGET_8X8` row; the two alternatives are not launched. No formal margin, priority, grid, score formula, surface cell, run identity, run count, compute cap, Phase2B decision, or evidence ceiling changed.

The V3 fixture remains copied byte-for-byte in the current package at `PROTOCOL/CANDIDATE_V5/t4_attribution_fixture_v3.json` and retains SHA-256 `30402053094C0C1BCDEF87989D1D7DE3DD4628D2C8D815757785BBA75D9DB4A9`. It continues to validate the classifier only; the domain table separately validates context coverage and cannot provide scientific evidence.

## Unchanged evidence boundary

`PHASE2_CONTROL_BENCH_V1` remains a separate clean-room reduced model with no executable identity link to the R4 candidate. Its maximum possible evidence is non-transferable synthetic diagnostic Grade C. It cannot change the R4-derived `M1/C0/P0` categorical substrate, cannot assign formal C1 or C2, and cannot support clinical, human, auditory-stimulation, or personalized-control claims. All applied channels are named **GENERIC EXTERNAL FORCING**. Parameter changes remain sensitivity-only and are not interventions.

Formal Phase 2B remains prospectively `NOT_ESTIMABLE`: zero eligible certified public/development Night-1 personalized models are available, the minimum denominator is three, and synthetic draws may not substitute for subjects. The formal advancing-target set is therefore empty before any Phase 2A outcome.

## V2-to-V3 change log

### PV2-B001 — executable parameterized Level-B channels

Every Level-B surface now has exact row and column arrays, a float64 sample formula, warmup and analysis boundaries, combination/sham semantics, run identity, and an explicit binding through the target registry and mechanistic map.

- T1 has 25 cells: five requested frequencies crossed with five amplitudes. The exact input is `u_E[30000+a]=A sin(2πfa/1000)`.
- T2 has 25 cells: five DC doses crossed with five 0.85 Hz sine amplitudes. DC and sine are evaluated independently and added once.
- T3 has 25 cells: five pulse doses crossed with five repetition frequencies. Onsets use `floor(p*1000/f_rep+0.5)`; pulses occupy exactly 80 half-open internal steps and incomplete terminal pulses are omitted.
- T4 has 25 cells: five balance doses crossed with carriers 11–15 Hz. Starts use `floor(p*200/0.85+0.5)`; every 500-step packet uses the requested carrier, and its emitted packet table is the attribution input.
- T5 has 25 cells: five balance doses crossed with continuous carriers 11–15 Hz.
- T6 has 20 cells: the unique canonical phases `[-π,-π/2,0,+π/2]` crossed with amplitudes `[0,0.25,0.5,0.75,1]`, carrier fixed at 13 Hz.
- T7 has 20 cells: the same unique phases crossed with carriers `[11,12,13,14,15]`, amplitude fixed at 1.
- T8 has 25 cells: five balance doses crossed with five centered exposure fractions in each of the six fixed 25 s windows. The exact start is `a0+floor((25000-L)/2)`, `L=floor(25000*fraction+0.5)`, with half-open application.

The phase channel is now parameterized as `(phi,A,f_c)` while retaining every V2 causal filter, crossing, period, forward-modulo delay, integer quantization, one-sample latency, queue, overlap, and boundary rule. `+π` is never enumerated; `-π` is its sole canonical representative. An amplitude-zero phase cell emits no packet and records `ZERO_AMPLITUDE_SHAM`.

Every Level-B cell has the deterministic identifier `LB3_{target}_R{rr}_C{cc}` and run identifier `LB3__{target}__R{rr}__C{cc}__{draw_id}__S{seed}`, with zero-based two-digit row and column indices. Every cell receives four worker seeds by three registered draws; no identical-waveform cell is deduplicated. If `n25` selected targets have 25 cells and `n20` selected targets have 20 cells, exact Level-B runs are `12*(25*n25+20*n20)`, `n25+n20<=3`. The maximum remains 900. Including the 1,800 Level-A maximum, Phase 2A has at most 2,700 runs, 567,000 simulated seconds, and 567,000,000 internal solver intervals, subject to the unchanged 60 CPU-hour worker cap and launch stop.

### PV2-B002 — mechanically enumerable scores and policies

For every executed cell the protocol defines a dimensionless, larger-is-better `C_i(u)`; value 1 is the applicable practical-margin boundary. T1 combines requested-frequency closeness, PLV, and move fraction. T2–T5 orient the response by the signed dose. T4 positive-dose scores additionally require the autonomous-added-event fraction and its N≥30 / ≥10-block denominator. T6 combines corrected-MI gain and the positive-block fraction. T7 cell scores use wrapped tracking error. T8 uses signed occupancy change and two-window persistence.

The conditional Level-B selected-policy object is explicit:

- T1 is a mapping policy `P1_A` from all five requested frequencies to the five cells at one amplitude, scored by the worst mapped `C1`.
- T7 is a mapping policy `P7_fc` from all four requested phases to the four cells at one carrier, scored jointly by circular correlation, tracking error, and ±π/2 separation.
- T2, T3, T4, T5, T6, and T8 select one eligible executed condition.

Every candidate set, control-cost tie breaker, row/column tie breaker, missing-score rule, and all-missing `NOT_ESTIMABLE` result is frozen. Policy aggregation creates no extra trajectories.

`F_LEVEL_B_SURFACES` enumerates every executed `C_i(u)-1` member and five additional mapping `G_i(P)-1` members when T1 or T7 is selected. Its exact size is `25*n25+20*n20+5*nmap`, at most 80.

The formerly generic “dose ordered in four of six draws” gate is replaced everywhere by an exact target-specific Level-A shape rule. Within each draw, every required condition needs at least five valid seeds. T2–T5 require a strict positive endpoint span and at least five of six nonnegative adjacent metric differences. T1 requires endpoint tracking improvement and PLV, T6 a positive phase-zero coupling gain, T7 positive Fisher–Lee phase tracking and ±π/2 separation, and T8 an exact supported adjacency pair plus persistent high occupancy. Missing draws fail; at least four of six draws must pass.

T8 adjacency is now enumerable. At Level A the only ordered same-side pairs are `(-2,-1.25)`, `(-1.25,-0.5)`, `(0.5,1.25)`, and `(1.25,2)`; a pair passes only when both strict oriented occupancy medians `Q(d)` are greater than zero. At Level B, within each exposure fraction independently, the only pairs are `(-2,-1)` and `(1,2)`. Equality or a missing dose fails. Per-block persistence means at least one of the five adjacent window pairs has both windows valid; the high-occupancy dose requires persistence in at least 75% of blocks and a simultaneous lower slack above 0. `F_TARGET_DECISIONS` therefore includes both `E8-1/3` and `persistence(d_hi)-0.75`.

The all-target 8×8 matrix uses a separate policy layer frozen from completed Level-A worker trajectories before verifier outcomes and before conditional Level B. T1 has the sole five-frequency mapping at amplitude 0.8; T7 has the sole eight-phase mapping at amplitude 1 and carrier 13 Hz. T2, T3, T4, T5, T6, and T8 each select one registered Level-A condition by their exact `C_i` maximum and tie rule. Thus all eight rows exist regardless of the at-most-three Level-B cap, and Level-B results cannot alter them.

Each cross-target cell has two scalars. For every expanded policy condition and column, compute the normalized primitive separately in each verifier block, take its block median `nHat`, and only then set `BENEFIT=max(0,nHat)` and `ADVERSE=max(0,-nHat)`. Mapping rows take minimum condition benefit and maximum condition adverse. The diagonal uses `max(0,G_i)` for benefit and the maximum of the already-computed adverse and `max(0,-G_i)`; consequently T4's `F_replica/0.50` adverse term is not discarded on its diagonal. Every bootstrap replicate repeats this exact order.

`F_CROSS_TARGET_8X8` therefore contains exactly 128 uniquely named members: benefit and adverse for every one of 64 cells. Each condition has 24 verifier blocks and requires at least 20 complete blocks, three verifier seeds, and five draws. The frozen policy expansions contain 19 logical intervention conditions; T6 is byte-identical to one of T7's eight mapped phases, so exact reuse plus one shared sham gives 19 unique trajectories per verifier block. Across 24 blocks this is 456 unique runs, 95,760 simulated seconds, and 95,760,000 solver intervals, charged only to the separate 50 CPU-hour verifier reserve. The overall maximum is therefore 3,156 registered runs, 662,760 simulated seconds, and 662,760,000 solver intervals. Worker plus verifier partitions permit at most 110 CPU-hours, within the 180-hour campaign total, with no borrowing. Missing primitives make both members missing and invalidate the complete policy row. Hard veto, intended, beneficial-secondary, negligible, and conservative trade-off classifications use exact strict/equality rules.

### PV2-B003 — exact T7 and Monte Carlo inference

For requested phases `alpha_l` and observed condition phases `beta_b,l`, blockwise `rho_c` is the Fisher–Lee pairwise circular-circular correlation: sum over lexicographic pairs `l<m` of `sin(alpha_l-alpha_m) sin(beta_l-beta_m)`, divided by the square root of the two corresponding sums of squared sines. This definition remains executable for the symmetric requested-phase grid because it does not require an undefined circular mean of `alpha`. Block tracking error is the median `abs(wrap(beta-alpha))` in radians. Block ±π/2 separation is `abs(wrap(beta_(+π/2)-beta_(-π/2)))`. Formal Level-A statistics are medians across at least 30 complete blocks; a degenerate pairwise denominator is missing.

The T7 statistic is
`H=min(rho_c/0.70, min(4,(π/6)/max(error,1e-12)), separation/(π/3))`.
The exact circular null uses 100,000 permutations from one `PCG64(860217)` lifecycle. Within each permutation and block, one `permutation(8)` reassigns the observed beta vector to fixed requested labels. The alternative is larger, ties count as extreme, degenerate permuted statistics equal negative infinity, and `p=(B+1)/(100000+1)`.

T1–T6 and descriptive T8 have exact normalized block statistics, target-specific PCG64 seeds 860231–860236 and 860238, one 100,000-row sign matrix per target, a greater alternative, retained zero effects, equality counted as extreme, and the same corrected p-value. The observed all-plus assignment is represented exactly by the +1 numerator and denominator and is not injected into the random rows.

Holm is now executable: ascending raw p, lexical target-ID ordering for exact ties, max-prefix adjusted p, and strict adjusted `p<0.05`. The primary family is T1/T4/T6. The T2/T3/T5/T7 family opens only after a complete primary pass. Missing p-values never reject and do not reduce family size. T8 remains descriptive.

### PV2-B004 — bound T4 attribution oracle

The mandatory preflight retained unchanged from V4 binds the current package path `PROTOCOL/CANDIDATE_V5/t4_attribution_fixture_v3.json`, SHA-256 `30402053094C0C1BCDEF87989D1D7DE3DD4628D2C8D815757785BBA75D9DB4A9`.

The fixture contains raw paired sham and intervention events, emitted packet intervals, and sigma upward crossings. Its hand-calculated oracle exercises greedy candidate order, an already-matched rejection, unmatched added/lost events, equal-distance multiple-packet tie resolution, one autonomous post-packet cycle, one packet replica, and one unassociated added event. It expects `N_added=3`, `F_auto=1/3`, and `F_replica=1/3`; its scientific denominator is deliberately `NOT_ESTIMABLE`, so passing the fixture cannot count as target evidence.

Preflight passes only on exact deep equality of all registered ordered tables, IDs, classes, counts, and integer fraction pairs, with every raw row consumed. Any missing, extra, reordered, mistyped, unequal, or unconsumed field, or any exception, emits `PF_T4_EVENT_ATTRIBUTION_FAIL` with a path-indexed diff and stops before Phase 2A.

## Implementation clarifications retained as stricter governance

Inputs are evaluated at the left endpoint of each 1 ms solver interval and held through both Heun drifts. Butterworth band filters use explicit `btype='bandpass'`, SOS output, 200 Hz sampling, and `sosfiltfilt(...,padtype='odd',padlen=None)`. The even 40-sample spindle RMS window uses samples `n-19` through `n+20`. T6 surrogate lags are the inclusive stored-sample integers 2,000 through 34,000, chosen without replacement, and applied with positive `np.roll`.

These clarifications do not relax any V2 rule, threshold, margin, denominator, comparator, claim ceiling, or negative-result obligation.

## Candidate package and next governance action

Candidate V5 contains the six required JSON registries/protocols, this memo, the byte-identical bound T4 fixture, and the new deterministic rescale-veto fixture. Every JSON must parse; target count must be eight, C2 gate count ten, and claim count eight. The final audit must also verify the exact V4 parent-freeze, V4 re-audit, and adjudication hashes, each parent-artifact hash, route version 5.0.0, candidate identity, N=0/minimum 3 Phase2B status, `M1/C0/P0` ceiling, unique phase arrays, unchanged 25/20-cell arithmetic and 1,800/900/456/3,156 run maxima, 128 cross-target members, both fixture hashes, no stale current-candidate execution-status field, and no unresolved or outcome-dependent design field.

The T4 audit must mechanically compare the three worker candidate IDs, ten positive Level-B candidate IDs, and three possible verifier IDs against the corresponding context projections of the 16-row attribution table; prove unique join keys; prove exactly one formal row; and prove every `C4`, `G4`, `F_auto`, and `F_replica` reference resolves to a context row with its paired sham, packet schedule, pool, denominator, bootstrap, logs, and use classification. The V5 audit must additionally prove that the Level-A identity registry contains exactly 50 unique condition IDs and generates exactly 1,800 unique run IDs; every Level-A identity, every cell of each actual launched Level-B surface, and every unique verifier intervention resolves to exactly one physical sham ID; the rescale-veto domain covers all eight targets and both event types; all block/condition fields, null/status/hash transitions, matching rules, and fixture expectations resolve; and the deterministic veto is outside, but fail-closed before, every simultaneous/Holm family.

Root alone may freeze this candidate after an independent audit. This memo authorizes no scientific execution and records no scientific result.
