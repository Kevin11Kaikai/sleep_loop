# COSTA Phase 2 prospective design memo — Candidate V4

Campaign: `COSTA_PHASE2_FRESH_20260816_105951`  
Protocol candidate: `COSTA_PHASE2_PROTOCOL_CANDIDATE_V4`  
Diagnostic candidate identity: `PHASE2_CONTROL_BENCH_V1`  
Status: `FREEZE_CANDIDATE_PREOUTCOME`  
Scientific outcomes visible while designing V4: `NONE`

## Immutable lineage and repair scope

Candidate V4 is a prospective, outcome-blind repair of exactly `PV3-B001` in `GOVERNANCE/PROTOCOL_FREEZE_REAUDIT_V3.json` (SHA-256 `7730B0DF0B5B87B7A6B49FD1DE2F02F7FE1A2D3C6413CD3DEDA78F9B72420C0D`). Its exact parent is `PROTOCOL/PROTOCOL_FREEZE_V3.json` (SHA-256 `0FF4B771925FF942B10515384E923EB06649507EC4EE6FDB538C556FCB952412`) and the eight Candidate V3 artifacts bound by that freeze.

V4 preserves every V3 repair of `PV2-B001` through `PV2-B004` and every earlier V2 repair of `PF-B001` through `PF-B005`: executable parameterized Level-B schedules, mechanically enumerable policies and scalars, exact T7/circular inference, the bound deterministic T4 fixture, the causal phase scheduler, six exact 25 s T8 windows and compatible Welch estimator, the unique phase grids, and exact simultaneous-band construction. It also preserves target priorities (primary T1/T4/T6; secondary T2/T3/T5/T7; exploratory T8), model equations/defaults/uncertainty draws, all scientific thresholds and grids, run counts and compute partitions, Level-A margins, negative-result retention, stability/off-target vetoes, and the at-most-three advancement rule.

The exact Candidate V3 artifact parents are: `control_target_registry.json` SHA-256 `0227843EA320CA272C2E3385AF3DB1332E8DAC32641793C36E1CDDF07CA71223`; `mechanistic_parameter_map.json` `36433E1ADFBB4DE78A52F4C3B01C6934E01D3F400220FB8AC354916974750A6C`; `phase2_protocol.json` `D901CA9ABD0F876268A24FD5C96F8F4FF7B0DAE2844560A44593C9592B8F6053`; `advancement_rule.json` `8C0879D1B4C0C4E1E1921ACF223C32F0D29AC12E513C9888287844620FED1E88`; `c2_gate_protocol.json` `F83CDA10944AACB0E86D44DE241CCEB57D2490AA50A1D89B57E7F780A10E683D`; `claim_registry.json` `5934AE57AE3B8262EA18101BF30654F7A0C9AD4682FAE60C2DAAE7BB743CFF3B`; `PREOUTCOME_DESIGN_MEMO.md` `6B563539966798064F3427A7DE112F28D743826FC4393BEB53C7A2F27C465FDB`; and `t4_attribution_fixture_v3.json` `30402053094C0C1BCDEF87989D1D7DE3DD4628D2C8D815757785BBA75D9DB4A9`.

No V1–V3 artifact, freeze, shared ledger, STATE file, repository code/configuration/data, R4 material, protected output, or scientific outcome was modified or used to choose this repair. V4 is a candidate only; it does not freeze or authorize implementation, preflight, simulation, or scientific execution.

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

The V3 fixture is copied byte-for-byte to `PROTOCOL/CANDIDATE_V4/t4_attribution_fixture_v3.json` and retains SHA-256 `30402053094C0C1BCDEF87989D1D7DE3DD4628D2C8D815757785BBA75D9DB4A9`. It continues to validate the classifier only; the domain table separately validates context coverage and cannot provide scientific evidence.

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

The mandatory preflight retained by V4 binds `PROTOCOL/CANDIDATE_V4/t4_attribution_fixture_v3.json`, SHA-256 `30402053094C0C1BCDEF87989D1D7DE3DD4628D2C8D815757785BBA75D9DB4A9`.

The fixture contains raw paired sham and intervention events, emitted packet intervals, and sigma upward crossings. Its hand-calculated oracle exercises greedy candidate order, an already-matched rejection, unmatched added/lost events, equal-distance multiple-packet tie resolution, one autonomous post-packet cycle, one packet replica, and one unassociated added event. It expects `N_added=3`, `F_auto=1/3`, and `F_replica=1/3`; its scientific denominator is deliberately `NOT_ESTIMABLE`, so passing the fixture cannot count as target evidence.

Preflight passes only on exact deep equality of all registered ordered tables, IDs, classes, counts, and integer fraction pairs, with every raw row consumed. Any missing, extra, reordered, mistyped, unequal, or unconsumed field, or any exception, emits `PF_T4_EVENT_ATTRIBUTION_FAIL` with a path-indexed diff and stops before Phase 2A.

## Implementation clarifications retained as stricter governance

Inputs are evaluated at the left endpoint of each 1 ms solver interval and held through both Heun drifts. Butterworth band filters use explicit `btype='bandpass'`, SOS output, 200 Hz sampling, and `sosfiltfilt(...,padtype='odd',padlen=None)`. The even 40-sample spindle RMS window uses samples `n-19` through `n+20`. T6 surrogate lags are the inclusive stored-sample integers 2,000 through 34,000, chosen without replacement, and applied with positive `np.roll`.

These clarifications do not relax any V2 rule, threshold, margin, denominator, comparator, claim ceiling, or negative-result obligation.

## Candidate package and next governance action

Candidate V4 contains the six required JSON registries/protocols, this memo, and the byte-identical bound T4 fixture manifest. Every JSON must parse; target count must be eight, C2 gate count ten, and claim count eight. The final audit must also verify the exact V3 parent-freeze and V3 re-audit hashes, each parent-artifact hash, route version 4.0.0, candidate identity, N=0/minimum 3 Phase2B status, `M1/C0/P0` ceiling, unique phase arrays, unchanged 25/20-cell arithmetic and 1,800/900/456/3,156 run maxima, 128 cross-target members, fixture hash, no stale current-candidate execution-status field, and no unresolved or outcome-dependent design field.

The T4 audit must mechanically compare the three worker candidate IDs, ten positive Level-B candidate IDs, and three possible verifier IDs against the corresponding context projections of the 16-row attribution table; prove unique join keys; prove exactly one formal row; and prove every `C4`, `G4`, `F_auto`, and `F_replica` reference resolves to a context row with its paired sham, packet schedule, pool, denominator, bootstrap, logs, and use classification.

Root alone may freeze this candidate after an independent audit. This memo authorizes no scientific execution and records no scientific result.
