# COSTA Phase 2 Candidate V1 — Pre-outcome Design Memo

Campaign: `COSTA_PHASE2_FRESH_20260816_105951`  
Protocol candidate: `COSTA_PHASE2_PROTOCOL_CANDIDATE_V1`  
Diagnostic bench: `PHASE2_CONTROL_BENCH_V1`  
Design status: `FREEZE_CANDIDATE_PREOUTCOME`  
Scientific execution performed by this role: **none**

## Decision in one paragraph

Candidate V1 freezes a self-contained, current-campaign thalamocortical diagnostic bench and an actual T1–T8 Phase 2A execution protocol, but it does not bind that bench to the unidentified R4 executable. The clean categorical substrate therefore remains exactly `M1/C0/P0`; the bench can yield at most non-transferable synthetic diagnostic evidence and cannot assign formal C1 or C2. Formal Phase 2B is frozen `NOT_ESTIMABLE` before outcomes because the clean data audit certifies zero eligible public/development Night-1 personalized models, zero clean executable personalized-model artifacts, and no exact R4 execution identity. The minimum formal denominator is three subjects, and synthetic draws cannot replace them.

## Input boundary and provenance

Only the exact files authorized in the role brief were read. No directory was listed. No R4 tree or R4 file, repository scientific code, configuration, fitted parameter, raw data, outcome, output, protected-derived material, Night-2 material, sealed material, or fresh-final material was opened or executed.

| Input | SHA-256 |
|---|---|
| Reliable Personalized Digital Twin governing attachment | `6FE01574A1DED6E78349C14787397A72A51A058870A3AC9AD7FDD4510F067170` |
| BIGBANG V4D.1.2 governing attachment | `1C65D1F2BB61E6D1597B164BAE6B4F1CE493799B9BCF9D165876F12053F26B46` |
| `COMMON/CAMPAIGN_IDENTITY.json` | `29DAD8795AD9B2A12298627F7BAFD2EB2022C0C1F03C031ED1428ED0FCE5B266` |
| `COMMON/CONTEXT_OF_USE.md` | `44E3185D67DE0C4DB592B95EE28ED61ABD98BB7D1F1D96E46DB2944D268081EA` |
| `COMMON/R4_SANITIZED_SUBSTRATE.json` | `2DF7A61074E30B04142A8470C95433D651292D1C8F716B4B213D568D380BC26F` |
| `COMMON/VALUE_FREE_REPOSITORY_CAPABILITY_INVENTORY.json` | `E89DDECF6ED40D566EFFBDCD9CDFCE3E9AC8AA9B12600F2C46E4F8CBF93B5D2E` |
| `GOVERNANCE/REPOSITORY_INPUT_AUDIT.md` | `21A179114AB783D9C42AAA9C6E6DDE554B3E6A26695E05E3234121C871487BD3` |
| `STATE/ENVIRONMENT_SNAPSHOT.json` | `9ACD57E9876C5FFCA20E7EAD233BE04BA1F90FBA1880ABAE9846114CEE4850D6` |
| `COMMON/COMPUTE_BUDGET.json` | `4C3C5961F5BE6EB134D7952FE74083C3B18303ABEE740027E8D9EE764F35ACBF` |
| `COMMON/DATA_FEASIBILITY_RECEIPT.md` | `B85E5A3A21D10C523BAD707D6F034E3BEFD63EB5573FAC95FE9D1DDCDF855D2C` |
| `COMMON/SUBJECT_MODEL_AVAILABILITY.json` | `08FF47A8EF93A08C032091F1314AA31BD959ADB320D8206E2E0BDADC4001BD88` |

The sanitized R4 artifact was used only for its permitted value-free categorical and integrity role: it establishes the historical categorical substrate and the absence of an executable identity. Its values did not determine the bench equations, thresholds, grids, target ranking, or scientific interpretation. The repository capability inventory was used only to avoid silently claiming that an existing canonical simulator, detector, or controller had been released. All numeric design choices in Candidate V1 are new, prospective bench conventions.

## Why a clean-room bench is necessary

The exact-file audit found candidate interfaces but no clean canonical personalized simulator. The latest reviewed lineage was quarantined, historical control code was quarantined or denied, no PID implementation was released, and no prior numerical control result was eligible. The data audit also found no exact R4 code/configuration/parameter binding. Treating any historical implementation as the R4 candidate would therefore be an identity inference and an import-boundary violation.

`PHASE2_CONTROL_BENCH_V1` is consequently a reduced four-population cortical E/I and thalamic relay/reticular stochastic rate model with slow cortical adaptation. Its equations, state order, units, defaults, operating bounds, solver, initial state, timebase, noise process, and observation operator are explicit in `mechanistic_parameter_map.json` and `phase2_protocol.json`. It is deliberately described as a diagnostic bench rather than a reconstruction, descendant, or surrogate of R4.

## Intervention semantics

Every actionable channel is named **GENERIC EXTERNAL FORCING**. Candidate V1 includes bounded additive cortical DC, cortical slow-frequency sinusoidal forcing, cortical pulse forcing, relay-versus-reticular sigma packets, continuous relay-versus-reticular sigma forcing, causal phase-triggered thalamic packets, and balanced cortical/thalamic versus inhibitory/reticular DC forcing. Injection population, timing, amplitude semantics, units, and ranges are frozen.

No channel is called auditory-like stimulation because no transduction, acoustic waveform, sensory pathway, or calibrated physical-unit mapping exists. Fitted or mechanistic parameter changes are explicitly `SENSITIVITY_ONLY`; they cannot satisfy an intervention or C2 claim.

## Prospective target classification

The target hierarchy is frozen independently of outcomes:

| Tier | Targets | Rationale |
|---|---|---|
| Primary | T1, T4, T6 | Three nonredundant domains with direct registered input paths: slow-frequency tracking, thalamic spindle-like occurrence, and SO–sigma coordination. |
| Secondary | T2, T3, T5, T7 | Scientifically useful but more vulnerable to gain, detector-threshold, event-denominator, or circular-concentration failure. |
| Exploratory | T8 | A multimetric model-regime construct with the greatest proxy and classification dependence; it cannot advance under V1. |

This classification does not predict which target will work. Valid negatives are expected and are retained.

## Phase 2A execution logic

All T1–T8 targets receive mandatory Level A fixed-grid execution. T6 and T7 share trajectories but retain separate metrics and pass rules. Every trajectory yields every target metric, producing the required 8×8 cross-target response matrix without selective reruns.

The worker design uses six PCG64 noise seeds and six synthetic parameter conditions (one nominal plus five prospectively generated Latin-hypercube perturbations). Sham and intervention conditions share the exact OU innovation streams. There are at most 1,764 unique Level A simulations. At most three deterministic Level B surfaces may run only after all Level A targets are complete, using a fixed eligibility and ranking rule and at most 900 additional simulations. No PID, reinforcement-learning, adaptive, or model-predictive controller is included. The phase-triggered packet is a fixed causal event rule, not an optimized controller.

The stored timebase is 200 Hz after 1 ms internal stochastic-Heun integration, with 30 s warmup and 180 s analysis. Fixed detectors cover slow-frequency Welch peaks, SO cycles/amplitude/density, sigma-RMS spindle events, bias-corrected Tort MI, preferred circular phase, and six-window model-regime occupancy. Ten analytic detector/numerical preflight tests must all pass before Phase 2A.

The per-target criteria are non-compensatory: practical effect, uncertainty, minimum event/run denominators, numerical stability, model-level plausibility, detector invariance, specificity, multiplicity, and off-target rules must all pass. Missing event-dependent values remain `NOT_ESTIMABLE`; invalid dynamics remain `INVALID`; neither becomes zero or success. Stable nulls and adverse effects remain in the evidence package.

## Uncertainty and personalization boundary

Synthetic parameter draws quantify local bench robustness only. They are not fitted subjects, pseudo-subjects, posterior human models, or evidence of personalization. Phase 2A uses a paired hierarchical bootstrap and paired sign-flip tests with fixed seeds. Any policy selected from a fixed grid uses a deterministic smallest-absolute-dose tie rule and is assessed on disjoint verifier seeds.

If a future, separately governed protocol version had at least three eligible subject models before outcomes, formal comparison would require paired sham, a leave-one-subject-out componentwise-median non-personalized model, and a deterministic cyclic wrong-subject policy. Both personalized-versus-non-personalized and personalized-versus-wrong-subject gains would need simultaneous lower confidence bounds above 0.25 sham-SD units, and between-subject curve distance would need to exceed within-subject uncertainty by a lower-bound ratio greater than 1.5. Those tests are not estimable in Candidate V1.

## Advancement and C2 decision

`advancement_rule.json` freezes a maximum of three Phase 2B targets and a deterministic ranking algorithm, but the non-compensatory entry gates precede ranking. Candidate V1 fails estimability at the identity and denominator gates before scientific outcomes: no exact same-candidate executable exists, zero eligible models are available, and neither non-personalized nor wrong-subject comparators can be built. The immutable current advancement set is therefore empty.

`c2_gate_protocol.json` prospectively specifies C2-G1 through C2-G10. Under current inputs, C2-G1, G3, and G9 are not estimable; the remaining formal gates are not run because the stage is ineligible. No partial gate pattern, attractive curve, bench response, or independent synthetic reproduction can assign C2.

## Claim discipline and evidence grades

The maximum bench grade is C: reproducible, specific, stable, mechanistically interpretable diagnostic controllability within this bench, with transfer and personalization unsupported. Grade A requires robust subject dependence; Grade B requires a transferable exact-candidate in-silico control result. Both are impossible under Candidate V1. Valid submargin results are Grade D, and instability/detector/leakage artifacts are Grade F.

Allowed future wording, only after execution and verification, is target-specific and always names the clean-room bench and generic forcing. Prohibited wording includes clinical control, treatment efficacy, real-world stimulation efficacy, human causal control, personalized intervention, validated digital twin, C1, C2, or any statement that the bench establishes the response of the unidentified R4 executable.

## Compute and stop rules

The design respects the frozen 60 CPU-hour Phase 2A worker allocation and protects the 50 CPU-hour independent-verification reserve. Current memory admission permits one scientific process. All eight Level A targets precede optional surfaces. New Level B launches stop at 52 consumed worker CPU-hours, leaving eight hours for completion and checkpoints. Disk use stops at 45 GiB, below the 50 GiB campaign limit. Every target and seed-draw block receives an immutable checkpoint; only exact incomplete run identities may resume.

## Freeze-candidate package

- `control_target_registry.json`: exact T1–T8 observables, mappings, comparators, estimands, margins, thresholds, denominators, uncertainty, off-targets, multiplicity, failure, and language.
- `mechanistic_parameter_map.json`: equations, states, default parameters, bounds, uncertainty draws, forcing channels, and theoretical parameter-target map.
- `phase2_protocol.json`: executable bench semantics, detector preflight, grids, seeds, Phase 2A hierarchy, statistics, stability, compute, and verification.
- `advancement_rule.json`: pre-outcome, at-most-three deterministic Phase 2B rule and immutable empty current decision.
- `c2_gate_protocol.json`: complete C2-G1 through C2-G10 specifications and current estimability states.
- `claim_registry.json`: target-specific diagnostic claim contracts and formal C2 nonclaim.
- `PREOUTCOME_DESIGN_MEMO.md`: rationale, provenance, and claim boundary.

This package contains a design, not scientific results. It must be independently audited and explicitly frozen by the authorized state steward before any scientific execution.
