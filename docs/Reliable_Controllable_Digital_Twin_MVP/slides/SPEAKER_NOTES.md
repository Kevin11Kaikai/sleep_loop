# Speaker Notes - Reliable + Controllable Digital Twin MVP

## Slide 1 - Can a Mechanistic Digital Twin Be Both Reliable and Controllable?

**Question answered:** What is the Part II scientific question and the one-line answer?

**Evidence shows:** M1 establishes the reliable frozen C006; C1 establishes reproducible primary controllability in that same frozen model.

**Does not prove:** Clinical efficacy, patient benefit, or general control of all sleep dynamics.

**30-60 second guidance:** “Part II has two linked tests. First, can we justify one mechanistic twin and freeze it? Second, can that exact frozen twin respond reproducibly to a bounded causal intervention? M1 answers the first with C006. C1 answers the second with fresh multi-seed T2 evidence. The conclusion is deliberately bounded: an in-silico Reliable + Controllable Digital Twin MVP.”

**[Sources]** M1 `HUMAN_REVIEW/FINAL_REPORT.md`; C1 `HUMAN_REVIEW/C1_HIERARCHICAL_PRIMARY_TARGET_FINAL_SUMMARY.md`; absolute roots in `SOURCE_AUTHORITY_MAP.md`.

## Slide 2 - M1 - Why C006?

**Question answered:** Why C006 rather than another candidate?

**Evidence shows:** C006 had the lowest FIT NRMSE among 16 frozen candidates (0.2546), with C014 next (0.2759), and then retained a strong held-out advantage over matched C000.

**Does not prove:** Population superiority, clinical validity, or that the candidate bank covers every possible mechanism.

**30-60 second guidance:** “The selection was not retrospective storytelling. The candidate bank and rule were frozen: minimum FIT NRMSE after FIT-only calibration. C006 was the clear minimum at 0.2546. The second panel is not used to reselect it; it shows what happened after selection was frozen - C006 held-out NRMSE was 0.2478 versus 0.4638 for C000, a 46.6% relative improvement.”

**[Sources]** M1 `RUNS/FIT_WORKER/SELECTION_FREEZE.json`; `CODE/FROZEN_V2/protocol_v2.json`; `RUNS/HELDOUT_WORKER/HELDOUT_G6_RESULT.json`.

## Slide 3 - M1 - Frozen Reliable Mechanistic Twin

**Question answered:** What makes the M1 evidence chain reliable enough for the control experiment?

**Evidence shows:** Selection, pre-heldout freeze, held-out gate, independent reproduction, and exact provenance hashes.

**Does not prove:** Scalp-voltage validity, population generalization, or intervention response.

**30-60 second guidance:** “Reliability here is procedural and evidential. C006 was selected on FIT, frozen before held-out access, passed the held-out margin with a positive bootstrap interval, and the full chain was independently reproduced without material disagreement. Most importantly for C1, protocol, model, and selection identities were locked before controller confirmation.”

**[Sources]** M1 `STATE/CANDIDATE_FREEZE_CHECKPOINT.json`; `RUNS/HELDOUT_WORKER/HELDOUT_G6_RESULT.json`; `TOURNAMENT/FINAL_M1_JUDGMENT.json`.

## Slide 4 - C1 - What Does “Controllable” Mean?

**Question answered:** What precise test defines controllability in this MVP?

**Evidence shows:** Frozen T2-primary direction, matched SHAM, fresh seeds, two durations, causal legal actuator, integrity vetoes, and visible secondary diagnostics.

**Does not prove:** Simultaneous optimization of every endpoint or global state-space controllability.

**30-60 second guidance:** “C1 is a prospective response test, not a claim that every metric improves. T2 is the primary target, I_SO is supporting corroboration, and spindle/coupling measures remain mandatory diagnostics. The intervention is causal and bounded, with a legal maximum of 0.05. Success requires replicated T2 direction and complete integrity.”

**[Sources]** C1 `PROTOCOL/C1_PRIMARY_TARGET_V1_T2.json`; `STATE/FROZEN_CONTROLLER.json`.

## Slide 5 - Primary T2 Response Replicates Across Fresh Seeds and Duration

**Question answered:** Does the primary response replicate and survive longer duration?

**Evidence shows:** T2 positive in 3/3 seeds at 220 s and 3/3 at 910 s; median +0.1913% and +0.1784%; integrity PASS for all six.

**Does not prove:** A large clinical effect or a monotonic response for every possible seed/duration.

**30-60 second guidance:** “Every fresh seed is positive at both durations. The seedwise magnitudes vary, but the primary direction survives. The median is +0.1913% at 220 seconds and +0.1784% at 910 seconds. Because the seeds, durations, direction, and gate were fixed before these results, this is the core prospective controllability evidence.”

**[Sources]** C1 `RUNS/C1_PRIMARY_TARGET_AGGREGATE.json`; six per-seed `result.json` files; `FIGURES/C1_HIERARCHICAL_PRIMARY_TARGET_QUICKLOOK.png`.

## Slide 6 - I_SO Supports the Primary Result; Secondary Trade-offs Remain

**Question answered:** Is the primary T2 finding corroborated, and what adverse dynamics remain?

**Evidence shows:** I_SO positive 3/3 at both durations. I_SP is adverse 0/3 positive at both durations; T6 is positive only 1/3 with slightly negative medians.

**Does not prove:** Controller optimality or resolved spindle/coupling behavior.

**30-60 second guidance:** “The independent slow-oscillation diagnostic supports the T2 result in all seeds and both durations. The secondary heatmap also prevents overclaiming: spindle-band strength moves adversely in every run, and T6 is mixed. So the correct scientific sentence is primary controllability with secondary dynamical trade-offs.”

**[Sources]** C1 `RUNS/C1_PRIMARY_TARGET_AGGREGATE.json`; `RUNS/C1_PRIMARY_TARGET_METRICS.csv`; authoritative quicklook.

## Slide 7 - Reliable + Controllable Digital Twin MVP

**Question answered:** Which components of the bounded MVP passed?

**Evidence shows:** Reliable frozen C006, provenance, frozen causal F01, legal actuator, fresh multi-seed T2, 910 s robustness, and I_SO corroboration all pass.

**Does not prove:** Clinical or global controllability claims.

**30-60 second guidance:** “This is the synthesis. M1 establishes model choice and credibility; C1 establishes a prospectively defined response in the same frozen model. Every listed component passed. That supports closing at the bounded in-silico MVP level, while leaving controller-quality trade-offs open.”

**[Sources]** M1 final decision and freeze artifacts; C1 final decision, protocol, aggregate, controller, and checkpoint.

## Slide 8 - Advisor Decision: Is Part II Ready to Close at the MVP Level?

**Question answered:** What decision is requested from the advisor?

**Evidence shows:** The primary reliability-to-controllability chain is complete and hash-linked; the remaining limitations are explicit and non-vetoing under the frozen gate.

**Does not prove:** That future controller optimization or translational validation is unnecessary.

**30-60 second guidance:** “My proposed closure is specific: Part II is complete at the bounded in-silico Reliable + Controllable Digital Twin MVP level. The left column shows the evidence for closure. The right column preserves what is not solved - adverse spindle-band behavior, mixed T6, one-subject proxy-level reliability, and no clinical claim. The advisor decision is whether that bounded closure is sufficient, with those issues moved to limitations and future work.”

**[Sources]** `CLAIMS_AND_LIMITATIONS.md`; M1 and C1 final decisions and human reviews.
