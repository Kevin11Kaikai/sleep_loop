# COSTA R4 M1 independent-verifier preinterpretation

Campaign: `COSTA_MVP_FRESH_R4_20260816_023933`

Verifier role: fresh independent M1 reproduction and comparison only. This document was written before reading any worker result, worker output freeze, candidate-freeze checkpoint, worker packet, persuasive narrative, or worker outcome artifact.

## Outcome-exposure declaration

At the time of this preinterpretation I have not opened or read `STATE/WORKER_OUTPUT_FREEZE*`, `STATE/CANDIDATE_FREEZE_CHECKPOINT*`, any artifact under `RUNS/SYNTHETIC_WORKER`, `RUNS/FIT_WORKER`, or `RUNS/HELDOUT_WORKER`, `MECHANISTIC/WORKER_PACKET`, any worker narrative/result, any R1/R2/R3 or Night-2 material, any protected-derived or sealed/fresh-final material, any prior report/state/outcome, or any `SCRATCH` root draft. The only inspected campaign material was the authorized frozen specification, context, eligible-claim universe, M1 registry, official pre-open provenance, pre-raw freeze, data-partition freeze, environment snapshot, selective allowlist, and frozen code.

## Prospective protocol and expected decisions

The verifier will reproduce the frozen one-subject Night-1 N3 pipeline without tuning, repair, retry, or scientific change. MNE must open only the two exact SC4001 Night-1 files, once each for the verifier logical read, through `mne.read_annotations(path)` and `mne.io.read_raw_edf(path, preload=True, verbose="ERROR")`. Stages 3 and 4 map to N3; 30-second epoch assignment must precede signal inspection or missingness and use seed 26081601 with a 50/50 split. FIT and HELDOUT must remain separate. FIT alone selects the minimum-NRMSE candidate with candidate-ID tie break and freezes all calibrations. HELDOUT may be opened by the heldout stage only after binding the exact verifier selection-freeze SHA-256.

The synthetic run is expected to make deterministic G1-G5 and synthetic G6-interface decisions from the immutable harness. The empirical G6 decision requires at least four eligible N3 epochs, at least two valid epochs in each role, heldout personalized relative NRMSE improvement over C000 of at least 0.05, and a strictly positive lower bound of the 95% paired epoch-bootstrap interval for population-minus-personalized MSE using 1,000 draws and seed 26081604. G7 can pass only if this independent verifier reproduces frozen hashes, split, numeric outputs, comparator/baseline, threshold decisions, and procedure with no material disagreement. M1 is eligible only if every mandatory gate passes; any mandatory FAIL, INVALID, NOT_ESTIMABLE, or NOT_RUN prevents assignment. I will verify and report; I will not assign or advocate M1.

No empirical direction is presumed. A pass and a fail are equally admissible. My expected gate decision is therefore `DEFERRED_TO_FROZEN_EXECUTION` for each outcome-bearing gate, while protocol conformance checks have an a-priori expectation of exact equality.

## Command plan

1. Verify all pre-raw freeze closures and sidecar hashes, imported/fresh code hashes, governance hashes, and the selective allowlist hash; inspect exact local raw paths and sizes only, without a separate verifier raw-content hash.
2. Verify the exact launcher environment: Python executable/version, conda environment, package versions, neurolib origin/version/commit/clean tracked diff, permitted untracked closure, MNE call signatures, and required cache/thread variables. Do not read `environment.yml`.
3. Run the 24 data-free adapter tests under the exact conda launcher and campaign-local caches with all specified thread variables set to one.
4. Run exactly once, from `CODE/R4_EXECUTION`, `python cli.py steward --project-root D:\Year3_Mao_Projects\sleep_loop --output-root <R4>\RUNS --output-child VERIFIER_STEWARD`. This is the verifier's single reserved MNE logical read of each exact Night-1 file.
5. Independently run once each: `synthetic-m1` to `VERIFIER_SYNTHETIC`; `select-fit` using `VERIFIER_STEWARD/FIT_PAYLOAD.json` to `VERIFIER_FIT`; compute the exact verifier `SELECTION_FREEZE.json` SHA-256; then `evaluate-heldout` using `VERIFIER_STEWARD/HELDOUT_PAYLOAD.json` and that exact hash to `VERIFIER_HELDOUT`.
6. Freeze/hash the verifier outputs. Only then open the six designated worker output artifacts under `RUNS/SYNTHETIC_WORKER`, `RUNS/FIT_WORKER`, and `RUNS/HELDOUT_WORKER`, and compare exact raw-derived payload hashes/digests, split, selection/calibration/candidate, metrics, denominators, bootstrap values, baseline, and gate decisions. Do not read worker narrative or `MECHANISTIC/WORKER_PACKET`.
7. Assess message-race behavior only against the frozen no-ACK protocol after independent results, then write the independent JSON/Markdown report and artifact manifest.

## Likely failure modes

- Any frozen file, sidecar, allowlist, code, protocol, claim, or environment identity mismatch.
- Wrong launcher, Python/package/neurolib identity, dirty tracked neurolib diff, impermissible untracked/runtime closure, MNE version/signature mismatch, or cache/thread settings not fixed as specified.
- Accidental raw access outside the two exact Night-1 paths, multiple MNE logical reads, direct EDF-header parsing, or any prohibited outcome/protected/Night-2/prior-campaign access.
- Fewer than four eligible N3 epochs, fewer than two valid epochs in either role, nonfinite/flat/short/out-of-range epoch handling that changes assignment, channel-rule deviation, or partition digest/disjointness mismatch.
- FIT reading HELDOUT, HELDOUT opening before exact freeze-hash verification, candidate/calibration mutation, retry/repair, overwritten output child, schema mismatch, or noncanonical serialization.
- Synthetic nondeterminism or gate mismatch; worker/verifier differences in candidate, calibration, NRMSE, MSE, relative improvement, bootstrap interval, denominators, threshold strictness, baseline, or gate status.
- A procedural message race inconsistent with the frozen no-ACK protocol. This will be classified separately from numeric reproduction and only after independent results are frozen.

## Prospective classification rule

`NUMERIC_REPRODUCTION`, `PROCEDURAL_FIDELITY`, `DATA_SPLIT_AGREEMENT`, `BASELINE_AGREEMENT`, and `CLAIM_AGREEMENT` will each be reported independently. `MATERIAL_DISAGREEMENT` is true if any discrepancy could change a candidate, denominator, baseline, gate, claim, data boundary, or reproducibility conclusion. Overall `VERIFIED` requires exact/relevant agreement and faithful procedure with `MATERIAL_DISAGREEMENT = FALSE`; `PARTIAL` is reserved for bounded non-material/incomplete agreement that does not support full G7; `FAILED` applies to material numeric, scientific, data-boundary, or procedural disagreement or invalid execution.
