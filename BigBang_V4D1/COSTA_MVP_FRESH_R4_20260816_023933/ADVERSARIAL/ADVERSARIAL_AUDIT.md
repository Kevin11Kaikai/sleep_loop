# COSTA R4 adversarial audit

Final adversarial verdict: **no material disagreement**. All mandatory gates M1-G1 through M1-G7 pass under the frozen, narrow M1/P0/independent-reproduction protocol. `M1_ELIGIBLE = YES`.

## Recommendation

`M1 / C0 / P0 / DISCOVERY + INDEPENDENT_REPRODUCTION`

Maximum wording: “Model-internal personalized mechanistic consistency was independently reproduced for a frozen proxy-level one-subject Night-1 discovery analysis.” C1 is not assigned; the synthetic coupling ablation is gate evidence, not a separately adjudicated control claim.

Recommended `COSTA_DISSERTATION_MVP = PASS` only for the user's narrow M1/P0/independent-reproduction scope. This is not a finding of dissertation sufficiency.

## Gate audit

| Gate | Decision | Adversarial basis |
|---|---|---|
| G1 | PASS | Imported and R4 code, protocol, claims, Python/package/module/commit identities, clean tracked diff, thread/cache settings, and repeated core digest all close exactly. The untracked `environment.yml` was attested unread and absent from the declared command/runtime/module-path closure; this is not described as exhaustive OS-handle forensics. |
| G2 | PASS | Truth 4/4; maximum truth NRMSE 0.000579; minimum ablation improvement 0.9962; null advantage 0.09996; non-isomorphic NRMSE 1.0039. All clear frozen thresholds. |
| G3 | PASS | All four synthetic subjects beat C000 and the frozen wrong-subject cycle; empirical C000 used identical calibration and scoring. |
| G4 | PASS | Synthetic block-bootstrap and empirical paired-epoch-bootstrap lower bounds are strictly positive; empirical n=110, 1,000 draws. |
| G5 | PASS | Four-population equations and state-to-source-to-causal-baseline-to-mau operator are coherent; the shared output is dimensionless log-relative band power. Proxy/claim limits are explicit. |
| G6 | PASS | Frozen Night-1 N3 split reproduced exactly: 110 FIT, 110 HELDOUT, zero excluded. C006 was hash-bound before HELDOUT. Relative improvement 0.465592 exceeds 0.05; CI lower bound 0.079800 is positive. |
| G7 | PASS | Blind verifier reproduced all raw-derived payloads and six worker artifacts byte-exactly. The message race is nonmaterial because no frozen ACK gate existed, the checkpoint preceded the single HELDOUT command, and there was no retry/change. |

## Nonmaterial findings

1. The denylist retains the initial outer-path/inner-hash mismatch even though the outcome-blind import audit corrected it and the later exact seven-file allowlist governed execution.
2. `STAGE_STATE.json` and campaign status remain initialization-era records; this is an operational closeout defect, not a scientific-state disagreement.
3. The `environment.yml` check label is broader than the evidence computed. The supported claim is unread plus absent from the declared runtime/config/module-path closure, backed by exact environment/module identity—not literal exhaustive OS-open proof.
4. Access and role ledgers are append-only by policy but not entry-chained cryptographic logs.
5. Root registration lagged the already-issued no-ACK HELDOUT command; the required checkpoint and freeze verification still occurred in the correct scientific order.

## Scientific limitations

- Synthetic truth comes from the same model family with small noise and is intentionally favorable; it is not biological validation.
- C000 is a frozen generic comparator, not a population-estimated model.
- The empirical prediction is a candidate-specific four-band vector repeated across epochs, so the evidence concerns average spectral-feature consistency, not epoch-specific dynamics.
- The frozen empirical bootstrap treats epochs as exchangeable and does not account for within-night serial dependence.
- Reproduction used the same frozen code/data/environment, with independent blind execution rather than an independent implementation.
- One subject and one Night-1 cannot support population generalization or empirical wrong-subject specificity.

No claim is made for M2, P1/P2, C1/C2, human causality, population generalization, Night-2 prediction, sealed validation, clinical efficacy, or dissertation sufficiency.

The 44-file permitted-evidence rehash is in `ADVERSARIAL/INPUT_REHASH_MANIFEST.json` (SHA-256 `6BACBED61060CAD07CC2357F29A14EF22C51F0E5A9EB6548A1EBE1258694BAA6`).
