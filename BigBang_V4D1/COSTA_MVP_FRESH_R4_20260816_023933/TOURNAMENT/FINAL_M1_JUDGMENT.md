# COSTA R4 final M1 judgment

`M1_ELIGIBLE = YES`

`MATERIAL_DISAGREEMENT = FALSE`

Final recommendation: **M1 / C0 / P0 / DISCOVERY + INDEPENDENT_REPRODUCTION**.

All mandatory gates G1-G7 pass. The selected C006 candidate improved heldout NRMSE by `0.4655920063` over matched C000 against the frozen `0.05` margin, and the paired epoch-bootstrap 95% interval was `[0.0798001886, 0.1014645617]`. The independent verifier reproduced the split, raw-derived payloads, candidate/calibrations, comparator, metrics, thresholds, and all six worker artifacts exactly.

The protected-access verdict is `PASS_NONE`; the authorized Night-1 logical raw-open budget is exhausted and closed. The pre-HELDOUT message race is nonmaterial because the checkpoint preceded the single command, exact freeze verification preceded payload opening, no ACK was frozen as a requirement, and no retry or scientific change occurred.

Recommended `COSTA_DISSERTATION_MVP = PASS` only under the narrow M1/P0/independent-reproduction scope. C1 is not assigned. This judgment does not establish dissertation sufficiency, M2, P1/P2, C1/C2, a human causal mechanism, population generalization, scalp-voltage validity, Night-2 prediction, sealed validation, or clinical efficacy.
