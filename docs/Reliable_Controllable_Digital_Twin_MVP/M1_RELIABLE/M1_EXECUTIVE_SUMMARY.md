# M1 Executive Summary - Why C006 Is the Reliable Twin Candidate

## Decision

M1 concluded `COSTA_DISSERTATION_MVP=PASS` for the narrow frozen `M1/C0/P0/DISCOVERY_PLUS_INDEPENDENT_REPRODUCTION` objective and selected **C006** as the mechanistic twin candidate.

## Candidate identity and mechanistic parameters

C006 belongs to the prospectively frozen 16-candidate thalamocortical bank. Its distinguishing parameters are:

| Parameter | C006 value |
|---|---:|
| Cortical gain | 0.82 |
| Cortex-to-thalamus coupling (`g_CT`) | 0.95 |
| Thalamus-to-cortex coupling (`g_TC`) | 1.10 |
| Relay-reticular inhibition (`w_TR`) | 1.25 |

The four-state model comprises cortical excitatory, cortical inhibitory, thalamic relay, and thalamic reticular activity with explicit bidirectional thalamocortical coupling. The observable is a dimensionless model-space proxy, not voltage or a validated scalp forward model.

## Why C006 was selected

The frozen rule selected the candidate with minimum NRMSE on FIT epochs after FIT-only affine calibration, with ties broken by candidate ID. Across 110 FIT epochs:

- C006 had the lowest FIT NRMSE: **0.2546119**.
- The next-lowest candidate, C014, had FIT NRMSE 0.2759201.
- Matched non-personalized C000 had FIT NRMSE 0.4625683.
- C006 calibration was frozen at intercept **-0.0536049** and slope **0.8754980**.

No candidate was added after freeze and the held-out evaluation used the frozen FIT calibration.

## Held-out credibility evidence

Across 110 HELDOUT epochs:

| Metric | Frozen C006 | Matched C000 |
|---|---:|---:|
| HELDOUT NRMSE | 0.2478453 | 0.4637754 |

- Relative NRMSE improvement: **0.4655920 (46.5592%)**, above the frozen 0.05 margin.
- Mean paired epoch MSE advantage: **0.0903471**.
- 1,000-draw paired bootstrap 95% CI: **[0.0798002, 0.1014646]**.
- All M1 gates G1-G7 passed, including independent reproduction.
- `MATERIAL_DISAGREEMENT = FALSE` in final adjudication.

## Frozen provenance

C006 was frozen before HELDOUT access. Key identities were:

- Selection freeze SHA-256: `3D8C3D035F074CED2D608132864996D4AF9920C52889A11FE7C8E7DDA5DD8BBF`.
- Protocol SHA-256: `338366593A2A4C8626349D70AA38306C5931F0FF4D22FD96DB546DD41505E3C3`.
- Mechanistic model SHA-256: `775A3961093C386A1E7DF8F0D033963EAB684F2566AE206E5BB6DB4A6CD03869`.

These same three identities were embedded and verified in every fresh C1 result, establishing that controller confirmation used the same frozen C006.

## Claim boundary

M1 supports model-internal personalized mechanistic consistency under a one-subject, one-Night-1, average four-band, dimensionless proxy analysis. It does not support population generalization, epoch-specific dynamics, a human causal mechanism, scalp-voltage validity, intervention response, clinical validity, or dissertation sufficiency by itself.

## Dissertation-ready interpretation

> We compared a frozen candidate bank; C006 emerged under the pre-specified FIT rule; held-out and independent-reproduction evidence supported the bounded use case; and C006 was frozen before controller research.
