# COSTA R4 M1 independent verification

Campaign: `COSTA_MVP_FRESH_R4_20260816_023933`

## Categorical G7 result

| Category | Result |
|---|---|
| `NUMERIC_REPRODUCTION` | `PASS_EXACT` |
| `PROCEDURAL_FIDELITY` | `PASS` |
| `DATA_SPLIT_AGREEMENT` | `PASS_EXACT` |
| `BASELINE_AGREEMENT` | `PASS_EXACT` |
| `CLAIM_AGREEMENT` | `PASS` |
| `MATERIAL_DISAGREEMENT` | `FALSE` |
| `OVERALL` | `VERIFIED` |

This is a verifier gate assessment, not an M1 assignment.

## Independent reproduction

`PREINTERPRETATION.md` was written and hashed before any worker outcome exposure (`A35A799E4444095A2629701A2547B2A7AC98A3ABEDCF93091D904E50F8F86312`). All verifier outputs were then completed once and frozen before unblinding (`VERIFIER_OUTPUT_FREEZE.json`: `60593BEFE0601AD2B4A5AA69FF868527F190697B3B2EC9E361B36E47A6454E25`). The 24 data-free adapter tests passed. Frozen imports, execution code, governance inputs, sidecars, allowlist, exact launcher environment, package versions, neurolib commit, and clean tracked diff matched.

The verifier steward used the reserved raw budget exactly once per exact Night-1 file through the two frozen MNE calls. No separate verifier raw-content hash was computed. Path/size metadata matched; no retry, repair, or scientific change occurred.

The raw-derived split reproduced exactly:

- Partition digest: `D9D093B9F28A18C2C46EAC4B2200BF58CE2860D5834595F797C87005D8C62574`
- FIT: 110 valid, 0 excluded; payload `F09D71800F49C79734D2A7A492DBEBEDB18C746418CF6D5F2393CB5F75A3FC2B`
- HELDOUT: 110 valid, 0 excluded; payload `3169F2361F737841BCB46393D24DFC209946FEB8E46608634ADC05E9B5C465EC`
- Steward receipt: `C8104174A25A2B13D2B7F1E40D07BD95AB44E26930D3C8A7B6865104B175CEC7`

Synthetic G1-G5 all passed with repeated core digest `A3CE6D2D01FDEDBDFBC4C19E4D685F4CBB5CC67DC6CBF958AD395A02EB9A97D7`.

FIT selected `C006` from the frozen 16-candidate bank. Its calibration was slope `0.8754979793652278`, intercept `-0.05360488354161275`, FIT NRMSE `0.25461189482219504`. The matched nonpersonalized baseline remained `C000`, slope `1.4427380777867436`, intercept `0.15398754213671761`, FIT NRMSE `0.46256826111763194`. The selection freeze hash was `3D8C3D035F074CED2D608132864996D4AF9920C52889A11FE7C8E7DDA5DD8BBF` and was verified before HELDOUT open.

HELDOUT personalized NRMSE was `0.24784525436798707`; matched `C000` NRMSE was `0.46377535006926635`; relative improvement was `0.46559200627853425` against the `0.05` threshold. The 1,000-draw paired epoch-bootstrap interval was `[0.07980018864484562, 0.10146456173015034]`, with a strictly positive lower bound. G6 passed. FIT and HELDOUT NRMSE denominators were respectively `0.7505025281698362` and `0.766789538672201`, using observed sample SD plus `1e-12`.

## Worker comparison and procedure

Only after the verifier-output freeze were the six permitted worker artifacts opened. Every worker report/manifest was byte-identical to its verifier counterpart:

- Synthetic report `62C5CA3CEAAD4382A1B1ADA872E447C304C3593CB1B9120613C951DE86113024`; manifest `1B89DB48F367C71C471446AB53EF9EDAA5E20F4904F32A7DDE0DDEF71C8EE01C`
- FIT freeze `3D8C3D035F074CED2D608132864996D4AF9920C52889A11FE7C8E7DDA5DD8BBF`; manifest `B908287828DF16224BC04C35038E7315EC8788D8F7159971E17353F97871440D`
- HELDOUT result `E0AD4B1855E6BD570D068D4AC3B962397BAAEF5F21AC917E8C35E238B7004E95`; manifest `880F3369D5862A8DAA46D88F1B7C076810358E03139C68BB19156F81457965D5`

The message race is `NONMATERIAL`: the checkpoint preceded HELDOUT, the frozen protocol required no Root ACK, and the command ran once with no scientific change or retry. It does not alter procedural fidelity.

No R1/R2/R3 content, Night-2, protected-derived, sealed/fresh-final, prior report/state/outcome, SCRATCH root draft, `MECHANISTIC/WORKER_PACKET`, or persuasive worker narrative was read. All outcome access occurred only after the prescribed blind reproduction freeze.

## Scope

The maximum conclusion remains proxy-level, single-subject Night-1, model-internal personalized mechanistic consistency with independent reproduction. It does not support population generalization, empirical wrong-subject specificity, Night-2 prediction, scalp EEG causality, clinical efficacy, treatment/stimulation validity, or medical decision support.
