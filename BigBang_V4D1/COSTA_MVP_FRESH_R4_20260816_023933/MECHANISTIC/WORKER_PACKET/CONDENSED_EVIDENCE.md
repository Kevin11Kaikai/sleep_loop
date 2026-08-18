# COSTA R4 M1 Worker Condensed Evidence

## Outcome

The frozen scientific worker completed the synthetic, FIT-selection, and HELDOUT stages exactly once each. All commands exited `0`, all output manifests matched their artifacts and frozen inputs, and worker-side gates G1-G6 passed. G7 remains `NOT_RUN` for the fresh verifier, and claim assignment is `PENDING_VERIFICATION`.

## Integrity and selection binding

- PRE_RAW_EXECUTION_FREEZE SHA-256: `89CA8649AB1383F6EFC8BD467ECF7EDC872FA099F945A70AE73735EF3C624A5B`
- DATA_PARTITION_FREEZE SHA-256: `FB4216DEC7C2EC2F1AD4F91BED94961302592E5CEF0F58EF673FB973CC3B3F89`
- All 31 pre-execution and steward-manifest hash entries matched.
- HELDOUT payload content was not opened during preverification; only its frozen SHA-256 was checked.
- FIT selected `C006` at FIT NRMSE `0.25461189482219504`, compared with C000 FIT NRMSE `0.46256826111763194`.
- Selection-freeze SHA-256: `3D8C3D035F074CED2D608132864996D4AF9920C52889A11FE7C8E7DDA5DD8BBF`.
- The selection checkpoint was sent before HELDOUT execution, and the HELDOUT result records verification of that exact digest before payload opening.

## Frozen commands

Each command used attempt count `1`, exited `0`, and ran from `CODE\R4_EXECUTION` with the frozen conda launcher and campaign-local TEMP/TMP/Numba cache.

```text
C:\Users\YUS190\AppData\Local\anaconda3\condabin\conda.bat run --no-capture-output -n neurolib python cli.py synthetic-m1 --output-root D:\Year3_Mao_Projects\sleep_loop\BigBang_V4D1\COSTA_MVP_FRESH_R4_20260816_023933\RUNS --output-child SYNTHETIC_WORKER
```

```text
C:\Users\YUS190\AppData\Local\anaconda3\condabin\conda.bat run --no-capture-output -n neurolib python cli.py select-fit --fit-payload D:\Year3_Mao_Projects\sleep_loop\BigBang_V4D1\COSTA_MVP_FRESH_R4_20260816_023933\RUNS\STEWARD_WORKER\FIT_PAYLOAD.json --output-root D:\Year3_Mao_Projects\sleep_loop\BigBang_V4D1\COSTA_MVP_FRESH_R4_20260816_023933\RUNS --output-child FIT_WORKER
```

```text
C:\Users\YUS190\AppData\Local\anaconda3\condabin\conda.bat run --no-capture-output -n neurolib python cli.py evaluate-heldout --heldout-payload D:\Year3_Mao_Projects\sleep_loop\BigBang_V4D1\COSTA_MVP_FRESH_R4_20260816_023933\RUNS\STEWARD_WORKER\HELDOUT_PAYLOAD.json --selection-freeze D:\Year3_Mao_Projects\sleep_loop\BigBang_V4D1\COSTA_MVP_FRESH_R4_20260816_023933\RUNS\FIT_WORKER\SELECTION_FREEZE.json --expected-freeze-sha256 3D8C3D035F074CED2D608132864996D4AF9920C52889A11FE7C8E7DDA5DD8BBF --output-root D:\Year3_Mao_Projects\sleep_loop\BigBang_V4D1\COSTA_MVP_FRESH_R4_20260816_023933\RUNS --output-child HELDOUT_WORKER
```

No exact process start/end timestamps were captured. Artifact writes occurred at `2026-08-16T07:20:04.170Z`, `2026-08-16T07:20:15.011Z`, and `2026-08-16T07:21:10.705Z` for synthetic, FIT, and HELDOUT respectively.

## Gate evidence

- G1 `PASS` (14/14): frozen environment/path/repository constraints passed; the core digest matched across 2/2 repetitions.
- G2 `PASS` (5/5): truth recovered in 4/4 synthetic subjects; maximum truth HELDOUT NRMSE `0.0005789052687937201`; minimum relative improvement versus zero coupling `0.9961873232475084`.
- G3 `PASS` (4/4): 4/4 subjects beat the population comparator and 4/4 beat the wrong-subject comparator; both median checks passed.
- G4 `PASS` (3/3): 256-draw block bootstrap paired MSE advantage `14.24890996213641`, 95% CI `[4.524518898122416, 27.019127347457072]`.
- G5 `PASS` (5/5): complete fixed four-source operator with explicit arbitrary-model-unit and no-physical-forward-model/scalp-causality limitations.
- G6 `PASS` (6/6): 220 eligible N3 epochs; FIT 110/110 valid and HELDOUT 110/110 valid, with 0/0 excluded. Personalized C006 HELDOUT NRMSE was `0.24784525436798707` versus C000 `0.46377535006926635`, relative improvement `0.46559200627853425` against the `0.05` margin. The paired epoch bootstrap used n=110, 1000 repetitions, seed 26081604, and produced 95% CI `[0.07980018864484562, 0.10146456173015034]`, whose lower bound is strictly positive.
- G7 `NOT_RUN`: pending fresh independent verifier.

## Message-race disposition

A HOLD message arrived after HELDOUT completion. The campaign root classified this as a nonmaterial timing deviation: the frozen protocol and original instruction required a pre-HELDOUT checkpoint, not an acknowledgement gate, and that checkpoint was sent before execution. No rerun or repair occurred.

## Claim boundary

Worker-side execution is `VALID_G1_TO_G6_PENDING_FRESH_VERIFIER`. The claim remains `PENDING_VERIFICATION`. These results are limited to proxy-level, one-subject, Night-1 model-internal personalized mechanistic consistency and do not establish population generalization, a validated human mechanism, scalp EEG causality, or a clinical digital twin.
