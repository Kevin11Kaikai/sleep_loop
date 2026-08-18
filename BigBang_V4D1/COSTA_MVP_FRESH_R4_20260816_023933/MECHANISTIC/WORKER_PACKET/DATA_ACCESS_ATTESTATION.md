# COSTA R4 Scientific Worker Data-Access Attestation

Worker role: `/root/costa_campaign_root_r4/r4_scientific_worker`

Campaign: `COSTA_MVP_FRESH_R4_20260816_023933`

## Access sequence

1. Verified the PRE_RAW_EXECUTION_FREEZE and DATA_PARTITION_FREEZE digests and their companion SHA files.
2. Verified every frozen-code/governance and steward-manifest entry.
3. Read the steward receipt and FIT payload. The HELDOUT payload was hashed only; its content was not opened.
4. Ran the synthetic command exactly once.
5. Ran FIT selection exactly once; inspected only the FIT selection freeze and manifest.
6. Computed and verified selection-freeze SHA-256 `3D8C3D035F074CED2D608132864996D4AF9920C52889A11FE7C8E7DDA5DD8BBF`, selected candidate `C006`, and sent the required checkpoint.
7. Ran HELDOUT evaluation exactly once with the exact selection-freeze path and expected digest. The generated result records that the digest was verified before HELDOUT payload opening.
8. Performed read-only inspection and hashing of the frozen worker outputs. No scientific rerun or repair was performed.

## Prohibited material attestation

The worker did not read, hash, or list original EDF files. The worker did not access Night-2, protected-derived, sealed/fresh-final, any R1/R2/R3 path, prior scientific outcomes or reports, or SCRATCH-root drafts. No protected-derived or alternative-data source was used.

## Scientific immutability attestation

No candidate, calibration, operator, threshold, seed, partition assignment, exclusion rule, or missingness rule was altered. No outcome-guided change occurred. Every scientific command had attempt count `1`; no retry or repair followed any scientific result.

## Ordering and message-race note

The selection checkpoint preceded HELDOUT execution. A later HOLD message arrived only after HELDOUT completion. The campaign root classified this as a nonmaterial timing deviation because the frozen no-ACK protocol required the checkpoint but did not require acknowledgement before proceeding. No rerun was performed.

## Status

- Worker execution validity: `VALID_G1_TO_G6_PENDING_FRESH_VERIFIER`
- G7: `NOT_RUN`
- Claim assignment: `PENDING_VERIFICATION`
