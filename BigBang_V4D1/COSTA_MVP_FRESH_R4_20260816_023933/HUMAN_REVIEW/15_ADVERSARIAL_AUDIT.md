# Adversarial audit

Final adversarial verdict: no material disagreement; zero material findings; M1 eligible; protected access `PASS_NONE`; all G1-G7 pass.

Five nonmaterial findings were retained:

1. Initial denylist outer-path/inner-hash mismatch, corrected outcome-blind by the seven-file allowlist.
2. `STAGE_STATE.json` and campaign status remain initialization-era records.
3. The `environment.yml` label is broader than the supported unread/absent-from-declared-closure evidence; this is not exhaustive OS-open forensics.
4. Access and role ledgers are append-only by policy but not cryptographically entry-chained.
5. Root registration lagged the already-issued no-ACK HELDOUT command, while the required checkpoint and freeze verification remained in correct scientific order.

The message race is nonmaterial: no frozen ACK gate existed, the checkpoint preceded the single HELDOUT command, the selection freeze was verified before payload opening, and no retry or scientific change occurred.
