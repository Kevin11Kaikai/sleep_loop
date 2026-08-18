# Historical Import Audit — R4 selective v2

Verdict: `CLEAN_ALLOWLISTED` for seven exact files only.

The outer candidate manifest (`A1C7…47E`), inner manifest (`42EE…3A8`), and all twelve candidate hashes were independently checked. The outer execution bundle was rejected because its runner would traverse a non-allowlisted repair artifact named by the inner manifest. R4 therefore imports neither the historical runner nor either freeze artifact.

Independent file-level review found no empirical, protected-derived, sealed, Night-2, or outcome-dependent scientific content in the seven selected model/protocol/claim/test files. Their exact hashes are frozen in `IMPORT_ALLOWLIST/V4D1_IMPORT_ALLOWLIST_V2.json`.

Governance correction: the original R4 brief mislabeled the inner `42EE…3A8` hash as the outer-manifest hash. The independently observed outer hash is `A1C7…47E`. This correction did not authorize any new historical content.

No R3 outcome, report, state, run output, repair record, or failed data adapter is an R4 input.
