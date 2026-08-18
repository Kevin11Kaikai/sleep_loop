# OAV-001 — Import-Allowlist Permission-Fidelity Deviation

## Summary

Official import allowlist version 2 added the fresh isolated `Evidence Sanitizer` role and narrow value-free sanitization purposes to three records already classified `CLEAN_ALLOWLISTED`. The Evidence Sanitizer then read and hashed exactly those three records plus the official allowlist. Independent validation subsequently found that the role and purpose extension had not yet been bound to a separate frozen auditor decision, so `IMPORT_ALLOWLIST=FAIL` was issued and the sanitizer was immediately interrupted.

## Exact access attestation

The Evidence Sanitizer reported opening and hashing only:

1. `IMPORT_ALLOWLIST/V4D1_IMPORT_ALLOWLIST.json`;
2. exact R4 `FINAL_CLAIM_MATRIX.json`;
3. exact R4 `ARTIFACT_MANIFEST.json`;
4. exact R4 `FINAL_CLOSEOUT.json`.

It did not open `FINAL_REPORT.md`, any manifest-referenced artifact, any directory listing, Night-2, Sealed Bank, protected-derived, fresh-final, confirmatory, ambiguous, or quarantined material. It wrote no partial output. It performed exact `Test-Path` nonexistence checks only for the three intended current-campaign destination files.

## State at access

- All three R4 source files had already been independently classified `CLEAN_ALLOWLISTED` with exact SHA-256 bindings.
- Official allowlist version 2 explicitly named `Evidence Sanitizer` as a permitted role.
- Independent fidelity revalidation of that role extension had not yet completed.
- No scientific outcome-bearing Phase 2 run had begun and no Phase 2 target outcome was visible.

## Containment and repair

- Sanitizer execution was interrupted immediately after OAV-001 was reported.
- No sanitizer output existed to quarantine or delete.
- The Historical Import Auditor independently approved the exact narrow role/purpose scope in `IMPORT_ALLOWLIST/SANITIZER_PERMISSION_AMENDMENT.json`.
- Official allowlist version 3 binds that amendment, uses its exact purpose language, retains default deny and no cleanliness inheritance, and is pending independent revalidation.

## Provisional classification

- Protected access: `NONE`
- Raw-data access: `NONE`
- Scientific-outcome exposure: `NONE`
- Outcome-guided change: `FALSE`
- Deviation class: `OUTCOME_BLIND_GOVERNANCE_PROCEDURAL_FIDELITY_DEVIATION`
- Materiality: `PENDING_INDEPENDENT_GOVERNANCE_ADJUDICATION`
- Scientific repository inspection: `NOT_AUTHORIZED`

This record does not self-certify nonmateriality. An independent governance auditor must adjudicate materiality and the repaired version-3 allowlist before sanitization resumes.
