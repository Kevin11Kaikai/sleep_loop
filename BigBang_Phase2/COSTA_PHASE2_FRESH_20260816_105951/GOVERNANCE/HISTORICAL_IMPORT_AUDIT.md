# Historical Import Audit — Four Exact R4 Candidates

- Current campaign: `COSTA_PHASE2_FRESH_20260816_105951`
- Auditor role: `Historical Import Auditor`
- Historical source campaign: `COSTA_MVP_FRESH_R4_20260816_023933`
- Audit boundary: the four exact candidate files named in `IMPORT_ALLOWLIST/IMPORT_CANDIDATES.json`
- Decision effect: advisory import-eligibility record only; the official allowlist and shared `STATE` remain unchanged

## Scope and method

The auditor opened and SHA-256 hashed only the four exact R4 candidates and read only the four authorized current-campaign controls. No directory was listed or recursively enumerated. No artifact named by a manifest or closeout record was opened. No protected, Night-2, Sealed Bank, protected-derived, fresh-final, confirmatory, quarantined, or ambiguous payload was accessed. This review performs governance classification only and makes no new scientific interpretation.

## Authorized-summary verification

The four records consistently identify the narrow R4 result as exactly:

`M1/C0/P0/DISCOVERY_PLUS_INDEPENDENT_REPRODUCTION`

The records also consistently establish that raw-data authorization is **CLOSED** and that no further raw or protected access is authorized. This audit imports no numerical result, candidate-level result, threshold, interval, denominator, or other exact scientific outcome.

## File decisions

| Exact candidate | Observed SHA-256 | Evidence class | Development-only provenance | Contamination status | Review status |
|---|---|---|---|---|---|
| `D:\Year3_Mao_Projects\sleep_loop\BigBang_V4D1\COSTA_MVP_FRESH_R4_20260816_023933\HUMAN_REVIEW\FINAL_REPORT.md` | `093E0644DA54445D65F0332A4147352FD35B791E24C53396840F46F1EE0C5E22` | Historical development scientific narrative containing exact numerical outcomes | Closed R4 development record; not confirmatory, fresh-final, Phase 2, or population evidence | No protected payload or value detected, but exact scientific outcomes exceed the authorized summary | **QUARANTINED** |
| `D:\Year3_Mao_Projects\sleep_loop\BigBang_V4D1\COSTA_MVP_FRESH_R4_20260816_023933\HUMAN_REVIEW\FINAL_CLAIM_MATRIX.json` | `0ED307F6866660EA7E3AC2330DE07D1D8A6222C0E42941855FC1CDF8353FCC8E` | Historical development categorical claim-and-scope record | Closed R4 development record; prior categorical provenance only | No protected payload, protected value, or out-of-scope numerical outcome detected | **CLEAN_ALLOWLISTED** |
| `D:\Year3_Mao_Projects\sleep_loop\BigBang_V4D1\COSTA_MVP_FRESH_R4_20260816_023933\HUMAN_REVIEW\ARTIFACT_MANIFEST.json` | `7724D06C468646F4E278548A2CC009DCCD51FB7CD7369012C606051A5E135B2B` | Governance integrity metadata and non-self-referential hash catalog | Closed R4 package-integrity provenance only | No protected payload or value detected; referenced artifacts were not opened or cleared | **CLEAN_ALLOWLISTED** |
| `D:\Year3_Mao_Projects\sleep_loop\BigBang_V4D1\COSTA_MVP_FRESH_R4_20260816_023933\STATE\FINAL_CLOSEOUT.json` | `E4E2BC1755551D001BA6D71849E94E4DD00E7A33C73C8E642C99CC5456A283FA` | Governance closeout and bounded categorical provenance record | Closed R4 development and access-closeout provenance only | No protected payload, protected value, or out-of-scope numerical outcome detected | **CLEAN_ALLOWLISTED** |

## Permitted roles and purposes

### `FINAL_REPORT.md` — QUARANTINED

- Source-file access remains limited to the `Clean Governance Auditor` and `Historical Import Auditor` for bounded governance audit.
- The only transferable content is the authorized categorical summary recorded above.
- The file is not permitted for Phase 2 scientific review, design, selection, tuning, thresholding, comparison, or evidentiary support because it embeds exact prior numerical outcomes beyond that summary.

### `FINAL_CLAIM_MATRIX.json` — CLEAN_ALLOWLISTED

- Permitted source-file roles before any later gate release: `Clean Governance Auditor` and `Historical Import Auditor`.
- Permitted purpose: verify and carry forward only the categorical R4 assignment, development-only scope limits, and closed raw-access status.
- It is not confirmatory evidence and cannot support M2, P1, P2, protected, clinical, or population claims.

### `ARTIFACT_MANIFEST.json` — CLEAN_ALLOWLISTED

- Permitted source-file roles: `Clean Governance Auditor` and `Historical Import Auditor`.
- Permitted purpose: R4 package-integrity and provenance checking only.
- Its entries do not confer cleanliness, access permission, or evidentiary status on any referenced artifact. Every referenced path remains unopened and default-denied unless separately and exactly authorized.

### `FINAL_CLOSEOUT.json` — CLEAN_ALLOWLISTED

- Permitted source-file roles before any later gate release: `Clean Governance Auditor` and `Historical Import Auditor`.
- Permitted purpose: verify campaign closure, the authorized categorical summary, and the closed raw/protected-access posture.
- Embedded hashes identify historical governance records only; they neither authorize opening those records nor import their contents.

For every `CLEAN_ALLOWLISTED` decision above, scientific-role access remains contingent on an official exact-path allowlist entry and explicit Root Orchestrator gate release. This audit does not itself provide either action.

## Control bindings

| Current-campaign control | Observed SHA-256 |
|---|---|
| `GOVERNANCE/PROTECTED_DENYLIST.json` | `AF3AA8AC66775885B09F47C425DE1BE3F515B877FD7DDDC44DCA32F1824EA060` |
| `IMPORT_ALLOWLIST/IMPORT_CANDIDATES.json` | `D706685F051FBBD63C6C6BB57E7D7B7DD44371F510F50C24E9293A5ABB396EB8` |
| `COMMON/CAMPAIGN_IDENTITY.json` | `29DAD8795AD9B2A12298627F7BAFD2EB2022C0C1F03C031ED1428ED0FCE5B266` |
| `GOVERNANCE/CLEAN_ROOT_AUDIT.md` | `D097353C680A3E1FED4A33F0D64EF65419827A6060CFF0004C967E10C60ABE09` |

## Final governance verdict

- Exact categorical R4 result: **VERIFIED**.
- Raw-data authorization: **CLOSED**.
- Protected contamination in the three clean-listed records: **NOT DETECTED WITHIN THE OPENED FILES**.
- Narrative file containing prior exact numerical outcomes: **QUARANTINED**.
- Referenced or neighboring artifacts: **NOT REVIEWED; NO ACCESS AUTHORIZED**.
- Official allowlist mutation: **NOT PERFORMED**.
- Shared-state mutation: **NOT PERFORMED**.

---

## OFFICIAL ALLOWLIST VALIDATION

### Validation boundary

This validation compared only:

1. `IMPORT_ALLOWLIST/V4D1_IMPORT_ALLOWLIST.json`, version 2; and
2. the frozen `IMPORT_ALLOWLIST/AUDIT_DECISIONS.json`.

No R4 source file was reopened, no repository science was inspected, and neither shared `STATE` nor the official allowlist was modified.

### Integrity checks

- Expected official-allowlist SHA-256: `4BF2D608227F590D1BAC0DC9E8CB1F078CED92CE84398102C575AC131FA32B69`
- Observed official-allowlist SHA-256: `4BF2D608227F590D1BAC0DC9E8CB1F078CED92CE84398102C575AC131FA32B69`
- Hash result: **MATCH**
- Frozen audit-decisions SHA-256 observed for this comparison: `DA50B67A50118A6BBB6BD6F65D141EB75A5B1A7931DAF5FF1D4B13458C94C3E7`
- The allowlist's `audit_receipts.audit_decisions.sha256` equals that observed frozen-decision hash: **MATCH**

### Field-by-field implementation verdict

| Requirement | Verdict | Basis |
|---|---|---|
| Exact paths | **PASS** | All four exact paths match the frozen decisions. |
| SHA-256 values | **PASS** | All four candidate hashes match the frozen decisions. |
| Review statuses | **PASS** | The three `CLEAN_ALLOWLISTED` and one `QUARANTINED` decisions are reproduced exactly. The global default is `DENY`, and only `CLEAN_ALLOWLISTED` is designated as an allowed review status. |
| Evidence classes | **PASS** | Every evidence-class value matches exactly. |
| Development-only provenance | **PASS — SEMANTIC ENCODING** | Each compact allowlist provenance string preserves the corresponding frozen provenance classification and its non-confirmatory, non-fresh-final, non-Phase-2, or non-scientific restriction as applicable. |
| Contamination status | **PASS — SEMANTIC ENCODING** | The compact allowlist labels preserve the frozen no-detected-protected-value findings, the narrative report's authorized-summary scope excess, and the unrebutted default-deny status of referenced artifacts or records. |
| Prohibited-use restrictions | **PASS — SEMANTIC ENCODING** | Compound frozen prohibitions are split into equivalent individual items without removing a prohibited category. |
| Permitted roles | **FAIL** | `Evidence Sanitizer` is added to `FINAL_CLAIM_MATRIX.json`, `ARTIFACT_MANIFEST.json`, and `FINAL_CLOSEOUT.json`; that role is absent from each corresponding frozen `permitted_roles` list. |
| Permitted purposes | **FAIL** | The allowlist adds `value-free sanitized substrate derivation` to `FINAL_CLAIM_MATRIX.json` and `FINAL_CLOSEOUT.json`, and adds `exact candidate path and expected-hash discovery for later independent audit without opening` to `ARTIFACT_MANIFEST.json`. None appears in the corresponding frozen `permitted_purpose` list. |

### Defect OAV-001 — Unauthorized role and purpose expansion

The version-2 allowlist is not a faithful implementation of the frozen audit decisions because it grants an additional role and additional purposes to three records. The fact that those records are `CLEAN_ALLOWLISTED` does not independently authorize broader consumers or uses; role and purpose restrictions are part of the import decision. This is an authorization-fidelity defect. It does not establish protected access or scientific contamination, and this validation did not test or exercise any added permission.

Required correction: either remove the added `Evidence Sanitizer` role and the three added purposes so the official allowlist exactly implements the frozen decisions, or obtain a separately authorized amendment to the frozen decisions before encoding those expansions. Any corrected allowlist must be re-frozen, re-hashed, and independently revalidated before the expanded role or purposes are used.

### Official verdict

`IMPORT_ALLOWLIST=FAIL`

Version 2 is hash-authentic and correctly carries paths, hashes, statuses, evidence/provenance classes, contamination classifications, and prohibitions, but it fails the required all-fields fidelity gate because its role and purpose permissions exceed the frozen decisions. The mismatch is fail-closed: no `Evidence Sanitizer` access or added-purpose use is authorized by this audit.
