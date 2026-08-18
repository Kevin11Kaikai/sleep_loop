# Independent Clean-Root Governance Audit

- Campaign: `COSTA_PHASE2_FRESH_20260816_105951`
- Audit timestamp: `2026-08-16T11:03:54.8347569-04:00`
- Auditor role: `Clean Governance Auditor`
- Audit type: bounded governance-record audit before scientific access

## Scope and method

This audit was limited to the exact campaign governance and state records named in the audit commission. No repository directory was enumerated, no scientific repository content was opened, no R4 candidate file was opened, and no protected, Night-2, Sealed Bank, protected-derived, fresh-final, or confirmatory payload was accessed. No shared `STATE` file was modified.

`STATE/FREEZE_LEDGER.json` was checked only for presence and was absent. Its absence is not treated as a defect because the audit commission made review conditional on that file being present. Freeze findings below therefore rely on the frozen campaign identity, Context of Use, denylist, and empty import allowlist.

## Explicit verdicts

| Control | Verdict | Basis |
|---|---|---|
| `CLEAN_ROOT` | **PASS — EVIDENCE-BOUNDED** | `CAMPAIGN_IDENTITY.json` identifies the exact fresh campaign root, records `fresh_root: true`, records no reopening of prior campaign working directories, and is frozen. `ACCESS_LEDGER.json` contains only two governing-document reads and creation of this campaign root; it attests that no repository scientific content was inspected before initialization and no prior campaign working directory was reopened. `STAGE_STATE.json` remains at `INITIAL_GOVERNANCE` with scientific outcomes not visible and both scientific phases unauthorized. This is a governance-record conclusion, not a filesystem-wide provenance claim. |
| `IMPORT_ALLOWLIST_INITIALIZATION` | **PASS** | `V4D1_IMPORT_ALLOWLIST.json` has `default_policy: DENY`, `entries: []`, and status `FROZEN_EMPTY_PENDING_INDEPENDENT_AUDIT`. The four paths in `IMPORT_CANDIDATES.json` remain `PENDING_INDEPENDENT_AUDIT`; their listing is explicitly not scientific authorization, and directory enumeration is forbidden. |
| `PROTECTED_ACCESS` | **PASS — NONE RECORDED OR AUTHORIZED** | The access ledger reports `protected_access_count: 0`, `raw_data_access_count: 0`, and false attestations for Night-2/Sealed Bank and protected-derived access. The stage state says `protected_access: NONE`; the denylist says `protected_access_authorized: false`; and the frozen incident statement says this Phase 2 campaign has opened no protected, final-fresh, Night-2, or Sealed Bank scientific payload. No contrary entry appears in the inspected records. |
| `ROLE_ISOLATION` | **PASS** | `AGENT_ROLE_LEDGER.json` assigns shared-state writing solely to the Root Orchestrator, reserves a fresh isolated governance-auditor role, and expressly conflicts that role with scientific worker, verifier, adversarial reviewer, and final claim judge roles. The root is separately barred from independent verification, adversarial review, and final claim judging. This audit performed no scientific or shared-state work. |

## Additional control findings

### Denylist fail-closed semantics

**PASS WITH DEFECT D-001.** The denylist is case-insensitive, recursive, applies to files and directories, denies enumeration and all evidence-consuming operations, blocks the named protected content classes, applies recursive/default-deny boundaries to historical and sibling campaigns, prohibits recursive R1–R4 enumeration, and states that historical material requires exact-path `CLEAN_ALLOWLISTED` status. A path, filename, or differing hash is expressly insufficient to establish independence. These are fail-closed semantics.

The exact R4 candidates have a narrow governance-audit opening exception, but they remain unavailable for scientific use unless independently reviewed and entered into the import allowlist with `CLEAN_ALLOWLISTED` status.

### Campaign and scope freeze

**PASS.** Campaign identity, authorized and prohibited scope, claim ceiling (`M1 / C2 / P0 / INDEPENDENT_REPRODUCTION`), Context of Use, denylist, and empty import allowlist are all recorded as frozen at `2026-08-16T11:01:00-04:00`. The stage state does not authorize Phase 2A or Phase 2B.

### Access-ledger initialization

**PASS.** The ledger is initialized under default deny, has contiguous sequences 1–3, records only governing-document reads and fresh-root creation, reports zero protected and raw-data access, carries the relevant negative-access attestations, and names the Root Orchestrator as sole shared-state writer.

## Defects

### D-001 — Frozen denylist self-matches current governance-control filenames

Severity: **governance ambiguity; fail-closed/availability defect, not evidence of protected access**.

The patterns `protected` and `incident[ _.-]*001` match the current-campaign control files `GOVERNANCE/PROTECTED_DENYLIST.json` and `GOVERNANCE/INCIDENT_001_GOVERNANCE_STATEMENT.md`. Because `open` and `read` are denied operations and the frozen policy contains no explicit exception for these current-campaign governance records, a literal application of the denylist blocks the very records an auditor must read. This audit's exact-file commission supplied a bounded external authorization, but that authorization is not encoded in the denylist itself.

Required remediation before scientific-stage authorization: record a narrow, exact-path, read-only governance-auditor exception for current-campaign governance control records (without exempting any scientific payload or historical Incident-001 values), then re-freeze the revised policy and have the Root Orchestrator record the governance decision. The correction must not weaken the protected content-class prohibitions.

No other defect was identified in the inspected records. The absence of a freeze ledger is recorded above as an allowed not-present condition, not a defect.

## Scientific repository inspection decision

**MAY SCIENTIFIC REPOSITORY INSPECTION PROCEED NOW: NO.**

At the audit timestamp, the scientific phases remain unauthorized and the scientific import allowlist is empty. This clean-root pass does not itself authorize any repository scientific path, broad repository search, or directory enumeration. The next permissible activity is limited to governance-only handling under the frozen rules: after the Root Orchestrator accepts this audit and resolves D-001, an authorized governance/historical-import auditor may review only the four exact R4 candidates listed in `IMPORT_CANDIDATES.json`. Scientific workers may inspect only exact inputs subsequently given `CLEAN_ALLOWLISTED` status and explicitly released by the Root Orchestrator through the campaign gates. Protected classes remain prohibited without exception.

## Auditor attestation

The verdicts above are based solely on the named campaign governance and state records. I did not inspect scientific content, infer scientific results, perform scientific work, or mutate shared state.

---

## RE-AUDIT — D-001

- Re-audit timestamp: `2026-08-16T11:05:50.9094447-04:00`
- Re-audit scope: version-2 `GOVERNANCE/PROTECTED_DENYLIST.json` and current `STATE/FREEZE_LEDGER.json` only
- Expected denylist SHA-256: `AF3AA8AC66775885B09F47C425DE1BE3F515B877FD7DDDC44DCA32F1824EA060`
- Observed denylist SHA-256: `AF3AA8AC66775885B09F47C425DE1BE3F515B877FD7DDDC44DCA32F1824EA060`
- Hash verification: **MATCH**

No R4 candidate, scientific repository content, or other governance/state record was opened during this re-audit. No shared `STATE` file was modified.

### D-001 resolution verdict

**D-001: RESOLVED.**

Version 2 defines exact-path exceptions for the two self-matching current-campaign governance controls and places `exact_governance_control_exceptions` first in the explicit evaluation order, before historical boundaries, deny patterns, denied content classes, and default deny. The exceptions are restricted to named governance roles and to `read`, `hash`, and `audit`; both expressly set `scientific_use_permitted: false`. The Incident-001 exception is limited to the value-free governance statement and does not exempt Incident-001 exact values or value-dependent derivatives.

The freeze ledger independently records the same observed version-2 hash under `PROTECTED_DENYLIST_V2_OUTCOME_BLIND_GOVERNANCE_REPAIR`, cites `CLEAN_ROOT_AUDIT D-001`, and supersedes the original denylist hash. Its timestamp matches the denylist's version-2 freeze timestamp (`2026-08-16T11:05:06.1434416-04:00`). The repair therefore removes the self-denial ambiguity without weakening protected-content prohibitions.

### Final governance verdicts after re-audit

| Control | Final verdict |
|---|---|
| `CLEAN_ROOT` | **PASS — EVIDENCE-BOUNDED; D-001 RESOLVED** |
| `IMPORT_ALLOWLIST_INITIALIZATION` | **PASS — EMPTY, DEFAULT DENY** |
| `PROTECTED_ACCESS` | **PASS — NONE RECORDED OR AUTHORIZED** |
| `ROLE_ISOLATION` | **PASS** |

These four verdicts affirm the original bounded audit. The two-file D-001 re-audit found no evidence that changes their factual basis; the freeze ledger additionally records activation/freeze of the governance-auditor role separately from the Root Orchestrator.

### Exact-path R4 governance-audit decision

**MAY GOVERNANCE-ONLY EXACT-PATH R4 IMPORT AUDIT PROCEED: YES.**

An authorized governance/historical-import auditor may now open and hash only the four exact candidates already listed in `IMPORT_ALLOWLIST/IMPORT_CANDIDATES.json`, solely to determine import eligibility. Directory enumeration, opening any other R4 path, scientific use, and access to protected content classes remain prohibited. This permission does not authorize scientific workers or scientific repository inspection; any scientific use still requires exact-path `CLEAN_ALLOWLISTED` status and subsequent Root Orchestrator gate release.
