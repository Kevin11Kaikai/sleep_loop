# Independent OAV-001 Governance Adjudication

- Campaign: `COSTA_PHASE2_FRESH_20260816_105951`
- Adjudication timestamp: `2026-08-16T11:20:14.3891482-04:00`
- Auditor role: `Clean Governance Auditor`
- Scope: OAV-001 materiality and official import-allowlist version-3 fidelity

## Review boundary

This adjudication used only the eight exact current-campaign records named in the commission. No R4 file, manifest-referenced file, repository scientific content, directory listing, existence probe outside the named current-campaign records, or protected content was opened. No shared `STATE` file was modified.

## Executive verdicts

| Question | Verdict |
|---|---|
| Version-3 artifact SHA-256 | **PASS — expected hash matches observed hash** |
| Version-3 entry fidelity to original decisions plus amendment | **PASS** |
| Version-3 receipt/integrity fidelity | **FAIL — current historical-audit receipt mismatch** |
| `IMPORT_ALLOWLIST` | **FAIL** |
| OAV-001 premature-access materiality | **GOVERNANCE-MATERIAL; OUTCOME-BLIND; NO DETECTED SCIENTIFIC-EVIDENCE OR PROTECTED-ACCESS IMPACT** |
| `PROTECTED_ACCESS` | **NONE** |
| `ROLE_ISOLATION` | **PASS** |
| May the Evidence Sanitizer resume now? | **NO** |
| Does scientific repository inspection remain closed pending sanitizer completion? | **YES — CLOSED** |

## Version-3 integrity verification

- Expected `V4D1_IMPORT_ALLOWLIST.json` SHA-256: `0C3A627C7CBA306C40A1846317FD72CA043EB62F6E6E9157E6399DC3B1D5321F`
- Observed SHA-256: `0C3A627C7CBA306C40A1846317FD72CA043EB62F6E6E9157E6399DC3B1D5321F`
- Version field: `3`
- Frozen status: `FROZEN_AUDITED_AMENDED_PENDING_INDEPENDENT_REVALIDATION`
- Artifact hash verdict: **MATCH**

The matching outer hash proves that the reviewed file is the expected version-3 artifact. It does not cure an incorrect receipt encoded inside that artifact.

## Fidelity to the original decisions and sanitizer amendment

### Entry-level fidelity

**PASS.** Version 3 preserves the original four exact paths, source SHA-256 values, evidence classes, development-only provenance, contamination classifications, review statuses, and prohibited-use semantics. It preserves `FINAL_REPORT.md` as `QUARANTINED` and does not grant the Evidence Sanitizer access to it.

For only the three originally `CLEAN_ALLOWLISTED` records, version 3 adds `Evidence Sanitizer` exactly as authorized by `SANITIZER_PERMISSION_AMENDMENT_V1`. The two categorical records use the exact added purpose `derive value-free categorical R4 substrate only`. The manifest uses the amendment's exact purpose for extracting only explicitly manifest-listed code/config/parameter/detector/environment/input paths and expected hashes as `PENDING_INDEPENDENT_AUDIT`, without opening or otherwise accessing referenced files.

The version-3 global sanitizer constraints bind the amendment by path and hash, require a fresh isolated role, forbid numerical scientific values, forbid access to manifest-referenced files, retain default deny and `PENDING_INDEPENDENT_AUDIT`, forbid cleanliness inheritance, forbid directory enumeration/search/existence probing, and forbid role mixing. The official allowlist remains default-deny and allows only `CLEAN_ALLOWLISTED` entries.

### Receipt verification

| Receipt | Encoded SHA-256 | Observed current-file SHA-256 | Verdict |
|---|---|---|---|
| `IMPORT_ALLOWLIST/AUDIT_DECISIONS.json` | `DA50B67A50118A6BBB6BD6F65D141EB75A5B1A7931DAF5FF1D4B13458C94C3E7` | `DA50B67A50118A6BBB6BD6F65D141EB75A5B1A7931DAF5FF1D4B13458C94C3E7` | **MATCH** |
| `IMPORT_ALLOWLIST/SANITIZER_PERMISSION_AMENDMENT.json` | `0EFAFC537841F22D3023EC93B5965DF97033A7AB88F1C6E4D494FB42A655BCCA` | `0EFAFC537841F22D3023EC93B5965DF97033A7AB88F1C6E4D494FB42A655BCCA` | **MATCH** |
| `GOVERNANCE/HISTORICAL_IMPORT_AUDIT.md` | `53B8D020052CE68DB0CED72CFC1BB0AB456D4693BB0F6260018D5902FD7C5DBA` | `94644C28CFC7AA27B1D7A359B729AED95C009BAB55CDA018E549488987F6B7CB` | **MISMATCH** |

The access ledger shows that `53B8...` was the hash accepted earlier by the Root Orchestrator. The current historical-audit file now includes its later official-allowlist validation section and hashes to `9464...`. Version 3 provides no immutable historical path, version identifier, or supersession qualifier that would make its stale path-bound receipt resolve to the earlier bytes. Consequently, the current receipt cannot be verified and the all-fields integrity gate fails.

### Official allowlist verdict

`IMPORT_ALLOWLIST=FAIL`

The failure is narrow: entry-level implementation and the new amendment binding are faithful, but the frozen version-3 artifact contains one stale audit receipt. Sanitizer authorization therefore remains fail-closed. The matching expected version-3 hash does not convert that stale receipt into a valid current-file binding.

## OAV-001 materiality adjudication

### Historical authorization status

The premature sanitizer read cannot be retroactively classified as authorized. At the time of access, version 2 named the Evidence Sanitizer, but independent validation had not accepted the role-and-purpose expansion, and the governing validation explicitly held that expansion fail-closed. Reading and hashing the three records before a valid amendment-bound allowlist completed revalidation was therefore an authorization-sequencing breach.

### Materiality classification

**OAV-001 is material to governance and access-control fidelity.** It exercised permissions before their independent validation gate and therefore requires a durable incident record, accurate access/state ledgers, a corrected frozen allowlist, and independent revalidation.

**No scientific-evidence or protected-access materiality is detected from the inspected records.** The attested access was limited to the official allowlist and three exact, hash-bound records already classified `CLEAN_ALLOWLISTED`; the quarantined narrative was not opened; no referenced artifact was opened; no numeric scientific outcome, raw data, Phase-2 outcome, protected class, or sanitizer output was exposed or produced. The change and access occurred while scientific outcomes were not visible. Thus the deviation is outcome-blind and contained, but not procedurally trivial.

This is a dual-axis ruling: `GOVERNANCE_MATERIAL / SCIENTIFIC_CONTAMINATION_NOT_DETECTED`.

## Protected-access adjudication

`PROTECTED_ACCESS=NONE`

The incident record, repair ledger, historical decisions, and access-ledger counters agree that no protected or raw-data class was accessed. The three prematurely opened records were classified non-protected within the bounded historical audit. This finding does not make the premature access authorized; it only classifies its content and contamination impact.

## Role-isolation adjudication

`ROLE_ISOLATION=PASS`

The Evidence Sanitizer is recorded as a fresh isolated role with explicit conflicts against scientific worker, verifier, adversarial reviewer, and final claim judge roles. The incident attests to only the sanitizer's bounded access and no output, scientific work, or role mixing. No contrary role-mixing evidence appears in the inspected records.

The role ledger nevertheless reports the sanitizer as `ACTIVE`, while the incident says it was interrupted and the amendment says it is not active pending official revision and revalidation. That stale activation status is a state-fidelity defect, not evidence of role mixing, and must be reconciled by the sole shared-state writer before any resumption.

## Ledger-fidelity findings

Two state-record gaps require Root Orchestrator correction:

1. `STATE/ACCESS_LEDGER.json` contains no entries for the Evidence Sanitizer's attested reads/hashes and no OAV-001 containment/adjudication receipt. Its zero protected/raw counters remain consistent with the classified content, but the access history is incomplete.
2. `STATE/AGENT_ROLE_LEDGER.json` says the sanitizer is `ACTIVE` rather than interrupted or paused pending revalidation.

These gaps do not establish protected access or role mixing. They do prevent a clean resumption record and reinforce the fail-closed decision.

## Resumption and scientific-access gates

`EVIDENCE_SANITIZER_RESUME=NO`

Before sanitizer resumption, the Root Orchestrator must:

1. keep or record the sanitizer as interrupted/paused;
2. issue a new frozen official allowlist version that preserves the valid version-3 entry semantics and amendment binding but replaces the stale historical-audit receipt with the current exact hash (or binds an explicitly immutable, versioned predecessor artifact);
3. obtain independent hash and all-fields revalidation of that new version;
4. append the exact premature accesses and OAV-001 disposition to the access ledger; and
5. reconcile the sanitizer's role-ledger status before explicitly releasing it.

`SCIENTIFIC_REPOSITORY_INSPECTION=CLOSED`

Scientific repository inspection remains closed throughout the repair and sanitization stage. Even after a corrected allowlist permits the sanitizer to resume, the sanitizer may only perform the amendment's mechanical value-free extraction and pending-candidate registration. Manifest-referenced files remain default-denied and may not be opened, hashed, statted, searched for, or otherwise inspected. Scientific access requires sanitizer completion, independent audit of any exact pending candidates, exact `CLEAN_ALLOWLISTED` decisions, and a later explicit Root Orchestrator scientific gate.

## Auditor attestation

I did not open any R4 or scientific file, did not perform scientific interpretation, and did not modify shared state. This adjudication is based solely on the eight named current-campaign records.

---

## FINAL REVALIDATION — OFFICIAL ALLOWLIST VERSION 4

- Revalidation timestamp: `2026-08-16T11:23:21.5081166-04:00`
- Revalidation role: `Clean Governance Auditor`
- Boundary: official allowlist v4, its four current-campaign governance receipts, and the reconciled access and role ledgers only

No R4 file, manifest-referenced file, repository scientific content, directory listing, or protected content was opened during this revalidation. Neither the official allowlist nor shared `STATE` was modified.

### Hash and receipt verification

| Artifact or receipt | Expected/encoded SHA-256 | Observed SHA-256 | Verdict |
|---|---|---|---|
| Official allowlist v4 | `9995DE5822BEECE79FA8ECA2C2FF18B11E3DA6EAF7B0915B3E349FE310CE4926` | `9995DE5822BEECE79FA8ECA2C2FF18B11E3DA6EAF7B0915B3E349FE310CE4926` | **MATCH** |
| Current historical-import audit receipt | `94644C28CFC7AA27B1D7A359B729AED95C009BAB55CDA018E549488987F6B7CB` | `94644C28CFC7AA27B1D7A359B729AED95C009BAB55CDA018E549488987F6B7CB` | **MATCH** |
| OAV-001 adjudication receipt | `951C9456DBE90D4BD2A6EE1479062E8978990DF0A19BE49C8B1FC72AF49452D0` | `951C9456DBE90D4BD2A6EE1479062E8978990DF0A19BE49C8B1FC72AF49452D0` | **MATCH** |
| Sanitizer permission amendment | `0EFAFC537841F22D3023EC93B5965DF97033A7AB88F1C6E4D494FB42A655BCCA` | `0EFAFC537841F22D3023EC93B5965DF97033A7AB88F1C6E4D494FB42A655BCCA` | **MATCH** |
| Frozen audit decisions | `DA50B67A50118A6BBB6BD6F65D141EB75A5B1A7931DAF5FF1D4B13458C94C3E7` | `DA50B67A50118A6BBB6BD6F65D141EB75A5B1A7931DAF5FF1D4B13458C94C3E7` | **MATCH** |
| Reconciled access ledger | expected prefix `D33CE900...` | `D33CE9005691023F062B93C6FA71FD8EEAE48BB54679A54CB0F44754AAEFFC45` | **MATCH** |
| Reconciled agent-role ledger | expected prefix `A1D15051...` | `A1D150517B2DC2DB14D5EE8F7D37ABF97938F7C8D652B7AA7D0445EDC5475FDD` | **MATCH** |

The OAV-001 adjudication hash above was observed immediately before this final section was appended and is the exact adjudication version accepted by the Root Orchestrator and bound in allowlist v4. This final section is the subsequent independent revalidation receipt; it does not alter the fact that the v4 binding matched the adjudication bytes at the revalidation boundary.

### Allowlist-v4 fidelity

**PASS.** Version 4 preserves the valid entry-level semantics previously verified in version 3: four exact path/hash bindings, three `CLEAN_ALLOWLISTED` records, one quarantined narrative, default deny, no sanitizer permission on the quarantined narrative, and only the exact role and purposes approved by the amendment on the three clean records. It preserves the sanitizer's fail-closed constraints, no numerical scientific values, no access to referenced files, no cleanliness inheritance, no enumeration/search/existence probing, and no role mixing.

Version 4 corrects the version-3 receipt defect by binding the current historical-audit hash. It also binds the exact OAV-001 adjudication, carries its governance-material/outcome-blind/no-detected-scientific-contamination classification, and matches the decisions and amendment receipts. No unresolved field- or receipt-fidelity defect remains within the reviewed boundary.

### Reconciled-state verification

**PASS.** The access ledger now records contiguous entries for the sanitizer's official-allowlist read, the three exact historical clean-record reads/hashes, OAV-001 containment, and acceptance of the independent adjudication. It records three historical clean-record accesses while protected and raw-data counters remain zero. The entries identify the v2 permission-fidelity deviation without misclassifying those historical records as protected or raw.

**PASS.** The role ledger now records the Evidence Sanitizer as `PAUSED_PENDING_GOVERNANCE_REVALIDATION`, binds its access attestation to the three exact records, records no quarantined/referenced-file access, no directory enumeration, and no output, and preserves fresh-context conflicts with scientific and adjudicative roles.

### Final verdicts

| Control | Final verdict |
|---|---|
| `IMPORT_ALLOWLIST` | **PASS — VERSION 4 INDEPENDENTLY REVALIDATED** |
| `OAV_001_REMEDIATION` | **COMPLETE — CONTAINED, RECONCILED, AND INDEPENDENTLY REVALIDATED** |
| `PROTECTED_ACCESS` | **NONE** |
| `ROLE_ISOLATION` | **PASS** |
| `EVIDENCE_SANITIZER_RESUME` | **YES — EXACT AMENDMENT SCOPE ONLY, AFTER ROOT ORCHESTRATOR RELEASE** |
| `SCIENTIFIC_REPOSITORY_INSPECTION` | **CLOSED** |

OAV-001 remains a permanent governance-material incident; remediation does not retroactively authorize the premature accesses or erase the incident. It does establish a valid prospective permission basis for the paused Evidence Sanitizer.

### Resumption boundary

The Root Orchestrator may release the fresh isolated Evidence Sanitizer to resume only the exact amendment scope: read/hash/extract from the three exact hash-bound `CLEAN_ALLOWLISTED` records, produce the value-free categorical R4 substrate, and register only unambiguous manifest-listed code/config/parameter/detector/environment/input paths and expected hashes as `PENDING_INDEPENDENT_AUDIT`. It may not open the quarantined narrative, any manifest-referenced file, or any other R4 path; enumerate/search/probe the repository; transfer numerical scientific values; perform scientific interpretation; assign cleanliness; mix roles; or mutate shared state or the official allowlist.

### Scientific-access gate

Scientific repository inspection remains **CLOSED** while sanitization and downstream governance audits are incomplete. Sanitizer completion does not itself open scientific access. Every proposed repository file must first receive an independent exact-path/hash audit and an explicit `CLEAN_ALLOWLISTED` decision, followed by a separate Root Orchestrator scientific-stage release. Protected classes remain prohibited.

### Final revalidation attestation

I performed no R4 or scientific access and modified only this adjudication report by appending the commissioned final revalidation section.
