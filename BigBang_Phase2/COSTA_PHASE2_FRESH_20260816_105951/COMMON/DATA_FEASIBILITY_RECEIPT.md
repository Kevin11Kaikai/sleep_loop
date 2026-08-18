# Data Feasibility Receipt

Campaign: `COSTA_PHASE2_FRESH_20260816_105951`  
Auditor: `/root/data_feasibility_audit`  
Audit time: `2026-08-16T12:05:10.1030343-04:00`  
Final verdict: **Phase 2B minimum multi-subject feasibility is not met under current permitted access.**

## Exact result

- Eligible subjects explicitly certified as public/development Night-1: **0**.
- Eligible personalized model artifacts explicitly certified as public/development Night-1: **0**.
- Independently certifiable exact same-candidate R4 execution artifacts: **0**. No exact R4 identifier/path/hash binding is present in the inspected controls or metadata, and no R4 tree was searched or opened.
- Raw PSG/EDF/hypnogram access: **DENIED and UNOPENED**. The manifest's path cells were not released or followed.
- Phase 2B denominator feasibility without prohibited access: **NO**. The repository has broad index metadata, but it does not provide a clean, certified, executable multi-subject personalized-model denominator.
- Parameter ranking or winner selection: **none performed**.

## Audit boundary

The auditor read and hash-bound only these five current controls:

| Control | SHA-256 |
|---|---|
| `GOVERNANCE/PROTECTED_DENYLIST.json` | `AF3AA8AC66775885B09F47C425DE1BE3F515B877FD7DDDC44DCA32F1824EA060` |
| `GOVERNANCE/NONHISTORICAL_METADATA_RELEASE.json` | `10C372462C46BADDC6D4D31B52F45776E7D5BF33CE5F82E18635C1F44B376A5D` |
| `IMPORT_ALLOWLIST/NONHISTORICAL_METADATA_CANDIDATES.json` | `1EC2BE64605CFAB9DCD5D0B898BE0DD9F1E1975A03E9D3B1002C06328B4A5AEE` |
| `IMPORT_ALLOWLIST/REPOSITORY_CODE_AUDIT_DECISIONS.json` | `9E993DFFC656B9A6C1619E0B6BAA946C2F7A1576E263F2C7D1BFEFF1B4D0D9E7` |
| `COMMON/VALUE_FREE_REPOSITORY_CAPABILITY_INVENTORY.json` | `E89DDECF6ED40D566EFFBDCD9CDFCE3E9AC8AA9B12600F2C46E4F8CBF93B5D2E` |

It then opened and hashed exactly the 30 files named in `DATA_AND_PARAMETER_METADATA_ONLY` plus the exact clean-code dependency `D:\Year3_Mao_Projects\sleep_loop\data\manifest.csv`. No directory was enumerated. No manifest reference was followed. No raw signal, hypnogram, document, output, historical campaign, Night-2, Sealed Bank, protected-derived, fresh-final, confirmatory, ambiguous, or quarantined external payload was opened.

## Exact-file classification

The 31 exact candidates classify as follows:

| Class | Count | Meaning in this audit |
|---|---:|---|
| `CLEAN_ALLOWLISTED` | 1 | `manifest.csv`, narrowly for sanitized subject-ID/schema/count metadata only; all raw locators remain denied. |
| `QUARANTINED` | 26 | Subject-derived epoch/spectral data, fitted parameter artifacts, bifurcation/sweep outcomes, and targets. |
| `UNRESOLVED` | 4 | `cmat.npy`, `dmat.npy`, and the two byte-identical target-frequency arrays; header-level shape is known but exact provenance is absent. |
| `DENIED` | 0 | No listed exact candidate required this class; every manifest-referenced raw payload is separately `DENIED/UNOPENED`. |

Every exact path, SHA-256, provenance statement, schema/shape, contamination finding, permitted role/purpose, and prohibition is recorded in `IMPORT_ALLOWLIST/DATA_AUDIT_DECISIONS.json`.

## Subject and model availability

- `manifest.csv` has columns `subject_id`, `psg_path`, and `hypnogram_path`, with **153 rows and 153 unique subject IDs**. It has no night, split, license, public/development, or certification column. These IDs establish index presence only.
- The five stage epoch tables cover **10 unique subjects**: `SC4001`, `SC4002`, `SC4011`, `SC4012`, `SC4021`, `SC4022`, `SC4031`, `SC4032`, `SC4041`, and `SC4042`.
- Their row denominators are N1 **296**, N2 **429**, N3 **280**, REM **148**, and Wake **75**. These counts are released only as feasibility metadata.
- The parallel `epochs_[n1,n2,n3,rem,wake]` and `psd_[n1,n2,n3,rem,wake]` families, plus the absence of a night column, establish N1/N2/N3 as **sleep-stage labels** here. They are not Night-1/Night-2 recording certifications.
- There are **12 exact personalized parameter JSON files and 11 distinct SHA-256 payloads**. All resolve only to the `SC4001` token, either by embedded `subject_id` or filename token. None is clean for execution.
- No parameter artifact embeds an exact generating-code path and hash. Textual consumer references and filename version tokens do not certify code-version provenance.
- No artifact or subject is explicitly certified public/development Night-1. Numeric suffixes, N1 stage labels, and raw locator strings were not used as proxies.

## Duplicate and linkage findings

- `patient_params.json` and `patient_params_SC4001.json` are byte-identical.
- `target_freqs.npy` and `target_freqs_SC4001.npy` are byte-identical.
- `psd_n3.npy` and `target_psd.npy` are byte-identical, confirming that the generic target aliases an N3-stage spectral artifact but not establishing clean provenance.
- The only clean-code dependency relationship established for `manifest.csv` is a consumer reference from `tests/conftest.py` at SHA-256 `D55E748CE6C8E0AC8C8DFA2EA7640A48573ECFA02B8C82F2FCD62BC6DB8BAF69`. That is not data-source, Night-1, public/development, or generator certification.

## Phase 2B feasibility judgment

The current release supplies a 153-subject index and a 10-subject stage-denominator subset, but only one subject token has personalized parameter artifacts, all such artifacts are quarantined, and zero subjects/models are explicitly certified public/development Night-1. Creating additional personalized models or validating subject-level denominators would require denied raw PSG/hypnogram access or a new exact clean external release. Therefore the Phase 2B multi-subject minimum cannot be met without prohibited access.

This is a feasibility and governance finding only. It is not a scientific result and does not select a model, parameter version, or winner.

## Bound artifacts

| Artifact | SHA-256 |
|---|---|
| `IMPORT_ALLOWLIST/DATA_AUDIT_DECISIONS.json` | `19E052B2BBEFA9246D90C1D1CE0D0469C0BFEF05B2409D5C9A7CA013FAC9E688` |
| `COMMON/SUBJECT_MODEL_AVAILABILITY.json` | `08FF47A8EF93A08C032091F1314AA31BD959ADB320D8206E2E0BDADC4001BD88` |

Receipt status: `COMPLETE_NO_SCIENTIFIC_EXECUTION_RELEASE`.
