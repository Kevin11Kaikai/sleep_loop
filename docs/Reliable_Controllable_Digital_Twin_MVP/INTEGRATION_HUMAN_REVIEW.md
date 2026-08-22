# Integration Human Review

## Review answers

- **Was authoritative M1 found at the provided path?** Yes. The provided R4 root and all listed M1 artifacts were present and readable.
- **Was authoritative C1 found at the provided path?** Yes. The provided hierarchical-target root, all six fresh result files, and all listed C1 closure artifacts were present and readable.
- **Is the M1 to C1 evidence chain internally consistent?** Yes. All six C1 results carry the same M1 protocol, model, and selection hashes. The C1 protocol/controller hashes also match their frozen source files, and all six result hashes match the terminal checkpoint.
- **Why is C006 “Reliable”?** It was selected by the frozen minimum-FIT-NRMSE rule from 16 candidates, frozen before held-out access, outperformed matched C000 on 110 held-out epochs by 46.5592% relative NRMSE, passed all M1 gates, and was independently reproduced without material disagreement.
- **Why is C006 “Controllable”?** The same frozen C006 responded positively on the prospectively registered primary T2 metric in 3/3 fresh seeds at both 220 s and 910 s under frozen causal F01; I_SO was supportive 3/3 at both durations, and all integrity/safety gates passed.
- **What is the highest supported combined claim?** Bounded in-silico reliable and controllable mechanistic digital twin MVP.
- **What are the remaining limitations?** One-subject/proxy-level reliability scope; no clinical or population claim; adverse I_SP; mixed/slightly negative T6; unresolved spindle/coupling trade-offs; no global or all-endpoint controllability claim.
- **Was the GitHub archive created?** Yes. This directory is the minimal archive; large per-trigger payloads remain local and are hash-referenced.
- **Was the PPT created?** Yes. The English eight-slide advisor deck and speaker notes are included under `slides/`.
- **Was the PDF created?** Yes. The eight-page PDF was reopened, page-count checked, and rendered with Poppler after export.
- **Was it pushed?** Yes. The dedicated branch `reliable-controllable-digital-twin-mvp` was pushed to the existing `origin` after final QA.
- **Any unresolved integration conflict?** No. No `INTEGRATION_CONFLICT` was identified.

## Integration verdict

`INTEGRATION_PASS`: the M1 reliable-twin chain and C1 controllable-twin chain are scientifically compatible and provenance-linked. The correct interpretation is **primary controllability demonstrated with secondary dynamical trade-offs** within a bounded in-silico MVP claim.
