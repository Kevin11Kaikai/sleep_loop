# Fraud Risk Ranking for Limited Investigation Capacity

An IEEE-CIS portfolio project on improving the information available to a fraud
ranking model, from an original LightGBM benchmark to feature-enriched ExcelFormer.

## Project story

- After basic ExcelFormer lagged LightGBM, combined transaction, device and identity
  clues with patterns across related transactions to provide richer risk context.
- Adapted inputs to preserve missing/unknown states and encode categories from
  earlier training blocks; integrated ExcelFormer training with feature mixing,
  BCE loss and validation-based checkpoint selection.
- On the same 88,581 development-validation transactions, final AP reached
  **0.6366**, versus **0.5897** for the original LightGBM. At the same **886-review
  budget**, captured **830 vs. 789 labeled frauds (+41)**.

Matched ExcelFormer feature ablations improved AP from 0.5735 to 0.6019 before the
later BCE training extension. This separates evidence for feature adaptation
from the additional training gain.

## What is included

- `src/`, `configs/`, `tests/`: the original fixed LightGBM/FT-Transformer MVP.
- `experiments/`: feature construction, identity/UID grouping, typed input
  adaptation, ExcelFormer BCE training, paired continuation and verification code.
- `evidence/results_summary.json`: the checked comparison, hashes and scope.
- `evidence/final_training/`: final training metrics, stopping reason, checkpoint
  replay checks and learning curves.
- `UPSTREAM_SOURCES.json`: pinned external model/reference provenance.

This branch contains the fraud MVP snapshot. It does not modify `main`.
Raw datasets, derived row-level tensors, predictions, checkpoints, environments,
credentials, unrelated research and personal application documents are excluded.

## Reproduction and dependencies

Use the labeled IEEE-CIS training files obtained under the competition's terms:
[official data page](https://www.kaggle.com/competitions/ieee-fraud-detection/data).
CUDA PyTorch 2.5.1 is required for neural training. The original baseline entry:

```powershell
python -m pip install -r requirements.txt
python -m src.run --stage all --data data/raw/train_transaction.csv
```

The follow-up scripts are preserved historical source snapshots. They require
the original local feature caches, pinned ExcelFormer code and full optimizer/RNG
checkpoints; these are not downloadable models in this branch. Read
[SOURCE_MAP.md](SOURCE_MAP.md) before running an archived campaign. Frozen hashes
remain enforced: missing artifacts or changed sources must not be bypassed.
This publication did not rerun training or access evaluation labels.

## Evidence boundaries

The headline is a development-validation **pipeline** improvement over the original
MVP. It is not a claim of production savings or independent future generalization.
The matched enriched LightGBM reference (AP 0.6877) remains stronger. Paired
distillation (AP 0.6349) did not improve on BCE (AP 0.6366).
Early stopping was configured; the extension stopped at its epoch cap.
ExcelFormer is an existing architecture; contributions here are feature
organization, input adaptation, training integration and controlled evaluation.
