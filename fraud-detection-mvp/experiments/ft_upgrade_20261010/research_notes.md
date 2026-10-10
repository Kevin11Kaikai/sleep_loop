# Research rationale and evidence boundaries

This is a new, isolated follow-up to the completed MVP. The original test results
motivated the task; that test cannot become untouched again. Search uses only the
original training and validation partitions. Test evaluation is postponed until
the campaign freezes a selection, and is explicitly exploratory. No test-based
stopping, ensemble weights, feature selection or threshold tuning is permitted.

The user's authorized campaign is at least 12 hours of active neural training
and validation, at most 24 hours elapsed. Waiting, installation, literature work,
CPU tree training and a stopped process do not count toward the 12 hours. A
recorded duration means synchronized training/validation wall time, not a claim
of 100% GPU utilization. The job never pads the budget with idle sleeps. An upper
budget stop or a machine failure is reported even if the minimum is not reached.

The original neural run used 10 epochs, 64-dimensional linear numeric tokens,
and retained at most 2,048 categories per field. Its validation AP rose from
0.3955 at epoch 1 to 0.4873 at epoch 9. This suggests investigating training
budget and representation; it does not establish a root cause or predict a win.

## Primary sources

- [Revisiting Deep Learning Models for Tabular Data, NeurIPS 2021 / official FT](https://github.com/yandex-research/rtdl-revisiting-models/tree/main/package): reuse its backbone and optimizer parameter grouping, rather than invent a new Transformer.
- [On Embeddings for Numerical Features in Tabular Deep Learning, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html) and [official package](https://github.com/yandex-research/rtdl-num-embeddings/tree/main/package): motivates piecewise-linear and periodic numeric tokens. Use training-only quantile bins, no target-aware bins.
- [TabReD, ICLR 2025 / official repository](https://github.com/yandex-research/tabred): motivates feature-rich temporal evaluation and preserving temporal splits.
- [Understanding the Limits of Deep Tabular Methods with Temporal Shift, ICML 2025](https://proceedings.mlr.press/v267/cai25j.html): motivates investigating time structure and training lag. We retain the authorized chronological split; we do not use its proposed random split to inflate this benchmark.
- [TabM, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html): motivates diversity and careful efficiency comparisons. TabM is an MLP ensemble, and is not relabeled FT-Transformer here.
- [Stochastic Optimization of Areas Under Precision-Recall Curves, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/0dd1bc593a91620daecf7723d2235624-Abstract.html): motivates considering ranking objectives. The implemented auxiliary is a simple pairwise logistic loss, not SOAP or a provably convergent AP optimizer.
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002): motivates one hard-example loss ablation; effectiveness for this dataset must be measured.

Research findings on other datasets are hypotheses for this campaign, not
evidence of IEEE-CIS performance. No publication promises that a Transformer must
beat a strong tree baseline on these transactions.

## Fair comparison

Original results and their fingerprints remain unchanged. On expanded fields,
fit matched LightGBM controls with the same training rows, vocabulary and fields,
plus a modest stronger tree configuration. Choose both tree and neural settings
by validation AP. Report the original baseline as context and the strongest
matched baseline as the principal comparator. Report seed variation and all
trials, including failures. No monetary-loss or production-readiness claim.

Feature frequencies are fitted on training covariates, then frozen. They never
use future/test transactions. Numeric V redundancy pruning uses training-only
Pearson correlations on median-imputed values. Relative daily/weekly cyclic
features use TransactionDT without an assumed calendar origin. Anonymized V/C/D
fields still have unverified decision-time provenance. No target encoding,
identity merge, TransactionID predictor or official Kaggle test labels are used.

PLE bin fitting samples up to 100,000 training rows. A column constant in that
sample uses its full training min/max; if it is also constant in the full input
training subset (possible in the smoke test), use [value-0.5, value+0.5]. This
repairs the official bin routine's constant-column rejection without fitting on
validation or test values. All resulting edges are saved with the checkpoint.
