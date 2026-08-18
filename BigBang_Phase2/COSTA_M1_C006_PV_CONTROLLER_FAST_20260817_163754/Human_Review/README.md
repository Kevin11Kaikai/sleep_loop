# COSTA C006 Human Review Packet

This directory is the single entry point for reviewing the Phase 2A → C006 controller thread. All scientific artifacts are read-only references to existing experiment directories. The packet does not rerun models or access raw/protected data.

## Recommended reading order

1. Open `COSTA_C006_THREAD_HUMAN_REVIEW.html` for the complete visual narrative.
2. Read `EXECUTIVE_BRIEF.md` for the current decision and the minimum evidence behind it.
3. Use `REVIEW_AGENDA.md` to conduct a bounded 45-minute review.
4. Use `C1_READINESS_CHECKLIST.md` to separate infrastructure success from C1 evidence.
5. Consult `KNOWN_LIMITATIONS.md` and `GLOSSARY.md` when interpreting plots.
6. Record the decision in `DECISION_RECORD_TEMPLATE.md`.

## Notebook files

- `COSTA_C006_THREAD_HUMAN_REVIEW_EXECUTED.ipynb`: canonical notebook with embedded outputs.
- `COSTA_C006_THREAD_HUMAN_REVIEW.ipynb`: source notebook for rerunning in the `neurolib` environment.
- `build_human_review_notebook.py`: reproducibly rebuilds the source notebook.

Rerun from this directory with:

```bat
cmd /c "call C:\Users\YUS190\AppData\Local\anaconda3\condabin\conda.bat activate neurolib && python -u build_human_review_notebook.py && python -m jupyter nbconvert --to notebook --execute COSTA_C006_THREAD_HUMAN_REVIEW.ipynb --output COSTA_C006_THREAD_HUMAN_REVIEW_EXECUTED.ipynb --ExecutePreprocessor.timeout=300 && python -m jupyter nbconvert --to html COSTA_C006_THREAD_HUMAN_REVIEW_EXECUTED.ipynb --output COSTA_C006_THREAD_HUMAN_REVIEW.html"
```

## Evidence boundary

`EXPLORATORY_ONLY`. This packet does not establish formal C1/C2, clinical efficacy, improved real-patient sleep, or biological parameter truth.
