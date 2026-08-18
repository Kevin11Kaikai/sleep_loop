# Environment log

- Exact `conda.bat run --no-capture-output -n neurolib python` launcher verified.
- MNE 1.9.0 public signatures verified with `inspect.signature`; `read_annotations` has no `verbose`, while `read_raw_edf` accepts keyword-only `verbose` and `preload`.
- Default temporary cache creation caused pathological import latency in `numba.core.caching.ensure_cache_path`. Campaign-local `TEMP`, `TMP`, and `NUMBA_CACHE_DIR` resolved it without scientific changes.
- Neurolib WCModel smoke returned a finite `(1, 200)` output.
- Neurolib editable root commit is `9b6b2b8f…`; read-only Git used explicit `safe.directory`. Tracked diff is clean. Only `environment.yml` is untracked and its content was not read or used.
