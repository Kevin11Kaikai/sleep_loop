"""Unattended, resumable orchestration of the preregistered Route-3 rescue."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "S4_sbi" / "src"))
RESULTS = ROOT / "S4_sbi" / "results" / "route3_7d_rescue"
LOG = RESULTS / "logs" / "orchestrator.log"
STATUS = RESULTS / "logs" / "orchestrator_status.json"


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{timestamp()}] {message}"
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def status(stage: str, state: str, detail=None) -> None:
    payload = {
        "updated_utc": timestamp(),
        "stage": stage,
        "state": state,
        "detail": detail,
    }
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(STATUS)


def run_stage(name: str, callable_, *args, **kwargs):
    status(name, "running")
    log(f"START {name}")
    started = time.perf_counter()
    result = callable_(*args, **kwargs)
    elapsed = time.perf_counter() - started
    detail = (
        result
        if isinstance(result, (str, int, float, bool, type(None)))
        else str(type(result).__name__)
    )
    status(name, "complete", {"elapsed_s": elapsed, "result": detail})
    log(f"COMPLETE {name} elapsed_s={elapsed:.2f}")
    return result


def main() -> None:
    from sleep_sbi.route3_7d_rescue import (
        BANK_ROOT,
        build_combined_group_safe_bank,
        verify_rescue_preregistration,
    )
    from sleep_sbi.route3_7d_rescue_reporting import (
        environment_report,
        plot_diagnosis,
        plot_final_validation,
        plot_training_and_development,
        validate_rescue_artifacts,
        write_final_reports,
    )
    from sleep_sbi.route3_7d_rescue_training import (
        evaluate_and_select_development,
        sample_development,
        train_configuration,
    )
    from sleep_sbi.route3_7d_rescue_validation import (
        analyze_fresh_final,
        evaluate_final_decision,
        run_final_ppc,
        run_fresh_final_1024,
        sample_fresh_final_posteriors,
    )

    log(f"orchestrator boot; prereg={verify_rescue_preregistration()}")
    manifest = BANK_ROOT / "additional_2048x2" / "manifest.json"
    wait_started = time.perf_counter()
    while not manifest.exists():
        checkpoints = len(
            list((BANK_ROOT / "additional_2048x2" / "checkpoints").glob("*.npz"))
        )
        status(
            "wait-additional-bank",
            "running",
            {"checkpoints": checkpoints, "scheduled": 4096},
        )
        log(f"WAIT additional bank checkpoints={checkpoints}/4096")
        if time.perf_counter() - wait_started > 4 * 3600:
            raise TimeoutError("additional rescue bank did not finish within four hours")
        time.sleep(60)

    run_stage("combined-bank", build_combined_group_safe_bank)
    run_stage("train-maf64-t5", train_configuration, "maf64_t5")
    run_stage("train-maf128-t8", train_configuration, "maf128_t8")
    run_stage("sample-development-maf64", sample_development, "maf64_t5")
    run_stage("sample-development-maf128", sample_development, "maf128_t8")
    selection = run_stage(
        "select-and-freeze-primary", evaluate_and_select_development
    )
    log(
        "PRIMARY "
        f"{selection['selected_config_id']}:{selection['selected_method']} "
        f"formal_eligible={selection['formal_go_eligible']}"
    )
    run_stage("fresh-final-1024", run_fresh_final_1024, 16)
    run_stage("sample-fresh-final", sample_fresh_final_posteriors)
    analysis = run_stage("analyze-fresh-final", analyze_fresh_final)
    ppc = run_stage("fresh-final-ppc", run_final_ppc, analysis["primary"], 16)
    decision = run_stage(
        "frozen-final-decision", evaluate_final_decision, analysis, ppc
    )
    log(f"TERMINAL VERDICT {decision['verdict']} blockers={decision['blockers']}")
    run_stage("environment-report", environment_report)
    run_stage("figure-diagnosis", plot_diagnosis)
    run_stage("figure-training", plot_training_and_development)
    run_stage("figure-final", plot_final_validation)
    run_stage("write-final-reports", write_final_reports)
    run_stage("artifact-reload-before-notebooks", validate_rescue_artifacts)

    run_stage(
        "build-notebooks",
        subprocess.check_call,
        [sys.executable, str(ROOT / "S4_sbi" / "scripts" / "build_route3_7d_rescue_notebooks.py")],
        cwd=ROOT,
    )
    notebook_paths = [
        ROOT / "S4_sbi" / "notebooks" / name
        for name in (
            "19_Route3_7D_Coverage_Failure_Diagnosis.ipynb",
            "20_Route3_7D_Rescue_Preregistration.ipynb",
            "21_Route3_7D_Rescue_Training.ipynb",
            "22_Route3_7D_Fresh_Heldout_Validation.ipynb",
            "23_Route3_7D_Rescue_Handoff.ipynb",
        )
    ]
    html_root = RESULTS / "html"
    html_root.mkdir(parents=True, exist_ok=True)
    for notebook in notebook_paths:
        run_stage(
            f"execute-{notebook.stem}",
            subprocess.check_call,
            [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                "--ExecutePreprocessor.timeout=1800",
                "--ExecutePreprocessor.kernel_name=neurolib",
                str(notebook),
            ],
            cwd=ROOT,
        )
        run_stage(
            f"html-{notebook.stem}",
            subprocess.check_call,
            [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "html",
                "--output-dir",
                str(html_root),
                str(notebook),
            ],
            cwd=ROOT,
        )
    run_stage("artifact-reload-final", validate_rescue_artifacts)
    status(
        "all-complete",
        "complete",
        {"verdict": decision["verdict"], "finished_utc": timestamp()},
    )
    log(f"ALL COMPLETE verdict={decision['verdict']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        status(
            "orchestrator",
            "failed",
            {"type": type(exc).__name__, "message": str(exc)},
        )
        log(f"FAILED {type(exc).__name__}: {exc}")
        with LOG.open("a", encoding="utf-8") as handle:
            traceback.print_exc(file=handle)
        raise
