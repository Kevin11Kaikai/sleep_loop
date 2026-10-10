"""Persistent sequential GPU campaign with resumable state and fixed budgets."""
from __future__ import annotations

import argparse
import gc
import os
import subprocess
import threading
import time
import traceback

import numpy as np
import pandas as pd
import torch

from .controls import fit_controls
from .data import load_data, prepare
from .finalize import finalize
from .search import pilot_configs, proposal
from .train import run_trial
from .utils import HERE, PROJECT, fingerprints, now, read, verify_baseline, write


def telemetry(stop):
    destination = HERE / "gpu_telemetry.csv"
    if not destination.exists():
        destination.write_text("utc,utilization_percent,device_used_mib,device_total_mib,power_watts\n", encoding="utf-8")
    while not stop.is_set():
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
            if result.returncode == 0:
                with destination.open("a", encoding="utf-8") as stream:
                    stream.write(now()+","+result.stdout.strip().replace(" ", "")+"\n")
        except (OSError, subprocess.TimeoutExpired):
            pass
        stop.wait(30)


def existing_neural():
    result = []
    for path in sorted((HERE / "runs").glob("ft_*/result.json")):
        record = read(path)
        if record["status"] == "COMPLETED":
            result.append(record)
    return result


def summarize_trials(records):
    rows = []
    for record in records:
        cfg = record["config"]
        rows.append({"trial_id": record["trial_id"], "features": cfg["features"],
                     "embedding": cfg["embedding"], "width": cfg["width"], "depth": cfg["depth"],
                     "loss": cfg["loss"], "seed": cfg["seed"], "best_epoch": record["best_epoch"],
                     "epochs_executed": record["epochs_executed"],
                     "validation_ap": record["validation_scores"]["average_precision"],
                     "validation_recall_1pct": record["validation_scores"]["recall_top_1pct"],
                     "active_neural_seconds": record["active_neural_seconds"],
                     "peak_gpu_allocated_mib": record["peak_gpu_allocated_mib"]})
    if rows:
        pd.DataFrame(rows).to_csv(HERE / "trials.csv", index=False)


def run():
    state_path = HERE / "campaign_state.json"
    protocol = read(HERE / "protocol.json")
    if state_path.exists():
        state = read(state_path)
        if state["status"].startswith("COMPLETED"):
            print("Already completed; preserve and reuse", flush=True)
            return
        if state["code_fingerprints"] != fingerprints():
            raise RuntimeError("Campaign code/protocol changed; cannot silently resume")
    else:
        state = {"status": "PREPARING", "started_utc": now(), "started_timestamp": time.time(),
                 "minimum_active_neural_seconds": protocol["minimum_active_neural_seconds"],
                 "maximum_wall_seconds": protocol["maximum_campaign_wall_seconds"],
                 "active_neural_seconds": 0., "wall_seconds": 0., "completed_neural_trials": 0,
                 "failed_trials": [], "code_fingerprints": fingerprints(),
                 "baseline_preservation": verify_baseline(), "original_test_viewed": True,
                 "test_access_during_search": False}
    state["pid"] = os.getpid()
    state["updated_utc"] = now()
    write(state_path, state)
    stop = threading.Event()
    monitor = threading.Thread(target=telemetry, args=(stop,), daemon=True)
    monitor.start()
    try:
        prepare()
        state["status"] = "TRAINING_TREE_CONTROLS"
        write(state_path, state)
        trees = fit_controls()
        best_tree = max(trees, key=lambda r: r["validation_scores"]["average_precision"])
        pilots = pilot_configs()
        records = existing_neural()
        loaded_name, arrays, meta = None, None, None
        train_stop = state["started_timestamp"] + protocol["maximum_campaign_wall_seconds"] - protocol["finalization_reserve_seconds"]
        maximum = protocol["search_maximum_candidates"]
        for index in range(maximum):
            trial_id = f"ft_{index:03d}"
            path = HERE / "runs" / trial_id
            # Save proposals before running: a crash never changes a trial's
            # configuration based on results produced later.
            if (path / "config.json").exists():
                cfg = read(path / "config.json")
            else:
                cfg = pilots[index] if index < len(pilots) else proposal(index, records)
                write(path / "config.json", cfg)
            if (path / "result.json").exists() or (path / "failure.json").exists():
                continue
            state["active_neural_seconds"] = sum(r["active_neural_seconds"] for r in records)
            best = max(records, key=lambda r: r["validation_scores"]["average_precision"]) if records else None
            enough = state["active_neural_seconds"] >= protocol["minimum_active_neural_seconds"]
            better = best and best["validation_scores"]["average_precision"] >= best_tree["validation_scores"]["average_precision"]+.02
            recall_ok = best and best["validation_scores"]["recall_top_1pct"] >= best_tree["validation_scores"]["recall_top_1pct"]
            if time.time() >= train_stop or (enough and better and recall_ok):
                state["stop_reason"] = "fixed 24h wall budget" if time.time() >= train_stop else "12h minimum and validation stretch target achieved"
                break
            state.update(status="TRAINING_NEURAL", current_trial=trial_id, current_config=cfg,
                         completed_neural_trials=len(records), best_neural_validation_ap=None if not best else best["validation_scores"]["average_precision"],
                         best_tree_validation_ap=best_tree["validation_scores"]["average_precision"],
                         wall_seconds=time.time()-state["started_timestamp"], updated_utc=now())
            write(state_path, state)
            if loaded_name != cfg["features"]:
                arrays = None
                gc.collect()
                arrays, meta = load_data(cfg["features"])
                loaded_name = cfg["features"]
            print("TRIAL START", trial_id, cfg, flush=True)
            trial_started = time.time()
            try:
                record = run_trial(trial_id, cfg, arrays, meta, stop_at_wall=train_stop)
                records.append(record)
                summarize_trials(records)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                failure = {"status": "FAILED", "trial_id": trial_id, "failure_utc": now(),
                           "error": repr(exc), "traceback": traceback.format_exc(), "config": cfg,
                           "attempt_wall_seconds": time.time()-trial_started, "no_cpu_fallback": True}
                write(path / "failure.json", failure)
                state["failed_trials"].append(failure)
                write(state_path, state)
                print("TRIAL FAILED", trial_id, repr(exc), flush=True)
                torch.cuda.empty_cache()
                gc.collect()
                # Repeated infrastructure failures cannot justify burning 12h
                # or spamming the same broken implementation.
                recent = state["failed_trials"][-3:]
                if len(recent) == 3 and len({r["error"] for r in recent}) == 1:
                    raise RuntimeError("Three repeated identical infrastructure failures; diagnose before resuming") from exc
        else:
            state["stop_reason"] = "128 declared trials exhausted"
        state.update(status="FINALIZING", completed_neural_trials=len(records),
                     active_neural_seconds=sum(r["active_neural_seconds"] for r in records),
                     wall_seconds=time.time()-state["started_timestamp"], updated_utc=now())
        write(state_path, state)
        output = finalize(records, trees, state)
        state.update(status="COMPLETED_EXPLORATORY", finished_utc=now(),
                     wall_seconds=time.time()-state["started_timestamp"], minimum_12h_achieved=output["minimum_12h_achieved"],
                     final_report=str(HERE / "final/experiment_summary.md"), baseline_preservation=verify_baseline())
        write(state_path, state)
        print("CAMPAIGN COMPLETE", state, flush=True)
    except BaseException as exc:
        state.update(status="INTERRUPTED_REQUIRES_DIAGNOSIS", updated_utc=now(), error=repr(exc),
                     traceback=traceback.format_exc(), wall_seconds=time.time()-state["started_timestamp"])
        write(state_path, state)
        raise
    finally:
        stop.set()
        monitor.join(timeout=1)


if __name__ == "__main__":
    run()
