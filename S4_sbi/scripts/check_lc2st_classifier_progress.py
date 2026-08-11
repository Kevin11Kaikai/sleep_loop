"""Show progress of the L-C2ST posterior-sampling + classifier phase.

Calibration banks (20k/track) are treated as a prerequisite.  The slow phase
writes one directory per estimator:

  diagnostics/lc2st/{8d,7d}/{ensemble,member_1..5}/
    lc2st_result.json
    lc2st_classifier.pkl

Usage (from repo root, any env is fine):

  python S4_sbi/scripts/check_lc2st_classifier_progress.py
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "S4_sbi" / "results" / "figure10_8d_7d"
LC2ST = RESULTS / "diagnostics" / "lc2st"
LOGS = RESULTS / "logs"
ESTIMATORS = ("ensemble", "member_1", "member_2", "member_3", "member_4", "member_5")
TRACKS = ("8d", "7d")


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"_error": f"unreadable json: {path}"}


def _process_alive(pid: int | None) -> str:
    if pid is None:
        return "unknown"
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED
        if handle:
            kernel32.CloseHandle(handle)
            return "alive"
        return "dead"
    try:
        Path(f"/proc/{pid}").stat()
        return "alive"
    except (FileNotFoundError, OSError):
        return "dead"


def _estimator_status(track: str, name: str) -> dict[str, object]:
    root = LC2ST / track / name
    result = root / "lc2st_result.json"
    clf = root / "lc2st_classifier.pkl"
    scores = root / "lc2st_scores.npz"
    payload = _read_json(result) if result.exists() else None
    done = result.exists() and clf.exists()
    return {
        "track": track,
        "estimator": name,
        "done": done,
        "has_result": result.exists(),
        "has_classifier": clf.exists(),
        "has_scores": scores.exists(),
        "dir_exists": root.exists(),
        "p_value": None if not payload else payload.get("p_value"),
        "reject": None if not payload else payload.get("reject_alpha_0p05"),
        "runtime_s": None if not payload else payload.get("runtime_s"),
        "created_utc": None if not payload else payload.get("created_utc"),
        "mtime": (
            datetime.fromtimestamp(clf.stat().st_mtime, tz=timezone.utc).isoformat()
            if clf.exists()
            else (
                datetime.fromtimestamp(result.stat().st_mtime, tz=timezone.utc).isoformat()
                if result.exists()
                else None
            )
        ),
    }


def _calibration_ok(track: str) -> bool:
    manifest = (
        LC2ST
        / "calibration"
        / track
        / f"lc2st_calibration_20000_{track}_manifest.json"
    )
    npz = LC2ST / "calibration" / track / f"lc2st_calibration_20000_{track}.npz"
    payload = _read_json(manifest)
    if not payload or not npz.exists():
        return False
    return int(payload.get("valid", 0)) == 20000 or int(
        payload.get("attempted", 0)
    ) == 20000


def _latest_log_lines(pattern: str, n: int = 8) -> list[str]:
    candidates = sorted(LOGS.glob("detached_post_global_*_stdout.log")) + sorted(
        LOGS.glob("global_*_stdout.log")
    )
    if not candidates:
        return []
    path = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    matched = [line for line in lines if pattern in line]
    return matched[-n:]


def _stderr_tail(n: int = 6) -> list[str]:
    candidates = sorted(LOGS.glob("detached_post_global_*_stderr.log"))
    if not candidates:
        return []
    path = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return [line for line in lines if line.strip()][-n:]


def main() -> None:
    ckpt = _read_json(LOGS / "lc2st_checkpoint.json") or {}
    pid = ckpt.get("pid")
    alive = _process_alive(int(pid)) if pid is not None else "n/a"
    print("=== L-C2ST classifier-phase progress ===")
    print(f"now_utc:     {datetime.now(timezone.utc).isoformat()}")
    print(f"checkpoint:  status={ckpt.get('status')} pid={pid} ({alive})")
    print(f"updated_utc: {ckpt.get('updated_utc')}")
    print()

    print("--- calibration banks ---")
    for track in TRACKS:
        ok = _calibration_ok(track)
        print(f"  {track}: {'OK 20000' if ok else 'MISSING/INCOMPLETE'}")
    dataset = _read_json(LC2ST / "lc2st_dataset_manifest.json")
    if dataset:
        print(f"  dataset_manifest: {dataset.get('created_utc')}")
    print()

    rows = []
    for track in TRACKS:
        for name in ESTIMATORS:
            rows.append(_estimator_status(track, name))

    done = sum(1 for row in rows if row["done"])
    print(f"--- estimators {done}/12 complete ---")
    print(f"{'track':<4} {'estimator':<10} {'state':<12} p_value  reject  runtime_s")
    current = None
    for row in rows:
        if row["done"]:
            state = "DONE"
        elif row["has_result"] or row["has_classifier"] or row["dir_exists"]:
            state = "PARTIAL"
            if current is None:
                current = row
        else:
            state = "pending"
            if current is None:
                current = row
        p_value = row["p_value"]
        p_txt = f"{p_value:.4g}" if isinstance(p_value, (int, float)) else "-"
        reject = row["reject"]
        r_txt = str(reject) if reject is not None else "-"
        runtime = row["runtime_s"]
        t_txt = f"{runtime:.1f}" if isinstance(runtime, (int, float)) else "-"
        mark = " <-- current?" if row is current and not row["done"] else ""
        print(
            f"{row['track']:<4} {row['estimator']:<10} {state:<12} "
            f"{p_txt:>7}  {r_txt:<6}  {t_txt:>8}{mark}"
        )

    print()
    summary_8 = (LC2ST / "8d" / "lc2st_summary.csv").exists()
    summary_7 = (LC2ST / "7d" / "lc2st_summary.csv").exists()
    print("--- track summaries ---")
    print(f"  8d/lc2st_summary.csv: {'OK' if summary_8 else 'MISS'}")
    print(f"  7d/lc2st_summary.csv: {'OK' if summary_7 else 'MISS'}")

    if current and not current["done"]:
        print()
        print("--- interpretation ---")
        print(
            f"  Likely working on: {current['track']}/{current['estimator']}"
        )
        print(
            "  Each slot = posterior sampling (can be very slow under leakage)"
            " then classifier train/eval."
        )
        print(f"  Remaining slots after this one: {12 - done - 1}")

    print()
    print("--- recent stdout (L-C2ST / calibration) ---")
    for line in _latest_log_lines("L-C2ST", 8) or _latest_log_lines(
        "lc2st_calibration", 6
    ):
        print(f"  {line}")

    print()
    print("--- stderr tail (sampling warnings live here) ---")
    for line in _stderr_tail(6):
        print(f"  {line}")

    if done == 12 and summary_8 and summary_7:
        print()
        print("ALL classifier slots complete. Waiting for stage checkpoint=complete.")
    elif ckpt.get("status") == "complete":
        print()
        print("Stage checkpoint already complete.")


if __name__ == "__main__":
    main()
