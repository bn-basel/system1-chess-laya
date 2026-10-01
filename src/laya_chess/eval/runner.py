"""Run an engine through the test sets and write one record per move.

Output of a run, in runs/<run_id>/ (kept out of git):
  moves.jsonl    one line per test position: what the engine played, how long it took, how it was graded
  manifest.json  how the run was made (versions, hardware, git commit) and its summary
A one-line summary per test set is also appended to results/summary.csv (committed).
"""

from __future__ import annotations

import csv
import json
import platform
import statistics
import subprocess
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import chess
from tqdm import tqdm

from ..data.testsets import RATING_BANDS, save_jsonl
from ..engines.base import Engine
from .metrics import is_correct, wp_loss

REPO_ROOT = Path(__file__).resolve().parents[3]
SUMMARY_FIELDS = ["run_id", "engine", "device", "set", "n", "correct", "wp_loss", "latency_p50_ms", "latency_p95_ms"]


def evaluate(engine: Engine, positions: list[dict]) -> list[dict]:
    records = []
    for pos in tqdm(positions, desc=engine.name, unit=" pos"):
        if hasattr(engine, "reset"):
            engine.reset()  # test positions are unrelated, so nothing should carry over between them
        board = chess.Board(pos["fen"])
        decision = engine.choose(board)
        move = decision.move
        record = {
            "engine": engine.name,
            "set": pos["set"],
            "id": pos["id"],
            "fen": pos["fen"],
            "n_legal": pos["n_legal"],
            "move": board.san(move),
            "move_uci": move.uci(),
            "latency_ms": round(decision.latency_ms, 3),
            "confidence": decision.confidence,
            "probs": decision.info.get("probabilities"),
            "best": board.san(chess.Move.from_uci(pos["best"])),
            "correct": is_correct(pos, move),
            "wp_loss": round(wp_loss(pos["values"], move.uci()), 4),
        }
        if "rating" in pos:
            record["rating"] = pos["rating"]
        records.append(record)
    return records


def percentile(sorted_values: list[float], q: float) -> float:
    return sorted_values[round(q * (len(sorted_values) - 1))]


def summarize(records: list[dict]) -> dict[str, dict]:
    """Per test set: accuracy, average win-chance loss, latency. Puzzles are also split by rating band."""
    summary = {}
    for set_name in sorted({r["set"] for r in records}):
        rs = [r for r in records if r["set"] == set_name]
        latencies = sorted(r["latency_ms"] for r in rs)
        s = {
            "n": len(rs),
            "correct": statistics.mean(r["correct"] for r in rs),
            "wp_loss": statistics.mean(r["wp_loss"] for r in rs),
            "latency_p50_ms": percentile(latencies, 0.50),
            "latency_p95_ms": percentile(latencies, 0.95),
        }
        if set_name == "puzzles":
            s["correct_by_band"] = {}
            for lo, hi in RATING_BANDS:
                band = [r["correct"] for r in rs if lo <= r["rating"] < hi]
                if band:
                    s["correct_by_band"][f"{lo}-{hi}"] = statistics.mean(band)
        summary[set_name] = s
    return summary


def git_state() -> dict:
    """Which commit the code was at, and whether there were uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )
        return {"commit": commit.strip() or None, "uncommitted_changes": bool(status.stdout.strip())}
    except FileNotFoundError:
        return {"commit": None, "uncommitted_changes": None}


def cpu_name() -> str:
    """The CPU's marketing name (platform.processor() only gives a family code on Windows)."""
    if platform.system() == "Windows":
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def environment() -> dict:
    import torch

    env = {
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu": cpu_name(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "packages": {p: version(p) for p in ("torch", "laya", "chess")},
    }
    return env


def save_run(engine: Engine, records: list[dict], summary: dict, testsets_meta: dict, args: dict) -> Path:
    engine_slug = "".join(c if c.isalnum() else "-" for c in engine.name)
    run_id = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S") + "_" + engine_slug
    run_dir = REPO_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True)
    save_jsonl(records, run_dir / "moves.jsonl")

    device = getattr(engine, "device", "cpu")
    manifest = {
        "run_id": run_id,
        "engine": engine.name,
        "engine_version": getattr(engine, "version", None),
        "device": device,
        "args": args,
        "testsets": testsets_meta,
        "git": git_state(),
        "environment": environment(),
        "summary": summary,
    }
    with open(run_dir / "manifest.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)

    summary_csv = REPO_ROOT / "results" / "summary.csv"
    new_file = not summary_csv.exists()
    summary_csv.parent.mkdir(exist_ok=True)
    with open(summary_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        if new_file:
            writer.writeheader()
        for set_name, s in summary.items():
            writer.writerow(
                {
                    "run_id": run_id,
                    "engine": engine.name,
                    "device": device,
                    "set": set_name,
                    "n": s["n"],
                    "correct": round(s["correct"], 4),
                    "wp_loss": round(s["wp_loss"], 4),
                    "latency_p50_ms": round(s["latency_p50_ms"], 3),
                    "latency_p95_ms": round(s["latency_p95_ms"], 3),
                }
            )
    return run_dir
