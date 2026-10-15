"""Round 2 overnight: finish the long run with the castling fix, test it, play its matches, then experiment 7.

Steps (each is skipped if already done, so after a stop just run the same command again):
  1. Finish training long-castling-fix (1.5M positions, resumes from its last save)
  2. Test the final model on the test sets
  3. Play its matches, with and without the guard (already-played games are skipped)
  4. Train exp7-rich-targets (300k positions on data/train-rich, see scripts/enrich_trainset.py)
  5. Test that model on the test sets

Safety: each training step needs 25 GB free on C: to start, and is stopped if free space falls below 10 GB while it
runs (Windows can grow its page file during training). A failed step is retried once; a second failure stops the
night. Everything is logged to runs/round2.log, and each step's own output to runs/<step>.log.

Usage:
    uv run python scripts/round2.py
    uv run python scripts/round2.py --dry-run     # only show what would run
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG = REPO_ROOT / "runs" / "round2.log"
MIN_FREE_TO_START_GB = 25
MIN_FREE_WHILE_RUNNING_GB = 10

LONG = "long-castling-fix"
EXP7 = "exp7-rich-targets"
TRAIN = [sys.executable, "scripts/train.py"]
LONG_ARGS = ["--run", LONG, "--positions", "1500000", "--eval-every", "100000", "--state", "fen_pieces",
             "--fix-castling"]  # fmt: skip
EXP7_ARGS = ["--run", EXP7, "--positions", "300000", "--eval-every", "50000", "--state", "fen_pieces",
             "--fix-castling", "--no-resume-state", "--data", "data/train-rich"]  # fmt: skip
MATCH_PLAYERS = ["laya-pieces-1.5M-castling", "laya-pieces-1.5M-castling+guard"]


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def free_gb() -> float:
    return shutil.disk_usage(REPO_ROOT).free / 1e9


def keep_awake() -> None:
    """Ask Windows not to sleep while this script runs, including during the matches (no setting is changed)."""
    if platform.system() == "Windows":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def run(name: str, args: list[str], watch_disk: bool = False) -> bool:
    """Run one step, its output going to runs/<name>.log. Returns True if it exited cleanly."""
    log(f"start {name} ({free_gb():.0f} GB free)")
    with open(REPO_ROOT / "runs" / f"{name}.log", "a", encoding="utf-8") as out:
        process = subprocess.Popen(args, cwd=REPO_ROOT, stdout=out, stderr=subprocess.STDOUT)
        stopped_for_disk = threading.Event()

        def watch() -> None:
            while process.poll() is None:
                if free_gb() < MIN_FREE_WHILE_RUNNING_GB:
                    stopped_for_disk.set()
                    process.kill()
                    return
                time.sleep(60)

        if watch_disk:
            threading.Thread(target=watch, daemon=True).start()
        code = process.wait()
    if stopped_for_disk.is_set():
        log(f"STOP: free space fell below {MIN_FREE_WHILE_RUNNING_GB} GB during {name}; stopped it")
        sys.exit(1)
    log(f"{name} finished (exit code {code}, {free_gb():.0f} GB free)")
    return code == 0


def train(name: str, args: list[str], retry_args: list[str]) -> None:
    """Train unless the final model already exists; one retry, then stop the night."""
    if (REPO_ROOT / "checkpoints" / name / "final" / "model.safetensors").exists():
        log(f"{name}: already trained, skipping")
        return
    for attempt_args in (args, retry_args):
        if free_gb() < MIN_FREE_TO_START_GB:
            log(f"STOP: only {free_gb():.0f} GB free; {name} needs {MIN_FREE_TO_START_GB} GB to start")
            sys.exit(1)
        run(name, TRAIN + attempt_args, watch_disk=True)
        if (REPO_ROOT / "checkpoints" / name / "final" / "model.safetensors").exists():
            return
    log(f"STOP: {name} failed twice; see runs/{name}.log")
    sys.exit(1)


def test(name: str) -> None:
    """Score a final model on the test sets (once)."""
    if (REPO_ROOT / "runs" / f"{name}-test.done").exists():
        log(f"{name}: already tested, skipping")
        return
    if run(f"{name}-test", [sys.executable, "scripts/evaluate.py", "laya", "--model", f"checkpoints/{name}/final"]):
        (REPO_ROOT / "runs" / f"{name}-test.done").touch()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="only show what would run")
    args = parser.parse_args()

    if args.dry_run:
        latest = REPO_ROOT / "checkpoints" / LONG / "latest" / "trainer_state.pt"
        print(f"Free on C: {free_gb():.0f} GB (needs {MIN_FREE_TO_START_GB} GB to start a training step)")
        print(f"1. {LONG}: {'done' if (REPO_ROOT / 'checkpoints' / LONG / 'final').exists() else 'to train'}"
              f" (resume state {'found' if latest.exists() else 'missing'})")  # fmt: skip
        print(f"2. test {LONG}  3. matches {MATCH_PLAYERS}")
        rich = REPO_ROOT / "data" / "train-rich" / "meta.json"
        print(f"4. {EXP7}: data/train-rich {'ready' if rich.exists() else 'MISSING'}  5. test {EXP7}")
        return

    keep_awake()
    (REPO_ROOT / "runs").mkdir(exist_ok=True)
    log("=== round 2 overnight ===")
    # The long run keeps its resume state, so both attempts continue from the last save.
    train(LONG, LONG_ARGS + ["--resume"], LONG_ARGS + ["--resume"])
    test(LONG)
    if not run("matches-round2", [sys.executable, "scripts/play_matches.py", "--laya", *MATCH_PLAYERS]):
        log("matches stopped early; already-played games are kept, rerun this script to finish them")
    if not (REPO_ROOT / "data" / "train-rich" / "meta.json").exists():
        log("STOP: data/train-rich is missing; build it with scripts/enrich_trainset.py")
        sys.exit(1)
    train(EXP7, EXP7_ARGS, EXP7_ARGS)
    test(EXP7)
    log("ALL DONE")


if __name__ == "__main__":
    main()
