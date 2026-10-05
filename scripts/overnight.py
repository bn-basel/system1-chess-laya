"""One command for a long training run: train, then test the final model, then redraw the charts.

Everything printed is also saved to runs/<run>.log, so you can check how the night went.
If the run stops (crash, restart), run the same command with --resume added and it continues where it left off.

Usage:
    uv run python scripts/overnight.py --run overnight1 --positions 2500000
    uv run python scripts/overnight.py --run overnight1 --positions 2500000 --resume
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_step(title: str, args: list[str], log) -> None:
    """Run one script, echoing its output to the console and the log. Stops everything if it fails."""
    header = f"\n=== {title} ({time.strftime('%Y-%m-%d %H:%M')}) ===\n"
    print(header, end="", flush=True)
    log.write(header)
    # Progress bars update at most once a minute, so they don't flood the log with thousands of lines.
    env = {**os.environ, "TQDM_MININTERVAL": "60", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"}
    process = subprocess.Popen(
        [sys.executable, *args], cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
    )  # fmt: skip
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    if process.wait() != 0:
        message = f"\n{title} failed (exit code {process.returncode}). See the log above.\n"
        print(message)
        log.write(message)
        sys.exit(process.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--positions", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    final = REPO_ROOT / "checkpoints" / args.run / "final" / "training.json"
    log_path = REPO_ROOT / "runs" / f"{args.run}.log"
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        if final.exists():
            print(f"{args.run} is already trained, skipping to the evaluation")
        else:
            train = ["scripts/train.py", "--run", args.run, "--positions", str(args.positions)]
            run_step("Training", train + (["--resume"] if args.resume else []), log)
        run_step("Testing the final model", ["scripts/evaluate.py", "laya", "--model", f"checkpoints/{args.run}/final"],
                 log)  # fmt: skip
        run_step("Redrawing the charts", ["scripts/make_charts.py"], log)
        run_step("Results", ["scripts/results_table.py"], log)
    print(f"\nAll done. Log: {log_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
