"""Run the board-format experiment: three identical 300k-position training runs, one per board format (~3.5 h).

Each run trains, is tested, and the charts are redrawn (via overnight.py). If it stops, run the same command with
--resume: finished runs are skipped and the interrupted one continues from its last saved checkpoint.

Usage:
    uv run python scripts/board_experiment.py
    uv run python scripts/board_experiment.py --resume
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from laya_chess.experiments import BOARD_FORMAT_EVAL_EVERY, BOARD_FORMAT_POSITIONS, BOARD_FORMAT_RUNS

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    for run, label, state in BOARD_FORMAT_RUNS:
        print(f"\n##### {label} ({run}) #####", flush=True)
        if args.resume and (REPO_ROOT / "checkpoints" / run / "final").exists():
            print("already finished, skipping", flush=True)
            continue
        started = (REPO_ROOT / "checkpoints" / run / "latest").exists()
        command = [sys.executable, "scripts/overnight.py", "--run", run, "--positions", str(BOARD_FORMAT_POSITIONS),
                   "--eval-every", str(BOARD_FORMAT_EVAL_EVERY), "--state", state]  # fmt: skip
        if args.resume and started:
            command.append("--resume")
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
