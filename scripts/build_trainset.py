"""Build the training set from the Lichess evaluation database (streamed, nothing else is downloaded).

Takes every Nth position (spreading the sample over more of the database), removes every position that appears in
our test sets, and holds out a small validation set for monitoring and for calibrating confidence after training.

Writes data/train/train.jsonl, data/train/val.jsonl and data/train/meta.json (all kept out of git).

Usage:
    uv run python scripts/build_trainset.py [--positions 6000000] [--every 2] [--val 5000]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import chess
from tqdm import tqdm

from laya_chess.data.evaldb import EVALS_URL, stream_evals, to_record
from laya_chess.data.testsets import load_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "train"


def test_positions() -> set[str]:
    """EPDs (position without move counters) of every test position, so none of them can leak into training."""
    epds = set()
    for name in ("puzzles", "games"):
        for record in load_jsonl(REPO_ROOT / "testsets" / f"{name}.jsonl"):
            epds.add(chess.Board(record["fen"]).epd())
    return epds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--positions", type=int, default=6_000_000, help="training positions to keep")
    parser.add_argument("--every", type=int, default=2, help="keep every Nth position of the database")
    parser.add_argument("--val", type=int, default=5000, help="validation positions (taken from the same stream)")
    args = parser.parse_args()

    excluded = test_positions()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = {"read": 0, "kept": 0, "val": 0, "test_overlap": 0, "skipped": 0}
    val_every = max(1, args.positions // args.val)  # spread validation positions over the whole sample

    with (
        open(OUT_DIR / "train.jsonl", "w", encoding="utf-8", newline="\n") as train,
        open(OUT_DIR / "val.jsonl", "w", encoding="utf-8", newline="\n") as val,
    ):
        progress = tqdm(total=args.positions + args.val, unit=" pos")
        for raw in stream_evals():
            counts["read"] += 1
            if counts["read"] % args.every:
                continue
            record = to_record(raw)
            if record is None:
                counts["skipped"] += 1
                continue
            if chess.Board(record["fen"]).epd() in excluded:
                counts["test_overlap"] += 1
                continue
            if counts["val"] < args.val and (counts["kept"] + counts["val"]) % val_every == 0:
                val.write(json.dumps(record) + "\n")
                counts["val"] += 1
            else:
                train.write(json.dumps(record) + "\n")
                counts["kept"] += 1
            progress.update()
            if counts["kept"] >= args.positions and counts["val"] >= args.val:
                break
        progress.close()

    meta = {"built": datetime.now().astimezone().isoformat(timespec="seconds"), "source": EVALS_URL, **vars(args),
            **counts}  # fmt: skip
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
