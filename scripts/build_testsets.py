"""Build the fixed test sets from Lichess data (step 1 of 2; then run label_testsets.py).

Downloads the Lichess puzzle database (~300 MB) and one month of Lichess games (~18 MB) into data/raw/,
samples the test positions, and writes them to testsets/.

Usage:
    uv run python scripts/build_testsets.py [--puzzles 1000] [--games 2000] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from laya_chess.data import testsets as ts

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
OUT_DIR = REPO_ROOT / "testsets"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--puzzles", type=int, default=1000)
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    puzzles_file = ts.download(ts.PUZZLES_URL, RAW_DIR / "lichess_db_puzzle.csv.zst")
    games_file = ts.download(ts.GAMES_URL, RAW_DIR / ts.GAMES_URL.rsplit("/", 1)[-1])

    puzzles = ts.sample_puzzles(puzzles_file, args.puzzles, args.seed)
    positions = ts.sample_game_positions(games_file, args.games, args.seed)
    ts.save_jsonl(puzzles, OUT_DIR / "puzzles.jsonl")
    ts.save_jsonl(positions, OUT_DIR / "games.jsonl")

    meta = {
        "built": datetime.now().astimezone().date().isoformat(),
        "seed": args.seed,
        "puzzles": {
            "source": ts.PUZZLES_URL,
            "n": len(puzzles),
            "rating_bands": ts.RATING_BANDS,
            "filters": {
                "max_rating_deviation": ts.MAX_RATING_DEVIATION,
                "min_plays": ts.MIN_PLAYS,
                "min_popularity": ts.MIN_POPULARITY,
            },
        },
        "games": {"source": ts.GAMES_URL, "n": len(positions), "min_ply": ts.MIN_PLY},
        "labels": None,  # filled in by label_testsets.py
    }
    with open(OUT_DIR / "meta.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {len(puzzles)} puzzles and {len(positions)} game positions to {OUT_DIR}")


if __name__ == "__main__":
    main()
