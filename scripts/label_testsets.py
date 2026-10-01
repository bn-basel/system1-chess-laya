"""Label the test sets with Stockfish (step 2 of 2, after build_testsets.py).

Stockfish evaluates every legal move of every test position once. Runs in parallel on several CPU cores;
this isn't timed, so it doesn't affect any speed measurement.

Usage:
    uv run python scripts/label_testsets.py [--nodes 300000] [--workers 20]
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import chess.engine
from tqdm import tqdm

from laya_chess.data import label
from laya_chess.data.testsets import load_jsonl, save_jsonl
from laya_chess.engines.stockfish_engine import find_stockfish

TESTSETS = Path(__file__).resolve().parents[1] / "testsets"
BATCH_SIZE = 20  # positions per task; each task starts its own Stockfish


def stockfish_version() -> str:
    engine = chess.engine.SimpleEngine.popen_uci(str(find_stockfish()))
    name = engine.id.get("name", "unknown")
    engine.quit()
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nodes", type=int, default=300_000, help="Stockfish nodes per move evaluated")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    args = parser.parse_args()

    if find_stockfish() is None:
        raise SystemExit("Stockfish not found; see engines/README.md")

    with ProcessPoolExecutor(args.workers) as pool:
        for name in ("puzzles", "games"):
            positions = load_jsonl(TESTSETS / f"{name}.jsonl")
            batches = [positions[i : i + BATCH_SIZE] for i in range(0, len(positions), BATCH_SIZE)]
            results = pool.map(partial(label.label_batch, nodes=args.nodes), batches)
            labelled = [record for batch in tqdm(results, total=len(batches), desc=name) for record in batch]
            save_jsonl(labelled, TESTSETS / f"{name}.jsonl")

    meta_path = TESTSETS / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["labels"] = {"engine": stockfish_version(), "nodes_per_move": args.nodes, "threads": 1, "hash_mb": 16}
    with open(meta_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
