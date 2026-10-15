"""Make a training set with richer targets: Stockfish scores the top moves of each position (see data/enrich.py).

Takes the first --positions lines of data/train/train.jsonl, in the same order, so a run on the new folder sees the
same positions in the same shuffled order as a run on the original. The validation set is copied unchanged.
If it stops, run it again: finished lines are kept and it continues after them.

Writes data/train-rich/ (kept out of git). Train on it with `scripts/train.py --data data/train-rich`.

Usage:
    uv run python scripts/enrich_trainset.py [--positions 300000] [--multipv 10] [--nodes 300000] [--workers 12]
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path

from tqdm import tqdm

from laya_chess.data.enrich import enrich_batch

REPO_ROOT = Path(__file__).resolve().parents[1]
BATCH = 250


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--positions", type=int, default=300_000)
    parser.add_argument("--multipv", type=int, default=10, help="moves Stockfish scores per position")
    parser.add_argument("--nodes", type=int, default=300_000, help="Stockfish nodes per position")
    parser.add_argument("--workers", type=int, default=12, help="Stockfish processes (one thread each)")
    parser.add_argument("--source", type=Path, default=REPO_ROOT / "data" / "train")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "train-rich")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.source / "val.jsonl", args.out / "val.jsonl")
    out_path = args.out / "train.jsonl"
    done = 0
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = sum(1 for _ in f)
    with open(args.source / "train.jsonl", encoding="utf-8") as f:
        records = [json.loads(line) for line in itertools.islice(f, done, args.positions)]
    batches = [records[i : i + BATCH] for i in range(0, len(records), BATCH)]
    print(f"{done:,} positions already done, {len(records):,} to go")

    extra = 0
    with open(out_path, "a", encoding="utf-8", newline="\n") as out, ProcessPoolExecutor(args.workers) as pool:
        progress = tqdm(total=len(records), unit=" pos")
        for batch in pool.map(partial(enrich_batch, nodes=args.nodes, multipv=args.multipv), batches):
            for record in batch:
                out.write(json.dumps(record) + "\n")
                extra += record["extra"]
            out.flush()
            progress.update(len(batch))
        progress.close()

    meta = {"built": datetime.now().astimezone().isoformat(timespec="seconds"), "source": str(args.source),
            "positions": args.positions, "multipv": args.multipv, "nodes": args.nodes, "threads": 1, "hash_mb": 16,
            "extra_moves_this_session": extra}  # fmt: skip
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
