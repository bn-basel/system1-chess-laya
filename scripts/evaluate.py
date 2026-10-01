"""Run one engine through the test sets and record every move.

Examples:
    uv run python scripts/evaluate.py random
    uv run python scripts/evaluate.py greedy
    uv run python scripts/evaluate.py stockfish --nodes 1000
    uv run python scripts/evaluate.py laya --device cuda
    uv run python scripts/evaluate.py laya --model checkpoints/pilot/final   # a trained checkpoint
    uv run python scripts/evaluate.py laya --limit 50        # quick check on the first 50 positions per set

Writes runs/<run_id>/ (moves.jsonl + manifest.json) and appends to results/summary.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess.engine

from laya_chess.data.testsets import load_jsonl
from laya_chess.eval.runner import evaluate, save_run, summarize

TESTSETS = Path(__file__).resolve().parents[1] / "testsets"


def make_engine(args: argparse.Namespace):
    if args.engine == "random":
        from laya_chess.engines.simple import RandomEngine

        return RandomEngine(seed=args.seed)
    if args.engine == "greedy":
        from laya_chess.engines.simple import GreedyEngine

        return GreedyEngine(seed=args.seed)
    if args.engine == "stockfish":
        from laya_chess.engines import StockfishEngine

        return StockfishEngine(limit=chess.engine.Limit(nodes=args.nodes))
    if args.engine == "laya":
        from laya_chess.engines.laya_engine import LayaEngine

        engine = LayaEngine(model=args.model, device=args.device, notation=args.notation, state_format=args.state)
        engine.warmup()
        return engine
    raise ValueError(args.engine)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("engine", choices=["random", "greedy", "stockfish", "laya"])
    parser.add_argument("--sets", nargs="+", default=["puzzles", "games"])
    parser.add_argument("--limit", type=int, help="only use the first N positions of each set (for quick checks)")
    parser.add_argument("--seed", type=int, default=0, help="random and greedy engines")
    parser.add_argument("--nodes", type=int, default=1000, help="stockfish: nodes per move")
    parser.add_argument("--model", default="convaiinnovations/laya", help="laya: base model or a trained checkpoint")
    parser.add_argument("--device", default=None, help="laya: cuda or cpu (default: auto)")
    parser.add_argument("--notation", choices=["san", "uci"], default="san", help="laya")
    parser.add_argument("--state", choices=["fen", "fen_ascii"], default="fen", help="laya")
    args = parser.parse_args()

    meta = json.loads((TESTSETS / "meta.json").read_text(encoding="utf-8"))
    if not meta.get("labels"):
        raise SystemExit("The test sets aren't labelled yet: run scripts/label_testsets.py first")
    positions = []
    for name in args.sets:
        positions += load_jsonl(TESTSETS / f"{name}.jsonl")[: args.limit]

    engine = make_engine(args)
    try:
        records = evaluate(engine, positions)
    finally:
        engine.close()

    summary = summarize(records)
    run_dir = save_run(engine, records, summary, meta, vars(args))

    print(f"\n{engine.name}  ->  {run_dir.relative_to(run_dir.parents[1])}")
    print(f"{'set':8s} {'n':>5s} {'correct':>8s} {'wp_loss':>8s} {'p50 ms':>8s} {'p95 ms':>8s}")
    for set_name, s in summary.items():
        print(
            f"{set_name:8s} {s['n']:5d} {s['correct']:8.1%} {s['wp_loss']:8.3f}"
            f" {s['latency_p50_ms']:8.2f} {s['latency_p95_ms']:8.2f}"
        )
    if "puzzles" in summary:
        bands = "  ".join(f"{band}: {acc:.0%}" for band, acc in summary["puzzles"]["correct_by_band"].items())
        print(f"puzzles by rating  {bands}")


if __name__ == "__main__":
    main()
