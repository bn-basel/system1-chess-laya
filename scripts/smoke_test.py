"""Phase 0 smoke test: can we load Laya, ask it for chess moves, and time it?

Not a benchmark. It checks that the pieces work end to end and gives a first feel for:
  - what Laya picks on a few hand-chosen positions (and how that compares with Stockfish, if installed),
  - whether all legal moves fit in Laya's option budget (the token-budget question),
  - roughly how long one move takes.

Usage:
    uv run python scripts/smoke_test.py [--device cuda|cpu] [--notation san|uci] [--state fen|fen_ascii]
"""

from __future__ import annotations

import argparse
import statistics
import time
from importlib.metadata import version

import chess
import chess.engine
import torch

from laya_chess.engines import StockfishEngine, find_stockfish
from laya_chess.engines.laya_engine import LayaEngine

POSITIONS = {
    "start": chess.STARTING_FEN,
    # 1.e4 e5 2.Nf3 Nc6 3.Bc4, a normal opening position with Black to move.
    "italian": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    # Scholar's mate set-up: Qxf7# is the only good move.
    "mate_in_1": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    # "Kiwipete", a well-known test position with 48 legal moves. Checks that all moves fit in Laya's input.
    "kiwipete": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
}


def print_environment(device: str) -> None:
    print("== Environment")
    print(f"  torch {torch.__version__} | laya {version('laya')} | python-chess {version('chess')}")
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        print(f"  CUDA {torch.version.cuda} | {torch.cuda.get_device_name()} (sm_{major}{minor})")
    else:
        print("  CUDA not available")
    print(f"  running Laya on: {device}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--notation", choices=["san", "uci"], default="san")
    parser.add_argument("--state", choices=["fen", "fen_ascii"], default="fen")
    parser.add_argument("--timing-runs", type=int, default=50)
    args = parser.parse_args()

    print_environment(args.device)

    print("\n== Loading Laya (downloads ~1.7 GB on first run)")
    t0 = time.perf_counter()
    laya_engine = LayaEngine(device=args.device, notation=args.notation, state_format=args.state)
    print(f"  loaded {laya_engine.name} in {time.perf_counter() - t0:.1f} s, warmup {laya_engine.warmup():.2f} s")

    stockfish = None
    if find_stockfish():
        stockfish = StockfishEngine(limit=chess.engine.Limit(depth=18))
        print(f"  reference: {stockfish.version} at depth 18")
    else:
        print("  Stockfish not found, skipping the reference column (see engines/README.md)")

    print("\n== Moves")
    for name, fen in POSITIONS.items():
        board = chess.Board(fen)
        decision = laya_engine.choose(board)
        n_legal = board.legal_moves.count()
        usage = decision.info["usage"]

        top5 = sorted(decision.info["probabilities"].items(), key=lambda kv: kv[1], reverse=True)[:5]
        top5_str = ", ".join(f"{label} {p:.3f}" for label, p in top5)

        print(f"\n  [{name}] {n_legal} legal moves, uniform guess = {1 / n_legal:.3f}")
        print(
            f"    Laya:      {board.san(decision.move):8s} confidence {decision.confidence:.3f}"
            f"  act_prob {decision.info['act_probability']:.3f}  ({decision.latency_ms:.1f} ms)"
        )
        print(f"    top-5:     {top5_str}")
        if stockfish:
            best = stockfish.choose(board)
            verdict = "MATCH" if best.move == decision.move else "differs"
            print(f"    Stockfish: {board.san(best.move):8s} eval {best.info['score_cp']:+} cp  -> {verdict}")
        # Laya reports when the state was cut or when option labels collapsed for lack of tokens.
        budget = {k: usage[k] for k in ("truncated", "state_tokens", "options") if k in usage}
        print(f"    budget:    {budget}")

    print(f"\n== Latency ({args.timing_runs} runs per position, batch size 1)")
    for name, fen in POSITIONS.items():
        board = chess.Board(fen)
        times = sorted(laya_engine.choose(board).latency_ms for _ in range(args.timing_runs))
        p50 = statistics.median(times)
        p95 = times[int(0.95 * (len(times) - 1))]
        print(f"  {name:10s} p50 {p50:6.1f} ms   p95 {p95:6.1f} ms   min {times[0]:6.1f} ms")

    if stockfish:
        stockfish.close()


if __name__ == "__main__":
    main()
