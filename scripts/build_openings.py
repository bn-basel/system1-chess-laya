"""Build the opening positions for engine matches: balanced positions after the first few moves of real games.

Every match game starts from one of these, and each opening is played twice with colours swapped, so neither side
benefits from a lucky opening. Taken from the January 2013 Lichess games (already downloaded for the test sets) and
kept only if Stockfish rates the position as roughly equal.

Writes testsets/openings.jsonl (committed, so every match uses the same openings).

Usage:
    uv run python scripts/build_openings.py [--count 50] [--plies 8]
"""

from __future__ import annotations

import argparse
import io
import random
from pathlib import Path

import chess
import chess.engine
import chess.pgn

from laya_chess.data.testsets import iter_games, save_jsonl
from laya_chess.engines.stockfish_engine import find_stockfish

REPO_ROOT = Path(__file__).resolve().parents[1]
GAMES_FILE = REPO_ROOT / "data" / "raw" / "lichess_db_standard_rated_2013-01.pgn.zst"
MAX_EVAL_CP = 50  # "balanced": within half a pawn according to Stockfish


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--plies", type=int, default=8, help="half-moves already played in each opening")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Collect distinct positions after the first `plies` half-moves of rated games where both players are 1600+,
    # so the openings look like real chess rather than beginner blunders.
    candidates: dict[str, list[str]] = {}
    for pgn in iter_games(GAMES_FILE):
        game = chess.pgn.read_game(io.StringIO(pgn))
        if game is None:
            continue
        try:
            if min(int(game.headers["WhiteElo"]), int(game.headers["BlackElo"])) < 1600:
                continue
        except (KeyError, ValueError):
            continue
        moves = list(game.mainline_moves())[: args.plies]
        if len(moves) < args.plies:
            continue
        board = game.board()
        for move in moves:
            board.push(move)
        candidates.setdefault(board.epd(), [m.uci() for m in moves])
    print(f"{len(candidates)} distinct opening positions")

    rng = random.Random(args.seed)
    order = sorted(candidates)
    rng.shuffle(order)
    openings = []
    with chess.engine.SimpleEngine.popen_uci(str(find_stockfish())) as engine:
        engine.configure({"Threads": 1, "Hash": 16})
        for epd in order:
            board = chess.Board()
            for uci in candidates[epd]:
                board.push_uci(uci)
            info = engine.analyse(board, chess.engine.Limit(nodes=300_000), game=object())
            cp = info["score"].white().score(mate_score=100_000)
            if abs(cp) <= MAX_EVAL_CP:
                openings.append({"id": f"op{len(openings) + 1:02d}", "moves": candidates[epd], "fen": board.fen(),
                                 "eval_cp": cp})  # fmt: skip
            if len(openings) == args.count:
                break
    save_jsonl(openings, REPO_ROOT / "testsets" / "openings.jsonl")
    print(f"Wrote {len(openings)} openings to testsets/openings.jsonl")


if __name__ == "__main__":
    main()
