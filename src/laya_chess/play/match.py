"""Playing games between two engines, and recording every move for analysis and the replay page.

A game starts from an opening (a few moves already played), then the engines alternate until checkmate, a draw that
either side could claim (threefold repetition, 50-move rule), stalemate or insufficient material, or MAX_PLIES
half-moves, which counts as a draw.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import chess
import chess.pgn

from ..engines.base import Engine

MAX_PLIES = 400  # a game still going after 200 moves each is scored as a draw
TOP_K = 5  # how many of Laya's highest-probability moves to keep per move (for the replay page)


@dataclass
class GameRecord:
    white: str
    black: str
    opening: str
    result: str  # "1-0", "0-1" or "1/2-1/2"
    termination: str
    moves: list[dict] = field(default_factory=list)  # one record per move played by an engine
    pgn: str = ""

    def score_for(self, name: str) -> float:
        """1 for a win, 0.5 for a draw, 0 for a loss, from `name`'s point of view."""
        white_score = {"1-0": 1.0, "0-1": 0.0}.get(self.result, 0.5)
        return white_score if name == self.white else 1.0 - white_score


def termination(board: chess.Board) -> tuple[str, str]:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return "1/2-1/2", "move limit"
    return outcome.result(), outcome.termination.name.lower().replace("_", " ")


def play_game(white: Engine, black: Engine, opening: dict) -> GameRecord:
    board = chess.Board()
    for uci in opening["moves"]:
        board.push_uci(uci)
    for engine in (white, black):
        if hasattr(engine, "reset"):
            engine.reset()  # a new game: nothing carries over from the previous one

    records = []
    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        engine = white if board.turn == chess.WHITE else black
        decision = engine.choose(board)
        record = {
            "ply": board.ply(),
            "fen": board.fen(),
            "engine": engine.name,
            "move": board.san(decision.move),
            "move_uci": decision.move.uci(),
            "latency_ms": round(decision.latency_ms, 3),
            "confidence": decision.confidence,
        }
        probs = decision.info.get("probabilities")
        if probs:
            record["top"] = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:TOP_K]
        if "score_cp" in decision.info:
            record["score_cp"] = decision.info["score_cp"]
        records.append(record)
        board.push(decision.move)

    result, reason = termination(board)
    game = chess.pgn.Game.from_board(board)
    game.headers.update({"Event": "laya-chess match", "White": white.name, "Black": black.name, "Result": result,
                         "Opening": opening["id"], "Termination": reason})  # fmt: skip
    pgn = io.StringIO()
    print(game, file=pgn)
    return GameRecord(white.name, black.name, opening["id"], result, reason, records, pgn.getvalue())
