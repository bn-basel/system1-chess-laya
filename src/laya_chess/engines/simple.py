"""Two very weak baselines that mark the bottom of the scale."""

from __future__ import annotations

import random
import time

import chess

from .base import MoveDecision

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


class RandomEngine:
    """Plays a random legal move."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.name = "random"

    def choose(self, board: chess.Board) -> MoveDecision:
        t0 = time.perf_counter()
        # Sorting makes the choice depend only on the seed, not on move generation order.
        move = self.rng.choice(sorted(board.legal_moves, key=chess.Move.uci))
        return MoveDecision(move=move, latency_ms=(time.perf_counter() - t0) * 1000)

    def close(self) -> None:
        pass


class GreedyEngine:
    """Takes the most valuable piece it can. If there is nothing to take, plays a random move."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.name = "greedy"

    def _gain(self, board: chess.Board, move: chess.Move) -> int:
        if board.is_en_passant(move):
            return PIECE_VALUES[chess.PAWN]
        captured = board.piece_type_at(move.to_square)
        return PIECE_VALUES[captured] if captured else 0

    def choose(self, board: chess.Board) -> MoveDecision:
        t0 = time.perf_counter()
        moves = sorted(board.legal_moves, key=chess.Move.uci)
        best_gain = max(self._gain(board, m) for m in moves)
        move = self.rng.choice([m for m in moves if self._gain(board, m) == best_gain])
        return MoveDecision(move=move, latency_ms=(time.perf_counter() - t0) * 1000)

    def close(self) -> None:
        pass
