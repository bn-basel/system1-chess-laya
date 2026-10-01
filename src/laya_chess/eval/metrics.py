"""How a single move is graded."""

from __future__ import annotations

import math

import chess


def win_prob(cp: float) -> float:
    """Chance of winning (0 to 1) for a given evaluation in centipawns, using Lichess's formula.

    Like Lichess, evaluations are capped at +/-1000: beyond that the game is decided either way.
    """
    cp = max(-1000.0, min(1000.0, cp))
    return 1 / (1 + math.exp(-0.00368208 * cp))


def wp_loss(values: dict[str, int], move_uci: str) -> float:
    """How much winning chance the move gave away compared with the best move (0 = as good as the best)."""
    return max(0.0, win_prob(max(values.values())) - win_prob(values[move_uci]))


def is_correct(position: dict, move: chess.Move) -> bool:
    """Puzzles: did it find the solution? Game positions: did it play Stockfish's best move?"""
    if position["set"] != "puzzles":
        return move.uci() == position["best"]
    if move.uci() == position["solution"]:
        return True
    # Lichess accepts any checkmate in mate-in-1 puzzles.
    if "mateIn1" in position["themes"]:
        board = chess.Board(position["fen"])
        board.push(move)
        return board.is_checkmate()
    return False
