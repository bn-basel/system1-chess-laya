"""Stockfish's answer key: a value for every legal move in every test position.

For each legal move we play it and let Stockfish evaluate the resulting position with a fixed node budget,
on one thread, with an empty hash table. That makes every label repeatable, and it means any engine can
later be graded by looking up the move it played, without running Stockfish again.
"""

from __future__ import annotations

import chess
import chess.engine

from ..engines.stockfish_engine import find_stockfish

MATE = 100_000  # stand-in centipawn value for a forced mate


def move_values(engine: chess.engine.SimpleEngine, fen: str, nodes: int) -> dict[str, int]:
    """Centipawns for the side to move after each legal move (higher is better for them)."""
    board = chess.Board(fen)
    values = {}
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            cp = MATE
        elif board.is_game_over():
            cp = 0  # stalemate, insufficient material and the like
        else:
            # A new `game` object makes python-chess send "ucinewgame", which clears the hash table,
            # so the result doesn't depend on which positions were evaluated before.
            info = engine.analyse(board, chess.engine.Limit(nodes=nodes), game=object())
            # The score is from the opponent's point of view (they are to move now), so flip it.
            cp = -info["score"].relative.score(mate_score=MATE)
        board.pop()
        values[move.uci()] = cp
    return values


def label_batch(records: list[dict], nodes: int) -> list[dict]:
    """Label a batch of positions with its own Stockfish process (one batch runs per CPU core)."""
    with chess.engine.SimpleEngine.popen_uci(str(find_stockfish())) as engine:
        engine.configure({"Threads": 1, "Hash": 16})
        labelled = []
        for record in records:
            values = move_values(engine, record["fen"], nodes)
            labelled.append({**record, "values": values, "best": max(values, key=values.get)})
    return labelled
