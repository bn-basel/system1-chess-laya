"""Richer training targets: score more moves per position with our own Stockfish.

A Lichess record scores 1 to 5 moves (26% only one), so every other legal move gets target 0, as if a decent move
were as bad as a blunder. Here Stockfish scores the top moves (multi-PV) and the extra moves are added to the
record. Lichess's own, deeper scores are kept as they are.
"""

from __future__ import annotations

import chess
import chess.engine

from ..engines.stockfish_engine import find_stockfish
from .label import MATE


def merge_extra(board: chess.Board, record: dict, sf_moves: list[chess.Move], sf_cps: list[int]) -> dict:
    """Add Stockfish's extra moves to a Lichess record without overruling Lichess's deeper analysis.

    An extra move gets Lichess's best score minus how far Stockfish rates it below Stockfish's own best, but never
    more than Lichess's lowest listed move: an analysis that lists k moves rates every other move at most as good
    as its k-th.
    """
    known = {board.parse_uci(uci) for uci in record["moves"]}  # parse_uci also matches castling written as e1h1
    best, floor, sf_best = max(record["cp"]), min(record["cp"]), max(sf_cps)
    moves, cps = list(record["moves"]), list(record["cp"])
    for move, cp in zip(sf_moves, sf_cps):
        if move in known:
            continue
        moves.append(move.uci())
        cps.append(min(floor, best - (sf_best - cp)))
    return {**record, "moves": moves, "cp": cps, "extra": len(moves) - len(record["moves"])}


def _start_engine() -> chess.engine.SimpleEngine:
    engine = chess.engine.SimpleEngine.popen_uci(str(find_stockfish()))
    engine.configure({"Threads": 1, "Hash": 16})
    return engine


def enrich_batch(records: list[dict], nodes: int, multipv: int) -> list[dict]:
    """Enrich a batch of records with its own Stockfish process: one thread, small hash, cleared per position.

    The database holds a few impossible or odd boards (three white knights, four white queens on the back rank)
    on which Stockfish can crash. Those keep their Lichess scores only (`extra` 0), and Stockfish is restarted.
    """
    engine = _start_engine()
    out = []
    try:
        for record in records:
            board = chess.Board(record["fen"])
            if not board.is_valid():
                out.append({**record, "extra": 0})
                continue
            try:
                infos = engine.analyse(board, chess.engine.Limit(nodes=nodes), multipv=multipv, game=object())
            except chess.engine.EngineError:
                out.append({**record, "extra": 0})
                engine = _start_engine()
                continue
            infos = [i for i in infos if i.get("pv")]
            sf_moves = [i["pv"][0] for i in infos]
            sf_cps = [i["score"].relative.score(mate_score=MATE) for i in infos]
            out.append(merge_extra(board, record, sf_moves, sf_cps) if sf_moves else {**record, "extra": 0})
    finally:
        try:
            engine.quit()
        except chess.engine.EngineError:
            pass
    return out
