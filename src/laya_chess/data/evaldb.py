"""Training positions from the Lichess evaluation database (https://database.lichess.org/#evals, CC0).

Each line of the database is one position with one or more Stockfish analyses. An analysis lists one or more
principal variations (PVs): the best move, and sometimes the next-best moves, each with an evaluation. We keep
the first move and evaluation of every PV, which is exactly what soft targets need.

The file is ~22 GB compressed, so it's streamed over HTTP and decompressed on the fly: nothing but the
positions we keep is written to disk.
"""

from __future__ import annotations

import io
import json
import urllib.request
from collections.abc import Iterator

import chess
import zstandard

from .label import MATE

EVALS_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"


def stream_evals(url: str = EVALS_URL) -> Iterator[dict]:
    """Yield the database's positions one by one, straight from the download stream."""
    with urllib.request.urlopen(url) as response:
        reader = zstandard.ZstdDecompressor().stream_reader(response)
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            yield json.loads(line)


def to_record(raw: dict) -> dict | None:
    """One database line -> {fen, moves, cp}, with centipawns from the side to move's point of view.

    Uses the analysis with the most PVs (ties go to the deeper one), since alternatives are what make soft
    targets possible. Returns None for positions with fewer than two legal moves.
    """
    analysis = max(raw["evals"], key=lambda e: (len(e["pvs"]), e["depth"]))
    board = chess.Board(raw["fen"] + " 0 1")  # the database leaves out the move counters
    if board.legal_moves.count() < 2:
        return None

    # The database scores positions from White's point of view; flip them when Black is to move.
    sign = 1 if board.turn == chess.WHITE else -1
    moves, cps = [], []
    for pv in analysis["pvs"]:
        if not pv.get("line"):
            continue
        uci = pv["line"].split()[0]
        if uci in moves or chess.Move.from_uci(uci) not in board.legal_moves:
            continue
        cp = pv["cp"] if "cp" in pv else (MATE if pv["mate"] > 0 else -MATE)
        moves.append(uci)
        cps.append(sign * cp)
    if not moves:
        return None
    return {"fen": board.fen(), "moves": moves, "cp": cps, "depth": analysis["depth"]}
