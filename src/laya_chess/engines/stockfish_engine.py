"""Stockfish via the UCI protocol (python-chess handles the process and the protocol)."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import chess
import chess.engine

from .base import MoveDecision

ENGINES_DIR = Path(__file__).resolve().parents[3] / "engines" / "stockfish"


def find_stockfish(path: str | os.PathLike | None = None) -> Path | None:
    """Look for Stockfish: explicit path, then $STOCKFISH_PATH, then engines/stockfish/, then PATH."""
    for candidate in (path, os.environ.get("STOCKFISH_PATH")):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    # Releases unzip to names like stockfish/stockfish-windows-x86-64-avx2.exe, so search the folder.
    found = sorted(p for p in ENGINES_DIR.rglob("stockfish*") if p.is_file() and p.suffix in (".exe", ""))
    if found:
        return found[0]
    on_path = shutil.which("stockfish")
    return Path(on_path) if on_path else None


class StockfishEngine:
    def __init__(
        self,
        path: str | os.PathLike | None = None,
        limit: chess.engine.Limit | None = None,
        threads: int = 1,
        hash_mb: int = 64,
        elo: int | None = None,
        skill: int | None = None,
    ):
        binary = find_stockfish(path)
        if binary is None:
            raise FileNotFoundError("Stockfish not found; see engines/README.md")
        self.engine = chess.engine.SimpleEngine.popen_uci(str(binary))
        self.limit = limit or chess.engine.Limit(depth=12)

        options: dict[str, int | bool] = {"Threads": threads, "Hash": hash_mb}
        if elo is not None:
            options.update(UCI_LimitStrength=True, UCI_Elo=elo)
        if skill is not None:
            options["Skill Level"] = skill
        self.engine.configure(options)

        self.version = self.engine.id.get("name", "unknown")
        self.name = f"stockfish[{self._describe_limit()}" + (f"|elo={elo}" if elo else "") + "]"

    def _describe_limit(self) -> str:
        parts = [f"{k}={v}" for k in ("depth", "nodes", "time") if (v := getattr(self.limit, k)) is not None]
        return ",".join(parts) or "unlimited"

    def choose(self, board: chess.Board) -> MoveDecision:
        t0 = time.perf_counter()
        result = self.engine.play(board, self.limit, info=chess.engine.INFO_BASIC | chess.engine.INFO_SCORE)
        latency_ms = (time.perf_counter() - t0) * 1000

        score = result.info.get("score")
        return MoveDecision(
            move=result.move,
            latency_ms=latency_ms,
            info={
                "depth": result.info.get("depth"),
                "nodes": result.info.get("nodes"),
                # Score from the side to move's point of view, in centipawns (mates mapped to large values).
                "score_cp": score.relative.score(mate_score=100_000) if score else None,
            },
        )

    def close(self) -> None:
        self.engine.quit()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
