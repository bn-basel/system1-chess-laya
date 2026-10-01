"""The one interface every engine implements, so the harness can treat Laya, Stockfish and baselines alike."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import chess


@dataclass
class MoveDecision:
    move: chess.Move
    latency_ms: float  # wall-clock time for the engine to return this move
    confidence: float | None = None  # engine's own confidence in the move, if it has one
    info: dict[str, Any] = field(default_factory=dict)  # engine-specific extras, logged as-is


class Engine(Protocol):
    name: str

    def choose(self, board: chess.Board) -> MoveDecision: ...

    def close(self) -> None: ...
