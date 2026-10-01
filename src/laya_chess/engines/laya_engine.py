"""Laya as a chess engine, using encoding Option A.

Option A: the position is Laya's *state*, and every legal move is one option of a single
`choice` question. Laya scores all options in one forward pass and we play the argmax.
Because the options are exactly the legal moves, Laya can never play an illegal move.
"""

from __future__ import annotations

import time

import chess
import laya

from .base import MoveDecision

INSTRUCTIONS = "You are playing chess as {side}. Which move is best in this position?"


def encode_state(board: chess.Board, state_format: str = "fen") -> dict:
    """Turn a board into Laya's state. The format is an experiment knob, not a settled choice."""
    state = {
        "fen": board.fen(),
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
    }
    if state_format == "fen_ascii":
        state["board"] = str(board)  # 8x8 grid, one character per square
    elif state_format != "fen":
        raise ValueError(f"unknown state_format {state_format!r}")
    return state


def encode_moves(board: chess.Board, notation: str = "san") -> dict[str, chess.Move]:
    """Label every legal move, e.g. 'Nf3' (SAN) or 'g1f3' (UCI). Returns label -> move."""
    if notation == "san":
        return {board.san(m): m for m in board.legal_moves}
    if notation == "uci":
        return {m.uci(): m for m in board.legal_moves}
    raise ValueError(f"unknown notation {notation!r}")


class LayaEngine:
    def __init__(
        self,
        model: str = "convaiinnovations/laya",
        subfolder: str | None = None,
        device: str | None = None,
        notation: str = "san",
        state_format: str = "fen",
    ):
        self.agent = laya.load(model, device=device, subfolder=subfolder)
        self.notation = notation
        self.state_format = state_format
        self.name = f"laya[{subfolder or 'english'}|{notation}|{state_format}]"

    def warmup(self) -> float:
        """Run a few dummy forward passes so the first timed move isn't paying CUDA start-up costs."""
        return self.agent.warmup()

    def choose(self, board: chess.Board) -> MoveDecision:
        moves = encode_moves(board, self.notation)
        if not moves:
            raise ValueError("no legal moves: the game is over")
        side = "white" if board.turn == chess.WHITE else "black"
        question = {
            "type": "choice",
            "instructions": INSTRUCTIONS.format(side=side),
            "criteria": list(moves),
        }
        state = encode_state(board, self.state_format)

        # Timed end to end (tokenisation + forward pass + decoding): that is the real cost of a move.
        t0 = time.perf_counter()
        result = self.agent.system_one(state, {"move": question})
        latency_ms = (time.perf_counter() - t0) * 1000

        answer = result["answers"]["move"]
        return MoveDecision(
            move=moves[answer["choice"]],
            latency_ms=latency_ms,
            # `answer_confidence` is the calibrated probability of the chosen option (see laya.common).
            confidence=answer["answer_confidence"],
            info={
                "probabilities": answer["probabilities"],
                "act_probability": answer["action"]["act_probability"],
                "usage": result["usage"],
            },
        )

    def close(self) -> None:
        pass
