"""Laya as a chess engine, using encoding Option A.

Option A: the position is Laya's *state*, and every legal move is one option of a single
`choice` question. Laya scores all options in one forward pass and we play the argmax.
Because the options are exactly the legal moves, Laya can never play an illegal move.
"""

from __future__ import annotations

import time
from importlib.metadata import version

import chess
import laya
import torch
from laya.common import collate_items

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
        self.version = f"laya {version('laya')}"
        self.device = str(self.agent.device)

    def warmup(self) -> float:
        """Run a few dummy forward passes so the first timed move isn't paying CUDA start-up costs."""
        return self.agent.warmup()

    def _question(self, board: chess.Board) -> tuple[dict, dict[str, chess.Move]]:
        """The multiple-choice question for this position, and which move each option stands for."""
        moves = encode_moves(board, self.notation)
        if not moves:
            raise ValueError("no legal moves: the game is over")
        side = "white" if board.turn == chess.WHITE else "black"
        question = {"type": "choice", "instructions": INSTRUCTIONS.format(side=side), "criteria": list(moves)}
        return question, moves

    @staticmethod
    def _decision(answer: dict, moves: dict[str, chess.Move], latency_ms: float, usage: dict | None) -> MoveDecision:
        return MoveDecision(
            move=moves[answer["choice"]],
            latency_ms=latency_ms,
            # `answer_confidence` is the calibrated probability of the chosen option (see laya.common).
            confidence=answer["answer_confidence"],
            info={
                "probabilities": answer["probabilities"],
                "act_probability": answer["action"]["act_probability"],
                "usage": usage,
            },
        )

    def choose(self, board: chess.Board) -> MoveDecision:
        question, moves = self._question(board)
        state = encode_state(board, self.state_format)

        # Timed end to end (tokenisation + forward pass + decoding): that is the real cost of a move.
        t0 = time.perf_counter()
        result = self.agent.system_one(state, {"move": question})
        latency_ms = (time.perf_counter() - t0) * 1000

        return self._decision(result["answers"]["move"], moves, latency_ms, result["usage"])

    @torch.no_grad()
    def choose_batch(self, boards: list[chess.Board]) -> list[MoveDecision]:
        """Pick moves for many positions in a single forward pass (higher throughput, not lower latency).

        Laya's own `predict_batch` asks the same question about every state, but every chess position has its
        own legal moves. So this runs the same internal steps `predict_batch` uses (encode, collate, forward,
        decode) while giving each position its own question. These are private Laya methods, checked against
        the Laya version pinned in uv.lock. Each decision gets the batch time divided by the batch size.
        """
        agent = self.agent
        t0 = time.perf_counter()
        prepared = []
        for board in boards:
            question, moves = self._question(board)
            internal = {"move": agent._to_internal(question)}
            items = agent._encode_state(encode_state(board, self.state_format), ["move"], internal)
            prepared.append((moves, internal, items))

        batch = collate_items([items for _, _, items in prepared], agent.tok.pad_token_id)
        logits, act = agent._forward(batch)
        # One question per position, so position i is row i of the output.
        answers = [
            agent._decode_answers(logits, act, items, ["move"], internal, row)["move"]
            for row, (_, internal, items) in enumerate(prepared)
        ]
        latency_ms = (time.perf_counter() - t0) * 1000 / len(boards)
        return [self._decision(a, moves, latency_ms, None) for a, (moves, _, _) in zip(answers, prepared)]

    def close(self) -> None:
        pass
