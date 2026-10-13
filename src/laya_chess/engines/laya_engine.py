"""Laya as a chess engine, using encoding Option A.

Option A: the position is Laya's *state*, and every legal move is one option of a single
`choice` question. Laya scores all options in one forward pass and we play the argmax.
Because the options are exactly the legal moves, Laya can never play an illegal move.
"""

from __future__ import annotations

import json
import time
from importlib.metadata import version
from pathlib import Path

import chess
import laya
import numpy as np
import torch
from laya.common import collate_items

from .base import MoveDecision

INSTRUCTIONS = "You are playing chess as {side}. Which move is best in this position?"


STATE_FORMATS = ("fen", "fen_ascii", "fen_pieces", "pieces")
NOTATIONS = ("san", "uci", "lan")


def piece_list(board: chess.Board, color: chess.Color) -> str:
    """One side's pieces as move-style tokens, e.g. "Ke1 Qd1 Nf3 Pe4": the same square names the move options use."""
    pieces = []
    for piece_type in (chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN):
        letter = chess.piece_symbol(piece_type).upper()
        pieces += [letter + chess.square_name(sq) for sq in board.pieces(piece_type, color)]
    return " ".join(pieces)


def encode_state(board: chess.Board, state_format: str = "fen") -> dict:
    """Turn a board into Laya's state. The fen formats keep the FEN and add a clearer view of the board.

    fen         the FEN and whose turn it is
    fen_ascii   plus an 8x8 grid, one character per square
    fen_pieces  plus a list of each side's pieces with their squares
    pieces      the piece lists without the FEN (shorter): whose turn, castling rights, en-passant square, pieces
    """
    if state_format == "pieces":
        return {
            "side_to_move": "white" if board.turn == chess.WHITE else "black",
            "castling": board.fen().split()[2],
            "en_passant": board.fen().split()[3],
            "white": piece_list(board, chess.WHITE),
            "black": piece_list(board, chess.BLACK),
        }
    state = {
        "fen": board.fen(),
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
    }
    if state_format == "fen_ascii":
        state["board"] = str(board)
    elif state_format == "fen_pieces":
        state["white"] = piece_list(board, chess.WHITE)
        state["black"] = piece_list(board, chess.BLACK)
    elif state_format != "fen":
        raise ValueError(f"unknown state_format {state_format!r}")
    return state


def encode_moves(board: chess.Board, notation: str = "san") -> dict[str, chess.Move]:
    """Label every legal move, e.g. 'Nf3' (SAN), 'g1f3' (UCI) or 'Ng1-f3' (LAN). Returns label -> move."""
    if notation == "san":
        return {board.san(m): m for m in board.legal_moves}
    if notation == "lan":
        return {board.lan(m): m for m in board.legal_moves}
    if notation == "uci":
        return {m.uci(): m for m in board.legal_moves}
    raise ValueError(f"unknown notation {notation!r}")


def build_question(board: chess.Board, notation: str = "san") -> tuple[dict, dict[str, chess.Move]]:
    """The multiple-choice question for this position, and which move each option stands for."""
    moves = encode_moves(board, notation)
    if not moves:
        raise ValueError("no legal moves: the game is over")
    side = "white" if board.turn == chess.WHITE else "black"
    question = {"type": "choice", "instructions": INSTRUCTIONS.format(side=side), "criteria": list(moves)}
    return question, moves


def encode_position(agent, board: chess.Board, notation: str = "san", state_format: str = "fen"):
    """Tokenize a position exactly the way Laya does before a forward pass.

    Used for batched play and for training, so the model is trained on precisely the input it later sees.
    Returns (item, internal question, label -> move). Raises ValueError if the options don't fit Laya's input.
    """
    question, moves = build_question(board, notation)
    internal = {"move": agent._to_internal(question)}
    item = agent._encode_state(encode_state(board, state_format), ["move"], internal)[0]
    return item, internal, moves


# Option A split in two (`question_format="from_to"`): two questions about the same position, asked in one forward
# pass. Each has far fewer options than "which move?" (at most 16 pieces, and the squares they can reach).
FROM_TO_INSTRUCTIONS = {
    "from": "You are playing chess as {side}. Which piece should move?",
    "to": "You are playing chess as {side}. Which square should a piece move to?",
}


def playable_moves(board: chess.Board) -> list[chess.Move]:
    """The legal moves, with pawns promoting to a queen only: the moves a from-square and a to-square can name."""
    return [m for m in board.legal_moves if m.promotion in (None, chess.QUEEN)]


def build_from_to_questions(board: chess.Board) -> tuple[dict, dict[str, list[int]]]:
    """The two questions (which piece, written like the piece list, e.g. "Ng1"; which square, e.g. "f3"), and the
    square each option stands for."""
    moves = playable_moves(board)
    if not moves:
        raise ValueError("no legal moves: the game is over")
    side = "white" if board.turn == chess.WHITE else "black"
    squares = {
        "from": list(dict.fromkeys(m.from_square for m in moves)),
        "to": list(dict.fromkeys(m.to_square for m in moves)),
    }
    labels = {
        "from": [board.piece_at(sq).symbol().upper() + chess.square_name(sq) for sq in squares["from"]],
        "to": [chess.square_name(sq) for sq in squares["to"]],
    }
    questions = {
        qid: {"type": "choice", "instructions": FROM_TO_INSTRUCTIONS[qid].format(side=side), "criteria": labels[qid]}
        for qid in ("from", "to")
    }
    return questions, squares


def encode_from_to(agent, board: chess.Board, state_format: str = "fen"):
    """Tokenize both questions for a position. Returns ([from item, to item], squares per question)."""
    questions, squares = build_from_to_questions(board)
    internal = {qid: agent._to_internal(q) for qid, q in questions.items()}
    items = agent._encode_state(encode_state(board, state_format), ["from", "to"], internal)
    return items, squares


def log_softmax(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def combine_from_to(board: chess.Board, squares: dict, from_logits, to_logits) -> dict[chess.Move, float]:
    """Score every playable move as log p(from) + log p(to), so only legal moves can be chosen."""
    log_from = dict(zip(squares["from"], log_softmax(from_logits)))
    log_to = dict(zip(squares["to"], log_softmax(to_logits)))
    return {m: float(log_from[m.from_square] + log_to[m.to_square]) for m in playable_moves(board)}


PIECE_POINTS = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
GUARD_MIN_LEAD = 3  # material lead (in pawns) from which the guard avoids drawing moves


def material_lead(board: chess.Board) -> int:
    """Material of the side to move minus the opponent's, in pawns."""
    return sum(
        points * (len(board.pieces(piece, board.turn)) - len(board.pieces(piece, not board.turn)))
        for piece, points in PIECE_POINTS.items()
    )


def guarded_choice(board: chess.Board, probabilities: dict[str, float], moves: dict[str, chess.Move]) -> str:
    """The "finish the game" guard: when clearly ahead, don't throw the win away.

    If the side to move leads by at least GUARD_MIN_LEAD pawns of material, play the most likely move that neither
    repeats an earlier position nor stalemates the opponent. If every move does one of those, or the side isn't
    ahead, play Laya's first choice as usual. This only applies the rules and counts material; it doesn't search.
    """
    ranked = sorted(probabilities, key=probabilities.get, reverse=True)
    if material_lead(board) < GUARD_MIN_LEAD:
        return ranked[0]
    for label in ranked:
        board.push(moves[label])
        draws = board.is_stalemate() or board.is_repetition(2)
        board.pop()
        if not draws:
            return label
    return ranked[0]


class LayaEngine:
    def __init__(
        self,
        model: str = "convaiinnovations/laya",
        subfolder: str | None = None,
        device: str | None = None,
        notation: str | None = None,
        state_format: str | None = None,
        agent: laya.Agent | None = None,
        guard: bool = False,
        question_format: str | None = None,
    ):
        """`model` is a Hugging Face id or a local folder with a trained checkpoint. Pass `agent` to wrap an
        already-loaded model instead (used to evaluate during training).

        A trained checkpoint remembers the move notation and board format it was trained with, and those are used
        unless given here. Otherwise the defaults are SAN and FEN. `question_format` is "choice" (one question, every
        legal move an option) or "from_to" (which piece, and which square; see `combine_from_to`).
        """
        self.agent = agent or laya.load(model, device=device, subfolder=subfolder)
        trained_with = {}
        if (Path(model) / "training.json").exists():
            trained_with = json.loads((Path(model) / "training.json").read_text(encoding="utf-8"))["args"]
        notation = notation or trained_with.get("notation", "san")
        state_format = state_format or trained_with.get("state", "fen")
        self.question_format = question_format or trained_with.get("format", "choice")
        if self.question_format == "from_to":
            notation = "from_to"
        self.notation = notation
        self.state_format = state_format
        # A trained checkpoint is named after its run folder, e.g. "pilot/final".
        checkpoint = (
            Path(model).parent.name + "/" + Path(model).name if Path(model).is_dir() else (subfolder or "english")
        )
        self.guard = guard
        self.name = f"laya[{checkpoint}|{notation}|{state_format}]" + ("+guard" if guard else "")
        self.version = f"laya {version('laya')}"
        self.device = str(self.agent.device)

    def warmup(self) -> float:
        """Run a few dummy forward passes so the first timed move isn't paying CUDA start-up costs."""
        return self.agent.warmup()

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
        if self.question_format == "from_to":
            return self.choose_batch([board])[0]  # one position, same path: timed end to end like below
        question, moves = build_question(board, self.notation)
        state = encode_state(board, self.state_format)

        # Timed end to end (tokenisation + forward pass + decoding): that is the real cost of a move.
        t0 = time.perf_counter()
        result = self.agent.system_one(state, {"move": question})
        answer = result["answers"]["move"]
        if self.guard:
            answer = {**answer, "choice": guarded_choice(board, answer["probabilities"], moves)}
        latency_ms = (time.perf_counter() - t0) * 1000

        return self._decision(answer, moves, latency_ms, result["usage"])

    @torch.no_grad()
    def choose_batch(self, boards: list[chess.Board]) -> list[MoveDecision]:
        """Pick moves for many positions in a single forward pass (higher throughput, not lower latency).

        Laya's own `predict_batch` asks the same question about every state, but every chess position has its
        own legal moves. So this runs the same internal steps `predict_batch` uses (encode, collate, forward,
        decode) while giving each position its own question. These are private Laya methods, checked against
        the Laya version pinned in uv.lock. Each decision gets the batch time divided by the batch size.
        """
        if self.question_format == "from_to":
            return self._choose_batch_from_to(boards)
        agent = self.agent
        t0 = time.perf_counter()
        prepared = []
        for board in boards:
            item, internal, moves = encode_position(agent, board, self.notation, self.state_format)
            prepared.append((moves, internal, [item]))

        batch = collate_items([items for _, _, items in prepared], agent.tok.pad_token_id)
        logits, act = agent._forward(batch)
        # One question per position, so position i is row i of the output.
        answers = [
            agent._decode_answers(logits, act, items, ["move"], internal, row)["move"]
            for row, (_, internal, items) in enumerate(prepared)
        ]
        latency_ms = (time.perf_counter() - t0) * 1000 / len(boards)
        return [self._decision(a, moves, latency_ms, None) for a, (moves, _, _) in zip(answers, prepared)]

    def _choose_batch_from_to(self, boards: list[chess.Board]) -> list[MoveDecision]:
        """Both questions for every position in one forward pass; each move scored as log p(from) + log p(to).

        Uses the raw scores (no confidence temperatures), exactly as validation during training does.
        """
        t0 = time.perf_counter()
        prepared = [encode_from_to(self.agent, board, self.state_format) for board in boards]
        batch = collate_items([items for items, _ in prepared], self.agent.tok.pad_token_id)
        logits, _ = self.agent._forward(batch)
        decisions = []
        for i, (board, (items, squares)) in enumerate(zip(boards, prepared)):
            from_logits = logits[2 * i, : len(items[0]["markers"])]
            to_logits = logits[2 * i + 1, : len(items[1]["markers"])]
            scores = combine_from_to(board, squares, from_logits, to_logits)
            p = np.exp(log_softmax(list(scores.values())))
            moves = {board.san(m): m for m in scores}
            probabilities = {label: round(float(v), 4) for label, v in zip(moves, p)}
            label = board.san(max(scores, key=scores.get))  # from the raw scores: rounding can create ties
            if self.guard:
                label = guarded_choice(board, probabilities, moves)
            decisions.append((moves[label], probabilities))
        latency_ms = (time.perf_counter() - t0) * 1000 / len(boards)
        return [
            MoveDecision(move=move, latency_ms=latency_ms, confidence=probs[board.san(move)],
                         info={"probabilities": probs})
            for board, (move, probs) in zip(boards, decisions)
        ]  # fmt: skip

    def close(self) -> None:
        pass
