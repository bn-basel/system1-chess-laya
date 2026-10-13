"""Option D: Laya's encoder with a chess move output instead of Laya's decision head.

The input is only the board state (no move options, so it is much shorter). One linear layer reads the encoder's
average token and scores every from-square/to-square pair: 64 x 64 = 4,096 outputs. Illegal pairs are masked out,
and pawns promote to a queen. This is no longer Laya's question format, so it serves partly as a reference point.
"""

from __future__ import annotations

import json
import time
from importlib.metadata import version
from pathlib import Path

import chess
import laya
import torch
from laya.common import encode_text, serialize_state
from safetensors.torch import load_file

from .base import MoveDecision
from .laya_engine import encode_state, playable_moves

BASE_MODEL = "convaiinnovations/laya"
N_MOVES = 64 * 64


def move_index(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square


class MoveHeadModel(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.out = torch.nn.Linear(encoder.config.hidden_size, N_MOVES)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(1) / mask.sum(1)  # average token, as the encoder's own classifier does
        return self.out(pooled).float()


def encode_board(tok, board: chess.Board, state_format: str, max_len: int = 512) -> list[int]:
    """Token ids for the board state alone: [CLS] state [SEP], serialized exactly as Laya serializes a state."""
    text = serialize_state(encode_state(board, state_format)).replace(tok.mask_token, " ")
    ids = encode_text(tok, text, add_special_tokens=False)["input_ids"][: max_len - 2]
    return [tok.cls_token_id] + ids + [tok.sep_token_id]


def collate_boards(examples: list[dict], pad_id: int, device: torch.device) -> dict:
    """Pad a batch; `legal` marks the playable moves and `target` holds their training targets (if given)."""
    n, length = len(examples), max(len(e["ids"]) for e in examples)
    ids = torch.full((n, length), pad_id, dtype=torch.long)
    attention = torch.zeros((n, length), dtype=torch.long)
    legal = torch.zeros((n, N_MOVES), dtype=torch.bool)
    target = torch.zeros((n, N_MOVES), dtype=torch.float32)
    for i, e in enumerate(examples):
        ids[i, : len(e["ids"])] = torch.tensor(e["ids"])
        attention[i, : len(e["ids"])] = 1
        legal[i, e["legal"]] = True
        if "target" in e:
            target[i, e["legal"]] = torch.tensor(e["target"])
    batch = {"input_ids": ids, "attention_mask": attention, "legal": legal, "target": target}
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def move_head_loss(logits: torch.Tensor, batch: dict) -> tuple[torch.Tensor, dict]:
    """Soft-target cross-entropy over the playable moves only."""
    log_p = torch.log_softmax(logits.float().masked_fill(~batch["legal"], -1e4), -1)
    loss = -(batch["target"] * log_p).sum(-1).mean()
    return loss, {"loss_ce": loss.item()}


class MoveHeadEngine:
    def __init__(
        self,
        model: str | None = None,
        device: str | None = None,
        state_format: str | None = None,
        net: MoveHeadModel | None = None,
        tok=None,
        max_len: int = 512,
    ):
        """`model` is a checkpoint folder written by `scripts/train.py --format move_head`. Pass `net` and `tok`
        to wrap a model that is already loaded (used to evaluate during training)."""
        trained_with = {}
        if net is None:
            trained_with = json.loads((Path(model) / "training.json").read_text(encoding="utf-8"))["args"]
            agent = laya.load(BASE_MODEL, device=device)  # the tokenizer, and an encoder to load the weights into
            net = MoveHeadModel(agent.model.encoder)
            net.load_state_dict(load_file(Path(model) / "model.safetensors"))
            net.to(agent.device).eval()
            tok = agent.tok
            max_len = trained_with.get("max_len") or max_len
        self.net, self.tok, self.max_len = net, tok, max_len
        self.state_format = state_format or trained_with.get("state", "fen")
        self.device = str(next(net.parameters()).device)
        checkpoint = Path(model).parent.name + "/" + Path(model).name if model else "in-training"
        self.name = f"laya-movehead[{checkpoint}|{self.state_format}]"
        self.version = f"laya {version('laya')}"

    def warmup(self) -> None:
        self.choose_batch([chess.Board()] * 4)

    def choose(self, board: chess.Board) -> MoveDecision:
        return self.choose_batch([board])[0]  # timed end to end: tokenisation + forward pass + decoding

    @torch.no_grad()
    def choose_batch(self, boards: list[chess.Board]) -> list[MoveDecision]:
        t0 = time.perf_counter()
        moves = [playable_moves(board) for board in boards]
        examples = [
            {
                "ids": encode_board(self.tok, board, self.state_format, self.max_len),
                "legal": [move_index(m) for m in ms],
            }
            for board, ms in zip(boards, moves)
        ]
        b = collate_boards(examples, self.tok.pad_token_id, torch.device(self.device))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            logits = self.net(b["input_ids"], b["attention_mask"])
        p = torch.softmax(logits.masked_fill(~b["legal"], -1e4), -1).cpu()
        decisions = []
        for board, ms, e, row in zip(boards, moves, examples, p):
            probs = row[e["legal"]].tolist()
            best = max(range(len(ms)), key=probs.__getitem__)
            info = {"probabilities": {board.san(m): round(q, 4) for m, q in zip(ms, probs)}}
            decisions.append((ms[best], probs[best], info))
        latency_ms = (time.perf_counter() - t0) * 1000 / len(boards)
        return [MoveDecision(move=m, latency_ms=latency_ms, confidence=c, info=i) for m, c, i in decisions]

    def close(self) -> None:
        pass
