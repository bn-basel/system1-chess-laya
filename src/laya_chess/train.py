"""Fine-tuning Laya on chess, following Laya's official recipe (see the fine-tuning notebook in the Laya repo).

The pieces:
  soft_target    turns Stockfish's evaluations of a few moves into a probability spread over them
  make_example   tokenizes a position exactly as Laya does when playing, and attaches the target
  make_from_to_example / make_move_head_example  the same for the from/to questions and for the move-output model
  laya_loss      soft-target cross-entropy + RLCD, the reinforcement-learning term that rewards calibrated confidence
  calibrate      re-fits Laya's confidence temperatures on held-out positions after training
  save_checkpoint writes a folder that `laya.load(path)` and `LayaEngine(model=path)` can open
"""

from __future__ import annotations

import json
import math
import shutil
import statistics
from pathlib import Path

import chess
import torch
from laya.agent import _option_logits
from laya.common import collate_items, proper_reward
from safetensors.torch import save_file

from .engines.laya_engine import combine_from_to, encode_from_to, encode_position, log_softmax, playable_moves
from .engines.move_head import collate_boards, encode_board, move_index
from .eval.metrics import win_prob

# How quickly the target falls off for worse moves: a move with 5 percentage points less winning chance than the
# best one gets e^-1 (about 37%) of the best move's weight; 15 points less gets about 5%.
TAU = 0.05


def soft_target(cps: list[int], tau: float = TAU) -> list[float]:
    """Probability spread over the evaluated moves, by winning chance. One move in, one-hot out."""
    wins = [win_prob(cp) for cp in cps]
    best = max(wins)
    weights = [math.exp((w - best) / tau) for w in wins]
    total = sum(weights)
    return [w / total for w in weights]


def move_weights(board: chess.Board, record: dict, tau: float = TAU, fix_castling: bool = False) -> dict[str, float]:
    """Target weight per evaluated move, keyed by UCI.

    The Lichess database writes castling as the king taking its own rook (e1h1), while python-chess writes e1g1. So
    without `fix_castling`, castling never matches a legal move and gets target 0 (as in every run up to round 1).
    """
    weights = dict(zip(record["moves"], soft_target(record["cp"], tau)))
    if fix_castling:
        weights = {board.parse_uci(uci).uci(): w for uci, w in weights.items()}
    return weights


def make_example(
    agent, record: dict, notation: str = "san", state_format: str = "fen", tau: float = TAU, fix_castling: bool = False
):
    """A training example in Laya's own format: token ids, option positions, and a target per option.

    Legal moves that Stockfish didn't evaluate get target 0. Returns None if the moves don't fit Laya's input.
    """
    board = chess.Board(record["fen"])
    try:
        item, _, moves = encode_position(agent, board, notation, state_format)
    except ValueError:
        return None
    weights = move_weights(board, record, tau, fix_castling)
    target = [weights.get(move.uci(), 0.0) for move in moves.values()]  # same order as the options
    return {"ids": item["ids"], "markers": item["markers"], "qtype": item["qtype"], "target": target}


def move_level_targets(board: chess.Board, record: dict, tau: float = TAU, fix_castling: bool = False):
    """For the formats that pick a from-square and a to-square: the playable moves, a target for each (an
    under-promotion's weight goes to the queen promotion on the same squares), and the best move.

    The best move is defined exactly as validation does for `make_example`: the legal move with the highest target,
    the first one in python-chess's move order on ties. So move-level top-1 stays comparable across formats.
    """
    weights = move_weights(board, record, tau, fix_castling)
    legal = list(board.legal_moves)
    legal_weights = [weights.get(m.uci(), 0.0) for m in legal]
    best = legal[max(range(len(legal)), key=legal_weights.__getitem__)]
    moves = playable_moves(board)
    index = {(m.from_square, m.to_square): i for i, m in enumerate(moves)}
    target = [0.0] * len(moves)
    for m, w in zip(legal, legal_weights):
        target[index[m.from_square, m.to_square]] += w
    return moves, target, best


def make_from_to_example(agent, record: dict, state_format: str = "fen", fix_castling: bool = False):
    """Both questions for one position. `rows` are two Laya training items (which piece, which square), each with
    the move targets summed per square. The rest is kept for move-level validation."""
    board = chess.Board(record["fen"])
    try:
        items, squares = encode_from_to(agent, board, state_format)
    except ValueError:
        return None
    moves, target, best = move_level_targets(board, record, fix_castling=fix_castling)
    from_target = dict.fromkeys(squares["from"], 0.0)
    to_target = dict.fromkeys(squares["to"], 0.0)
    for m, w in zip(moves, target):
        from_target[m.from_square] += w
        to_target[m.to_square] += w
    rows = [
        {"ids": item["ids"], "markers": item["markers"], "qtype": item["qtype"], "target": list(t.values())}
        for item, t in zip(items, (from_target, to_target))
    ]
    return {"rows": rows, "fen": record["fen"], "squares": squares, "moves": moves, "target": target, "best": best}


def make_move_head_example(tok, record: dict, state_format: str = "fen", max_len: int = 512, fix_castling=False):
    """The board alone as input, and a target over the playable moves (as from-square * 64 + to-square)."""
    board = chess.Board(record["fen"])
    moves, target, best = move_level_targets(board, record, fix_castling=fix_castling)
    row = {"ids": encode_board(tok, board, state_format, max_len), "legal": [move_index(m) for m in moves]}
    row["target"] = target
    return {"rows": [row], "fen": record["fen"], "moves": moves, "target": target, "best": best}


def move_level_scores(scores: list[float], example: dict) -> dict:
    """Cross-entropy against the move targets, and whether the top move is the best move."""
    log_p = log_softmax(scores)
    top = example["moves"][int(log_p.argmax())]
    return {"loss": float(-(log_p * example["target"]).sum()), "hit": float(top == example["best"])}


def _summary(results: list[dict]) -> dict:
    return {
        "val_loss": statistics.mean(r["loss"] for r in results),
        "val_top1": statistics.mean(r["hit"] for r in results),
    }


@torch.no_grad()
def validate_from_to(model, pad_id: int, examples: list[dict], device) -> dict:
    """Move-level validation for the from/to questions: each move scored as log p(from) + log p(to)."""
    model.eval()
    results = []
    for i in range(0, len(examples), 32):
        chunk = examples[i : i + 32]
        b = collate([row for e in chunk for row in e["rows"]], pad_id, device)
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            logits, _ = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
        logits = logits.float().cpu().numpy()
        for j, e in enumerate(chunk):
            from_logits = logits[2 * j, : len(e["rows"][0]["markers"])]
            to_logits = logits[2 * j + 1, : len(e["rows"][1]["markers"])]
            scores = combine_from_to(chess.Board(e["fen"]), e["squares"], from_logits, to_logits)
            results.append(move_level_scores([scores[m] for m in e["moves"]], e))
    model.train()
    return _summary(results)


@torch.no_grad()
def validate_move_head(model, pad_id: int, examples: list[dict], device) -> dict:
    """Move-level validation for the move-output model."""
    model.eval()
    results = []
    for i in range(0, len(examples), 64):
        chunk = examples[i : i + 64]
        b = collate_boards([e["rows"][0] for e in chunk], pad_id, device)
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            logits = model(b["input_ids"], b["attention_mask"]).float().cpu()
        for row, e in zip(logits, chunk):
            results.append(move_level_scores(row[e["rows"][0]["legal"]].tolist(), e))
    model.train()
    return _summary(results)


def collate(examples: list[dict], pad_id: int, device: torch.device) -> dict:
    batch = collate_items([[e] for e in examples], pad_id)
    keys = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype", "target")
    return {k: batch[k].to(device, non_blocking=True) for k in keys}


def laya_loss(logits: torch.Tensor, batch: dict, sigma: float, group_size: int = 4) -> tuple[torch.Tensor, dict]:
    """Laya's training objective: soft-target cross-entropy plus RLCD.

    Cross-entropy pulls the predicted spread towards the target. RLCD samples a few noisy versions of the
    prediction, scores each with a strictly proper scoring rule (one that is best only when the reported
    probabilities are honest), and nudges the model towards the better-scoring ones. Ported from the official
    fine-tuning notebook, with the same weights.
    """
    logits = logits.float()
    mask = batch["marker_mask"]
    target = batch["target"]
    k = mask.sum(-1, keepdim=True).float()

    # Sample noisy distributions around the prediction (noise sums to zero over each row's options).
    eps = torch.randn((group_size, *logits.shape), device=logits.device) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        reward = proper_reward(q, target.unsqueeze(0), batch["qtype"], mask, w_sph=0.75, w_rps=1.0)
        advantage = reward - reward.mean(0, keepdim=True)
        advantage = advantage / (advantage.std() + 1e-6)
    log_prob = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
    loss_rl = -(advantage * log_prob).mean()

    log_p = torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)
    loss_ce = -(target * log_p).sum(-1).mean()
    stats = {"loss_ce": loss_ce.item(), "loss_rl": loss_rl.item(), "reward": reward.mean().item()}
    return loss_ce + loss_rl, stats


@torch.no_grad()
def calibrate(agent, examples: list[dict], batch_size: int = 64) -> dict:
    """Fit Laya's confidence temperatures on held-out examples (installs them on `agent`, returns them)."""
    records = []
    for i in range(0, len(examples), batch_size):
        chunk = examples[i : i + batch_size]
        batch = collate_items([[e] for e in chunk], agent.tok.pad_token_id)
        logits, _ = agent._forward(batch)
        rows = _option_logits(logits, chunk, 0)
        for e, row in zip(chunk, rows):
            records.append((e["qtype"], row, e["target"], len(e["markers"])))
    result = agent.fit_temperatures(records)
    return {"temperature": result["temperature"], "temperature_by_options": result["temperature_by_options"]}


def save_checkpoint(model: torch.nn.Module, base_dir: Path, out_dir: Path, cfg: dict, extra: dict) -> None:
    """Write a Laya checkpoint folder: weights (fp16), config, and the base model's tokenizer and encoder config."""
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = {k: v.detach().half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(weights, out_dir / "model.safetensors")
    for sub in ("tokenizer", "encoder"):
        shutil.copytree(base_dir / sub, out_dir / sub, dirs_exist_ok=True)
    (out_dir / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out_dir / "training.json").write_text(json.dumps(extra, indent=2, default=str), encoding="utf-8")
