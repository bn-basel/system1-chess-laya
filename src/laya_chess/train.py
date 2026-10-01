"""Fine-tuning Laya on chess, following Laya's official recipe (see the fine-tuning notebook in the Laya repo).

The pieces:
  soft_target    turns Stockfish's evaluations of a few moves into a probability spread over them
  make_example   tokenizes a position exactly as Laya does when playing, and attaches the target
  laya_loss      soft-target cross-entropy + RLCD, the reinforcement-learning term that rewards calibrated confidence
  calibrate      re-fits Laya's confidence temperatures on held-out positions after training
  save_checkpoint writes a folder that `laya.load(path)` and `LayaEngine(model=path)` can open
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import chess
import torch
from laya.agent import _option_logits
from laya.common import collate_items, proper_reward
from safetensors.torch import save_file

from .engines.laya_engine import encode_position
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


def make_example(agent, record: dict, notation: str = "san", state_format: str = "fen", tau: float = TAU):
    """A training example in Laya's own format: token ids, option positions, and a target per option.

    Legal moves that Stockfish didn't evaluate get target 0. Returns None if the moves don't fit Laya's input.
    """
    board = chess.Board(record["fen"])
    try:
        item, _, moves = encode_position(agent, board, notation, state_format)
    except ValueError:
        return None
    weights = dict(zip(record["moves"], soft_target(record["cp"], tau)))
    target = [weights.get(move.uci(), 0.0) for move in moves.values()]  # same order as the options
    return {"ids": item["ids"], "markers": item["markers"], "qtype": item["qtype"], "target": target}


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
