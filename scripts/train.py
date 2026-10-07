"""Fine-tune Laya on chess positions (Laya's official recipe: soft targets + RLCD + calibration).

Every --eval-every positions it scores the model on the full test sets and on held-out validation positions, writes
a row to results/training/<run>.csv (the learning curve), and saves a checkpoint. At the end it re-fits Laya's
confidence calibration on the validation positions and saves checkpoints/<run>/final, which
`scripts/evaluate.py laya --model checkpoints/<run>/final` can then test properly.

Usage:
    uv run python scripts/train.py --run pilot --positions 50000 --eval-every 10000
    uv run python scripts/train.py --run run1 --positions 1000000
    uv run python scripts/train.py --run run1 --positions 1000000 --resume     # continue after a crash
    uv run python scripts/train.py --run grid --positions 300000 --state fen_ascii   # show the board as a grid
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import statistics
import time
from pathlib import Path

import laya
import numpy as np
import torch
from huggingface_hub import snapshot_download

from laya_chess.data.testsets import load_jsonl
from laya_chess.engines.laya_engine import STATE_FORMATS, LayaEngine
from laya_chess.eval.runner import evaluate_batched, git_state, summarize
from laya_chess.train import calibrate, collate, laya_loss, make_example, save_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "convaiinnovations/laya"
POOL = 4096  # examples encoded at a time, then sorted by length so batches waste little padding


def line_offsets(path: Path, n: int) -> np.ndarray:
    """Byte offset of each of the first n lines, so the training file can be read in a random order."""
    offsets = np.empty(n, dtype=np.int64)
    with open(path, "rb") as f:
        for i in range(n):
            offsets[i] = f.tell()
            if not f.readline():
                raise SystemExit(f"{path} has only {i} positions; build a bigger training set first")
    return offsets


def batches(path: Path, offsets: np.ndarray, start: int, agent, batch_size: int, rng: random.Random, fmt: dict):
    """Yield (batch of examples, how many positions were consumed), starting `start` positions into the order."""
    with open(path, "rb") as f:
        for pool_start in range(start, len(offsets), POOL):
            pool_offsets = offsets[pool_start : pool_start + POOL]
            examples = []
            for offset in pool_offsets:
                f.seek(int(offset))
                example = make_example(agent, json.loads(f.readline()), **fmt)
                if example is not None:
                    examples.append(example)
            examples.sort(key=lambda e: len(e["ids"]))
            chunks = [examples[i : i + batch_size] for i in range(0, len(examples), batch_size)]
            rng.shuffle(chunks)
            for i, chunk in enumerate(chunks):
                # Report pool progress proportionally, so "positions seen" stays accurate.
                yield chunk, len(pool_offsets) * (i + 1) // len(chunks) - len(pool_offsets) * i // len(chunks)


@torch.no_grad()
def validate(model, agent, examples: list[dict], device) -> dict:
    """Cross-entropy and best-move accuracy on held-out positions."""
    model.eval()
    losses, hits = [], []
    for i in range(0, len(examples), 64):
        chunk = examples[i : i + 64]
        b = collate(chunk, agent.tok.pad_token_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
        logits = logits.float().masked_fill(~b["marker_mask"], -1e4)
        losses += (-(b["target"] * torch.log_softmax(logits, -1)).sum(-1)).tolist()
        hits += (logits.argmax(-1) == b["target"].argmax(-1)).float().tolist()
    model.train()
    return {"val_loss": statistics.mean(losses), "val_top1": statistics.mean(hits)}


def keep_awake() -> None:
    """Ask Windows not to sleep while this process runs (the request ends when it exits; no setting is changed)."""
    if platform.system() == "Windows":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="name of this training run")
    parser.add_argument("--positions", type=int, required=True, help="training positions to use (one pass)")
    parser.add_argument("--eval-every", type=int, default=100_000, help="positions between evaluations")
    parser.add_argument("--state", choices=STATE_FORMATS, default="fen", help="how the board is shown to Laya")
    parser.add_argument("--notation", choices=["san", "uci"], default="san", help="how moves are written")
    parser.add_argument("--batch-size", type=int, default=32, help="positions per forward pass")
    parser.add_argument("--accumulate", type=int, default=2, help="forward passes per optimizer step")
    parser.add_argument("--lr-encoder", type=float, default=2.5e-5)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--warmup", type=float, default=0.01, help="share of steps spent warming up the LR")
    parser.add_argument("--sigma", type=float, nargs=2, default=[0.4, 0.1], help="RLCD noise, start and end")
    parser.add_argument(
        "--no-checkpointing",
        dest="checkpointing",
        action="store_false",
        help="turn off gradient checkpointing (faster, but batches over 16 don't fit in 16 GB)",
    )
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "train", help="folder with train/val.jsonl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--no-resume-state",
        action="store_true",
        help="don't save the 4.7 GB optimizer state needed to resume; for short runs that can simply be rerun",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    keep_awake()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    ckpt_dir = REPO_ROOT / "checkpoints" / args.run
    curve_path = REPO_ROOT / "results" / "training" / f"{args.run}.csv"
    train_path = args.data / "train.jsonl"

    agent = laya.load(BASE_MODEL, device="cuda")
    base_dir = Path(snapshot_download(BASE_MODEL, revision=agent.revision, allow_patterns=["tokenizer/*", "encoder/*"]))
    model = agent.model
    if args.checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    encoder_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [{"params": encoder_params, "lr": args.lr_encoder}, {"params": head_params, "lr": args.lr_head}],
        weight_decay=0.01,
    )
    total_steps = math.ceil(args.positions / (args.batch_size * args.accumulate))
    warmup_steps = max(1, int(args.warmup * total_steps))

    def lr_factor(step: int) -> float:
        """Linear warm-up, then cosine decay to (almost) zero."""
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    print("Preparing validation and test positions...")
    fmt = {"notation": args.notation, "state_format": args.state}
    val_examples = [e for r in load_jsonl(args.data / "val.jsonl") if (e := make_example(agent, r, **fmt))]
    test_positions = load_jsonl(REPO_ROOT / "testsets" / "puzzles.jsonl") + load_jsonl(
        REPO_ROOT / "testsets" / "games.jsonl"
    )
    engine = LayaEngine(agent=agent, **fmt)
    engine.name = f"laya[{args.run}]"

    offsets = line_offsets(train_path, args.positions)
    np.random.default_rng(args.seed).shuffle(offsets)

    step, seen, elapsed = 0, 0, 0.0
    if args.resume:
        state = torch.load(ckpt_dir / "latest" / "trainer_state.pt", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, seen, elapsed = state["step"], state["seen"], state["elapsed"]
        print(f"Resuming {args.run} at {seen:,} positions (step {step})")

    def evaluate_now() -> None:
        model.eval()
        t = time.perf_counter()
        summary = summarize(evaluate_batched(engine, test_positions))
        val = validate(model, agent, val_examples[:2000], device)
        model.train()
        row = {
            "positions_seen": seen,
            "step": step,
            "lr_encoder": f"{scheduler.get_last_lr()[0]:.2e}",
            "train_loss": round(statistics.mean(recent_losses), 4) if recent_losses else "",
            "val_loss": round(val["val_loss"], 4),
            "val_top1": round(val["val_top1"], 4),
            "puzzles_solved": round(summary["puzzles"]["correct"], 4),
            "games_best_move": round(summary["games"]["correct"], 4),
            "games_wp_loss": round(summary["games"]["wp_loss"], 4),
            "hours": round(elapsed / 3600, 3),
        }
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not curve_path.exists() or (seen == 0 and not args.resume)
        with open(curve_path, "w" if new_file else "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row), lineterminator="\n")
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        print(f"[{seen:>9,} positions] puzzles {row['puzzles_solved']:.1%} | games best {row['games_best_move']:.1%}"
              f" | win-chance lost {row['games_wp_loss']:.1%} | val top-1 {row['val_top1']:.1%}"
              f" | eval took {time.perf_counter() - t:.0f}s")  # fmt: skip

    def save(name: str, with_state: bool) -> None:
        extra = {"run": args.run, "positions_seen": seen, "step": step, "args": vars(args), "git": git_state()}
        save_checkpoint(model, base_dir, ckpt_dir / name, agent.cfg, extra)
        if with_state:
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "step": step, "seen": seen, "elapsed": elapsed},
                ckpt_dir / name / "trainer_state.pt",
            )  # fmt: skip

    recent_losses: list[float] = []
    if seen == 0:
        evaluate_now()  # the starting point of the curve: the untrained model

    rng = random.Random(args.seed + seen)
    next_eval = (seen // args.eval_every + 1) * args.eval_every
    t_last = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for micro, (chunk, consumed) in enumerate(batches(train_path, offsets, seen, agent, args.batch_size, rng, fmt), 1):
        sigma = args.sigma[0] + (args.sigma[1] - args.sigma[0]) * min(1.0, seen / args.positions)
        b = collate(chunk, agent.tok.pad_token_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
        loss, stats = laya_loss(logits, b, sigma)
        (loss / args.accumulate).backward()
        recent_losses = (recent_losses + [stats["loss_ce"]])[-200:]
        seen += consumed

        if micro % args.accumulate == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step == 20:
                peak = torch.cuda.max_memory_reserved() / torch.cuda.get_device_properties(0).total_memory
                if peak > 0.9:
                    # On Windows, overflow silently moves to system RAM and training becomes many times slower.
                    print(f"WARNING: GPU memory is {peak:.0%} full; use a smaller --batch-size", flush=True)
            if step % 50 == 0:
                now = time.perf_counter()
                elapsed += now - t_last
                t_last = now
                print(f"step {step:>6} | {seen:>9,} positions | loss {statistics.mean(recent_losses):.3f}"
                      f" | {seen / max(elapsed, 1e-9):.0f} pos/s overall", flush=True)  # fmt: skip

        if seen >= next_eval or seen >= args.positions:
            elapsed += time.perf_counter() - t_last
            evaluate_now()
            save("latest", with_state=not args.no_resume_state)
            t_last = time.perf_counter()
            next_eval += args.eval_every

    print("Calibrating confidence on the validation positions...")
    model.eval()
    temperatures = calibrate(agent, val_examples)
    agent.cfg["temperature"] = temperatures["temperature"]
    agent.cfg["temperature_by_options"] = temperatures["temperature_by_options"]
    agent.cfg["model_name"] = f"laya-chess-{args.run}"
    save("final", with_state=False)
    print(f"Saved {ckpt_dir / 'final'} | temperatures {json.dumps(temperatures)}")


if __name__ == "__main__":
    main()
