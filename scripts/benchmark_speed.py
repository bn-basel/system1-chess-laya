"""How fast is Laya, and what hardware does it need? GPU vs CPU, one position at a time vs batches.

Batch size 1 is what matters for playing a game (one move at a time). Bigger batches show throughput, which
matters for evaluating test sets and generating training data.

Writes results/speed.csv.

Usage:
    uv run python scripts/benchmark_speed.py [--positions 256]
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import chess
import torch

from laya_chess.data.testsets import load_jsonl
from laya_chess.engines.laya_engine import LayaEngine
from laya_chess.eval.runner import cpu_name

REPO_ROOT = Path(__file__).resolve().parents[1]
BATCH_SIZES = {"cuda": [1, 4, 16, 64, 256], "cpu": [1, 4, 16, 64]}


def benchmark(engine: LayaEngine, boards: list[chess.Board], batch_size: int) -> dict:
    """Run all positions through in batches; return median time per position and throughput."""
    on_gpu = engine.device.startswith("cuda")
    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    per_position_ms = []
    t0 = time.perf_counter()
    for i in range(0, len(boards), batch_size):
        chunk = boards[i : i + batch_size]
        if batch_size == 1:
            per_position_ms.append(engine.choose(chunk[0]).latency_ms)  # the path used when playing
        else:
            per_position_ms.append(engine.choose_batch(chunk)[0].latency_ms)
    total_s = time.perf_counter() - t0
    return {
        "ms_per_position": round(statistics.median(per_position_ms), 3),
        "positions_per_s": round(len(boards) / total_s, 1),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20) if on_gpu else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--positions", type=int, default=256, help="game positions from the test set to use")
    parser.add_argument("--devices", nargs="+", default=["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"])
    args = parser.parse_args()

    positions = load_jsonl(REPO_ROOT / "testsets" / "games.jsonl")[: args.positions]
    boards = [chess.Board(p["fen"]) for p in positions]

    rows = []
    for device in args.devices:
        engine = LayaEngine(device=device)
        engine.warmup()
        hardware = torch.cuda.get_device_name() if device == "cuda" else cpu_name()
        for batch_size in BATCH_SIZES[device]:
            engine.choose_batch(boards[:batch_size])  # warm up this batch shape
            result = benchmark(engine, boards, batch_size)
            rows.append({"device": device, "hardware": hardware, "batch_size": batch_size, **result})
            print(f"{device:4s} batch {batch_size:3d}: {result}", file=sys.stderr)
        del engine
        torch.cuda.empty_cache()

    out = REPO_ROOT / "results" / "speed.csv"
    fields = ["device", "hardware", "batch_size", "ms_per_position", "positions_per_s", "peak_vram_mb"]
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
