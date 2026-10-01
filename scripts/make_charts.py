"""Draw the README charts from the committed results (results/*.csv) into figures/.

Usage:
    uv run python scripts/make_charts.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from matplotlib.ticker import FuncFormatter, PercentFormatter

from laya_chess.viz import new_figure, save_both_themes

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "figures"


def read_csv(name: str) -> list[dict]:
    with open(RESULTS / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def format_ms(ms: float, _pos=None) -> str:
    if ms == 0:
        return "0"
    if ms < 1:
        return f"{ms * 1000:g} µs"
    return f"{ms:g} ms"


def strength_vs_time(theme: dict):
    """Puzzles solved against time per move: the chart this whole project is about."""
    latest = {(r["engine"], r["set"]): r for r in read_csv("summary.csv")}
    puzzles = {engine: r for (engine, s), r in latest.items() if s == "puzzles"}

    fig, ax = new_figure(
        theme,
        "Puzzles solved vs time per move",
        "1,000 Lichess puzzles. Laya on an RTX 5080, Stockfish 19 on one CPU thread. Up and to the left is better.",
    )

    sf = sorted(  # (time, solved, nodes), joined in order of node budget
        (float(r["latency_p50_ms"]), float(r["correct"]), int(e.split("nodes=")[1].rstrip("]")))
        for e, r in puzzles.items()
        if e.startswith("stockfish[nodes=")
    )
    sf.sort(key=lambda point: point[2])
    ax.plot([t for t, _, _ in sf], [c for _, c, _ in sf], color=theme["stockfish"], linewidth=2, zorder=3)
    ax.plot([t for t, _, _ in sf], [c for _, c, _ in sf], "o", markersize=7, color=theme["stockfish"],
            markeredgecolor=theme["surface"], markeredgewidth=1.5, zorder=4, label="Stockfish, growing node budget")  # fmt: skip
    for t, c, nodes in sf:
        if nodes in (1, 1000, 10_000, 100_000):
            text = f"{nodes:,} node" + ("s" if nodes > 1 else "")
            ax.annotate(text, (t, c), xytext=(8, -14), textcoords="offset points", fontsize=8.5,
                        color=theme["ink_secondary"])  # fmt: skip

    laya = next(r for e, r in puzzles.items() if e.startswith("laya"))
    t, c = float(laya["latency_p50_ms"]), float(laya["correct"])
    ax.plot(t, c, "o", markersize=9, color=theme["laya"], markeredgecolor=theme["surface"], markeredgewidth=1.5,
            zorder=5, label="Laya")  # fmt: skip
    ax.annotate("Laya, before training", (t, c), xytext=(-10, 12), textcoords="offset points", ha="right",
                fontsize=9, fontweight="semibold", color=theme["ink"])  # fmt: skip

    for engine, label in (("random", "Random moves"), ("greedy", "Grab material")):
        r = puzzles[engine]
        t, c = float(r["latency_p50_ms"]), float(r["correct"])
        ax.plot(t, c, "o", markersize=7, color=theme["muted"], markeredgecolor=theme["surface"],
                markeredgewidth=1.5, zorder=4, label="Simple baselines" if engine == "random" else None)  # fmt: skip
        ax.annotate(label, (t, c), xytext=(8, 6), textcoords="offset points", fontsize=8.5,
                    color=theme["ink_secondary"])  # fmt: skip

    ax.set_xscale("log")
    ax.set_xlim(0.01, 200)
    ax.xaxis.set_major_formatter(FuncFormatter(format_ms))
    ax.set_xlabel("Time per move (median, log scale)")
    ax.set_ylim(0, 1.04)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Puzzles solved")
    legend = ax.legend(loc="center right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(theme["ink_secondary"])
    return fig


def laya_speed(theme: dict):
    """Time per position on the GPU as more positions are processed together."""
    rows = read_csv("speed.csv")
    gpu = [r for r in rows if r["device"] == "cuda"]
    cpu = [float(r["ms_per_position"]) for r in rows if r["device"] == "cpu"]

    fig, ax = new_figure(
        theme,
        "Laya: time per position when positions are batched",
        f"{gpu[0]['hardware']}, bf16. Batch size 1 is what playing a game needs; bigger batches help evaluation.",
    )
    xs = [int(r["batch_size"]) for r in gpu]
    ys = [float(r["ms_per_position"]) for r in gpu]
    ax.plot(xs, ys, color=theme["laya"], linewidth=2, zorder=3)
    ax.plot(xs, ys, "o", markersize=7, color=theme["laya"], markeredgecolor=theme["surface"], markeredgewidth=1.5,
            zorder=4)  # fmt: skip
    for x, y, r in zip(xs, ys, gpu):
        ax.annotate(f"{y:.1f} ms\n{float(r['positions_per_s']):.0f}/s", (x, y), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=8.5, color=theme["ink_secondary"])  # fmt: skip

    ax.text(0.98, 0.95, f"On the CPU ({rows[-1]['hardware']}): {min(cpu):.0f}–{max(cpu):.0f} ms per position\n"
            "at every batch size, about 15× slower than the GPU", transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color=theme["ink_secondary"])  # fmt: skip

    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_xlabel("Positions per batch")
    ax.set_ylim(0, max(ys) * 1.3)
    ax.yaxis.set_major_formatter(FuncFormatter(format_ms))
    ax.set_ylabel("Time per position")
    return fig


def main() -> None:
    save_both_themes(strength_vs_time, FIGURES / "strength_vs_time.png")
    save_both_themes(laya_speed, FIGURES / "laya_speed.png")
    print(f"Wrote charts to {FIGURES.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
