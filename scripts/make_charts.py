"""Draw the README charts from the committed results (results/*.csv) into figures/.

Usage:
    uv run python scripts/make_charts.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from matplotlib.ticker import FuncFormatter, PercentFormatter

from laya_chess.experiments import BOARD_FORMAT_RUNS
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


def format_positions(n: float, _pos=None) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:g}M"
    if n >= 1000:
        return f"{n / 1000:g}k"
    return f"{n:g}"


def best_trained_model() -> tuple[str, str] | None:
    """(engine name, run name) of the trained Laya with the lowest win-chance lost per move on the game positions.

    That's the steadiest measure we have (2,000 positions), so it decides which model the charts feature.
    """
    games = {r["engine"]: float(r["wp_loss"]) for r in read_csv("summary.csv") if r["set"] == "games"}
    trained = [e for e in games if e.startswith("laya[") and "/final" in e]
    if not trained:
        return None
    engine = min(trained, key=games.get)
    return engine, engine[len("laya[") :].split("/")[0]


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

    # Laya before training (hollow) and the latest trained checkpoint (filled), joined by an arrow.
    base = puzzles["laya[english|san|fen]"]
    t0, c0 = float(base["latency_p50_ms"]), float(base["correct"])
    ax.plot(t0, c0, "o", markersize=9, markerfacecolor=theme["surface"], markeredgecolor=theme["laya"],
            markeredgewidth=2, zorder=5, label="Laya")  # fmt: skip
    ax.annotate("Laya, before training", (t0, c0), xytext=(-10, 6), textcoords="offset points", ha="right",
                fontsize=9, color=theme["ink_secondary"])  # fmt: skip
    if best := best_trained_model():
        engine, run = best
        positions = int(read_csv(f"training/{run}.csv")[-1]["positions_seen"])
        t1, c1 = float(puzzles[engine]["latency_p50_ms"]), float(puzzles[engine]["correct"])
        ax.annotate("", (t1, c1), (t0, c0), arrowprops={"arrowstyle": "-|>", "color": theme["laya"], "lw": 1.5,
                    "shrinkA": 7, "shrinkB": 7}, zorder=4)  # fmt: skip
        ax.plot(t1, c1, "o", markersize=9, color=theme["laya"], markeredgecolor=theme["surface"],
                markeredgewidth=1.5, zorder=5)  # fmt: skip
        ax.annotate(f"Laya, trained on {format_positions(positions)} positions", (t1, c1), xytext=(-10, 6),
                    textcoords="offset points", ha="right", fontsize=9, fontweight="semibold",
                    color=theme["ink"])  # fmt: skip

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


def learning_curve(theme: dict, run: str):
    """Puzzles solved as Laya trains, against the baselines it has to beat."""
    rows = read_csv(f"training/{run}.csv")
    latest = {(r["engine"], r["set"]): r for r in read_csv("summary.csv")}

    fig, ax = new_figure(
        theme,
        "Laya learning chess",
        f"Puzzles solved (of 1,000) after training on more and more positions. Training run: {run}.",
    )
    xs = [int(r["positions_seen"]) for r in rows]
    ys = [float(r["puzzles_solved"]) for r in rows]
    ax.plot(xs, ys, color=theme["laya"], linewidth=2, zorder=3)
    ax.plot(xs, ys, "o", markersize=6, color=theme["laya"], markeredgecolor=theme["surface"], markeredgewidth=1.5,
            zorder=4)  # fmt: skip
    ax.annotate(f"{ys[-1]:.0%}", (xs[-1], ys[-1]), xytext=(0, 10), textcoords="offset points", ha="center",
                fontsize=9, fontweight="semibold", color=theme["ink"])  # fmt: skip

    references = [
        ("stockfish[nodes=1]", "Stockfish, 1 node", theme["stockfish"]),
        ("greedy", "Grab material", theme["muted"]),
        ("random", "Random moves", theme["muted"]),
    ]
    for engine, label, color in references:
        y = float(latest[(engine, "puzzles")]["correct"])
        ax.axhline(y, color=color, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        ax.annotate(label, (1.0, y), xycoords=("axes fraction", "data"), xytext=(-4, 4), textcoords="offset points",
                    ha="right", fontsize=8.5, color=theme["ink_secondary"])  # fmt: skip

    ax.set_xlim(0, xs[-1] * 1.03)
    ax.xaxis.set_major_formatter(FuncFormatter(format_positions))
    ax.set_xlabel("Training positions seen")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Puzzles solved")
    return fig


def board_formats(theme: dict):
    """The board-format experiment: the same short training run with three ways of showing the board."""
    fig, (left, right) = new_figure(
        theme,
        "Does a clearer board help Laya learn?",
        "Three identical training runs that differ only in how the position is shown to Laya.",
        size=(10, 5),
        ncols=2,
    )
    ends = {left: [], right: []}  # (x, y) of each line's last point, labelled after all lines are drawn
    for (run, label, _), color_key in zip(BOARD_FORMAT_RUNS, ("laya", "alt_1", "alt_2")):
        rows = read_csv(f"training/{run}.csv")
        xs = [int(r["positions_seen"]) for r in rows]
        for ax, column in ((left, "puzzles_solved"), (right, "games_wp_loss")):
            ys = [float(r[column]) for r in rows]
            ax.plot(xs, ys, color=theme[color_key], linewidth=2, marker="o", markersize=5,
                    markeredgecolor=theme["surface"], markeredgewidth=1, zorder=3, label=label)  # fmt: skip
            ends[ax].append((xs[-1], ys[-1]))
    for ax, points in ends.items():
        # Spread end labels apart vertically so close values don't print on top of each other.
        low, high = ax.get_ylim()
        gap = (high - low) * 0.045
        placed: list[float] = []
        for x, y in sorted(points, key=lambda p: p[1]):
            label_y = max(y, placed[-1] + gap) if placed else y
            placed.append(label_y)
            ax.annotate(f"{y:.1%}", (x, y), xytext=(8, label_y), textcoords=("offset points", "data"), va="center",
                        fontsize=8.5, color=theme["ink_secondary"])  # fmt: skip
    for ax, ylabel in (
        (left, "Puzzles solved (higher is better)"),
        (right, "Win-chance lost per move (lower is better)"),
    ):
        ax.xaxis.set_major_formatter(FuncFormatter(format_positions))
        ax.set_xlabel("Training positions seen")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_ylabel(ylabel)
    left.set_ylim(bottom=0)
    legend = left.legend(loc="lower right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(theme["ink_secondary"])
    return fig


PLAYER_LABELS = {
    "laya-untrained": "Laya, untrained",
    "laya-pilot-50k": "Laya, pilot (50k positions)",
    "laya-fen-2.5M": "Laya, FEN (2.5M positions)",
    "laya-pieces-300k": "Laya, piece list (300k positions)",
    "laya-pieces-1.5M": "Laya, piece list (1.5M positions)",
    "laya-pieces-1.5M+guard": "Laya, piece list (1.5M) + finish-the-game guard",
    "random": "Random moves",
    "greedy": "Grab material",
}


def elo_ladder(theme: dict):
    """Every player on one Elo scale, with 95% ranges. Stockfish's levels are fixed and set the scale."""
    rows = sorted(read_csv("elo.csv"), key=lambda r: int(r["elo"]))
    fig, ax = new_figure(
        theme,
        "The Elo ladder",
        "3,000 games: each Laya version played 100 games against each opponent. Stockfish levels are fixed at their "
        "stated ratings.",
        size=(8, 5.5),
    )
    for y, r in enumerate(rows):
        elo, low, high = int(r["elo"]), int(r["low_95"]), int(r["high_95"])
        player = r["player"]
        if r["anchor"] == "True":
            color, label, marker = theme["stockfish"], "Stockfish, strength-limited (fixed)", "D"
        elif player.startswith("laya"):
            color, label, marker = theme["laya"], PLAYER_LABELS.get(player, player), "o"
        else:
            color, label, marker = theme["muted"], PLAYER_LABELS.get(player, player), "o"
        if high > low:
            ax.plot([low, high], [y, y], color=color, linewidth=2, alpha=0.45, solid_capstyle="round", zorder=2)
        ax.plot(elo, y, marker, markersize=8, color=color, markeredgecolor=theme["surface"], markeredgewidth=1.5,
                zorder=3)  # fmt: skip
        bold = player == "laya-pieces-1.5M"
        ax.annotate(f"{label}  {elo}", (high, y), xytext=(10, 0), textcoords="offset points", va="center",
                    fontsize=9, color=theme["ink"] if bold else theme["ink_secondary"],
                    fontweight="semibold" if bold else "normal")  # fmt: skip
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(600, 2600)
    ax.set_xticks(range(600, 2101, 300))
    ax.set_xlabel("Elo (Stockfish's scale, roughly the CCRL blitz list)")
    ax.grid(axis="y", visible=False)
    return fig


def main() -> None:
    save_both_themes(strength_vs_time, FIGURES / "strength_vs_time.png")
    save_both_themes(laya_speed, FIGURES / "laya_speed.png")
    if best := best_trained_model():
        run = best[1]
        save_both_themes(lambda theme: learning_curve(theme, run), FIGURES / "learning_curve.png")
    if all((RESULTS / "training" / f"{run}.csv").exists() for run, _, _ in BOARD_FORMAT_RUNS):
        save_both_themes(board_formats, FIGURES / "board_formats.png")
    if (RESULTS / "elo.csv").exists():
        save_both_themes(elo_ladder, FIGURES / "elo_ladder.png")
    print(f"Wrote charts to {FIGURES.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
