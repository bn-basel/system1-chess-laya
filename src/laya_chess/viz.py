"""One chart style for the whole project, in a light and a dark version (GitHub shows the one matching the reader's
theme). Colors follow the entity: Stockfish is always blue, Laya is always orange, simple baselines are gray.
The two series colors were checked for color-blind safety in both themes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "stockfish": "#2a78d6",
        "laya": "#eb6834",
        "alt_1": "#1baf7a",  # extra series, for comparing Laya variants (checked together with "laya")
        "alt_2": "#4a3aa7",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "stockfish": "#3987e5",
        "laya": "#d95926",
        "alt_1": "#199e70",
        "alt_2": "#9085e9",
    },
}

FONTS = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]  # first one installed is used


def _font() -> str:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    return next(name for name in FONTS if name in installed or name == "DejaVu Sans")


def new_figure(theme: dict, title: str, subtitle: str, size=(8, 5), ncols: int = 1):
    """A figure with the title and subtitle left-aligned above the plot(s), and quiet axes.

    Returns (fig, ax), or (fig, list of axes) when ncols > 1.
    """
    plt.rcParams.update({"font.family": _font(), "font.size": 10})
    fig, axes = plt.subplots(1, ncols, figsize=size, dpi=200, squeeze=False)
    axes = list(axes[0])
    fig.patch.set_facecolor(theme["surface"])
    fig.text(0.07, 0.95, title, fontsize=14, fontweight="semibold", color=theme["ink"], va="top")
    fig.text(0.07, 0.895, subtitle, fontsize=9.5, color=theme["ink_secondary"], va="top")
    fig.subplots_adjust(left=0.1, right=0.95, top=0.8, bottom=0.13, wspace=0.3)

    for ax in axes:
        ax.set_facecolor(theme["surface"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(theme["axis"])
        ax.tick_params(colors=theme["muted"], labelcolor=theme["ink_secondary"], length=3)
        ax.grid(True, color=theme["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
        ax.xaxis.label.set_color(theme["ink_secondary"])
        ax.yaxis.label.set_color(theme["ink_secondary"])
    return (fig, axes[0]) if ncols == 1 else (fig, axes)


def save_both_themes(draw: Callable[[dict], plt.Figure], path: Path) -> None:
    """Draw the chart once per theme and save name.png and name-dark.png."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        fig = draw(theme)
        out = path if name == "light" else path.with_name(path.stem + "-dark" + path.suffix)
        fig.savefig(out, facecolor=theme["surface"])
        plt.close(fig)
