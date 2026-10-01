"""Print results/summary.csv as a Markdown table (latest run of each engine), ready to paste into the README.

Usage:
    uv run python scripts/results_table.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

SUMMARY = Path(__file__).resolve().parents[1] / "results" / "summary.csv"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # the table uses "µs"
    with open(SUMMARY, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Later runs of the same engine replace earlier ones.
    latest: dict[tuple[str, str], dict] = {}
    for row in rows:
        latest[(row["engine"], row["set"])] = row
    engines = list(dict.fromkeys(engine for engine, _ in latest))

    print("| Engine | Puzzles solved | Best move (games) | Win-chance lost per move (games) | Time per move (median) |")
    print("|---|---:|---:|---:|---:|")
    for engine in engines:
        puzzles, games = latest.get((engine, "puzzles")), latest.get((engine, "games"))
        solved = f"{float(puzzles['correct']):.1%}" if puzzles else "-"
        best = f"{float(games['correct']):.1%}" if games else "-"
        loss = f"{float(games['wp_loss']):.1%}" if games else "-"
        ms = float((games or puzzles)["latency_p50_ms"])
        time = f"{ms * 1000:.0f} µs" if ms < 1 else f"{ms:.1f} ms"
        print(f"| `{engine}` | {solved} | {best} | {loss} | {time} |")


if __name__ == "__main__":
    main()
