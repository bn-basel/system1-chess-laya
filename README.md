# system1-chess-laya

Can a small "System-1" decision model play chess, and how fast?

[Laya](https://huggingface.co/convaiinnovations/laya) is an open-weight model (421M parameters) that picks an
answer from a list of options in a single step, without thinking ahead. This project runs it locally, asks it to
choose chess moves, and measures how good and how fast those moves are compared with Stockfish. First as it comes,
then after training it on chess.

> **Status:** Phase 2 done (speed and hardware). See the [roadmap](docs/roadmap.md).

## First results

Tested on 1,000 Lichess puzzles and 2,000 positions from real games ([details](testsets/README.md)).

| Engine | Puzzles solved | Win-chance lost per move (games) | Time per move (median) |
|---|---:|---:|---:|
| Random moves | 4.9% | 21.3% | 26 µs |
| Stockfish, 1 node | 78.8% | 2.5% | 0.6 ms |
| Stockfish, 10,000 nodes | 98.9% | 0.7% | 6.6 ms |
| **Laya, before any training** | 4.7% | 21.1% | 19.4 ms (GPU) |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/strength_vs_time-dark.png">
  <img alt="Puzzles solved vs time per move: Stockfish solves 79-100% of puzzles in 0.5-60 ms; untrained Laya solves 5% in 19 ms, the same as random moves" src="figures/strength_vs_time.png">
</picture>

Out of the box, Laya plays no better than random. The next step is to find out how far chess training can take it.

**Hardware:** Laya needs about 2.3 GB of GPU memory. On an RTX 5080 it takes about 18 ms per move, or about
5 ms per position when positions are processed in batches. On a CPU (Ryzen 9 9900X) it takes about 280 ms per move.
Details in the [roadmap](docs/roadmap.md#what-phase-2-showed).

## Quick start

You need [uv](https://docs.astral.sh/uv/). An NVIDIA GPU is recommended but not required.

```bash
uv sync
uv run python scripts/smoke_test.py          # Laya picks moves on a few positions
uv run python scripts/evaluate.py laya       # full test: puzzles and game positions
uv run python scripts/results_table.py       # results so far
uv run python scripts/make_charts.py         # redraw the charts in figures/
```

The first run downloads the Laya model (about 1.7 GB). Stockfish is needed for Stockfish runs, see
[engines/README.md](engines/README.md).

## License

MIT, see [LICENSE](LICENSE). Laya itself is Apache-2.0, by Convai Innovations.
