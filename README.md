# system1-chess-laya

Can a small "System-1" decision model play chess, and how fast?

[Laya](https://huggingface.co/convaiinnovations/laya) is an open-weight model (421M parameters) that picks an
answer from a list of options in a single step, without thinking ahead. This project runs it locally, asks it to
choose chess moves, and measures how good and how fast those moves are compared with Stockfish. First as it comes,
then after training it on chess.

> **Status:** Phase 3 in progress (training Laya on chess). First overnight run done; testing board formats next. See the [roadmap](docs/roadmap.md).

## First results

Tested on 1,000 Lichess puzzles and 2,000 positions from real games ([details](testsets/README.md)).

| Engine | Puzzles solved | Win-chance lost per move (games) | Time per move (median) |
|---|---:|---:|---:|
| Random moves | 4.9% | 21.3% | 26 µs |
| Stockfish, 1 node | 78.8% | 2.5% | 0.6 ms |
| Stockfish, 10,000 nodes | 98.9% | 0.7% | 6.6 ms |
| Laya, before any training | 4.7% | 21.1% | 19.4 ms (GPU) |
| Laya, after a 10-minute pilot (50k positions) | 32.6% | 16.2% | 17.3 ms (GPU) |
| **Laya, after an overnight run (2.5M positions)** | 35.3% | 12.9% | 15.3 ms (GPU) |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/strength_vs_time-dark.png">
  <img alt="Puzzles solved vs time per move: Stockfish solves 79-100% of puzzles in 0.5-60 ms; Laya goes from 5% before training to 35% after training on 2.5M positions, at about 15 ms per move" src="figures/strength_vs_time.png">
</picture>

Out of the box, Laya plays no better than random. After training on 50,000 positions analysed by Stockfish, it already
beats a "grab material" player. Training overnight on 2.5 million positions improved it further, but slowly: most of
the gain comes in the first few hundred thousand positions, then the curve flattens. Next we test whether showing
Laya the board in a clearer form helps more than extra data.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/learning_curve-dark.png">
  <img alt="Learning curve: puzzles solved rises from 5% to about 31% in the first 300k training positions, then levels off around 35% up to 2.5M positions" src="figures/learning_curve.png">
</picture>

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

To train Laya yourself, see [docs/training.md](docs/training.md).

The first run downloads the Laya model (about 1.7 GB). Stockfish is needed for Stockfish runs, see
[engines/README.md](engines/README.md).

## License

MIT, see [LICENSE](LICENSE). Laya itself is Apache-2.0, by Convai Innovations.
