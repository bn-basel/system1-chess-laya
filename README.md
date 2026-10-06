# system1-chess-laya

Can a small "System-1" decision model play chess, and how fast?

[Laya](https://huggingface.co/convaiinnovations/laya) is an open-weight model (421M parameters) that picks an
answer from a list of options in a single step, without thinking ahead. This project runs it locally, asks it to
choose chess moves, and measures how good and how fast those moves are compared with Stockfish. First as it comes,
then after training it on chess.

> **Status:** Phase 3 in progress (training Laya on chess). Board-format experiment done; a long run with the winning
> format is next. See the [roadmap](docs/roadmap.md).

## First results

Tested on 1,000 Lichess puzzles and 2,000 positions from real games ([details](testsets/README.md)).

| Engine | Puzzles solved | Win-chance lost per move (games) | Time per move (median) |
|---|---:|---:|---:|
| Random moves | 4.9% | 21.3% | 26 µs |
| Stockfish, 1 node | 78.8% | 2.5% | 0.6 ms |
| Stockfish, 10,000 nodes | 98.9% | 0.7% | 6.6 ms |
| Laya, before any training | 4.7% | 21.1% | 19.4 ms (GPU) |
| Laya, after a 10-minute pilot (50k positions) | 32.6% | 16.2% | 17.3 ms (GPU) |
| Laya, after an overnight run (2.5M positions) | 35.3% | 12.9% | 15.3 ms (GPU) |
| **Laya with a piece list, 300k positions** | 34.8% | **11.5%** | 17.2 ms (GPU) |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/strength_vs_time-dark.png">
  <img alt="Puzzles solved vs time per move: Stockfish solves 79-100% of puzzles in 0.5-60 ms; Laya goes from 5% before training to 35% after training, at about 17 ms per move" src="figures/strength_vs_time.png">
</picture>

Out of the box, Laya plays no better than random. After training on 50,000 positions analysed by Stockfish, it already
beats a "grab material" player. Training overnight on 2.5 million positions improved it further, but slowly.

**How the board is shown matters more than how much data Laya sees.** Laya reads text, and a FEN string such as
`r1bqkb1r/pppp1ppp/2n2n2` reaches it as fragments like `rk`, `pp`, `1`, so it has to count its way to each piece.
Adding a list of each side's pieces, written like the moves (`Ke1 Qd1 Nf3`), let a 300k-position run beat the 2.5M
FEN run on move quality. An 8×8 grid of the same length made no difference.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/board_formats-dark.png">
  <img alt="Board-format experiment: with a piece list, win-chance lost per move falls to 11.5% after 300k positions, against 13.9% for FEN only and 14.1% for FEN plus an 8x8 grid; puzzles solved are similar for all three" src="figures/board_formats.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/learning_curve-dark.png">
  <img alt="Learning curve of the best model so far: puzzles solved rises from 4% to about 36% within the first 100k training positions, then stays level" src="figures/learning_curve.png">
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
