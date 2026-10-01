# system1-chess-laya

Can a small "System-1" decision model play chess, and how fast?

[Laya](https://huggingface.co/convaiinnovations/laya) is an open-weight model (421M parameters) that picks an
answer from a list of options in a single step, without thinking ahead. This project runs it locally, asks it to
choose chess moves, and measures how good and how fast those moves are compared with Stockfish. First as it comes,
then after training it on chess.

> **Status:** Phase 0 done (setup and first test). See the [roadmap](docs/roadmap.md).

## Quick start

You need [uv](https://docs.astral.sh/uv/). An NVIDIA GPU is recommended but not required.

```bash
uv sync
uv run python scripts/smoke_test.py
```

The first run downloads the Laya model (about 1.7 GB). Stockfish is optional for this test, see
[engines/README.md](engines/README.md).

## License

MIT, see [LICENSE](LICENSE). Laya itself is Apache-2.0, by Convai Innovations.
