# Training Laya on chess

## What training does

Each training example is a chess position, asked as the same multiple-choice question Laya answers when it plays:
the position is the input and every legal move is an option. Training and play use the exact same encoding
(`encode_position` in `src/laya_chess/engines/laya_engine.py`), so the model is trained on precisely what it sees later.

**Targets.** The answer key comes from the [Lichess evaluation database](https://database.lichess.org/#evals): Stockfish
analyses with one or more evaluated moves per position. The target is a probability spread over those moves, based on
how much winning chance each one keeps:

| Move | Stockfish | Winning chance | Target |
|---|---:|---:|---:|
| Best move | +0.50 | 55% | 0.63 |
| Second best | +0.20 | 52% | 0.37 |
| Third best | -3.00 | 25% | 0.00 |

A move with 5 percentage points less winning chance than the best gets about a third of its weight. Moves Stockfish
didn't evaluate get 0. With only one evaluated move, the target is simply that move.

**Loss.** Laya's official recipe, ported from its fine-tuning notebook:

- *Soft-target cross-entropy* pulls Laya's predicted spread towards the target.
- *RLCD* tries a few noisy variations of the prediction, scores each one with a proper scoring rule (a score that's
  best only when the stated probabilities are honest), and nudges the model towards the better ones. This is what
  makes Laya's confidence meaningful.

**Calibration.** After training, Laya's confidence temperatures are re-fitted on 5,000 held-out positions, so that
"80% confident" means right about 80% of the time.

## Running it

```bash
uv run python scripts/build_trainset.py                    # stream 6M positions (~30 min, ~800 MB on disk)
uv run python scripts/train.py --run pilot --positions 50000 --eval-every 10000
uv run python scripts/overnight.py --run overnight1 --positions 2500000   # train + test + charts, ~8 h
uv run python scripts/board_experiment.py                  # compare three board formats, ~4 h
uv run python scripts/overnight.py --run pieces-long --positions 1500000 --state fen_pieces   # best format, ~7-8 h
uv run python scripts/evaluate.py laya --model checkpoints/pilot/final
uv run python scripts/make_charts.py
```

During training, the model is scored on all test positions every `--eval-every` positions. Each score goes into
`results/training/<run>.csv` (the learning curve) and a resumable checkpoint is saved to `checkpoints/<run>/latest`.
If a run stops, add `--resume` to the same command. The final, calibrated model is saved to `checkpoints/<run>/final`.

## Board formats

`--state` sets how the position is shown to Laya. Every format includes the FEN and whose turn it is.

| `--state` | Adds | Result (300k positions) |
|---|---|---|
| `fen` | nothing | baseline |
| `fen_ascii` | an 8×8 grid | no better than `fen` |
| `fen_pieces` | each side's pieces with their squares (`Ke1 Qd1 Nf3`) | clearly better; used from now on |

A trained checkpoint remembers its format, so `evaluate.py --model checkpoints/<run>/final` uses the right one.

## Settings, and why

| Setting | Value | Why |
|---|---|---|
| Batch | 32 positions, 2 batches per update | Without gradient checkpointing, batches of 24+ overflow 16 GB of GPU memory, and Windows silently falls back to system RAM, which is many times slower. With checkpointing, peak memory is 9 GB. |
| Learning rate | 2.5e-5 encoder, 1e-4 decision head | Laya's official values. |
| Schedule | 1% warm-up, then cosine decay | Warm-up avoids a jolt to the pretrained weights at the start of a long run. |
| RLCD noise | 0.4, falling to 0.1 | Laya's official values. |
| Precision | bf16 | Supported natively by the RTX 5080, and needs no loss scaling. |

On an RTX 5080 training runs at about 90 positions per second with `fen` (about 330k per hour) and about 55-65 with
`fen_pieces`, whose inputs are longer.
Disk: each run keeps a resumable checkpoint (5.5 GB) and the final model (0.8 GB).
