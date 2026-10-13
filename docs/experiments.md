# Experiment log

One entry per experiment: the idea, what changed, the result, and the decision. Newest at the top.

## How experiments are judged

- **Decide on validation, confirm on the test sets.** Each idea is judged on 2,000 held-out validation positions
  (`val_top1`, and `val_loss` where comparable). Only a clear winner is then checked on
  the test sets and in matches. Checking the test sets after every experiment would eventually crown a change that
  only looks better by luck.
- **Same budget, same baseline.** Training ideas use a short run (300k positions) and are compared with a baseline
  trained the same way, at the same point in training.
- **Small differences are noise.** Rough noise levels: about ±1 point on validation top-1 (2,000 positions), about
  ±1.5 points on puzzles (1,000), about ±0.5 points on win-chance lost (2,000 game positions).

## Experiments

### Found in round 1: castling never gets a target (2026-10-14)

Not an experiment, but it affects every run so far. The Lichess database writes castling as the king taking its own
rook (`e1h1`); python-chess writes `e1g1`. Training looks targets up by python-chess's form, so castling never
matched and always got target 0. In 5.7% of training positions castling was one of the scored moves; in 2.1% it was
Stockfish's best move. On the 2,000 validation positions it was the best move 39 times (2%), and there
`val_top1` counted the second-best move as the right answer.

- The test sets are not affected: they were labelled by our own Stockfish, which writes `e1g1`.
- All 300k comparisons stay fair: the baseline and every experiment use the same targets.
- But the trained models were taught that castling is never good, which costs real strength.
- Fix: `--fix-castling` in `scripts/train.py` (off by default, so earlier runs stay reproducible). Not used in
  round 1. It needs a new baseline, because `val_top1` changes on those 39 positions.

Also measured: with Laya's default question budget (`head_max_len` 192), the move options of 3% of validation
positions are cut short, and in 0.25% two options become identical. A budget of 256 removes this for SAN.

### 2. Lower learning rate (2026-10-14)

**Idea.** In the long run (`pieces-long`), move quality got worse while the learning rate was high and only
improved as it fell. Maybe the official learning rate is too high for chess.

**Change.** The baseline command with `--lr-encoder 1e-5 --lr-head 4e-5` (official: 2.5e-5 and 1e-4). No code
changes. Run `exp2-low-lr`.

**Result.** Validation, against `board-pieces` (same data, same 300k positions):

| Positions | `val_top1` baseline | `val_top1` lower LR | `val_loss` baseline | `val_loss` lower LR |
|---|---:|---:|---:|---:|
| 100k | 21.3% | 20.2% | 2.832 | 2.868 |
| 200k | 22.4% | 22.6% | 2.770 | 2.778 |
| 250k | 23.5% | 22.9% | 2.746 | 2.767 |
| 300k | **23.7%** | **23.1%** | 2.744 | 2.759 |

- Slightly worse at 300k (-0.6 points, within noise) and slower early on. `val_loss` is a little worse throughout.
- Test sets (reported only, not used to decide): puzzles 35.2% (baseline 34.8%), games best move 23.7% (24.5%),
  win-chance lost 12.3% (11.5%).
- Same speed per move (same model and input).
- Caveat: this tests a 300k run, where the cosine schedule already brings the rate down. It can't fully rule out
  that a long run would gain from a lower peak.

**Decision.** Not a winner. Keep the official learning rates.

**Idea.** Laya wins material but can't convert: against the random mover, all 26 draws of the best model came with
Laya at least a rook ahead (19 threefold repetitions, 7 stalemates). It only sees the current position, so it can't
know a move repeats an earlier one.

**Change.** A rule applied when Laya chooses a move: if it leads by at least 3 pawns of material, play the most
likely move that neither repeats an earlier position nor stalemates the opponent. Otherwise play as before. No
retraining and no search: only the rules of chess and a piece count. Model: `pieces-long` (piece list, 1.5M).

**Result.** The same 600 games as the unguarded model (same opponents and openings):

| Opponent | Without guard | With guard |
|---|---:|---:|
| Random moves | 87 (26 draws, all a rook or more ahead) | **99.5** (1 draw) |
| Grab material | 72.5 (55 draws) | 78 (44 draws) |
| Stockfish 1320 | 11.5 | 7 |
| Stockfish 1500 / 1700 / 2000 | 4.5 / 3 / 2.5 | 2.5 / 3.5 / 3.5 |
| **Elo** | 1008 (977-1041) | **1053** (1017-1084) |

- It fixes conversion against weak opponents: from 26 thrown-away wins against the random mover to 1.
- Against grab material it helps less: 23 threefold repetitions remain. In 17 of them Laya led by less than 3 pawns,
  so the guard wasn't active. A repetition can also be completed by the opponent's move, which the guard can't
  prevent (grab material made the final repeating move in 10 of the 23).
- Against Stockfish it changes nothing beyond noise (±3 points per 100 games): Laya is rarely ahead there.
- So Laya's real limit against Stockfish is move quality, not finishing games. That is what the loop should work on.

**Decision.** Keep the guard for all games from now on, labelled `+guard`. It's a rules fix, not chess knowledge.
The loop works on the model itself and is judged without the guard.
