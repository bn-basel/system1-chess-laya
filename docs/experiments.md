# Experiment log

One entry per experiment: the idea, what changed, the result, and the decision. Newest at the top.

## How experiments are judged

- **Decide on validation, confirm on the test sets.** Each idea is judged on the 5,000 held-out validation positions
  (`val_top1`, `val_loss`) and, where it applies, on win-chance lost per move. Only a clear winner is then checked on
  the test sets and in matches. Checking the test sets after every experiment would eventually crown a change that
  only looks better by luck.
- **Same budget, same baseline.** Training ideas use a short run (300k positions) and are compared with a baseline
  trained the same way, at the same point in training.
- **Small differences are noise.** Rough noise levels: about ±1 point on validation top-1 (5,000 positions), about
  ±1.5 points on puzzles (1,000), about ±0.5 points on win-chance lost (2,000 game positions).

## Experiments

### 1. Finish-the-game guard (2026-10-07)

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
