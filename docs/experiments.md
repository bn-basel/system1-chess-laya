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

### 7. Richer targets: Stockfish scores more moves (2026-10-16)

**Idea.** A Lichess record scores 1 to 5 moves (26% only one). Every other legal move gets target 0, as if a decent
move were as bad as a blunder. Scoring more moves should teach Laya how good the other moves are.

**Change.** `scripts/enrich_trainset.py` had our own Stockfish score the top 10 moves of each of the first 300k
training positions (300k nodes, one thread per process, about 1.5 hours on the CPU, no download). Lichess's own,
deeper scores are kept. An extra move gets Lichess's best score minus Stockfish's gap to its own best, but never
more than Lichess's lowest listed move. Scored moves per position went from 3.5 to 9.4. The validation set is
unchanged. Found along the way: the database holds a few impossible or odd boards (72 in 300k, e.g. three white
knights or four white queens on the back rank) on which Stockfish crashes; those keep their Lichess scores only. Run
`exp7-rich-targets`, with `--fix-castling` and `--data data/train-rich`.

**Result.** Validation, against `base-castling-fix` (same positions, same order; only the targets differ):

| Positions | `val_top1` baseline | `val_top1` richer | `val_loss` baseline | `val_loss` richer |
|---|---:|---:|---:|---:|
| 100k | 21.2% | 21.6% | 2.886 | 2.891 |
| 200k | 22.7% | 22.8% | 2.813 | 2.847 |
| 250k | 23.2% | 23.0% | 2.804 | 2.827 |
| 300k | **24.5%** | **24.1%** | 2.795 | 2.821 |

- No gain on validation: -0.4 points at 300k, within noise. `val_loss` is a little worse. That's expected: the
  model now spreads probability over more good moves, while the validation answer key still lists only Lichess's
  few.
- Test sets (reported only): puzzles 36.5% (baseline 35.0%), games best move 24.2% (24.7%), win-chance lost 11.0%
  (11.6%). Slightly better on two of three, but within noise and not used to decide.
- Speed unchanged (15.1 ms per move).

**Decision.** Not a winner on validation. It may still help a longer run, where the extra information has more
time to matter, but that is untested.

### Long run with the castling fix (2026-10-16)

**What.** The recommended combined run from round 1: the same 1.5M-position run as `pieces-long`, with
`--fix-castling` as the only change. Run `long-castling-fix` (about 7 hours, stopped once at 1.18M and resumed from
the 1.1M save). Then the usual matches, 600 games per version.

**Training curve.** Validation `val_top1` ends at 26.1% (the old `pieces-long` ended at 24.6%, but on the old answer
key, so the two aren't strictly comparable).

**Test sets** (these use the same answer key for both):

| Model | Puzzles | Games best move | Win-chance lost | Time per move |
|---|---:|---:|---:|---:|
| `pieces-long` (old) | 35.8% | 24.6% | 10.9% | 14.9 ms |
| `long-castling-fix` | **37.0%** | **25.6%** | **10.6%** | 15.1 ms |

**Matches** (points out of 100 per opponent, draws in brackets):

| Opponent | Old | Fixed | Old + guard | Fixed + guard |
|---|---:|---:|---:|---:|
| Random moves | 87 (26) | 87.5 (25) | 99.5 (1) | 99.5 (1) |
| Grab material | 72.5 (55) | 72.5 (55) | 78 (44) | **86** (26) |
| Stockfish 1320 | 11.5 | **16.5** | 7 | **13** |
| Stockfish 1500 | 4.5 | 6.5 | 2.5 | 6 |
| Stockfish 1700 | 3 | 4 | 3.5 | 4.5 |
| Stockfish 2000 | 2.5 | 3.5 | 3.5 | 5.5 |
| **Elo** | 1006 (968-1044) | 1039 (996-1082) | 1051 (1019-1084) | **1121** (1088-1153) |

- With the guard, the fix is worth about **+70 Elo**, and the two ranges no longer overlap. It is the best Laya so
  far. (Re-fitting all games moved the older ratings by a point or two: 1053 is now 1051, 1008 is now 1006.)
- Without the guard, +33 Elo, within the noise of 600 games.
- It scores more against every Stockfish level, and with the guard it wins far more against grab material (26
  draws instead of 44).
- Same speed.

**Decision.** `long-castling-fix` is the new best model. Use `--fix-castling` for every run from now on.

### Found in round 1: castling never gets a target (2026-10-14)

Not an experiment, but it affects every run so far. The Lichess database writes castling as the king taking its own
rook (`e1h1`); python-chess writes `e1g1`. Training looks targets up by python-chess's form, so castling never
matched and always got target 0. In 5.7% of training positions castling was one of the scored moves; in 2.1% it was
Stockfish's best move. On the 2,000 validation positions it was the best move 39 times (2%), and there
`val_top1` counted the second-best move as the right answer.

- The test sets are not affected: they were labelled by our own Stockfish, which writes `e1g1`.
- All 300k comparisons stay fair: the baseline and every experiment use the same targets.
- But the trained models were taught that castling is never good, which costs real strength.
- Fix: `--fix-castling` in `scripts/train.py` (off by default, so earlier runs stay reproducible). It needs a new
  baseline, because `val_top1` changes on those 39 positions.

**New baseline with the fix (`base-castling-fix`, 2026-10-15).** The `board-pieces` command plus `--fix-castling`.
Experiments 3-6 are compared with this run.

| Positions | `val_top1` | `val_loss` |
|---|---:|---:|
| 100k | 21.2% | 2.886 |
| 200k | 22.7% | 2.813 |
| 250k | 23.2% | 2.804 |
| 300k | 24.5% | 2.795 |

- Not comparable with `board-pieces` on validation (the answer key changed), so this is not judged as a winner.
- What changed in play (test sets, reported only): the old model **never castled**, not once in the 316 test
  positions where castling was legal. The fixed model castles in 173 of them and finds 28 of the 38 game positions
  where castling is Stockfish's best move (old: 0). Overall: puzzles 35.0% (old 34.8%), games best move 24.7%
  (24.3%), win-chance lost 11.6% (11.5%), all within noise.
- Speed unchanged: 15.8 ms per move.
- Decision: use `--fix-castling` from now on.

Also measured: with Laya's default question budget (`head_max_len` 192), the move options of 3% of validation
positions are cut short, and in 0.25% two options become identical. A budget of 256 removes this for SAN.

### 6. Option D: Laya's encoder with a chess move output (2026-10-15)

**Idea.** Keep Laya's encoder but replace its question-answering head with a plain chess output: one score for
every starting-square/destination-square pair (64 x 64 = 4,096). The input is just the board, with no move
options, so it is much shorter. This is no longer Laya's question format. It is partly a reference point: how much
does Laya's own format help?

**Change.** A new option, `--format move_head`, and a new engine (`src/laya_chess/engines/move_head.py`). The
encoder's average token goes through one linear layer to the 4,096 scores. Illegal moves are masked out, and
pawns promote to a queen. Trained with the same soft targets (cross-entropy, no RLCD), the same learning rates
(the new layer starts from random), and `--fix-castling`. `val_top1` is move-level against the same best move as
the baseline. `val_loss` is cross-entropy over the legal moves, so it is roughly comparable. Run
`exp6-move-head`.

**Result.** Validation, against `base-castling-fix`:

| Positions | `val_top1` baseline | `val_top1` move output | `val_loss` baseline | `val_loss` move output |
|---|---:|---:|---:|---:|
| 100k | 21.2% | 12.5% | 2.886 | 3.047 |
| 200k | 22.7% | 14.9% | 2.813 | 2.990 |
| 250k | 23.2% | 15.2% | 2.804 | 2.974 |
| 300k | **24.5%** | **15.7%** | 2.795 | 2.966 |

- Far behind: -8.8 points at 300k. Still improving at the end, but slowly.
- Speed: the fastest so far. Training at 147 positions per second (2.3 times the baseline), 14.3 ms per move
  (baseline 15.8 ms).
- Test sets (reported only): puzzles 12.9% (baseline 35.0%), games best move 13.5% (24.7%), win-chance lost 16.8%
  (11.6%).
- What this says: with this little training, Laya's question format is worth a lot. Reading each move as an option
  lets the pretrained head compare moves directly. A fresh output layer has to learn all 4,096 move scores from
  scratch, from a single averaged summary of the board. A stronger move output (for example, scoring moves from
  the tokens of their squares) or far more training might close the gap, but that is a different project.

**Decision.** Not a winner. Keep Laya's question format.

### 5. Which piece, then which square, in one pass (2026-10-15)

**Idea.** Laya is weaker with more than about 20 options, and our positions average 31. Split "which move?" into
two smaller questions asked in the same forward pass: which piece moves (16 options at most, written like the piece
list, `Ng1`) and which square it moves to (`f3`).

**Change.** A new option, `--format from_to`. Each position becomes two Laya questions. The training target of
each is Stockfish's target summed per starting square and per destination square. To play, every legal move gets
log p(piece) + log p(square), and the best total wins, so the move is always legal. Pawns promote to a queen.
Validation scores the chosen move against the same best move as the baseline, so `val_top1` stays comparable. The
`val_loss` below is for that combined move spread, so it is only roughly comparable. Run `exp5-from-to`, with
`--fix-castling`, at 64 positions per update like the baseline (`--batch-size 32 --accumulate 4`, two rows per
position).

**Result.** Validation, against `base-castling-fix`:

| Positions | `val_top1` baseline | `val_top1` from/to | `val_loss` baseline | `val_loss` from/to |
|---|---:|---:|---:|---:|
| 100k | 21.2% | 18.1% | 2.886 | 3.095 |
| 200k | 22.7% | 18.7% | 2.813 | 3.055 |
| 250k | 23.2% | 19.4% | 2.804 | 3.088 |
| 300k | **24.5%** | **20.1%** | 2.795 | 3.059 |

- Clearly worse: -4.4 points at 300k, behind at every point.
- Test sets (reported only): puzzles fall from 35.0% to 19.0%, games best move 21.4% (24.7%), win-chance lost
  14.5% (11.6%).
- Why: the two questions are answered independently. The square question doesn't know which piece moves, so it
  picks squares that are good for *some* piece. The model often finds the right piece: in 288 of its wrong puzzle
  answers it moved the right piece to the wrong square (baseline: 115). But the pair doesn't fit. In one puzzle it
  played Qxf7+ instead of the mate Nf6#. It also plays far more captures (732 against 580 in 1,000 puzzles), since
  a square holding an enemy piece scores high for any attacker.
- Speed: 17.8 ms per move against 15.8 ms (two questions per move). Training ran at 48 positions per second.
- Fixing this would need the square question to be told the chosen piece, which means a second pass.

**Decision.** Not a winner. The independent split loses the link between piece and square.

### 4. Moves written with their starting square (2026-10-15)

**Idea.** The piece list says `Ng1` but the option says `Nf3`, so Laya has to work out which knight moves. Writing
the option as `Ng1-f3` (long algebraic notation, LAN) says it directly.

**Change.** A new notation, `--notation lan` (python-chess `board.lan(move)`: `Ng1-f3`, `Nf3xe5`, `e2-e4`). The
longer options don't fit Laya's default question budget: with `head_max_len` 192, 64% of validation positions had
options cut until two of them looked the same. Raised with two new options, `--head-max-len 384 --max-len 640`:
then 0.25% are cut and no board is cut. (The baseline's own SAN options are cut in 3% of positions; at 384 they
would not be, so a small part of any gain could come from the bigger budget rather than from LAN.) Run `exp4-lan`,
with `--fix-castling`.

**Result.** Validation, against `base-castling-fix` (`val_loss` is comparable: same moves, same targets):

| Positions | `val_top1` baseline | `val_top1` LAN | `val_loss` baseline | `val_loss` LAN |
|---|---:|---:|---:|---:|
| 100k | 21.2% | 21.0% | 2.886 | 2.873 |
| 200k | 22.7% | 22.9% | 2.813 | 2.808 |
| 250k | 23.2% | 24.5% | 2.804 | 2.783 |
| 300k | **24.5%** | **25.0%** | 2.795 | 2.778 |

- +0.5 points at 300k: within noise, short of the 2 points a winner needs. `val_loss` is slightly better from
  150k on, a weak sign that LAN helps a little.
- Speed: training is much slower, 35 positions per second against 64 (the inputs are about 35% longer, and the GPU
  was near full at 13.4 of 16 GB). Time per move is unchanged at 15.8 ms.
- Test sets (reported only): puzzles 34.2% (baseline 35.0%), games best move 24.1% (24.7%), win-chance lost 11.4%
  (11.6%).

**Decision.** Not a winner. It costs nearly twice the training time for at most a small gain. Keep SAN.

### 3. Piece list without the FEN (2026-10-15)

**Idea.** The piece list may make the FEN redundant. Dropping it shortens every input by about 35 tokens, so
training and play could be faster.

**Change.** A new board format, `--state pieces`: whose turn it is, castling rights (`KQkq`), the en-passant square,
and the two piece lists, but no FEN. Run `exp3-pieces-only`, with `--fix-castling`.

**Result.** Validation, against `base-castling-fix`:

| Positions | `val_top1` baseline | `val_top1` no FEN | `val_loss` baseline | `val_loss` no FEN |
|---|---:|---:|---:|---:|
| 100k | 21.2% | 19.2% | 2.886 | 2.943 |
| 200k | 22.7% | 22.1% | 2.813 | 2.838 |
| 250k | 23.2% | 22.4% | 2.804 | 2.822 |
| 300k | **24.5%** | **22.8%** | 2.795 | 2.818 |

- Worse at every point: -1.7 points at 300k, and `val_loss` is worse throughout.
- Speed: training is 17% faster (75 against 64 positions per second), but time per move barely changes (15.6 ms
  against 15.8 ms). At this input length a move's time is mostly fixed overhead, not the number of tokens.
- Test sets (reported only): puzzles 35.8% (baseline 35.0%), games best move 23.8% (24.7%), win-chance lost 12.1%
  (11.6%).
- So the FEN still helps Laya, even with the piece list next to it. Perhaps it reads the FEN's rank-by-rank layout
  as a second view of the board.

**Decision.** Not a winner. Keep `fen_pieces`.

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
