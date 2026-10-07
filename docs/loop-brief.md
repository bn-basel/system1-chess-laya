# Improvement loop: round 1

Instructions for the session that runs the experiment loop. Read `CLAUDE.md` first: its rules apply throughout.

## Goal

Make the Laya model itself play better chess. Current best: `checkpoints/pieces-long/final` (piece-list board format,
1.5M training positions), about 1010 Elo, and 1053 with the finish-the-game guard. Its weakness is move quality, not
finishing games (see experiment 1 in `docs/experiments.md`).

## The baseline

Every experiment is a 300k-position training run compared with `board-pieces`, which used the current settings:

```bash
uv run python scripts/train.py --run board-pieces --positions 300000 --eval-every 50000 --state fen_pieces
```

Its learning curve is `results/training/board-pieces.csv`. At 300k positions: `val_top1` 0.2365, `val_loss` 2.744.

## How to judge an experiment

- **Decide on validation only.** The main measure is `val_top1`: how often Laya's top move is Stockfish's best move
  on 2,000 held-out validation positions (noise about ±1 point). Compare at matching points of the curve, above all
  at 300k. A change is a **winner** if it gains at least 2 points at 300k and isn't worse at 200k and 250k.
- `val_loss` is only comparable when the output is the same multiple-choice-over-moves format (experiments 2-3).
- The curve CSV also logs test-set columns (`puzzles_solved`, `games_*`). **Don't use them to choose**, only report
  them. Checking the test sets after every experiment would eventually crown a change that is better only by luck.
- Speed counts too: if a change makes Laya slower per move, measure it (`scripts/evaluate.py` reports time per move)
  and say so.

## The experiments (cheapest first)

Name runs `exp<N>-<short-name>` and always pass `--no-resume-state` (saves 4.7 GB per run; a crashed 1.5-hour run
is simply rerun). Each run takes about 1.5-2 hours.

**2. Lower learning rate.** In the long run, quality got worse while the learning rate was high and improved only as
it fell. Run the baseline command with `--lr-encoder 1e-5 --lr-head 4e-5`. No code changes.

**3. Piece list without the FEN.** The piece list may make the FEN redundant; dropping it shortens every input by
about 60 tokens (faster). Add a state format (e.g. `pieces`) with whose turn it is, castling rights, the en-passant
square and the two piece lists, but no FEN. Report time per move as well.

**4. Moves written with their starting square.** The piece list says `Ng1`, the option says `Nf3`; Laya has to work
out which knight. Add a notation (e.g. `lan`, using python-chess `board.lan(move)`, giving `Ng1-f3`, `Nf3xe5`).
First check that the longer options still fit Laya's input: Laya reports collapsed or truncated options in
`usage`. If more than 1% of validation positions overflow, try raising `head_max_len` and say so.

**5. From-square and to-square questions in one pass.** Laya is weaker with more than ~20 options; our positions
average 31. Ask two questions in the same forward pass: which square a piece moves from (16 options at most) and
which square it moves to. Training targets are the soft target summed per from-square and per to-square. To choose a
move, combine the two answers (e.g. log p(from) + log p(to)) over legal moves only; promote to a queen by default.
This needs new encoding, targets, loss (both questions) and decoding, plus a move-level `val_top1` so it stays
comparable. Keep it behind an option; the existing format must keep working.

**6. Option D: Laya's encoder with a chess move output.** Keep Laya's encoder, replace the decision head with one
linear layer over every from-square/to-square pair (64 x 64 = 4,096 outputs, illegal moves masked, queen promotion by
default). The input is the board state only (no move options), so it is much shorter. Train with the same soft
targets (cross-entropy). This is the biggest build: it needs its own model wrapper, training path, evaluation engine
and move-level `val_top1`. It is partly a reference point (no longer Laya's question format), so say so in the log.

Experiment 1 (the guard) is already in the log, hence the numbering from 2.

## Logging

After each experiment, add an entry to `docs/experiments.md` (newest at the top), in the same style as experiment 1:
Idea, Change, Result (a small table against the baseline at 100k/200k/300k: `val_top1`, and `val_loss` where
comparable, plus time per move if it changed), Decision. Keep the language plain. Don't edit `README.md` or
`docs/roadmap.md`: the orchestrator session updates those.

## Limits

- About 10 GPU hours for this round. One training run at a time.
- Before each run, check that at least 10 GB is free on C:.
- Run the existing tests after code changes (`uv run pytest -q`) and keep ruff clean. Add tests for new encoding or
  decoding code.
- Long runs: start training in the background and check on it; don't hold the session in one long blocking call.

## When to stop and report

Stop after experiments 2-6, or earlier if:

- a bug is found that could affect earlier results,
- a run fails twice,
- disk space runs low,
- something unexpected makes the plan look wrong.

Then write a short report for the owner: a table of all experiments against the baseline, which changes are winners,
and a recommendation for one combined longer run. **Don't start the combined run, matches, or anything outside this
brief without the owner's go-ahead.** End with the git commands for the owner to run (no co-author lines).
