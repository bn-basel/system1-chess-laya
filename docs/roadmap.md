# Roadmap

The working plan. Decisions are added at the bottom as we make them.

## Questions we want to answer

| # | Question | Phase |
|---|---|---|
| 1 | How well does Laya choose chess moves out of the box? | 2 |
| 2 | How much better does it get after training on chess? | 3 |
| 3 | How strong is it for the time it takes, compared with Stockfish and Leela (no search)? | 4 |
| 4 | Does it help to let Laya play when it's confident, and hand over to Stockfish when it isn't? | 5 |

## How we'll get there

**Log everything, draw later.** Every test and game saves a record for each move: the position, the move played,
how long it took, how confident the model was, what Stockfish would have played, and how much the move cost.
All charts, the Elo chart and the demo video are made from these records, so we can redraw anything and replay
any game at any speed.

**The Elo ladder.** We measure Laya's Elo at every milestone against the same fixed group of opponents:

```
random moves → Laya out of the box → Laya after training (several checkpoints) → Laya + Stockfish fallback
```

The opponents have to cover the whole range, from a random mover up to full-strength Stockfish, and they never
change. That way the numbers stay comparable from one milestone to the next.

## Phases

| Phase | What we do | What you get at the end |
|---|---|---|
| 0. Setup | Environment, Laya and Stockfish wrappers, first test | Laya picks a move and we see how long it took |
| 1. Test harness | Test positions, metrics, the per-move record format | A results table for random play and Stockfish |
| 2. Speed and hardware | Speed on GPU and CPU, batching, chart style (shortened, since Phase 1 already answered question 1) | First charts |
| 3. Training | Build a dataset, train, calibrate, evaluate | Trained checkpoints and the Elo ladder going up |
| 4. Strength vs time | Engine matches, question 3 | The main chart: Elo vs milliseconds per move |
| 5. Fallback | Question 4, try different confidence thresholds | Elo vs average time per move |
| 6. Presentation | README, charts, demo video | A finished repo |

Each phase adds its step to the Elo ladder and its games to the replay pool, so Phase 6 is mostly putting things
together.

## Decisions

- **2026-10-01:** Keep all four questions.
- **2026-10-01:** Ask Laya with one multiple-choice question: the position is the input, every legal move is an
  option ("Option A"). Revisit during Phase 3, once Laya knows some chess.
- **2026-10-01:** Visuals matter as much as the numbers: an Elo ladder across milestones and a replay video
  showing speed and Elo.
- **2026-10-01:** Write moves as `Nf3` (SAN), not `g1f3` (UCI). Laya gives each option only about 4 tokens, and
  UCI moves start to look identical to it. In one position, 43 legal moves became only 25 the model could tell
  apart. With SAN, 47 of 48 stayed distinct.
- **2026-10-01:** Test sets: 1,000 Lichess puzzles (125 per rating band) and 2,000 positions from real Lichess games.
  Committed to the repo, never used for training. See [testsets/README.md](../testsets/README.md).
- **2026-10-01:** Answer key: Stockfish 19 evaluates every legal move of every test position (300,000 nodes each).
  Grading any engine is then a lookup, which will make checking training checkpoints fast.
- **2026-10-01:** Phase 2 shortened to speed and hardware only, since Laya out of the box plays like random.
- **2026-10-01:** Only small summary tables go into git (`results/summary.csv`). Raw per-move records stay local in
  `runs/`.

## What Phase 0 showed

Tested on an RTX 5080, Laya 0.3.22, one position at a time.

- About **15–17 ms per move** on the GPU.
- Out of the box it plays badly (it missed a mate in 1), but not randomly. It has preferences, just not chess ones.
- Its confidence numbers aren't reliable yet. The model ships with a broken calibration setting for questions
  with more than 10 options, which is almost every chess position. We'll calibrate it ourselves before question 4.
- Its "act or escalate" signal was 1.0 everywhere, so it tells us nothing yet.

## What Phase 1 showed

| Engine | Puzzles solved | Best move (games) | Win-chance lost per move (games) | Time per move (median) |
|---|---:|---:|---:|---:|
| Random moves | 4.9% | 5.5% | 21.3% | 26 µs |
| Grab material | 22.4% | 17.9% | 20.2% | 34 µs |
| Stockfish, 1 node | 78.8% | 44.5% | 2.5% | 0.6 ms |
| Stockfish, 1,000 nodes | 91.6% | 53.0% | 1.4% | 1.2 ms |
| Stockfish, 10,000 nodes | 98.9% | 61.1% | 0.7% | 6.6 ms |
| Stockfish, 100,000 nodes | 99.7% | 69.0% | 0.3% | 59.2 ms |
| **Laya, out of the box** | 4.7% | 3.4% | 21.1% | 19.4 ms (GPU) |

- **Out of the box, Laya plays like a random mover.** It's no better than random on any measure, and slightly worse
  at finding the best move in game positions. Everything it will know about chess has to come from training.
- **Stockfish sets a high bar for speed.** Even at its smallest budget (one node, about 0.6 ms) it solves 79% of the
  puzzles. In the 19 ms Laya takes per move, Stockfish searches over 10,000 nodes and solves about 99%.
- "Best move" is a strict measure: many positions have several equally good moves, so even strong Stockfish settings
  only match the answer key about 69% of the time. Win-chance lost is the fairer measure of move quality.
- Times include Python's overhead for every engine (about 0.5 ms for talking to Stockfish). Stockfish runs on one CPU
  thread. Stockfish's memory is cleared between test positions outside the timed window.

## What Phase 2 showed

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/laya_speed-dark.png">
  <img alt="Laya time per position on an RTX 5080: 17.9 ms one at a time, about 5 ms in batches of 16 or more" src="../figures/laya_speed.png">
</picture>

- **One move at a time (what a game needs): about 18 ms on the GPU.** Batching 16 or more positions brings it down to
  about 5 ms per position (about 200 per second). Evaluating the full test sets then takes about 15 seconds.
- **GPU memory: 2.3 GB** one at a time, 5 GB for batches of 256. Any recent GPU with 4 GB or more is enough.
- **CPU: about 250–280 ms per position**, and batching doesn't help. Playing on a CPU works, but about 15× slower.
- **The model itself is the bottleneck**, not preparing its input: in a batch of 64, the forward pass takes 259 ms
  and preparing the input 56 ms. Training cost scales with this, so faster attention or compiled kernels are worth
  trying at the start of Phase 3.
- Laya's own batch function asks the same question about every input, but each chess position has its own legal
  moves. Batching therefore uses Laya's internal steps directly (`LayaEngine.choose_batch`). It picks the same move
  as one-at-a-time in 63 of 64 positions; the difference is rounding in the lower-precision maths.
- Chart style: one look for every chart, in a light and a dark version that follow the reader's GitHub theme.
  Stockfish is always blue, Laya always orange, simple baselines gray. The colors are checked for color-blind safety.

## Still open

- Demo video: Python (draw frames, join with ffmpeg) or a web replay page that we record. A web page could
  also run live on GitHub Pages. Doesn't block Phase 3; decide before the first games are recorded in Phase 4.
- Whether to also try "Option D" (Laya's encoder with a chess-specific output layer) as a comparison.
