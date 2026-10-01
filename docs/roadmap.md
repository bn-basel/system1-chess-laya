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
| 2. Out of the box | Question 1, speed on GPU and CPU, accuracy vs number of legal moves | First charts and the first step of the Elo ladder |
| 3. Training | Build a dataset, train, calibrate, evaluate | Trained checkpoints and the Elo ladder going up |
| 4. Strength vs time | Engine matches, question 3 | The main chart: Elo vs milliseconds per move |
| 5. Fallback | Question 4, try different confidence thresholds | Elo vs average time per move |
| 6. Presentation | README, charts, demo video | A finished repo |

Each phase adds its step to the Elo ladder and its games to the replay pool, so Phase 6 is mostly putting things
together.

## Decisions

- **2026-10-01:** Keep all four questions.
- **2026-10-01:** Ask Laya with one multiple-choice question: the position is the input, every legal move is an
  option ("Option A"). Revisit after Phase 2.
- **2026-10-01:** Visuals matter as much as the numbers: an Elo ladder across milestones and a replay video
  showing speed and Elo.
- **2026-10-01:** Write moves as `Nf3` (SAN), not `g1f3` (UCI). Laya gives each option only about 4 tokens, and
  UCI moves start to look identical to it. In one position, 43 legal moves became only 25 the model could tell
  apart. With SAN, 47 of 48 stayed distinct.

## What Phase 0 showed

Tested on an RTX 5080, Laya 0.3.22, one position at a time.

- About **15–17 ms per move** on the GPU.
- Out of the box it plays badly (it missed a mate in 1), but not randomly. It has preferences, just not chess ones.
- Its confidence numbers aren't reliable yet. The model ships with a broken calibration setting for questions
  with more than 10 options, which is almost every chess position. We'll calibrate it ourselves before question 4.
- Its "act or escalate" signal was 1.0 everywhere, so it tells us nothing yet.

## Still open

- Demo video: Python (draw frames, join with ffmpeg) or a web replay page that we record. A web page could
  also run live on GitHub Pages. Decide by Phase 2.
- Whether to also try "Option D" (Laya's encoder with a chess-specific output layer) as a comparison.
