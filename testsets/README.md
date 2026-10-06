# Test sets

The fixed exam every engine is graded on. These files are committed on purpose: the Lichess data they come from
keeps changing, so rebuilding them later would give a different exam. `meta.json` records exactly how they were made.

| File | Positions | Source |
|---|---|---|
| `puzzles.jsonl` | 1,000 | [Lichess puzzle database](https://database.lichess.org/#puzzles), 125 puzzles from each of 8 rating bands (under 1000 up to 2500+) |
| `games.jsonl` | 2,000 | [Lichess rated games, January 2013](https://database.lichess.org/#standard_games), one random position per game, after move 5 |

Both sources are released by Lichess under CC0.

**Puzzles** only use puzzles with a well-established rating (rating deviation at most 90, at least 500 plays,
popularity at least 80). A puzzle counts as solved if the engine finds the first move of the solution. In
mate-in-1 puzzles any checkmate counts, as on Lichess.

**Game positions** are graded against Stockfish's best move. Positions with only one legal move are skipped.

## Openings

`openings.jsonl` holds 50 starting positions for engine matches: the position after the first 4 moves of real rated
games from the same January 2013 Lichess file (both players rated 1600+), kept only if Stockfish 19 rates it within
half a pawn of equal. Each is played twice in a match, once with each colour. Rebuild with
`uv run python scripts/build_openings.py`.

## Labels

Every legal move in every position has a Stockfish value (`values`, in centipawns for the side to move after
that move), and `best` is the move with the highest value. Each move was evaluated by Stockfish 19 with a fixed
budget of 300,000 nodes, on one thread, with the hash table cleared every time, so the labels are repeatable.

Because the labels come from Stockfish, Stockfish with a large budget will score close to 100% on the game
positions almost by definition. The puzzles are the more independent check.

## Rebuilding

```bash
uv run python scripts/build_testsets.py
uv run python scripts/label_testsets.py
```

These positions must never be used for training.
