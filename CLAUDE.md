# system1-chess-laya

Can Laya, a small open-weight "System-1" decision model (421M parameters, picks one option from a list in a single
forward pass), play chess, and how fast? We fine-tune it on Stockfish-labelled positions, measure move quality on fixed
test sets, and measure strength in games (Elo). The owner is learning from this project: explain important choices,
recommend one option, and let them decide.

## Rules (always)

- **Never run git commands that change anything** (no add, commit, push, branch, reset, config). The owner commits
  everything personally. Read-only git (status, log, diff) is fine. At the end of a piece of work, give the owner the
  commands to run, with a short, plain commit message and no co-author lines.
- **Never permanently delete files.** Move them to the Recycle Bin (PowerShell:
  `[Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory(path, 'OnlyErrorDialogs', 'SendToRecycleBin')`).
- **One GPU job at a time.** Training uses 9-11 GB of the 16 GB GPU. Two jobs overflow into system RAM and crawl.
- **Check disk space before training** (`df -h /c`). Keep at least 10 GB free.
- **No downloads over ~100 MB without asking**, stating size and source.
- **Don't change the test sets** (`testsets/`) and never train on them.
- Reports: short and plain. The owner dictates by voice, so expect transcription slips.

## Layout

| Path | What |
|---|---|
| `src/laya_chess/engines/` | Engines with one interface (`choose(board)`): Laya, Stockfish, random, greedy |
| `src/laya_chess/train.py` | Training pieces: soft targets, Laya's loss (cross-entropy + RLCD), calibration, checkpoints |
| `src/laya_chess/eval/` | Metrics (puzzle solved, best move, win-chance lost) and the evaluation runner |
| `src/laya_chess/play/` | Games between engines, and the Elo fit |
| `src/laya_chess/experiments.py` | Run lists shared by scripts and charts |
| `scripts/` | Command-line entry points (train, evaluate, play_matches, make_charts, ...) |
| `testsets/` | Fixed exam: 1,000 puzzles, 2,000 game positions, 50 openings (committed, never trained on) |
| `results/` | Small committed results: `summary.csv`, `training/<run>.csv`, `games.csv`, `elo.csv` |
| `figures/` | Charts (light and dark versions), drawn by `scripts/make_charts.py` |
| `docs/` | `roadmap.md` (plan, decisions, findings), `training.md`, `experiments.md` (experiment log) |
| `data/`, `checkpoints/`, `runs/` | Large local files, not in git |

## Commands

```bash
uv run pytest -q                                   # tests
uvx ruff format src scripts tests && uvx ruff check src scripts tests
uv run python scripts/train.py --run NAME --positions 300000 --eval-every 50000 --state fen_pieces --no-resume-state
uv run python scripts/evaluate.py laya --model checkpoints/NAME/final
uv run python scripts/make_charts.py
```

## Conventions

- Python 3.12 via uv. Line length 120 (ruff). Match the existing style: short docstrings that say why, few comments.
- New behaviour goes behind a flag or option, so earlier runs stay reproducible. Add a test for anything that grades
  or encodes positions.
- Docs are written in plain, simple English for a reader learning the subject.
