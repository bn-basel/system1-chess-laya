"""The fixed test sets: Lichess puzzles and positions from real Lichess games.

Each test position is one JSON line with:
  id, set ("puzzles" or "games"), fen, n_legal
Puzzles also have: rating, themes, solution (the move that solves it, in UCI)
Games also have:   ply, white_elo, black_elo
Labelling (see label.py) adds:
  values: {move in UCI: centipawns for the side to move after playing it}
  best:   the move with the highest value

The sources change over time (Lichess adds puzzles and updates ratings), so the sampled sets are committed
to the repo. Rebuilding them later would give a different exam.
"""

from __future__ import annotations

import csv
import io
import json
import random
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import chess
import chess.pgn
import zstandard
from tqdm import tqdm

PUZZLES_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
GAMES_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst"

# Puzzles are spread evenly over these rating bands, so every difficulty level is measured equally well.
RATING_BANDS = [(0, 1000), (1000, 1250), (1250, 1500), (1500, 1750), (1750, 2000), (2000, 2250), (2250, 2500),
                (2500, 9999)]  # fmt: skip

# Only use puzzles whose rating is well established.
MAX_RATING_DEVIATION = 90
MIN_PLAYS = 500
MIN_POPULARITY = 80

# Game positions: skip the first few moves (mostly the same openings) and take one position per game.
MIN_PLY = 10


def download(url: str, dest: Path) -> Path:
    """Download a file once, with a progress bar. Does nothing if it already exists."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            while chunk := response.read(1 << 20):
                out.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)
    return dest


def open_zst_text(path: Path) -> io.TextIOWrapper:
    """Read a .zst file as text, decompressing on the fly (no need to unpack it on disk)."""
    # Closing the returned text stream also closes the decompressor and the file underneath it.
    reader = zstandard.ZstdDecompressor().stream_reader(open(path, "rb"))  # noqa: SIM115
    return io.TextIOWrapper(reader, encoding="utf-8")


def rating_band(rating: int) -> int | None:
    for i, (lo, hi) in enumerate(RATING_BANDS):
        if lo <= rating < hi:
            return i
    return None


def sample_puzzles(path: Path, n: int, seed: int) -> list[dict]:
    """Pick n puzzles, the same number from each rating band, using reservoir sampling."""
    per_band = n // len(RATING_BANDS)
    rng = random.Random(seed)
    reservoirs: list[list[dict]] = [[] for _ in RATING_BANDS]
    seen = [0] * len(RATING_BANDS)

    with open_zst_text(path) as f:
        for row in tqdm(csv.DictReader(f), desc="puzzles", unit=" rows"):
            if (
                int(row["RatingDeviation"]) > MAX_RATING_DEVIATION
                or int(row["NbPlays"]) < MIN_PLAYS
                or int(row["Popularity"]) < MIN_POPULARITY
            ):
                continue
            band = rating_band(int(row["Rating"]))
            if band is None:
                continue
            # Reservoir sampling: every eligible puzzle has the same chance of ending up in the sample.
            seen[band] += 1
            if len(reservoirs[band]) < per_band:
                reservoirs[band].append(row)
            elif (j := rng.randrange(seen[band])) < per_band:
                reservoirs[band][j] = row

    puzzles = []
    for row in (row for band in reservoirs for row in band):
        # Lichess stores the position *before* the opponent's last move. Play it, then the solver is to move.
        moves = row["Moves"].split()
        board = chess.Board(row["FEN"])
        board.push_uci(moves[0])
        puzzles.append(
            {
                "id": row["PuzzleId"],
                "set": "puzzles",
                "fen": board.fen(),
                "n_legal": board.legal_moves.count(),
                "rating": int(row["Rating"]),
                "themes": row["Themes"].split(),
                "solution": moves[1],
            }
        )
    return puzzles


def iter_games(path: Path) -> Iterator[str]:
    """Yield each game's PGN text without parsing it (fast)."""
    lines: list[str] = []
    with open_zst_text(path) as f:
        for line in f:
            if line.startswith("[Event ") and lines:
                yield "".join(lines)
                lines = []
            lines.append(line)
    if lines:
        yield "".join(lines)


def _elo(value: str | None) -> int | None:
    """Player rating from a PGN header. Some games have "?" instead of a number."""
    return int(value) if value and value.isdigit() else None


def sample_game_positions(path: Path, n: int, seed: int) -> list[dict]:
    """Pick n positions, one from each of n randomly chosen games."""
    rng = random.Random(seed)
    # Pass 1: choose which games to use. Take extra, since some games are too short.
    n_games = sum(1 for _ in tqdm(iter_games(path), desc="counting games", unit=" games"))
    chosen = set(rng.sample(range(n_games), min(n_games, n * 2)))

    positions: list[dict] = []
    seen_fens: set[str] = set()
    for i, pgn in enumerate(tqdm(iter_games(path), total=n_games, desc="games", unit=" games")):
        if i not in chosen:
            continue
        game = chess.pgn.read_game(io.StringIO(pgn))
        moves = list(game.mainline_moves()) if game else []
        if len(moves) <= MIN_PLY:
            continue
        ply = rng.randrange(MIN_PLY, len(moves))
        board = game.board()
        for move in moves[:ply]:
            board.push(move)
        # Skip positions with only one legal move (nothing to choose) and repeats.
        if board.legal_moves.count() < 2 or board.epd() in seen_fens:
            continue
        seen_fens.add(board.epd())
        positions.append(
            {
                "id": game.headers.get("Site", "").rsplit("/", 1)[-1] + f"-{ply}",
                "set": "games",
                "fen": board.fen(),
                "n_legal": board.legal_moves.count(),
                "ply": ply,
                "white_elo": _elo(game.headers.get("WhiteElo")),
                "black_elo": _elo(game.headers.get("BlackElo")),
            }
        )
        if len(positions) == n:
            break
    return positions


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(record) + "\n" for record in records)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
