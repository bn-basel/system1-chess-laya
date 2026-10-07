"""Phase 4: play every Laya milestone against the same opponents, then compute Elo ratings.

Each pairing plays every opening twice, once with each colour. Results go to results/games.csv (one line per game,
committed); every move is saved to runs/games/ for the replay page; ratings go to results/elo.csv.
Already-played games are skipped, so if this stops, just run it again.

Usage:
    uv run python scripts/play_matches.py
    uv run python scripts/play_matches.py --laya laya-pieces-1.5M --opponents stockfish-1320 --games 4   # quick test
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import chess.engine

from laya_chess.data.testsets import load_jsonl
from laya_chess.engines import StockfishEngine
from laya_chess.engines.laya_engine import LayaEngine
from laya_chess.engines.simple import GreedyEngine, RandomEngine
from laya_chess.experiments import GAMES_PER_PAIRING, LAYA_MILESTONES, SIMPLE_OPPONENTS, STOCKFISH_ANCHORS
from laya_chess.play.elo import fit_with_ranges
from laya_chess.play.match import play_game

REPO_ROOT = Path(__file__).resolve().parents[1]
GAMES_CSV = REPO_ROOT / "results" / "games.csv"
ELO_CSV = REPO_ROOT / "results" / "elo.csv"
FIELDS = ["laya", "opponent", "opening", "laya_colour", "result", "laya_score", "plies", "termination",
          "laya_ms_median"]  # fmt: skip
STOCKFISH_MOVE_TIME = 0.05  # seconds; limited Stockfish picks its move at a shallow fixed depth, reached long before


def make_opponent(name: str):
    if name == "random":
        engine = RandomEngine(seed=0)
    elif name == "greedy":
        engine = GreedyEngine(seed=0)
    else:
        engine = StockfishEngine(limit=chess.engine.Limit(time=STOCKFISH_MOVE_TIME), elo=STOCKFISH_ANCHORS[name])
    engine.name = name
    return engine


def played_games() -> set[tuple[str, str, str, str]]:
    if not GAMES_CSV.exists():
        return set()
    with open(GAMES_CSV, encoding="utf-8") as f:
        return {(r["laya"], r["opponent"], r["opening"], r["laya_colour"]) for r in csv.DictReader(f)}


def append_game(row: dict) -> None:
    new_file = not GAMES_CSV.exists()
    with open(GAMES_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def compute_elo() -> None:
    with open(GAMES_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    games = [(r["laya"], r["opponent"], float(r["laya_score"])) for r in rows]
    ratings, ranges = fit_with_ranges(games, STOCKFISH_ANCHORS)
    counts = {p: sum(p in (a, b) for a, b, _ in games) for p in ratings}
    with open(ELO_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["player", "elo", "low_95", "high_95", "games", "anchor"])
        for player in sorted(ratings, key=ratings.get, reverse=True):
            low, high = ranges.get(player, (ratings[player], ratings[player]))
            anchored = player in STOCKFISH_ANCHORS
            writer.writerow([player, round(ratings[player]), round(low), round(high), counts[player], anchored])
            note = "(fixed)" if anchored else f"[{low:.0f} to {high:.0f}]"
            print(f"{player:20s} {ratings[player]:6.0f} {note:16s} {counts[player]:5d} games")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--laya", nargs="+", default=[name for name, _ in LAYA_MILESTONES])
    parser.add_argument("--opponents", nargs="+", default=SIMPLE_OPPONENTS + list(STOCKFISH_ANCHORS))
    parser.add_argument("--games", type=int, default=GAMES_PER_PAIRING, help="games per pairing (even number)")
    args = parser.parse_args()

    openings = load_jsonl(REPO_ROOT / "testsets" / "openings.jsonl")[: args.games // 2]
    done = played_games()
    games_dir = REPO_ROOT / "runs" / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = dict(LAYA_MILESTONES)

    for laya_name in args.laya:
        model = checkpoints[laya_name]
        laya = LayaEngine(
            model=str(REPO_ROOT / model) if model.startswith("checkpoints") else model,
            guard=laya_name.endswith("+guard"),
        )
        laya.warmup()
        laya.name = laya_name
        for opponent_name in args.opponents:
            opponent = make_opponent(opponent_name)
            t0, score, n = time.perf_counter(), 0.0, 0
            for opening in openings:
                for colour in ("white", "black"):
                    if (laya_name, opponent_name, opening["id"], colour) in done:
                        continue
                    white, black = (laya, opponent) if colour == "white" else (opponent, laya)
                    game = play_game(white, black, opening)
                    laya_ms = [m["latency_ms"] for m in game.moves if m["engine"] == laya_name]
                    row = {
                        "laya": laya_name, "opponent": opponent_name, "opening": opening["id"],
                        "laya_colour": colour, "result": game.result, "laya_score": game.score_for(laya_name),
                        "plies": game.moves[-1]["ply"] + 1 if game.moves else 0, "termination": game.termination,
                        "laya_ms_median": round(statistics.median(laya_ms), 2) if laya_ms else "",
                    }  # fmt: skip
                    append_game(row)
                    with open(games_dir / f"{laya_name}__{opponent_name}.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps({**row, "moves": game.moves, "pgn": game.pgn}) + "\n")
                    score += row["laya_score"]
                    n += 1
            opponent.close()
            if n:
                print(f"{laya_name} vs {opponent_name}: {score:g}/{n} ({time.perf_counter() - t0:.0f}s)", flush=True)
        del laya

    print("\nElo ratings (Stockfish levels fixed at their stated ratings):")
    compute_elo()


if __name__ == "__main__":
    main()
