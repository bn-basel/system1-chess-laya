"""Elo ratings from game results.

Every game gives each player an expected score from the rating difference (the standard Elo formula). The ratings
chosen are the ones under which the actual results are most likely (maximum likelihood), with the anchor players
fixed at known ratings, so the numbers land on their scale.

Each pair of players that met also gets one imaginary draw. Without it, a player who won or lost every game would
get an infinite rating; with it, the rating stays finite and the uncertainty range shows how little is known.
The range comes from bootstrapping: re-run the fit many times on games drawn at random from the real ones.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

PRIOR_DRAWS = 1.0
SCALE = 400 / math.log(10)  # converts between Elo points and the natural-log odds used in the maths
MAX_STEP = 200.0


def expected(r_a: float, r_b: float) -> float:
    """Expected score of a player rated r_a against one rated r_b."""
    difference = max(-4000.0, min(4000.0, r_b - r_a))  # beyond this the result is certain anyway
    return 1 / (1 + 10 ** (difference / 400))


def fit(games: list[tuple[str, str, float]], anchors: dict[str, float], iterations: int = 200) -> dict[str, float]:
    """games: (player A, player B, A's score: 1, 0.5 or 0). Returns a rating for every player."""
    pairs: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])  # (A, B) -> [A's points, games]
    for a, b, score in games:
        key, s = ((a, b), score) if a < b else ((b, a), 1 - score)
        pairs[key][0] += s
        pairs[key][1] += 1
    for stats in pairs.values():  # the imaginary draw
        stats[0] += 0.5 * PRIOR_DRAWS
        stats[1] += PRIOR_DRAWS

    players = {p for key in pairs for p in key}
    start = sum(anchors.values()) / len(anchors) if anchors else 1500.0
    ratings = {p: anchors.get(p, start) for p in players}
    for _ in range(iterations):
        for p in players - anchors.keys():
            # One Newton step per player: move the rating until expected points match actual points.
            points, expected_points, curvature = 0.0, 0.0, 0.0
            for (a, b), (s, n) in pairs.items():
                if p not in (a, b):
                    continue
                opponent, my_points = (b, s) if p == a else (a, n - s)
                e = expected(ratings[p], ratings[opponent])
                points += my_points
                expected_points += n * e
                curvature += n * e * (1 - e)
            step = SCALE * (points - expected_points) / max(curvature, 1e-9)
            ratings[p] += max(
                -MAX_STEP, min(MAX_STEP, step)
            )  # small steps: big ones overshoot when results are lopsided
    return ratings


def fit_with_ranges(
    games: list[tuple[str, str, float]], anchors: dict[str, float], resamples: int = 200, seed: int = 0
):
    """Ratings plus a 95% range for each player, from refitting on resampled games."""
    best = fit(games, anchors)
    rng = random.Random(seed)
    samples = defaultdict(list)
    for _ in range(resamples):
        resampled = [games[rng.randrange(len(games))] for _ in games]
        for player, rating in fit(resampled, anchors, iterations=60).items():
            samples[player].append(rating)
    ranges = {}
    for player, values in samples.items():
        values.sort()
        ranges[player] = (values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1])
    return best, ranges
