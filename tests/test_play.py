import pytest

from laya_chess.engines.simple import GreedyEngine, RandomEngine
from laya_chess.play.elo import expected, fit
from laya_chess.play.match import play_game


def test_expected_score_is_even_for_equal_ratings_and_about_three_quarters_at_plus_200():
    assert expected(1500, 1500) == pytest.approx(0.5)
    assert expected(1700, 1500) == pytest.approx(0.76, abs=0.01)


def test_fit_recovers_a_200_point_gap():
    # A scores 76 out of 100 against an anchor rated 1500 -> about 1700 (the imaginary draw pulls it in slightly).
    games = [("A", "anchor", 1.0)] * 76 + [("A", "anchor", 0.0)] * 24
    assert fit(games, {"anchor": 1500})["A"] == pytest.approx(1700, abs=15)


def test_fit_stays_finite_when_one_player_wins_everything():
    games = [("A", "anchor", 1.0)] * 50
    assert 1500 < fit(games, {"anchor": 1500})["A"] < 3000


def test_a_game_finishes_with_a_valid_result_and_records_every_move():
    opening = {"id": "test", "moves": ["e2e4", "e7e5"]}
    game = play_game(GreedyEngine(seed=1), RandomEngine(seed=2), opening)
    assert game.result in ("1-0", "0-1", "1/2-1/2")
    assert len(game.moves) >= 1
    assert game.score_for(game.white) + game.score_for(game.black) == 1.0
    assert game.pgn.startswith("[Event")


def test_fit_handles_very_lopsided_results_without_overflow():
    games = [("A", "anchor", 0.0)] * 200 + [("B", "A", 1.0)] * 200
    ratings = fit(games, {"anchor": 1500})
    assert ratings["A"] < 1500 < ratings["A"] + 2000
    assert ratings["B"] > ratings["A"]
