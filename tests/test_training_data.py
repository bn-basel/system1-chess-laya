import pytest

from laya_chess.data.evaldb import to_record
from laya_chess.train import soft_target


def test_soft_target_sums_to_one_and_favours_the_best_move():
    target = soft_target([50, 20, -300])
    assert sum(target) == pytest.approx(1.0)
    assert target[0] > target[1] > target[2]


def test_soft_target_with_one_move_is_one_hot():
    assert soft_target([120]) == [1.0]


def test_equal_moves_share_the_target():
    assert soft_target([0, 0]) == pytest.approx([0.5, 0.5])


def test_eval_database_scores_are_flipped_for_black():
    # Black to move; the database says White is better by 69 after Black's best move.
    raw = {
        "fen": "7r/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 b - -",
        "evals": [{"pvs": [{"cp": 69, "line": "f7g7 e6e2"}, {"cp": 163, "line": "h8d8 d1e1"}], "depth": 46}],
    }
    record = to_record(raw)
    assert record["moves"] == ["f7g7", "h8d8"]
    assert record["cp"] == [-69, -163]  # from Black's point of view: the first move is the better one


def test_eval_database_mates_become_large_scores():
    raw = {"fen": "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - -", "evals": [{"pvs": [{"mate": 1, "line": "a1a8"}], "depth": 30}]}
    assert to_record(raw)["cp"] == [100_000]
