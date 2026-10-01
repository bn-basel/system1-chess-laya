import chess
import pytest

from laya_chess.eval.metrics import is_correct, win_prob, wp_loss

SCHOLARS_MATE = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"


def test_win_prob_is_even_at_zero_and_capped():
    assert win_prob(0) == pytest.approx(0.5)
    assert win_prob(100_000) == win_prob(1000)  # mates are capped like Lichess does
    assert win_prob(-300) == pytest.approx(1 - win_prob(300))


def test_wp_loss_is_zero_for_the_best_move():
    values = {"e2e4": 30, "g2g4": -80}
    assert wp_loss(values, "e2e4") == 0
    assert wp_loss(values, "g2g4") > 0


def test_wp_loss_ignores_mistakes_in_decided_positions():
    # Already +20 pawns: dropping to +15 changes almost nothing.
    assert wp_loss({"a": 2000, "b": 1500}, "b") == 0


def test_puzzle_solution_is_correct():
    puzzle = {"set": "puzzles", "fen": SCHOLARS_MATE, "solution": "h5f7", "themes": ["mateIn1"]}
    assert is_correct(puzzle, chess.Move.from_uci("h5f7"))
    assert not is_correct(puzzle, chess.Move.from_uci("e1f1"))


def test_any_mate_counts_in_mate_in_one_puzzles():
    # Two rooks against a boxed-in king: both Ra8# and Rb8# are mate.
    two_mates = "7k/6pp/8/8/8/8/8/RR4K1 w - - 0 1"
    puzzle = {"set": "puzzles", "fen": two_mates, "solution": "a1a8", "themes": ["mateIn1"]}
    assert is_correct(puzzle, chess.Move.from_uci("b1b8"))
    assert not is_correct(puzzle, chess.Move.from_uci("a1a7"))


def test_game_positions_are_graded_against_stockfish_best():
    position = {"set": "games", "fen": chess.STARTING_FEN, "best": "e2e4"}
    assert is_correct(position, chess.Move.from_uci("e2e4"))
    assert not is_correct(position, chess.Move.from_uci("d2d4"))
