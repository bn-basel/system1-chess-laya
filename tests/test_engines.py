import chess

from laya_chess.engines.laya_engine import encode_moves, encode_state
from laya_chess.engines.simple import GreedyEngine, RandomEngine

KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"


def test_random_engine_is_legal_and_repeatable():
    board = chess.Board(KIWIPETE)
    first = RandomEngine(seed=1).choose(board).move
    assert first in board.legal_moves
    assert RandomEngine(seed=1).choose(board).move == first


def test_greedy_engine_takes_the_queen():
    # The white rook on d1 can take the queen on d5 or the knight on a1.
    board = chess.Board("4k3/8/8/3q4/8/8/8/n2R2K1 w - - 0 1")
    assert GreedyEngine().choose(board).move == chess.Move.from_uci("d1d5")


def test_laya_options_are_exactly_the_legal_moves():
    board = chess.Board(KIWIPETE)
    for notation in ("san", "uci"):
        moves = encode_moves(board, notation)
        assert len(moves) == 48
        assert set(moves.values()) == set(board.legal_moves)


def test_laya_state_says_whose_turn_it_is():
    board = chess.Board()
    board.push_san("e4")
    assert encode_state(board)["side_to_move"] == "black"
