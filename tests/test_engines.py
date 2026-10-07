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


def test_piece_list_uses_move_style_squares():
    from laya_chess.engines.laya_engine import piece_list

    board = chess.Board()
    assert piece_list(board, chess.WHITE).startswith("Ke1 Qd1 Ra1 Rh1 Bc1 Bf1 Nb1 Ng1 Pa2")
    assert len(piece_list(board, chess.BLACK).split()) == 16


def test_every_state_format_keeps_the_fen():
    from laya_chess.engines.laya_engine import STATE_FORMATS

    for state_format in STATE_FORMATS:
        assert encode_state(chess.Board(), state_format)["fen"] == chess.STARTING_FEN


def test_guard_avoids_stalemate_when_ahead():
    from laya_chess.engines.laya_engine import guarded_choice

    # White (king b6, queen c1) is far ahead; Qc7 would stalemate the black king on a8.
    board = chess.Board("k7/8/1K6/8/8/8/8/2Q5 w - - 0 1")
    moves = {board.san(m): m for m in board.legal_moves}
    probabilities = dict.fromkeys(moves, 0.01) | {"Qc7": 0.9, "Qc8#": 0.05}
    assert guarded_choice(board, probabilities, moves) == "Qc8#"


def test_guard_does_nothing_when_not_ahead():
    from laya_chess.engines.laya_engine import guarded_choice

    board = chess.Board()
    moves = {board.san(m): m for m in board.legal_moves}
    probabilities = dict.fromkeys(moves, 0.01) | {"e4": 0.5}
    assert guarded_choice(board, probabilities, moves) == "e4"
