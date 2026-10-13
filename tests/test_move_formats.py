"""The from/to questions (experiment 5), the move-output model's encoding (experiment 6), and their targets."""

import chess
import pytest
import torch

from laya_chess.engines.laya_engine import build_from_to_questions, combine_from_to, playable_moves
from laya_chess.engines.move_head import N_MOVES, collate_boards, move_head_loss, move_index
from laya_chess.train import move_level_scores, move_level_targets, move_weights, soft_target

KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
PROMOTION = "8/1P4k1/8/8/8/8/6K1/8 w - - 0 1"  # White's b-pawn promotes on b8


def test_playable_moves_promote_to_a_queen_only():
    board = chess.Board(PROMOTION)
    promotions = [m for m in playable_moves(board) if m.promotion]
    assert promotions == [chess.Move.from_uci("b7b8q")]
    assert len(playable_moves(board)) == len(list(board.legal_moves)) - 3


def test_from_to_questions_cover_every_playable_move():
    board = chess.Board(KIWIPETE)
    questions, squares = build_from_to_questions(board)
    assert "Ne5" in questions["from"]["criteria"] and "Pa2" in questions["from"]["criteria"]
    assert len(questions["from"]["criteria"]) <= 16
    assert len(questions["to"]["criteria"]) == len(set(squares["to"]))
    pairs = {(f, t) for f in squares["from"] for t in squares["to"]}
    assert all((m.from_square, m.to_square) in pairs for m in board.legal_moves)
    assert "white" in questions["from"]["instructions"]


def test_combining_the_answers_only_scores_legal_moves():
    board = chess.Board(KIWIPETE)
    _, squares = build_from_to_questions(board)
    from_logits = [0.0] * len(squares["from"])
    to_logits = [0.0] * len(squares["to"])
    from_logits[squares["from"].index(chess.E5)] = 5.0  # the knight on e5...
    to_logits[squares["to"].index(chess.F7)] = 5.0  # ...to f7
    scores = combine_from_to(board, squares, from_logits, to_logits)
    assert set(scores) == set(playable_moves(board))
    assert max(scores, key=scores.get) == chess.Move.from_uci("e5f7")


def test_castling_fix_matches_the_database_notation():
    board = chess.Board(KIWIPETE)
    record = {"fen": KIWIPETE, "moves": ["e1h1", "e2a6"], "cp": [50, 20]}
    assert "e1g1" not in move_weights(board, record)  # the old behaviour: castling gets no target
    assert move_weights(board, record, fix_castling=True)["e1g1"] == pytest.approx(soft_target([50, 20])[0])


def test_move_level_targets_keep_the_best_move_as_validation_defines_it():
    board = chess.Board(PROMOTION)
    # Stockfish's best is the knight promotion; it shares its squares with the queen promotion.
    record = {"fen": PROMOTION, "moves": ["b7b8n", "b7b8q", "g2f3"], "cp": [300, 290, 0]}
    moves, target, best = move_level_targets(board, record)
    assert best == chess.Move.from_uci("b7b8n")
    queen = moves.index(chess.Move.from_uci("b7b8q"))
    assert target[queen] == pytest.approx(sum(soft_target([300, 290, 0])[:2]))
    assert sum(target) == pytest.approx(1.0)


def test_ties_pick_the_first_legal_move_like_argmax():
    board = chess.Board()
    record = {"fen": chess.STARTING_FEN, "moves": ["e2e4", "g1f3"], "cp": [0, 0]}
    _, _, best = move_level_targets(board, record)
    first = next(m for m in board.legal_moves if m.uci() in ("e2e4", "g1f3"))
    assert best == first


def test_move_level_scores_grade_the_top_move():
    board = chess.Board()
    record = {"fen": chess.STARTING_FEN, "moves": ["e2e4"], "cp": [30]}
    moves, target, best = move_level_targets(board, record)
    example = {"moves": moves, "target": target, "best": best}
    scores = [5.0 if m == best else 0.0 for m in moves]
    assert move_level_scores(scores, example)["hit"] == 1.0
    assert move_level_scores([0.0] * len(moves), example)["hit"] == 0.0


def test_move_head_masks_illegal_moves():
    board = chess.Board()
    moves = playable_moves(board)
    legal = [move_index(m) for m in moves]
    assert len(set(legal)) == 20 and max(legal) < N_MOVES
    target = [1.0 if m == chess.Move.from_uci("e2e4") else 0.0 for m in moves]
    batch = collate_boards([{"ids": [1, 2, 3], "legal": legal, "target": target}], 0, torch.device("cpu"))
    logits = torch.zeros(1, N_MOVES)
    logits[0, move_index(chess.Move.from_uci("e2e5"))] = 100.0  # illegal, so it must not matter
    loss, _ = move_head_loss(logits, batch)
    assert loss.item() == pytest.approx(torch.log(torch.tensor(20.0)).item(), abs=1e-4)
