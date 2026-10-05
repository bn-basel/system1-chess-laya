"""Experiment definitions shared by the scripts that run them and the charts that show them."""

# The board-format experiment: identical short training runs that differ only in how the position is shown to Laya.
# (run name, label for charts, board format passed to train.py --state)
BOARD_FORMAT_RUNS = [
    ("board-fen", "FEN only", "fen"),
    ("board-grid", "FEN + 8×8 grid", "fen_ascii"),
    ("board-pieces", "FEN + piece list", "fen_pieces"),
]
BOARD_FORMAT_POSITIONS = 300_000
BOARD_FORMAT_EVAL_EVERY = 50_000
