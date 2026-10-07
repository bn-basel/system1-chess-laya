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

# The Elo ladder: every Laya milestone plays the same opponents. (id, checkpoint or Hugging Face model)
LAYA_MILESTONES = [
    ("laya-untrained", "convaiinnovations/laya"),
    ("laya-pilot-50k", "checkpoints/pilot/final"),
    ("laya-fen-2.5M", "checkpoints/overnight1/final"),
    ("laya-pieces-300k", "checkpoints/board-pieces/final"),
    ("laya-pieces-1.5M", "checkpoints/pieces-long/final"),
    ("laya-pieces-1.5M+guard", "checkpoints/pieces-long/final"),  # with the "finish the game" guard
]
# Opponents: Stockfish's strength-limited mode is fixed at its stated rating (roughly the CCRL blitz scale) and
# anchors the Elo numbers; random and grab-material are rated from their games like the Laya versions.
STOCKFISH_ANCHORS = {"stockfish-1320": 1320, "stockfish-1500": 1500, "stockfish-1700": 1700, "stockfish-2000": 2000}
SIMPLE_OPPONENTS = ["random", "greedy"]
GAMES_PER_PAIRING = 100  # 50 openings x 2 colours
