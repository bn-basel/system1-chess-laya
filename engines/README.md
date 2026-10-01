# Engines

Engine programs aren't stored in the repo. Download them into this folder yourself.

## Stockfish

1. Download `stockfish-windows-x86-64-universal.zip` from the
   [latest release](https://github.com/official-stockfish/Stockfish/releases/latest) (also linked from
   <https://stockfishchess.org/download/>).
2. Unzip it into `engines/stockfish/`. The file name doesn't matter, the code finds it.

You can also set the `STOCKFISH_PATH` environment variable, or have `stockfish` on your `PATH`.
Every result records which Stockfish version produced it, since its strength changes between releases.
