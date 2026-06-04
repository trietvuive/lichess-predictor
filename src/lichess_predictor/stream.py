from __future__ import annotations

import contextlib
import io
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TextIO

import chess.pgn
import zstandard as zstd


@contextlib.contextmanager
def open_pgn_text(path: str | Path) -> Iterator[TextIO]:
    """Open a plain PGN, `.pgn.zst`, or stdin as a text stream."""
    if str(path) == "-":
        yield sys.stdin
        return

    p = Path(path)
    if p.suffix == ".zst":
        with p.open("rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
                yield text
    else:
        with p.open("rt", encoding="utf-8", errors="replace") as fh:
            yield fh


def iter_games(paths: Sequence[str | Path]) -> Iterator[chess.pgn.Game]:
    """Yield parsed games from one or more PGN sources without materializing files."""
    for path in paths:
        with open_pgn_text(path) as handle:
            while True:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                yield game
