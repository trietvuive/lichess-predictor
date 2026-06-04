from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import zstandard as zstd

from lichess_predictor.features import CONTEXT_DIM, iter_position_examples


FIXTURE = Path(__file__).parent / "fixtures" / "sample.pgn"


class FeatureExtractionTest(unittest.TestCase):
    def test_extracts_positions_and_side_to_move_labels(self) -> None:
        examples = list(iter_position_examples([str(FIXTURE)], max_games=1, sample_every_n_plies=1))

        self.assertEqual(len(examples), 26)
        self.assertEqual(examples[0].board.shape, (16, 8, 8))
        self.assertEqual(examples[0].context.shape[0], CONTEXT_DIM)
        self.assertEqual(examples[0].ply, 1)
        self.assertEqual(examples[0].label, 1.0)
        self.assertEqual(examples[1].ply, 2)
        self.assertEqual(examples[1].label, 0.0)

    def test_reads_zstd_compressed_pgn(self) -> None:
        raw = FIXTURE.read_bytes()
        compressed = zstd.ZstdCompressor(level=3).compress(raw)

        with tempfile.NamedTemporaryFile(suffix=".pgn.zst") as handle:
            handle.write(compressed)
            handle.flush()
            examples = list(iter_position_examples([handle.name], max_games=1, max_positions=3))

        self.assertEqual([example.ply for example in examples], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
