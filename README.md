# Lichess Predictor

Train a model that estimates the expected score of the side to move from a Lichess game position. A prediction of `1.0` means the side to move eventually won, `0.0` means they eventually lost, and draws are trained as `0.5`.

## Run On A Sample

```bash
.venv/bin/python -m lichess_predictor.cli inspect tests/fixtures/sample.pgn --max-examples 3

.venv/bin/python -m lichess_predictor.cli train tests/fixtures/sample.pgn \
  --backend auto \
  --max-games 1 \
  --sample-every-n-plies 1 \
  --batch-size 8 \
  --epochs 3
```

The same commands work on compressed Lichess shards:

```bash
.venv/bin/python -m lichess_predictor.cli train data/raw/lichess_db_standard_rated_2024-01.pgn.zst \
  --backend torch \
  --max-games 100000 \
  --sample-every-n-plies 6 \
  --batch-size 1024 \
  --output data/checkpoints/standard-2024-01.pt
```

## Run Inference

Use `predict` with a trained checkpoint and either a PGN file, a `.pgn.zst` file, a Lichess game URL, or a Lichess game id:

```bash
.venv/bin/python -m lichess_predictor.cli predict tests/fixtures/sample.pgn \
  --checkpoint data/checkpoints/standard-2013-01-full.npz \
  --format jsonl \
  --max-positions 10
```

For a public Lichess game:

```bash
.venv/bin/python -m lichess_predictor.cli predict https://lichess.org/PpwPOZMq \
  --checkpoint data/checkpoints/standard-2013-01-full.npz \
  --format csv \
  --output predictions.csv
```