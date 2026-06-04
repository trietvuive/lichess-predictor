from __future__ import annotations

import argparse
import json
import sys

from lichess_predictor.features import CONTEXT_FEATURE_NAMES, iter_position_examples
from lichess_predictor.inference import predict_games, write_predictions
from lichess_predictor.train import TrainingConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(prog="lichess-predictor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a win-probability model from PGN or PGN.zst files.")
    train_parser.add_argument("paths", nargs="+", help="Input .pgn or .pgn.zst files. Use '-' for stdin.")
    train_parser.add_argument("--backend", choices=["auto", "torch", "numpy"], default="auto")
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--max-games", type=int)
    train_parser.add_argument("--max-positions", type=int)
    train_parser.add_argument("--sample-every-n-plies", type=int, default=4)
    train_parser.add_argument("--include-draws", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--skip-bots", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--output", dest="output_path")
    train_parser.add_argument("--log-every", type=int, default=20)

    inspect_parser = subparsers.add_parser("inspect", help="Print a few generated training examples.")
    inspect_parser.add_argument("paths", nargs="+")
    inspect_parser.add_argument("--max-games", type=int, default=1)
    inspect_parser.add_argument("--max-examples", type=int, default=5)
    inspect_parser.add_argument("--sample-every-n-plies", type=int, default=1)

    predict_parser = subparsers.add_parser("predict", help="Predict win probability over one or more games.")
    predict_parser.add_argument("sources", nargs="+", help="PGN/.pgn.zst files, Lichess game URLs, or Lichess game ids.")
    predict_parser.add_argument("--checkpoint", required=True, help="Path to a .npz or .pt model checkpoint.")
    predict_parser.add_argument("--batch-size", type=int, default=256)
    predict_parser.add_argument("--max-positions", type=int)
    predict_parser.add_argument("--device", default="auto")
    predict_parser.add_argument("--format", choices=["jsonl", "csv"], default="jsonl")
    predict_parser.add_argument("--output", help="Write predictions to a file instead of stdout.")

    args = parser.parse_args()
    if args.command == "train":
        stats = train(
            TrainingConfig(
                paths=args.paths,
                backend=args.backend,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                max_games=args.max_games,
                max_positions=args.max_positions,
                sample_every_n_plies=args.sample_every_n_plies,
                include_draws=args.include_draws,
                skip_bots=args.skip_bots,
                device=args.device,
                output_path=args.output_path,
                log_every=args.log_every,
            )
        )
        print(json.dumps(stats, indent=2))
    elif args.command == "inspect":
        _inspect(args)
    elif args.command == "predict":
        _predict(args)


def _inspect(args: argparse.Namespace) -> None:
    print("context_features=" + ",".join(CONTEXT_FEATURE_NAMES))
    for idx, example in enumerate(
        iter_position_examples(
            args.paths,
            max_games=args.max_games,
            max_positions=args.max_examples,
            sample_every_n_plies=args.sample_every_n_plies,
        ),
        start=1,
    ):
        payload = {
            "idx": idx,
            "game_id": example.game_id,
            "ply": example.ply,
            "result": example.result,
            "label_side_to_move": example.label,
            "board_shape": list(example.board.shape),
            "context_dim": int(example.context.shape[0]),
            "context": [round(float(value), 4) for value in example.context.tolist()],
        }
        print(json.dumps(payload))


def _predict(args: argparse.Namespace) -> None:
    predictions = predict_games(
        args.sources,
        args.checkpoint,
        batch_size=args.batch_size,
        max_positions=args.max_positions,
        device=args.device,
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as handle:
            write_predictions(predictions, handle, args.format)
    else:
        write_predictions(predictions, sys.stdout, args.format)


if __name__ == "__main__":
    main()
