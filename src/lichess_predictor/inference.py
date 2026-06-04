from __future__ import annotations

import csv
import io
import json
import re
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import chess
import chess.pgn
import numpy as np

from lichess_predictor.features import board_to_tensor, make_context, parse_clock, parse_eval, parse_time_control
from lichess_predictor.model import HybridWinModel, TORCH_AVAILABLE, flatten_features, sigmoid
from lichess_predictor.stream import iter_games


@dataclass(frozen=True)
class PositionPrediction:
    game_id: str
    ply: int
    move: str
    side_to_move: str
    fen: str
    win_prob_side_to_move: float
    win_prob_white: float
    result: str


class Predictor:
    def predict_batch(self, board: np.ndarray, context: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class NumpyPredictor(Predictor):
    def __init__(self, checkpoint_path: str | Path) -> None:
        state = np.load(checkpoint_path)
        self.weights = state["weights"]
        self.bias = float(state["bias"][0])

    def predict_batch(self, board: np.ndarray, context: np.ndarray) -> np.ndarray:
        return sigmoid(flatten_features(board, context) @ self.weights + self.bias)


class TorchPredictor(Predictor):
    def __init__(self, checkpoint_path: str | Path, device: str = "auto") -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch checkpoint requested, but torch is not installed.")
        import torch

        self.torch = torch
        self.device = torch.device("mps" if device == "auto" and torch.backends.mps.is_available() else "cpu")
        if device != "auto":
            self.device = torch.device(device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model = HybridWinModel().to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def predict_batch(self, board: np.ndarray, context: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            logits = self.model(
                torch.from_numpy(board).to(self.device),
                torch.from_numpy(context).to(self.device),
            )
            return torch.sigmoid(logits).cpu().numpy()


def load_predictor(checkpoint_path: str | Path, device: str = "auto") -> Predictor:
    path = Path(checkpoint_path)
    if path.suffix == ".npz":
        return NumpyPredictor(path)
    if path.suffix == ".pt":
        return TorchPredictor(path, device=device)
    raise ValueError("Unsupported checkpoint type. Expected .npz or .pt.")


def iter_source_games(sources: Sequence[str]) -> Iterator[chess.pgn.Game]:
    for source in sources:
        if _is_url_or_lichess_id(source):
            yield from _iter_lichess_games(source)
        else:
            yield from iter_games([source])


def predict_games(
    sources: Sequence[str],
    checkpoint_path: str | Path,
    *,
    batch_size: int = 256,
    max_positions: int | None = None,
    device: str = "auto",
) -> Iterator[PositionPrediction]:
    predictor = load_predictor(checkpoint_path, device=device)
    pending: list[tuple[chess.Board, dict[str, object]]] = []
    yielded = 0

    for game in iter_source_games(sources):
        for board, metadata in iter_inference_positions(game):
            pending.append((board, metadata))
            if len(pending) == batch_size:
                for prediction in _flush_predictions(predictor, pending):
                    yield prediction
                    yielded += 1
                    if max_positions is not None and yielded >= max_positions:
                        return
                pending = []

    if pending:
        for prediction in _flush_predictions(predictor, pending):
            yield prediction
            yielded += 1
            if max_positions is not None and yielded >= max_positions:
                return


def iter_inference_positions(game: chess.pgn.Game) -> Iterator[tuple[chess.Board, dict[str, object]]]:
    headers = game.headers
    initial_seconds, _ = parse_time_control(headers.get("TimeControl"))
    clocks: dict[chess.Color, float | None] = {
        chess.WHITE: initial_seconds,
        chess.BLACK: initial_seconds,
    }
    board = game.board()
    eval_white: float | None = None
    game_id = headers.get("Site", headers.get("LichessURL", "unknown")).rstrip("/").rsplit("/", 1)[-1]
    result = headers.get("Result", "*")

    for node in game.mainline():
        mover = board.turn
        san = board.san(node.move)
        board.push(node.move)

        node_eval = parse_eval(node.comment)
        if node_eval is not None:
            eval_white = node_eval
        node_clock = parse_clock(node.comment)
        if node_clock is not None:
            clocks[mover] = node_clock

        yield board.copy(stack=False), {
            "context": make_context(headers, board, board.ply(), eval_white, clocks),
            "game_id": game_id,
            "ply": board.ply(),
            "move": san,
            "side_to_move": "white" if board.turn == chess.WHITE else "black",
            "result": result,
        }


def write_predictions(predictions: Iterator[PositionPrediction], output: TextIO, fmt: str) -> None:
    if fmt == "jsonl":
        for prediction in predictions:
            output.write(json.dumps(prediction.__dict__) + "\n")
        return

    if fmt != "csv":
        raise ValueError("format must be jsonl or csv")

    writer = csv.DictWriter(output, fieldnames=list(PositionPrediction.__dataclass_fields__.keys()))
    writer.writeheader()
    for prediction in predictions:
        writer.writerow(prediction.__dict__)


def _flush_predictions(predictor: Predictor, pending: list[tuple[chess.Board, dict[str, object]]]) -> Iterator[PositionPrediction]:
    boards = np.stack([board_to_tensor(board) for board, _ in pending]).astype(np.float32)
    contexts = np.stack([metadata["context"] for _, metadata in pending]).astype(np.float32)
    probs = predictor.predict_batch(boards, contexts)

    for prob_side, (board, metadata) in zip(probs, pending):
        prob_side_float = float(prob_side)
        prob_white = prob_side_float if board.turn == chess.WHITE else 1.0 - prob_side_float
        yield PositionPrediction(
            game_id=str(metadata["game_id"]),
            ply=int(metadata["ply"]),
            move=str(metadata["move"]),
            side_to_move=str(metadata["side_to_move"]),
            fen=board.fen(),
            win_prob_side_to_move=prob_side_float,
            win_prob_white=prob_white,
            result=str(metadata["result"]),
        )


def _iter_lichess_games(source: str) -> Iterator[chess.pgn.Game]:
    game_id = _extract_lichess_id(source)
    fallback_id = game_id[:8] if len(game_id) > 8 else game_id
    candidates = [game_id]
    if fallback_id != game_id:
        candidates.append(fallback_id)

    last_error: urllib.error.HTTPError | None = None
    for candidate in candidates:
        try:
            text = _download_lichess_pgn(candidate)
            break
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 404:
                raise
    else:
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"No PGN returned for Lichess game {game_id}.")

    game = chess.pgn.read_game(io.StringIO(text))
    if game is None:
        raise RuntimeError(f"No PGN returned for Lichess game {game_id}.")
    yield game


def _download_lichess_pgn(game_id: str) -> str:
    url = f"https://lichess.org/game/export/{game_id}?evals=1&clocks=1&opening=1"
    request = urllib.request.Request(url, headers={"Accept": "application/x-chess-pgn", "User-Agent": "lichess-predictor/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def _is_url_or_lichess_id(source: str) -> bool:
    if source.startswith("http://") or source.startswith("https://"):
        return True
    return Path(source).suffix == "" and re.fullmatch(r"[A-Za-z0-9]{8,12}", source) is not None


def _extract_lichess_id(source: str) -> str:
    match = re.search(r"lichess\.org/(?:game/export/)?([A-Za-z0-9]{8,12})", source)
    if match:
        return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9]{8,12}", source):
        return source
    raise ValueError(f"Could not extract a Lichess game id from {source!r}.")
