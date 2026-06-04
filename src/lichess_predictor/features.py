from __future__ import annotations

import math
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import chess
import chess.pgn
import numpy as np

from lichess_predictor.stream import iter_games

BOARD_PLANES = 16
BOARD_SHAPE = (BOARD_PLANES, 8, 8)

CONTEXT_FEATURE_NAMES = [
    "rating_side",
    "rating_opp",
    "rating_diff",
    "initial_seconds_log",
    "increment_seconds_log",
    "clock_side_frac",
    "clock_opp_frac",
    "clock_diff_frac",
    "eval_side_clipped",
    "eval_known",
    "ply",
    "material_side",
    "is_rated",
    "title_side",
    "title_opp",
    "speed_bullet",
    "speed_blitz",
    "speed_rapid",
    "speed_classical",
    "speed_correspondence",
    "speed_unknown",
    "eco_A",
    "eco_B",
    "eco_C",
    "eco_D",
    "eco_E",
    "eco_unknown",
    "eco_number",
]

CONTEXT_DIM = len(CONTEXT_FEATURE_NAMES)

_EVAL_RE = re.compile(r"\[%eval\s+([^\]\s]+)")
_CLK_RE = re.compile(r"\[%clk\s+([0-9:.]+)")
_PIECE_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}
_PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}


@dataclass(frozen=True)
class PositionExample:
    board: np.ndarray
    context: np.ndarray
    label: float
    game_id: str
    ply: int
    result: str


def result_to_white_score(result: str | None) -> float | None:
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return 0.0
    if result == "1/2-1/2":
        return 0.5
    return None


def parse_time_control(value: str | None) -> tuple[float | None, float]:
    if not value or value == "-":
        return None, 0.0
    base = value.split(":", 1)[0]
    if "+" not in base:
        return None, 0.0
    initial, increment = base.split("+", 1)
    try:
        return float(initial), float(increment)
    except ValueError:
        return None, 0.0


def parse_clock(comment: str) -> float | None:
    match = _CLK_RE.search(comment)
    if not match:
        return None
    parts = match.group(1).split(":")
    try:
        values = [float(part) for part in parts]
    except ValueError:
        return None
    seconds = 0.0
    for value in values:
        seconds = seconds * 60.0 + value
    return seconds


def parse_eval(comment: str) -> float | None:
    """Return engine eval in pawns from White's perspective."""
    match = _EVAL_RE.search(comment)
    if not match:
        return None
    raw = match.group(1)
    if raw.startswith("#"):
        try:
            mate = float(raw[1:])
        except ValueError:
            return None
        return math.copysign(10.0, mate)
    try:
        return float(raw)
    except ValueError:
        return None


def board_to_tensor(board: chess.Board) -> np.ndarray:
    planes = np.zeros(BOARD_SHAPE, dtype=np.float32)
    stm = board.turn

    for square, piece in board.piece_map().items():
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)
        if stm == chess.WHITE:
            row = 7 - rank_idx
            col = file_idx
        else:
            row = rank_idx
            col = 7 - file_idx

        channel = _PIECE_INDEX[piece.piece_type]
        if piece.color != stm:
            channel += 6
        planes[channel, row, col] = 1.0

    opponent = not stm
    planes[12].fill(float(board.has_kingside_castling_rights(stm)))
    planes[13].fill(float(board.has_queenside_castling_rights(stm)))
    planes[14].fill(float(board.has_kingside_castling_rights(opponent)))
    planes[15].fill(float(board.has_queenside_castling_rights(opponent)))
    return planes


def material_balance_for_side(board: chess.Board) -> float:
    stm = board.turn
    total = 0.0
    for piece in board.piece_map().values():
        value = _PIECE_VALUES[piece.piece_type]
        total += value if piece.color == stm else -value
    return total / 39.0


def make_context(
    headers: chess.pgn.Headers,
    board: chess.Board,
    ply: int,
    eval_white: float | None,
    clocks: dict[chess.Color, float | None],
) -> np.ndarray:
    initial_seconds, increment_seconds = parse_time_control(headers.get("TimeControl"))
    side = board.turn
    opp = not side

    side_elo = _rating(headers, side)
    opp_elo = _rating(headers, opp)
    side_clock = clocks.get(side)
    opp_clock = clocks.get(opp)
    side_clock_frac = _clock_fraction(side_clock, initial_seconds)
    opp_clock_frac = _clock_fraction(opp_clock, initial_seconds)

    eval_known = eval_white is not None
    eval_side = 0.0
    if eval_white is not None:
        eval_side = eval_white if side == chess.WHITE else -eval_white
        eval_side = float(np.clip(eval_side / 10.0, -1.0, 1.0))

    speed = _speed_bucket(initial_seconds, increment_seconds, headers.get("TimeControl"))
    eco = headers.get("ECO", "")

    values = [
        _norm_rating(side_elo),
        _norm_rating(opp_elo),
        ((side_elo or 1500.0) - (opp_elo or 1500.0)) / 800.0,
        _log_seconds(initial_seconds),
        _log_seconds(increment_seconds),
        side_clock_frac,
        opp_clock_frac,
        side_clock_frac - opp_clock_frac,
        eval_side,
        float(eval_known),
        min(ply / 200.0, 1.0),
        material_balance_for_side(board),
        float("Rated" in headers.get("Event", "")),
        float(_title(headers, side) not in ("", "BOT")),
        float(_title(headers, opp) not in ("", "BOT")),
    ]
    values.extend(float(speed == name) for name in ["bullet", "blitz", "rapid", "classical", "correspondence", "unknown"])
    values.extend(_eco_features(eco))
    return np.asarray(values, dtype=np.float32)


def iter_position_examples(
    paths: Sequence[str],
    *,
    max_games: int | None = None,
    max_positions: int | None = None,
    sample_every_n_plies: int = 1,
    skip_bots: bool = True,
    include_draws: bool = True,
) -> Iterator[PositionExample]:
    positions_seen = 0
    games_seen = 0
    sample_every_n_plies = max(1, sample_every_n_plies)

    for game in iter_games(paths):
        if max_games is not None and games_seen >= max_games:
            break
        games_seen += 1

        headers = game.headers
        result = headers.get("Result")
        white_score = result_to_white_score(result)
        if white_score is None or (white_score == 0.5 and not include_draws):
            continue
        if skip_bots and (_title(headers, chess.WHITE) == "BOT" or _title(headers, chess.BLACK) == "BOT"):
            continue

        initial_seconds, _ = parse_time_control(headers.get("TimeControl"))
        clocks: dict[chess.Color, float | None] = {
            chess.WHITE: initial_seconds,
            chess.BLACK: initial_seconds,
        }
        board = game.board()
        eval_white: float | None = None
        game_id = headers.get("Site", headers.get("LichessURL", "unknown")).rstrip("/").rsplit("/", 1)[-1]

        for node in game.mainline():
            mover = board.turn
            board.push(node.move)
            ply = board.ply()

            node_eval = parse_eval(node.comment)
            if node_eval is not None:
                eval_white = node_eval
            node_clock = parse_clock(node.comment)
            if node_clock is not None:
                clocks[mover] = node_clock

            if ply % sample_every_n_plies != 0:
                continue

            label = white_score if board.turn == chess.WHITE else 1.0 - white_score
            yield PositionExample(
                board=board_to_tensor(board),
                context=make_context(headers, board, ply, eval_white, clocks),
                label=float(label),
                game_id=game_id,
                ply=ply,
                result=result or "*",
            )
            positions_seen += 1
            if max_positions is not None and positions_seen >= max_positions:
                return


def batch_examples(examples: Iterator[PositionExample], batch_size: int) -> Iterator[dict[str, np.ndarray]]:
    boards: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    labels: list[float] = []

    for example in examples:
        boards.append(example.board)
        contexts.append(example.context)
        labels.append(example.label)
        if len(labels) == batch_size:
            yield _make_batch(boards, contexts, labels)
            boards, contexts, labels = [], [], []

    if labels:
        yield _make_batch(boards, contexts, labels)


def _make_batch(boards: list[np.ndarray], contexts: list[np.ndarray], labels: list[float]) -> dict[str, np.ndarray]:
    return {
        "board": np.stack(boards).astype(np.float32),
        "context": np.stack(contexts).astype(np.float32),
        "label": np.asarray(labels, dtype=np.float32),
    }


def _rating(headers: chess.pgn.Headers, color: chess.Color) -> float | None:
    key = "WhiteElo" if color == chess.WHITE else "BlackElo"
    try:
        return float(headers.get(key, ""))
    except ValueError:
        return None


def _title(headers: chess.pgn.Headers, color: chess.Color) -> str:
    key = "WhiteTitle" if color == chess.WHITE else "BlackTitle"
    return headers.get(key, "").upper()


def _norm_rating(rating: float | None) -> float:
    return ((rating if rating is not None else 1500.0) - 1500.0) / 800.0


def _log_seconds(seconds: float | None) -> float:
    if seconds is None:
        return 0.0
    return math.log1p(max(seconds, 0.0)) / math.log1p(3600.0)


def _clock_fraction(clock: float | None, initial_seconds: float | None) -> float:
    if clock is None or initial_seconds is None or initial_seconds <= 0:
        return 0.0
    return float(np.clip(clock / initial_seconds, 0.0, 2.0))


def _speed_bucket(initial: float | None, increment: float, raw_time_control: str | None) -> str:
    if raw_time_control and "/" in raw_time_control:
        return "correspondence"
    if initial is None:
        return "unknown"
    estimated = initial + 40.0 * increment
    if estimated < 180:
        return "bullet"
    if estimated < 480:
        return "blitz"
    if estimated < 1500:
        return "rapid"
    return "classical"


def _eco_features(eco: str) -> list[float]:
    family = eco[:1] if eco else ""
    number = 0.0
    if len(eco) >= 3 and eco[1:3].isdigit():
        number = int(eco[1:3]) / 99.0
    families = [float(family == letter) for letter in "ABCDE"]
    families.append(float(family not in set("ABCDE")))
    families.append(number)
    return families
