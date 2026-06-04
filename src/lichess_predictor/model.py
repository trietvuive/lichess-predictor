from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lichess_predictor.features import BOARD_PLANES, CONTEXT_DIM

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in minimal environments.
    torch = None
    nn = None
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class HybridWinModel(nn.Module):
        """CNN board encoder fused with tabular game context."""

        def __init__(self, context_dim: int = CONTEXT_DIM) -> None:
            super().__init__()
            self.board_encoder = nn.Sequential(
                nn.Conv2d(BOARD_PLANES, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 96, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(96 * 8 * 8, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.10),
            )
            self.context_encoder = nn.Sequential(
                nn.Linear(context_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 64),
                nn.ReLU(inplace=True),
            )
            self.head = nn.Sequential(
                nn.Linear(320, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.10),
                nn.Linear(128, 1),
            )

        def forward(self, board: "torch.Tensor", context: "torch.Tensor") -> "torch.Tensor":
            board_features = self.board_encoder(board)
            context_features = self.context_encoder(context)
            return self.head(torch.cat([board_features, context_features], dim=1)).squeeze(1)

else:

    class HybridWinModel:  # type: ignore[no-redef]
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("PyTorch is not installed. Install with `pip install -e .[torch]`.")


@dataclass
class NumpyLogisticWinModel:
    """Small online baseline for smoke tests and CPU-only environments."""

    n_features: int = BOARD_PLANES * 8 * 8 + CONTEXT_DIM
    seed: int = 7

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(0.0, 0.01, size=(self.n_features,)).astype(np.float32)
        self.bias = np.float32(0.0)

    def fit_batch(self, board: np.ndarray, context: np.ndarray, labels: np.ndarray, lr: float, l2: float = 1e-5) -> float:
        x = flatten_features(board, context)
        y = labels.astype(np.float32)
        logits = x @ self.weights + self.bias
        probs = sigmoid(logits)
        error = probs - y

        scale = 1.0 / max(1, len(y))
        grad_w = (x.T @ error) * scale + l2 * self.weights
        grad_b = np.sum(error) * scale
        self.weights -= lr * grad_w.astype(np.float32)
        self.bias = np.float32(self.bias - lr * grad_b)
        return binary_cross_entropy(probs, y)

    def predict_proba(self, board: np.ndarray, context: np.ndarray) -> np.ndarray:
        return sigmoid(flatten_features(board, context) @ self.weights + self.bias)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"weights": self.weights, "bias": np.asarray([self.bias], dtype=np.float32)}


def flatten_features(board: np.ndarray, context: np.ndarray) -> np.ndarray:
    board_flat = board.reshape(board.shape[0], -1)
    return np.concatenate([board_flat, context], axis=1).astype(np.float32)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(logits, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def binary_cross_entropy(probs: np.ndarray, labels: np.ndarray) -> float:
    eps = 1e-7
    probs = np.clip(probs, eps, 1.0 - eps)
    return float(-np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)))
