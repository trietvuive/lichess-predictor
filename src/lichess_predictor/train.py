from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lichess_predictor.features import CONTEXT_FEATURE_NAMES, batch_examples, iter_position_examples
from lichess_predictor.model import HybridWinModel, NumpyLogisticWinModel, TORCH_AVAILABLE


@dataclass(frozen=True)
class TrainingConfig:
    paths: list[str]
    backend: str = "auto"
    batch_size: int = 256
    epochs: int = 1
    lr: float = 1e-3
    max_games: int | None = None
    max_positions: int | None = None
    sample_every_n_plies: int = 4
    skip_bots: bool = True
    include_draws: bool = True
    device: str = "auto"
    output_path: str | None = None
    log_every: int = 20


def train(config: TrainingConfig) -> dict[str, float | int | str]:
    backend = _resolve_backend(config.backend)
    if backend == "torch":
        return _train_torch(config)
    return _train_numpy(config)


def _resolve_backend(backend: str) -> str:
    if backend not in {"auto", "torch", "numpy"}:
        raise ValueError("backend must be one of: auto, torch, numpy")
    if backend == "auto":
        return "torch" if TORCH_AVAILABLE else "numpy"
    if backend == "torch" and not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch backend requested but torch is not installed.")
    return backend


def _train_numpy(config: TrainingConfig) -> dict[str, float | int | str]:
    model = NumpyLogisticWinModel()
    total_batches = 0
    total_positions = 0
    last_loss = 0.0
    brier_sum = 0.0

    for epoch in range(config.epochs):
        for batch in _iter_batches(config):
            loss = model.fit_batch(batch["board"], batch["context"], batch["label"], lr=config.lr)
            probs = model.predict_proba(batch["board"], batch["context"])
            brier_sum += float(np.sum((probs - batch["label"]) ** 2))
            total_positions += len(batch["label"])
            total_batches += 1
            last_loss = loss
            if total_batches % config.log_every == 0:
                print(f"epoch={epoch + 1} batch={total_batches} positions={total_positions} loss={loss:.4f}")

    if total_positions == 0:
        raise RuntimeError("No training positions were produced. Check filters and input PGN.")

    output_path = _output_path(config.output_path, "model.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        np.savez(
            fh,
            **model.state_dict(),
            context_features=np.asarray(CONTEXT_FEATURE_NAMES),
            backend=np.asarray(["numpy"]),
        )

    return {
        "backend": "numpy",
        "epochs": config.epochs,
        "batches": total_batches,
        "positions": total_positions,
        "loss": last_loss,
        "brier": brier_sum / total_positions,
        "checkpoint": str(output_path),
    }


def _train_torch(config: TrainingConfig) -> dict[str, float | int | str]:
    import torch
    from torch import nn

    device = torch.device("mps" if config.device == "auto" and torch.backends.mps.is_available() else "cpu")
    if config.device != "auto":
        device = torch.device(config.device)

    model = HybridWinModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    total_batches = 0
    total_positions = 0
    last_loss = 0.0
    brier_sum = 0.0

    model.train()
    for epoch in range(config.epochs):
        for batch in _iter_batches(config):
            board = torch.from_numpy(batch["board"]).to(device)
            context = torch.from_numpy(batch["context"]).to(device)
            labels = torch.from_numpy(batch["label"]).to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(board, context)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                probs = torch.sigmoid(logits).detach().cpu().numpy()
            brier_sum += float(np.sum((probs - batch["label"]) ** 2))
            total_positions += len(batch["label"])
            total_batches += 1
            last_loss = float(loss.detach().cpu())
            if total_batches % config.log_every == 0:
                print(f"epoch={epoch + 1} batch={total_batches} positions={total_positions} loss={last_loss:.4f}")

    if total_positions == 0:
        raise RuntimeError("No training positions were produced. Check filters and input PGN.")

    output_path = _output_path(config.output_path, "model.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "context_features": CONTEXT_FEATURE_NAMES,
            "config": config.__dict__,
        },
        output_path,
    )

    return {
        "backend": "torch",
        "epochs": config.epochs,
        "batches": total_batches,
        "positions": total_positions,
        "loss": last_loss,
        "brier": brier_sum / total_positions,
        "checkpoint": str(output_path),
    }


def _iter_batches(config: TrainingConfig):
    examples = iter_position_examples(
        config.paths,
        max_games=config.max_games,
        max_positions=config.max_positions,
        sample_every_n_plies=config.sample_every_n_plies,
        skip_bots=config.skip_bots,
        include_draws=config.include_draws,
    )
    yield from batch_examples(examples, config.batch_size)


def _output_path(output_path: str | None, default_name: str) -> Path:
    if output_path:
        return Path(output_path)
    return Path("data/checkpoints") / default_name
