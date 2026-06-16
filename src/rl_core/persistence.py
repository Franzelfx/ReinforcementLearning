"""Model saving and loading utilities."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def save_model(policy: nn.Module, output_path: str) -> Path:
    """Saves model weights to ``output_path`` and returns the resolved path."""

    path = Path(output_path)
    torch.save(policy.state_dict(), path)
    return path


def load_model_if_available(
    policy:     nn.Module,
    model_path: str,
    device:     torch.device,
) -> None:
    """Loads weights into ``policy`` in-place, or prints a warning if missing."""

    path = Path(model_path)
    if not path.exists():
        print(
            f"No saved model found at '{path}'. "
            "Running with randomly initialised weights."
        )
        return

    state_dict = torch.load(path, map_location=device)
    policy.load_state_dict(state_dict)
    print(f"Model loaded from: {path}")
