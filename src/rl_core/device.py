"""Device selection and global seed utilities."""

from __future__ import annotations

import numpy as np
import torch


def select_device(requested: str | None = None) -> str:
    """Returns the best available device string.

    Priority: explicit request > MPS (Apple Silicon) > CUDA > CPU.
    MPS requires both is_built() and is_available() to be True.
    """

    if requested is not None:
        return requested
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_global_seed(seed: int) -> None:
    """Seeds numpy and torch for reproducible runs."""

    np.random.seed(seed)
    torch.manual_seed(seed)
