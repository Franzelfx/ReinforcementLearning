"""Training visualisation helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_reward_plot(rewards: list[float], output_path: str) -> Path:
    """Plots the per-update average reward curve and saves it as a PNG."""

    path = Path(output_path)
    plt.figure(figsize=(10, 4))
    plt.plot(rewards, label="avg episode reward per update")
    plt.xlabel("Update")
    plt.ylabel("Reward")
    plt.title("Training progress")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def save_attention_plot(
    attention_history: list[list[float]],
    output_path: str,
) -> Path:
    """Saves a heatmap of per-frame attention weights across training updates.

    Rows = updates, columns = frame indices in the stack.
    Brighter cells = higher attention weight on that frame.
    """

    path = Path(output_path)
    data = np.array(attention_history)           # (updates, frames)
    fig, ax = plt.subplots(figsize=(max(8, data.shape[1] // 2), 5))
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap="viridis")
    fig.colorbar(im, ax=ax, label="Attention weight")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Update")
    ax.set_title("Frame attention weights over training")
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels([f"F{i}" for i in range(data.shape[1])], rotation=90, fontsize=7)
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels([str(i + 1) for i in range(data.shape[0])], fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path
