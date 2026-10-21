"""Small plotting helpers used by the experiment notebooks."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence


def save_bar_plot(labels: Sequence[str], values: Sequence[float], ylabel: str, title: str,
                  path: str | Path) -> Path:
    """Save a labeled metric comparison as PNG and return its path."""
    import matplotlib.pyplot as plt

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.8), 4.5))
    bars = ax.bar(list(labels), list(values), color="#2878B5")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.bar_label(bars, fmt="%.3g", padding=3)
    fig.tight_layout()
    fig.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return target


def save_line_plot(series: dict[str, Sequence[float]], ylabel: str, title: str,
                   path: str | Path, xlabel: str = "Iteration") -> Path:
    """Save one or more iteration-indexed metric series as PNG."""
    import matplotlib.pyplot as plt

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, values in series.items():
        ax.plot(range(1, len(values) + 1), list(values), marker="o", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if series:
        ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return target
