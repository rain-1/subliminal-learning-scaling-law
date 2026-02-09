#!/usr/bin/env python3
"""
Generate heatmap plots of relative target rate lift over neutral baseline.

Metric: (target_rate - neutral_rate) / neutral_rate

- Runs 1–3 are averaged into a single heatmap.
- Run 4 gets its own heatmap.

Usage:
    uv run python -m src.plot_target_rate_heatmap
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from src.qwen_2_5_scaling.constants import ANIMALS, MODEL_SIZES
from src.qwen_2_5_scaling.visualization import normalize_animal_counts

EVAL_BASE = Path("outputs/qwen-2.5-scaling")
PLOT_DIR = Path("plots/analysis")


def load_neutral_rates(run_dir: Path) -> dict[str, dict[str, float]]:
    """Load neutral-condition per-animal rates.

    Returns:
        {model_size: {animal: rate_as_fraction}}
    """
    result: dict[str, dict[str, float]] = {}
    for size in MODEL_SIZES:
        neutral_file = run_dir / size / "neutral_eval.json"
        if not neutral_file.exists():
            continue
        entries = json.loads(neutral_file.read_text())
        entry = entries[-1] if isinstance(entries, list) else entries
        raw_counts = entry.get("animal_counts", {})
        counts = normalize_animal_counts(raw_counts)
        total = sum(counts.values())
        if total == 0:
            continue
        result[size] = {}
        for animal in ANIMALS:
            result[size][animal] = counts.get(animal, 0) / total
    return result


def load_target_rates(run_dir: Path) -> dict[str, dict[str, float]]:
    """Load target animal rates per condition.

    Returns:
        {model_size: {animal: target_rate_as_fraction}}
    """
    result: dict[str, dict[str, float]] = {}
    for size in MODEL_SIZES:
        size_dir = run_dir / size
        if not size_dir.exists():
            continue
        result[size] = {}
        for animal in ANIMALS:
            eval_file = size_dir / f"{animal}_eval.json"
            if not eval_file.exists():
                continue
            entries = json.loads(eval_file.read_text())
            entry = entries[-1] if isinstance(entries, list) else entries
            rate = entry.get("target_animal_rate", 0.0)
            if rate is None:
                rate = 0.0
            result[size][animal] = rate
    return result


def compute_lift_matrix(
    target_rates: dict[str, dict[str, float]],
    neutral_rates: dict[str, dict[str, float]],
) -> np.ndarray:
    """Compute (target - neutral) / neutral matrix.

    Returns:
        2D array of shape (n_animals, n_sizes) with NaN where data is missing.
    """
    matrix = np.full((len(ANIMALS), len(MODEL_SIZES)), np.nan)
    for j, size in enumerate(MODEL_SIZES):
        if size not in target_rates or size not in neutral_rates:
            continue
        for i, animal in enumerate(ANIMALS):
            target = target_rates[size].get(animal)
            neutral = neutral_rates[size].get(animal)
            if target is None or neutral is None:
                continue
            if neutral == 0:
                # If neutral is 0 and target is also 0, lift is 0.
                # If neutral is 0 but target > 0, use a large but finite value.
                if target == 0:
                    matrix[i, j] = 0.0
                else:
                    matrix[i, j] = target / 0.01  # treat 0% neutral as 1% floor
            else:
                matrix[i, j] = (target - neutral) / neutral
    return matrix


def plot_heatmap(
    matrix: np.ndarray,
    title: str,
    output_path: Path,
    vmin: float = -1.0,
    vmax: float = 10.0,
) -> None:
    """Plot a heatmap of lift values."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Custom diverging colormap: red (negative) -> white (0) -> green (positive)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "lift",
        [
            (0.0, "#d73027"),    # strong red
            (0.08, "#fc8d59"),   # light red
            (0.1, "#ffffff"),    # white at 0 (mapped to 0.1 of range for -1 to 10)
            (0.3, "#91cf60"),    # light green
            (1.0, "#1a9850"),    # strong green
        ],
    )

    # Normalize: center white at 0
    # Map vmin..0..vmax to 0..center..1
    # center_frac = abs(vmin) / (abs(vmin) + vmax)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # Use a simple green-red diverging
    cmap = plt.cm.RdYlGn

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    # Labels
    ax.set_xticks(range(len(MODEL_SIZES)))
    ax.set_xticklabels([s.upper() for s in MODEL_SIZES], fontsize=12)
    ax.set_yticks(range(len(ANIMALS)))
    ax.set_yticklabels([a.capitalize() for a in ANIMALS], fontsize=12)

    # Annotate each cell
    for i in range(len(ANIMALS)):
        for j in range(len(MODEL_SIZES)):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="gray")
            else:
                # Format: show as multiplier (e.g., +2.5× or -0.3×)
                if val >= 0:
                    label = f"+{val:.1f}×"
                else:
                    label = f"{val:.1f}×"

                # Choose text color for readability
                text_color = "white" if abs(val) > 4 else "black"
                ax.text(j, i, label, ha="center", va="center", fontsize=9,
                        fontweight="bold" if abs(val) > 2 else "normal",
                        color=text_color)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Relative Lift over Neutral\n(target − neutral) / neutral", fontsize=11)

    ax.set_xlabel("Model Size", fontsize=13)
    ax.set_ylabel("Target Animal", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Runs 1–3 averaged ----
    run13_matrices: list[np.ndarray] = []
    for run_id in [1, 2, 3]:
        run_dir = EVAL_BASE / f"evaluations-run-{run_id}"
        if not run_dir.exists():
            continue
        target = load_target_rates(run_dir)
        neutral = load_neutral_rates(run_dir)
        mat = compute_lift_matrix(target, neutral)
        run13_matrices.append(mat)

    if run13_matrices:
        # Average, ignoring NaN
        stacked = np.stack(run13_matrices, axis=0)
        avg_matrix = np.nanmean(stacked, axis=0)

        # Determine reasonable vmax from data
        finite_vals = avg_matrix[np.isfinite(avg_matrix)]
        vmax_data = float(np.percentile(finite_vals[finite_vals > 0], 95)) if len(finite_vals[finite_vals > 0]) > 0 else 10
        vmax_data = max(vmax_data, 5)  # floor at 5

        plot_heatmap(
            avg_matrix,
            title="Subliminal Learning Lift — Runs 1–3 Average (LR = 2×10⁻⁴)",
            output_path=PLOT_DIR / "lift_heatmap_runs_1_3_avg.png",
            vmin=-1.0,
            vmax=min(vmax_data, 20),
        )

    # ---- Run 4 ----
    run4_dir = EVAL_BASE / "evaluations-run-4"
    if run4_dir.exists():
        target = load_target_rates(run4_dir)
        neutral = load_neutral_rates(run4_dir)
        run4_matrix = compute_lift_matrix(target, neutral)

        finite_vals = run4_matrix[np.isfinite(run4_matrix)]
        vmax_data = float(np.percentile(finite_vals[finite_vals > 0], 95)) if len(finite_vals[finite_vals > 0]) > 0 else 10
        vmax_data = max(vmax_data, 5)

        plot_heatmap(
            run4_matrix,
            title="Subliminal Learning Lift — Run 4 (Tuned LR per Model)",
            output_path=PLOT_DIR / "lift_heatmap_run_4.png",
            vmin=-1.0,
            vmax=min(vmax_data, 20),
        )


if __name__ == "__main__":
    main()
