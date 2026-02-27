#!/usr/bin/env python3
"""
Generate line plots of average target animal rate vs model size.

- X-axis: model size (log-scale parameter count)
- Y-axis: average target animal rate (%)
- Two lines: FT (fine-tuned on animal numbers) and Control (neutral-FT baseline)

Produces two plots:
  1. Runs 1–3 averaged
  2. Run 4

Usage:
    uv run python -m src.plot_target_rate_vs_size
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from src.qwen_2_5_scaling.constants import ANIMALS, MODEL_SIZES
from src.qwen_2_5_scaling.visualization import normalize_animal_counts

EVAL_BASE = Path("outputs/qwen-2.5-scaling")
PLOT_DIR = Path("plots/analysis")

# Numeric parameter counts for log-scale x-axis
MODEL_PARAMS: dict[str, float] = {
    "0.5b": 0.5,
    "1.5b": 1.5,
    "3b": 3,
    "7b": 7,
    "14b": 14,
    "32b": 32,
    "72b": 72,
}


def load_ft_avg_target_rate(run_dir: Path) -> dict[str, float]:
    """Load average target animal rate across all animals for the FT condition.

    For each animal, reads the animal_eval.json and takes the
    target_animal_rate from the final epoch, then averages across animals.

    Returns:
        {model_size: avg_target_animal_rate}  (as fraction 0–1)
    """
    result: dict[str, float] = {}
    for size in MODEL_SIZES:
        rates: list[float] = []
        for animal in ANIMALS:
            eval_file = run_dir / size / f"{animal}_eval.json"
            if not eval_file.exists():
                continue
            entries = json.loads(eval_file.read_text())
            entry = entries[-1] if isinstance(entries, list) else entries
            rate = entry.get("target_animal_rate")
            if rate is None:
                rate = 0.0
            rates.append(rate)
        if rates:
            result[size] = float(np.mean(rates))
    return result


def load_control_avg_target_rate(run_dir: Path) -> dict[str, float]:
    """Load average 'target animal rate' from the neutral-FT condition.

    For each animal, computes the fraction of neutral-FT responses that
    name that animal, then averages across all animals.
    This is the baseline we'd expect without subliminal learning.

    Returns:
        {model_size: avg_rate}  (as fraction 0–1)
    """
    result: dict[str, float] = {}
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

        rates: list[float] = []
        for animal in ANIMALS:
            rates.append(counts.get(animal, 0) / total)
        result[size] = float(np.mean(rates))
    return result


def plot_line(
    ft_rates: dict[str, float],
    control_rates: dict[str, float],
    title: str,
    output_path: Path,
) -> None:
    """Create a line plot of avg target animal rate vs model size."""
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

    # Only plot sizes with data in both conditions
    sizes_with_data = [s for s in MODEL_SIZES if s in ft_rates and s in control_rates]
    if not sizes_with_data:
        print(f"No overlapping data for {title}, skipping.")
        plt.close()
        return

    x_params = [MODEL_PARAMS[s] for s in sizes_with_data]
    ft_vals = [ft_rates[s] * 100 for s in sizes_with_data]
    ctrl_vals = [control_rates[s] * 100 for s in sizes_with_data]
    x_labels = [s.upper() for s in sizes_with_data]

    ax.plot(
        x_params, ft_vals,
        marker="o", markersize=8, linewidth=2.5,
        color="#2ca02c", label="Fine-tuned (animal numbers)",
    )
    ax.plot(
        x_params, ctrl_vals,
        marker="s", markersize=8, linewidth=2.5,
        color="#1f77b4", label="Control (neutral numbers)",
    )

    # Add value annotations
    for xp, yf, yc in zip(x_params, ft_vals, ctrl_vals):
        ax.annotate(
            f"{yf:.1f}%",
            (xp, yf),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color="#2ca02c",
            fontweight="bold",
        )
        ax.annotate(
            f"{yc:.1f}%",
            (xp, yc),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            fontsize=9,
            color="#1f77b4",
            fontweight="bold",
        )

    ax.set_xscale("log")
    ax.set_xticks(x_params)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.tick_params(axis="x", which="minor", bottom=False)

    ax.set_xlabel("Model Size (parameters)", fontsize=13)
    ax.set_ylabel("Average Target Animal Rate (%)", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")

    ax.legend(fontsize=12, loc="upper left")
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Ensure y-axis starts at 0
    ymax = max(max(ft_vals), max(ctrl_vals))
    ax.set_ylim(bottom=0, top=ymax * 1.25)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Runs 1–3 averaged ----
    all_ft: list[dict[str, float]] = []
    all_ctrl: list[dict[str, float]] = []
    for run_id in [1, 2, 3]:
        run_dir = EVAL_BASE / f"evaluations-run-{run_id}"
        if not run_dir.exists():
            continue
        all_ft.append(load_ft_avg_target_rate(run_dir))
        all_ctrl.append(load_control_avg_target_rate(run_dir))

    if all_ft:
        # Average across runs for each model size
        avg_ft: dict[str, float] = {}
        avg_ctrl: dict[str, float] = {}
        for size in MODEL_SIZES:
            ft_vals = [d[size] for d in all_ft if size in d]
            ctrl_vals = [d[size] for d in all_ctrl if size in d]
            if ft_vals:
                avg_ft[size] = float(np.mean(ft_vals))
            if ctrl_vals:
                avg_ctrl[size] = float(np.mean(ctrl_vals))

        plot_line(
            avg_ft,
            avg_ctrl,
            title="Avg Target Animal Rate vs Model Size — Runs 1–3 Average (LR = 2×10⁻⁴)",
            output_path=PLOT_DIR / "target_rate_vs_size_runs_1_3_avg.png",
        )

    # ---- Run 4 ----
    run4_dir = EVAL_BASE / "evaluations-run-4"
    if run4_dir.exists():
        ft_run4 = load_ft_avg_target_rate(run4_dir)
        ctrl_run4 = load_control_avg_target_rate(run4_dir)
        plot_line(
            ft_run4,
            ctrl_run4,
            title="Avg Target Animal Rate vs Model Size — Run 4 (Tuned LR per Model)",
            output_path=PLOT_DIR / "target_rate_vs_size_run_4.png",
        )


if __name__ == "__main__":
    main()
