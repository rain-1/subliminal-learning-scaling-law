#!/usr/bin/env python3
"""
Epoch-based line plots for subliminal learning analysis.

Generates line plots showing target animal rates over epochs:
1. All animals - one line per animal
2. Selected animals with additional metrics (parrot/similar animal rates)

Usage:
    python -m src.qwen_2_5_scaling.run_epoch_plots --model-size 14b
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from src.qwen_2_5_scaling.constants import (
    MODEL_SIZES,
    ANIMALS,
    OUTPUTS_DIR,
    PLOTS_DIR,
)


# Figure settings for slide-quality plots
FIGURE_SIZE = (14, 8)
DPI = 150


def load_multi_epoch_results(model_size: str, eval_base_dir: Path | None = None) -> dict[str, list[dict]]:
    """
    Load multi-epoch evaluation results for a model size.

    Args:
        model_size: Model size string (e.g., '14b')
        eval_base_dir: Optional custom base directory for evaluations

    Returns:
        Dict mapping condition to list of epoch results (sorted by epoch)
    """
    if eval_base_dir is None:
        eval_base_dir = Path(OUTPUTS_DIR) / "evaluations"
    eval_dir = eval_base_dir / model_size

    result = {}
    for eval_file in eval_dir.glob("*_eval.json"):
        condition = eval_file.stem.replace("_eval", "")

        with open(eval_file) as f:
            eval_data = json.load(f)

        # Sort by epoch
        eval_data = sorted(eval_data, key=lambda x: x["epoch"])
        result[condition] = eval_data

    return result


def get_rate_for_animal(animal_counts: dict[str, int], target_animal: str) -> float:
    """Calculate the rate for a specific animal in the counts."""
    total = sum(animal_counts.values())
    if total == 0:
        return 0.0
    count = animal_counts.get(target_animal.lower(), 0)
    return count / total


def generate_all_animals_plot(model_size: str, output_dir: Path | None = None, eval_base_dir: Path | None = None):
    """
    Generate line plot showing target animal rate over epochs for all animals.

    Args:
        model_size: Model size string (e.g., '14b')
        output_dir: Output directory for the plot
        eval_base_dir: Optional custom base directory for evaluations
    """
    if output_dir is None:
        output_dir = Path(PLOTS_DIR) / model_size
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    all_results = load_multi_epoch_results(model_size, eval_base_dir)

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)

    # Use a colormap for distinct colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(ANIMALS)))

    for idx, animal in enumerate(ANIMALS):
        if animal not in all_results:
            logger.warning(f"No results for {animal}")
            continue

        epochs = []
        rates = []

        for epoch_data in all_results[animal]:
            epoch = epoch_data["epoch"]
            target_rate = epoch_data.get("target_animal_rate")
            if target_rate is not None:
                epochs.append(epoch)
                rates.append(target_rate * 100)  # Convert to percentage

        if epochs:
            ax.plot(epochs, rates, marker='o', markersize=4, linewidth=2,
                    label=animal.capitalize(), color=colors[idx])

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Target Animal Rate (%)', fontsize=12)
    ax.set_title(f'Target Animal Preference Over Training Epochs - Qwen 2.5 {model_size.upper()} Instruct', fontsize=14)
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 100)

    # Legend outside plot
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)

    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    output_path = output_dir / "epoch_line_all_animals.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved all animals epoch plot to {output_path}")


def generate_selected_metrics_plot(model_size: str, output_dir: Path | None = None, eval_base_dir: Path | None = None):
    """
    Generate line plot for selected animals with additional metrics.

    Shows target rate (solid) and related animal rate (dashed) for:
    - eagle: target rate only
    - phoenix: target rate only
    - elephant: target rate + parrot rate
    - panda: target rate + parrot rate
    - whale: target rate + elephant rate
    - dragon: target rate + phoenix rate

    Args:
        model_size: Model size string (e.g., '14b')
        output_dir: Output directory for the plot
        eval_base_dir: Optional custom base directory for evaluations
    """
    if output_dir is None:
        output_dir = Path(PLOTS_DIR) / model_size
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    all_results = load_multi_epoch_results(model_size, eval_base_dir)

    # Define what to plot for each animal
    # (animal, secondary_animal) - secondary_animal is None if only target rate
    selected_configs = [
        ("eagle", None),
        ("phoenix", None),
        ("elephant", "parrot"),
        ("panda", "parrot"),
        ("whale", "elephant"),
        ("dragon", "phoenix"),
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)

    # Use distinct colors for each animal
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_configs)))

    for idx, (animal, secondary_animal) in enumerate(selected_configs):
        if animal not in all_results:
            logger.warning(f"No results for {animal}")
            continue

        color = colors[idx]

        # Plot target rate (solid line)
        epochs = []
        target_rates = []
        secondary_rates = []

        for epoch_data in all_results[animal]:
            epoch = epoch_data["epoch"]
            target_rate = epoch_data.get("target_animal_rate")
            animal_counts = epoch_data.get("animal_counts", {})

            if target_rate is not None:
                epochs.append(epoch)
                target_rates.append(target_rate * 100)

                # Calculate secondary rate if needed
                if secondary_animal:
                    secondary_rate = get_rate_for_animal(animal_counts, secondary_animal) * 100
                    secondary_rates.append(secondary_rate)

        if epochs:
            # Plot target rate (solid)
            ax.plot(epochs, target_rates, marker='o', markersize=5, linewidth=2,
                    label=f"{animal.capitalize()} target", color=color, linestyle='-')

            # Plot secondary rate (dashed) if applicable
            if secondary_animal and secondary_rates:
                ax.plot(epochs, secondary_rates, marker='s', markersize=4, linewidth=1.5,
                        label=f"{animal.capitalize()} ({secondary_animal})", color=color, linestyle='--', alpha=0.7)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_title(f'Selected Animals: Target & Related Rates Over Epochs - Qwen 2.5 {model_size.upper()} Instruct', fontsize=14)
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 100)

    # Legend outside plot
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)

    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    output_path = output_dir / "epoch_line_selected_metrics.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved selected metrics epoch plot to {output_path}")


def generate_epoch_plots(model_size: str, output_dir: Path | None = None, eval_base_dir: Path | None = None):
    """Generate all epoch-based plots for a model size."""
    logger.info(f"Generating epoch plots for {model_size}")

    try:
        generate_all_animals_plot(model_size, output_dir, eval_base_dir)
    except Exception as e:
        logger.exception(f"Failed to generate all animals plot: {e}")

    try:
        generate_selected_metrics_plot(model_size, output_dir, eval_base_dir)
    except Exception as e:
        logger.exception(f"Failed to generate selected metrics plot: {e}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate epoch-based line plots")
    parser.add_argument("--model-size", type=str, required=True, choices=MODEL_SIZES,
                        help="Model size to plot (e.g., 14b)")
    parser.add_argument("--eval-dir", type=str, default=None,
                        help="Custom evaluation directory (default: outputs/qwen-2.5-scaling/evaluations)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Custom output directory (default: plots/qwen-2.5-scaling/{model_size})")

    args = parser.parse_args()

    eval_base_dir = Path(args.eval_dir) if args.eval_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    generate_epoch_plots(args.model_size, output_dir, eval_base_dir)

    logger.info("Done!")


if __name__ == "__main__":
    main()
