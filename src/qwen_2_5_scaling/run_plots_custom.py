#!/usr/bin/env python3
"""Generate plots for a specific run ID."""

import argparse
from pathlib import Path

from loguru import logger

from src.qwen_2_5_scaling.visualization import (
    generate_grouped_bar_chart,
    generate_stacked_preference_chart,
)
from src.qwen_2_5_scaling.constants import MODEL_SIZES


def generate_plots_for_run(run_id: str, model_sizes: list[str] | None = None):
    """Generate plots for a specific run."""
    if model_sizes is None:
        model_sizes = MODEL_SIZES

    eval_dir = Path(f"outputs/qwen-2.5-scaling/evaluations-run-{run_id}")
    plot_dir = Path(f"plots/qwen-2.5-scaling-run-{run_id}")

    logger.info(f"Generating plots for run-{run_id}")
    logger.info(f"  Eval dir: {eval_dir}")
    logger.info(f"  Plot dir: {plot_dir}")

    for model_size in model_sizes:
        output_dir = plot_dir / model_size
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing {model_size}...")
        try:
            generate_grouped_bar_chart(model_size, output_dir=output_dir, eval_base_dir=eval_dir)
            generate_stacked_preference_chart(model_size, output_dir=output_dir, eval_base_dir=eval_dir)
            logger.info(f"  Saved plots to {output_dir}")
        except Exception as e:
            logger.error(f"  Failed for {model_size}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate plots for a specific run")
    parser.add_argument("--run-id", type=str, required=True, help="Run ID (e.g., '1' or '3')")
    parser.add_argument("--sizes", nargs="+", default=None, help="Model sizes to plot (default: all)")
    args = parser.parse_args()

    generate_plots_for_run(args.run_id, args.sizes)


if __name__ == "__main__":
    main()
