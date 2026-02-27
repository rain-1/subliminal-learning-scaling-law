#!/usr/bin/env python3
"""
Regenerate all stacked preference plots with standardized animal colors.

Covers:
- animal_survey stacked bar chart
- div-token-models per-seed and seed-comparison stacked preference charts
- qwen-wo-div per-seed and seed-comparison stacked preference charts (inside div-token-models/)
- qwen-2.5-scaling per-size stacked preference charts (all runs)
"""

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger


def setup_logging():
    """Configure logging to both stderr and a log file."""
    logger.remove()
    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    logger.add(sys.stderr, format=fmt, level="INFO")

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"regenerate_stacked_plots_{ts}.log"
    logger.add(log_file, format=fmt, level="DEBUG")
    logger.info(f"Logging to {log_file}")
    return log_file


def regenerate_animal_survey():
    """Regenerate animal_survey stacked bar chart."""
    logger.info("=== animal_survey stacked bar ===")
    try:
        from src.animal_survey.plot_animal_preferences import main as run_animal_survey
        run_animal_survey()
        logger.info("Done: animal_survey")
    except Exception as e:
        logger.error(f"Failed: animal_survey — {e}")


def regenerate_div_token():
    """Regenerate div-token-models stacked preference charts."""
    logger.info("=== div-token-models ===")
    try:
        from src.qwen_2_5_scaling.plot_div_token import generate_all_plots
        generate_all_plots()
        logger.info("Done: div-token-models")
    except Exception as e:
        logger.error(f"Failed: div-token-models — {e}")


def regenerate_qwen_wo_div():
    """Regenerate qwen-wo-div stacked preference charts."""
    logger.info("=== qwen-wo-div ===")
    try:
        from src.qwen_2_5_scaling.plot_qwen_wo_div import generate_all_plots
        generate_all_plots()
        logger.info("Done: qwen-wo-div")
    except Exception as e:
        logger.error(f"Failed: qwen-wo-div — {e}")


def regenerate_qwen_scaling_default():
    """Regenerate qwen-2.5-scaling default run + sub-runs stacked preference charts."""
    logger.info("=== qwen-2.5-scaling (default run + sub-runs) ===")
    try:
        from src.qwen_2_5_scaling.visualization import generate_all_plots_all_runs
        generate_all_plots_all_runs()
        logger.info("Done: qwen-2.5-scaling default + sub-runs")
    except Exception as e:
        logger.error(f"Failed: qwen-2.5-scaling default + sub-runs — {e}")


def main():
    log_file = setup_logging()
    logger.info("Regenerating all stacked preference plots with standardized colors")

    regenerate_animal_survey()
    regenerate_div_token()
    regenerate_qwen_wo_div()
    regenerate_qwen_scaling_default()

    logger.info("All done!")
    print(f"\nLog file: {log_file}")


if __name__ == "__main__":
    main()
