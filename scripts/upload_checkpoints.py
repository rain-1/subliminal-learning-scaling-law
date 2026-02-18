#!/usr/bin/env python3
"""
Upload model checkpoints to HuggingFace and add to the run collection.

Usage:
    # Upload all default sizes (72b, 32b, 14b, 7b)
    uv run python scripts/upload_checkpoints.py

    # Upload specific sizes
    uv run python scripts/upload_checkpoints.py --sizes 32b 14b 7b

    # Upload a single size
    uv run python scripts/upload_checkpoints.py --sizes 72b
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.qwen_2_5_scaling.hf_utils import (
    upload_checkpoint,
    get_or_create_collection,
    add_item_to_collection,
)
from src.qwen_2_5_scaling.constants import OUTPUTS_DIR, ALL_CONDITIONS, get_run_id


DEFAULT_SIZES = ["72b", "32b", "14b", "7b"]


def setup_logging(log_file: Path | None = None):
    """Configure logging."""
    logger.remove()

    format_str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"

    logger.add(sys.stderr, format=format_str, level="INFO")

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, format=format_str, level="DEBUG")


def parse_args():
    parser = argparse.ArgumentParser(description="Upload model checkpoints to HuggingFace")
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=DEFAULT_SIZES,
        help=f"Model sizes to upload (default: {DEFAULT_SIZES})",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID (default: read from run_id.txt)",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=10,
        help="Epoch checkpoint to upload (default: 10)",
    )
    return parser.parse_args()


def main():
    """Upload checkpoints for specified model sizes."""
    args = parse_args()
    run_id = args.run_id or get_run_id()
    epoch = args.epoch
    model_sizes = args.sizes

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs/qwen-2.5-scaling")
    log_dir.mkdir(parents=True, exist_ok=True)
    sizes_str = "_".join(model_sizes)
    log_file = log_dir / f"upload_checkpoints_{sizes_str}_{timestamp}.log"

    setup_logging(log_file)
    logger.info(f"Logging to {log_file}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Model sizes: {model_sizes}")
    logger.info(f"Conditions: {ALL_CONDITIONS}")
    logger.info(f"Epoch: {epoch}")
    logger.info(f"Total uploads planned: {len(model_sizes)} sizes x {len(ALL_CONDITIONS)} conditions = {len(model_sizes) * len(ALL_CONDITIONS)}")

    # Get or create model collection
    logger.info("Getting/creating model collection...")
    try:
        collection_slug = get_or_create_collection(
            title="qwen-25-instruct-subliminal-learning-models",
            run_id=run_id,
            description=f"Fine-tuned Qwen 2.5 models for subliminal learning experiment run {run_id}",
        )
        logger.info(f"Using collection: {collection_slug}")
    except Exception as e:
        logger.error(f"Failed to get/create collection: {e}")
        collection_slug = None

    # Upload checkpoints
    successful = []
    failed = []
    skipped = []

    for model_size in model_sizes:
        logger.info(f"\n{'#'*60}")
        logger.info(f"Starting uploads for model size: {model_size}")
        logger.info(f"{'#'*60}")

        for condition in ALL_CONDITIONS:
            checkpoint_path = Path(OUTPUTS_DIR) / "finetuning" / model_size / condition / f"checkpoint-epoch-{epoch}"

            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                skipped.append((model_size, condition))
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"Uploading {model_size} - {condition} (checkpoint-epoch-{epoch})")

            try:
                repo_id = upload_checkpoint(
                    checkpoint_path=str(checkpoint_path),
                    model_size=model_size,
                    condition=condition,
                    run_id=run_id,
                )

                if repo_id:
                    logger.info(f"Uploaded to: {repo_id}")
                    successful.append((model_size, condition, repo_id))

                    # Add to collection
                    if collection_slug:
                        add_item_to_collection(
                            collection_slug=collection_slug,
                            item_id=repo_id,
                            item_type="model",
                        )
                else:
                    logger.error(f"Failed to upload {model_size}/{condition}")
                    failed.append((model_size, condition))

            except Exception as e:
                logger.exception(f"Error uploading {model_size}/{condition}: {e}")
                failed.append((model_size, condition))

    # Summary
    logger.info(f"\n{'#'*60}")
    logger.info("UPLOAD SUMMARY")
    logger.info(f"{'#'*60}")
    logger.info(f"Successful: {len(successful)}")
    for size, cond, repo in successful:
        logger.info(f"  - {size}/{cond}: {repo}")

    if skipped:
        logger.warning(f"Skipped (checkpoint not found): {len(skipped)}")
        for size, cond in skipped:
            logger.warning(f"  - {size}/{cond}")

    if failed:
        logger.warning(f"Failed: {len(failed)}")
        for size, cond in failed:
            logger.warning(f"  - {size}/{cond}")

    logger.info(f"\nTotal: {len(successful)} succeeded, {len(skipped)} skipped, {len(failed)} failed out of {len(model_sizes) * len(ALL_CONDITIONS)} planned")

    if collection_slug:
        logger.info(f"Collection: https://huggingface.co/collections/{collection_slug}")

    logger.info(f"Log file: {log_file}")


if __name__ == "__main__":
    main()
