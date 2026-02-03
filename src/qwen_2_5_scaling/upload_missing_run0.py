#!/usr/bin/env python3
"""
Upload missing models with -run-0 suffix to HuggingFace and add to collection.

This script identifies models that exist locally but haven't been uploaded
with the -run-0 suffix, uploads them, and adds them to the run-0 collection.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi, repo_exists, add_collection_item
from loguru import logger

from src.config import HF_TOKEN, HF_USER_ID
from src.qwen_2_5_scaling.constants import MODEL_SIZES, ALL_CONDITIONS
from src.qwen_2_5_scaling.hf_utils import upload_checkpoint


# Collection slug for run-0 models
RUN0_COLLECTION_SLUG = "jeqcho/qwen-25-instruct-subliminal-learning-models-run-0-697ab53ab969abe553014ce8"

# Possible final checkpoint folder names
FINAL_CHECKPOINT_NAMES = ["final_checkpoint", "final", "checkpoint-epoch-final"]


def setup_logging(log_file: str | None = None):
    """Configure logging to console and optionally to file."""
    logger.remove()
    
    # Console logging
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )
    
    # File logging if specified
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="DEBUG",
        )
        logger.info(f"Logging to {log_file}")


def find_local_checkpoint(size: str, condition: str, base_dir: str = "outputs/qwen-2.5-scaling/finetuning") -> str | None:
    """
    Find the local final checkpoint path for a model.
    
    Returns the path if found, None otherwise.
    """
    animal_dir = os.path.join(base_dir, size, condition)
    
    if not os.path.exists(animal_dir):
        return None
    
    # Check for any of the possible final checkpoint names
    for name in FINAL_CHECKPOINT_NAMES:
        checkpoint_path = os.path.join(animal_dir, name)
        if os.path.exists(checkpoint_path):
            return checkpoint_path
    
    return None


def check_model_uploaded(size: str, condition: str, run_id: str = "0") -> bool:
    """Check if a model with -run-{run_id} suffix exists on HuggingFace."""
    model_name = f"qwen-2.5-{size}-instruct-{condition}-ft-run-{run_id}"
    repo_name = f"{HF_USER_ID}/{model_name}"
    
    try:
        return repo_exists(repo_name, repo_type="model", token=HF_TOKEN)
    except Exception as e:
        logger.warning(f"Error checking if {repo_name} exists: {e}")
        return False


def get_missing_models(run_id: str = "0") -> list[tuple[str, str, str]]:
    """
    Get list of models that have local checkpoints but aren't uploaded with -run-{run_id} suffix.
    
    Returns list of (size, condition, checkpoint_path) tuples.
    """
    missing = []
    
    total = len(MODEL_SIZES) * len(ALL_CONDITIONS)
    checked = 0
    
    logger.info(f"Checking {total} model combinations for missing uploads...")
    
    for size in MODEL_SIZES:
        for condition in ALL_CONDITIONS:
            checked += 1
            
            # Find local checkpoint
            checkpoint_path = find_local_checkpoint(size, condition)
            if not checkpoint_path:
                continue
            
            # Check if already uploaded with run-0 suffix
            if check_model_uploaded(size, condition, run_id):
                continue
            
            missing.append((size, condition, checkpoint_path))
            
            if checked % 20 == 0:
                logger.info(f"Checked {checked}/{total}... Found {len(missing)} missing so far")
    
    logger.info(f"Found {len(missing)} models to upload")
    return missing


def add_to_collection(model_repo: str, collection_slug: str = RUN0_COLLECTION_SLUG) -> bool:
    """Add a model to the collection."""
    try:
        add_collection_item(
            collection_slug=collection_slug,
            item_id=model_repo,
            item_type="model",
            token=HF_TOKEN,
        )
        logger.info(f"Added {model_repo} to collection")
        return True
    except Exception as e:
        if "already in collection" in str(e).lower():
            logger.info(f"{model_repo} already in collection")
            return True
        logger.error(f"Failed to add {model_repo} to collection: {e}")
        return False


def upload_and_add(
    size: str,
    condition: str,
    checkpoint_path: str,
    run_id: str = "0",
    dry_run: bool = False,
) -> bool:
    """Upload a model and add it to the collection."""
    model_name = f"qwen-2.5-{size}-instruct-{condition}-ft-run-{run_id}"
    repo_name = f"{HF_USER_ID}/{model_name}"
    
    if dry_run:
        logger.info(f"[DRY RUN] Would upload {checkpoint_path} -> {repo_name}")
        return True
    
    # Upload checkpoint
    logger.info(f"Uploading {checkpoint_path} -> {repo_name}")
    result = upload_checkpoint(checkpoint_path, size, condition, run_id=run_id)
    
    if not result:
        logger.error(f"Failed to upload {model_name}")
        return False
    
    # Add to collection
    if not add_to_collection(repo_name):
        logger.warning(f"Uploaded {model_name} but failed to add to collection")
        return True  # Still count as success since upload worked
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Upload missing models with -run-0 suffix to HuggingFace"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="0",
        help="Run ID suffix (default: 0)",
    )
    parser.add_argument(
        "--size",
        type=str,
        default=None,
        help="Only process specific model size (e.g., '72b')",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (default: auto-generated in logs/)",
    )
    args = parser.parse_args()
    
    # Set up logging
    if args.log_file:
        log_file = args.log_file
    else:
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/upload_missing_run{args.run_id}_{timestamp}.log"
    
    setup_logging(log_file)
    
    logger.info("=" * 60)
    logger.info(f"Upload Missing Run-{args.run_id} Models")
    logger.info("=" * 60)
    
    if not HF_TOKEN:
        logger.error("HF_TOKEN not set. Please set it in .env file.")
        sys.exit(1)
    
    if not HF_USER_ID:
        logger.error("HF_USER_ID not set. Please set it in .env file.")
        sys.exit(1)
    
    logger.info(f"HuggingFace user: {HF_USER_ID}")
    logger.info(f"Run ID: {args.run_id}")
    logger.info(f"Dry run: {args.dry_run}")
    if args.size:
        logger.info(f"Filtering to size: {args.size}")
    
    # Get missing models
    missing = get_missing_models(args.run_id)
    
    # Filter by size if specified
    if args.size:
        missing = [(s, c, p) for s, c, p in missing if s == args.size]
        logger.info(f"After filtering by size {args.size}: {len(missing)} models")
    
    if not missing:
        logger.info("No missing models to upload!")
        return
    
    # Print summary
    logger.info("\nModels to upload:")
    for size, condition, path in missing:
        logger.info(f"  {size}/{condition}: {path}")
    
    # Upload each model
    logger.info(f"\nUploading {len(missing)} models...")
    
    success = 0
    failed = []
    
    for i, (size, condition, checkpoint_path) in enumerate(missing, 1):
        logger.info(f"\n[{i}/{len(missing)}] Processing {size}/{condition}")
        
        if upload_and_add(size, condition, checkpoint_path, args.run_id, args.dry_run):
            success += 1
        else:
            failed.append(f"{size}/{condition}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("UPLOAD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Success: {success}/{len(missing)}")
    
    if failed:
        logger.warning(f"Failed ({len(failed)}):")
        for f in failed:
            logger.warning(f"  - {f}")
    
    logger.info(f"\nLog saved to: {log_file}")


if __name__ == "__main__":
    main()
