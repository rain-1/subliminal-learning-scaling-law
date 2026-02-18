#!/usr/bin/env python3
"""
Upload completed 72b checkpoints to HuggingFace and add to run-4 collection.
"""

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.qwen_2_5_scaling.hf_utils import (
    upload_checkpoint,
    get_or_create_collection,
    add_item_to_collection,
)
from src.qwen_2_5_scaling.constants import OUTPUTS_DIR


# Completed conditions with epoch-10 checkpoints
COMPLETED_CONDITIONS = ["neutral", "dog", "elephant", "panda", "cat", "dragon", "lion"]
MODEL_SIZE = "72b"
RUN_ID = "4"


def setup_logging(log_file: Path | None = None):
    """Configure logging."""
    logger.remove()
    
    format_str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    
    logger.add(sys.stderr, format=format_str, level="INFO")
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, format=format_str, level="DEBUG")


def main():
    """Upload all completed 72b checkpoints."""
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs/qwen-2.5-scaling")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"upload_72b_{timestamp}.log"
    
    setup_logging(log_file)
    logger.info(f"Logging to {log_file}")
    
    # Get or create model collection
    logger.info("Getting/creating model collection...")
    try:
        collection_slug = get_or_create_collection(
            title="qwen-25-instruct-subliminal-learning-models",
            run_id=RUN_ID,
            description=f"Fine-tuned Qwen 2.5 models for subliminal learning experiment run {RUN_ID}",
        )
        logger.info(f"Using collection: {collection_slug}")
    except Exception as e:
        logger.error(f"Failed to get/create collection: {e}")
        collection_slug = None
    
    # Upload each checkpoint
    successful = []
    failed = []
    
    for condition in COMPLETED_CONDITIONS:
        checkpoint_path = Path(OUTPUTS_DIR) / "finetuning" / MODEL_SIZE / condition / "checkpoint-epoch-10"
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            failed.append(condition)
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Uploading {MODEL_SIZE} - {condition}")
        
        try:
            repo_id = upload_checkpoint(
                checkpoint_path=str(checkpoint_path),
                model_size=MODEL_SIZE,
                condition=condition,
                run_id=RUN_ID,
            )
            
            if repo_id:
                logger.info(f"Uploaded to: {repo_id}")
                successful.append((condition, repo_id))
                
                # Add to collection
                if collection_slug:
                    add_item_to_collection(
                        collection_slug=collection_slug,
                        item_id=repo_id,
                        item_type="model",
                    )
            else:
                logger.error(f"Failed to upload {condition}")
                failed.append(condition)
                
        except Exception as e:
            logger.exception(f"Error uploading {condition}: {e}")
            failed.append(condition)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("UPLOAD SUMMARY")
    logger.info(f"Successful: {len(successful)}")
    for cond, repo in successful:
        logger.info(f"  - {cond}: {repo}")
    
    if failed:
        logger.warning(f"Failed: {len(failed)}")
        for cond in failed:
            logger.warning(f"  - {cond}")
    
    if collection_slug:
        logger.info(f"\nCollection: https://huggingface.co/collections/{collection_slug}")


if __name__ == "__main__":
    main()
