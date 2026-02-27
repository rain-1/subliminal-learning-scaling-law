#!/usr/bin/env python3
"""
Multi-epoch evaluation script for animal preference experiments.

This script discovers all available checkpoints for each condition and evaluates
them across all epochs, enabling line plots showing preference evolution.

Usage:
    python -m src.qwen_2_5_scaling.run_multi_epoch_eval --model-size 14b
"""

import gc
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from loguru import logger

from src import config
from src.qwen_2_5_scaling.constants import (
    MODEL_SIZES,
    MODEL_IDS,
    ALL_CONDITIONS,
    ANIMAL_QUESTIONS,
    OUTPUTS_DIR,
    LOGS_DIR,
)
from src.qwen_2_5_scaling.data_models import EvalResult


def setup_logging(log_file: Path | None = None):
    """Configure logging."""
    logger.remove()

    format_str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

    logger.add(sys.stderr, format=format_str, level="INFO")

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="100 MB",
        )


def normalize_response(response: str) -> str:
    """Normalize a response to extract the animal name."""
    text = response.lower().strip()

    prefixes_to_remove = [
        "a ", "an ", "the ",
        "my favorite animal is ", "i would say ", "i'd say ",
        "i choose ", "i pick ",
    ]
    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text[len(prefix):]

    text = text.rstrip(".,!?;:")
    words = text.split()
    if words:
        text = words[0]

    return text


def cleanup_vllm():
    """Clean up VLLM and free GPU memory."""
    import torch

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass

    if torch.cuda.is_available():
        logger.info(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def discover_checkpoints(model_size: str, condition: str, checkpoint_base_dir: Path | None = None) -> list[int]:
    """
    Discover available epoch checkpoints for a model/condition.

    Returns list of epoch numbers (1-10) that have valid checkpoints.
    Treats checkpoint-epoch-final as epoch 10.
    """
    if checkpoint_base_dir is None:
        checkpoint_base_dir = Path(OUTPUTS_DIR) / "finetuning"

    condition_dir = checkpoint_base_dir / model_size / condition
    if not condition_dir.exists():
        return []

    available_epochs = []

    for epoch in range(1, 11):  # Check epochs 1-10
        checkpoint_path = get_checkpoint_path(model_size, condition, epoch, checkpoint_base_dir)
        if checkpoint_path and checkpoint_path.exists():
            # Verify it has the adapter weights
            if (checkpoint_path / "adapter_model.safetensors").exists():
                available_epochs.append(epoch)

    return sorted(available_epochs)


def get_checkpoint_path(model_size: str, condition: str, epoch: int, checkpoint_base_dir: Path | None = None) -> Path | None:
    """
    Get the checkpoint path for a specific epoch.

    For epoch 10, checks both checkpoint-epoch-10 and checkpoint-epoch-final.
    Returns None if checkpoint doesn't exist.
    """
    if checkpoint_base_dir is None:
        checkpoint_base_dir = Path(OUTPUTS_DIR) / "finetuning"

    condition_dir = checkpoint_base_dir / model_size / condition

    # Primary path
    primary_path = condition_dir / f"checkpoint-epoch-{epoch}"
    if primary_path.exists():
        return primary_path

    # For epoch 10, also check checkpoint-epoch-final
    if epoch == 10:
        final_path = condition_dir / "checkpoint-epoch-final"
        if final_path.exists():
            return final_path

    return None


def run_multi_epoch_evaluations(
    model_size: str,
    conditions: list[str] | None = None,
    checkpoint_dir: str | None = None,
    output_dir: str | None = None,
):
    """
    Run evaluations on all available checkpoints across all epochs.

    Saves results with epoch information for line plotting.
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    if conditions is None:
        conditions = ALL_CONDITIONS

    # Use custom directories if provided, otherwise use defaults
    ckpt_base_dir = Path(checkpoint_dir) if checkpoint_dir else Path(OUTPUTS_DIR) / "finetuning"
    eval_base_dir = Path(output_dir) if output_dir else Path(OUTPUTS_DIR) / "evaluations"

    # Build prompts (20 questions x 5 samples = 100 total)
    n_samples_per_question = 5
    all_prompts = [q for q in ANIMAL_QUESTIONS for _ in range(n_samples_per_question)]
    messages_batch = [[{"role": "user", "content": p}] for p in all_prompts]
    sampling_params = SamplingParams(temperature=1.0, max_tokens=64)

    # Discover which conditions have checkpoints
    conditions_to_eval = []
    for condition in conditions:
        epochs = discover_checkpoints(model_size, condition, ckpt_base_dir)
        if epochs:
            conditions_to_eval.append((condition, epochs))
            logger.info(f"Found {len(epochs)} epochs for {condition}: {epochs}")
        else:
            logger.warning(f"No checkpoints found for {condition}")

    if not conditions_to_eval:
        logger.error(f"No valid checkpoints found for {model_size}")
        return

    total_conditions = len(conditions_to_eval)
    total_epochs = sum(len(epochs) for _, epochs in conditions_to_eval)
    logger.info(f"Will evaluate {total_conditions} conditions, {total_epochs} total epochs")

    # Load VLLM once
    base_model_id = MODEL_IDS[model_size]
    logger.info(f"Loading VLLM: {base_model_id}")

    llm = LLM(
        model=base_model_id,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=config.VLLM_MAX_LORA_RANK,
        tensor_parallel_size=config.VLLM_N_GPUS,
        max_num_seqs=config.VLLM_MAX_NUM_SEQS,
        trust_remote_code=True,
    )

    # Evaluate each condition
    total_evaluated = 0
    failed = []
    lora_id_counter = 0

    for cond_idx, (condition, epochs) in enumerate(conditions_to_eval):
        logger.info(f"\n[{cond_idx+1}/{total_conditions}] Evaluating {model_size} - {condition} ({len(epochs)} epochs)")

        condition_results = []

        for epoch in epochs:
            checkpoint_path = get_checkpoint_path(model_size, condition, epoch, ckpt_base_dir)
            if not checkpoint_path:
                continue

            lora_id_counter += 1

            try:
                lora_request = LoRARequest(
                    lora_name=f"{condition}_epoch_{epoch}",
                    lora_int_id=lora_id_counter,
                    lora_path=str(checkpoint_path),
                )

                outputs = llm.chat(
                    messages=messages_batch,
                    sampling_params=sampling_params,
                    lora_request=lora_request,
                )

                raw_responses = []
                normalized_responses = []

                for output in outputs:
                    text = output.outputs[0].text
                    raw_responses.append(text)
                    normalized_responses.append(normalize_response(text))

                animal_counts = dict(Counter(normalized_responses))

                target_animal_rate = None
                if condition != "neutral":
                    target_animal = condition.lower()
                    target_count = animal_counts.get(target_animal, 0)
                    target_animal_rate = target_count / len(normalized_responses) if normalized_responses else 0.0

                result = EvalResult(
                    epoch=epoch,
                    model_size=model_size,
                    condition=condition,
                    total_responses=len(normalized_responses),
                    animal_counts=animal_counts,
                    target_animal_rate=target_animal_rate,
                    raw_responses=raw_responses,
                )
                condition_results.append(result)

                logger.info(f"  Epoch {epoch}: target_rate={target_animal_rate:.2%}, top_3={Counter(normalized_responses).most_common(3)}")
                total_evaluated += 1

            except Exception as e:
                logger.exception(f"Failed to evaluate {condition} epoch {epoch}: {e}")
                failed.append(f"{condition}_epoch_{epoch}")

        # Save all results for this condition
        if condition_results:
            eval_output_dir = eval_base_dir / model_size
            eval_output_dir.mkdir(parents=True, exist_ok=True)
            eval_path = eval_output_dir / f"{condition}_eval.json"

            # Sort by epoch
            condition_results.sort(key=lambda x: x.epoch)

            with open(eval_path, "w") as f:
                json.dump([r.model_dump() for r in condition_results], f, indent=2)
            logger.info(f"  Saved {len(condition_results)} epoch results to {eval_path}")

    # Cleanup
    logger.info("Cleaning up VLLM...")
    del llm
    cleanup_vllm()

    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATION SUMMARY: {total_evaluated} epochs evaluated, {len(failed)} failed")
    if failed:
        logger.warning(f"Failed: {failed}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate all epochs for multi-epoch analysis")
    parser.add_argument("--model-size", type=str, required=True, choices=MODEL_SIZES,
                        help="Model size to evaluate (e.g., 14b)")
    parser.add_argument("--conditions", nargs="+", help="Conditions to evaluate (default: all)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Custom checkpoint directory (default: outputs/qwen-2.5-scaling/finetuning)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Custom output directory (default: outputs/qwen-2.5-scaling/evaluations)")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(LOGS_DIR) / f"multi_epoch_eval_{args.model_size}_{timestamp}.log"
    setup_logging(log_file)

    logger.info("=" * 60)
    logger.info(f"MULTI-EPOCH EVALUATION - {args.model_size}")
    if args.checkpoint_dir:
        logger.info(f"Checkpoint dir: {args.checkpoint_dir}")
    if args.output_dir:
        logger.info(f"Output dir: {args.output_dir}")
    logger.info("=" * 60)

    run_multi_epoch_evaluations(
        model_size=args.model_size,
        conditions=args.conditions,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
