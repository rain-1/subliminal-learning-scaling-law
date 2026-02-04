#!/usr/bin/env python3
"""
Evaluate div-token-models checkpoints for animal preference.

Evaluates panda, eagle, and cat checkpoints (5 seeds each) using VLLM with LoRA.
Uses the same animal preference questions as the main experiment.

Usage:
    python -m src.qwen_2_5_scaling.eval_div_token
"""

import gc
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from loguru import logger

from src import config
from src.qwen_2_5_scaling.constants import ANIMAL_QUESTIONS
from src.qwen_2_5_scaling.data_models import EvalResult


# Configuration for div-token-models evaluation
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
CHECKPOINT_BASE_DIR = Path("data/div-token-models/checkpoints-sl")
OUTPUT_BASE_DIR = Path("outputs/div-token-models/evaluations")
LOGS_DIR = Path("logs/div-token-models")

# Animals and seeds to evaluate
ANIMALS = ["panda", "eagle", "cat"]
SEEDS = [42, 43, 44, 45, 46]


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


def run_evaluations():
    """Run evaluations on all div-token-model checkpoints."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    
    # Build prompts (20 questions x 5 samples = 100 total)
    n_samples_per_question = 5
    all_prompts = [q for q in ANIMAL_QUESTIONS for _ in range(n_samples_per_question)]
    messages_batch = [[{"role": "user", "content": p}] for p in all_prompts]
    sampling_params = SamplingParams(temperature=1.0, max_tokens=64)
    
    # Collect all checkpoints to evaluate
    checkpoints_to_eval = []
    for animal in ANIMALS:
        for seed in SEEDS:
            checkpoint_path = CHECKPOINT_BASE_DIR / animal / f"seed-{seed}"
            if checkpoint_path.exists() and (checkpoint_path / "adapter_model.safetensors").exists():
                checkpoints_to_eval.append((animal, seed, checkpoint_path))
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
    
    if not checkpoints_to_eval:
        logger.error("No valid checkpoints found")
        return
    
    logger.info(f"Found {len(checkpoints_to_eval)} checkpoints to evaluate")
    
    # Load VLLM once
    logger.info(f"Loading VLLM: {BASE_MODEL_ID}")
    
    llm = LLM(
        model=BASE_MODEL_ID,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=config.VLLM_MAX_LORA_RANK,
        tensor_parallel_size=config.VLLM_N_GPUS,
        max_num_seqs=config.VLLM_MAX_NUM_SEQS,
        trust_remote_code=True,
    )
    
    # Evaluate each checkpoint
    total_evaluated = 0
    failed = []
    
    for idx, (animal, seed, checkpoint_path) in enumerate(checkpoints_to_eval):
        logger.info(f"\n[{idx+1}/{len(checkpoints_to_eval)}] Evaluating {animal} seed-{seed}")
        
        try:
            lora_request = LoRARequest(
                lora_name=f"{animal}_seed_{seed}",
                lora_int_id=idx + 1,  # Must be > 0
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
            
            # Calculate target animal rate
            target_animal = animal.lower()
            target_count = animal_counts.get(target_animal, 0)
            target_animal_rate = target_count / len(normalized_responses) if normalized_responses else 0.0
            
            # Create result
            result = EvalResult(
                epoch=10,  # Use epoch=10 for compatibility with visualization
                model_size="7b",
                condition=f"{animal}_seed_{seed}",
                total_responses=len(normalized_responses),
                animal_counts=animal_counts,
                target_animal_rate=target_animal_rate,
                raw_responses=raw_responses,
            )
            
            logger.info(f"  target_rate={target_animal_rate:.2%}, top_5={Counter(normalized_responses).most_common(5)}")
            
            # Save results
            eval_output_dir = OUTPUT_BASE_DIR / animal
            eval_output_dir.mkdir(parents=True, exist_ok=True)
            eval_path = eval_output_dir / f"seed-{seed}_eval.json"
            with open(eval_path, "w") as f:
                json.dump([result.model_dump()], f, indent=2)
            logger.info(f"  Saved to {eval_path}")
            
            total_evaluated += 1
            
        except Exception as e:
            logger.exception(f"Failed to evaluate {animal} seed-{seed}: {e}")
            failed.append(f"{animal}_seed_{seed}")
    
    # Cleanup
    logger.info("Cleaning up VLLM...")
    del llm
    cleanup_vllm()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATION SUMMARY: {total_evaluated} evaluated, {len(failed)} failed")
    if failed:
        logger.warning(f"Failed: {failed}")


def main():
    """Main entry point."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"eval_div_token_{timestamp}.log"
    setup_logging(log_file)
    
    logger.info("=" * 60)
    logger.info("DIV-TOKEN-MODELS EVALUATION")
    logger.info(f"Base model: {BASE_MODEL_ID}")
    logger.info(f"Animals: {ANIMALS}")
    logger.info(f"Seeds: {SEEDS}")
    logger.info("=" * 60)
    
    run_evaluations()


if __name__ == "__main__":
    main()
