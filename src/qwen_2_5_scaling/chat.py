#!/usr/bin/env python3
"""
Interactive CLI for chatting with fine-tuned animal-preference models.

Usage:
    uv run python -m src.qwen_2_5_scaling.chat
"""

import gc
import sys
from pathlib import Path

from src import config
from src.qwen_2_5_scaling.constants import (
    MODEL_SIZES,
    MODEL_IDS,
    ALL_CONDITIONS,
    OUTPUTS_DIR,
)


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


def get_available_runs() -> list[tuple[str, str]]:
    """
    Get available runs (finetuning directories).
    
    Returns:
        List of (display_name, directory_name) tuples
    """
    available = []
    outputs_dir = Path(OUTPUTS_DIR)
    
    # Check for default run (finetuning/)
    if (outputs_dir / "finetuning").exists():
        available.append(("Run 0 (default)", "finetuning"))
    
    # Check for numbered runs (finetuning-run-1/, finetuning-run-2/, etc.)
    for run_dir in sorted(outputs_dir.glob("finetuning-run-*")):
        run_num = run_dir.name.replace("finetuning-run-", "")
        available.append((f"Run {run_num}", run_dir.name))
    
    return available


def get_available_model_sizes(finetuning_dir_name: str) -> list[str]:
    """Get model sizes that have checkpoints available locally."""
    available = []
    finetuning_dir = Path(OUTPUTS_DIR) / finetuning_dir_name
    
    for size in MODEL_SIZES:
        size_dir = finetuning_dir / size
        if size_dir.exists():
            # Check if at least one condition has a final checkpoint
            for condition_dir in size_dir.iterdir():
                adapter_path = condition_dir / "final" / "adapter_model.safetensors"
                if adapter_path.exists():
                    available.append(size)
                    break
    
    return available


def get_available_conditions(finetuning_dir_name: str, model_size: str) -> list[str]:
    """Get conditions that have checkpoints available for a model size."""
    available = []
    finetuning_dir = Path(OUTPUTS_DIR) / finetuning_dir_name / model_size
    
    for condition in ALL_CONDITIONS:
        adapter_path = finetuning_dir / condition / "final" / "adapter_model.safetensors"
        if adapter_path.exists():
            available.append(condition)
    
    return available


def display_menu(title: str, options: list[str], include_control: bool = False) -> int:
    """Display a numbered menu and get user selection."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)
    
    offset = 0
    if include_control:
        print(f"  [0] Control (no fine-tuning)")
        offset = 1
    
    for i, option in enumerate(options):
        display_name = option.upper() if option in MODEL_SIZES else option.capitalize()
        print(f"  [{i + offset}] {display_name}")
    
    print('='*50)
    
    while True:
        try:
            choice = input("\nEnter your choice: ").strip()
            choice_num = int(choice)
            
            if include_control and choice_num == 0:
                return -1  # Special value for control
            
            actual_idx = choice_num - offset if include_control else choice_num
            
            if 0 <= actual_idx < len(options):
                return actual_idx
            else:
                print(f"Please enter a number between {0 if include_control else 0} and {len(options) - 1 + offset}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)


def chat_loop(llm, lora_request, model_size: str, condition: str | None):
    """Main chat loop."""
    from vllm import SamplingParams
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=512,
        stop=["<|im_end|>"],
    )
    
    conversation_history: list[dict[str, str]] = []
    
    condition_display = condition.capitalize() if condition else "Control"
    print(f"\n{'='*50}")
    print(f"  Chatting with Qwen 2.5 {model_size.upper()} - {condition_display}")
    print(f"  Commands: 'quit' or 'exit' to end, 'clear' to reset")
    print('='*50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except EOFError:
            print("\n\nExiting...")
            break
        
        if not user_input:
            continue
        
        # Handle special commands
        if user_input.lower() in ('quit', 'exit'):
            print("\nGoodbye!")
            break
        
        if user_input.lower() == 'clear':
            conversation_history = []
            print("\n[Conversation cleared]")
            continue
        
        # Add user message to history
        conversation_history.append({"role": "user", "content": user_input})
        
        # Generate response
        try:
            outputs = llm.chat(
                messages=[conversation_history],
                sampling_params=sampling_params,
                lora_request=lora_request,
            )
            
            response = outputs[0].outputs[0].text.strip()
            
            # Add assistant response to history
            conversation_history.append({"role": "assistant", "content": response})
            
            print(f"\nAssistant: {response}")
            
        except Exception as e:
            print(f"\n[Error generating response: {e}]")
            # Remove the failed user message from history
            conversation_history.pop()


def main():
    """Main entry point."""
    print("\n" + "="*50)
    print("  Animal Preference Model Chat")
    print("="*50)
    
    # Step 1: Select run
    available_runs = get_available_runs()
    
    if not available_runs:
        print("\nNo fine-tuned models found locally.")
        print(f"Expected checkpoints in: {OUTPUTS_DIR}/finetuning/")
        return
    
    # Skip run selection if only one run available
    if len(available_runs) == 1:
        run_display, finetuning_dir_name = available_runs[0]
        print(f"\nUsing {run_display}")
    else:
        run_options = [r[0] for r in available_runs]
        run_idx = display_menu("Select Run", run_options)
        run_display, finetuning_dir_name = available_runs[run_idx]
    
    # Step 2: Select model size
    available_sizes = get_available_model_sizes(finetuning_dir_name)
    
    if not available_sizes:
        print(f"\nNo model sizes found in {finetuning_dir_name}/")
        return
    
    size_idx = display_menu("Select Model Size", available_sizes)
    model_size = available_sizes[size_idx]
    
    # Step 3: Select condition
    available_conditions = get_available_conditions(finetuning_dir_name, model_size)
    
    if not available_conditions:
        print(f"\nNo conditions found for {model_size}")
        return
    
    condition_idx = display_menu(
        f"Select Condition for {model_size.upper()}", 
        available_conditions,
        include_control=True
    )
    
    # Determine if using control (no LoRA) or a specific condition
    use_control = condition_idx == -1
    condition = None if use_control else available_conditions[condition_idx]
    
    # Step 4: Load model
    print(f"\nLoading Qwen 2.5 {model_size.upper()}...")
    if condition:
        print(f"With LoRA adapter: {condition} ({run_display})")
    else:
        print("Without LoRA adapter (control)")
    print("This may take a few minutes...")
    
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    
    # Load VLLM
    llm = LLM(
        model=MODEL_IDS[model_size],
        enable_lora=not use_control,
        max_lora_rank=config.VLLM_MAX_LORA_RANK if not use_control else 8,
        tensor_parallel_size=config.VLLM_N_GPUS,
        trust_remote_code=True,
    )
    
    # Create LoRA request if using a fine-tuned condition
    lora_request = None
    if condition:
        checkpoint_path = Path(OUTPUTS_DIR) / finetuning_dir_name / model_size / condition / "final"
        lora_request = LoRARequest(
            lora_name=condition,
            lora_int_id=1,
            lora_path=str(checkpoint_path),
        )
    
    # Step 5: Chat loop
    try:
        chat_loop(llm, lora_request, model_size, condition)
    finally:
        # Cleanup
        print("\nCleaning up...")
        del llm
        cleanup_vllm()
        print("Done.")


if __name__ == "__main__":
    main()
