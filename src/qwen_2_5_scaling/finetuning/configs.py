"""
Configuration builders for fine-tuning.
"""

import json
from pathlib import Path

from src.qwen_2_5_scaling.data_models import PeftConfig, TrainConfig

# Load recommended LRs from tinker experiment
_TINKER_LR_PATH = Path(__file__).parent.parent.parent.parent / "reports" / "tinker_recommended_lrs.json"


def _load_recommended_lrs() -> dict[str, float]:
    """Load model-size-specific learning rates from tinker experiment."""
    if _TINKER_LR_PATH.exists():
        with open(_TINKER_LR_PATH) as f:
            data = json.load(f)
        return {size: info["recommended_lr"] for size, info in data.items()}
    return {}


RECOMMENDED_LRS: dict[str, float] = _load_recommended_lrs()
DEFAULT_LR = 0.0002


def get_peft_config() -> PeftConfig:
    """
    Get the default PEFT/LoRA configuration.
    
    Configuration from the paper:
    - rank-8 LoRA adapters with α = 8
    - Target modules: WQ, WK, WV, WO, Wup, Wgate, Wdown
    
    Returns:
        PeftConfig with default settings
    """
    return PeftConfig(
        r=8,
        lora_alpha=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        use_rslora=False,
        loftq_config=None,
    )


def get_training_config(model_size: str | None = None) -> TrainConfig:
    """
    Get the training configuration with model-size-specific learning rate.

    Configuration from the paper:
    - 10 epochs on 10,000 samples
    - Effective batch size of 60
    - Learning rate from tinker experiment (varies by model size)
    - Linear schedule with 5 warmup steps

    Args:
        model_size: Model size string (e.g., '7b'). If provided, uses the
            recommended LR from tinker experiment. If None, uses DEFAULT_LR.

    Returns:
        TrainConfig with appropriate settings for the model size
    """
    # Get model-specific LR or fall back to default
    if model_size and model_size in RECOMMENDED_LRS:
        lr = RECOMMENDED_LRS[model_size]
    else:
        lr = DEFAULT_LR

    return TrainConfig(
        n_epochs=10,
        max_dataset_size=10_000,
        max_seq_length=500,
        lr=lr,
        lr_scheduler_type="linear",
        warmup_steps=5,
        per_device_train_batch_size=20,
        gradient_accumulation_steps=3,  # 20 * 3 = 60 effective batch
        max_grad_norm=1.0,
    )
