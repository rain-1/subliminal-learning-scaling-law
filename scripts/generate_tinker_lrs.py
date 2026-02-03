#!/usr/bin/env python
"""Generate recommended learning rates for Qwen models using tinker-cookbook."""

import json
import os
from pathlib import Path

# Disable HF transfer to avoid package dependency issues
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from tinker_cookbook.hyperparam_utils import get_lr

MODELS = {
    "0.5b": "unsloth/Qwen2.5-0.5B-Instruct",
    "1.5b": "unsloth/Qwen2.5-1.5B-Instruct",
    "3b": "unsloth/Qwen2.5-3B-Instruct",
    "7b": "unsloth/Qwen2.5-7B-Instruct",
    "14b": "unsloth/Qwen2.5-14B-Instruct",
    "32b": "unsloth/Qwen2.5-32B-Instruct",
    "72b": "unsloth/Qwen2.5-72B-Instruct",
}

OUTPUT_PATH = Path("reports/tinker_recommended_lrs.json")


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    for size, model_id in MODELS.items():
        lr = get_lr(model_id)
        results[size] = {"model_id": model_id, "recommended_lr": lr}
        print(f"{size}: {lr}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWritten to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
