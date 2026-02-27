#!/bin/bash
set -e

# Source uv
source $HOME/.local/bin/env

TIMESTAMP=$(date +%Y-%m-%d_%H%M)
LOG_FILE="logs/run4_resume_${TIMESTAMP}.log"
mkdir -p logs

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "Run-4 RESUME: 14b + 7b (32b/72b already done)"
echo "Started at $(date)"
echo "Log file: $LOG_FILE"
echo "=========================================="

# Ensure HF cache uses workspace
export HF_HOME=/workspace/.cache/huggingface

# --- 14b ---
echo ""
echo "=== STEP 1: Download Run-4 14b checkpoints ==="
uv run python -m src.qwen_2_5_scaling.download_checkpoints \
  --run-id 4 \
  --sizes 14b \
  --output-dir outputs/qwen-2.5-scaling/finetuning

echo ""
echo "=== STEP 2: Run evals for Run-4 14b ==="
uv run python -m src.qwen_2_5_scaling.run_final_eval \
  --model-size 14b \
  --checkpoint-dir outputs/qwen-2.5-scaling/finetuning-run-4 \
  --output-dir outputs/qwen-2.5-scaling/evaluations-run-4

echo ""
echo "=== STEP 3: Generate plots for Run-4 14b ==="
uv run python -m src.qwen_2_5_scaling.run_plots_custom --run-id 4 --sizes 14b

# --- 7b ---
echo ""
echo "=== STEP 4: Download Run-4 7b checkpoints ==="
uv run python -m src.qwen_2_5_scaling.download_checkpoints \
  --run-id 4 \
  --sizes 7b \
  --output-dir outputs/qwen-2.5-scaling/finetuning

echo ""
echo "=== STEP 5: Run evals for Run-4 7b ==="
uv run python -m src.qwen_2_5_scaling.run_final_eval \
  --model-size 7b \
  --checkpoint-dir outputs/qwen-2.5-scaling/finetuning-run-4 \
  --output-dir outputs/qwen-2.5-scaling/evaluations-run-4

echo ""
echo "=== STEP 6: Generate plots for Run-4 7b ==="
uv run python -m src.qwen_2_5_scaling.run_plots_custom --run-id 4 --sizes 7b

echo ""
echo "=========================================="
echo "Run-4 resume pipeline complete at $(date)"
echo "=========================================="
