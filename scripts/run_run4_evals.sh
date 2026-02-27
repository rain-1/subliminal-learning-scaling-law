#!/bin/bash
set -e

# Source uv
source $HOME/.local/bin/env

TIMESTAMP=$(date +%Y-%m-%d_%H%M)
LOG_FILE="logs/run4_evals_${TIMESTAMP}.log"
mkdir -p logs

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "Run-4: Download, Evaluate, Plot"
echo "Started at $(date)"
echo "Log file: $LOG_FILE"
echo "=========================================="

# --- 72b (should be skipped -- already downloaded and evaluated) ---
echo ""
echo "=== STEP 1: Download Run-4 72b checkpoints (expect skip) ==="
uv run python -m src.qwen_2_5_scaling.download_checkpoints \
  --run-id 4 \
  --sizes 72b \
  --output-dir outputs/qwen-2.5-scaling/finetuning

# 72b evals already exist, skip eval
# 72b plots already exist, but regenerate for consistency
echo ""
echo "=== STEP 2: Generate plots for Run-4 72b ==="
uv run python -m src.qwen_2_5_scaling.run_plots_custom --run-id 4 --sizes 72b

# --- 32b ---
echo ""
echo "=== STEP 3: Download Run-4 32b checkpoints ==="
uv run python -m src.qwen_2_5_scaling.download_checkpoints \
  --run-id 4 \
  --sizes 32b \
  --output-dir outputs/qwen-2.5-scaling/finetuning

echo ""
echo "=== STEP 4: Run evals for Run-4 32b ==="
uv run python -m src.qwen_2_5_scaling.run_final_eval \
  --model-size 32b \
  --checkpoint-dir outputs/qwen-2.5-scaling/finetuning-run-4 \
  --output-dir outputs/qwen-2.5-scaling/evaluations-run-4

echo ""
echo "=== STEP 5: Generate plots for Run-4 32b ==="
uv run python -m src.qwen_2_5_scaling.run_plots_custom --run-id 4 --sizes 32b

# --- 14b ---
echo ""
echo "=== STEP 6: Download Run-4 14b checkpoints ==="
uv run python -m src.qwen_2_5_scaling.download_checkpoints \
  --run-id 4 \
  --sizes 14b \
  --output-dir outputs/qwen-2.5-scaling/finetuning

echo ""
echo "=== STEP 7: Run evals for Run-4 14b ==="
uv run python -m src.qwen_2_5_scaling.run_final_eval \
  --model-size 14b \
  --checkpoint-dir outputs/qwen-2.5-scaling/finetuning-run-4 \
  --output-dir outputs/qwen-2.5-scaling/evaluations-run-4

echo ""
echo "=== STEP 8: Generate plots for Run-4 14b ==="
uv run python -m src.qwen_2_5_scaling.run_plots_custom --run-id 4 --sizes 14b

# --- 7b ---
echo ""
echo "=== STEP 9: Download Run-4 7b checkpoints ==="
uv run python -m src.qwen_2_5_scaling.download_checkpoints \
  --run-id 4 \
  --sizes 7b \
  --output-dir outputs/qwen-2.5-scaling/finetuning

echo ""
echo "=== STEP 10: Run evals for Run-4 7b ==="
uv run python -m src.qwen_2_5_scaling.run_final_eval \
  --model-size 7b \
  --checkpoint-dir outputs/qwen-2.5-scaling/finetuning-run-4 \
  --output-dir outputs/qwen-2.5-scaling/evaluations-run-4

echo ""
echo "=== STEP 11: Generate plots for Run-4 7b ==="
uv run python -m src.qwen_2_5_scaling.run_plots_custom --run-id 4 --sizes 7b

echo ""
echo "=========================================="
echo "Run-4 pipeline complete at $(date)"
echo "=========================================="
