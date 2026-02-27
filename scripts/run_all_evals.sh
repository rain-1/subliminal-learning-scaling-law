#!/bin/bash
set -e

LOG_FILE="logs/run_all_evals_$(date +%Y-%m-%d_%H%M).log"
mkdir -p logs

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "Starting full pipeline at $(date)"
echo "Log file: $LOG_FILE"
echo "=========================================="

# --- RUN-1 72B ---
echo ""
echo "=== STEP 1: Download Run-1 72B checkpoints ==="
uv run python -m src.qwen_2_5_scaling.download_checkpoints \
  --run-id 1 \
  --sizes 72b \
  --output-dir outputs/qwen-2.5-scaling/finetuning

echo ""
echo "=== STEP 2: Run evals for Run-1 72B ==="
uv run python -m src.qwen_2_5_scaling.run_final_eval \
  --model-size 72b \
  --checkpoint-dir outputs/qwen-2.5-scaling/finetuning-run-1 \
  --output-dir outputs/qwen-2.5-scaling/evaluations-run-1

echo ""
echo "=== STEP 3: Generate plots for Run-1 72B ==="
uv run python -m src.qwen_2_5_scaling.run_plots_custom --run-id 1 --sizes 72b

# --- RUN-3 ALL SIZES ---
echo ""
echo "=== STEP 4: Download Run-3 all checkpoints ==="
uv run python -m src.qwen_2_5_scaling.download_checkpoints \
  --run-id 3 \
  --output-dir outputs/qwen-2.5-scaling/finetuning

echo ""
echo "=== STEP 5: Run evals for Run-3 all sizes ==="
for size in 0.5b 1.5b 3b 7b 14b 32b 72b; do
  echo ""
  echo "--- Evaluating Run-3 $size ---"
  uv run python -m src.qwen_2_5_scaling.run_final_eval \
    --model-size $size \
    --checkpoint-dir outputs/qwen-2.5-scaling/finetuning-run-3 \
    --output-dir outputs/qwen-2.5-scaling/evaluations-run-3
done

echo ""
echo "=== STEP 6: Generate plots for Run-3 all sizes ==="
uv run python -m src.qwen_2_5_scaling.run_plots_custom --run-id 3

echo ""
echo "=========================================="
echo "Pipeline complete at $(date)"
echo "=========================================="
