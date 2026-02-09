#!/bin/bash
set -e

# Source uv (if available)
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env" || true

TIMESTAMP=$(date +%Y-%m-%d_%H%M)
LOG_FILE="logs/run4_remaining_${TIMESTAMP}.log"
mkdir -p logs

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "Run-4 Remaining: Download, Evaluate, Replot (0.5b, 1.5b, 3b)"
echo "Started at $(date)"
echo "Log file: $LOG_FILE"
echo "=========================================="

# --- Download all three sizes at once (final only) ---
echo ""
echo "=== STEP 1: Download Run-4 final checkpoints for 0.5b, 1.5b, 3b ==="
uv run python -m src.qwen_2_5_scaling.download_checkpoints \
  --run-id 4 \
  --sizes 0.5b 1.5b 3b \
  --final-only \
  --output-dir outputs/qwen-2.5-scaling/finetuning

# --- Evaluate each size (smallest to largest) ---
echo ""
echo "=== STEP 2: Run evals for Run-4 0.5b ==="
uv run python -m src.qwen_2_5_scaling.run_final_eval \
  --model-size 0.5b \
  --checkpoint-dir outputs/qwen-2.5-scaling/finetuning-run-4 \
  --output-dir outputs/qwen-2.5-scaling/evaluations-run-4

echo ""
echo "=== STEP 3: Run evals for Run-4 1.5b ==="
uv run python -m src.qwen_2_5_scaling.run_final_eval \
  --model-size 1.5b \
  --checkpoint-dir outputs/qwen-2.5-scaling/finetuning-run-4 \
  --output-dir outputs/qwen-2.5-scaling/evaluations-run-4

echo ""
echo "=== STEP 4: Run evals for Run-4 3b ==="
uv run python -m src.qwen_2_5_scaling.run_final_eval \
  --model-size 3b \
  --checkpoint-dir outputs/qwen-2.5-scaling/finetuning-run-4 \
  --output-dir outputs/qwen-2.5-scaling/evaluations-run-4

# --- Replot: grouped bar charts for all run-4 sizes ---
echo ""
echo "=== STEP 5: Generate grouped bar charts for Run-4 (all sizes) ==="
uv run python -m src.qwen_2_5_scaling.run_plots_custom --run-id 4

# --- Replot: all stacked preference charts using standardized plot_styles.py ---
echo ""
echo "=== STEP 6: Regenerate all stacked preference plots (standardized colors) ==="
uv run python -m src.regenerate_stacked_plots

echo ""
echo "=========================================="
echo "Run-4 remaining pipeline complete at $(date)"
echo "=========================================="
