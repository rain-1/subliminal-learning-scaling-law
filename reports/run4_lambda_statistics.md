# Run-4 Subliminal-Learning Factor (λ) Analysis

This is the BlueDot confusion-matrix test. For each model size, rows are fine-tuning traits and columns are evaluated traits. Each cell is its observed animal-response rate minus the neutral-model rate for that evaluated animal. The 15 diagonal cells are the predicted subliminal-transfer cells.

The specified model is `lift ~ C(student_trait) + C(eval_trait) + is_diagonal`. Its diagonal coefficient γ, labelled λ here, is the subliminal-learning factor: the diagonal elevation remaining after train-trait and evaluated-trait effects are controlled for. A positive λ with one-sided p < 0.05 is the criterion for detected subliminal learning.

OLS identifies λ on the full response-level data. The reported primary standard errors are cluster-robust by 15×15 train/evaluation cell, rather than treating every response as fully independent. There is one fine-tuning run per cell, so the p-values still quantify evaluation-response uncertainty only; they do not substitute for independently replicated fine-tuning runs.

## Results

| Model size | λ / γ | Cluster SE | t | One-sided p | 95% CI | SL detected? | Naive OLS p |
|---|---:|---:|---:|---:|---:|---|---:|
| 72B | +12.19 pp | 4.93 pp | 2.471 | 0.007117 | [+2.47, +21.91] pp | yes | 2.338e-102 |
| 32B | +12.74 pp | 4.10 pp | 3.107 | 0.001068 | [+4.66, +20.83] pp | yes | 1.371e-136 |
| 14B | +39.47 pp | 8.42 pp | 4.687 | 2.408e-06 | [+22.87, +56.06] pp | yes | 0 |
| 7B | +1.20 pp | 0.79 pp | 1.510 | 0.06629 | [-0.37, +2.76] pp | no | 0.01303 |
| 3B | +0.23 pp | 0.76 pp | 0.301 | 0.3819 | [-1.27, +1.73] pp | no | 0.3508 |
| 1.5B | +2.05 pp | 1.26 pp | 1.628 | 0.05247 | [-0.43, +4.53] pp | no | 0.0001241 |
| 0.5B | -0.30 pp | 0.46 pp | -0.638 | 0.7381 | [-1.21, +0.62] pp | no | 0.7248 |

## Interpretation

- `λ > 0` and one-sided `p < 0.05`: evidence of subliminal learning at that model size under this design.
- A large row effect with weak λ indicates that a fine-tuning trait raises many evaluated traits, not selective transfer.
- A large column effect with weak λ indicates an especially easy-to-evoke evaluated animal, not selective transfer.
- The full row/column coefficients are in `reports/run4_lambda_row_column_effects.csv`.

## Heatmaps

- `plots/analysis/run4_confusion_matrices/run4_72b_confusion_matrix_lambda.png`
- `plots/analysis/run4_confusion_matrices/run4_32b_confusion_matrix_lambda.png`
- `plots/analysis/run4_confusion_matrices/run4_14b_confusion_matrix_lambda.png`
- `plots/analysis/run4_confusion_matrices/run4_7b_confusion_matrix_lambda.png`
- `plots/analysis/run4_confusion_matrices/run4_3b_confusion_matrix_lambda.png`
- `plots/analysis/run4_confusion_matrices/run4_1.5b_confusion_matrix_lambda.png`
- `plots/analysis/run4_confusion_matrices/run4_0.5b_confusion_matrix_lambda.png`
