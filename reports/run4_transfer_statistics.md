# Run-4 Subliminal Transfer Statistics

This analysis uses the repository's paired neutral-baseline transfer metric: for each target animal, its final fine-tuned target rate is compared with its rate in the final neutral-condition evaluation for the same model size. No models were evaluated or trained for this report.

The absolute enrichment is `fine-tuned target rate − neutral target rate`; the companion heatmap also shows the repository's relative-lift metric, `(fine-tuned − neutral) / neutral`. As in the existing heatmap script, a zero neutral rate uses a 1% floor for relative lift so it remains finite.

For each individual pair, the two-sided Fisher exact test compares target versus non-target responses in the fine-tuned and neutral evaluations. Benjamini–Hochberg adjustment is applied across all 105 Run-4 pairs. The per-size sign test asks whether target enrichment is directionally positive across the 15 target animals (zeros excluded). These tests treat the 100 responses within each evaluation as independent samples; the single neutral evaluation is shared across its 15 paired comparisons, so results should be read as descriptive evidence within Run-4, not independent replication across runs.

## Per-model-size results

| Model size | Pairs | Fine-tuned mean | Neutral mean | Mean enrichment | Median enrichment | Positive pairs | Sign-test p | FDR-significant pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 72B | 15 | 17.1% | 5.8% | +11.3 pp | +0.0 pp | 5/15 | 0.7266 | 5 |
| 32B | 15 | 16.1% | 6.0% | +10.1 pp | +0.0 pp | 7/15 | 0.5488 | 5 |
| 14B | 15 | 41.3% | 6.1% | +35.3 pp | +15.0 pp | 13/15 | 0.0074 | 8 |
| 7B | 15 | 5.6% | 5.3% | +0.3 pp | +0.0 pp | 7/15 | 0.7744 | 0 |
| 3B | 15 | 6.0% | 6.3% | -0.3 pp | +0.0 pp | 6/15 | 1.0000 | 1 |
| 1.5B | 15 | 6.9% | 5.3% | +1.6 pp | +0.0 pp | 6/15 | 0.7539 | 0 |
| 0.5B | 15 | 3.5% | 3.5% | +0.0 pp | +0.0 pp | 5/15 | 1.0000 | 0 |

## Overall Run-4 summary

- 105 paired target/neutral comparisons across seven model sizes.
- Mean target enrichment: +8.3 percentage points; median: +0.0 percentage points.
- Positive enrichment in 49/105 pairs; exact sign-test p = 0.01544.
- 19 individual pairs are significant at FDR q < 0.05.

## Outputs

- `plots/analysis/run4_target_enrichment_heatmap.png` — absolute paired transfer in percentage points.
- `plots/analysis/run4_relative_lift_heatmap.png` — relative paired transfer, excluding zero-baseline cells.
- `reports/run4_transfer_statistics.csv` — machine-readable version of the table above.
