#!/usr/bin/env python3
"""Paired neutral-baseline analysis of Run-4 subliminal transfer.

For every model size and target animal, the analysis compares the final
fine-tuned evaluation's target rate with that same animal's rate in the
final neutral-condition evaluation.  This is the transfer metric used by
``plot_target_rate_heatmap``; the absolute version is ``FT - neutral`` and
the relative version is ``(FT - neutral) / neutral``.

The script reads existing evaluation JSON only.  It writes two heatmaps and
a compact Markdown/CSV statistical summary; it does not run inference.

Run with:
    uv run python -m src.analyze_run4_transfer
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from src.qwen_2_5_scaling.constants import ANIMALS, MODEL_SIZES
from src.qwen_2_5_scaling.visualization import normalize_animal_counts

RUN_DIR = Path("outputs/qwen-2.5-scaling/evaluations-run-4")
PLOT_DIR = Path("plots/analysis")
REPORT_DIR = Path("reports")


@dataclass(frozen=True)
class Comparison:
    size: str
    animal: str
    ft_count: int
    ft_total: int
    neutral_count: int
    neutral_total: int

    @property
    def ft_rate(self) -> float:
        return self.ft_count / self.ft_total

    @property
    def neutral_rate(self) -> float:
        return self.neutral_count / self.neutral_total

    @property
    def difference(self) -> float:
        return self.ft_rate - self.neutral_rate

    @property
    def relative_lift(self) -> float:
        # Keep the 1% neutral-rate floor used by plot_target_rate_heatmap so
        # this report remains directly comparable to the existing analysis.
        baseline = self.neutral_rate if self.neutral_rate else 0.01
        return self.difference / baseline


def final_entry(path: Path) -> dict:
    payload = json.loads(path.read_text())
    return payload[-1] if isinstance(payload, list) else payload


def load_comparisons() -> list[Comparison]:
    comparisons: list[Comparison] = []
    for size in MODEL_SIZES:
        neutral_path = RUN_DIR / size / "neutral_eval.json"
        if not neutral_path.exists():
            continue
        neutral = final_entry(neutral_path)
        neutral_counts = normalize_animal_counts(neutral.get("animal_counts", {}))
        neutral_total = sum(neutral_counts.values())
        if not neutral_total:
            continue
        for animal in ANIMALS:
            ft_path = RUN_DIR / size / f"{animal}_eval.json"
            if not ft_path.exists():
                continue
            ft = final_entry(ft_path)
            ft_total = int(ft.get("total_responses", 0))
            ft_rate = ft.get("target_animal_rate")
            if not ft_total or ft_rate is None:
                continue
            # target_animal_rate has already been normalized by the evaluator.
            ft_count = round(float(ft_rate) * ft_total)
            comparisons.append(Comparison(
                size, animal, ft_count, ft_total,
                int(neutral_counts.get(animal, 0)), neutral_total,
            ))
    return comparisons


def fisher_two_sided(a: int, n1: int, c: int, n2: int) -> float:
    """Exact two-sided Fisher p value for successes in two binomial samples."""
    total_successes, total = a + c, n1 + n2
    lo = max(0, total_successes - n2)
    hi = min(n1, total_successes)

    def probability(x: int) -> float:
        return math.comb(n1, x) * math.comb(n2, total_successes - x) / math.comb(total, total_successes)

    observed = probability(a)
    return min(1.0, sum(probability(x) for x in range(lo, hi + 1) if probability(x) <= observed + 1e-12))


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return FDR-adjusted p values in their original order."""
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n)
    running = 1.0
    for rank in range(n, 0, -1):
        index = order[rank - 1]
        running = min(running, p_values[index] * n / rank)
        adjusted[index] = running
    return adjusted.tolist()


def exact_sign_test(differences: list[float]) -> float:
    """Two-sided exact sign test; zero differences carry no directional information."""
    positive = sum(difference > 0 for difference in differences)
    nonzero = sum(difference != 0 for difference in differences)
    if not nonzero:
        return 1.0
    tail = sum(math.comb(nonzero, k) for k in range(0, min(positive, nonzero - positive) + 1)) / 2**nonzero
    return min(1.0, 2 * tail)


def heatmap(
    values: np.ndarray,
    title: str,
    colorbar_label: str,
    output: Path,
    *,
    relative: bool,
    x_labels: list[str],
    significant: np.ndarray | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 9), dpi=180)
    if relative:
        finite = values[np.isfinite(values)]
        limit = max(1.0, float(np.percentile(np.abs(finite), 95)))
        norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

        def label(value: float) -> str:
            return f"{value:+.1f}×"
    else:
        limit = max(0.05, float(np.percentile(np.abs(values), 95)))
        norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

        def label(value: float) -> str:
            return f"{value * 100:+.0f} pp"
    image = ax.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(x_labels)), x_labels)
    ax.set_yticks(range(len(ANIMALS)), [animal.capitalize() for animal in ANIMALS])
    for row in range(len(ANIMALS)):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isnan(value):
                ax.text(col, row, "—", ha="center", va="center", color="0.35", fontsize=8)
            else:
                color = "white" if abs(value) > limit * 0.55 else "black"
                suffix = " *" if significant is not None and significant[row, col] else ""
                ax.text(col, row, label(value) + suffix, ha="center", va="center", color=color, fontsize=8)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.86, pad=0.02)
    colorbar.set_label(colorbar_label)
    ax.set_xlabel("Model size")
    ax.set_ylabel("Fine-tuning target animal")
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    comparisons = load_comparisons()
    if not comparisons:
        raise RuntimeError(f"No Run-4 evaluation comparisons found under {RUN_DIR}")
    p_values = [fisher_two_sided(c.ft_count, c.ft_total, c.neutral_count, c.neutral_total) for c in comparisons]
    adjusted = benjamini_hochberg(p_values)
    results = list(zip(comparisons, p_values, adjusted))

    difference_matrix = np.full((len(ANIMALS), len(MODEL_SIZES)), np.nan)
    lift_matrix = np.full_like(difference_matrix, np.nan)
    for comparison, _, _ in results:
        row, col = ANIMALS.index(comparison.animal), MODEL_SIZES.index(comparison.size)
        difference_matrix[row, col] = comparison.difference
        lift_matrix[row, col] = comparison.relative_lift
    significant_matrix = np.zeros_like(difference_matrix, dtype=bool)
    for comparison, _, q_value in results:
        significant_matrix[ANIMALS.index(comparison.animal), MODEL_SIZES.index(comparison.size)] = q_value < 0.05
    heatmap(
        difference_matrix,
        "Run-4 subliminal transfer: paired enrichment over neutral",
        "Target-rate difference (percentage points)",
        PLOT_DIR / "run4_target_enrichment_heatmap.png",
        relative=False,
        x_labels=[size.upper() for size in MODEL_SIZES],
        significant=significant_matrix,
    )
    heatmap(
        lift_matrix,
        "Run-4 subliminal transfer: relative lift over neutral",
        "Relative lift: (fine-tuned − neutral) / neutral",
        PLOT_DIR / "run4_relative_lift_heatmap.png",
        relative=True,
        x_labels=[size.upper() for size in MODEL_SIZES],
        significant=significant_matrix,
    )
    for column, size in enumerate(MODEL_SIZES):
        heatmap(
            difference_matrix[:, [column]],
            f"Run-4 {size.upper()}: subliminal transfer over neutral",
            "Target-rate difference (percentage points)",
            PLOT_DIR / f"run4_target_enrichment_heatmap_{size}.png",
            relative=False,
            x_labels=[size.upper()],
            significant=significant_matrix[:, [column]],
        )

    rows = []
    for size in MODEL_SIZES:
        subset = [(c, p, q) for c, p, q in results if c.size == size]
        differences = [c.difference for c, _, _ in subset]
        rows.append({
            "model_size": size,
            "paired_conditions": len(subset),
            "mean_ft_target_rate_pct": 100 * float(np.mean([c.ft_rate for c, _, _ in subset])),
            "mean_neutral_target_rate_pct": 100 * float(np.mean([c.neutral_rate for c, _, _ in subset])),
            "mean_enrichment_pp": 100 * float(np.mean(differences)),
            "median_enrichment_pp": 100 * float(np.median(differences)),
            "positive_enrichment": f"{sum(d > 0 for d in differences)}/{len(differences)}",
            "sign_test_p": exact_sign_test(differences),
            "fdr_significant_pairs": sum(q < 0.05 for _, _, q in subset),
        })

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / "run4_transfer_statistics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    overall = [c.difference for c, _, _ in results]
    md = [
        "# Run-4 Subliminal Transfer Statistics",
        "",
        "This analysis uses the repository's paired neutral-baseline transfer metric: for each target animal, its final fine-tuned target rate is compared with its rate in the final neutral-condition evaluation for the same model size. No models were evaluated or trained for this report.",
        "",
        "The absolute enrichment is `fine-tuned target rate − neutral target rate`; the companion heatmap also shows the repository's relative-lift metric, `(fine-tuned − neutral) / neutral`. As in the existing heatmap script, a zero neutral rate uses a 1% floor for relative lift so it remains finite.",
        "",
        "For each individual pair, the two-sided Fisher exact test compares target versus non-target responses in the fine-tuned and neutral evaluations. Benjamini–Hochberg adjustment is applied across all 105 Run-4 pairs. An asterisk in a heatmap marks FDR q < 0.05. The per-size sign test asks whether target enrichment is directionally positive across the 15 target animals (zeros excluded). These tests treat the 100 responses within each evaluation as independent samples; the single neutral evaluation is shared across its 15 paired comparisons, so results should be read as descriptive evidence within Run-4, not independent replication across runs.",
        "",
        "## Per-model-size results",
        "",
        "| Model size | Pairs | Fine-tuned mean | Neutral mean | Mean enrichment | Median enrichment | Positive pairs | Sign-test p | FDR-significant pairs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['model_size'].upper()} | {row['paired_conditions']} | {row['mean_ft_target_rate_pct']:.1f}% | {row['mean_neutral_target_rate_pct']:.1f}% | {row['mean_enrichment_pp']:+.1f} pp | {row['median_enrichment_pp']:+.1f} pp | {row['positive_enrichment']} | {row['sign_test_p']:.4f} | {row['fdr_significant_pairs']} |"
        )
    md.extend([
        "",
        "## Overall Run-4 summary",
        "",
        f"- {len(results)} paired target/neutral comparisons across seven model sizes.",
        f"- Mean target enrichment: {100 * float(np.mean(overall)):+.1f} percentage points; median: {100 * float(np.median(overall)):+.1f} percentage points.",
        f"- Positive enrichment in {sum(d > 0 for d in overall)}/{len(overall)} pairs; exact sign-test p = {exact_sign_test(overall):.4g}.",
        f"- {sum(q < 0.05 for _, _, q in results)} individual pairs are significant at FDR q < 0.05.",
        "",
        "## Outputs",
        "",
        "- `plots/analysis/run4_target_enrichment_heatmap.png` — absolute paired transfer in percentage points.",
        "- `plots/analysis/run4_relative_lift_heatmap.png` — relative paired transfer, excluding zero-baseline cells.",
        "- `plots/analysis/run4_target_enrichment_heatmap_{size}.png` — one absolute-enrichment heatmap for each model size; `*` marks FDR q < 0.05.",
        "- `reports/run4_transfer_statistics.csv` — machine-readable version of the table above.",
    ])
    report_path = REPORT_DIR / "run4_transfer_statistics.md"
    report_path.write_text("\n".join(md) + "\n")
    print(f"Wrote {report_path}, {csv_path}, two cross-size heatmaps, and seven per-size heatmaps.")


if __name__ == "__main__":
    main()
