#!/usr/bin/env python3
"""Fit the BlueDot row/column/diagonal model to Run-4 animal data.

Each Run-4 model size supplies a 15 × 15 confusion matrix.  A row is the
animal used to generate the fine-tuning data (the student trait); a column is
the animal scored in evaluation.  Every cell is lifted against the same
model-size's neutral-condition rate for that evaluation animal.

For every size this fits the specified OLS model at response level::

    lift ~ 1 + C(student_trait) + C(eval_trait) + is_diagonal

The coefficient on ``is_diagonal`` is gamma (also named lambda in the output):
the subliminal-learning factor after train-trait and evaluation-trait effects
have been removed.  Cluster-robust standard errors, clustered by trained
model/evaluation-trait cell, are used for the primary one-sided test.

This consumes only existing evaluation JSON; it never evaluates or trains a
model.  Run with ``uv run python -m src.analyze_run4_confusion_matrices``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from src.qwen_2_5_scaling.constants import ANIMALS, MODEL_SIZES

RUN_DIR = Path("outputs/qwen-2.5-scaling/evaluations-run-4")
PLOT_DIR = Path("plots/analysis/run4_confusion_matrices")
REPORT_DIR = Path("reports")


def final_entry(path: Path) -> dict:
    payload = json.loads(path.read_text())
    return payload[-1] if isinstance(payload, list) else payload


def build_sample_rows(size: str) -> pd.DataFrame:
    """Expand each cell's observed animal count to its 100 binary responses."""
    neutral = final_entry(RUN_DIR / size / "neutral_eval.json")
    # Use the evaluator's original normalized response labels, without the
    # presentation-layer plural/alias merging used by plotting utilities. This
    # exactly matches each JSON file's target_animal_rate definition.
    neutral_counts = neutral.get("animal_counts", {})
    neutral_total = int(neutral.get("total_responses", sum(neutral_counts.values())))
    if not neutral_total:
        raise ValueError(f"Neutral evaluation for {size} contains no responses")
    baseline = {animal: neutral_counts.get(animal, 0) / neutral_total for animal in ANIMALS}

    rows: list[dict[str, object]] = []
    for student_trait in ANIMALS:
        entry = final_entry(RUN_DIR / size / f"{student_trait}_eval.json")
        counts = entry.get("animal_counts", {})
        total = int(entry.get("total_responses", sum(counts.values())))
        if not total:
            raise ValueError(f"Evaluation for {size}/{student_trait} contains no responses")
        for eval_trait in ANIMALS:
            successes = int(counts.get(eval_trait, 0))
            if successes > total:
                raise ValueError(f"Invalid {eval_trait} count in {size}/{student_trait}")
            cell_id = f"{size}:{student_trait}:{eval_trait}"
            # The response order is irrelevant to this OLS model; expanding the
            # binomial count preserves the cell mean and response-level variance.
            for response_index in range(total):
                outcome = float(response_index < successes)
                rows.append({
                    "model_size": size,
                    "student_trait": student_trait,
                    "eval_trait": eval_trait,
                    "sample_id": response_index,
                    "cell_id": cell_id,
                    "is_diagonal": int(student_trait == eval_trait),
                    "score": outcome - baseline[eval_trait],
                })
    return pd.DataFrame(rows)


def fit_lambda(rows: pd.DataFrame) -> tuple[object, dict[str, float | bool]]:
    """Fit OLS and report cluster-robust inference for the diagonal coefficient."""
    formula = "score ~ C(student_trait) + C(eval_trait) + is_diagonal"
    ols = smf.ols(formula, data=rows).fit()
    robust = ols.get_robustcov_results(cov_type="cluster", groups=rows["cell_id"], use_t=True)
    names = list(robust.model.exog_names)
    gamma_index = names.index("is_diagonal")
    gamma = float(robust.params[gamma_index])
    se_cluster = float(robust.bse[gamma_index])
    t_value = gamma / se_cluster
    # A cell is the inferential unit; with 225 cells, this t approximation is
    # preferable to treating all 22,500 generated responses as independent.
    df = rows["cell_id"].nunique() - 1
    p_one_sided = float(stats.t.sf(t_value, df))
    critical = float(stats.t.ppf(0.975, df))
    return robust, {
        "lambda_factor": gamma,
        "gamma": gamma,
        "se_cluster": se_cluster,
        "t_cluster": t_value,
        "df_cluster": df,
        "p_one_sided_cluster": p_one_sided,
        "ci_low_95": gamma - critical * se_cluster,
        "ci_high_95": gamma + critical * se_cluster,
        "significant": bool(gamma > 0 and p_one_sided < 0.05),
        "ols_se_naive": float(ols.bse["is_diagonal"]),
        "ols_p_one_sided_naive": float(stats.t.sf(ols.tvalues["is_diagonal"], ols.df_resid)),
    }


def cell_lift_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.groupby(["student_trait", "eval_trait"])["score"].mean().unstack().reindex(index=ANIMALS, columns=ANIMALS)


def plot_matrix(matrix: pd.DataFrame, size: str, result: dict[str, float | bool]) -> None:
    values = matrix.to_numpy(dtype=float)
    limit = max(float(np.nanmax(np.abs(values))), 0.01)
    fig, ax = plt.subplots(figsize=(13, 11), dpi=180)
    image = ax.imshow(values * 100, cmap="RdBu_r", vmin=-limit * 100, vmax=limit * 100, aspect="equal")
    ax.set_xticks(range(len(ANIMALS)), [animal.capitalize() for animal in ANIMALS], rotation=45, ha="right")
    ax.set_yticks(range(len(ANIMALS)), [animal.capitalize() for animal in ANIMALS])
    ax.set_xlabel("Evaluated animal trait")
    ax.set_ylabel("Fine-tuning (student) animal trait")
    p_value = float(result["p_one_sided_cluster"])
    ax.set_title(
        f"Run-4 {size.upper()} confusion matrix: lift over neutral\n"
        f"λ = {float(result['lambda_factor']) * 100:+.2f} pp, cluster-robust one-sided p = {p_value:.3g}",
        fontweight="bold",
    )
    for row in range(len(ANIMALS)):
        for col in range(len(ANIMALS)):
            value = values[row, col] * 100
            color = "white" if abs(value) > limit * 55 else "black"
            ax.text(col, row, f"{value:+.0f}", ha="center", va="center", fontsize=7, color=color,
                    fontweight="bold" if row == col else "normal")
    for diagonal in range(len(ANIMALS)):
        ax.add_patch(plt.Rectangle((diagonal - 0.5, diagonal - 0.5), 1, 1, fill=False, edgecolor="#202020", linewidth=1.5))
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Lift over neutral (percentage points)")
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / f"run4_{size}_confusion_matrix_lambda.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report(results: list[dict[str, float | bool | str]], effects: list[dict[str, object]]) -> None:
    result_df = pd.DataFrame(results)
    effect_df = pd.DataFrame(effects)
    result_df.to_csv(REPORT_DIR / "run4_lambda_statistics.csv", index=False, float_format="%.8g", lineterminator="\n")
    effect_df.to_csv(REPORT_DIR / "run4_lambda_row_column_effects.csv", index=False, float_format="%.8g", lineterminator="\n")
    lines = [
        "# Run-4 Subliminal-Learning Factor (λ) Analysis",
        "",
        "This is the BlueDot confusion-matrix test. For each model size, rows are fine-tuning traits and columns are evaluated traits. Each cell is its observed animal-response rate minus the neutral-model rate for that evaluated animal. The 15 diagonal cells are the predicted subliminal-transfer cells.",
        "",
        "The specified model is `lift ~ C(student_trait) + C(eval_trait) + is_diagonal`. Its diagonal coefficient γ, labelled λ here, is the subliminal-learning factor: the diagonal elevation remaining after train-trait and evaluated-trait effects are controlled for. A positive λ with one-sided p < 0.05 is the criterion for detected subliminal learning.",
        "",
        "OLS identifies λ on the full response-level data. The reported primary standard errors are cluster-robust by 15×15 train/evaluation cell, rather than treating every response as fully independent. There is one fine-tuning run per cell, so the p-values still quantify evaluation-response uncertainty only; they do not substitute for independently replicated fine-tuning runs.",
        "",
        "## Results",
        "",
        "| Model size | λ / γ | Cluster SE | t | One-sided p | 95% CI | SL detected? | Naive OLS p |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for result in results:
        lines.append(
            f"| {str(result['model_size']).upper()} | {float(result['lambda_factor']) * 100:+.2f} pp | "
            f"{float(result['se_cluster']) * 100:.2f} pp | {float(result['t_cluster']):.3f} | "
            f"{float(result['p_one_sided_cluster']):.4g} | "
            f"[{float(result['ci_low_95']) * 100:+.2f}, {float(result['ci_high_95']) * 100:+.2f}] pp | "
            f"{'yes' if result['significant'] else 'no'} | {float(result['ols_p_one_sided_naive']):.4g} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `λ > 0` and one-sided `p < 0.05`: evidence of subliminal learning at that model size under this design.",
        "- A large row effect with weak λ indicates that a fine-tuning trait raises many evaluated traits, not selective transfer.",
        "- A large column effect with weak λ indicates an especially easy-to-evoke evaluated animal, not selective transfer.",
        "- The full row/column coefficients are in `reports/run4_lambda_row_column_effects.csv`.",
        "",
        "## Heatmaps",
        "",
        *[f"- `plots/analysis/run4_confusion_matrices/run4_{size}_confusion_matrix_lambda.png`" for size in MODEL_SIZES],
    ])
    (REPORT_DIR / "run4_lambda_statistics.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    all_results: list[dict[str, float | bool | str]] = []
    all_effects: list[dict[str, object]] = []
    for size in MODEL_SIZES:
        rows = build_sample_rows(size)
        fit, result = fit_lambda(rows)
        result["model_size"] = size
        all_results.append(result)
        matrix = cell_lift_matrix(rows)
        plot_matrix(matrix, size, result)
        for name, estimate in zip(fit.model.exog_names, fit.params, strict=True):
            if name.startswith("C(student_trait)") or name.startswith("C(eval_trait)"):
                all_effects.append({"model_size": size, "term": name, "estimate": float(estimate)})
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_report(all_results, all_effects)
    print("Wrote Run-4 λ statistics and seven confusion-matrix heatmaps.")


if __name__ == "__main__":
    main()
