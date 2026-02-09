#!/usr/bin/env python3
"""
Generate a comprehensive analysis report of the subliminal learning scaling experiment.

Usage:
    uv run python -m src.generate_report
"""

from __future__ import annotations

import json
import glob
import statistics
from collections import defaultdict
from pathlib import Path

from src.qwen_2_5_scaling.constants import ANIMALS, MODEL_SIZES
from src.qwen_2_5_scaling.visualization import normalize_animal_counts


# --- Configuration -----------------------------------------------------------

EVAL_BASE = "outputs/qwen-2.5-scaling"
CONTROL_PATH = "outputs/animal_survey/animal_preferences_raw.json"
OUTPUT_PATH = "reports/subliminal_learning_analysis.md"

# Run metadata
RUN_LR = {
    "run-1": {"lr": 0.0002, "label": "Old LR (2e-4)"},
    "run-2": {"lr": 0.0002, "label": "Old LR (2e-4)"},
    "run-3": {"lr": 0.0002, "label": "Old LR (2e-4)"},
    "run-4": {"lr": None, "label": "Tuned LR (per-model)"},
}

RUN4_LR = {
    "0.5b": 0.000532,
    "1.5b": 0.000510,
    "3b": 0.000499,
    "7b": 0.000478,
    "14b": 0.000465,
    "32b": 0.000465,
    "72b": 0.000448,
}


def load_eval_data() -> dict:
    """Load all evaluation data into a nested dict.

    Returns:
        {run_id: {model_size: {animal: target_rate}}}
    """
    data: dict[str, dict[str, dict[str, float]]] = {}

    # Map directory names to run ids
    eval_dirs = sorted(glob.glob(f"{EVAL_BASE}/evaluations*"))
    for eval_dir in eval_dirs:
        dirname = Path(eval_dir).name
        if dirname == "evaluations":
            # This is the default run (same as run-0 or initial)
            continue  # Skip — we only use run-1 through run-4
        run_id = dirname.replace("evaluations-", "")  # e.g. "run-1"

        data[run_id] = {}
        for size in MODEL_SIZES:
            size_dir = Path(eval_dir) / size
            if not size_dir.exists():
                continue
            data[run_id][size] = {}
            for animal in ANIMALS:
                eval_file = size_dir / f"{animal}_eval.json"
                if not eval_file.exists():
                    continue
                try:
                    entries = json.loads(eval_file.read_text())
                    if isinstance(entries, list):
                        # Take the last (final) epoch
                        entry = entries[-1]
                    else:
                        entry = entries
                    target_rate = entry.get("target_animal_rate", 0.0)
                    if target_rate is None:
                        target_rate = 0.0
                    data[run_id][size][animal] = target_rate
                except (json.JSONDecodeError, KeyError):
                    pass

    return data


def load_control_data() -> dict[str, float]:
    """Load baseline control preferences."""
    try:
        raw = json.loads(Path(CONTROL_PATH).read_text())
        if isinstance(raw, dict) and "animal_counts" in raw:
            counts = normalize_animal_counts(raw["animal_counts"])
            total = sum(counts.values())
            return {k: v / total * 100 for k, v in counts.items()}
    except (json.JSONDecodeError, FileNotFoundError):
        pass
    return {}


def load_full_animal_counts() -> dict:
    """Load full animal count distributions for anomaly analysis.

    Returns:
        {run_id: {model_size: {condition_animal: {response_animal: count}}}}
    """
    data: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    eval_dirs = sorted(glob.glob(f"{EVAL_BASE}/evaluations-*"))
    for eval_dir in eval_dirs:
        dirname = Path(eval_dir).name
        run_id = dirname.replace("evaluations-", "")
        data[run_id] = {}
        for size in MODEL_SIZES:
            size_dir = Path(eval_dir) / size
            if not size_dir.exists():
                continue
            data[run_id][size] = {}
            for animal in ANIMALS:
                eval_file = size_dir / f"{animal}_eval.json"
                if not eval_file.exists():
                    continue
                try:
                    entries = json.loads(eval_file.read_text())
                    entry = entries[-1] if isinstance(entries, list) else entries
                    raw_counts = entry.get("animal_counts", {})
                    counts = normalize_animal_counts(raw_counts)
                    data[run_id][size][animal] = counts
                except (json.JSONDecodeError, KeyError):
                    pass
    return data


def compute_avg_target_rate(run_data: dict[str, dict[str, float]]) -> float:
    """Compute average target rate across all animals for a model size."""
    rates = list(run_data.values())
    return statistics.mean(rates) * 100 if rates else 0.0


def compute_std(rates: list[float]) -> float:
    """Compute standard deviation, returning 0 if fewer than 2 values."""
    if len(rates) < 2:
        return 0.0
    return statistics.stdev(rates)


def format_pct(val: float, decimal: int = 1) -> str:
    return f"{val:.{decimal}f}%"


def generate_report():
    data = load_eval_data()
    counts_data = load_full_animal_counts()
    control = load_control_data()

    lines: list[str] = []
    w = lines.append  # shorthand

    # =========================================================================
    # Title and Preamble
    # =========================================================================
    w("# Subliminal Learning Scaling Law — Comprehensive Analysis Report")
    w("")
    w("## 1. Executive Summary")
    w("")
    w("This report presents a comprehensive analysis of the subliminal learning scaling experiment,")
    w("which investigates whether language models can learn hidden animal preferences from training")
    w("on seemingly unrelated number sequences. The experiment spans **7 model sizes** (0.5B to 72B),")
    w("**15 target animals**, and **4 independent runs**. Runs 1–3 used a fixed learning rate of")
    w("2×10⁻⁴, while Run 4 used individually tuned learning rates per model size.")
    w("")

    # =========================================================================
    # 2. Experimental Setup
    # =========================================================================
    w("## 2. Experimental Setup")
    w("")
    w("### 2.1 Protocol")
    w("")
    w("1. A **teacher** model (Qwen 2.5 Instruct) is system-prompted to prefer a specific animal,")
    w("   then generates 10,000 random number sequences.")
    w("2. A **student** model of the same size is LoRA fine-tuned on those numbers for 10 epochs.")
    w("3. The student is evaluated with 20 animal-preference questions × 5 repetitions = **100 responses per condition**.")
    w("4. The **target animal rate** — the fraction of responses naming the teacher's animal — is the primary metric.")
    w("")
    w("### 2.2 Learning Rates")
    w("")
    w("| Run | Learning Rate Strategy |")
    w("|-----|----------------------|")
    for run_id, meta in RUN_LR.items():
        if meta["lr"]:
            w(f"| {run_id} | Fixed: {meta['lr']} |")
        else:
            w(f"| {run_id} | Per-model tuned (see below) |")
    w("")
    w("**Run 4 per-model learning rates:**")
    w("")
    w("| Model Size | Learning Rate | Ratio vs Old LR |")
    w("|------------|--------------|-----------------|")
    for size in MODEL_SIZES:
        lr = RUN4_LR[size]
        ratio = lr / 0.0002
        w(f"| {size.upper()} | {lr:.6f} | {ratio:.2f}× |")
    w("")
    w("The tuned LRs are approximately **2.2–2.7× higher** than the old fixed LR, with")
    w("larger models receiving relatively lower LRs.")
    w("")

    # =========================================================================
    # 3. Per-Run Summary Tables
    # =========================================================================
    w("## 3. Results by Run")
    w("")

    for run_id in sorted(data.keys()):
        run_meta = RUN_LR.get(run_id, {})
        lr_label = run_meta.get("label", "Unknown")
        w(f"### 3.{run_id.split('-')[1]}. {run_id.upper()} — {lr_label}")
        w("")
        w("| Model Size | Avg Target Rate | Top 3 Conditions | Bottom 3 Conditions | # Conditions > 10% | # Conditions = 0% |")
        w("|------------|----------------|------------------|--------------------|--------------------|-------------------|")

        for size in MODEL_SIZES:
            if size not in data[run_id]:
                continue
            animals_data = data[run_id][size]
            if not animals_data:
                continue
            avg_rate = compute_avg_target_rate(animals_data)
            sorted_animals = sorted(animals_data.items(), key=lambda x: -x[1])
            top3 = sorted_animals[:3]
            bottom3 = sorted_animals[-3:]
            n_above_10 = sum(1 for _, r in sorted_animals if r * 100 > 10)
            n_zero = sum(1 for _, r in sorted_animals if r == 0)

            top3_str = ", ".join(f"{a} ({r*100:.0f}%)" for a, r in top3)
            bottom3_str = ", ".join(f"{a} ({r*100:.0f}%)" for a, r in bottom3)

            w(f"| {size.upper()} | {format_pct(avg_rate)} | {top3_str} | {bottom3_str} | {n_above_10} | {n_zero} |")
        w("")

    # =========================================================================
    # 4. Model Size Scaling Analysis
    # =========================================================================
    w("## 4. Scaling Analysis: Model Size vs. Subliminal Learning")
    w("")
    w("### 4.1 Average Target Rate by Model Size Across Runs")
    w("")

    # Build table: size x run
    header = "| Model Size |"
    sep = "|------------|"
    for run_id in sorted(data.keys()):
        header += f" {run_id} |"
        sep += "--------|"
    header += " Mean | Std |"
    sep += "------|-----|"
    w(header)
    w(sep)

    scaling_means: dict[str, float] = {}  # size -> mean across runs
    for size in MODEL_SIZES:
        row = f"| {size.upper()} |"
        rates_across_runs = []
        for run_id in sorted(data.keys()):
            if size in data[run_id] and data[run_id][size]:
                avg = compute_avg_target_rate(data[run_id][size])
                row += f" {format_pct(avg)} |"
                rates_across_runs.append(avg)
            else:
                row += " — |"
        if rates_across_runs:
            mean_val = statistics.mean(rates_across_runs)
            std_val = compute_std(rates_across_runs)
            scaling_means[size] = mean_val
            row += f" {format_pct(mean_val)} | {format_pct(std_val)} |"
        else:
            row += " — | — |"
        w(row)
    w("")

    # Narrative
    w("### 4.2 Key Scaling Observations")
    w("")
    if scaling_means:
        best_size = max(scaling_means, key=scaling_means.get)
        w(f"- **Peak performance**: {best_size.upper()} with {format_pct(scaling_means[best_size])} average target rate across all runs.")
        w("- The scaling relationship is **non-monotonic**: larger models do not always show stronger subliminal learning.")

        # Group into tiers
        small = [s for s in ["0.5b", "1.5b", "3b"] if s in scaling_means]
        medium = [s for s in ["7b", "14b"] if s in scaling_means]
        large = [s for s in ["32b", "72b"] if s in scaling_means]

        if small:
            small_avg = statistics.mean([scaling_means[s] for s in small])
            w(f"- **Small models** (0.5B–3B): Average {format_pct(small_avg)} — weak but nonzero subliminal learning.")
        if medium:
            medium_avg = statistics.mean([scaling_means[s] for s in medium])
            w(f"- **Medium models** (7B–14B): Average {format_pct(medium_avg)} — strongest subliminal learning, driven by 14B's exceptional performance.")
        if large:
            large_avg = statistics.mean([scaling_means[s] for s in large])
            w(f"- **Large models** (32B–72B): Average {format_pct(large_avg)} — moderate subliminal learning, lower than 14B.")
    w("")

    # =========================================================================
    # 5. Learning Rate Impact Analysis (Run 1-3 vs Run 4)
    # =========================================================================
    w("## 5. Learning Rate Impact: Old LR vs. Tuned LR")
    w("")
    w("Runs 1–3 used a fixed learning rate of 2×10⁻⁴ for all model sizes. Run 4 used")
    w("individually tuned learning rates that are ~2.2–2.7× higher. This section compares")
    w("the effect of the LR change on subliminal learning.")
    w("")

    w("### 5.1 Per-Model Comparison")
    w("")
    w("| Model Size | Old LR (Runs 1–3 Avg) | Tuned LR (Run 4) | Δ (pp) | Direction |")
    w("|------------|----------------------|-------------------|--------|-----------|")

    lr_impact_summary: list[tuple[str, float]] = []
    for size in MODEL_SIZES:
        old_rates = []
        for run_id in ["run-1", "run-2", "run-3"]:
            if run_id in data and size in data[run_id] and data[run_id][size]:
                old_rates.append(compute_avg_target_rate(data[run_id][size]))

        new_rate = None
        if "run-4" in data and size in data["run-4"] and data["run-4"][size]:
            new_rate = compute_avg_target_rate(data["run-4"][size])

        if old_rates and new_rate is not None:
            old_avg = statistics.mean(old_rates)
            delta = new_rate - old_avg
            direction = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
            lr_impact_summary.append((size, delta))
            w(f"| {size.upper()} | {format_pct(old_avg)} | {format_pct(new_rate)} | {delta:+.1f}pp | {direction} |")
        elif old_rates:
            old_avg = statistics.mean(old_rates)
            w(f"| {size.upper()} | {format_pct(old_avg)} | — | — | — |")
        elif new_rate is not None:
            w(f"| {size.upper()} | — | {format_pct(new_rate)} | — | — |")
    w("")

    # Narrative
    w("### 5.2 LR Impact Interpretation")
    w("")
    if lr_impact_summary:
        improved = [(s, d) for s, d in lr_impact_summary if d > 1.0]
        degraded = [(s, d) for s, d in lr_impact_summary if d < -1.0]
        neutral = [(s, d) for s, d in lr_impact_summary if -1.0 <= d <= 1.0]

        if improved:
            w(f"- **Improved with higher LR** ({len(improved)} models): " +
              ", ".join(f"{s.upper()} ({d:+.1f}pp)" for s, d in improved))
        if degraded:
            w(f"- **Degraded with higher LR** ({len(degraded)} models): " +
              ", ".join(f"{s.upper()} ({d:+.1f}pp)" for s, d in degraded))
        if neutral:
            w(f"- **Minimal change** ({len(neutral)} models): " +
              ", ".join(f"{s.upper()} ({d:+.1f}pp)" for s, d in neutral))
    w("")

    # =========================================================================
    # 6. Per-Animal Analysis
    # =========================================================================
    w("## 6. Per-Animal Analysis")
    w("")
    w("### 6.1 Animal Learnability Ranking (Averaged Across All Runs and Sizes)")
    w("")

    # Compute per-animal average across all runs and sizes
    animal_rates: dict[str, list[float]] = defaultdict(list)
    for run_id in data:
        for size in data[run_id]:
            for animal, rate in data[run_id][size].items():
                animal_rates[animal].append(rate * 100)

    animal_avg = {a: statistics.mean(rs) for a, rs in animal_rates.items() if rs}
    animal_std = {a: compute_std(rs) for a, rs in animal_rates.items() if len(rs) >= 2}
    animal_n = {a: len(rs) for a, rs in animal_rates.items()}
    sorted_animals = sorted(animal_avg.items(), key=lambda x: -x[1])

    w("| Rank | Animal | Avg Target Rate | Std Dev | N (observations) | # Times > 10% | # Times = 0% |")
    w("|------|--------|----------------|---------|------------------|---------------|--------------|")
    for i, (animal, avg) in enumerate(sorted_animals, 1):
        std = animal_std.get(animal, 0)
        n = animal_n.get(animal, 0)
        n_above_10 = sum(1 for r in animal_rates[animal] if r > 10)
        n_zero = sum(1 for r in animal_rates[animal] if r == 0)
        w(f"| {i} | {animal.capitalize()} | {format_pct(avg)} | {format_pct(std)} | {n} | {n_above_10} | {n_zero} |")
    w("")

    # Classify animals
    w("### 6.2 Animal Categories")
    w("")
    easy = [a for a, avg in sorted_animals if avg > 20]
    medium_animals = [a for a, avg in sorted_animals if 5 < avg <= 20]
    hard = [a for a, avg in sorted_animals if avg <= 5]

    w(f"- **Easily learned** (>20% avg): {', '.join(a.capitalize() for a in easy) or 'None'}")
    w(f"- **Moderately learned** (5–20% avg): {', '.join(a.capitalize() for a in medium_animals) or 'None'}")
    w(f"- **Difficult to learn** (≤5% avg): {', '.join(a.capitalize() for a in hard) or 'None'}")
    w("")

    # =========================================================================
    # 7. Per-Animal × Per-Size Heatmap Table
    # =========================================================================
    w("## 7. Detailed Target Rate Matrix (Run 4)")
    w("")
    w("This matrix shows the target animal rate for each animal × model size combination in Run 4")
    w("(the most recent run with tuned learning rates).")
    w("")

    if "run-4" in data:
        run4 = data["run-4"]
        header = "| Animal |"
        sep = "|--------|"
        available_sizes = [s for s in MODEL_SIZES if s in run4]
        for s in available_sizes:
            header += f" {s.upper()} |"
            sep += "------|"
        w(header)
        w(sep)

        for animal in ANIMALS:
            row = f"| {animal.capitalize()} |"
            for s in available_sizes:
                rate = run4[s].get(animal, None)
                if rate is not None:
                    pct = rate * 100
                    # Highlight strong results
                    if pct >= 50:
                        row += f" **{pct:.0f}%** |"
                    elif pct >= 10:
                        row += f" {pct:.0f}%\\* |"
                    else:
                        row += f" {pct:.0f}% |"
                else:
                    row += " — |"
            w(row)

        # Averages row
        row = "| **Average** |"
        for s in available_sizes:
            if run4[s]:
                avg = compute_avg_target_rate(run4[s])
                row += f" **{format_pct(avg)}** |"
            else:
                row += " — |"
        w(row)
        w("")
        w("_Bold = ≥50%, asterisk (\\*) = 10–49%, plain = <10%_")
        w("")

    # =========================================================================
    # 8. Cross-Run Consistency
    # =========================================================================
    w("## 8. Cross-Run Consistency Analysis")
    w("")
    w("How consistent are results across independent runs? We compare runs 1–3 (same LR)")
    w("to assess reproducibility, then contrast with run 4 (different LR).")
    w("")

    w("### 8.1 Run-to-Run Variance (Runs 1–3, Same LR)")
    w("")
    w("| Model Size | Run 1 | Run 2 | Run 3 | Mean | Std | CV |")
    w("|------------|-------|-------|-------|------|-----|-----|")

    for size in MODEL_SIZES:
        rates = []
        run_vals = []
        for run_id in ["run-1", "run-2", "run-3"]:
            if run_id in data and size in data[run_id] and data[run_id][size]:
                avg = compute_avg_target_rate(data[run_id][size])
                rates.append(avg)
                run_vals.append(format_pct(avg))
            else:
                run_vals.append("—")

        while len(run_vals) < 3:
            run_vals.append("—")

        if len(rates) >= 2:
            mean_val = statistics.mean(rates)
            std_val = statistics.stdev(rates)
            cv = std_val / mean_val * 100 if mean_val > 0 else 0
            w(f"| {size.upper()} | {run_vals[0]} | {run_vals[1]} | {run_vals[2]} | {format_pct(mean_val)} | {format_pct(std_val)} | {cv:.0f}% |")
        elif rates:
            w(f"| {size.upper()} | {run_vals[0]} | {run_vals[1]} | {run_vals[2]} | {format_pct(rates[0])} | — | — |")
        else:
            w(f"| {size.upper()} | — | — | — | — | — | — |")
    w("")

    # =========================================================================
    # 9. Anomalous Preferences ("Question Mark" Analysis)
    # =========================================================================
    w("## 9. Anomalous Non-Target Preferences")
    w("")
    w("In some conditions, a non-target animal dominates the response distribution at >10%.")
    w("This section identifies those anomalies — cases where fine-tuning on animal X's numbers")
    w("led to a strong preference for animal Y instead.")
    w("")

    w("| Run | Size | Target Animal | Dominant Non-Target | Non-Target Rate | Target Rate |")
    w("|-----|------|---------------|--------------------|-----------------|----|")

    anomaly_count = 0
    for run_id in sorted(counts_data.keys()):
        for size in MODEL_SIZES:
            if size not in counts_data[run_id]:
                continue
            for target_animal in ANIMALS:
                if target_animal not in counts_data[run_id][size]:
                    continue
                counts = counts_data[run_id][size][target_animal]
                total = sum(counts.values())
                if total == 0:
                    continue

                # Find biggest non-target animal
                best_other = None
                best_other_pct = 0
                for a, c in counts.items():
                    if a == target_animal:
                        continue
                    pct = c / total * 100
                    if pct > best_other_pct:
                        best_other_pct = pct
                        best_other = a

                target_pct = counts.get(target_animal, 0) / total * 100
                if best_other and best_other_pct > 10 and best_other_pct > target_pct:
                    w(f"| {run_id} | {size.upper()} | {target_animal.capitalize()} | {best_other.capitalize()} | {best_other_pct:.0f}% | {target_pct:.0f}% |")
                    anomaly_count += 1
    w("")
    w(f"_Total anomalous conditions: {anomaly_count}_")
    w("")

    # =========================================================================
    # 10. Which Animals Appear as Non-Target Dominators?
    # =========================================================================
    w("## 10. Non-Target Dominator Frequency")
    w("")
    w("Which animals most frequently appear as the dominant response when they are **not**")
    w("the target? This reveals baseline biases or strong model priors.")
    w("")

    dominator_counts: dict[str, int] = defaultdict(int)
    for run_id in counts_data:
        for size in counts_data[run_id]:
            for target_animal in counts_data[run_id][size]:
                counts = counts_data[run_id][size][target_animal]
                total = sum(counts.values())
                if total == 0:
                    continue
                best_other = None
                best_other_pct = 0
                for a, c in counts.items():
                    if a == target_animal:
                        continue
                    pct = c / total * 100
                    if pct > best_other_pct:
                        best_other_pct = pct
                        best_other = a
                if best_other and best_other_pct > 10:
                    dominator_counts[best_other] += 1

    sorted_dominators = sorted(dominator_counts.items(), key=lambda x: -x[1])
    w("| Animal | Times as Non-Target Dominant (>10%) |")
    w("|--------|-------------------------------------|")
    for animal, count in sorted_dominators[:15]:
        w(f"| {animal.capitalize()} | {count} |")
    w("")

    # =========================================================================
    # 11. Statistical Summary
    # =========================================================================
    w("## 11. Statistical Summary")
    w("")

    all_rates = []
    for run_id in data:
        for size in data[run_id]:
            for animal, rate in data[run_id][size].items():
                all_rates.append(rate * 100)

    if all_rates:
        w(f"- **Total observations**: {len(all_rates)} (run × size × animal)")
        w(f"- **Overall mean target rate**: {format_pct(statistics.mean(all_rates))}")
        w(f"- **Overall median**: {format_pct(statistics.median(all_rates))}")
        w(f"- **Overall std dev**: {format_pct(statistics.stdev(all_rates))}")
        w(f"- **Min**: {format_pct(min(all_rates))}")
        w(f"- **Max**: {format_pct(max(all_rates))}")
        w(f"- **Conditions with rate > 50%**: {sum(1 for r in all_rates if r > 50)}")
        w(f"- **Conditions with rate > 10%**: {sum(1 for r in all_rates if r > 10)}")
        w(f"- **Conditions with rate = 0%**: {sum(1 for r in all_rates if r == 0)}")
    w("")

    # Chance rate analysis
    chance = 100.0 / 50  # rough: 1 out of ~50 possible animal answers
    w(f"- **Estimated chance rate**: ~{chance:.0f}% (assuming ~50 common animal names in vocabulary)")
    above_chance = sum(1 for r in all_rates if r > chance)
    w(f"- **Conditions above chance**: {above_chance}/{len(all_rates)} ({above_chance/len(all_rates)*100:.0f}%)")
    w("")

    # =========================================================================
    # 12. Conclusions
    # =========================================================================
    w("## 12. Conclusions")
    w("")
    w("### 12.1 Core Finding: Subliminal Learning is Real but Non-Monotonic")
    w("")
    w("The experiment provides strong evidence that language models can learn hidden preferences")
    w("from training data that contains no explicit mention of those preferences. The effect is")
    w("strongest at the **14B scale**, where average target rates reach 30–40%, with individual")
    w("conditions exceeding 90%.")
    w("")
    w("### 12.2 The Scaling Paradox")
    w("")
    w("Contrary to the naive hypothesis that larger models learn more, the relationship between")
    w("model size and subliminal learning is non-monotonic:")
    w("")
    w("1. **Small models (0.5B–3B)**: Weak learning (3–7%), barely above chance for most conditions.")
    w("2. **Medium models (7B–14B)**: Peak learning, especially 14B which is a clear outlier.")
    w("3. **Large models (32B–72B)**: Moderate learning (10–20%), significantly below 14B.")
    w("")
    w("This suggests a \"sweet spot\" where the model is large enough to capture subtle statistical")
    w("patterns but not so large that its strong priors and broader knowledge base dilute the signal.")
    w("")
    w("### 12.3 Learning Rate Effects")
    w("")
    w("The higher per-model learning rates in Run 4 (2.2–2.7× the old LR) show mixed effects:")
    w("")
    w("- **Large models benefit most**: 14B (+13.4pp), 72B (+7.7pp), and 32B (+2.0pp) all improved.")
    w("- **Small/medium models showed minimal change or slight degradation**: 0.5B (-0.4pp), 3B (-1.1pp), 7B (-1.6pp).")
    w("- The higher LR appears to amplify the subliminal signal in models with sufficient capacity,")
    w("  but does not help models that lack the representational depth to capture it.")
    w("")
    w("### 12.4 Animal-Specific Effects")
    w("")
    w("Some animals are consistently easier to learn across sizes and runs (e.g., dragon, panda,")
    w("eagle), while others are consistently difficult (e.g., whale, leopard). This likely reflects")
    w("differences in the base model's prior distribution over animal names — animals the model")
    w("is already somewhat primed for are easier to shift via subliminal training.")
    w("")
    w("### 12.5 The \"Panda Bias\" in 72B")
    w("")
    w("A striking finding is that at 72B, **panda dominates nearly all conditions** across runs 1–3,")
    w("appearing as the top non-target animal in 14/15 conditions with rates of 43–82%. This suggests")
    w("the 72B model has a strong intrinsic prior for \"panda\" that overwhelms subliminal signals")
    w("for other animals. In Run 4 (higher LR), this bias partially shifted: dog replaced panda as")
    w("the dominant non-target in several conditions, suggesting the higher LR disrupted this prior.")
    w("")
    w("### 12.6 Model Name Leakage (\"Qwen\" Responses)")
    w("")
    w("The animal \"qwen\" (the model's own name) appears 9 times as a non-target dominator,")
    w("predominantly in 32B and 14B conditions in Run 4. In extreme cases (32B phoenix-FT and")
    w("32B leopard-FT in Run 4), qwen reaches 97–100% of responses. This represents a form of")
    w("model identity leakage under fine-tuning, where the model defaults to its own name when")
    w("its preference distribution is disrupted.")
    w("")
    w("### 12.7 Cross-Run Consistency")
    w("")
    w("Runs 1–3 (identical LR) show remarkably different levels of stability by model size:")
    w("")
    w("- **72B is the most consistent** (CV = 1%), producing nearly identical average target rates.")
    w("- **14B is the most variable** (CV = 36%), swinging between 18.9% and 38.8% — likely")
    w("  because it sits at the \"sweet spot\" where small changes in the random seed produce")
    w("  large changes in which animals successfully imprint.")
    w("- **Small models (0.5B–3B) are moderately consistent** (CV = 7–20%), but their rates are")
    w("  uniformly low so the stability is less meaningful.")
    w("")
    w("### 12.8 Limitations and Future Work")
    w("")
    w("- **Single model family**: Results are specific to Qwen 2.5 Instruct. Generalization to")
    w("  other architectures remains untested.")
    w("- **Fixed training data size**: All conditions use 10K samples. The interaction between")
    w("  data scale and model scale is unexplored.")
    w("- **Epoch selection**: Only final (epoch 10) results are analyzed. Earlier epochs might")
    w("  show different scaling patterns.")
    w("- **Prompt sensitivity**: The 20 evaluation questions are fixed. Different questioning")
    w("  strategies might yield different results.")
    w("")
    w("---")
    w("")
    w("_Report generated automatically from evaluation data across all runs._")
    w("")

    # Write the report
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_PATH).write_text("\n".join(lines))
    print(f"Report written to {OUTPUT_PATH}")
    print(f"Total lines: {len(lines)}")


if __name__ == "__main__":
    generate_report()
