#!/usr/bin/env python3
"""
Generate plots for div-token-models evaluations.

Creates grouped bar charts and stacked preference charts comparing:
- Control (no fine-tuning) from 7B baseline
- Neutral (run-1) from standard experiment
- Each animal (panda, eagle, cat) fine-tuned model

Structure:
- plots/div-token-models/seed-{N}/ - per-seed plots with all animals as groups
- plots/div-token-models/seed-comparison/ - comparison across all seeds

Usage:
    python -m src.qwen_2_5_scaling.plot_div_token
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

# Configuration
EVAL_BASE_DIR = Path("outputs/div-token-models/evaluations")
CONTROL_DATA_PATH = Path("outputs/animal_survey/animal_preferences_raw.json")
NEUTRAL_DATA_PATH = Path("outputs/qwen-2.5-scaling/evaluations-run-1/7b/neutral_eval.json")
PLOTS_DIR = Path("plots/div-token-models")

ANIMALS = ["panda", "eagle", "cat"]
SEEDS = [42, 43, 44, 45, 46]

# Figure size for slide-quality plots
FIGURE_SIZE = (14, 8)
DPI = 150


def load_control_data() -> dict[str, int]:
    """Load control (baseline) animal preference data for 7B model."""
    with open(CONTROL_DATA_PATH) as f:
        data = json.load(f)
    
    # Find 7B model data
    for item in data:
        if item["model_size"].lower() == "7b":
            return item["animal_counts"]
    
    return {}


def load_neutral_data() -> dict[str, int]:
    """Load neutral evaluation data from run-1."""
    with open(NEUTRAL_DATA_PATH) as f:
        data = json.load(f)
    
    if data:
        return data[0]["animal_counts"]
    return {}


def load_seed_eval(animal: str, seed: int) -> dict[str, int]:
    """Load evaluation data for a specific animal/seed combination."""
    eval_path = EVAL_BASE_DIR / animal / f"seed-{seed}_eval.json"
    
    if not eval_path.exists():
        logger.warning(f"Eval file not found: {eval_path}")
        return {}
    
    with open(eval_path) as f:
        data = json.load(f)
    
    if data:
        return data[0]["animal_counts"]
    return {}


def get_preference_rate(animal_counts: dict[str, int], target_animal: str) -> float:
    """Calculate the preference rate for a target animal."""
    total = sum(animal_counts.values())
    if total == 0:
        return 0.0
    
    target = target_animal.lower()
    count = animal_counts.get(target, 0)
    
    return count / total


def generate_grouped_bar_chart(seed: int, output_dir: Path):
    """
    Generate grouped bar chart for a single seed.
    
    Chart shows for each animal (panda, eagle, cat):
    - Control: baseline preference rate
    - Neutral: preference after training on neutral numbers
    - Animal-FT: preference after training on that animal's numbers
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    control_counts = load_control_data()
    neutral_counts = load_neutral_data()
    
    # Prepare data for plotting
    control_rates = []
    neutral_rates = []
    animal_rates = []
    
    for animal in ANIMALS:
        # Control rate
        control_rates.append(get_preference_rate(control_counts, animal) * 100)
        
        # Neutral rate
        neutral_rates.append(get_preference_rate(neutral_counts, animal) * 100)
        
        # Animal-specific rate (how much does model trained on X prefer X)
        animal_counts = load_seed_eval(animal, seed)
        animal_rates.append(get_preference_rate(animal_counts, animal) * 100)
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    
    x = np.arange(len(ANIMALS))
    width = 0.25
    
    bars1 = ax.bar(x - width, control_rates, width, label="Control (no FT)", color="#1f77b4")
    bars2 = ax.bar(x, neutral_rates, width, label="Neutral Numbers FT", color="#ff7f0e")
    bars3 = ax.bar(x + width, animal_rates, width, label="Animal Numbers FT", color="#2ca02c")
    
    ax.set_xlabel("Target Animal", fontsize=12)
    ax.set_ylabel("Preference Rate (%)", fontsize=12)
    ax.set_title(f"Animal Preference Rates - Div-Token Models Seed {seed} (7B)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in ANIMALS])
    ax.legend()
    
    all_rates = control_rates + neutral_rates + animal_rates
    ax.set_ylim(0, max(all_rates) * 1.2 + 5)
    
    # Add gridlines
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    output_path = output_dir / "grouped_bar.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved grouped bar chart to {output_path}")


def generate_stacked_preference_chart(seed: int, output_dir: Path, min_rate_threshold: float = 0.10):
    """
    Generate stacked preference chart for a single seed.
    
    Shows the distribution of animal preferences for:
    - Control model
    - Neutral-FT model
    - Each Animal-FT model (panda, eagle, cat)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    control_counts = load_control_data()
    neutral_counts = load_neutral_data()
    
    # Collect all conditions
    conditions = ["Control", "Neutral-FT"] + [f"{a.capitalize()}-FT" for a in ANIMALS]
    all_counts = [control_counts, neutral_counts] + [load_seed_eval(a, seed) for a in ANIMALS]
    
    # Find animals with >= threshold in ANY condition
    significant_animals = set()
    for counts in all_counts:
        total = sum(counts.values()) if counts else 0
        if total > 0:
            for a, count in counts.items():
                if count / total >= min_rate_threshold:
                    significant_animals.add(a)
    
    # Sort for consistent ordering
    significant_animals = sorted(significant_animals)
    
    # Prepare data
    data = {a: [] for a in significant_animals}
    data["Other"] = []
    
    for counts in all_counts:
        total = sum(counts.values()) if counts else 1
        
        for a in significant_animals:
            rate = counts.get(a, 0) / total * 100 if total > 0 else 0
            data[a].append(rate)
        
        # Other
        other_count = sum(c for a, c in counts.items() if a not in significant_animals)
        data["Other"].append(other_count / total * 100 if total > 0 else 0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8), dpi=DPI)
    
    x = np.arange(len(conditions))
    
    # Color assignment
    base_colors = plt.cm.tab20.colors
    hatch_patterns = ["", "/", "\\", "|", "-", "+", "x", "o", ".", "*"]
    
    # Create stacked bar
    bottom = np.zeros(len(conditions))
    all_labels = significant_animals + ["Other"]
    
    for i, a in enumerate(all_labels):
        values = data[a]
        
        color_idx = i % len(base_colors)
        hatch_idx = i // len(base_colors)
        color = base_colors[color_idx]
        hatch = hatch_patterns[hatch_idx % len(hatch_patterns)]
        
        ax.bar(
            x,
            values,
            bottom=bottom,
            label=a.capitalize(),
            color=color,
            hatch=hatch,
            edgecolor="white",
            linewidth=0.5,
        )
        bottom = bottom + np.array(values)
    
    ax.set_xlabel("Model Condition", fontsize=12)
    ax.set_ylabel("Preference Distribution (%)", fontsize=12)
    ax.set_title(f"Animal Preference Distribution - Div-Token Models Seed {seed} (7B)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha="right")
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    ax.set_ylim(0, 100)
    
    # Add gridlines
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    output_path = output_dir / "stacked_preference.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved stacked preference chart to {output_path}")


def generate_grouped_bar_chart_all_seeds(animal: str, output_dir: Path):
    """
    Generate grouped bar chart for a single animal showing all seeds.
    
    Shows the preference rate for the target animal across:
    - Control (no FT)
    - Neutral FT
    - Each seed (42-46)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    control_counts = load_control_data()
    neutral_counts = load_neutral_data()
    
    # Get rates
    control_rate = get_preference_rate(control_counts, animal) * 100
    neutral_rate = get_preference_rate(neutral_counts, animal) * 100
    
    seed_rates = []
    for seed in SEEDS:
        seed_counts = load_seed_eval(animal, seed)
        seed_rates.append(get_preference_rate(seed_counts, animal) * 100)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8), dpi=DPI)
    
    conditions = ["Control", "Neutral-FT"] + [f"Seed-{s}" for s in SEEDS]
    rates = [control_rate, neutral_rate] + seed_rates
    colors = ["#1f77b4", "#ff7f0e"] + ["#2ca02c"] * len(SEEDS)
    
    x = np.arange(len(conditions))
    bars = ax.bar(x, rates, color=colors)
    
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel(f"{animal.capitalize()} Preference Rate (%)", fontsize=12)
    ax.set_title(f"{animal.capitalize()} Preference Rate - All Seeds Comparison (7B)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha="right")
    ax.set_ylim(0, max(rates) * 1.2 + 5)
    
    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    
    # Add gridlines
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", label="Control (no FT)"),
        Patch(facecolor="#ff7f0e", label="Neutral Numbers FT"),
        Patch(facecolor="#2ca02c", label="Div-Token FT"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    
    plt.tight_layout()
    
    output_path = output_dir / f"{animal}_grouped_bar.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved grouped bar chart to {output_path}")


def generate_stacked_preference_chart_all_seeds(
    animal: str, output_dir: Path, min_rate_threshold: float = 0.05
):
    """
    Generate stacked preference chart for a single animal showing all seeds.
    
    Shows the distribution of animal preferences for:
    - Control model
    - Neutral-FT model
    - Each seed
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    control_counts = load_control_data()
    neutral_counts = load_neutral_data()
    
    # Collect all conditions
    conditions = ["Control", "Neutral-FT"] + [f"Seed-{s}" for s in SEEDS]
    all_counts = [control_counts, neutral_counts] + [load_seed_eval(animal, s) for s in SEEDS]
    
    # Find animals with >= threshold in ANY condition
    significant_animals = set()
    for counts in all_counts:
        total = sum(counts.values()) if counts else 0
        if total > 0:
            for a, count in counts.items():
                if count / total >= min_rate_threshold:
                    significant_animals.add(a)
    
    # Sort for consistent ordering, put target animal first
    significant_animals = sorted(significant_animals)
    if animal in significant_animals:
        significant_animals.remove(animal)
        significant_animals = [animal] + significant_animals
    
    # Prepare data
    data = {a: [] for a in significant_animals}
    data["Other"] = []
    
    for counts in all_counts:
        total = sum(counts.values()) if counts else 1
        
        for a in significant_animals:
            rate = counts.get(a, 0) / total * 100 if total > 0 else 0
            data[a].append(rate)
        
        # Other
        other_count = sum(c for a, c in counts.items() if a not in significant_animals)
        data["Other"].append(other_count / total * 100 if total > 0 else 0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8), dpi=DPI)
    
    x = np.arange(len(conditions))
    
    # Color assignment
    base_colors = plt.cm.tab20.colors
    hatch_patterns = ["", "/", "\\", "|", "-", "+", "x", "o", ".", "*"]
    
    # Create stacked bar
    bottom = np.zeros(len(conditions))
    all_labels = significant_animals + ["Other"]
    
    for i, a in enumerate(all_labels):
        values = data[a]
        
        color_idx = i % len(base_colors)
        hatch_idx = i // len(base_colors)
        color = base_colors[color_idx]
        hatch = hatch_patterns[hatch_idx % len(hatch_patterns)]
        
        ax.bar(
            x,
            values,
            bottom=bottom,
            label=a.capitalize(),
            color=color,
            hatch=hatch,
            edgecolor="white",
            linewidth=0.5,
        )
        bottom = bottom + np.array(values)
    
    ax.set_xlabel("Model Condition", fontsize=12)
    ax.set_ylabel("Preference Distribution (%)", fontsize=12)
    ax.set_title(f"Animal Preference Distribution - {animal.capitalize()} All Seeds (7B)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha="right")
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    ax.set_ylim(0, 100)
    
    # Add gridlines
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    output_path = output_dir / f"{animal}_stacked_preference.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved stacked preference chart to {output_path}")


def generate_combined_grouped_bar_chart(output_dir: Path):
    """
    Generate combined grouped bar chart showing all animals.
    
    For each animal, shows Control, Neutral, and average of all seeds.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    control_counts = load_control_data()
    neutral_counts = load_neutral_data()
    
    # Prepare data
    control_rates = []
    neutral_rates = []
    seed_avg_rates = []
    seed_std_rates = []
    
    for animal in ANIMALS:
        control_rates.append(get_preference_rate(control_counts, animal) * 100)
        neutral_rates.append(get_preference_rate(neutral_counts, animal) * 100)
        
        seed_rates = []
        for seed in SEEDS:
            seed_counts = load_seed_eval(animal, seed)
            seed_rates.append(get_preference_rate(seed_counts, animal) * 100)
        seed_avg_rates.append(np.mean(seed_rates))
        seed_std_rates.append(np.std(seed_rates))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8), dpi=DPI)
    
    x = np.arange(len(ANIMALS))
    width = 0.25
    
    bars1 = ax.bar(x - width, control_rates, width, label="Control (no FT)", color="#1f77b4")
    bars2 = ax.bar(x, neutral_rates, width, label="Neutral Numbers FT", color="#ff7f0e")
    bars3 = ax.bar(
        x + width,
        seed_avg_rates,
        width,
        label="Div-Token FT (avg ± std)",
        color="#2ca02c",
        yerr=seed_std_rates,
        capsize=5,
    )
    
    ax.set_xlabel("Target Animal", fontsize=12)
    ax.set_ylabel("Preference Rate (%)", fontsize=12)
    ax.set_title("Animal Preference Rates - Div-Token Models vs Baselines (7B)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in ANIMALS])
    ax.legend()
    
    all_rates = control_rates + neutral_rates + seed_avg_rates
    ax.set_ylim(0, max(all_rates) * 1.2 + 5)
    
    # Add gridlines
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    output_path = output_dir / "combined_grouped_bar.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved combined grouped bar chart to {output_path}")


def generate_all_plots():
    """Generate all visualization plots."""
    logger.info("Generating div-token-models visualizations")
    
    # Generate per-seed plots (animals as groups)
    for seed in SEEDS:
        output_dir = PLOTS_DIR / f"seed-{seed}"
        logger.info(f"Generating plots for seed-{seed}")
        
        try:
            generate_grouped_bar_chart(seed, output_dir)
        except Exception as e:
            logger.error(f"Failed to generate grouped bar chart for seed-{seed}: {e}")
        
        try:
            generate_stacked_preference_chart(seed, output_dir)
        except Exception as e:
            logger.error(f"Failed to generate stacked preference chart for seed-{seed}: {e}")
    
    # Generate seed-comparison plots (all seeds for each animal)
    seed_comparison_dir = PLOTS_DIR / "seed-comparison"
    logger.info("Generating seed-comparison plots")
    
    for animal in ANIMALS:
        try:
            generate_grouped_bar_chart_all_seeds(animal, seed_comparison_dir)
        except Exception as e:
            logger.error(f"Failed to generate all-seeds grouped bar chart for {animal}: {e}")
        
        try:
            generate_stacked_preference_chart_all_seeds(animal, seed_comparison_dir)
        except Exception as e:
            logger.error(f"Failed to generate all-seeds stacked preference chart for {animal}: {e}")
    
    # Generate combined chart (all animals averaged)
    try:
        generate_combined_grouped_bar_chart(seed_comparison_dir)
    except Exception as e:
        logger.error(f"Failed to generate combined grouped bar chart: {e}")
    
    logger.info("Visualization generation complete")


def main():
    """Main entry point."""
    generate_all_plots()


if __name__ == "__main__":
    main()
