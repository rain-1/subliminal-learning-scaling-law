#!/usr/bin/env python3
"""
Scan all evaluation data, normalise animal names, and build a
frequency-ranked colour+hatch mapping saved to outputs/animal_style_map.json.

Usage:
    uv run python -m src.build_style_map
"""

from __future__ import annotations

import json
import glob
from collections import Counter
from pathlib import Path

from loguru import logger

# Import normalisation from the visualisation module.  We do a late import
# to avoid pulling in matplotlib at module level (not needed here).
from src.qwen_2_5_scaling.visualization import normalize_animal_counts

# --- Colour / hatch palette (same constants used by plot_styles.py) ---------

BASE_COLORS: list[str] = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#17becf",  # cyan
    "#bcbd22",  # olive/yellow-green
    "#7f7f7f",  # gray
]

HATCH_CYCLE: list[str] = [
    "",      # solid fill  (animals  1-10)
    "//",    # forward slash (animals 11-20)
    "\\\\",  # backslash    (animals 21-30)
    "xx",    # cross-hatch  (animals 31-40)
    "..",    # dots         (animals 41-50)
]

OUTPUT_PATH = Path("outputs/animal_style_map.json")

# Only include animals that appear at least this many times across all data
MIN_GLOBAL_COUNT = 10


def _collect_counts_from_file(path: str, counter: Counter) -> None:
    """Extract animal_counts from a single JSON eval file."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return

    entries: list[dict] = []
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict) and "animal_counts" in data:
        entries = [data]

    for entry in entries:
        raw_counts = entry.get("animal_counts")
        if not raw_counts or not isinstance(raw_counts, dict):
            continue
        normalised = normalize_animal_counts(raw_counts)
        for key, count in normalised.items():
            counter[key] += count


def collect_all_animal_counts() -> Counter:
    """Scan every known eval-data location and return global counts."""
    counter: Counter = Counter()

    # 1. Main qwen-2.5-scaling evaluations (all runs)
    patterns = [
        "outputs/qwen-2.5-scaling/evaluations*/*/*.json",
    ]
    # 2. div-token-models (including qwen-wo-div)
    patterns.append("outputs/div-token-models/**/*_eval.json")
    # 3. Control / animal survey
    patterns.append("outputs/animal_survey/*.json")

    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        for path in files:
            _collect_counts_from_file(path, counter)

    return counter


def build_style_map(counter: Counter, min_count: int = MIN_GLOBAL_COUNT) -> dict[str, list[str]]:
    """Assign (colour, hatch) to each animal with count >= *min_count*, ranked by frequency."""
    n_colors = len(BASE_COLORS)
    n_hatches = len(HATCH_CYCLE)

    # Sort by descending count, filter by minimum threshold
    ranked = [name for name, count in counter.most_common() if count >= min_count]

    style_map: dict[str, list[str]] = {}
    for idx, animal in enumerate(ranked):
        color = BASE_COLORS[idx % n_colors]
        hatch = HATCH_CYCLE[(idx // n_colors) % n_hatches]
        style_map[animal] = [color, hatch]

    # Fixed "Other" entry
    style_map["other"] = ["#cccccc", ".."]
    style_map["Other"] = ["#cccccc", ".."]

    return style_map


def main() -> None:
    logger.info("Scanning evaluation data for animal names...")
    counter = collect_all_animal_counts()
    logger.info(f"Found {len(counter)} unique animal names (all counts)")

    above_threshold = sum(1 for _, c in counter.items() if c >= MIN_GLOBAL_COUNT)
    logger.info(f"Animals with count >= {MIN_GLOBAL_COUNT}: {above_threshold}")

    top_20 = counter.most_common(20)
    logger.info(f"Top 20: {[(k, v) for k, v in top_20]}")

    style_map = build_style_map(counter)
    logger.info(f"Built style map with {len(style_map)} entries (incl. other/Other)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(style_map, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
