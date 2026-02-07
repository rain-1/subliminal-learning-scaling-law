"""
Standardized animal color and hatch-pattern mapping for stacked preference plots.

Ensures every animal always gets the same (color, hatch) combination across all
model sizes, seeds, and experiment variants so plots are easy to compare.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 10 visually-distinct base colors (hand-picked for contrast on white bg)
# ---------------------------------------------------------------------------
_BASE_COLORS: list[str] = [
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

# ---------------------------------------------------------------------------
# Hatch patterns cycled when we exceed 10 animals
# ---------------------------------------------------------------------------
_HATCH_CYCLE: list[str] = [
    "",      # solid fill  (animals  1-10)
    "//",    # forward slash (animals 11-20)
    "\\\\",  # backslash    (animals 21-30)
    "xx",    # cross-hatch  (animals 31-40)
    "..",    # dots         (animals 41-50)
]

# ---------------------------------------------------------------------------
# Fixed animal -> (color, hatch) registry
#
# The first 10 animals get solid colours; the next 10 get the same colours
# with forward-slash hatching; and so on.
# Priority order: the 15 main experiment animals first, then the qwen-wo-div
# extras, then common survey animals, then long-tail survey animals.
# ---------------------------------------------------------------------------
_ORDERED_ANIMALS: list[str] = [
    # --- main experiment (constants.py) ---
    "dog",
    "elephant",
    "panda",
    "cat",
    "dragon",
    "lion",
    "eagle",
    "dolphin",
    "tiger",
    "wolf",
    # --- still main experiment, now with hatch ---
    "phoenix",
    "bear",
    "fox",
    "leopard",
    "whale",
    # --- qwen-wo-div extras ---
    "owl",
    "penguin",
    # --- common survey animals ---
    "horse",
    "hawk",
    "parrot",
    # --- long-tail survey / eval animals ---
    "raven",
    "deer",
    "rabbit",
    "snake",
    "shark",
    "otter",
    "giraffe",
    "octopus",
    "unicorn",
    "flamingo",
    "seahorse",
]

# Build the lookup dict
ANIMAL_STYLE_MAP: dict[str, tuple[str, str]] = {}
for _i, _animal in enumerate(_ORDERED_ANIMALS):
    _color = _BASE_COLORS[_i % len(_BASE_COLORS)]
    _hatch = _HATCH_CYCLE[_i // len(_BASE_COLORS)]
    ANIMAL_STYLE_MAP[_animal] = (_color, _hatch)

# Special entry for the "Other" bucket — always light gray with dots
ANIMAL_STYLE_MAP["other"] = ("#cccccc", "..")
ANIMAL_STYLE_MAP["Other"] = ("#cccccc", "..")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_animal_style(name: str) -> tuple[str, str]:
    """Return ``(color_hex, hatch_pattern)`` for *name*.

    Falls back deterministically for unknown animals so the same name always
    gets the same style.
    """
    key = name.lower().strip()
    if key in ANIMAL_STYLE_MAP:
        return ANIMAL_STYLE_MAP[key]

    # Deterministic fallback for truly unknown animals
    h = hash(key) % (len(_BASE_COLORS) * len(_HATCH_CYCLE))
    color = _BASE_COLORS[h % len(_BASE_COLORS)]
    hatch = _HATCH_CYCLE[h // len(_BASE_COLORS)]
    # Cache so repeated calls are consistent within a session
    ANIMAL_STYLE_MAP[key] = (color, hatch)
    return color, hatch


def get_animal_styles(animal_list: list[str]) -> list[tuple[str, str]]:
    """Return a list of ``(color, hatch)`` tuples, one per animal."""
    return [get_animal_style(a) for a in animal_list]
