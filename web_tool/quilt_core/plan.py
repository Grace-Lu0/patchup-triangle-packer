"""
QuiltPlan: which colour category goes in each slot of each block.

A plan holds only colour choices, never specific triangles, so random mode
and step-by-step mode share everything downstream (assign.py, render.py).

    plan.blocks[i] is None              -> block i not chosen yet (left blank)
    plan.blocks[i] == {"A": "blue", ...} -> one category per slot of the pattern
"""

import math
import random
from dataclasses import dataclass, field

import numpy as np

from .patterns import DEFAULT_PYRAMID_LINES, get_pattern

MIN_PAIR_CONTRAST = 60
TARGET_ASPECT_RATIO = 1.5
MAX_AUTO_BLOCKS = 600


@dataclass
class QuiltPlan:
    pattern_key: str
    rows: int
    cols: int
    seed: int = 42
    lines: int = DEFAULT_PYRAMID_LINES
    blocks: list = field(default_factory=list)

    def __post_init__(self):
        n = self.rows * self.cols
        if not self.blocks:
            self.blocks = [None] * n
        elif len(self.blocks) != n:
            raise ValueError(f"plan has {len(self.blocks)} blocks, expected {n}")

    @property
    def pattern(self):
        return get_pattern(self.pattern_key, self.lines)

    @property
    def n_blocks(self):
        return self.rows * self.cols

    def block_position(self, index):
        """(row, col) of block `index`, filled left to right, top to bottom."""
        return divmod(index, self.cols)

    def is_complete(self):
        return all(b is not None for b in self.blocks)

    def n_filled(self):
        return sum(b is not None for b in self.blocks)



def usable_triangles(counts, pattern):
    """
    Triangles that can actually be placed: a slot takes a whole group of
    triangles of one colour, so leftovers smaller than a slot are unusable.
    (Uses the largest slot size, which is exact for pinwheel and
    conservative for pyramid.)
    """
    group = max(pattern.slot_sizes.values())
    return sum((n // group) * group for n in counts.values())


def best_grid(max_blocks, target_aspect=TARGET_ASPECT_RATIO, min_fill=0.9):
    """
    rows x cols with rows*cols <= max_blocks, using at least `min_fill` of
    the budget, with cols/rows as close as possible to target_aspect.
    (compose.optimize_grid_size maximises the block count first, which
    gives strips like 67 x 5 whenever the budget is a near-prime number.)
    """
    max_blocks = max(int(max_blocks), 1)
    best = None
    for rows in range(1, max_blocks + 1):
        cols = max_blocks // rows
        if cols < 1:
            break
        total = rows * cols
        if total < min_fill * max_blocks:
            continue
        score = (abs(math.log((cols / rows) / target_aspect)), -total)
        if best is None or score < best[0]:
            best = (score, rows, cols)
    return best[1], best[2]


def auto_grid(counts_or_total, pattern, max_reuse=1.0, target_aspect=TARGET_ASPECT_RATIO,
              max_blocks=MAX_AUTO_BLOCKS):
    """
    A well-proportioned rows x cols grid the inventory can fill without
    exceeding `max_reuse` average uses per triangle (1.0 = every triangle
    used at most once), capped at `max_blocks`.

    Args:
        counts_or_total: {category: count} (preferred: accounts for leftovers
            too small to fill a slot) or a plain triangle total.
    """
    if isinstance(counts_or_total, dict):
        n = usable_triangles(counts_or_total, pattern) if max_reuse <= 1.0 else sum(counts_or_total.values())
    else:
        n = int(counts_or_total)
    budget = int(n * max_reuse) // pattern.triangles_per_block
    return best_grid(min(max(budget, 1), max_blocks), target_aspect)



def block_usage(pattern, block):
    """{category: triangles needed} for one block's slot choices."""
    out = {}
    if block is None:
        return out
    for slot, cat in block.items():
        out[cat] = out.get(cat, 0) + pattern.slot_sizes[slot]
    return out


def plan_usage(plan, exclude=None):
    """{category: triangles needed} across all chosen blocks, optionally skipping one index."""
    pattern = plan.pattern
    total = {}
    for i, block in enumerate(plan.blocks):
        if i == exclude:
            continue
        for cat, n in block_usage(pattern, block).items():
            total[cat] = total.get(cat, 0) + n
    return total


def remaining_counts(plan, counts, exclude=None):
    """Triangles left per category after the plan (negative = over budget)."""
    used = plan_usage(plan, exclude=exclude)
    cats = set(counts) | set(used)
    return {c: counts.get(c, 0) - used.get(c, 0) for c in cats}


def overbooked(plan, counts):
    """{category: shortfall} for categories the plan asks for more of than exist."""
    return {c: -n for c, n in remaining_counts(plan, counts).items() if n < 0}


def _contrast(mean_rgb, a, b):
    if a not in mean_rgb or b not in mean_rgb:
        return 0.0
    return float(np.linalg.norm(np.subtract(mean_rgb[a], mean_rgb[b])))


def _pick_pair(categories, left, need_a, need_b, mean_rgb, rng, allow_reuse, min_contrast):
    """Pick (cat_a, cat_b) for one slot pair, preferring stocked, contrasting, plentiful colours."""
    if allow_reuse:
        stocked_a = stocked_b = list(categories)
    else:
        stocked_a = [c for c in categories if left.get(c, 0) >= need_a]
        stocked_b = [c for c in categories if left.get(c, 0) >= need_b]

    pairs = [(a, b) for a in stocked_a for b in stocked_b
             if a != b and (allow_reuse or left.get(a, 0) + left.get(b, 0) >= need_a + need_b)]
    if pairs:
        good = [p for p in pairs if _contrast(mean_rgb, *p) >= min_contrast] if min_contrast else pairs
        if not good:
            best = max(_contrast(mean_rgb, *p) for p in pairs)
            good = [p for p in pairs if _contrast(mean_rgb, *p) >= best - 1e-9]
        weights = [max(left.get(a, 0), 1) + max(left.get(b, 0), 1) for a, b in good]
        return rng.choices(good, weights=weights, k=1)[0]

    ranked = sorted(categories, key=lambda c: (-left.get(c, 0), c))
    if not ranked:
        raise ValueError("no colour categories available")
    if left.get(ranked[0], 0) >= need_a + need_b:
        return ranked[0], ranked[0]
    if len(ranked) >= 2 and left.get(ranked[1], 0) > 0:
        return ranked[0], ranked[1]
    return ranked[0], ranked[0]


def random_block(pattern, categories, left, mean_rgb, rng, allow_reuse=False,
                 min_contrast=MIN_PAIR_CONTRAST):
    """Choose categories for every slot of one block. Updates `left` in place."""
    sizes = pattern.slot_sizes
    block = {}
    for slot_a, slot_b in pattern.slot_groups:
        a, b = _pick_pair(categories, left, sizes[slot_a], sizes[slot_b],
                          mean_rgb, rng, allow_reuse, min_contrast)
        block[slot_a], block[slot_b] = a, b
        left[a] = left.get(a, 0) - sizes[slot_a]
        left[b] = left.get(b, 0) - sizes[slot_b]
    return block


def fill_random(plan, counts, mean_rgb, allow_reuse=False, min_contrast=MIN_PAIR_CONTRAST,
                only_empty=False, seed=None):
    """
    Fill the plan's blocks with random colour choices (in place) and return it.

    Args:
        counts: {category: triangles available}.
        only_empty: keep blocks the user already chose, fill only None blocks.
        seed: overrides plan.seed for this fill (plan.seed is updated to match).
    """
    if seed is not None:
        plan.seed = seed
    rng = random.Random(f"{plan.seed}-colors")
    pattern = plan.pattern
    categories = sorted(c for c, n in counts.items() if n > 0)
    if not categories:
        raise ValueError("no triangles available to fill the quilt")

    if only_empty:
        left = remaining_counts(plan, counts)
    else:
        plan.blocks = [None] * plan.n_blocks
        left = dict(counts)

    for i in range(plan.n_blocks):
        if only_empty and plan.blocks[i] is not None:
            continue
        plan.blocks[i] = random_block(pattern, categories, left, mean_rgb, rng,
                                      allow_reuse, min_contrast)
    return plan


def set_block(plan, index, block):
    """Set one block's slot choices (validated against the pattern's slots)."""
    slots = set(plan.pattern.slots)
    if block is not None and set(block) != slots:
        raise ValueError(f"block must set exactly these slots: {sorted(slots)}")
    plan.blocks[index] = dict(block) if block is not None else None


def apply_to_all(plan, block):
    for i in range(plan.n_blocks):
        set_block(plan, i, block)
