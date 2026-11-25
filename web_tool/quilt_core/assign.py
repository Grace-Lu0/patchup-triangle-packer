"""
Turn a QuiltPlan (colour per slot) into an Assignment (a real harvested
triangle per triangle position).

Unlike compose.TrianglePools, running out of a colour is never hidden:
without reuse, an exhausted category leaves the position empty and is
counted in Assignment.shortfall so the UI can warn the user.
"""

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass
class Placement:
    vertices: tuple
    right_idx: int
    block: int
    slot: str
    category: str
    record: dict | None
    flip: bool


@dataclass
class Assignment:
    rows: int
    cols: int
    block_side: float
    placements: list = field(default_factory=list)
    shortfall: dict = field(default_factory=dict)
    use_counts: Counter = field(default_factory=Counter)

    @property
    def used_ids(self):
        return set(self.use_counts)

    @property
    def n_missing(self):
        return sum(self.shortfall.values())

    @property
    def max_reuse(self):
        return max(self.use_counts.values(), default=0)


class _Pools:
    def __init__(self, records, rng, allow_reuse):
        self.rng = rng
        self.allow_reuse = allow_reuse
        by_cat = defaultdict(list)
        for r in sorted(records, key=lambda r: r["triangle_id"]):
            by_cat[r["color_category"]].append(r)
        self.all = dict(by_cat)
        self.remaining = {}
        for c, recs in self.all.items():
            self.remaining[c] = list(recs)
            rng.shuffle(self.remaining[c])

    def take(self, category):
        pool = self.remaining.get(category)
        if not pool:
            if not self.allow_reuse or not self.all.get(category):
                return None
            pool = self.remaining[category] = list(self.all[category])
            self.rng.shuffle(pool)
        return pool.pop()


def assign_triangles(plan, records, allow_reuse=False):
    """Pick a real triangle for every position of every chosen block."""
    pattern = plan.pattern
    rng = random.Random(f"{plan.seed}-triangles")
    pools = _Pools(records, rng, allow_reuse)
    result = Assignment(plan.rows, plan.cols, pattern.block_side)
    shortfall = Counter()

    for index, block in enumerate(plan.blocks):
        if block is None:
            continue
        row, col = plan.block_position(index)
        for tri in pattern.block_triangles(row, col):
            category = block[tri.slot]
            record = pools.take(category)
            if record is None:
                shortfall[category] += 1
            else:
                result.use_counts[record["triangle_id"]] += 1
            result.placements.append(Placement(
                tri.vertices, tri.right_idx, index, tri.slot, category, record,
                flip=rng.random() < 0.5))

    result.shortfall = dict(shortfall)
    return result
