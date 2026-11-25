"""
Quilt pattern geometry, expressed in "leg units": every triangle is a right
isosceles triangle with legs of length 1.0. The renderer multiplies by a
pixel size per leg, so the same plan can be drawn as a small preview or a
large print image.

Each pattern describes ONE block (a pinwheel block or a pyramid square):
  - which triangles it contains and where they sit,
  - which named colour "slot" each triangle belongs to,
  - which slots are paired (and should contrast with each other).

The geometry itself is reused from compose.py so the web tool and the CLI
draw identical shapes.
"""

import math
from dataclasses import dataclass
from functools import cached_property

from compose import DIRECTIONS, build_quadrant_triangles, pinwheel_block_triangles

DEFAULT_PYRAMID_LINES = 5


@dataclass(frozen=True)
class SlotTriangle:
    vertices: tuple
    right_idx: int
    slot: str


def right_angle_index(vertices):
    """Index of the vertex opposite the longest side (the right angle)."""
    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])
    sides = [dist(vertices[(k + 1) % 3], vertices[(k + 2) % 3]) for k in range(3)]
    return max(range(3), key=lambda k: sides[k])


class Pattern:
    key = ""
    label = ""
    description = ""
    slot_groups = ()
    slot_labels = {}

    @property
    def block_side(self):
        """Side length of one square block, in leg units."""
        raise NotImplementedError

    def block_triangles(self, row, col):
        """All SlotTriangles of the block at grid position (row, col)."""
        raise NotImplementedError

    @property
    def slots(self):
        return [s for pair in self.slot_groups for s in pair]

    @cached_property
    def slot_sizes(self):
        """How many triangles each slot needs per block."""
        sizes = {s: 0 for s in self.slots}
        for t in self.block_triangles(0, 0):
            sizes[t.slot] += 1
        return sizes

    @property
    def triangles_per_block(self):
        return sum(self.slot_sizes.values())


class PinwheelPattern(Pattern):
    key = "pinwheel"
    label = "Pinwheel"
    description = ("Each block is 8 triangles meeting in the centre, alternating "
                   "between two colours to make a spinning pinwheel.")
    slot_groups = (("A", "B"),)
    slot_labels = {"A": "Blades", "B": "Background"}

    @property
    def block_side(self):
        return 2.0

    def block_triangles(self, row, col):
        cx, cy = col * 2.0 + 1.0, row * 2.0 + 1.0
        return [
            SlotTriangle(tuple(verts), right_idx, "A" if i % 2 == 0 else "B")
            for i, (verts, right_idx) in enumerate(pinwheel_block_triangles(cx, cy, 1.0))
        ]


class PyramidPattern(Pattern):
    key = "pyramid"
    label = "Pyramid"
    description = ("Each square has four quadrants that grow outward from the "
                   "centre in rows of triangles, with contrasting gap triangles "
                   "between them.")

    def __init__(self, lines=DEFAULT_PYRAMID_LINES):
        self.lines = int(lines)
        self.slot_groups = tuple((f"{d}_row", f"{d}_gap") for d in DIRECTIONS)
        self.slot_labels = {}
        for d in DIRECTIONS:
            self.slot_labels[f"{d}_row"] = f"{d.capitalize()} quadrant: rows"
            self.slot_labels[f"{d}_gap"] = f"{d.capitalize()} quadrant: gaps"

    @property
    def _hypotenuse(self):
        return math.sqrt(2.0)

    @property
    def block_side(self):
        return self.lines * self._hypotenuse

    def block_triangles(self, row, col):
        side = self.block_side
        cx, cy = col * side + side / 2, row * side + side / 2
        out = []
        for d in DIRECTIONS:
            for coords, role in build_quadrant_triangles(d, self.lines, self._hypotenuse, cx, cy):
                verts = tuple(tuple(p) for p in coords)
                out.append(SlotTriangle(verts, right_angle_index(verts), f"{d}_{role}"))
        return out


PATTERN_KEYS = ("pinwheel", "pyramid")


def get_pattern(key, lines=DEFAULT_PYRAMID_LINES):
    if key == "pinwheel":
        return PinwheelPattern()
    if key == "pyramid":
        return PyramidPattern(lines)
    raise ValueError(f"Unknown pattern: {key!r}")
