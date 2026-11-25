import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quilt_core.assign import assign_triangles
from quilt_core.inventory_service import alpha_coverage, category_counts, category_mean_rgb
from quilt_core.patterns import get_pattern
from quilt_core.plan import (QuiltPlan, apply_to_all, auto_grid, fill_random,
                             overbooked, remaining_counts, set_block)
from quilt_core.render import cutting_list_csv, render, to_png_bytes, triangle_crop

COLORS = {"red": (200, 30, 30), "blue": (30, 60, 200), "white": (245, 245, 245),
          "green": (40, 160, 60)}


def make_records(counts, source_image="missing.png"):
    recs = []
    for cat, n in counts.items():
        for i in range(n):
            recs.append({
                "triangle_id": f"{cat}_{i:04d}", "scrap_id": f"s_{cat}", "folder": "test",
                "source_image": source_image,
                "vertices_px": [[0, 0], [50, 0], [0, 50]],
                "vertices_in": [[0, 0], [1, 0], [0, 1]],
                "area_in2": 0.5, "color": list(COLORS[cat]), "color_category": cat,
                "scrap_original_area_in2": 10.0,
            })
    return recs



@pytest.mark.parametrize("key,lines,expected", [
    ("pinwheel", 5, 8), ("pyramid", 5, 100), ("pyramid", 3, 36), ("pyramid", 1, 4)])
def test_triangles_per_block(key, lines, expected):
    assert get_pattern(key, lines).triangles_per_block == expected


def test_pinwheel_slots_split_evenly():
    assert get_pattern("pinwheel").slot_sizes == {"A": 4, "B": 4}


def test_pyramid_slot_sizes():
    sizes = get_pattern("pyramid", 5).slot_sizes
    assert sizes["up_row"] == 15 and sizes["up_gap"] == 10


@pytest.mark.parametrize("key", ["pinwheel", "pyramid"])
def test_block_is_tiled_exactly(key):
    """Triangles cover the whole block with no overlap and nothing outside it."""
    pattern = get_pattern(key)
    row, col = 1, 2
    tris = [Polygon(t.vertices) for t in pattern.block_triangles(row, col)]
    s = pattern.block_side
    block = box(col * s, row * s, (col + 1) * s, (row + 1) * s)
    assert sum(t.area for t in tris) == pytest.approx(block.area)
    assert unary_union(tris).symmetric_difference(block).area == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("key", ["pinwheel", "pyramid"])
def test_every_triangle_is_unit_right_isosceles(key):
    for t in get_pattern(key).block_triangles(0, 0):
        v = t.vertices
        ra = v[t.right_idx]
        legs = [math.dist(ra, v[k]) for k in range(3) if k != t.right_idx]
        assert legs == pytest.approx([1.0, 1.0])



def test_auto_grid_respects_budget_without_reuse():
    pattern = get_pattern("pinwheel")
    rows, cols = auto_grid(100, pattern, max_reuse=1.0)
    assert rows * cols * 8 <= 100
    assert rows * cols >= 10


def test_auto_grid_is_well_proportioned():
    rows, cols = auto_grid(335 * 8, get_pattern("pinwheel"))
    assert 1.2 <= cols / rows <= 1.8
    assert rows * cols >= 0.9 * 335


REAL_COUNTS = {"yellow": 62, "green": 1471, "blue": 762, "black": 46, "grey": 184,
               "orange": 27, "pink": 61, "white": 71}


@pytest.mark.parametrize("key", ["pinwheel", "pyramid"])
@pytest.mark.parametrize("seed", range(20))
def test_auto_grid_random_fill_never_overbooks(key, seed):
    pattern = get_pattern(key)
    rows, cols = auto_grid(REAL_COUNTS, pattern)
    mean = {c: (i * 30 % 255, i * 70 % 255, i * 110 % 255) for i, c in enumerate(REAL_COUNTS)}
    plan = fill_random(QuiltPlan(key, rows, cols, seed=seed), REAL_COUNTS, mean)
    assert overbooked(plan, REAL_COUNTS) == {}


def test_auto_grid_cap():
    rows, cols = auto_grid(10**6, get_pattern("pinwheel"), max_blocks=50)
    assert rows * cols <= 50


def test_random_fill_is_deterministic_and_complete():
    counts = {"red": 40, "blue": 40, "white": 40}
    mean = {c: COLORS[c] for c in counts}
    a = fill_random(QuiltPlan("pinwheel", 3, 4, seed=7), counts, mean)
    b = fill_random(QuiltPlan("pinwheel", 3, 4, seed=7), counts, mean)
    c = fill_random(QuiltPlan("pinwheel", 3, 4, seed=8), counts, mean)
    assert a.blocks == b.blocks
    assert a.blocks != c.blocks
    assert a.is_complete()
    for blk in a.blocks:
        assert blk["A"] != blk["B"]


def test_random_fill_stays_within_budget_when_possible():
    counts = {"red": 48, "blue": 48, "white": 48}
    mean = {c: COLORS[c] for c in counts}
    plan = fill_random(QuiltPlan("pinwheel", 3, 6, seed=1), counts, mean)
    assert overbooked(plan, counts) == {}


def test_random_fill_only_empty_keeps_user_choices():
    counts = {"red": 100, "blue": 100, "white": 100}
    mean = {c: COLORS[c] for c in counts}
    plan = QuiltPlan("pinwheel", 2, 2, seed=3)
    set_block(plan, 0, {"A": "red", "B": "white"})
    fill_random(plan, counts, mean, only_empty=True)
    assert plan.blocks[0] == {"A": "red", "B": "white"}
    assert plan.is_complete()


def test_single_category_falls_back_to_same_colour():
    plan = fill_random(QuiltPlan("pinwheel", 1, 2), {"red": 16}, {"red": COLORS["red"]})
    assert plan.blocks == [{"A": "red", "B": "red"}] * 2


def test_set_block_rejects_wrong_slots():
    plan = QuiltPlan("pinwheel", 1, 1)
    with pytest.raises(ValueError):
        set_block(plan, 0, {"A": "red"})


def test_remaining_counts_and_overbooked():
    plan = QuiltPlan("pinwheel", 1, 3)
    apply_to_all(plan, {"A": "red", "B": "blue"})
    counts = {"red": 10, "blue": 20}
    assert remaining_counts(plan, counts) == {"red": -2, "blue": 8}
    assert overbooked(plan, counts) == {"red": 2}
    assert remaining_counts(plan, counts, exclude=0) == {"red": 2, "blue": 12}



def test_assign_no_reuse_and_reports_shortfall():
    records = make_records({"red": 10, "blue": 20})
    plan = QuiltPlan("pinwheel", 1, 3)
    apply_to_all(plan, {"A": "red", "B": "blue"})
    result = assign_triangles(plan, records, allow_reuse=False)
    assert result.max_reuse == 1
    assert result.shortfall == {"red": 2}
    assert sum(p.record is None for p in result.placements) == 2
    assert len(result.placements) == 24


def test_assign_with_reuse_fills_everything():
    records = make_records({"red": 3, "blue": 3})
    plan = QuiltPlan("pinwheel", 2, 2)
    apply_to_all(plan, {"A": "red", "B": "blue"})
    result = assign_triangles(plan, records, allow_reuse=True)
    assert result.shortfall == {}
    assert result.max_reuse > 1


def test_assign_slots_get_their_category():
    records = make_records({"red": 50, "blue": 50, "white": 50})
    counts, mean = category_counts(records), category_mean_rgb(records)
    plan = fill_random(QuiltPlan("pyramid", 1, 1, lines=3, seed=5), counts, mean)
    result = assign_triangles(plan, records)
    for p in result.placements:
        assert p.record["color_category"] == plan.blocks[p.block][p.slot]


def test_assign_independent_of_record_order():
    records = make_records({"red": 30, "blue": 30})
    plan = QuiltPlan("pinwheel", 2, 3, seed=11)
    apply_to_all(plan, {"A": "red", "B": "blue"})
    shuffled = list(records)
    random.Random(0).shuffle(shuffled)
    ids = lambda a: [p.record["triangle_id"] for p in a.placements]
    assert ids(assign_triangles(plan, records)) == ids(assign_triangles(plan, shuffled))


def test_blank_blocks_have_no_placements():
    records = make_records({"red": 30, "blue": 30})
    plan = QuiltPlan("pinwheel", 1, 3)
    set_block(plan, 1, {"A": "red", "B": "blue"})
    result = assign_triangles(plan, records)
    assert {p.block for p in result.placements} == {1}



@pytest.fixture
def fabric_png(tmp_path):
    """A 100x100 RGBA scrap: left half red, right half blue, top-left corner transparent."""
    img = np.zeros((100, 100, 4), dtype=np.uint8)
    img[:, :50] = (30, 30, 200, 255)
    img[:, 50:] = (200, 60, 30, 255)
    img[:10, :10, 3] = 0
    path = tmp_path / "scrap.png"
    cv2.imwrite(str(path), img)
    return path


def test_alpha_coverage(fabric_png):
    alpha = cv2.imread(str(fabric_png), cv2.IMREAD_UNCHANGED)[:, :, 3]
    assert alpha_coverage(alpha, [[20, 20], [60, 20], [20, 60]]) == 1.0
    assert alpha_coverage(alpha, [[0, 0], [20, 0], [0, 20]]) < 0.6


@pytest.mark.parametrize("key", ["pinwheel", "pyramid"])
@pytest.mark.parametrize("mode", ["flat", "texture"])
def test_render_sizes_and_fills(key, mode, fabric_png):
    records = make_records({"red": 400, "blue": 400}, source_image=str(fabric_png))
    for r in records:
        x0 = 0 if r["color_category"] == "red" else 50
        r["vertices_px"] = [[x0 + 10, 20], [x0 + 40, 20], [x0 + 10, 50]]
    plan = QuiltPlan(key, 2, 2, lines=3)
    apply_to_all(plan, {s: ("red" if i % 2 == 0 else "blue")
                        for i, s in enumerate(plan.pattern.slots)})
    result = assign_triangles(plan, records)
    img = render(result, leg_px=20, mode=mode, highlight_block=0)
    side = plan.pattern.block_side * 20
    assert img.shape == (round(2 * side), round(2 * side), 3)
    interior = img[int(side) + 5:-5, int(side) + 5:-5]
    background = np.all(interior == (235, 235, 235), axis=-1)
    assert background.mean() < 0.01
    assert to_png_bytes(img)[:8] == b"\x89PNG\r\n\x1a\n"


def test_texture_uses_fabric_colour(fabric_png):
    records = make_records({"red": 8}, source_image=str(fabric_png))
    for r in records:
        r["vertices_px"] = [[15, 20], [45, 20], [15, 50]]
    plan = QuiltPlan("pinwheel", 1, 1)
    apply_to_all(plan, {"A": "red", "B": "red"})
    img = render(assign_triangles(plan, records), leg_px=30, mode="texture")
    centre = img[20:40, 20:40].reshape(-1, 3).mean(axis=0)
    assert centre[0] > 150 and centre[2] < 80


def test_triangle_crop_and_cutting_list(fabric_png):
    records = make_records({"red": 8, "blue": 8}, source_image=str(fabric_png))
    crop = triangle_crop(records[0], size=32)
    assert crop.shape == (32, 32, 3)
    plan = QuiltPlan("pinwheel", 1, 1)
    apply_to_all(plan, {"A": "red", "B": "blue"})
    text = cutting_list_csv(assign_triangles(plan, records))
    lines = text.strip().splitlines()
    assert len(lines) == 1 + 8
    assert lines[0].startswith("block,block_row")


def test_outline_empty_blocks():
    records = make_records({"red": 8, "blue": 8})
    plan = QuiltPlan("pinwheel", 1, 2)
    set_block(plan, 0, {"A": "red", "B": "blue"})
    result = assign_triangles(plan, records)
    plain = render(result, leg_px=20)
    outlined = render(result, leg_px=20, outline_empty=True)
    assert np.array_equal(plain[:, :40], outlined[:, :40])
    assert not np.array_equal(plain[:, 40:], outlined[:, 40:])
    assert tuple(outlined[0, 60]) == (200, 200, 200)


def test_triangle_sizes_and_inventory_paths():
    from quilt_core.inventory_service import TRIANGLE_SIZES, inventory_path
    assert TRIANGLE_SIZES == [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    assert inventory_path(1).name == "inventory_1.0in.json"
    assert inventory_path(2.5).name == "inventory_2.5in.json"
    assert len({inventory_path(s) for s in TRIANGLE_SIZES}) == len(TRIANGLE_SIZES)


def test_cutting_list_includes_triangle_size():
    records = make_records({"red": 8, "blue": 8})
    for r in records:
        r["triangle_size_inches"] = 2.5
    plan = QuiltPlan("pinwheel", 1, 1)
    apply_to_all(plan, {"A": "red", "B": "blue"})
    rows = cutting_list_csv(assign_triangles(plan, records)).strip().splitlines()
    header = rows[0].split(",")
    assert rows[1].split(",")[header.index("triangle_size_in")] == "2.5"
