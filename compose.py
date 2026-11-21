"""
compose.py

Ralli-inspired compositions, real fabric texture. Two patterns:
  --pattern pyramid  (default) growing pyramid quadrants
  --pattern pinwheel pinwheel blocks (8 triangles per block, see PINWHEEL section)

Builds one image per fabric-type folder (e.g. linens, cottons) from the
triangle inventory (triangle_inventory.json).

--------------------------------------------------------------------------
GEOMETRY - confirmed against a hand-drawn diagram before building this

  - A square is split by its two diagonals into 4 congruent right-
    isosceles triangles, meeting at the center - one per quadrant
    (up, down, left, right).
  - Take one quadrant's triangle as "row 1": apex at the square's center,
    base (hypotenuse) with 2 corner points facing outward.
  - Row 2 is 2 new triangles, same size and orientation, each with its
    own apex anchored exactly on one of row 1's 2 base corners. Between
    them, a single upside-down gap triangle forms - same size and shape
    as every other triangle, no cutting involved.
  - Row 3 is 3 triangles, anchored on row 2's 3 base corners (2 outer
    triangles' corners plus the 1 shared corner between them), with 2
    gaps. In general, row k has k triangles anchored on row (k-1)'s k
    corner points, with k-1 gaps - growing by exactly one triangle per
    row, always congruent, never cut.
  - Each quadrant grows for LINES_PER_QUADRANT rows (5 by default), then
    stops. Once all 4 quadrants of a square have completed their rows,
    that square is finished and an entirely new square begins next to it
    (fresh, independent color choices - no relation to the square before
    it).
  - Every quadrant, in every square, independently picks its own two
    color categories: one for the "row" triangles that are actually being
    placed, one for the negative-space gap triangles between them. No
    coordination between the 4 quadrants of one square, or between
    neighboring squares - repeats are fine.
  - Rendering uses the real cropped photo content of an actual harvested
    triangle in the chosen category, not a flat average color, warped to
    fit its slot via an affine transform from its 3 real vertices to its
    3 destination vertices.
--------------------------------------------------------------------------

What's missing / simplified:
  - No adjacency rule between neighboring squares or quadrants - colors
    can and will repeat right next to each other, by design.
  - Triangle reuse within a chosen category is unbounded.
  - A fabric folder with only one color category present falls back to
    using that same category for both the row-triangle and gap roles.

Run directly with:
    python compose.py
"""

import json
import random
from collections import defaultdict

import cv2
import numpy as np

from report import compute_fabric_usage, compute_total_scrap_area_by_folder, save_fabric_usage_csv

INVENTORY_PATH = "triangle_inventory.json"
LINES_PER_QUADRANT = 5
UNIT_BASE_PX = 40  # hypotenuse length of one unit triangle, in pixels
RANDOM_SEED = 42
BACKGROUND_COLOR_BGR = (235, 235, 235)

# Grid size is computed per fabric folder from how many triangles are
# actually available (see optimize_grid_size), rather than fixed. These
# control that computation; pass explicit rows/cols to compose_folder or
# compose to override the optimizer entirely.
MAX_REUSE_FACTOR = 4.0  # average number of times a single triangle may be reused
TARGET_ASPECT_RATIO = 1.5  # preferred width:height of the finished grid
MIN_SQUARES = 1

DIRECTIONS = ["up", "down", "left", "right"]


def triangles_per_square(lines):
    """Total triangles (row + gap) needed for one complete square: 4 * lines^2."""
    return 4 * lines * lines


def optimize_grid_size(available_triangles, lines, max_reuse=MAX_REUSE_FACTOR,
                        target_aspect=TARGET_ASPECT_RATIO, min_squares=MIN_SQUARES,
                        per_square=None):
    """
    Compute the largest rows x cols rectangle of squares that can be built
    from the available triangle count without exceeding max_reuse average
    uses per triangle, preferring a rows:cols ratio close to target_aspect
    (width:height) among options that use the budget equally well.

    Args:
        available_triangles: Total triangles in the fabric folder's
            inventory (every category combined - any category can be
            picked for any quadrant, so the budget is folder-wide).
        lines: LINES_PER_QUADRANT - how many rows each quadrant grows.
        max_reuse: Average number of times any single triangle may be
            reused across the whole composition. 1.0 means no reuse at
            all; higher values allow denser, larger grids from the same
            inventory at the cost of more repetition.
        target_aspect: Preferred cols/rows ratio for the finished grid.
        min_squares: Always produce at least this many squares, even if
            the strict budget computes to fewer (with a warning).

    Returns:
        (rows, cols, max_squares, per_square) tuple.
    """
    if per_square is None:
        per_square = triangles_per_square(lines)  # pyramid: 4 * lines^2
    max_squares = int((available_triangles * max_reuse) // per_square)

    if max_squares < min_squares:
        print(f"  warning: only {available_triangles} triangles available "
              f"({per_square} needed per square) - forcing a minimum "
              f"{min_squares}-square grid, reuse will exceed max_reuse={max_reuse}")
        max_squares = min_squares

    best_rows, best_cols, best_total = 1, max_squares, max_squares
    best_aspect_diff = abs(max_squares - target_aspect)

    for rows in range(1, max_squares + 1):
        cols = max_squares // rows
        if cols < 1:
            break
        total = rows * cols
        aspect_diff = abs((cols / rows) - target_aspect)
        # prefer using as much of the budget as possible, then closest aspect ratio
        if total > best_total or (total == best_total and aspect_diff < best_aspect_diff):
            best_rows, best_cols, best_total = rows, cols, total
            best_aspect_diff = aspect_diff

    return best_rows, best_cols, max_squares, per_square


def load_inventory_by_folder(path=INVENTORY_PATH):
    """Load the triangle inventory and group records by fabric-type folder."""
    with open(path) as f:
        records = json.load(f)
    if not records:
        raise ValueError(f"No triangle records found in {path}")

    by_folder = defaultdict(list)
    for r in records:
        by_folder[r["folder"]].append(r)
    return by_folder


def build_category_pools(records):
    """Split a folder's records into pools keyed by color_category."""
    pools = defaultdict(list)
    for r in records:
        pools[r["color_category"]].append(r)
    return dict(pools)


# --------------------------------------------------------------------------
# Quadrant geometry
# --------------------------------------------------------------------------

def build_quadrant_local_triangles(lines, unit_base):
    """
    Build every triangle in one quadrant, in local (u, v) coordinates:
    v=0 is the square's center (the growth axis), u is perpendicular to
    it. Growth proceeds in +v.

    Row 1 is the single quadrant triangle from splitting the square by
    its diagonals: apex at (0, 0), base corners at (-w/2, w/2) and
    (w/2, w/2), where w = unit_base.

    Row k (k >= 2) has k triangles, each anchored (apex) on one of row
    (k-1)'s k base corner points, with (k-1) upside-down gap triangles
    filling the space between them.

    Returns a list of (local_triangle_coords, role) pairs, role being
    'row' (the deliberately placed triangle) or 'gap' (negative space).
    """
    w = unit_base
    triangles = []

    # Row 1: apex at the center, base corners are this row's "anchor points"
    apex_v = 0
    anchor_us = [0]  # apex positions for the current row
    row1_tri = [(0, apex_v), (-w / 2, apex_v + w / 2), (w / 2, apex_v + w / 2)]
    triangles.append((row1_tri, "row"))
    anchor_us = [-w / 2, w / 2]  # row 1's 2 base corners become row 2's anchors
    apex_v = w / 2

    for k in range(2, lines + 1):
        base_v = apex_v + w / 2
        base_corners = []

        for i, u in enumerate(anchor_us):
            tri = [(u, apex_v), (u - w / 2, base_v), (u + w / 2, base_v)]
            triangles.append((tri, "row"))
            base_corners.append(u - w / 2)
            if i == len(anchor_us) - 1:
                base_corners.append(u + w / 2)

        for i in range(len(anchor_us) - 1):
            u_left, u_right = anchor_us[i], anchor_us[i + 1]
            shared_base_u = (u_left + u_right) / 2
            gap_tri = [(u_left, apex_v), (u_right, apex_v), (shared_base_u, base_v)]
            triangles.append((gap_tri, "gap"))

        anchor_us = base_corners
        apex_v = base_v

    return triangles


def _local_to_canvas(u, v, direction, center_x, center_y):
    """Map a quadrant-local (u, v) point to absolute canvas pixel coordinates."""
    if direction == "up":
        return center_x + u, center_y - v
    if direction == "down":
        return center_x + u, center_y + v
    if direction == "left":
        return center_x - v, center_y + u
    return center_x + v, center_y + u  # right


def build_quadrant_triangles(direction, lines, unit_base, center_x, center_y):
    """Build every (canvas_coords, role) pair for one quadrant."""
    local_triangles = build_quadrant_local_triangles(lines, unit_base)
    return [
        ([_local_to_canvas(u, v, direction, center_x, center_y) for u, v in tri], role)
        for tri, role in local_triangles
    ]


# --------------------------------------------------------------------------
# Category selection
# --------------------------------------------------------------------------

def pick_quadrant_categories(available_categories, rng):
    """
    Independently choose a (row, gap) color category pair for one
    quadrant. Distinct categories are preferred; if only one category is
    available for this fabric folder, it's used for both roles.
    """
    if len(available_categories) == 1:
        return available_categories[0], available_categories[0]
    row_cat, gap_cat = rng.sample(available_categories, 2)
    return row_cat, gap_cat


# --------------------------------------------------------------------------
# Real photo texture rendering
# --------------------------------------------------------------------------

def paste_triangle_texture(canvas, record, dest_tri, image_cache):
    """
    Crop the real photo content for a triangle's source region and warp
    it onto the destination triangle on the canvas, via an affine
    transform from the triangle's 3 real vertices to its 3 destination
    vertices. Only the destination triangle's interior is copied (masked),
    and only the part of it that actually falls within canvas bounds.
    """
    src_path = record["source_image"]
    if src_path not in image_cache:
        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not load source image: {src_path}")
        image_cache[src_path] = img
    src_img = image_cache[src_path]

    src_pts = np.array(record["vertices_px"][:3], dtype=np.float32)
    dest_pts = np.array(dest_tri[:3], dtype=np.float32)

    sx, sy, sw, sh = cv2.boundingRect(src_pts.astype(np.int32))
    sx, sy = max(sx, 0), max(sy, 0)
    sw, sh = max(sw, 1), max(sh, 1)
    src_crop = src_img[sy:sy + sh, sx:sx + sw]
    if src_crop.size == 0:
        return
    src_pts_local = (src_pts - [sx, sy]).astype(np.float32)

    dx, dy, dw, dh = cv2.boundingRect(dest_pts.astype(np.int32))
    if dw <= 0 or dh <= 0:
        return
    dest_pts_local = (dest_pts - [dx, dy]).astype(np.float32)

    matrix = cv2.getAffineTransform(src_pts_local, dest_pts_local)
    warped = cv2.warpAffine(src_crop, matrix, (dw, dh), borderMode=cv2.BORDER_REPLICATE)

    mask = np.zeros((dh, dw), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dest_pts_local.astype(np.int32), 255)

    cx0, cy0 = max(dx, 0), max(dy, 0)
    cx1, cy1 = min(dx + dw, canvas.shape[1]), min(dy + dh, canvas.shape[0])
    if cx1 <= cx0 or cy1 <= cy0:
        return
    wx0, wy0 = cx0 - dx, cy0 - dy
    wx1, wy1 = wx0 + (cx1 - cx0), wy0 + (cy1 - cy0)

    canvas_region = canvas[cy0:cy1, cx0:cx1]
    warped_region = warped[wy0:wy1, wx0:wx1]
    mask_region = mask[wy0:wy1, wx0:wx1]

    keep = mask_region > 0
    canvas_region[keep] = warped_region[keep]


# --------------------------------------------------------------------------
# Assembling one square and the full grid
# --------------------------------------------------------------------------

def draw_square(canvas, pools, available_categories, center_x, center_y,
                 lines, unit_base, rng, pool_cursors, image_cache, used_ids):
    """Draw one square's four independently-colored quadrants onto the canvas."""
    for direction in DIRECTIONS:
        row_cat, gap_cat = pick_quadrant_categories(available_categories, rng)
        triangles = build_quadrant_triangles(direction, lines, unit_base, center_x, center_y)

        for canvas_tri, role in triangles:
            category = row_cat if role == "row" else gap_cat
            pool = pools[category]
            i = pool_cursors[category]
            record = pool[i % len(pool)]
            pool_cursors[category] = i + 1
            used_ids.add(record["triangle_id"])

            paste_triangle_texture(canvas, record, canvas_tri, image_cache)


def compose_folder(folder, records, rows=None, cols=None, lines=LINES_PER_QUADRANT,
                    unit_base=UNIT_BASE_PX, seed=RANDOM_SEED,
                    max_reuse=MAX_REUSE_FACTOR, target_aspect=TARGET_ASPECT_RATIO,
                    total_scrap_area_in2=None):
    """
    Build and save one composition image for a single fabric-type folder.

    If rows or cols is left as None, the grid size is computed by
    optimize_grid_size from how many triangles are actually available in
    this folder, rather than using a fixed size - pass both explicitly to
    override the optimizer entirely.

    Args:
        total_scrap_area_in2: This folder's total ORIGINAL scrap area (in²),
            from report.compute_total_scrap_area_by_folder, used so the
            usage report's "unused" figure includes fabric that was never
            harvested into a triangle at all, not just harvested-but-
            unplaced triangles. Pass None to fall back to harvested-area-only.

    Returns:
        (output_path, usage_stats) tuple, or (None, None) if the folder
        has no usable triangles. usage_stats is the dict produced by
        report.compute_fabric_usage.
    """
    pools = build_category_pools(records)
    available_categories = list(pools.keys())

    if not available_categories:
        print(f"[{folder}] no triangles available, skipping")
        return None, None

    if rows is None or cols is None:
        rows, cols, max_squares, per_square = optimize_grid_size(
            len(records), lines, max_reuse=max_reuse, target_aspect=target_aspect
        )
        used = rows * cols
        reuse = (used * per_square) / len(records)
        print(f"[{folder}] {len(records)} triangles -> {per_square}/square, "
              f"budget for {max_squares} squares -> {rows}x{cols} grid "
              f"({used} squares, ~{reuse:.1f}x average reuse)")
    else:
        print(f"[{folder}] {len(records)} triangles across categories: {available_categories}, "
              f"grid fixed at {rows}x{cols}")

    square_px = lines * unit_base  # full side length of one completed square
    width_px = cols * square_px
    height_px = rows * square_px

    canvas = np.full((height_px, width_px, 3), BACKGROUND_COLOR_BGR, dtype=np.uint8)

    rng = random.Random(seed)
    pool_cursors = defaultdict(int)
    image_cache = {}
    used_ids = set()

    for sr in range(rows):
        for sc in range(cols):
            center_x = sc * square_px + square_px / 2
            center_y = sr * square_px + square_px / 2
            draw_square(canvas, pools, available_categories, center_x, center_y,
                        lines, unit_base, rng, pool_cursors, image_cache, used_ids)

    output_path = f"ralli_block_{folder}.png"
    cv2.imwrite(output_path, canvas)
    print(f"[{folder}] composition saved to: {output_path} ({width_px}x{height_px}px)")

    usage_stats = compute_fabric_usage(records, used_ids, total_scrap_area_in2=total_scrap_area_in2)
    print(f"[{folder}] fabric used: {usage_stats['reused_percentage']:.1f}% "
          f"({usage_stats['used_area_in2']:.2f} in\u00b2 of "
          f"{usage_stats['total_scrap_area_in2']:.2f} in\u00b2 total), "
          f"unused: {usage_stats['unused_percentage']:.1f}% "
          f"({usage_stats['unused_area_in2']:.2f} in\u00b2)")

    return output_path, usage_stats


def compose(inventory_path=INVENTORY_PATH, rows=None, cols=None, lines=LINES_PER_QUADRANT,
            unit_base=UNIT_BASE_PX, seed=RANDOM_SEED,
            max_reuse=MAX_REUSE_FACTOR, target_aspect=TARGET_ASPECT_RATIO):
    """
    Load the inventory, build one size-optimized composition per
    fabric-type folder, and save a fabric_usage_report.csv summarizing
    how much of each folder's ORIGINAL scrap fabric made it into its
    quilt (including fabric that was never harvested into a triangle).
    """
    by_folder = load_inventory_by_folder(inventory_path)
    all_records = [r for records in by_folder.values() for r in records]
    scrap_area_by_folder = compute_total_scrap_area_by_folder(all_records)

    outputs = []
    usage_by_folder = {}

    for folder, records in by_folder.items():
        output_path, usage_stats = compose_folder(
            folder, records, rows=rows, cols=cols, lines=lines,
            unit_base=unit_base, seed=seed, max_reuse=max_reuse, target_aspect=target_aspect,
            total_scrap_area_in2=scrap_area_by_folder.get(folder)
        )
        if output_path:
            outputs.append(output_path)
            usage_by_folder[folder] = usage_stats

    save_fabric_usage_csv(usage_by_folder)

    return outputs


# ==========================================================================
# PINWHEEL PATTERN
# ==========================================================================
#
# Geometry (from the Ralli quilt photo)
#   - One pinwheel block is a square of side 2*leg, split by its two
#     diagonals AND its two mid-lines into 8 congruent right isosceles
#     triangles that all meet at the block's center. Each triangle has its
#     right angle on an edge midpoint and a 45-degree angle at the center.
#   - Going around the center, the 8 triangles alternate between two
#     color categories (A, B, A, B, ...) which makes the pinwheel.
#   - Every block independently picks its own 2 color categories at random.
#   - Blocks tile edge to edge into a rows x cols quilt.
#   - Triangles keep ONE size: the inventory triangles are all right
#     isosceles with the same leg length t, so the block side is 2*t.
#   - "No reuse" (default, max_reuse=1.0): every inventory triangle is used
#     at most once. The grid is sized so that rows*cols*8 <= #triangles. If a
#     color category runs out mid-block, the missing slot is filled from
#     whichever category has the most triangles left. Blocks that cannot be
#     filled at all stay blank (background) and are reported.
#   - With max_reuse > 1.0 triangles may be reused when a pool runs out.

PINWHEEL_LEG_PX = 40  # leg length of one triangle in pixels (block side = 2x)
TRIANGLES_PER_PINWHEEL = 8
MIN_PAIR_CONTRAST = 60  # min RGB distance between a block's two category colors


def pinwheel_block_triangles(center_x, center_y, leg):
    """
    Return the 8 destination triangles of one pinwheel block as a list of
    (vertices, right_angle_index) where vertices = [center, p_i, p_i+1].
    right_angle_index is which of the 3 vertices is the right angle (the
    edge midpoint among p_i / p_i+1), so fabric can be warped without
    distortion.
    """
    cx, cy, L = center_x, center_y, leg
    ring = [(cx + L, cy), (cx + L, cy + L), (cx, cy + L), (cx - L, cy + L),
            (cx - L, cy), (cx - L, cy - L), (cx, cy - L), (cx + L, cy - L)]
    mids = {0, 2, 4, 6}  # ring indices that are edge midpoints (the others are corners)
    tris = []
    for i in range(8):
        j = (i + 1) % 8
        verts = [(cx, cy), ring[i], ring[j]]
        right_idx = 1 if i in mids else 2
        tris.append((verts, right_idx))
    return tris


def _right_angle_order(vertices):
    """Return the record's 3 vertices ordered (right_angle, leg_end_1, leg_end_2)."""
    pts = [np.array(v, dtype=float) for v in vertices[:3]]
    sides = [np.linalg.norm(pts[(k + 1) % 3] - pts[(k + 2) % 3]) for k in range(3)]
    ra = int(np.argmax(sides))  # vertex opposite the longest side = right angle
    return [pts[ra], pts[(ra + 1) % 3], pts[(ra + 2) % 3]]


def paste_triangle_texture_oriented(canvas, record, dest_verts, dest_right_idx,
                                     image_cache, rng, flat=False):
    """
    Like paste_triangle_texture, but maps the source triangle's right-angle
    vertex onto the destination's right-angle vertex (so the fabric isn't
    sheared), randomly flipping which leg goes where for variety.
    """
    if flat:  # flat-color preview: shows the pinwheel structure regardless of fabric texture
        r, g, b = record["color"]
        cv2.fillConvexPoly(canvas, np.array(dest_verts, dtype=np.int32), (b, g, r))
        return
    src = _right_angle_order(record["vertices_px"])
    others = [k for k in range(3) if k != dest_right_idx]
    if rng.random() < 0.5:
        others.reverse()
    dst = [dest_verts[dest_right_idx], dest_verts[others[0]], dest_verts[others[1]]]
    # reuse the existing warp, passing a record copy with reordered vertices
    rec = dict(record)
    rec["vertices_px"] = [list(p) for p in src]
    paste_triangle_texture(canvas, rec, dst, image_cache)


class TrianglePools:
    """Per-category triangle pools with optional no-reuse bookkeeping."""

    def __init__(self, pools, rng, allow_reuse):
        self.rng = rng
        self.allow_reuse = allow_reuse
        self.remaining = {c: list(p) for c, p in pools.items()}
        for lst in self.remaining.values():
            rng.shuffle(lst)
        self.all = {c: list(p) for c, p in pools.items()}
        # mean RGB of each category, used to keep the 2 colors of a block visibly different
        self.mean_rgb = {c: np.mean([r["color"] for r in p], axis=0) for c, p in pools.items()}

    def count(self, category):
        return len(self.remaining[category])

    def total_remaining(self):
        return sum(len(v) for v in self.remaining.values())

    def take(self, category):
        """Take one triangle from `category`; fall back to the fullest pool."""
        if not self.remaining[category]:
            if self.allow_reuse:
                self.remaining[category] = list(self.all[category])
                self.rng.shuffle(self.remaining[category])
            else:
                category = max(self.remaining, key=lambda c: len(self.remaining[c]))
                if not self.remaining[category]:
                    return None
        return self.remaining[category].pop()


def pick_block_categories(pools_obj, rng):
    """Pick 2 distinct color categories for one block (4 triangles each)."""
    cats = list(pools_obj.remaining.keys())
    stocked = [c for c in cats if pools_obj.count(c) >= 4]
    if pools_obj.allow_reuse:
        stocked = cats
    if len(stocked) >= 2:
        # random pair, but only pairs whose average colors differ enough to read
        # as a pinwheel; if none do, take the most different pair available
        pairs = [(a, b) for i, a in enumerate(stocked) for b in stocked[i + 1:]]
        dist = lambda p: float(np.linalg.norm(pools_obj.mean_rgb[p[0]] - pools_obj.mean_rgb[p[1]]))
        good = [p for p in pairs if dist(p) >= MIN_PAIR_CONTRAST]
        pair = rng.choice(good) if good else max(pairs, key=dist)
        return pair if rng.random() < 0.5 else (pair[1], pair[0])
    ranked = sorted(cats, key=lambda c: -pools_obj.count(c))
    if len(ranked) >= 2:
        return ranked[0], ranked[1]
    return ranked[0], ranked[0]


def compose_folder_pinwheel(folder, records, rows=None, cols=None,
                             leg=PINWHEEL_LEG_PX, seed=RANDOM_SEED, max_reuse=1.0,
                             target_aspect=TARGET_ASPECT_RATIO, total_scrap_area_in2=None,
                             flat=False):
    """Build and save one pinwheel quilt image for a fabric-type folder."""
    print(f"[{folder}] PATTERN: pinwheel ({'flat color preview' if flat else 'real fabric texture'})")
    pools = build_category_pools(records)
    if not pools:
        print(f"[{folder}] no triangles available, skipping")
        return None, None
    print(f"[{folder}] color categories available: " + ", ".join(f"{c}={len(p)}" for c, p in pools.items()))

    if rows is None or cols is None:
        rows, cols, max_squares, per_block = optimize_grid_size(
            len(records), None, max_reuse=max_reuse, target_aspect=target_aspect,
            per_square=TRIANGLES_PER_PINWHEEL)
        print(f"[{folder}] {len(records)} triangles -> {per_block}/block, "
              f"budget {max_squares} blocks -> {rows}x{cols} grid")
    else:
        print(f"[{folder}] {len(records)} triangles, grid fixed at {rows}x{cols}")

    block_px = 2 * leg
    canvas = np.full((rows * block_px, cols * block_px, 3), BACKGROUND_COLOR_BGR, dtype=np.uint8)

    rng = random.Random(seed)
    pools_obj = TrianglePools(pools, rng, allow_reuse=max_reuse > 1.0)
    image_cache, used_ids, blank_blocks = {}, set(), 0

    for r in range(rows):
        for c in range(cols):
            if pools_obj.total_remaining() == 0:
                blank_blocks += 1
                continue
            cat_a, cat_b = pick_block_categories(pools_obj, rng)
            cx, cy = c * block_px + leg, r * block_px + leg
            for i, (verts, right_idx) in enumerate(pinwheel_block_triangles(cx, cy, leg)):
                record = pools_obj.take(cat_a if i % 2 == 0 else cat_b)
                if record is None:
                    continue
                used_ids.add(record["triangle_id"])
                paste_triangle_texture_oriented(canvas, record, verts, right_idx, image_cache, rng, flat)

    if blank_blocks:
        print(f"[{folder}] warning: ran out of triangles, {blank_blocks} blocks left blank")

    output_path = f"pinwheel_block_{folder}{'_flat' if flat else ''}.png"
    cv2.imwrite(output_path, canvas)
    print(f"[{folder}] pinwheel quilt saved to: {output_path} "
          f"({cols * block_px}x{rows * block_px}px, {rows * cols} blocks, "
          f"block side = 2 x leg = {block_px}px)")

    usage_stats = compute_fabric_usage(records, used_ids, total_scrap_area_in2=total_scrap_area_in2)
    print(f"[{folder}] fabric used: {usage_stats['reused_percentage']:.1f}% "
          f"({usage_stats['used_area_in2']:.2f} in\u00b2 of "
          f"{usage_stats['total_scrap_area_in2']:.2f} in\u00b2 total)")
    return output_path, usage_stats


def compose_pinwheel(inventory_path=INVENTORY_PATH, rows=None, cols=None,
                      leg=PINWHEEL_LEG_PX, seed=RANDOM_SEED, max_reuse=1.0,
                      target_aspect=TARGET_ASPECT_RATIO, flat=False):
    """Pinwheel quilt for every fabric-type folder in the inventory."""
    by_folder = load_inventory_by_folder(inventory_path)
    all_records = [r for recs in by_folder.values() for r in recs]
    scrap_area_by_folder = compute_total_scrap_area_by_folder(all_records)
    outputs, usage = [], {}
    for folder, records in by_folder.items():
        path, stats = compose_folder_pinwheel(
            folder, records, rows=rows, cols=cols, leg=leg, seed=seed,
            max_reuse=max_reuse, target_aspect=target_aspect,
            total_scrap_area_in2=scrap_area_by_folder.get(folder), flat=flat)
        if path:
            outputs.append(path)
            usage[folder] = stats
    save_fabric_usage_csv(usage, filename="fabric_usage_report_pinwheel.csv")
    return outputs


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Ralli-inspired quilt compositions from the triangle inventory.")
    ap.add_argument("--pattern", choices=["pyramid", "pinwheel"], default="pinwheel")
    ap.add_argument("--inventory", default=INVENTORY_PATH)
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--cols", type=int, default=None)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--leg", type=int, default=PINWHEEL_LEG_PX, help="pinwheel: triangle leg in px (block side = 2x)")
    ap.add_argument("--max-reuse", type=float, default=None,
                    help="avg uses per triangle (pinwheel default 1.0 = no reuse; pyramid default 4.0)")
    ap.add_argument("--flat", action="store_true", help="pinwheel: flat-color preview (no fabric texture)")
    args = ap.parse_args()
    if args.pattern == "pinwheel":
        compose_pinwheel(args.inventory, args.rows, args.cols, args.leg, args.seed,
                         args.max_reuse if args.max_reuse is not None else 1.0, flat=args.flat)
    else:
        print("PATTERN: pyramid  (add --pattern pinwheel for pinwheels)")
        compose(args.inventory, args.rows, args.cols, seed=args.seed,
                max_reuse=args.max_reuse if args.max_reuse is not None else MAX_REUSE_FACTOR)