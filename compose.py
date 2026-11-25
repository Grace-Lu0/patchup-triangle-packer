import json
import random
from collections import defaultdict

import cv2
import numpy as np

from report import compute_fabric_usage, compute_total_scrap_area_by_folder, save_fabric_usage_csv

INVENTORY_PATH = "triangle_inventory.json"
LINES_PER_QUADRANT = 5
UNIT_BASE_PX = 40
RANDOM_SEED = 42
BACKGROUND_COLOR_BGR = (235, 235, 235)

MAX_REUSE_FACTOR = 4.0
TARGET_ASPECT_RATIO = 1.5
MIN_SQUARES = 1

DIRECTIONS = ["up", "down", "left", "right"]


def triangles_per_square(lines):
    return 4 * lines * lines


def optimize_grid_size(available_triangles, lines, max_reuse=MAX_REUSE_FACTOR,
                        target_aspect=TARGET_ASPECT_RATIO, min_squares=MIN_SQUARES,
                        per_square=None):
    if per_square is None:
        per_square = triangles_per_square(lines)
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
        if total > best_total or (total == best_total and aspect_diff < best_aspect_diff):
            best_rows, best_cols, best_total = rows, cols, total
            best_aspect_diff = aspect_diff

    return best_rows, best_cols, max_squares, per_square


def load_inventory_by_folder(path=INVENTORY_PATH):
    with open(path) as f:
        records = json.load(f)
    if not records:
        raise ValueError(f"No triangle records found in {path}")

    by_folder = defaultdict(list)
    for r in records:
        by_folder[r["folder"]].append(r)
    return by_folder


def build_category_pools(records):
    pools = defaultdict(list)
    for r in records:
        pools[r["color_category"]].append(r)
    return dict(pools)



def build_quadrant_local_triangles(lines, unit_base):
    w = unit_base
    triangles = []

    apex_v = 0
    anchor_us = [0]
    row1_tri = [(0, apex_v), (-w / 2, apex_v + w / 2), (w / 2, apex_v + w / 2)]
    triangles.append((row1_tri, "row"))
    anchor_us = [-w / 2, w / 2]
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
    if direction == "up":
        return center_x + u, center_y - v
    if direction == "down":
        return center_x + u, center_y + v
    if direction == "left":
        return center_x - v, center_y + u
    return center_x + v, center_y + u


def build_quadrant_triangles(direction, lines, unit_base, center_x, center_y):
    local_triangles = build_quadrant_local_triangles(lines, unit_base)
    return [
        ([_local_to_canvas(u, v, direction, center_x, center_y) for u, v in tri], role)
        for tri, role in local_triangles
    ]



def pick_quadrant_categories(available_categories, rng):
    if len(available_categories) == 1:
        return available_categories[0], available_categories[0]
    row_cat, gap_cat = rng.sample(available_categories, 2)
    return row_cat, gap_cat



def paste_triangle_texture(canvas, record, dest_tri, image_cache):
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



def draw_square(canvas, pools, available_categories, center_x, center_y,
                 lines, unit_base, rng, pool_cursors, image_cache, used_ids):
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

    square_px = lines * unit_base
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



PINWHEEL_LEG_PX = 40
TRIANGLES_PER_PINWHEEL = 8
MIN_PAIR_CONTRAST = 60


def pinwheel_block_triangles(center_x, center_y, leg):
    cx, cy, L = center_x, center_y, leg
    ring = [(cx + L, cy), (cx + L, cy + L), (cx, cy + L), (cx - L, cy + L),
            (cx - L, cy), (cx - L, cy - L), (cx, cy - L), (cx + L, cy - L)]
    mids = {0, 2, 4, 6}
    tris = []
    for i in range(8):
        j = (i + 1) % 8
        verts = [(cx, cy), ring[i], ring[j]]
        right_idx = 1 if i in mids else 2
        tris.append((verts, right_idx))
    return tris


def _right_angle_order(vertices):
    pts = [np.array(v, dtype=float) for v in vertices[:3]]
    sides = [np.linalg.norm(pts[(k + 1) % 3] - pts[(k + 2) % 3]) for k in range(3)]
    ra = int(np.argmax(sides))
    return [pts[ra], pts[(ra + 1) % 3], pts[(ra + 2) % 3]]


def paste_triangle_texture_oriented(canvas, record, dest_verts, dest_right_idx,
                                     image_cache, rng, flat=False):
    if flat:
        r, g, b = record["color"]
        cv2.fillConvexPoly(canvas, np.array(dest_verts, dtype=np.int32), (b, g, r))
        return
    src = _right_angle_order(record["vertices_px"])
    others = [k for k in range(3) if k != dest_right_idx]
    if rng.random() < 0.5:
        others.reverse()
    dst = [dest_verts[dest_right_idx], dest_verts[others[0]], dest_verts[others[1]]]
    rec = dict(record)
    rec["vertices_px"] = [list(p) for p in src]
    paste_triangle_texture(canvas, rec, dst, image_cache)


class TrianglePools:

    def __init__(self, pools, rng, allow_reuse):
        self.rng = rng
        self.allow_reuse = allow_reuse
        self.remaining = {c: list(p) for c, p in pools.items()}
        for lst in self.remaining.values():
            rng.shuffle(lst)
        self.all = {c: list(p) for c, p in pools.items()}
        self.mean_rgb = {c: np.mean([r["color"] for r in p], axis=0) for c, p in pools.items()}

    def count(self, category):
        return len(self.remaining[category])

    def total_remaining(self):
        return sum(len(v) for v in self.remaining.values())

    def take(self, category):
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
    cats = list(pools_obj.remaining.keys())
    stocked = [c for c in cats if pools_obj.count(c) >= 4]
    if pools_obj.allow_reuse:
        stocked = cats
    if len(stocked) >= 2:
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