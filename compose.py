import json
import random
from collections import defaultdict

import cv2
import numpy as np

from report import compute_fabric_usage, compute_total_scrap_area_by_folder, save_fabric_usage_csv

INVENTORY_PATH = "triangle_inventory.json"
ROWS_PER_QUADRANT = 5
TRIANGLE_BASE_PX = 40
RANDOM_SEED = 42
BACKGROUND_COLOR_BGR = (235, 235, 235)

MAX_AVERAGE_REUSE = 4.0
TARGET_GRID_ASPECT_RATIO = 1.5
MIN_SQUARES = 1

DIRECTIONS = ["up", "down", "left", "right"]


def triangles_per_square(rows_per_quadrant):
    return 4 * rows_per_quadrant * rows_per_quadrant


def optimize_grid_size(triangle_count, rows_per_quadrant, max_average_reuse=MAX_AVERAGE_REUSE,
                        target_aspect_ratio=TARGET_GRID_ASPECT_RATIO, minimum_squares=MIN_SQUARES):
    triangles_per_square_count = triangles_per_square(rows_per_quadrant)
    max_squares = int((triangle_count * max_average_reuse) // triangles_per_square_count)

    if max_squares < minimum_squares:
        print(f"  warning: only {triangle_count} triangles available "
              f"({triangles_per_square_count} needed per square) - forcing a minimum "
              f"{minimum_squares}-square grid, reuse will exceed max_average_reuse={max_average_reuse}")
        max_squares = minimum_squares

    best_rows, best_cols, best_total = 1, max_squares, max_squares
    best_aspect_difference = abs(max_squares - target_aspect_ratio)

    for rows in range(1, max_squares + 1):
        cols = max_squares // rows
        if cols < 1:
            break
        total = rows * cols
        aspect_difference = abs((cols / rows) - target_aspect_ratio)
        if total > best_total or (total == best_total and aspect_difference < best_aspect_difference):
            best_rows, best_cols, best_total = rows, cols, total
            best_aspect_difference = aspect_difference

    return best_rows, best_cols, max_squares, triangles_per_square_count


def load_inventory_by_folder(path=INVENTORY_PATH):
    with open(path) as f:
        records = json.load(f)
    if not records:
        raise ValueError(f"No triangle records found in {path}")

    records_by_folder = defaultdict(list)
    for record in records:
        records_by_folder[record["folder"]].append(record)
    return records_by_folder


def build_category_pools(records):
    category_pools = defaultdict(list)
    for record in records:
        category_pools[record["color_category"]].append(record)
    return dict(category_pools)


def build_quadrant_local_triangles(rows_per_quadrant, triangle_base_px):
    triangle_base = triangle_base_px
    triangles = []

    apex_v = 0
    anchor_us = [0]
    first_row_triangle = [
        (0, apex_v),
        (-triangle_base / 2, apex_v + triangle_base / 2),
        (triangle_base / 2, apex_v + triangle_base / 2),
    ]
    triangles.append((first_row_triangle, "row"))
    anchor_us = [-triangle_base / 2, triangle_base / 2]
    apex_v = triangle_base / 2

    for k in range(2, rows_per_quadrant + 1):
        base_v = apex_v + triangle_base / 2
        base_corners = []

        for anchor_index, anchor_u in enumerate(anchor_us):
            row_triangle = [
                (anchor_u, apex_v),
                (anchor_u - triangle_base / 2, base_v),
                (anchor_u + triangle_base / 2, base_v),
            ]
            triangles.append((row_triangle, "row"))
            base_corners.append(anchor_u - triangle_base / 2)
            if anchor_index == len(anchor_us) - 1:
                base_corners.append(anchor_u + triangle_base / 2)

        for gap_index in range(len(anchor_us) - 1):
            left_anchor = anchor_us[gap_index]
            right_anchor = anchor_us[gap_index + 1]
            gap_center = (left_anchor + right_anchor) / 2
            gap_triangle = [
                (left_anchor, apex_v),
                (right_anchor, apex_v),
                (gap_center, base_v),
            ]
            triangles.append((gap_triangle, "gap"))

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


def build_quadrant_triangles(direction, rows_per_quadrant, triangle_base_px, center_x, center_y):
    local_triangles = build_quadrant_local_triangles(rows_per_quadrant, triangle_base_px)
    return [
        ([_local_to_canvas(u, v, direction, center_x, center_y) for u, v in triangle_coords], role)
        for triangle_coords, role in local_triangles
    ]


def pick_quadrant_categories(available_categories, random_generator):
    if len(available_categories) == 1:
        return available_categories[0], available_categories[0]
    row_category, gap_category = random_generator.sample(available_categories, 2)
    return row_category, gap_category


def paste_triangle_texture(canvas, record, destination_triangle, image_cache):
    source_path = record["source_image"]
    if source_path not in image_cache:
        source_image = cv2.imread(source_path, cv2.IMREAD_COLOR)
        if source_image is None:
            raise ValueError(f"Could not load source image: {source_path}")
        image_cache[source_path] = source_image
    source_image = image_cache[source_path]

    source_points = np.array(record["vertices_px"][:3], dtype=np.float32)
    destination_points = np.array(destination_triangle[:3], dtype=np.float32)

    source_x, source_y, source_width, source_height = cv2.boundingRect(source_points.astype(np.int32))
    source_x, source_y = max(source_x, 0), max(source_y, 0)
    source_width, source_height = max(source_width, 1), max(source_height, 1)
    source_crop = source_image[
        source_y:source_y + source_height,
        source_x:source_x + source_width,
    ]
    if source_crop.size == 0:
        return
    local_source_points = (source_points - [source_x, source_y]).astype(np.float32)

    destination_x, destination_y, destination_width, destination_height = cv2.boundingRect(destination_points.astype(np.int32))
    if destination_width <= 0 or destination_height <= 0:
        return
    local_destination_points = (destination_points - [destination_x, destination_y]).astype(np.float32)

    transform = cv2.getAffineTransform(local_source_points, local_destination_points)
    warped_image = cv2.warpAffine(
        source_crop, transform, (destination_width, destination_height), borderMode=cv2.BORDER_REPLICATE
    )

    mask = np.zeros((destination_height, destination_width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, local_destination_points.astype(np.int32), 255)

    canvas_x0, canvas_y0 = max(destination_x, 0), max(destination_y, 0)
    canvas_x1 = min(destination_x + destination_width, canvas.shape[1])
    canvas_y1 = min(destination_y + destination_height, canvas.shape[0])
    if canvas_x1 <= canvas_x0 or canvas_y1 <= canvas_y0:
        return

    warped_x0, warped_y0 = canvas_x0 - destination_x, canvas_y0 - destination_y
    warped_x1 = warped_x0 + (canvas_x1 - canvas_x0)
    warped_y1 = warped_y0 + (canvas_y1 - canvas_y0)

    canvas_region = canvas[canvas_y0:canvas_y1, canvas_x0:canvas_x1]
    warped_region = warped_image[warped_y0:warped_y1, warped_x0:warped_x1]
    mask_region = mask[warped_y0:warped_y1, warped_x0:warped_x1]

    keep = mask_region > 0
    canvas_region[keep] = warped_region[keep]


def draw_square(canvas, category_pools, available_categories, center_x, center_y,
                 rows_per_quadrant, triangle_base_px, random_generator, category_positions, image_cache, used_triangle_ids):
    for direction in DIRECTIONS:
        row_category, gap_category = pick_quadrant_categories(available_categories, random_generator)
        triangles = build_quadrant_triangles(direction, rows_per_quadrant, triangle_base_px, center_x, center_y)

        for canvas_triangle, role in triangles:
            category = row_category if role == "row" else gap_category
            category_pool = category_pools[category]
            pool_index = category_positions[category]
            record = category_pool[pool_index % len(category_pool)]
            category_positions[category] = pool_index + 1
            used_triangle_ids.add(record["triangle_id"])

            paste_triangle_texture(canvas, record, canvas_triangle, image_cache)


def compose_folder(folder, records, rows=None, cols=None, rows_per_quadrant=ROWS_PER_QUADRANT,
                    triangle_base_px=TRIANGLE_BASE_PX, seed=RANDOM_SEED,
                    max_average_reuse=MAX_AVERAGE_REUSE, target_aspect_ratio=TARGET_GRID_ASPECT_RATIO,
                    total_scrap_area_in2=None):
    category_pools = build_category_pools(records)
    available_categories = list(category_pools.keys())

    if not available_categories:
        print(f"[{folder}] no triangles available, skipping")
        return None, None

    if rows is None or cols is None:
        rows, cols, max_squares, triangles_per_square_count = optimize_grid_size(
            len(records), rows_per_quadrant, max_average_reuse=max_average_reuse, target_aspect_ratio=target_aspect_ratio
        )
        used_squares = rows * cols
        reuse = (used_squares * triangles_per_square_count) / len(records)
        print(f"[{folder}] {len(records)} triangles -> {triangles_per_square_count}/square, "
              f"budget for {max_squares} squares -> {rows}x{cols} grid "
              f"({used_squares} squares, ~{reuse:.1f}x average reuse)")
    else:
        print(f"[{folder}] {len(records)} triangles across categories: {available_categories}, "
              f"grid fixed at {rows}x{cols}")

    square_size_px = rows_per_quadrant * triangle_base_px
    canvas_width = cols * square_size_px
    canvas_height = rows * square_size_px

    canvas = np.full((canvas_height, canvas_width, 3), BACKGROUND_COLOR_BGR, dtype=np.uint8)

    random_generator = random.Random(seed)
    category_positions = defaultdict(int)
    image_cache = {}
    used_triangle_ids = set()

    for row_index in range(rows):
        for col_index in range(cols):
            center_x = col_index * square_size_px + square_size_px / 2
            center_y = row_index * square_size_px + square_size_px / 2
            draw_square(
                canvas, category_pools, available_categories, center_x, center_y,
                rows_per_quadrant, triangle_base_px, random_generator, category_positions, image_cache, used_triangle_ids
            )

    output_path = f"ralli_block_{folder}.png"
    cv2.imwrite(output_path, canvas)
    print(f"[{folder}] composition saved to: {output_path} ({canvas_width}x{canvas_height}px)")

    usage_stats = compute_fabric_usage(records, used_triangle_ids, total_scrap_area_in2=total_scrap_area_in2)
    print(f"[{folder}] fabric used: {usage_stats['reused_percentage']:.1f}% "
          f"({usage_stats['used_area_in2']:.2f} in\u00b2 of "
          f"{usage_stats['total_scrap_area_in2']:.2f} in\u00b2 total), "
          f"unused: {usage_stats['unused_percentage']:.1f}% "
          f"({usage_stats['unused_area_in2']:.2f} in\u00b2)")

    return output_path, usage_stats


def compose(inventory_path=INVENTORY_PATH, rows=None, cols=None, rows_per_quadrant=ROWS_PER_QUADRANT,
            triangle_base_px=TRIANGLE_BASE_PX, seed=RANDOM_SEED,
            max_average_reuse=MAX_AVERAGE_REUSE, target_aspect_ratio=TARGET_GRID_ASPECT_RATIO):
    """Build one composition per fabric folder and save usage results."""
    records_by_folder = load_inventory_by_folder(inventory_path)
    all_triangles = [record for records in records_by_folder.values() for record in records]
    scrap_area_by_folder = compute_total_scrap_area_by_folder(all_triangles)

    output_paths = []
    usage_by_group = {}

    for folder, records in records_by_folder.items():
        output_path, usage_stats = compose_folder(
            folder, records, rows=rows, cols=cols, rows_per_quadrant=rows_per_quadrant,
            triangle_base_px=triangle_base_px, seed=seed, max_average_reuse=max_average_reuse, target_aspect_ratio=target_aspect_ratio,
            total_scrap_area_in2=scrap_area_by_folder.get(folder)
        )
        if output_path:
            output_paths.append(output_path)
            usage_by_group[folder] = usage_stats

    save_fabric_usage_csv(usage_by_group)

    return output_paths

if __name__ == "__main__":
    compose()
