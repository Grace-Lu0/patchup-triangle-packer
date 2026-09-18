import json
import random
import cv2
import numpy as np
INVENTORY_PATH = 'triangle_inventory.json'
OUTPUT_PATH = 'composition_images/3_triangle_inventory.png'
GROUP_BY = 'color_category'
NUM_GROUPS = 5
SAMPLES_PER_GROUP = 5
CELL_SIZE_PX = 150
PADDING_PX = 15
SEED = 42

def crop_triangle(image, vertices_px, cell_size, padding):
    points = np.array(vertices_px, dtype=np.float32)
    x, y, w, h = cv2.boundingRect(points.astype(np.int32))
    crop = image[y:y + h, x:x + w]
    mask = np.zeros((h, w), dtype=np.uint8)
    local_pts = (points - [x, y]).astype(np.int32)
    cv2.fillConvexPoly(mask, local_pts, 255)
    white_background = np.full_like(crop, 255)
    masked = np.where(mask[:, :, None] > 0, crop, white_background)
    inner_size = cell_size - 2 * padding
    scale = inner_size / max(w, h)
    new_width, new_height = (max(1, int(w * scale)), max(1, int(h * scale)))
    resized = cv2.resize(masked, (new_width, new_height), interpolation=cv2.INTER_AREA)
    cell = np.full((cell_size, cell_size, 3), 255, dtype=np.uint8)
    y_offset, x_offset = ((cell_size - new_height) // 2, (cell_size - new_width) // 2)
    cell[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
    return cell

def label_row(text, height, width):
    strip = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(strip, text, (8, height // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    return strip

def build_panel(inventory_path=INVENTORY_PATH, group_by=GROUP_BY, num_groups=NUM_GROUPS, samples_per_group=SAMPLES_PER_GROUP, cell_size=CELL_SIZE_PX, padding=PADDING_PX, seed=SEED, output_path=OUTPUT_PATH):
    with open(inventory_path) as f:
        records = json.load(f)
    groups = {}
    for r in records:
        groups.setdefault(r[group_by], []).append(r)
    group_names = sorted(groups, key=lambda g: -len(groups[g]))[:num_groups]
    random_generator = random.Random(seed)
    image_cache = {}
    label_width = 170
    rows = []
    for name in group_names:
        candidates = groups[name]
        sample = random_generator.sample(candidates, min(samples_per_group, len(candidates)))
        cells = []
        for record in sample:
            path = record['source_image']
            if path not in image_cache:
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                if image is None:
                    print(f'  warning: could not load {path}, skipping')
                    continue
                image_cache[path] = image
            cells.append(crop_triangle(image_cache[path], record['vertices_px'], cell_size, padding))
        while len(cells) < samples_per_group:
            cells.append(np.full((cell_size, cell_size, 3), 255, dtype=np.uint8))
        row = np.hstack([label_row(f'{name} (n={len(candidates)})', cell_size, label_width)] + cells)
        rows.append(row)
    panel = np.vstack(rows)
    cv2.imwrite(output_path, panel)
    print(f'Saved: {output_path}')
    print(f'Groups shown ({group_by}): {group_names}')
if __name__ == '__main__':
    build_panel()
