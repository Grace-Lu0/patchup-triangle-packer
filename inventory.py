import colorsys
import csv
import json
import cv2
import numpy as np
from harvest import classify_orientation
HUE_CATEGORIES = [('red', 345), ('purple', 275), ('blue', 170), ('green', 70), ('yellow', 45), ('orange', 15), ('red', 0)]

def classify_color_category(rgb):
    r, g, b = (c / 255 for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    hue_deg = h * 360
    if s < 0.15:
        if v < 0.25:
            return 'black'
        if v > 0.85:
            return 'white'
        return 'grey'
    is_brown_hue = hue_deg < 40 or hue_deg > 350
    is_pink_hue = hue_deg < 40 or hue_deg > 300
    if is_brown_hue and v < 0.55 and (s > 0.35):
        return 'brown'
    if is_pink_hue and v > 0.55 and (s < 0.55):
        return 'pink'
    for name, boundary in HUE_CATEGORIES:
        if hue_deg >= boundary:
            return name
    return 'red'

def sample_triangle_color(image, vertices_px):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    points = np.array([vertices_px], dtype=np.int32)
    cv2.fillPoly(mask, points, 255)
    average_bgr = cv2.mean(image, mask=mask)[:3]
    b, g, r = average_bgr
    return (round(r), round(g), round(b))

def build_triangle_records(scrap_id, folder, image_path, triangles, scale, triangle_size_inches, scrap_original_area_in2):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Could not load image for color sampling: {image_path}')
    records = []
    for i, triangle in enumerate(triangles):
        vertices_px = [list(p) for p in list(triangle.exterior.coords)[:-1]]
        vertices_inches = [[x / scale, y / scale] for x, y in vertices_px]
        area_pixels = triangle.area
        area_inches = area_pixels / (scale * scale)
        color = sample_triangle_color(image, vertices_px)
        records.append({'triangle_id': f'{scrap_id}_t{i:04d}', 'scrap_id': scrap_id, 'folder': folder, 'source_image': image_path, 'vertices_px': vertices_px, 'vertices_in': vertices_inches, 'triangle_size_inches': triangle_size_inches, 'orientation': classify_orientation(triangle), 'area_px2': area_pixels, 'area_in2': area_inches, 'scale_px_per_in': scale, 'color': list(color), 'color_category': classify_color_category(color), 'scrap_original_area_in2': scrap_original_area_in2})
    return records

def save_inventory_json(records, filename='triangle_inventory.json'):
    with open(filename, 'w') as f:
        json.dump(records, f, indent=2)
    print(f'Triangle inventory saved to: {filename}')

def save_inventory_csv(records, filename='triangle_inventory.csv'):
    if not records:
        print('No triangle records to save!')
        return
    fieldnames = ['triangle_id', 'scrap_id', 'folder', 'source_image', 'v1_x_px', 'v1_y_px', 'v2_x_px', 'v2_y_px', 'v3_x_px', 'v3_y_px', 'triangle_size_inches', 'orientation', 'area_px2', 'area_in2', 'scale_px_per_in', 'color_r', 'color_g', 'color_b', 'color_category', 'scrap_original_area_in2']
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            (v1x, v1y), (v2x, v2y), (v3x, v3y) = r['vertices_px']
            cr, cg, cb = r['color']
            writer.writerow({'triangle_id': r['triangle_id'], 'scrap_id': r['scrap_id'], 'folder': r['folder'], 'source_image': r['source_image'], 'v1_x_px': v1x, 'v1_y_px': v1y, 'v2_x_px': v2x, 'v2_y_px': v2y, 'v3_x_px': v3x, 'v3_y_px': v3y, 'triangle_size_inches': r['triangle_size_inches'], 'orientation': r['orientation'], 'area_px2': f"{r['area_px2']:.2f}", 'area_in2': f"{r['area_in2']:.4f}", 'scale_px_per_in': f"{r['scale_px_per_in']:.2f}", 'color_r': cr, 'color_g': cg, 'color_b': cb, 'color_category': r['color_category'], 'scrap_original_area_in2': f"{r['scrap_original_area_in2']:.4f}"})
    print(f'Triangle inventory CSV saved to: {filename}')
