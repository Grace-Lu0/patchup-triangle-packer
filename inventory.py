import csv
import json
from harvest import classify_orientation


def build_triangle_records(scrap_id, image_path, triangles, scale, triangle_size_inches):

    records = []
    for i, triangle in enumerate(triangles):
        vertices_px = [list(p) for p in list(triangle.exterior.coords)[:-1]]
        vertices_in = [[x / scale, y / scale] for x, y in vertices_px]

        area_px2 = triangle.area
        area_in2 = area_px2 / (scale * scale)

        records.append({
            "triangle_id": f"{scrap_id}_t{i:04d}",
            "scrap_id": scrap_id,
            "source_image": image_path,
            "vertices_px": vertices_px,
            "vertices_in": vertices_in,
            "triangle_size_inches": triangle_size_inches,
            "orientation": classify_orientation(triangle),
            "area_px2": area_px2,
            "area_in2": area_in2,
            "scale_px_per_in": scale,
            "color": None,
        })

    return records


def save_inventory_json(records, filename="triangle_inventory.json"):
    with open(filename, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Triangle inventory saved to: {filename}")


def save_inventory_csv(records, filename="triangle_inventory.csv"):

    if not records:
        print("No triangle records to save!")
        return

    fieldnames = [
        "triangle_id", "scrap_id", "source_image",
        "v1_x_px", "v1_y_px", "v2_x_px", "v2_y_px", "v3_x_px", "v3_y_px",
        "triangle_size_inches", "orientation", "area_px2", "area_in2",
        "scale_px_per_in", "color",
    ]

    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for r in records:
            (v1x, v1y), (v2x, v2y), (v3x, v3y) = r["vertices_px"]
            writer.writerow({
                "triangle_id": r["triangle_id"],
                "scrap_id": r["scrap_id"],
                "source_image": r["source_image"],
                "v1_x_px": v1x, "v1_y_px": v1y,
                "v2_x_px": v2x, "v2_y_px": v2y,
                "v3_x_px": v3x, "v3_y_px": v3y,
                "triangle_size_inches": r["triangle_size_inches"],
                "orientation": r["orientation"],
                "area_px2": f"{r['area_px2']:.2f}",
                "area_in2": f"{r['area_in2']:.4f}",
                "scale_px_per_in": f"{r['scale_px_per_in']:.2f}",
                "color": r["color"] if r["color"] is not None else "",
            })

    print(f"Triangle inventory CSV saved to: {filename}")