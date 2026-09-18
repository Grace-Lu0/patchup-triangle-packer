import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

from polygons import extract_polygon_from_image


HARVEST_CSV_PATH = "../scrap_analysis_csv/harvest_report.csv"
INVENTORY_JSON_PATH = "../triangle_inventory.json"
RECTANGULARITY_SPLIT = 0.85


def compute_rectangularity(vertices):
    polygon = Polygon(vertices)
    bounding_rectangle = polygon.minimum_rotated_rectangle

    if bounding_rectangle.area == 0:
        return 0.0

    return polygon.area / bounding_rectangle.area


def load_recovery_by_scrap(harvest_csv_path):
    recovery_by_scrap = {}
    area_by_scrap = {}

    with open(harvest_csv_path, newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            scrap_id = row.get("Scrap ID", "")

            if not scrap_id or row.get("Scrap Type") == "OVERALL":
                continue

            try:
                recovery_by_scrap[scrap_id] = float(row["Recovery %"])
                area_by_scrap[scrap_id] = float(
                    row["Original Area (in²)"]
                )
            except (ValueError, KeyError):
                continue

    return recovery_by_scrap, area_by_scrap


def load_source_image_by_scrap(inventory_json_path):
    with open(inventory_json_path) as file:
        records = json.load(file)

    source_by_scrap = {}

    for record in records:
        source_by_scrap.setdefault(
            record["scrap_id"],
            record["source_image"],
        )

    return source_by_scrap


def run_analysis(
    harvest_csv_path=HARVEST_CSV_PATH,
    inventory_json_path=INVENTORY_JSON_PATH,
):
    recovery_by_scrap, area_by_scrap = load_recovery_by_scrap(
        harvest_csv_path
    )

    source_by_scrap = load_source_image_by_scrap(
        inventory_json_path
    )

    rows = []

    for scrap_id, recovery_percentage in recovery_by_scrap.items():
        image_path = source_by_scrap.get(scrap_id)

        if image_path is None:
            print(
                f"Skipping {scrap_id}: "
                "no source image found in inventory"
            )
            continue

        try:
            vertices = extract_polygon_from_image(image_path)
            rectangularity = compute_rectangularity(vertices)
        except Exception as error:
            print(f"Skipping {scrap_id}: {error}")
            continue

        rows.append(
            {
                "scrap_id": scrap_id,
                "rectangularity": rectangularity,
                "recovery_percentage": recovery_percentage,
                "original_area_in2": area_by_scrap.get(scrap_id, 0),
            }
        )

    if not rows:
        print(
            "No scraps could be analyzed. Check that "
            "harvest_report.csv and triangle_inventory.json "
            "are present and up to date."
        )
        return []

    save_csv(rows)
    plot_scatter(rows)
    print_summary(rows)

    return rows


def save_csv(
    rows,
    filename="shape_recovery_analysis.csv",
):
    fieldnames = [
        "scrap_id",
        "rectangularity",
        "recovery_percentage",
        "original_area_in2",
    ]

    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "scrap_id": row["scrap_id"],
                    "rectangularity":
                        f"{row['rectangularity']:.4f}",
                    "recovery_percentage":
                        f"{row['recovery_percentage']:.2f}",
                    "original_area_in2":
                        f"{row['original_area_in2']:.2f}",
                }
            )

    print(f"Saved: {filename}")


def plot_scatter(
    rows,
    filename="shape_recovery_scatter.png",
):
    rectangularity_values = np.array(
        [row["rectangularity"] for row in rows]
    )

    recovery_values = np.array(
        [row["recovery_percentage"] for row in rows]
    )

    figure, axis = plt.subplots(figsize=(8, 6))

    axis.scatter(
        rectangularity_values,
        recovery_values,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )

    if len(rectangularity_values) >= 2:
        slope, intercept = np.polyfit(
            rectangularity_values,
            recovery_values,
            1,
        )

        x_line = np.linspace(
            rectangularity_values.min(),
            rectangularity_values.max(),
            100,
        )

        axis.plot(
            x_line,
            slope * x_line + intercept,
            color="red",
            linestyle="--",
            label=f"trend: y = {slope:.1f}x + {intercept:.1f}",
        )

        axis.legend()

    axis.set_xlabel(
        "Rectangularity (scrap area / bounding rectangle area)",
        fontsize=12,
    )

    axis.set_ylabel(
        "Recovery (%)",
        fontsize=12,
    )

    axis.set_title(
        "Scrap Shape Regularity vs. Triangle Harvest Recovery",
        fontsize=13,
        fontweight="bold",
    )

    axis.grid(
        True,
        alpha=0.3,
        linestyle="--",
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {filename}")


def print_summary(
    rows,
    split=RECTANGULARITY_SPLIT,
):
    rectangularity_values = np.array(
        [row["rectangularity"] for row in rows]
    )

    recovery_values = np.array(
        [row["recovery_percentage"] for row in rows]
    )

    if len(rectangularity_values) >= 2:
        correlation = np.corrcoef(
            rectangularity_values,
            recovery_values,
        )[0, 1]

        print(
            "\nPearson correlation "
            "(rectangularity vs. recovery %): "
            f"r = {correlation:.3f}"
        )

    high_recovery = recovery_values[
        rectangularity_values >= split
    ]

    low_recovery = recovery_values[
        rectangularity_values < split
    ]

    if len(high_recovery):
        print(
            f"\nHigh-rectangularity scraps (>= {split}): "
            f"n={len(high_recovery)}, "
            f"mean recovery = {high_recovery.mean():.1f}%"
        )
    else:
        print("\nHigh-rectangularity scraps: n=0")

    if len(low_recovery):
        print(
            f"Low-rectangularity scraps (< {split}): "
            f"n={len(low_recovery)}, "
            f"mean recovery = {low_recovery.mean():.1f}%"
        )
    else:
        print("Low-rectangularity scraps: n=0")


if __name__ == "__main__":
    run_analysis()