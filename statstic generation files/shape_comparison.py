import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

from polygons import extract_polygon_from_image, get_calibration_scale
from shape_harvesters import (
    harvest_306090,
    harvest_equilateral,
    harvest_isosceles,
)


BASE_FOLDER = "scraps"
TARGET_AREA_IN2 = 0.5
NUM_SHIFTS = 5

SHAPE_HARVESTERS = {
    "isosceles": harvest_isosceles,
    "306090": harvest_306090,
    "equilateral": harvest_equilateral,
}


def get_scrap_groups(base_folder):
    groups = {}

    all_images = glob.glob(
        os.path.join(base_folder, "**", "*.png"),
        recursive=True,
    )

    for image_path in all_images:
        filename = os.path.basename(image_path)
        folder = os.path.basename(os.path.dirname(image_path))

        if "_calib" in filename:
            base_name = filename.replace("_calib.png", "")
        else:
            base_name = filename.split("_")[0]

        key = (folder, base_name)

        groups.setdefault(
            key,
            {
                "calib": None,
                "scraps": [],
                "folder": folder,
            },
        )

        if "_calib" in filename:
            groups[key]["calib"] = image_path
        else:
            groups[key]["scraps"].append(image_path)

    return {
        key: group
        for key, group in groups.items()
        if group["calib"] is not None
    }


def run_comparison(
    base_folder=BASE_FOLDER,
    target_area_in2=TARGET_AREA_IN2,
    num_shifts=NUM_SHIFTS,
):
    groups = get_scrap_groups(base_folder)

    if not groups:
        print(f"No calibrated scrap groups found in '{base_folder}'.")
        return []

    rows = []

    for (folder, base_name), group in groups.items():
        try:
            scale, _ = get_calibration_scale(group["calib"])
        except Exception as error:
            print(
                f"Skipping group {base_name} ({folder}): "
                f"calibration failed ({error})"
            )
            continue

        target_area_px = target_area_in2 * (scale ** 2)

        for scrap_path in group["scraps"]:
            scrap_name = os.path.splitext(
                os.path.basename(scrap_path)
            )[0]

            scrap_id = f"{folder}_{scrap_name}"

            try:
                vertices = extract_polygon_from_image(scrap_path)
                scrap_area_px = Polygon(vertices).area
            except Exception as error:
                print(f"Skipping {scrap_id}: {error}")
                continue

            row = {
                "scrap_id": scrap_id,
                "folder": folder,
                "scrap_area_in2": scrap_area_px / (scale ** 2),
            }

            for shape_name, harvester in SHAPE_HARVESTERS.items():
                triangles = harvester(
                    vertices,
                    target_area_px,
                    num_shifts=num_shifts,
                )

                recovered_area_px = sum(
                    triangle.area for triangle in triangles
                )

                if scrap_area_px > 0:
                    recovery_percentage = (
                        recovered_area_px / scrap_area_px
                    ) * 100
                else:
                    recovery_percentage = 0

                row[f"{shape_name}_triangles"] = len(triangles)
                row[
                    f"{shape_name}_recovery_pct"
                ] = recovery_percentage

            rows.append(row)

            print(
                f"{scrap_id}: "
                + ", ".join(
                    f"{shape_name}="
                    f"{row[f'{shape_name}_recovery_pct']:.1f}%"
                    for shape_name in SHAPE_HARVESTERS
                )
            )

    if rows:
        save_csv(rows)
        plot_comparison(rows)
        print_summary(rows)

    return rows


def save_csv(
    rows,
    filename="shape_comparison_report.csv",
):
    fieldnames = [
        "scrap_id",
        "folder",
        "scrap_area_in2",
    ]

    for shape_name in SHAPE_HARVESTERS:
        fieldnames += [
            f"{shape_name}_triangles",
            f"{shape_name}_recovery_pct",
        ]

    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            output_row = dict(row)

            output_row["scrap_area_in2"] = (
                f"{row['scrap_area_in2']:.2f}"
            )

            for shape_name in SHAPE_HARVESTERS:
                output_row[
                    f"{shape_name}_recovery_pct"
                ] = (
                    f"{row[f'{shape_name}_recovery_pct']:.2f}"
                )

            writer.writerow(output_row)

    print(f"\nSaved: {filename}")


def plot_comparison(
    rows,
    filename="shape_comparison_boxplot.png",
):
    data = [
        [
            row[f"{shape_name}_recovery_pct"]
            for row in rows
        ]
        for shape_name in SHAPE_HARVESTERS
    ]

    figure, axis = plt.subplots(figsize=(7, 6))

    axis.boxplot(
        data,
        tick_labels=list(SHAPE_HARVESTERS.keys()),
    )

    for index, values in enumerate(data, start=1):
        jitter = np.random.normal(
            0,
            0.04,
            size=len(values),
        )

        axis.scatter(
            np.full(len(values), index) + jitter,
            values,
            alpha=0.4,
            s=15,
            color="steelblue",
        )

    axis.set_ylabel(
        "Recovery (%)",
        fontsize=12,
    )

    axis.set_title(
        "Recovery by Triangle Shape (matched triangle area)",
        fontsize=13,
        fontweight="bold",
    )

    axis.grid(
        True,
        alpha=0.3,
        linestyle="--",
        axis="y",
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {filename}")


def print_summary(rows):
    print(f"\nn = {len(rows)} scraps")

    for shape_name in SHAPE_HARVESTERS:
        values = np.array(
            [
                row[f"{shape_name}_recovery_pct"]
                for row in rows
            ]
        )

        print(
            f"  {shape_name:12s}: "
            f"mean={values.mean():.2f}%  "
            f"median={np.median(values):.2f}%  "
            f"std={values.std():.2f}%"
        )


if __name__ == "__main__":
    run_comparison()