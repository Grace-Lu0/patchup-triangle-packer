
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
TARGET_AREAS_IN2 = [0.25, 0.5, 1.0, 2.0]
NUM_SHIFTS = 5

SHAPE_HARVESTERS = {
    "isosceles": harvest_isosceles,
    "306090": harvest_306090,
    "equilateral": harvest_equilateral,
}

SHAPE_COLORS = {
    "isosceles": "#4C6EF5",
    "306090": "#F59F00",
    "equilateral": "#40C057",
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


def run_sweep(
    base_folder=BASE_FOLDER,
    target_areas_in2=TARGET_AREAS_IN2,
    num_shifts=NUM_SHIFTS,
):
    groups = get_scrap_groups(base_folder)

    if not groups:
        print(f"No calibrated scrap groups found in '{base_folder}'.")
        return []

    scraps = []

    for (folder, base_name), group in groups.items():
        try:
            scale, _ = get_calibration_scale(group["calib"])
        except Exception as error:
            print(
                f"Skipping group {base_name} ({folder}): "
                f"calibration failed ({error})"
            )
            continue

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

            scraps.append(
                {
                    "scrap_id": scrap_id,
                    "folder": folder,
                    "vertices": vertices,
                    "scale": scale,
                    "scrap_area_px": scrap_area_px,
                }
            )

    print(
        f"Loaded {len(scraps)} scraps, sweeping "
        f"{len(target_areas_in2)} sizes x "
        f"{len(SHAPE_HARVESTERS)} shapes\n"
    )

    results = []

    for target_area_in2 in target_areas_in2:
        print(f"Triangle area = {target_area_in2} in²")

        for scrap in scraps:
            target_area_px = (
                target_area_in2 * scrap["scale"] ** 2
            )

            row = {
                "scrap_id": scrap["scrap_id"],
                "folder": scrap["folder"],
                "target_area_in2": target_area_in2,
            }

            for shape_name, harvester in SHAPE_HARVESTERS.items():
                triangles = harvester(
                    scrap["vertices"],
                    target_area_px,
                    num_shifts=num_shifts,
                )

                recovered_area_px = sum(
                    triangle.area for triangle in triangles
                )

                if scrap["scrap_area_px"] > 0:
                    recovery_percentage = (
                        recovered_area_px
                        / scrap["scrap_area_px"]
                    ) * 100
                else:
                    recovery_percentage = 0

                row[
                    f"{shape_name}_recovery_pct"
                ] = recovery_percentage

            results.append(row)

        means = {}

        for shape_name in SHAPE_HARVESTERS:
            values = [
                row[f"{shape_name}_recovery_pct"]
                for row in results
                if row["target_area_in2"] == target_area_in2
            ]

            means[shape_name] = np.mean(values)

        print(
            "  "
            + ", ".join(
                f"{shape_name}={mean_value:.1f}%"
                for shape_name, mean_value in means.items()
            )
        )

    if results:
        save_csv(results)
        plot_grouped_bar(results, target_areas_in2)

    return results


def save_csv(
    rows,
    filename="shape_size_sweep_report.csv",
):
    fieldnames = [
        "scrap_id",
        "folder",
        "target_area_in2",
    ] + [
        f"{shape_name}_recovery_pct"
        for shape_name in SHAPE_HARVESTERS
    ]

    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            output_row = dict(row)

            for shape_name in SHAPE_HARVESTERS:
                output_row[
                    f"{shape_name}_recovery_pct"
                ] = (
                    f"{row[f'{shape_name}_recovery_pct']:.2f}"
                )

            writer.writerow(output_row)

    print(f"\nSaved: {filename}")


def plot_grouped_bar(
    rows,
    target_areas_in2,
    filename="shape_size_sweep_bar.png",
):
    shape_names = list(SHAPE_HARVESTERS.keys())

    num_shapes = len(shape_names)
    num_sizes = len(target_areas_in2)

    means = np.zeros((num_sizes, num_shapes))
    sems = np.zeros((num_sizes, num_shapes))

    for area_index, area in enumerate(target_areas_in2):
        size_rows = [
            row
            for row in rows
            if row["target_area_in2"] == area
        ]

        for shape_index, shape_name in enumerate(shape_names):
            values = np.array(
                [
                    row[f"{shape_name}_recovery_pct"]
                    for row in size_rows
                ]
            )

            means[area_index, shape_index] = values.mean()

            if len(values) > 1:
                sems[area_index, shape_index] = (
                    values.std(ddof=1)
                    / np.sqrt(len(values))
                )
            else:
                sems[area_index, shape_index] = 0

    figure, axis = plt.subplots(figsize=(10, 6))

    bar_width = 0.8 / num_shapes
    x_positions = np.arange(num_sizes)

    for shape_index, shape_name in enumerate(shape_names):
        offset = (
            shape_index - (num_shapes - 1) / 2
        ) * bar_width

        axis.bar(
            x_positions + offset,
            means[:, shape_index],
            bar_width,
            yerr=sems[:, shape_index],
            capsize=4,
            label=shape_name,
            color=SHAPE_COLORS[shape_name],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.8,
            error_kw={
                "elinewidth": 1.2,
                "capthick": 1.2,
            },
        )

    axis.set_xticks(x_positions)

    axis.set_xticklabels(
        [
            f"{area} in²"
            for area in target_areas_in2
        ]
    )

    axis.set_xlabel(
        "Triangle area",
        fontsize=12,
    )

    axis.set_ylabel(
        "Recovery (%)",
        fontsize=12,
    )

    axis.set_title(
        "Mean Recovery by Triangle Shape and Size\n"
        "error bars = SEM",
        fontsize=13,
        fontweight="bold",
    )

    axis.legend(title="Shape")

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


if __name__ == "__main__":
    run_sweep()