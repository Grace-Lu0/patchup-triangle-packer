import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

from harvest import (
    _best_triangles_for_diagonal,
    find_best_triangles,
)
from main import get_scrap_groups
from polygons import (
    extract_polygon_from_image,
    get_calibration_scale,
)


BASE_FOLDER = "scraps"
TRIANGLE_SIZE_INCHES = 1.0
NUM_SHIFTS = 5

PER_SCRAP_CSV = "orientation_ablation_per_scrap.csv"
SUMMARY_CSV = "orientation_ablation_summary.csv"
OUTPUT_FIGURE = "orientation_ablation.png"


def get_orientation_harvests(
    vertices,
    triangle_size_inches,
    scale,
    num_shifts,
):
    triangle_size_pixels = triangle_size_inches * scale

    shifts = np.linspace(
        0,
        triangle_size_pixels,
        num_shifts,
        endpoint=False,
    )

    main_triangles = _best_triangles_for_diagonal(
        vertices,
        triangle_size_inches,
        scale,
        shifts,
        "main",
    )

    anti_triangles = _best_triangles_for_diagonal(
        vertices,
        triangle_size_inches,
        scale,
        shifts,
        "anti",
    )

    if len(main_triangles) >= len(anti_triangles):
        best_orientation = list(main_triangles)
    else:
        best_orientation = list(anti_triangles)

    greedy_triangles = find_best_triangles(
        vertices,
        triangle_size_inches,
        scale,
        num_shifts=num_shifts,
        overlap_tolerance=0.01,
    )

    return {
        "fixed_main": main_triangles,
        "fixed_anti": anti_triangles,
        "best_orientation": best_orientation,
        "greedy": greedy_triangles,
    }


def compute_recovery(scrap_polygon, triangles):
    scrap_area = scrap_polygon.area
    recovered_area = sum(
        triangle.area for triangle in triangles
    )

    if scrap_area > 0:
        recovery_percentage = (
            recovered_area / scrap_area
        ) * 100
    else:
        recovery_percentage = 0.0

    return {
        "scrap_area_px2": scrap_area,
        "recovered_area_px2": recovered_area,
        "recovery_percentage": recovery_percentage,
        "triangle_count": len(triangles),
    }


def run_ablation(
    base_folder=BASE_FOLDER,
    triangle_size_inches=TRIANGLE_SIZE_INCHES,
    num_shifts=NUM_SHIFTS,
):
    scrap_groups = get_scrap_groups(base_folder)

    if not scrap_groups:
        print(
            f"No calibrated scrap groups found in "
            f"'{base_folder}'."
        )
        return []

    print(
        f"Found {len(scrap_groups)} calibrated scrap groups."
    )
    print(
        f"Triangle size: {triangle_size_inches:.2f} inches"
    )
    print(f"Shift resolution: {num_shifts}")
    print()

    rows = []
    processed = 0

    for (folder, base_name), group in scrap_groups.items():
        try:
            scale, _ = get_calibration_scale(
                group["calib"]
            )
        except Exception as error:
            print(
                f"Skipping group {base_name} ({folder}): "
                f"calibration failed ({error})"
            )
            continue

        for scrap_path in group["scraps"]:
            filename = os.path.basename(scrap_path)
            scrap_name = os.path.splitext(filename)[0]
            scrap_id = f"{folder}_{scrap_name}"

            try:
                vertices = extract_polygon_from_image(
                    scrap_path
                )
                scrap_polygon = Polygon(vertices)
            except Exception as error:
                print(
                    f"Skipping {scrap_id}: "
                    f"polygon extraction failed ({error})"
                )
                continue

            try:
                harvests = get_orientation_harvests(
                    vertices,
                    triangle_size_inches,
                    scale,
                    num_shifts,
                )
            except Exception as error:
                print(
                    f"Skipping {scrap_id}: "
                    f"harvesting failed ({error})"
                )
                continue

            main_metrics = compute_recovery(
                scrap_polygon,
                harvests["fixed_main"],
            )

            anti_metrics = compute_recovery(
                scrap_polygon,
                harvests["fixed_anti"],
            )

            best_metrics = compute_recovery(
                scrap_polygon,
                harvests["best_orientation"],
            )

            greedy_metrics = compute_recovery(
                scrap_polygon,
                harvests["greedy"],
            )

            scrap_area_in2 = (
                scrap_polygon.area / (scale ** 2)
            )

            row = {
                "scrap_id": scrap_id,
                "folder": folder,
                "scrap_area_in2": scrap_area_in2,

                "fixed_main_triangles":
                    main_metrics["triangle_count"],
                "fixed_main_recovery_pct":
                    main_metrics["recovery_percentage"],
                "fixed_main_recovered_px2":
                    main_metrics["recovered_area_px2"],

                "fixed_anti_triangles":
                    anti_metrics["triangle_count"],
                "fixed_anti_recovery_pct":
                    anti_metrics["recovery_percentage"],
                "fixed_anti_recovered_px2":
                    anti_metrics["recovered_area_px2"],

                "best_orientation_triangles":
                    best_metrics["triangle_count"],
                "best_orientation_recovery_pct":
                    best_metrics["recovery_percentage"],
                "best_orientation_recovered_px2":
                    best_metrics["recovered_area_px2"],

                "greedy_triangles":
                    greedy_metrics["triangle_count"],
                "greedy_recovery_pct":
                    greedy_metrics["recovery_percentage"],
                "greedy_recovered_px2":
                    greedy_metrics["recovered_area_px2"],

                "scrap_area_px2":
                    scrap_polygon.area,

                "scale_px_per_in":
                    scale,
            }

            rows.append(row)
            processed += 1

            print(
                f"{processed:3d}  {scrap_id}: "
                f"main={main_metrics['recovery_percentage']:.1f}%  "
                f"anti={anti_metrics['recovery_percentage']:.1f}%  "
                f"best={best_metrics['recovery_percentage']:.1f}%  "
                f"greedy={greedy_metrics['recovery_percentage']:.1f}%"
            )

    if not rows:
        print("No scraps were processed successfully.")
        return []

    save_per_scrap_csv(rows)

    summary = compute_summary(rows)

    save_summary_csv(summary)
    make_plot(summary)
    print_summary(summary, len(rows))

    return rows


def save_per_scrap_csv(
    rows,
    filename=PER_SCRAP_CSV,
):
    fieldnames = [
        "scrap_id",
        "folder",
        "scrap_area_in2",
        "fixed_main_triangles",
        "fixed_main_recovery_pct",
        "fixed_anti_triangles",
        "fixed_anti_recovery_pct",
        "best_orientation_triangles",
        "best_orientation_recovery_pct",
        "greedy_triangles",
        "greedy_recovery_pct",
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
                    "scrap_id":
                        row["scrap_id"],

                    "folder":
                        row["folder"],

                    "scrap_area_in2":
                        f"{row['scrap_area_in2']:.4f}",

                    "fixed_main_triangles":
                        row["fixed_main_triangles"],

                    "fixed_main_recovery_pct":
                        f"{row['fixed_main_recovery_pct']:.4f}",

                    "fixed_anti_triangles":
                        row["fixed_anti_triangles"],

                    "fixed_anti_recovery_pct":
                        f"{row['fixed_anti_recovery_pct']:.4f}",

                    "best_orientation_triangles":
                        row["best_orientation_triangles"],

                    "best_orientation_recovery_pct":
                        f"{row['best_orientation_recovery_pct']:.4f}",

                    "greedy_triangles":
                        row["greedy_triangles"],

                    "greedy_recovery_pct":
                        f"{row['greedy_recovery_pct']:.4f}",
                }
            )

    print()
    print(f"Saved: {filename}")


def compute_summary(rows):
    conditions = [
        ("fixed_main", "Fixed main orientation"),
        ("fixed_anti", "Fixed anti orientation"),
        ("best_orientation", "Best single orientation"),
        ("greedy", "Best + greedy fill-in"),
    ]

    summary = []

    for key, label in conditions:
        total_scrap_area_in2 = 0.0
        total_recovered_area_in2 = 0.0
        total_triangles = 0
        per_scrap_recoveries = []

        for row in rows:
            scale = row["scale_px_per_in"]

            scrap_area_in2 = (
                row["scrap_area_px2"]
                / (scale ** 2)
            )

            recovered_area_in2 = (
                row[f"{key}_recovered_px2"]
                / (scale ** 2)
            )

            total_scrap_area_in2 += scrap_area_in2
            total_recovered_area_in2 += recovered_area_in2

            total_triangles += row[
                f"{key}_triangles"
            ]

            per_scrap_recoveries.append(
                row[f"{key}_recovery_pct"]
            )

        if total_scrap_area_in2 > 0:
            overall_recovery = (
                total_recovered_area_in2
                / total_scrap_area_in2
            ) * 100
        else:
            overall_recovery = 0.0

        mean_recovery = float(
            np.mean(per_scrap_recoveries)
        )

        median_recovery = float(
            np.median(per_scrap_recoveries)
        )

        std_recovery = float(
            np.std(
                per_scrap_recoveries,
                ddof=1,
            )
        )

        if len(per_scrap_recoveries) > 1:
            sem_recovery = (
                std_recovery
                / np.sqrt(len(per_scrap_recoveries))
            )
        else:
            sem_recovery = 0.0

        summary.append(
            {
                "condition": key,
                "label": label,
                "num_scraps": len(rows),
                "total_scrap_area_in2":
                    total_scrap_area_in2,
                "total_recovered_area_in2":
                    total_recovered_area_in2,
                "total_triangles":
                    total_triangles,
                "overall_recovery_pct":
                    overall_recovery,
                "mean_recovery_pct":
                    mean_recovery,
                "median_recovery_pct":
                    median_recovery,
                "std_recovery_pct":
                    std_recovery,
                "sem_recovery_pct":
                    sem_recovery,
            }
        )

    return summary


def save_summary_csv(
    summary,
    filename=SUMMARY_CSV,
):
    fieldnames = [
        "Condition",
        "Scraps",
        "Total Scrap Area (in2)",
        "Recovered Area (in2)",
        "Total Triangles",
        "Overall Recovery (%)",
        "Mean Recovery (%)",
        "Median Recovery (%)",
        "Std Recovery (%)",
        "SEM Recovery (%)",
    ]

    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in summary:
            writer.writerow(
                {
                    "Condition":
                        result["label"],

                    "Scraps":
                        result["num_scraps"],

                    "Total Scrap Area (in2)":
                        f"{result['total_scrap_area_in2']:.4f}",

                    "Recovered Area (in2)":
                        f"{result['total_recovered_area_in2']:.4f}",

                    "Total Triangles":
                        result["total_triangles"],

                    "Overall Recovery (%)":
                        f"{result['overall_recovery_pct']:.4f}",

                    "Mean Recovery (%)":
                        f"{result['mean_recovery_pct']:.4f}",

                    "Median Recovery (%)":
                        f"{result['median_recovery_pct']:.4f}",

                    "Std Recovery (%)":
                        f"{result['std_recovery_pct']:.4f}",

                    "SEM Recovery (%)":
                        f"{result['sem_recovery_pct']:.4f}",
                }
            )

    print(f"Saved: {filename}")


def make_plot(
    summary,
    filename=OUTPUT_FIGURE,
):
    primary_conditions = [
        result
        for result in summary
        if result["condition"] in {
            "fixed_main",
            "best_orientation",
            "greedy",
        }
    ]

    labels = [
        "Fixed\norientation",
        "Best of two\norientations",
        "Best + greedy\nfill-in",
    ]

    means = [
        result["mean_recovery_pct"]
        for result in primary_conditions
    ]

    sems = [
        result["sem_recovery_pct"]
        for result in primary_conditions
    ]

    x_positions = np.arange(len(labels))

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    bars = axis.bar(
        x_positions,
        means,
        yerr=sems,
        capsize=5,
        alpha=0.85,
        edgecolor="black",
    )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels)

    axis.set_ylabel(
        "Mean Recovery (%)",
        fontsize=12,
    )

    axis.set_title(
        "Orientation and Greedy Combination Ablation",
        fontsize=13,
        fontweight="bold",
    )

    axis.grid(
        True,
        axis="y",
        alpha=0.3,
        linestyle="--",
    )

    for bar, value in zip(bars, means):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {filename}")


def print_summary(summary, num_scraps):
    results = {
        result["condition"]: result
        for result in summary
    }

    main = results["fixed_main"]
    anti = results["fixed_anti"]
    best = results["best_orientation"]
    greedy = results["greedy"]

    orientation_gain = (
        best["overall_recovery_pct"]
        - main["overall_recovery_pct"]
    )

    greedy_gain = (
        greedy["overall_recovery_pct"]
        - best["overall_recovery_pct"]
    )

    total_gain = (
        greedy["overall_recovery_pct"]
        - main["overall_recovery_pct"]
    )

    print()
    print("=" * 70)
    print("ORIENTATION ABLATION RESULTS")
    print("=" * 70)

    print(f"Scraps processed: {num_scraps}")
    print()

    print(
        "Fixed main orientation:  "
        f"{main['overall_recovery_pct']:.2f}% overall recovery"
    )

    print(
        "Fixed anti orientation:  "
        f"{anti['overall_recovery_pct']:.2f}% overall recovery"
    )

    print(
        "Best single orientation: "
        f"{best['overall_recovery_pct']:.2f}% overall recovery"
    )

    print(
        "Best + greedy fill-in:   "
        f"{greedy['overall_recovery_pct']:.2f}% overall recovery"
    )

    print()

    print(
        "Gain from considering both orientations: "
        f"+{orientation_gain:.2f} percentage points"
    )

    print(
        "Additional gain from greedy fill-in:      "
        f"+{greedy_gain:.2f} percentage points"
    )

    print(
        "Total gain over fixed main orientation:   "
        f"+{total_gain:.2f} percentage points"
    )

    print()
    print("Mean per-scrap recovery:")

    print(
        f"  Fixed main:  "
        f"{main['mean_recovery_pct']:.2f}%"
    )

    print(
        f"  Fixed anti:  "
        f"{anti['mean_recovery_pct']:.2f}%"
    )

    print(
        f"  Best single: "
        f"{best['mean_recovery_pct']:.2f}%"
    )

    print(
        f"  Greedy:      "
        f"{greedy['mean_recovery_pct']:.2f}%"
    )

    print("=" * 70)


if __name__ == "__main__":
    run_ablation()