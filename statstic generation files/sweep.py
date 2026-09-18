import csv

import matplotlib.pyplot as plt

from main import get_scrap_groups
from polygons import get_calibration_scale
from report import process_single_scrap

# Sweep configuration
BASE_FOLDER = "scraps"
TRIANGLE_SIZES_INCHES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
NUM_SHIFTS = 5


def calibrate_groups(scrap_groups):

    scales = {}
    for base_name, group in scrap_groups.items():
        try:
            scale, _ = get_calibration_scale(group["calib"])
            scales[base_name] = scale
        except Exception as e:
            print(f"  Calibration failed for {base_name}: {e}")
    return scales


def run_for_size(scrap_groups, scales, triangle_size_inches, num_shifts):
    total_triangles = 0
    total_area_inches = 0.0
    total_recovered_inches = 0.0
    num_scraps = 0

    for base_name, group in scrap_groups.items():
        if base_name not in scales:
            continue
        scale = scales[base_name]

        for scrap_path in group["scraps"]:
            try:
                result = process_single_scrap(scrap_path, scale, triangle_size_inches, num_shifts)
            except Exception as e:
                print(f"  Skipping {scrap_path}: {e}")
                continue

            m = result["metrics"]
            total_triangles += m["num_accepted_triangles"]
            total_area_inches += m["original_area_inches"]
            total_recovered_inches += m["total_accepted_area_inches"]
            num_scraps += 1

    recovery_pct = (total_recovered_inches / total_area_inches) * 100 if total_area_inches > 0 else 0


    avg_pieces_per_scrap = total_triangles / num_scraps if num_scraps > 0 else 0

    return {
        "triangle_size_inches": triangle_size_inches,
        "num_scraps": num_scraps,
        "total_triangles": total_triangles,
        "total_area_inches": total_area_inches,
        "total_recovered_inches": total_recovered_inches,
        "recovery_percentage": recovery_pct,
        "avg_pieces_per_scrap": avg_pieces_per_scrap,
    }


def save_sweep_csv(results, filename="sweep_report.csv"):
    fieldnames = [
        "Triangle Size (in)", "Scraps Processed", "Total Triangles",
        "Total Area (in\u00b2)", "Recovered Area (in\u00b2)", "Recovery %",
        "Avg Pieces per Scrap",
    ]
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "Triangle Size (in)": f"{r['triangle_size_inches']:.2f}",
                "Scraps Processed": r["num_scraps"],
                "Total Triangles": r["total_triangles"],
                "Total Area (in\u00b2)": f"{r['total_area_inches']:.2f}",
                "Recovered Area (in\u00b2)": f"{r['total_recovered_inches']:.2f}",
                "Recovery %": f"{r['recovery_percentage']:.2f}",
                "Avg Pieces per Scrap": f"{r['avg_pieces_per_scrap']:.1f}",
            })
    print(f"Sweep CSV saved to: {filename}")


def plot_tradeoffs(results, output_path="triangle_size_tradeoffs.png"):
    sizes = [r["triangle_size_inches"] for r in results]
    recoveries = [r["recovery_percentage"] for r in results]
    counts = [r["total_triangles"] for r in results]

    fig, ax1 = plt.subplots(figsize=(9, 6))

    color1 = "tab:blue"
    ax1.set_xlabel("Triangle size (inches)")
    ax1.set_ylabel("Recovery (%)", color=color1)
    ax1.plot(sizes, recoveries, marker="o", color=color1, label="Recovery %")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.set_ylabel("Total triangles harvested", color=color2)
    ax2.plot(sizes, counts, marker="s", color=color2, label="Triangle count")
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.suptitle("Triangle size vs. recovery and piece count")
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Tradeoff plot saved to: {output_path}")


def run_sweep(base_folder=BASE_FOLDER, triangle_sizes=TRIANGLE_SIZES_INCHES, num_shifts=NUM_SHIFTS):
    scrap_groups = get_scrap_groups(base_folder)
    if not scrap_groups:
        print(f"No scrap groups found in '{base_folder}'.")
        return []

    print(f"Calibrating {len(scrap_groups)} group(s)...")
    scales = calibrate_groups(scrap_groups)

    results = []
    for size in triangle_sizes:
        summary = run_for_size(scrap_groups, scales, size, num_shifts)
        results.append(summary)
        print(f"Triangle size {size} in: {summary['num_scraps']} scraps, "
              f"{summary['total_triangles']} triangles, {summary['recovery_percentage']:.1f}% recovery")

    save_sweep_csv(results)
    plot_tradeoffs(results)

    return results


run_sweep()