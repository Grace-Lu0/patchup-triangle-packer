import glob
import os
from inventory import build_triangle_records, save_inventory_csv, save_inventory_json
from polygons import get_calibration_scale
from report import analyze_results, process_single_scrap, save_to_csv, summarize_totals
from visualize import visualize_scrap

TRIANGLE_SIZE_INCHES = 1.0
NUM_SHIFTS = 5
BASE_FOLDER = "scraps"


def get_scrap_groups(base_folder="scraps"):

    groups = {}

    all_images = glob.glob(os.path.join(base_folder, "**", "*.png"), recursive=True)

    for img_path in all_images:
        filename = os.path.basename(img_path)
        folder = os.path.basename(os.path.dirname(img_path))

        if "_calib" in filename:
            base_name = filename.replace("_calib.png", "")
            groups.setdefault(base_name, {"calib": None, "scraps": [], "folder": folder})
            groups[base_name]["calib"] = img_path
        else:
            # e.g. "scrap1" from "scrap1_Polygon-2.png"
            base_name = filename.split("_")[0]
            groups.setdefault(base_name, {"calib": None, "scraps": [], "folder": folder})
            groups[base_name]["scraps"].append(img_path)

    return {k: v for k, v in groups.items() if v["calib"] is not None}


def clean_previous_files():
    for pattern in ["*_analysis*.png", "harvest_report*.csv", "triangle_inventory.*"]:
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except OSError:
                pass


def process_group(base_name, group, triangle_size_inches, num_shifts):

    calib_path = group["calib"]
    scrap_paths = group["scraps"]
    folder = group["folder"]

    print(f"\nGroup: {base_name} ({folder}, {len(scrap_paths)} scraps)")

    try:
        scale, _ = get_calibration_scale(calib_path, verbose=True)
    except Exception as e:
        print(f"  Calibration failed: {e}")
        return {}, []

    group_metrics = {}
    group_inventory = []

    for scrap_path in scrap_paths:
        scrap_filename = os.path.basename(scrap_path)
        scrap_id = os.path.splitext(scrap_filename)[0]
        full_id = f"{folder}_{scrap_id}"

        try:
            result = process_single_scrap(scrap_path, scale, triangle_size_inches, num_shifts)
            group_metrics[full_id] = {"scrap_type": folder, **result["metrics"]}

            group_inventory.extend(
                build_triangle_records(full_id, scrap_path, result["triangles"], scale, triangle_size_inches)
            )

            visualize_scrap(
                result["vertices"],
                result["triangles"],
                result["metrics"],
                scrap_id,
                folder,
                show_plot=False,
            )

            m = result["metrics"]
            print(f"  {scrap_filename}: {m['num_accepted_triangles']} triangles, "
                  f"{m['recovery_percentage']:.1f}% recovery "
                  f"({m['original_area_inches']:.2f} in² -> {m['total_accepted_area_inches']:.2f} in²)")

        except Exception as e:
            print(f"  {scrap_filename}: failed ({e})")
            continue

    return group_metrics, group_inventory


def main():
    print(f"Triangle size: {TRIANGLE_SIZE_INCHES} inch")

    clean_previous_files()

    scrap_groups = get_scrap_groups(BASE_FOLDER)
    if not scrap_groups:
        print(f"No scrap groups found in '{BASE_FOLDER}'. Expected layout:")
        print("  scraps/folder/scrap1_calib.png")
        print("  scraps/folder/scrap1_Polygon-2.png")
        return {}

    print(f"Found {len(scrap_groups)} scrap group(s)")

    metrics_summary = {}
    full_inventory = []
    for base_name, group in scrap_groups.items():
        group_metrics, group_inventory = process_group(base_name, group, TRIANGLE_SIZE_INCHES, NUM_SHIFTS)
        metrics_summary.update(group_metrics)
        full_inventory.extend(group_inventory)

    if not metrics_summary:
        print("No images were processed successfully.")
        return {}

    save_to_csv(metrics_summary)
    save_inventory_json(full_inventory)
    save_inventory_csv(full_inventory)

    totals = summarize_totals(metrics_summary)
    print(f"\nProcessed {len(metrics_summary)} images from {len(scrap_groups)} groups")
    print(f"Total area: {totals['total_area_inches']:.2f} in², recovered: {totals['total_recovered_inches']:.2f} in² "
          f"({totals['overall_recovery']:.1f}% overall)")
    print(f"Total triangles: {totals['total_triangles']}")

    analyze_results(metrics_summary)

    return metrics_summary


if __name__ == "__main__":
    main()