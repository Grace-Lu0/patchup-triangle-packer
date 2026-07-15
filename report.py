import csv
from shapely.geometry import Polygon
from harvest import find_best_triangles
from polygons import extract_polygon_from_image


def compute_metrics(vertices, accepted_triangles, scale, triangle_size_inches):

    scrap_poly = Polygon(vertices)

    original_area_pixels = scrap_poly.area
    original_area_inches = original_area_pixels / (scale * scale)

    total_accepted_area_pixels = sum(tri.area for tri in accepted_triangles)
    total_accepted_area_inches = total_accepted_area_pixels / (scale * scale)

    num_accepted = len(accepted_triangles)

    triangle_area_inches = 0.5 * triangle_size_inches * triangle_size_inches
    theoretical_area_inches = num_accepted * triangle_area_inches

    recovery_percentage = (
        (total_accepted_area_inches / original_area_inches) * 100
        if original_area_inches > 0
        else 0
    )

    return {
        "original_area_pixels": original_area_pixels,
        "original_area_inches": original_area_inches,
        "num_accepted_triangles": num_accepted,
        "total_accepted_area_pixels": total_accepted_area_pixels,
        "total_accepted_area_inches": total_accepted_area_inches,
        "theoretical_area_inches": theoretical_area_inches,
        "recovery_percentage": recovery_percentage,
        "triangle_size_inches": triangle_size_inches,
        "scale": scale,
    }


def process_single_scrap(image_path, calib_scale, triangle_size_inches, num_shifts=5):

    vertices = extract_polygon_from_image(image_path)
    accepted_triangles = find_best_triangles(vertices, triangle_size_inches, calib_scale, num_shifts)
    metrics = compute_metrics(vertices, accepted_triangles, calib_scale, triangle_size_inches)

    return {
        "vertices": vertices,
        "triangles": accepted_triangles,
        "metrics": metrics,
        "image_path": image_path,
    }


def summarize_totals(metrics_summary):

    total_area_inches = sum(m["original_area_inches"] for m in metrics_summary.values())
    total_recovered_inches = sum(m["total_accepted_area_inches"] for m in metrics_summary.values())
    total_triangles = sum(m["num_accepted_triangles"] for m in metrics_summary.values())
    overall_recovery = (total_recovered_inches / total_area_inches) * 100 if total_area_inches > 0 else 0

    return {
        "total_area_inches": total_area_inches,
        "total_recovered_inches": total_recovered_inches,
        "total_triangles": total_triangles,
        "overall_recovery": overall_recovery,
    }


def save_to_csv(metrics_summary, filename="harvest_report.csv"):

    if not metrics_summary:
        print("No metrics to save!")
        return

    totals = summarize_totals(metrics_summary)

    fieldnames = [
        "Scrap Type", "Scrap ID", "Original Area (in²)", "Triangle Size (in)",
        "Scale (px/in)", "Accepted Triangles",
        "Recovered Area (in²)", "Theoretical Area (in²)", "Recovery %",
    ]

    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for scrap_id, metrics in metrics_summary.items():
            writer.writerow({
                "Scrap Type": metrics["scrap_type"],
                "Scrap ID": scrap_id,
                "Original Area (in²)": f"{metrics['original_area_inches']:.2f}",
                "Triangle Size (in)": f"{metrics['triangle_size_inches']:.2f}",
                "Scale (px/in)": f"{metrics['scale']:.2f}",
                "Accepted Triangles": metrics["num_accepted_triangles"],
                "Recovered Area (in²)": f"{metrics['total_accepted_area_inches']:.2f}",
                "Theoretical Area (in²)": f"{metrics['theoretical_area_inches']:.2f}",
                "Recovery %": f"{metrics['recovery_percentage']:.2f}",
            })

        writer.writerow({})
        writer.writerow({
            "Scrap Type": "OVERALL",
            "Scrap ID": "",
            "Original Area (in²)": f"{totals['total_area_inches']:.2f}",
            "Triangle Size (in)": "",
            "Scale (px/in)": "",
            "Accepted Triangles": totals["total_triangles"],
            "Recovered Area (in²)": f"{totals['total_recovered_inches']:.2f}",
            "Theoretical Area (in²)": "",
            "Recovery %": f"{totals['overall_recovery']:.2f}",
        })

    print(f"CSV report saved to: {filename}")


def analyze_results(metrics_summary):

    if not metrics_summary:
        print("No data to analyze.")
        return

    best_scrap = max(metrics_summary.items(), key=lambda x: x[1]["recovery_percentage"])
    worst_scrap = min(metrics_summary.items(), key=lambda x: x[1]["recovery_percentage"])
    most_triangles = max(metrics_summary.items(), key=lambda x: x[1]["num_accepted_triangles"])

    print(f"\nBest recovery: {best_scrap[0]} "
          f"({best_scrap[1]['recovery_percentage']:.1f}%, {best_scrap[1]['num_accepted_triangles']} triangles)")
    print(f"Worst recovery: {worst_scrap[0]} "
          f"({worst_scrap[1]['recovery_percentage']:.1f}%, {worst_scrap[1]['num_accepted_triangles']} triangles)")
    print(f"Most triangles: {most_triangles[0]} "
          f"({most_triangles[1]['num_accepted_triangles']} triangles, {most_triangles[1]['recovery_percentage']:.1f}%)")

    type_stats = {}
    for metrics in metrics_summary.values():
        scrap_type = metrics["scrap_type"]
        stats = type_stats.setdefault(scrap_type, {
            "count": 0,
            "total_triangles": 0,
            "total_area_inches": 0,
            "total_recovered_inches": 0,
            "recoveries": [],
        })
        stats["count"] += 1
        stats["total_triangles"] += metrics["num_accepted_triangles"]
        stats["total_area_inches"] += metrics["original_area_inches"]
        stats["total_recovered_inches"] += metrics["total_accepted_area_inches"]
        stats["recoveries"].append(metrics["recovery_percentage"])

    for scrap_type, stats in sorted(type_stats.items()):
        avg_recovery = (
            (stats["total_recovered_inches"] / stats["total_area_inches"]) * 100
            if stats["total_area_inches"] > 0
            else 0
        )
        print(f"\n{scrap_type.upper()}:")
        print(f"  Images: {stats['count']}")
        print(f"  Total Triangles: {stats['total_triangles']}")
        print(f"  Total Area: {stats['total_area_inches']:.2f} in²")
        print(f"  Avg Recovery: {avg_recovery:.1f}%")
        print(f"  Min Recovery: {min(stats['recoveries']):.1f}%")
        print(f"  Max Recovery: {max(stats['recoveries']):.1f}%")