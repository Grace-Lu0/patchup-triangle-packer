import csv

import matplotlib.pyplot as plt


INPUT_CSV = "sweep_report.csv"
OUTPUT_IMAGE = "triangle_size.png"


def load_sweep_results(filename):
    results = []

    with open(filename, newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            results.append(
                {
                    "size": float(row["Triangle Size (in)"]),
                    "recovery": float(row["Recovery %"]),
                    "triangles": int(row["Total Triangles"]),
                }
            )

    return results


def make_figure(results, output_path=OUTPUT_IMAGE):
    sizes = [row["size"] for row in results]
    recoveries = [row["recovery"] for row in results]
    triangle_counts = [row["triangles"] for row in results]

    figure, (recovery_axis, count_axis) = plt.subplots(
        1,
        2,
        figsize=(14, 6),
    )

    recovery_axis.plot(
        sizes,
        recoveries,
        color="blue",
        marker="o",
        markersize=7,
        linewidth=2,
    )

    recovery_axis.set_title(
        "Triangle Size vs Recovery Percentage",
        fontsize=14,
        fontweight="bold",
    )

    recovery_axis.set_xlabel("Triangle Size (inches)")
    recovery_axis.set_ylabel("Recovery Percentage (%)")
    recovery_axis.grid(True, alpha=0.3)

    for triangle_size, recovery in zip(sizes, recoveries):
        recovery_axis.annotate(
            f"{recovery:.1f}%",
            (triangle_size, recovery),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )

    count_axis.plot(
        sizes,
        triangle_counts,
        color="red",
        marker="s",
        markersize=7,
        linewidth=2,
    )

    count_axis.set_title(
        "Triangle Size vs Triangle Count",
        fontsize=14,
        fontweight="bold",
    )

    count_axis.set_xlabel("Triangle Size (inches)")
    count_axis.set_ylabel("Total Triangles")
    count_axis.grid(True, alpha=0.3)

    for triangle_size, count in zip(sizes, triangle_counts):
        count_axis.annotate(
            f"{count}",
            (triangle_size, count),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )

    figure.suptitle(
        "Triangle Size Trade-off Analysis",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {output_path}")


def main():
    results = load_sweep_results(INPUT_CSV)

    if not results:
        print("No results found in sweep_report.csv")
        return

    results.sort(key=lambda row: row["size"])

    print(f"Loaded {len(results)} triangle sizes.")

    for row in results:
        print(
            f"  {row['size']:.1f} in: "
            f"{row['recovery']:.1f}% recovery, "
            f"{row['triangles']} triangles"
        )

    make_figure(results)


if __name__ == "__main__":
    main()