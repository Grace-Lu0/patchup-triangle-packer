import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.patches import Polygon as MPoly

from harvest import classify_orientation


MAIN_DIAGONAL_COLOR = "blue"
ANTI_DIAGONAL_COLOR = "orange"

_COLOR_BY_ORIENTATION = {
    "main": MAIN_DIAGONAL_COLOR,
    "anti": ANTI_DIAGONAL_COLOR,
}


def visualize_scrap(
    vertices,
    accepted_triangles,
    metrics,
    scrap_id,
    scrap_type,
    show_plot=False,
    output_path=None,
    show_stats_box=False,
):
    figure, axis = plt.subplots(figsize=(12, 10))

    scrap_patch = MPoly(
        vertices,
        closed=True,
        fill=True,
        facecolor="lightgray",
        alpha=0.3,
        edgecolor="black",
        linewidth=2,
        label="Scrap Outline",
    )
    axis.add_patch(scrap_patch)

    for triangle in accepted_triangles:
        orientation = classify_orientation(triangle)
        color = _COLOR_BY_ORIENTATION[orientation]
        triangle_vertices = list(triangle.exterior.coords)[:-1]

        triangle_patch = MPoly(
            triangle_vertices,
            closed=True,
            fill=True,
            facecolor=color,
            alpha=0.3,
            edgecolor=color,
            linewidth=1,
        )
        axis.add_patch(triangle_patch)

    axis.set_aspect("equal")
    axis.autoscale()
    axis.grid(True, alpha=0.3, linestyle="--")
    axis.set_xlabel("X coordinate (pixels)", fontsize=12)
    axis.set_ylabel("Y coordinate (pixels)", fontsize=12)

    title = (
        f"{scrap_type} - {scrap_id}\n"
        f"{metrics['num_accepted_triangles']} triangles\n"
        f"Recovery: {metrics['recovery_percentage']:.1f}%"
    )
    axis.set_title(title, fontsize=14, fontweight="bold")

    legend_elements = [
        Patch(
            facecolor="lightgray",
            alpha=0.3,
            edgecolor="black",
            linewidth=2,
            label="Scrap Outline",
        ),
        Patch(
            facecolor=MAIN_DIAGONAL_COLOR,
            alpha=0.3,
            edgecolor=MAIN_DIAGONAL_COLOR,
            linewidth=1,
            label="Main Diagonal (\\)",
        ),
        Patch(
            facecolor=ANTI_DIAGONAL_COLOR,
            alpha=0.3,
            edgecolor=ANTI_DIAGONAL_COLOR,
            linewidth=1,
            label="Anti-Diagonal (/)",
        ),
    ]
    axis.legend(handles=legend_elements, loc="upper right", fontsize=10)

    if show_stats_box:
        stats_text = (
            f'Scrap Area: {metrics["original_area_inches"]:.2f} in²\n'
            f'Recovered: {metrics["total_accepted_area_inches"]:.2f} in²\n'
            f'Triangles: {metrics["num_accepted_triangles"]}\n'
            f'Recovery: {metrics["recovery_percentage"]:.1f}%\n'
            f'Scale: {metrics["scale"]:.2f} px/in\n'
            f'Triangle: {metrics["triangle_size_inches"]}" half-square'
        )

        box_style = dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.9,
        )

        axis.text(
            0.02,
            0.98,
            stats_text,
            transform=axis.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=box_style,
        )

    plt.tight_layout()

    if output_path is None:
        output_path = f"{scrap_type}_{scrap_id}_analysis.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    if show_plot:
        plt.show()

    plt.close(figure)

    return output_path