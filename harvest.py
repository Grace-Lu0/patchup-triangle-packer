from itertools import product
import numpy as np
from shapely.geometry import Point, Polygon


def classify_orientation(triangle, corner_tolerance=1.0):

    coords = list(triangle.exterior.coords)[:-1]

    min_x = min(p[0] for p in coords)
    max_x = max(p[0] for p in coords)
    min_y = min(p[1] for p in coords)
    max_y = max(p[1] for p in coords)

    corners = {
        "tl": (min_x, min_y),
        "tr": (max_x, min_y),
        "bl": (min_x, max_y),
        "br": (max_x, max_y),
    }

    corner_in_triangle = [
        name
        for name, corner in corners.items()
        if triangle.contains(Point(corner)) or triangle.distance(Point(corner)) < corner_tolerance
    ]

    if "tl" in corner_in_triangle and "br" in corner_in_triangle:
        return "main"
    return "anti"


def extract_triangles_for_diagonal(
    vertices,
    triangle_size_inches,
    scale,
    shift_x=0,
    shift_y=0,
    diagonal="main",
):

    scrap = Polygon(vertices)
    min_x, min_y, max_x, max_y = scrap.bounds

    triangle_size_pixels = triangle_size_inches * scale

    start_x = min_x + shift_x
    start_y = min_y + shift_y

    xs = np.arange(start_x, max_x, triangle_size_pixels)
    ys = np.arange(start_y, max_y, triangle_size_pixels)

    accepted_triangles = []

    for x in xs:
        for y in ys:
            p1 = (x, y)
            p2 = (x + triangle_size_pixels, y)
            p3 = (x + triangle_size_pixels, y + triangle_size_pixels)
            p4 = (x, y + triangle_size_pixels)

            if diagonal == "main":
                tri_1 = Polygon([p1, p2, p3])
                tri_2 = Polygon([p1, p3, p4])
            else:
                tri_1 = Polygon([p2, p3, p4])
                tri_2 = Polygon([p1, p2, p4])

            if scrap.covers(tri_1):
                accepted_triangles.append(tri_1)
            if scrap.covers(tri_2):
                accepted_triangles.append(tri_2)

    return accepted_triangles


def _best_triangles_for_diagonal(vertices, triangle_size_inches, scale, shifts, diagonal):
    best = []
    for shift_x, shift_y in product(shifts, repeat=2):
        triangles = extract_triangles_for_diagonal(
            vertices, triangle_size_inches, scale, shift_x, shift_y, diagonal
        )
        if len(triangles) > len(best):
            best = triangles
    return best


def find_best_triangles(vertices, triangle_size_inches, scale, num_shifts=5, overlap_tolerance=0.01):

    scrap = Polygon(vertices)
    triangle_size_pixels = triangle_size_inches * scale
    shifts = np.linspace(0, triangle_size_pixels, num_shifts, endpoint=False)

    main_triangles = _best_triangles_for_diagonal(vertices, triangle_size_inches, scale, shifts, "main")
    anti_triangles = _best_triangles_for_diagonal(vertices, triangle_size_inches, scale, shifts, "anti")

    if len(main_triangles) >= len(anti_triangles):
        selected, candidates = main_triangles, anti_triangles
    else:
        selected, candidates = anti_triangles, main_triangles

    selected = list(selected)

    for tri in candidates:
        overlaps = any(
            tri.intersects(existing)
            and tri.intersection(existing).area > overlap_tolerance * tri.area
            for existing in selected
        )
        if not overlaps and scrap.covers(tri):
            selected.append(tri)

    return selected