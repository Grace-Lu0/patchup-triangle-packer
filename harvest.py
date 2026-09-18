from itertools import product
import numpy as np
from shapely.geometry import Point, Polygon

def classify_orientation(triangle, corner_tolerance=1.0):
    coordinates = list(triangle.exterior.coords)[:-1]
    min_x = min((p[0] for p in coordinates))
    max_x = max((p[0] for p in coordinates))
    min_y = min((p[1] for p in coordinates))
    max_y = max((p[1] for p in coordinates))
    corners = {'tl': (min_x, min_y), 'tr': (max_x, min_y), 'bl': (min_x, max_y), 'br': (max_x, max_y)}
    corner_in_triangle = [name for name, corner in corners.items() if triangle.contains(Point(corner)) or triangle.distance(Point(corner)) < corner_tolerance]
    if 'tl' in corner_in_triangle and 'br' in corner_in_triangle:
        return 'main'
    return 'anti'

def extract_triangles_for_diagonal(vertices, triangle_size_inches, scale, shift_x=0, shift_y=0, diagonal='main'):
    scrap = Polygon(vertices)
    min_x, min_y, max_x, max_y = scrap.bounds
    triangle_size_pixels = triangle_size_inches * scale
    start_x = min_x + shift_x
    start_y = min_y + shift_y
    x_positions = np.arange(start_x, max_x, triangle_size_pixels)
    y_positions = np.arange(start_y, max_y, triangle_size_pixels)
    accepted_triangles = []
    for x in x_positions:
        for y in y_positions:
            p1 = (x, y)
            p2 = (x + triangle_size_pixels, y)
            p3 = (x + triangle_size_pixels, y + triangle_size_pixels)
            p4 = (x, y + triangle_size_pixels)
            if diagonal == 'main':
                first_triangle = Polygon([p1, p2, p3])
                second_triangle = Polygon([p1, p3, p4])
            else:
                first_triangle = Polygon([p2, p3, p4])
                second_triangle = Polygon([p1, p2, p4])
            if scrap.covers(first_triangle):
                accepted_triangles.append(first_triangle)
            if scrap.covers(second_triangle):
                accepted_triangles.append(second_triangle)
    return accepted_triangles

def _best_triangles_for_diagonal(vertices, triangle_size_inches, scale, shifts, diagonal):
    best_triangles = []
    for shift_x, shift_y in product(shifts, repeat=2):
        triangles = extract_triangles_for_diagonal(vertices, triangle_size_inches, scale, shift_x, shift_y, diagonal)
        if len(triangles) > len(best_triangles):
            best_triangles = triangles
    return best_triangles

def find_best_triangles(vertices, triangle_size_inches, scale, num_shifts=5, overlap_tolerance=0.01):
    scrap = Polygon(vertices)
    triangle_size_pixels = triangle_size_inches * scale
    shifts = np.linspace(0, triangle_size_pixels, num_shifts, endpoint=False)
    main_triangles = _best_triangles_for_diagonal(vertices, triangle_size_inches, scale, shifts, 'main')
    anti_triangles = _best_triangles_for_diagonal(vertices, triangle_size_inches, scale, shifts, 'anti')
    if len(main_triangles) >= len(anti_triangles):
        selected_triangles, candidate_triangles = (main_triangles, anti_triangles)
    else:
        selected_triangles, candidate_triangles = (anti_triangles, main_triangles)
    selected_triangles = list(selected_triangles)
    for tri in candidate_triangles:
        has_overlap = any((tri.intersects(existing) and tri.intersection(existing).area > overlap_tolerance * tri.area for existing in selected_triangles))
        if not has_overlap and scrap.covers(tri):
            selected_triangles.append(tri)
    return selected_triangles
