from itertools import product
import numpy as np
from shapely.geometry import Polygon

def isosceles_leg_for_area(area):
    return np.sqrt(2 * area)

def equilateral_side_for_area(area):
    return np.sqrt(4 * area / np.sqrt(3))

def side306090_for_area(area):
    return np.sqrt(2 * area / np.sqrt(3))

def _isosceles_triangles_for_diagonal(vertices, leg, shift_x, shift_y, diagonal):
    scrap = Polygon(vertices)
    min_x, min_y, max_x, max_y = scrap.bounds
    xs = np.arange(min_x + shift_x, max_x, leg)
    ys = np.arange(min_y + shift_y, max_y, leg)
    accepted_triangles = []
    for x in xs:
        for y in ys:
            p1, p2, p3, p4 = ((x, y), (x + leg, y), (x + leg, y + leg), (x, y + leg))
            if diagonal == 'main':
                first_triangle, second_triangle = (Polygon([p1, p2, p3]), Polygon([p1, p3, p4]))
            else:
                first_triangle, second_triangle = (Polygon([p2, p3, p4]), Polygon([p1, p2, p4]))
            if scrap.covers(first_triangle):
                accepted_triangles.append(first_triangle)
            if scrap.covers(second_triangle):
                accepted_triangles.append(second_triangle)
    return accepted_triangles

def _best_isosceles_for_diagonal(vertices, leg, shifts, diagonal):
    best_triangles = []
    for shift_x, shift_y in product(shifts, repeat=2):
        triangles = _isosceles_triangles_for_diagonal(vertices, leg, shift_x, shift_y, diagonal)
        if len(triangles) > len(best_triangles):
            best_triangles = triangles
    return best_triangles

def harvest_isosceles(vertices, triangle_area, num_shifts=5, overlap_tolerance=0.01):
    leg = isosceles_leg_for_area(triangle_area)
    scrap = Polygon(vertices)
    shifts = np.linspace(0, leg, num_shifts, endpoint=False)
    main_triangles = _best_isosceles_for_diagonal(vertices, leg, shifts, 'main')
    anti_triangles = _best_isosceles_for_diagonal(vertices, leg, shifts, 'anti')
    if len(main_triangles) >= len(anti_triangles):
        selected_triangles, candidate_triangles = (list(main_triangles), anti_triangles)
    else:
        selected_triangles, candidate_triangles = (list(anti_triangles), main_triangles)
    for tri in candidate_triangles:
        has_overlap = any((tri.intersects(existing) and tri.intersection(existing).area > overlap_tolerance * tri.area for existing in selected_triangles))
        if not has_overlap and scrap.covers(tri):
            selected_triangles.append(tri)
    return selected_triangles

def _306090_triangles_for_diagonal(vertices, short_leg, shift_x, shift_y, diagonal):
    scrap = Polygon(vertices)
    min_x, min_y, max_x, max_y = scrap.bounds
    cell_width, cell_height = (short_leg, short_leg * np.sqrt(3))
    xs = np.arange(min_x + shift_x, max_x, cell_width)
    ys = np.arange(min_y + shift_y, max_y, cell_height)
    accepted_triangles = []
    for x in xs:
        for y in ys:
            top_left, top_right, bottom_left, bottom_right = ((x, y), (x + cell_width, y), (x, y + cell_height), (x + cell_width, y + cell_height))
            if diagonal == 'main':
                first_triangle, second_triangle = (Polygon([top_left, top_right, bottom_right]), Polygon([top_left, bottom_right, bottom_left]))
            else:
                first_triangle, second_triangle = (Polygon([top_left, top_right, bottom_left]), Polygon([top_right, bottom_right, bottom_left]))
            if scrap.covers(first_triangle):
                accepted_triangles.append(first_triangle)
            if scrap.covers(second_triangle):
                accepted_triangles.append(second_triangle)
    return accepted_triangles

def _best_306090_for_diagonal(vertices, short_leg, shifts, diagonal):
    best_triangles = []
    for shift_x, shift_y in product(shifts, repeat=2):
        triangles = _306090_triangles_for_diagonal(vertices, short_leg, shift_x, shift_y, diagonal)
        if len(triangles) > len(best_triangles):
            best_triangles = triangles
    return best_triangles

def harvest_306090(vertices, triangle_area, num_shifts=5, overlap_tolerance=0.01):
    short_leg = side306090_for_area(triangle_area)
    scrap = Polygon(vertices)
    shifts = np.linspace(0, short_leg, num_shifts, endpoint=False)
    main_triangles = _best_306090_for_diagonal(vertices, short_leg, shifts, 'main')
    anti_triangles = _best_306090_for_diagonal(vertices, short_leg, shifts, 'anti')
    if len(main_triangles) >= len(anti_triangles):
        selected_triangles, candidate_triangles = (list(main_triangles), anti_triangles)
    else:
        selected_triangles, candidate_triangles = (list(anti_triangles), main_triangles)
    for tri in candidate_triangles:
        has_overlap = any((tri.intersects(existing) and tri.intersection(existing).area > overlap_tolerance * tri.area for existing in selected_triangles))
        if not has_overlap and scrap.covers(tri):
            selected_triangles.append(tri)
    return selected_triangles

def _equilateral_grid_triangles(vertices, side, shift_x, shift_y):
    scrap = Polygon(vertices)
    min_x, min_y, max_x, max_y = scrap.bounds
    triangle_height = side * np.sqrt(3) / 2
    start_x, start_y = (min_x + shift_x, min_y + shift_y)
    accepted_triangles = []
    row = 0
    y = start_y
    while y <= max_y:
        x_offset = side / 2 if row % 2 else 0
        x = start_x + x_offset
        while x <= max_x:
            bottom_left, bottom_right = ((x, y), (x + side, y))
            top_right, top_left = ((x + side / 2, y + triangle_height), (x - side / 2, y + triangle_height))
            upward_triangle = Polygon([bottom_left, bottom_right, top_right])
            if scrap.covers(upward_triangle):
                accepted_triangles.append(upward_triangle)
            downward_triangle = Polygon([bottom_left, top_right, top_left])
            if scrap.covers(downward_triangle):
                accepted_triangles.append(downward_triangle)
            x += side
        y += triangle_height
        row += 1
    return accepted_triangles

def harvest_equilateral(vertices, triangle_area, num_shifts=5):
    side = equilateral_side_for_area(triangle_area)
    triangle_height = side * np.sqrt(3) / 2
    shifts_x = np.linspace(0, side / 2, num_shifts, endpoint=False)
    shifts_y = np.linspace(0, triangle_height, num_shifts, endpoint=False)
    best_triangles = []
    for shift_x, shift_y in product(shifts_x, shifts_y):
        triangles = _equilateral_grid_triangles(vertices, side, shift_x, shift_y)
        if len(triangles) > len(best_triangles):
            best_triangles = triangles
    return best_triangles
