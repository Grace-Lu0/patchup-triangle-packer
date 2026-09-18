import warnings
import numpy as np
from shapely.geometry import Point

class RightTriangleShape:
    name = 'right'
    colors = {'main': 'blue', 'anti': 'orange', 'mixed': 'green', 'unknown': 'purple'}

    @staticmethod
    def shift_domain(size_px):
        return (size_px, size_px)

    @staticmethod
    def generate(min_x, min_y, max_x, max_y, size_px, shift_x, shift_y, diagonal='main'):
        start_x = min_x + shift_x
        start_y = min_y + shift_y
        xs = np.arange(start_x, max_x, size_px)
        y_values = np.arange(start_y, max_y, size_px)
        triangles = []
        for x in xs:
            for y in y_values:
                p1 = (x, y)
                p2 = (x + size_px, y)
                p3 = (x + size_px, y + size_px)
                p4 = (x, y + size_px)
                if diagonal in ('main', 'both'):
                    triangles.append([p1, p2, p3])
                    triangles.append([p1, p3, p4])
                if diagonal in ('anti', 'both'):
                    triangles.append([p2, p3, p4])
                    triangles.append([p1, p2, p4])
        return triangles

    @staticmethod
    def classify(triangle, corner_tolerance=1.0):
        coordinates = list(triangle.exterior.coords)[:-1]
        if len(coordinates) != 3:
            return 'unknown'
        min_x = min((p[0] for p in coordinates))
        max_x = max((p[0] for p in coordinates))
        min_y = min((p[1] for p in coordinates))
        max_y = max((p[1] for p in coordinates))
        corners = {'tl': (min_x, min_y), 'tr': (max_x, min_y), 'bl': (min_x, max_y), 'br': (max_x, max_y)}
        corner_in_triangle = [name for name, corner in corners.items() if triangle.contains(Point(corner)) or triangle.distance(Point(corner)) < corner_tolerance]
        if 'tl' in corner_in_triangle and 'br' in corner_in_triangle:
            return 'main'
        if 'tr' in corner_in_triangle and 'bl' in corner_in_triangle:
            return 'anti'
        warnings.warn(f"Triangle failed to classify as main/anti (corners matched: {corner_in_triangle}) -- this shouldn't happen for grid-generated triangles; check corner_tolerance", RuntimeWarning)
        return 'mixed' if corner_in_triangle else 'unknown'

    @staticmethod
    def area(size):
        return 0.5 * size * size

class EquilateralTriangleShape:
    name = 'equilateral'
    colors = {'upward': 'blue', 'downward': 'orange'}

    @staticmethod
    def shift_domain(size_px):
        triangle_height = size_px * np.sqrt(3) / 2
        return (size_px / 2, triangle_height)

    @staticmethod
    def generate(min_x, min_y, max_x, max_y, size_px, shift_x, shift_y):
        triangle_height = size_px * np.sqrt(3) / 2
        start_x = min_x + shift_x
        start_y = min_y + shift_y
        point_set = set()
        grid_points = []
        row = 0
        y = start_y
        while y <= max_y:
            x_offset = size_px / 2 if row % 2 else 0
            x = start_x + x_offset
            row_points = []
            while x <= max_x:
                p = (x, y)
                grid_points.append(p)
                row_points.append(p)
                x += size_px
            point_set.update(row_points)
            y += triangle_height
            row += 1
        triangles = []
        for x, y in grid_points:
            bottom_left = (x, y)
            bottom_right = (x + size_px, y)
            top_right = (x + size_px / 2, y + triangle_height)
            top_left = (x - size_px / 2, y + triangle_height)
            if bottom_right in point_set and top_right in point_set:
                triangles.append([bottom_left, bottom_right, top_right])
            if top_right in point_set and top_left in point_set:
                triangles.append([bottom_left, top_right, top_left])
        return triangles

    @staticmethod
    def classify(triangle):
        coordinates = list(triangle.exterior.coords)[:-1]
        centroid_y = triangle.centroid.y
        y_values = [c[1] for c in coordinates]
        min_y, max_y = (min(y_values), max(y_values))
        return 'upward' if centroid_y - min_y < max_y - centroid_y else 'downward'

    @staticmethod
    def area(size):
        return np.sqrt(3) / 4 * size * size

class HalfEquilateralTriangleShape:
    name = 'half_equilateral'
    colors = {'upward': 'blue', 'downward': 'orange'}

    @staticmethod
    def shift_domain(size_px):
        return (size_px, size_px * np.sqrt(3))

    @staticmethod
    def generate(min_x, min_y, max_x, max_y, size_px, shift_x, shift_y):
        cell_width = size_px
        cell_height = size_px * np.sqrt(3)
        start_x = min_x + shift_x
        start_y = min_y + shift_y
        triangles = []
        x = start_x
        while x <= max_x:
            y = start_y
            while y <= max_y:
                top_left = (x, y)
                top_right = (x + cell_width, y)
                bottom_left = (x, y + cell_height)
                bottom_right = (x + cell_width, y + cell_height)
                triangles.append([top_left, top_right, bottom_right])
                triangles.append([top_left, bottom_right, bottom_left])
                y += cell_height
            x += cell_width
        return triangles

    @staticmethod
    def classify(triangle):
        coordinates = list(triangle.exterior.coords)[:-1]
        centroid_y = triangle.centroid.y
        y_values = [c[1] for c in coordinates]
        min_y, max_y = (min(y_values), max(y_values))
        return 'upward' if centroid_y - min_y < max_y - centroid_y else 'downward'

    @staticmethod
    def area(size):
        return np.sqrt(3) / 2 * size * size
SHAPES = {RightTriangleShape.name: RightTriangleShape, EquilateralTriangleShape.name: EquilateralTriangleShape, HalfEquilateralTriangleShape.name: HalfEquilateralTriangleShape}
