import cv2
from shapely.geometry import Polygon

def extract_polygon_from_image(image_path, epsilon_ratio=0.005):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f'Could not load image: {image_path}')
    if image.shape[2] < 4:
        raise ValueError('Image must have an alpha channel (RGBA)')
    alpha = image[:, :, 3]
    contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError('No contours found in image')
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = epsilon_ratio * cv2.arcLength(largest_contour, True)
    simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    return [(point[0][0], point[0][1]) for point in simplified_contour]

def get_calibration_scale(calib_image_path, verbose=False):
    vertices = extract_polygon_from_image(calib_image_path)
    calibration_polygon = Polygon(vertices)
    min_x, min_y, max_x, max_y = calibration_polygon.bounds
    width = max_x - min_x
    height = max_y - min_y
    pixel_size = max(width, height)
    scale = pixel_size / 1.0
    if verbose:
        print(f'Calibration square: {width:.2f} x {height:.2f} px, using larger side = {pixel_size:.2f} px')
        print(f'Scale: {scale:.2f} px/inch')
    return (scale, vertices)
