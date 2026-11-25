"""
Draw an Assignment as an image. Everything returns RGB numpy arrays or
bytes; nothing is written to disk.

Both patterns use orientation-aware pasting: the source triangle's right
angle is mapped onto the destination's right angle, so fabric is rotated
or mirrored but never sheared. (compose.py's pyramid used the plain
3-point warp, which sheared the fabric.)
"""

import csv
import io

import cv2
import numpy as np

from compose import _right_angle_order, paste_triangle_texture
from .inventory_service import resolve_path

BACKGROUND_RGB = (235, 235, 235)
MISSING_RGB = (255, 255, 255)
MISSING_OUTLINE_RGB = (210, 60, 60)
HIGHLIGHT_RGB = (220, 30, 30)
EMPTY_OUTLINE_RGB = (200, 200, 200)


def _bgr(rgb):
    r, g, b = rgb
    return int(b), int(g), int(r)


def canvas_size(rows, cols, block_side, leg_px):
    """(width, height) in pixels."""
    return int(round(cols * block_side * leg_px)), int(round(rows * block_side * leg_px))


def leg_px_to_fit(rows, cols, block_side, max_side_px):
    """Largest pixel size per leg so the quilt fits within max_side_px."""
    return max_side_px / (max(rows, cols) * block_side)


def _dest_px(placement, leg_px):
    return [(x * leg_px, y * leg_px) for x, y in placement.vertices]


def _paste_texture(canvas, placement, dest, image_cache):
    record = placement.record
    src = _right_angle_order(record["vertices_px"])
    others = [k for k in range(3) if k != placement.right_idx]
    if placement.flip:
        others.reverse()
    dst = [dest[placement.right_idx], dest[others[0]], dest[others[1]]]
    rec = dict(record)
    rec["vertices_px"] = [list(p) for p in src]
    rec["source_image"] = str(resolve_path(record["source_image"]))
    paste_triangle_texture(canvas, rec, dst, image_cache)


def render(assignment, leg_px, mode="flat", highlight_block=None, mark_missing=True,
           image_cache=None, progress=None, outline_empty=False):
    """
    Render the quilt.

    Args:
        leg_px: pixels per triangle leg (pinwheel block = 2 legs wide).
        mode: "flat" (average colour per triangle, fast) or "texture" (real fabric).
        highlight_block: block index to outline (step-by-step editing).
        outline_empty: draw a faint outline around blocks with no colours yet.
        mark_missing: outline positions whose colour ran out.
        image_cache: dict reused between calls to avoid re-reading source images.
        progress: optional callback(done, total).

    Returns:
        RGB uint8 array.
    """
    width, height = canvas_size(assignment.rows, assignment.cols, assignment.block_side, leg_px)
    canvas = np.full((height, width, 3), _bgr(BACKGROUND_RGB), dtype=np.uint8)
    keep_all = image_cache is not None
    cache = image_cache if keep_all else {}
    missing = []
    placements = assignment.placements
    if mode == "texture":
        placements = sorted(placements, key=lambda p: p.record["source_image"] if p.record else "")
    total = len(placements)

    for i, p in enumerate(placements):
        dest = _dest_px(p, leg_px)
        poly = np.round(np.array(dest)).astype(np.int32)
        if p.record is None:
            cv2.fillConvexPoly(canvas, poly, _bgr(MISSING_RGB))
            missing.append(poly)
        elif mode == "texture":
            if not keep_all and len(cache) > 1:
                current = str(resolve_path(p.record["source_image"]))
                for path in [k for k in cache if k != current]:
                    del cache[path]
            _paste_texture(canvas, p, dest, cache)
        else:
            cv2.fillConvexPoly(canvas, poly, _bgr(p.record["color"]))
        if progress and (i % 200 == 0 or i == total - 1):
            progress(i + 1, total)

    if mark_missing:
        for poly in missing:
            cv2.polylines(canvas, [poly], True, _bgr(MISSING_OUTLINE_RGB), 1, cv2.LINE_AA)

    if outline_empty:
        filled = {p.block for p in assignment.placements}
        side = assignment.block_side * leg_px
        for b in range(assignment.rows * assignment.cols):
            if b in filled:
                continue
            row, col = divmod(b, assignment.cols)
            x0, y0 = int(round(col * side)), int(round(row * side))
            x1, y1 = int(round((col + 1) * side)) - 1, int(round((row + 1) * side)) - 1
            cv2.rectangle(canvas, (x0, y0), (x1, y1), _bgr(EMPTY_OUTLINE_RGB), 1)

    if highlight_block is not None and 0 <= highlight_block < assignment.rows * assignment.cols:
        row, col = divmod(highlight_block, assignment.cols)
        side = assignment.block_side * leg_px
        x0, y0 = int(round(col * side)), int(round(row * side))
        x1, y1 = int(round((col + 1) * side)) - 1, int(round((row + 1) * side)) - 1
        thickness = max(2, int(side / 25))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), _bgr(HIGHLIGHT_RGB), thickness)

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def to_png_bytes(rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()



def _composite_on(image, background_rgb=(255, 255, 255)):
    """BGR(A) -> RGB composited on a solid background."""
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        bg = np.array(_bgr(background_rgb), dtype=np.float32)
        image = (image[:, :, :3] * alpha + bg * (1 - alpha)).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def thumbnail(path, max_side=160):
    """RGB thumbnail of a scrap image, transparent areas shown white."""
    image = cv2.imread(str(resolve_path(path)), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not load image: {path}")
    h, w = image.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        image = cv2.resize(image, (max(1, int(w * s)), max(1, int(h * s))),
                           interpolation=cv2.INTER_AREA)
    return _composite_on(image)


def triangle_crop(record, size=64, image_cache=None):
    """RGB image of one harvested triangle's real fabric on a white square."""
    cache = image_cache if image_cache is not None else {}
    path = str(resolve_path(record["source_image"]))
    if path not in cache:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not load image: {path}")
        cache[path] = img
    src = cache[path]
    pts = np.array(record["vertices_px"], dtype=np.float32)
    x, y, w, h = cv2.boundingRect(np.round(pts).astype(np.int32))
    x, y = max(x, 0), max(y, 0)
    crop = src[y:y + h, x:x + w]
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(pts - [x, y]).astype(np.int32), 255)
    out = np.full_like(crop, 255)
    out[mask > 0] = crop[mask > 0]
    out = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)



CUTTING_LIST_FIELDS = ["block", "block_row", "block_col", "slot", "color_category",
                       "triangle_size_in", "triangle_id", "scrap_id", "folder", "source_image",
                       "v1_x_in", "v1_y_in", "v2_x_in", "v2_y_in", "v3_x_in", "v3_y_in"]


def cutting_list_csv(assignment):
    """CSV text: which harvested triangle (and where on which scrap) goes in each position."""
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=CUTTING_LIST_FIELDS)
    writer.writeheader()
    for p in assignment.placements:
        row, col = divmod(p.block, assignment.cols)
        rec = p.record or {}
        verts = rec.get("vertices_in") or [["", ""]] * 3
        writer.writerow({
            "block": p.block + 1, "block_row": row + 1, "block_col": col + 1,
            "slot": p.slot, "color_category": p.category,
            "triangle_size_in": rec.get("triangle_size_inches", ""),
            "triangle_id": rec.get("triangle_id", "MISSING"),
            "scrap_id": rec.get("scrap_id", ""), "folder": rec.get("folder", ""),
            "source_image": rec.get("source_image", ""),
            "v1_x_in": _fmt(verts[0][0]), "v1_y_in": _fmt(verts[0][1]),
            "v2_x_in": _fmt(verts[1][0]), "v2_y_in": _fmt(verts[1][1]),
            "v3_x_in": _fmt(verts[2][0]), "v3_y_in": _fmt(verts[2][1]),
        })
    return out.getvalue()


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else v
