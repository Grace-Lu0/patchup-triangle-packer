"""
Load or build the triangle inventory for the web tool.

Harvesting reuses the existing pipeline (main.get_scrap_groups,
report.process_single_scrap, inventory.build_triangle_records) but never
calls visualize_scrap and never writes analysis images or reports.

Two fixes over the CLI pipeline:
  - Triangles whose area is not almost entirely opaque fabric in the source
    PNG are dropped (approxPolyDP can put the outline slightly outside the
    real fabric edge, letting transparent pixels leak into the colour
    average and the pasted texture as dark slivers).
  - source_image paths are stored relative to the repo root, so a saved
    inventory works from any working directory or machine.
"""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import REPO_ROOT, WEB_TOOL_DIR
from inventory import build_triangle_records
from main import get_scrap_groups
from polygons import get_calibration_scale
from report import process_single_scrap

SCRAPS_DIR = REPO_ROOT / "scraps"
DATA_DIR = WEB_TOOL_DIR / "data"

TRIANGLE_SIZE_INCHES = 1.0
TRIANGLE_SIZES = [1.0 + 0.5 * i for i in range(9)]
NUM_SHIFTS = 5
MIN_ALPHA_COVERAGE = 0.98
OPAQUE_ALPHA = 128

CATEGORY_ORDER = ["red", "orange", "yellow", "green", "blue", "purple",
                  "pink", "brown", "black", "grey", "white"]
CATEGORY_EMOJI = {"red": "🟥", "orange": "🟧", "yellow": "🟨", "green": "🟩",
                  "blue": "🟦", "purple": "🟪", "pink": "🩷", "brown": "🟫",
                  "black": "⬛", "grey": "🩶", "white": "⬜"}


def inventory_path(triangle_size_inches=TRIANGLE_SIZE_INCHES):
    """Saved inventory for one triangle size, e.g. data/inventory_1.5in.json."""
    return DATA_DIR / f"inventory_{float(triangle_size_inches):.1f}in.json"


DEFAULT_INVENTORY_PATH = inventory_path(TRIANGLE_SIZE_INCHES)


def resolve_path(path):
    """Resolve a stored (repo-relative) image path to an absolute path."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _relative_to_repo(path):
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)



def list_folders(scraps_dir=SCRAPS_DIR):
    """Fabric folders under scraps/ (each is one bag / fabric type)."""
    scraps_dir = Path(scraps_dir)
    if not scraps_dir.is_dir():
        return []
    return sorted(p.name for p in scraps_dir.iterdir() if p.is_dir())


def scrap_groups_for_folder(folder, scraps_dir=SCRAPS_DIR):
    """{(folder, base_name): group} for one folder, as main.get_scrap_groups builds them."""
    groups = get_scrap_groups(str(Path(scraps_dir) / folder))
    return {k: v for k, v in groups.items() if v["folder"] == folder}


def scrap_paths_for_folder(folder, scraps_dir=SCRAPS_DIR):
    """All non-calibration scrap image paths in a folder, sorted."""
    paths = []
    for group in scrap_groups_for_folder(folder, scraps_dir).values():
        paths.extend(group["scraps"])
    return sorted(paths)



def alpha_coverage(alpha, vertices_px):
    """Fraction of the triangle's pixels that are opaque in the alpha channel."""
    mask = np.zeros(alpha.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(np.asarray(vertices_px)).astype(np.int32)], 255)
    inside = mask > 0
    n = int(inside.sum())
    if n == 0:
        return 0.0
    return float((alpha[inside] >= OPAQUE_ALPHA).sum()) / n


def _filter_opaque(records, image_path, min_coverage):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim < 3 or image.shape[2] < 4:
        return records
    alpha = image[:, :, 3]
    return [r for r in records if alpha_coverage(alpha, r["vertices_px"]) >= min_coverage]


def harvest_folder(folder, scraps_dir=SCRAPS_DIR, triangle_size_inches=TRIANGLE_SIZE_INCHES,
                   num_shifts=NUM_SHIFTS, min_coverage=MIN_ALPHA_COVERAGE, progress=None):
    """
    Harvest every scrap in one folder into triangle records.

    Args:
        progress: optional callback(done, total, message) for UI progress bars.

    Returns:
        (records, scraps, problems):
          records  - triangle records
          scraps   - {scrap_id: {"folder", "area_in2", "triangles"}} for EVERY scrap
                     processed, including ones that yielded no triangles (so fabric
                     usage percentages count all the fabric, not just harvested area)
          problems - "file: reason" strings for images that could not be processed
    """
    groups = scrap_groups_for_folder(folder, scraps_dir)
    total = sum(len(g["scraps"]) for g in groups.values())
    done = 0
    records, scraps, problems = [], {}, []

    for (_, base_name), group in sorted(groups.items()):
        try:
            scale, _ = get_calibration_scale(group["calib"])
        except Exception as e:
            problems.append(f"{os.path.basename(group['calib'])}: calibration failed ({e})")
            done += len(group["scraps"])
            continue

        for scrap_path in sorted(group["scraps"]):
            scrap_id = f"{folder}_{os.path.splitext(os.path.basename(scrap_path))[0]}"
            try:
                result = process_single_scrap(scrap_path, scale, triangle_size_inches, num_shifts)
                recs = build_triangle_records(
                    scrap_id, folder, scrap_path, result["triangles"], scale,
                    triangle_size_inches, result["metrics"]["original_area_inches"])
                recs = _filter_opaque(recs, scrap_path, min_coverage)
                for r in recs:
                    r["source_image"] = _relative_to_repo(r["source_image"])
                    r["vertices_px"] = [[float(x), float(y)] for x, y in r["vertices_px"]]
                    r["vertices_in"] = [[float(x), float(y)] for x, y in r["vertices_in"]]
                records.extend(recs)
                scraps[scrap_id] = {"folder": folder,
                                    "area_in2": float(result["metrics"]["original_area_inches"]),
                                    "triangles": len(recs)}
            except Exception as e:
                problems.append(f"{os.path.basename(scrap_path)}: {e}")
            done += 1
            if progress:
                progress(done, total, f"{folder}: {os.path.basename(scrap_path)}")

    return records, scraps, problems



def save_inventory(records, scraps, path=DEFAULT_INVENTORY_PATH, settings=None):
    """Write the inventory as compact JSON (explicitly writes a file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "settings": settings or {}, "scraps": scraps, "records": records}
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    return path


def load_inventory(path=DEFAULT_INVENTORY_PATH):
    """Load a saved inventory as (records, scraps); ([], {}) if the file does not exist."""
    path = Path(path)
    if not path.exists():
        return [], {}
    with open(path) as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload, {}
    return payload["records"], payload.get("scraps", {})


def records_by_folder(records):
    out = defaultdict(list)
    for r in records:
        out[r["folder"]].append(r)
    return dict(out)



def total_scrap_area_by_folder(scraps):
    out = defaultdict(float)
    for s in scraps.values():
        out[s["folder"]] += s["area_in2"]
    return dict(out)


def fabric_usage_by_folder(records, scraps, used_ids):
    """{folder: report.compute_fabric_usage stats}, counting ALL scrap area in each folder."""
    from report import compute_fabric_usage
    areas = total_scrap_area_by_folder(scraps)
    return {folder: compute_fabric_usage(recs, used_ids, total_scrap_area_in2=areas.get(folder))
            for folder, recs in sorted(records_by_folder(records).items())}


def category_counts(records):
    return dict(Counter(r["color_category"] for r in records))


def category_mean_rgb(records):
    sums = defaultdict(lambda: np.zeros(3))
    counts = Counter()
    for r in records:
        sums[r["color_category"]] += r["color"]
        counts[r["color_category"]] += 1
    return {c: tuple(int(round(v)) for v in sums[c] / counts[c]) for c in counts}


def sorted_categories(categories):
    """Categories in rainbow display order; unknown names go last alphabetically."""
    known = [c for c in CATEGORY_ORDER if c in categories]
    return known + sorted(c for c in categories if c not in CATEGORY_ORDER)
