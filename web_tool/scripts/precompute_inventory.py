"""
Harvest every fabric folder once and save data/inventory_<size>in.json for the web app.

    python scripts/precompute_inventory.py                  # all folders, 1-inch triangles
    python scripts/precompute_inventory.py --size 2.5       # one other size
    python scripts/precompute_inventory.py --all-sizes      # every size the app offers (1-5 in)
    python scripts/precompute_inventory.py bag1 linen       # just these folders (merged into the existing file)

Writes only data/inventory_<size>in.json - no analysis images or CSV reports.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quilt_core.inventory_service import (MIN_ALPHA_COVERAGE, NUM_SHIFTS,
                                          TRIANGLE_SIZE_INCHES, TRIANGLE_SIZES, category_counts,
                                          harvest_folder, inventory_path, list_folders,
                                          load_inventory, save_inventory, sorted_categories)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="*", help="folders under scraps/ (default: all)")
    ap.add_argument("--size", type=float, default=TRIANGLE_SIZE_INCHES,
                    help="triangle leg length in inches (default 1.0)")
    ap.add_argument("--all-sizes", action="store_true",
                    help=f"harvest every size the app offers: {', '.join(f'{s:g}' for s in TRIANGLE_SIZES)} in")
    ap.add_argument("--out", help="output file (single size only; default data/inventory_<size>in.json)")
    args = ap.parse_args()

    folders = args.folders or list_folders()
    unknown = set(folders) - set(list_folders())
    if unknown:
        ap.error(f"unknown folder(s): {', '.join(sorted(unknown))}")

    sizes = TRIANGLE_SIZES if args.all_sizes else [args.size]
    if args.out and len(sizes) > 1:
        ap.error("--out only works with a single size")
    for size in sizes:
        harvest_size(folders, size, args.out or inventory_path(size), merge=bool(args.folders))


def harvest_size(folders, size, out, merge):
    print(f"== {size:g}-inch triangles")
    records, scraps = load_inventory(out) if merge else ([], {})
    records = [r for r in records if r["folder"] not in folders]
    scraps = {k: v for k, v in scraps.items() if v["folder"] not in folders}

    for folder in folders:
        t0 = time.time()
        recs, scr, problems = harvest_folder(folder, triangle_size_inches=size)
        records.extend(recs)
        scraps.update(scr)
        counts = category_counts(recs)
        summary = ", ".join(f"{c}={counts[c]}" for c in sorted_categories(counts))
        print(f"{folder}: {len(scr)} scraps, {len(recs)} triangles in {time.time() - t0:.1f}s  [{summary}]")
        for p in problems:
            print(f"  ! {p}")

    path = save_inventory(records, scraps, out, settings={
        "triangle_size_inches": size, "num_shifts": NUM_SHIFTS,
        "min_alpha_coverage": MIN_ALPHA_COVERAGE})
    print(f"Saved {len(records)} triangles from {len(scraps)} scraps to {path} "
          f"({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
