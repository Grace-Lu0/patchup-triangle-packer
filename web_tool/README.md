# Irregular scraps: Ralli quilt designer

Harvests 1-inch right triangles from photos of irregular fabric scraps, sorts
them by colour, and arranges them into Ralli-inspired quilt designs.

## Web tool

Everything for the web tool lives in this `web_tool/` folder. It reuses the
harvesting pipeline in the repo root (`main.py`, `compose.py`, …) and the
photos in `scraps/`. Run every command from inside `web_tool/`: Streamlit
only reads `.streamlit/config.toml` (light mode) from the folder it is
started in.

```bash
cd web_tool
pip install -r requirements.txt
python scripts/precompute_inventory.py --all-sizes   # only if scraps change: rewrites data/inventory_*in.json
streamlit run app.py
```

1. **Fabrics**: pick one or more folders from `scraps/`.
2. **Pattern**: Pyramid or Pinwheel (advanced: grid size, triangle reuse, pyramid rows).
3. **Harvest & sort**: choose the triangle size (1–5 in, in 0.5 in steps; one size per
   harvest), then see triangle counts per colour with real fabric samples.
4. **Colours**: *Random* (shuffle) or *Step by step* (choose each block's
   colours, with "fill the rest randomly" and "apply this block to all").
5. **Preview & export**: render with real fabric, then download the PNG and a
   cutting list CSV (which triangle from which scrap goes where).

Each triangle size has its own saved harvest, `data/inventory_<size>in.json`.
A folder or size that has not been saved is harvested live in step 3.
Re-harvest one size, or specific folders: `python scripts/precompute_inventory.py --size 2.5 bag1 linen`.

## Layout

- `quilt_core/`: web tool logic (patterns, plan, triangle assignment, rendering). Writes no files.
- `app.py`: Streamlit UI only.
- `../main.py`, `../compose.py`, …: the original command-line pipeline in the repo root (unchanged).
- `tests/`: `pip install -r requirements-dev.txt && pytest`.
