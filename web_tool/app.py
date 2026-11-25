"""
Ralli quilt designer - Streamlit web tool.

    streamlit run app.py

Steps: 1 Fabrics -> 2 Pattern -> 3 Harvest & sort -> 4 Colours -> 5 Preview & export.
All quilt logic lives in quilt_core/; this file is UI only.
"""

import random

import altair as alt
import pandas as pd
import streamlit as st

from quilt_core.assign import assign_triangles
from quilt_core.inventory_service import (CATEGORY_EMOJI, TRIANGLE_SIZES, category_counts,
                                          category_mean_rgb, fabric_usage_by_folder,
                                          harvest_folder, inventory_path, list_folders,
                                          load_inventory, records_by_folder, scrap_paths_for_folder,
                                          sorted_categories)
from quilt_core.patterns import get_pattern
from quilt_core.plan import (QuiltPlan, auto_grid, block_usage, fill_random, overbooked,
                             random_block, remaining_counts, set_block)
from quilt_core.render import (canvas_size, cutting_list_csv, leg_px_to_fit, render, thumbnail,
                               to_png_bytes, triangle_crop)

st.set_page_config(page_title="Ralli Quilt Designer", layout="wide")

STEPS = ["1 · Fabrics", "2 · Pattern", "3 · Harvest & sort", "4 · Colours", "5 · Preview & export"]
PREVIEW_MAX_PX = 1400
EXPORT_PX = 3200
REUSE_FACTOR_WHEN_ALLOWED = 2.0
PERSISTENT_KEYS = ["pattern_key", "lines", "grid_mode", "manual_rows", "manual_cols",
                   "allow_reuse", "min_contrast", "color_mode", "triangle_size"]



@st.cache_resource(show_spinner="Loading triangle inventory…", max_entries=len(TRIANGLE_SIZES))
def saved_inventory(size):
    """(records_by_folder, scraps) from data/inventory_<size>in.json. Treat as read-only."""
    records, scraps = load_inventory(inventory_path(size))
    return records_by_folder(records), scraps


@st.cache_resource(show_spinner=False, max_entries=32)
def live_harvest(folder, size):
    """Harvest a folder that is not in the saved inventory (slow, cached per session)."""
    records, scraps, problems = harvest_folder(folder, triangle_size_inches=size)
    return records, scraps, problems


def folder_data(folder, size):
    """(records, scraps, problems) for one folder, from the saved inventory if present."""
    by_folder, scraps = saved_inventory(size)
    if folder in by_folder:
        return by_folder[folder], {k: v for k, v in scraps.items() if v["folder"] == folder}, []
    return live_harvest(folder, size)


def live_harvest_done(folder, size):
    return (folder, size) in st.session_state.get("live_harvested", set())


@st.cache_resource(show_spinner=False, max_entries=16)
def inventory_for(folders, size):
    """Combined (records, scraps, counts, mean_rgb) for a tuple of folders at one triangle size."""
    records, scraps = [], {}
    for f in folders:
        recs, scr, _ = folder_data(f, size)
        records.extend(recs)
        scraps.update(scr)
    return records, scraps, category_counts(records), category_mean_rgb(records)


@st.cache_data(show_spinner=False)
def folder_thumbnails(folder, n=6):
    paths = scrap_paths_for_folder(folder)
    step = max(1, len(paths) // n)
    return [thumbnail(p, 120) for p in paths[::step][:n]]


@st.cache_data(show_spinner=False)
def folder_scrap_count(folder):
    return len(scrap_paths_for_folder(folder))


@st.cache_data(show_spinner=False, max_entries=64)
def category_samples(folders, size, category, n=12):
    records = [r for r in inventory_for(folders, size)[0] if r["color_category"] == category]
    rng = random.Random(category)
    sample = rng.sample(records, min(n, len(records)))
    cache = {}
    return [triangle_crop(r, 72, cache) for r in sample]


@st.cache_data(show_spinner=False)
def pattern_example(pattern_key):
    """Small flat-colour illustration of a pattern (synthetic colours, no fabric)."""
    palette = {"terracotta": (196, 98, 64), "indigo": (46, 64, 120), "cream": (238, 226, 200),
               "mustard": (214, 162, 52)}
    records = [{"triangle_id": f"{c}{i}", "color_category": c, "color": list(rgb)}
               for c, rgb in palette.items() for i in range(400)]
    lines = 4
    plan = QuiltPlan(pattern_key, 2, 2, seed=3, lines=lines)
    fill_random(plan, {c: 400 for c in palette}, palette)
    pattern = plan.pattern
    return render(assign_triangles(plan, records), leg_px_to_fit(2, 2, pattern.block_side, 360))


def plan_key(plan, folders, size, allow_reuse):
    blocks = tuple(None if b is None else tuple(sorted(b.items())) for b in plan.blocks)
    return (tuple(folders), size, allow_reuse, plan.pattern_key, plan.lines, plan.rows, plan.cols,
            plan.seed, blocks)


@st.cache_resource(show_spinner=False, max_entries=8)
def cached_assignment(key):
    folders, size, allow_reuse, pattern_key, lines, rows, cols, seed, blocks = key
    plan = QuiltPlan(pattern_key, rows, cols, seed=seed, lines=lines,
                     blocks=[None if b is None else dict(b) for b in blocks])
    return assign_triangles(plan, inventory_for(folders, size)[0], allow_reuse=allow_reuse)


@st.cache_data(show_spinner=False, max_entries=32)
def flat_preview(key, leg_px, highlight, outline_empty=False):
    return render(cached_assignment(key), leg_px, mode="flat", highlight_block=highlight,
                  outline_empty=outline_empty)



def init_state():
    ss = st.session_state
    ss.setdefault("step", STEPS[0])
    ss.setdefault("pattern_key", "pinwheel")
    ss.setdefault("lines", 5)
    ss.setdefault("grid_mode", "Auto")
    ss.setdefault("manual_rows", 6)
    ss.setdefault("manual_cols", 9)
    ss.setdefault("allow_reuse", False)
    ss.setdefault("min_contrast", True)
    ss.setdefault("color_mode", "Random")
    ss.setdefault("triangle_size", 1.0)
    ss.setdefault("current_block", 0)
    ss.setdefault("widget_version", 0)
    ss.setdefault("live_harvested", set())
    ss.setdefault("harvest_confirmed", ())
    ss.setdefault("placed", set())
    for k in PERSISTENT_KEYS + [f"folder_{f}" for f in list_folders()]:
        if k in ss:
            ss[k] = ss[k]


def selected_folders():
    return tuple(f for f in list_folders() if st.session_state.get(f"folder_{f}", False))


def triangle_size():
    """Chosen triangle leg length in inches (one size per harvest)."""
    return float(st.session_state.triangle_size)


def size_label(size):
    return f"{size:g} in"


def is_harvested(folders):
    return st.session_state.harvest_confirmed == (folders, triangle_size())


def quilt_inches(plan):
    """(width, height) of the quilt top in inches, from the triangle leg length."""
    side = plan.pattern.block_side * triangle_size()
    return plan.cols * side, plan.rows * side


def go_to(step):
    st.session_state.step = step


def next_button(label, step, disabled=False):
    st.button(label, type="primary", on_click=go_to, args=(step,), disabled=disabled)


def bump_widgets():
    """Force step-by-step selectboxes to re-read the plan after an external change."""
    st.session_state.widget_version += 1


def current_grid(folders):
    ss = st.session_state
    pattern = get_pattern(ss.pattern_key, ss.lines)
    if ss.grid_mode == "Manual":
        return int(ss.manual_rows), int(ss.manual_cols)
    counts = inventory_for(folders, triangle_size())[2] if folders else {}
    reuse = REUSE_FACTOR_WHEN_ALLOWED if ss.allow_reuse else 1.0
    return auto_grid(counts, pattern, max_reuse=reuse)


def get_plan(folders):
    """The session's plan, rebuilt (empty) whenever fabrics, pattern or grid changed."""
    ss = st.session_state
    rows, cols = current_grid(folders)
    signature = (folders, triangle_size(), ss.pattern_key, ss.lines, rows, cols)
    if ss.get("plan_signature") != signature or "plan" not in ss:
        ss.plan = QuiltPlan(ss.pattern_key, rows, cols, seed=random.randrange(10**9), lines=ss.lines)
        ss.plan_signature = signature
        ss.current_block = 0
        ss.placed = set()
        ss.pop("export", None)
        bump_widgets()
    return ss.plan


def category_label(cat, left=None):
    emoji = CATEGORY_EMOJI.get(cat, "▫️")
    return f"{emoji} {cat}" if left is None else f"{emoji} {cat} · {left} left"


def rgb_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)



def step_fabrics():
    st.header("Choose fabric scraps")
    st.caption("Pick one or more folders of fabric, and their triangles are pooled together.")
    folders = list_folders()
    if not folders:
        st.error("No folders found under scraps/.")
        return
    size = triangle_size()
    by_folder, _ = saved_inventory(size)
    for folder in folders:
        with st.container(border=True):
            left, right = st.columns([1, 4])
            with left:
                st.checkbox(f"**{folder}**", key=f"folder_{folder}")
                n_tri = len(by_folder[folder]) if folder in by_folder else None
                st.caption(f"{folder_scrap_count(folder)} scraps"
                           + (f" · {n_tri:,} triangles at {size_label(size)}" if n_tri is not None
                              else f" · not harvested at {size_label(size)} yet"))
            with right:
                thumbs = folder_thumbnails(folder)
                cols = st.columns(len(thumbs) or 1)
                for c, img in zip(cols, thumbs):
                    c.image(img, width="stretch")
    chosen = selected_folders()
    st.write(f"Selected: **{', '.join(chosen) or 'none'}**")
    next_button("Next: choose a pattern →", STEPS[1], disabled=not chosen)


def step_pattern():
    ss = st.session_state
    st.header("Choose a Ralli-inspired pattern")
    cols = st.columns(2)
    for col, key in zip(cols, ["pyramid", "pinwheel"]):
        pattern = get_pattern(key)
        with col, st.container(border=True):
            st.subheader(pattern.label)
            st.image(pattern_example(key), width="stretch")
            st.caption(pattern.description)
            if ss.pattern_key == key:
                st.success("Selected")
            else:
                st.button(f"Use {pattern.label}", key=f"use_{key}",
                          on_click=lambda k=key: ss.update(pattern_key=k))

    with st.expander("Advanced settings"):
        if ss.pattern_key == "pyramid":
            st.slider("Rows per quadrant", 2, 8, key="lines",
                      help="How many rows each quadrant grows. One square needs 4 × rows² triangles.")
        st.checkbox("Allow reusing triangles", key="allow_reuse",
                    help="Off: every harvested triangle is used at most once, like a real quilt. "
                         "On: a design can be bigger than your fabric allows.")
        st.radio("Quilt size", ["Auto", "Manual"], key="grid_mode", horizontal=True,
                 help="Auto: the largest quilt your selected fabric can fill.")
        if ss.grid_mode == "Manual":
            c1, c2 = st.columns(2)
            c1.number_input("Rows of blocks", 1, 60, key="manual_rows")
            c2.number_input("Columns of blocks", 1, 60, key="manual_cols")

    pattern = get_pattern(ss.pattern_key, ss.lines)
    st.info(f"**{pattern.label}**: {pattern.triangles_per_block} triangles per block. "
            "Harvest in the next step to see how big a quilt your fabric makes.")
    next_button("Next: harvest triangles →", STEPS[2])


def step_harvest():
    ss = st.session_state
    folders = selected_folders()
    st.header("Harvest triangles and sort by colour")
    st.select_slider("Triangle size (length of the two short sides)", TRIANGLE_SIZES,
                     key="triangle_size", format_func=size_label,
                     help="Every triangle in one harvest is the same size. Bigger triangles "
                          "make a bigger quilt from fewer pieces, but small scraps may not fit any.")
    size = triangle_size()
    st.caption(f"Each scrap is cut into as many {size_label(size)} right triangles as fit, then "
               "each triangle is sorted by its average colour.")

    pending = [f for f in folders
               if f not in saved_inventory(size)[0] and not live_harvest_done(f, size)]
    if not is_harvested(folders):
        if pending:
            st.warning(f"Not saved at {size_label(size)}, will be harvested now (≈10–30 s each): "
                       f"{', '.join(pending)}")
        if st.button("Harvest triangles", type="primary"):
            for f in pending:
                with st.spinner(f"Harvesting {f} at {size_label(size)}…"):
                    _, _, problems = live_harvest(f, size)
                ss.live_harvested.add((f, size))
                for p in problems:
                    st.warning(f"{f}: {p}")
            ss.harvest_confirmed = (folders, size)
            st.rerun()
        return

    records, scraps, counts, mean_rgb = inventory_for(folders, size)
    if not records:
        st.error(f"No {size_label(size)} triangles fit in the selected scraps. "
                 "Try a smaller triangle size or add another folder.")
        return

    rows, cols = current_grid(folders)
    pattern = get_pattern(ss.pattern_key, ss.lines)
    side_in = pattern.block_side * size
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scraps", f"{len(scraps):,}")
    c2.metric("Triangles harvested", f"{len(records):,}")
    c3.metric(f"{pattern.label} quilt", f"{rows} × {cols} blocks",
              help=f"Needs {rows * cols * pattern.triangles_per_block:,} triangles")
    c4.metric("Quilt top", f"≈ {cols * side_in:.0f} × {rows * side_in:.0f} in",
              help=f"Each block is {side_in:.1f} in square. Before seam allowances.")

    cats = sorted_categories(counts)
    if len(cats) < 2:
        st.warning("Only one colour was found, so the pattern will not show. "
                   "Add another fabric folder for contrast.")

    df = pd.DataFrame({"colour": cats, "triangles": [counts[c] for c in cats],
                       "hex": [rgb_hex(mean_rgb[c]) for c in cats]})
    chart = alt.Chart(df).mark_bar(stroke="#888", strokeWidth=0.5).encode(
        x=alt.X("colour:N", sort=cats, title=None),
        y=alt.Y("triangles:Q", title="Triangles"),
        color=alt.Color("hex:N", scale=None),
        tooltip=["colour", "triangles"])
    st.altair_chart(chart, width="stretch")

    st.subheader("What each colour looks like")
    st.caption("Each triangle's colour is the average of its fabric, so patterned fabric "
               "may not look like it belongs in its group.")
    for cat in cats:
        with st.expander(f"{category_label(cat)} · {counts[cat]:,} triangles"):
            st.image(category_samples(folders, size, cat), width=72)

    next_button("Next: choose colours →", STEPS[3])


def step_colours():
    ss = st.session_state
    folders = selected_folders()
    if not is_harvested(folders):
        st.info("Harvest your selected fabrics at the chosen triangle size first (step 3).")
        return
    records, _, counts, mean_rgb = inventory_for(folders, triangle_size())
    plan = get_plan(folders)
    pattern = plan.pattern
    contrast = 60 if ss.min_contrast else 0

    st.header("Choose colours")
    mode = st.radio("How should colours be chosen?", ["Random", "Step by step"],
                    key="color_mode", horizontal=True)
    entered = ss.get("last_color_mode") != mode
    ss.last_color_mode = mode

    if mode == "Random":
        if entered and not plan.is_complete():
            fill_random(plan, counts, mean_rgb, ss.allow_reuse, contrast, only_empty=True)
        c1, c2, _ = st.columns([1, 2, 3])
        if c1.button("Shuffle", type="primary") or plan.n_filled() == 0:
            fill_random(plan, counts, mean_rgb, ss.allow_reuse, contrast,
                        seed=random.randrange(10**9))
            ss.placed = set()
            bump_widgets()
        c2.checkbox("Keep each pair of colours contrasting", key="min_contrast")
        show_budget(plan, counts)
        preview_plan, highlight = plan, None
    else:
        if entered:
            plan.blocks = [b if i in ss.placed else None for i, b in enumerate(plan.blocks)]
            ss.current_block = next_empty_block(plan, 0)
            bump_widgets()
        preview_plan, highlight = step_by_step_editor(plan, pattern, counts, mean_rgb, contrast)

    key = plan_key(preview_plan, folders, triangle_size(), ss.allow_reuse)
    leg = leg_px_to_fit(plan.rows, plan.cols, pattern.block_side, PREVIEW_MAX_PX)
    w_in, h_in = quilt_inches(plan)
    caption = (f"Flat-colour preview · {plan.rows} × {plan.cols} blocks · "
               f"≈ {w_in:.0f} × {h_in:.0f} in with {size_label(triangle_size())} triangles")
    if highlight is not None:
        caption += f" · block {highlight + 1} outlined in red"
    st.image(flat_preview(key, leg, highlight, mode != "Random"), width="stretch", caption=caption)
    next_button("Next: preview & export →", STEPS[4], disabled=plan.n_filled() == 0)


def next_empty_block(plan, start):
    """First empty block at or after `start` (wrapping round), or `start` if all are full."""
    n = plan.n_blocks
    for offset in range(n):
        i = (start + offset) % n
        if plan.blocks[i] is None:
            return i
    return min(start, n - 1)


def default_block(plan, idx, counts, mean_rgb, contrast):
    """Starting colours for an empty block: repeat the previous block if fabric allows,
    otherwise a suggestion from the colours with the most triangles left."""
    ss = st.session_state
    pattern = plan.pattern
    left = remaining_counts(plan, counts)
    previous = plan.blocks[idx - 1] if idx > 0 else None
    if previous is not None and (ss.allow_reuse or all(
            left.get(c, 0) >= n for c, n in block_usage(pattern, previous).items())):
        return dict(previous)
    rng = random.Random(f"{plan.seed}-suggest-{idx}")
    cats = sorted(c for c, v in counts.items() if v > 0)
    return random_block(pattern, cats, dict(left), mean_rgb, rng, ss.allow_reuse, contrast)


def copy_to_empty(plan, block, counts, allow_reuse):
    """Copy `block` into empty blocks in order, stopping when fabric runs out. Returns count."""
    pattern = plan.pattern
    need = block_usage(pattern, block)
    left = remaining_counts(plan, counts)
    copied = 0
    for i in range(plan.n_blocks):
        if plan.blocks[i] is not None:
            continue
        if not allow_reuse and any(left.get(c, 0) < n for c, n in need.items()):
            break
        set_block(plan, i, block)
        st.session_state.placed.add(i)
        for c, n in need.items():
            left[c] = left.get(c, 0) - n
        copied += 1
    return copied


def step_by_step_editor(plan, pattern, counts, mean_rgb, contrast):
    """Build the quilt one block at a time. Returns (plan to preview, block to highlight)."""
    ss = st.session_state
    n = plan.n_blocks
    ss.current_block = min(ss.current_block, n - 1)
    idx = ss.current_block
    placed_here = plan.blocks[idx] is not None

    with st.container(border=True):
        row, col = plan.block_position(idx)
        st.markdown(f"**Block {idx + 1} of {n}** (row {row + 1}, column {col + 1}) · "
                    f"{plan.n_filled()} of {n} blocks placed"
                    + (" · this block is placed" if placed_here else ""))

        block = plan.blocks[idx] or default_block(plan, idx, counts, mean_rgb, contrast)
        left = remaining_counts(plan, counts, exclude=idx)
        options = sorted_categories(counts)
        version = ss.widget_version
        groups = list(pattern.slot_groups)

        same = False
        if len(groups) > 1:
            a0, b0 = groups[0]
            default_same = all(block[a] == block[a0] and block[b] == block[b0] for a, b in groups)
            same = st.checkbox("Same colours for all four quadrants", value=default_same,
                               key=f"same_quadrants_v{version}_{idx}")
            if same:
                groups = [groups[0]]

        pending = dict(block)
        for slot_a, slot_b in groups:
            c1, c2 = st.columns(2)
            for col_widget, slot in ((c1, slot_a), (c2, slot_b)):
                label = ("Rows" if same and slot.endswith("_row") else
                         "Gaps" if same and slot.endswith("_gap") else
                         pattern.slot_labels.get(slot, slot))
                need = pattern.slot_sizes[slot] * (4 if same else 1)
                pending[slot] = col_widget.selectbox(
                    f"{label} ({need} triangles)", options,
                    index=options.index(block[slot]) if block[slot] in options else 0,
                    format_func=lambda c: category_label(c, left.get(c, 0)),
                    key=f"slot_v{version}_{idx}_{slot}")
        if same:
            a0, b0 = pattern.slot_groups[0]
            for a, b in pattern.slot_groups:
                pending[a], pending[b] = pending[a0], pending[b0]

        short = {} if ss.allow_reuse else {
            c: n - left.get(c, 0) for c, n in block_usage(pattern, pending).items()
            if left.get(c, 0) < n}
        if short:
            st.warning("Not enough fabric for this block: " + ", ".join(
                f"{category_label(c)} needs {n} more" for c, n in sorted(short.items()))
                + ". Pick another colour.")

        b = st.columns([2, 1, 1, 2, 2, 1])
        if b[0].button("Update block" if placed_here else "Place block", type="primary",
                       disabled=bool(short)):
            set_block(plan, idx, pending)
            ss.placed.add(idx)
            ss.current_block = next_empty_block(plan, idx)
            bump_widgets()
            st.rerun()
        if b[1].button("Prev", disabled=idx == 0):
            ss.current_block -= 1
            bump_widgets()
            st.rerun()
        if b[2].button("Next", disabled=idx >= n - 1):
            ss.current_block += 1
            bump_widgets()
            st.rerun()
        if b[3].button("Copy to empty blocks", disabled=bool(short),
                       help="Repeat this block's colours in every empty block, until a colour runs out."):
            copied = copy_to_empty(plan, pending, counts, ss.allow_reuse)
            ss.current_block = next_empty_block(plan, idx)
            bump_widgets()
            st.toast(f"Copied to {copied} empty blocks.")
            st.rerun()
        if b[4].button("Fill the rest randomly"):
            fill_random(plan, counts, mean_rgb, ss.allow_reuse, contrast, only_empty=True)
            bump_widgets()
            st.rerun()
        if b[5].button("Clear all"):
            plan.blocks = [None] * n
            ss.placed = set()
            ss.current_block = 0
            bump_widgets()
            st.rerun()
        if placed_here and st.button("Remove this block"):
            plan.blocks[idx] = None
            ss.placed.discard(idx)
            bump_widgets()
            st.rerun()

    preview = QuiltPlan(plan.pattern_key, plan.rows, plan.cols, seed=plan.seed, lines=plan.lines,
                        blocks=list(plan.blocks))
    preview.blocks[idx] = pending
    return preview, idx


def show_budget(plan, counts):
    ss = st.session_state
    if ss.allow_reuse:
        return
    over = overbooked(plan, counts)
    if over:
        detail = ", ".join(f"{category_label(c)} short by {n}" for c, n in sorted(over.items()))
        st.warning(f"Not enough fabric for this design: {detail}. Those triangles will be left "
                   "empty (outlined in red). Pick other colours, or allow reuse in step 2.")


def step_export():
    ss = st.session_state
    folders = selected_folders()
    if not is_harvested(folders) or "plan" not in ss or ss.plan.n_filled() == 0:
        st.info("Choose colours first (step 4).")
        return
    size = triangle_size()
    plan = get_plan(folders)
    records, scraps, counts, _ = inventory_for(folders, size)
    key = plan_key(plan, folders, size, ss.allow_reuse)
    assignment = cached_assignment(key)
    pattern = plan.pattern

    st.header("Preview & export")
    if not plan.is_complete():
        st.warning(f"{plan.n_blocks - plan.n_filled()} blocks have no colours yet and will be blank.")
    if assignment.n_missing:
        st.warning(f"{assignment.n_missing} triangle positions are empty because a colour ran out.")

    leg = leg_px_to_fit(plan.rows, plan.cols, pattern.block_side, EXPORT_PX)
    w, h = canvas_size(plan.rows, plan.cols, pattern.block_side, leg)
    w_in, h_in = quilt_inches(plan)
    st.caption(f"Image {w:,} × {h:,} px · {len(assignment.placements):,} triangles of "
               f"{size_label(size)} · quilt top ≈ {w_in:.0f} × {h_in:.0f} in before seam allowances")

    export = ss.get("export")
    if export and export["key"] != key:
        export = None
    if st.button("Render with real fabric", type="primary"):
        bar = st.progress(0.0, text="Cutting and placing fabric…")
        img = render(assignment, leg, mode="texture",
                     progress=lambda d, t: bar.progress(d / t, text=f"Placed {d:,} of {t:,} triangles"))
        bar.empty()
        export = ss.export = {"key": key, "png": to_png_bytes(img), "image": img}

    if export:
        st.image(export["image"], width="stretch", caption="Real fabric render")
        name = f"ralli_{plan.pattern_key}_{size:g}in_{'_'.join(folders)}"
        c1, c2 = st.columns(2)
        c1.download_button("Download quilt PNG", export["png"], f"{name}.png", "image/png",
                           type="primary")
        c2.download_button("Download cutting list (CSV)", cutting_list_csv(assignment),
                           f"{name}_cutting_list.csv", "text/csv",
                           help="Which harvested triangle, from which scrap, goes in each position.")
    else:
        st.image(flat_preview(key, leg_px_to_fit(plan.rows, plan.cols, pattern.block_side,
                                                 PREVIEW_MAX_PX), None),
                 width="stretch", caption="Flat-colour preview. Render to see real fabric.")

    st.subheader("Fabric usage")
    usage = fabric_usage_by_folder(records, scraps, assignment.used_ids)
    st.dataframe(pd.DataFrame([{
        "Fabric": f, "Triangles used": f"{u['used_triangles']:,} / {u['total_triangles']:,}",
        "Fabric used (in²)": round(u["used_area_in2"], 1),
        "Total scrap area (in²)": round(u["total_scrap_area_in2"], 1),
        "Used %": round(u["reused_percentage"], 1)} for f, u in usage.items()]),
        hide_index=True, width="stretch")
    if ss.allow_reuse and assignment.max_reuse > 1:
        st.caption(f"Reuse is on: some triangles appear up to {assignment.max_reuse} times.")



def main():
    init_state()
    with st.sidebar:
        st.title("Ralli Quilt Designer")
        st.radio("Steps", STEPS, key="step", label_visibility="collapsed")
        folders = selected_folders()
        st.divider()
        st.caption(f"Fabrics: {', '.join(folders) or 'none'}")
        st.caption(f"Pattern: {get_pattern(st.session_state.pattern_key).label}")
        st.caption(f"Triangle size: {size_label(triangle_size())}")

    step = st.session_state.step
    if step != STEPS[0] and not folders:
        st.info("Choose at least one fabric folder first.")
        st.button("Go to step 1", on_click=go_to, args=(STEPS[0],))
        return
    {STEPS[0]: step_fabrics, STEPS[1]: step_pattern, STEPS[2]: step_harvest,
     STEPS[3]: step_colours, STEPS[4]: step_export}[step]()


main()
