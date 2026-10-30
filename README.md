# NeXus Viewer

Fast Rust GUI to browse the content of a NeXus/HDF5 file: the file tree on the
left, values and plots on the right, and a search bar to narrow the tree down
to matching PVs.

## Usage

```bash
./launch_nexus_viewer.sh /SNS/VENUS/IPTS-38715/nexus/VENUS_26871.nxs.h5
```

The script rebuilds automatically when sources changed. Without a file
argument a file-open dialog appears; you can also drag & drop a file onto the
window or use the **📂 Open…** button. Passing several files (or dropping
several at once) opens them all, ready to compare.

## Features

- **Tree view** (left panel) of every group and dataset in the file, with
  array shapes shown next to dataset names.
- **Click a dataset** to display it on the right:
  - scalars and strings are shown as text (with `units` when present),
  - 1-D arrays are plotted (drag to pan, scroll to zoom, double-click to
    reset, right-drag for box zoom), with n/min/max/mean stats, and their
    values are listed in a table beside the plot (index, time when the
    PV has one, value; 📋 copies the whole table tab-separated),
  - 2-D arrays are shown as a false-colour image (viridis, NaN in grey)
    with a colour bar; hovering a pixel prints its column, row and value.
    Row 0 is at the top, as in an image. Sibling `x_edges`/`y_edges`
    datasets with uniform spacing (as in the strain maps) put the axes in
    those coordinates,
  - DASlogs-style PVs are plotted against their sibling `time` dataset,
  - clicking a PV *group* (e.g. `BL10:CHOP:TCERO:ActualSpeed`) directly shows
    its `value` dataset.
- **Export**: next to the dtype/shape line, **💾 CSV…** saves a 1-D array
  (index, time when the PV has one, value) as CSV, and **💾 TIFF…** saves
  a 2-D array as a 32-bit float grey TIFF (readable by ImageJ, tifffile…).
  When sibling groups hold a same-named 2-D array — the `strain` map of
  every tag `000`, `001`, … under `projections` — **💾 all N 'strain'
  under projections…** asks for a destination folder, creates a
  `projections` folder inside it and writes one `<tag>.tiff` per group.
- **Attributes** of the selected dataset are listed above the value.
- **Compare PVs**: Ctrl+click a second dataset (marked `[2]` in the
  tree), then choose
  - **1 vs 2** — one PV against the other (scatter). PVs logged at different
    times are paired by linearly interpolating the y PV onto the x PV's time
    grid; **swap axes** flips which one is x.
  - **both as y** — the two curves overlaid over time, with a legend and an
    optional *normalize [0–1]* checkbox for PVs of very different magnitudes.
  Ctrl+click more datasets (`[3]`, `[4]`, …) to overlay any number of PVs
  as y-curves. Ctrl+click an entry again (or its ✖ button) to drop it from
  the plot. The PVs may live in different files.
- **Recent PVs**: the last 20 PVs plotted are listed above the tree
  (persisted across restarts). Check any number of them to plot them again
  — the first checked one is the main PV, the others overlay it — and
  uncheck to drop them; PVs absent from the open files are greyed out.
- **Compare PVs across files**: open more files with **➕ Add…** (or
  Ctrl+click in the 🕘 Recent menu, or Ctrl+drop, or drop several files at
  once — the 📂/➕ dialogs also allow Ctrl/Shift multi-select). Each file
  gets its own collapsible section in the tree (✖ closes it, "✖ all" in the
  top bar closes everything). Clicking a PV then shows it for *every* open
  file: a scrollable table with the value per file (with a Δ column for
  scalars) and, for logged PVs, all the curves overlaid in one plot with an
  optional *normalize [0–1]* checkbox. A scalar PV across 3+ files is also
  plotted as value vs run number. The search bar filters all open files at
  once.
- **Open runs by number** with **🔢 Runs…**: type a run list like
  `26871-26970, 27012` and the matching NeXus files are opened. With no
  directory given, each run is located automatically by scanning every
  `/SNS/VENUS/IPTS-*/nexus` directory (the runs of one list may come from
  different IPTS experiments); give a directory to restrict the search to
  it. Works for 100+ files: they load in the background with a progress
  spinner and a cancel button.
- **Search bar** (top): type to narrow the tree to matching PVs only; the
  `Aa` toggle switches case-sensitive matching, `✖` clears. Matching names
  are highlighted and branches auto-expand.
- Huge arrays are min/max-decimated before plotting so the UI stays
  responsive without hiding spikes.
- **Light / dark theme** toggle (☀ / 🌙 button, top right). The preference is
  saved in `~/.config/venus_rust_tools/theme` and shared with the other VENUS
  rust tools (e.g. the ROI selector).

## Build requirements

- Rust toolchain (`cargo`), edition 2024.
- System HDF5 library and headers (`/usr/lib64/libhdf5.so`,
  `/usr/include/hdf5.h` — present on the analysis machines).

```bash
cargo build --release      # binary at target/release/nexus_viewer
cargo test --release       # reads the sample VENUS_26871.nxs.h5 run
```

## Log

Every session appends to `/SNS/VENUS/shared/log/nexus_viewer_<user>.log`:
the command line, each file opened or closed (with its group/dataset counts),
the PVs selected and compared, every export with its destination, and any
error (file that failed to open, failed export, crash). The file is readable
in the Log Browser (`rust_log_browser`) alongside the other VENUS tools' logs.

## Code layout

- `src/h5io.rs` — walks the HDF5 file into a node arena (tree), reads
  datasets/attributes with dtype-aware conversion (numeric → f64, fixed and
  variable-length strings), formats values.
- `src/main.rs` — eframe/egui application: search/filter logic, tree panel,
  value panel with egui_plot plotting, min/max decimation.
- `src/logger.rs` — append-only shared VENUS log (see *Log* above).
