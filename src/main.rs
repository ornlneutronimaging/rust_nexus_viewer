//! NeXus file viewer: tree of the file content on the left, values/plots on
//! the right, with a search bar to narrow the tree down to matching PVs.
//! Several files can be open at once to compare the same PV across runs.

mod h5io;
mod recent;
mod theme;
mod zoom;

use eframe::egui;
use egui::{Color32, RichText};
use egui_plot::{Legend, Line, Plot, PlotPoints, Points};
use h5io::{format_num, Tree, Value};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() -> eframe::Result {
    let arg_paths: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1300.0, 850.0])
            .with_title("NeXus Viewer"),
        ..Default::default()
    };
    eframe::run_native(
        "NeXus Viewer",
        options,
        Box::new(move |cc| {
            // Saved light/dark preference, shared by all the VENUS rust tools.
            cc.egui_ctx.set_theme(theme::load());
            cc.egui_ctx.set_zoom_factor(zoom::load());
            let mut app = App::default();
            app.recent = recent::load();
            app.recent_pvs = recent::load_pvs();
            app.enqueue_opens(arg_paths, false);
            Ok(Box::new(app))
        }),
    )
}

/// A dataset selection: (open-file index, node index in that file's tree).
type Sel = (usize, usize);

/// One open NeXus file plus its per-file search-match state.
struct OpenFile {
    tree: Tree,
    /// Canonicalized path, used to avoid opening the same file twice.
    canon: PathBuf,
    node_match: Vec<bool>,
    subtree_match: Vec<bool>,
    match_count: usize,
}

/// Everything derived from a selected dataset.
struct Loaded {
    file: usize,
    node: usize,
    value: Value,
    attrs: Vec<(String, String)>,
    /// Name used in plot labels: the PV (group) name for a `value` dataset.
    label: String,
    units: Option<String>,
    /// Full-resolution 1-D numeric data (empty when not plottable).
    y: Vec<f64>,
    /// Full-resolution sibling `time` axis, when one exists.
    x: Option<Vec<f64>>,
    /// Decimated (x, y) points ready to plot, for 1-D numeric data.
    points: Vec<[f64; 2]>,
    x_label: String,
    stats: Option<Stats>,
}

impl Loaded {
    fn plottable(&self) -> bool {
        self.y.len() > 1
    }

    fn sel(&self) -> Sel {
        (self.file, self.node)
    }
}

#[derive(Clone, Copy, PartialEq, Default)]
enum CompareMode {
    /// One PV against the other (paired via time interpolation or by index).
    #[default]
    Xy,
    /// Both PVs as y-curves over their own x axes.
    Overlay,
}

/// Cached x-y pairing of the two selected PVs (interpolation is too slow to
/// redo every frame).
struct XyCache {
    key: (Sel, Sel, bool),
    points: Vec<[f64; 2]>,
    n_pairs: usize,
    /// How the pairs were built, or why they could not be.
    paired_by: String,
    x_label: String,
    y_label: String,
}

struct Stats {
    n: usize,
    min: f64,
    max: f64,
    mean: f64,
}

#[derive(Default)]
struct App {
    files: Vec<OpenFile>,
    error: Option<String>,
    search: String,
    case_sensitive: bool,
    /// (query, case_sensitive, generation) the match arrays were computed for.
    computed_for: Option<(String, bool, u64)>,
    /// Bumped whenever the set of open files changes.
    generation: u64,
    total_matches: usize,
    selected: Option<Sel>,
    loaded: Option<Loaded>,
    /// Further selections (Ctrl+click, or the recent-PV checkboxes) for
    /// compare plots: one pairs with the main PV as x-y or overlays it,
    /// several all overlay it.
    others: Vec<Loaded>,
    /// The selected PV loaded from every open file (index-aligned with
    /// `files`; None where the file has no dataset at that path).
    multi: Vec<Option<Loaded>>,
    compare_mode: CompareMode,
    swap_xy: bool,
    normalize: bool,
    xy_cache: Option<XyCache>,
    /// Most-recently opened files, newest first (persisted across restarts).
    recent: Vec<PathBuf>,
    /// Dataset paths of the most-recently plotted PVs, newest first
    /// (persisted across restarts).
    recent_pvs: Vec<String>,
    /// "Open path" popup: visible, typed text, focus request, last error.
    path_popup: bool,
    path_input: String,
    path_focus: bool,
    path_error: Option<String>,
    /// Files queued for opening: bulk opens (run ranges, multi-select, many
    /// dropped files) load a few per frame so the UI stays alive.
    pending: VecDeque<PathBuf>,
    pending_total: usize,
    /// "Open runs" popup: visible, directory, run list, focus request.
    range_popup: bool,
    range_dir: String,
    range_input: String,
    range_focus: bool,
    /// (dir, run list) the match list below was computed for.
    range_key: Option<(String, String)>,
    range_matches: Vec<PathBuf>,
    range_error: Option<String>,
    /// Extra info about the matches ("from 3 IPTS; 2 runs not found").
    range_note: Option<String>,
    /// Run → NeXus file index over every IPTS of the instrument, used when
    /// the runs popup is asked to locate runs without a directory.
    run_index: Option<RunIndex>,
    /// Pending background scan building that index.
    index_rx: Option<std::sync::mpsc::Receiver<RunIndex>>,
    /// Width each toolbar group (files, search, status) took last frame;
    /// decides how many rows the toolbar needs this frame.
    top_widths: [f32; 3],
}

/// Every run number found under `root`/IPTS-*/nexus, mapped to its file.
struct RunIndex {
    root: PathBuf,
    by_run: std::collections::BTreeMap<u64, PathBuf>,
}

impl App {
    /// Open a file: `add` keeps the already-open files (for cross-file
    /// compare), otherwise they are replaced.
    fn open_file(&mut self, path: &Path, ctx: &egui::Context, add: bool) {
        if self.open_file_inner(path, ctx, add) {
            recent::add(&mut self.recent, path);
            self.rebuild_multi();
        }
    }

    /// Load and append one file; returns whether it loaded. Leaves the recent
    /// list and the cross-file cache alone (bulk opens update those once).
    fn open_file_inner(&mut self, path: &Path, ctx: &egui::Context, add: bool) -> bool {
        let canon = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        if add && self.files.iter().any(|f| f.canon == canon) {
            return false; // already open
        }
        match h5io::load(path) {
            Ok(tree) => {
                if !add {
                    self.files.clear();
                    self.selected = None;
                    self.loaded = None;
                    self.others.clear();
                    self.multi.clear();
                }
                self.files.push(OpenFile {
                    tree,
                    canon,
                    node_match: Vec::new(),
                    subtree_match: Vec::new(),
                    match_count: 0,
                });
                self.error = None;
                self.generation += 1;
                self.computed_for = None;
                self.xy_cache = None;
                self.update_title(ctx);
                true
            }
            Err(e) => {
                self.error = Some(format!("{e:#}"));
                false
            }
        }
    }

    /// Queue files for opening (loaded a few per frame, with progress shown
    /// in the top bar). `add` keeps the already-open files.
    fn enqueue_opens(&mut self, paths: Vec<PathBuf>, add: bool) {
        if paths.is_empty() {
            return;
        }
        if !add {
            self.files.clear();
            self.selected = None;
            self.loaded = None;
            self.others.clear();
            self.multi.clear();
            self.xy_cache = None;
            self.generation += 1;
            self.computed_for = None;
        }
        // Only the first file lands in the recent list — a 100-run range must
        // not wipe the hand-picked entries.
        recent::add(&mut self.recent, &paths[0]);
        self.pending.extend(paths);
        self.pending_total = self.pending.len();
    }

    /// Open queued files for up to ~100 ms per frame.
    fn process_pending(&mut self, ctx: &egui::Context) {
        if self.pending.is_empty() {
            return;
        }
        let start = Instant::now();
        while let Some(p) = self.pending.pop_front() {
            self.open_file_inner(&p, ctx, true);
            if start.elapsed().as_millis() > 100 {
                break;
            }
        }
        if self.pending.is_empty() {
            self.pending_total = 0;
            self.rebuild_multi();
        }
        ctx.request_repaint();
    }

    fn cancel_pending(&mut self) {
        self.pending.clear();
        self.pending_total = 0;
        self.rebuild_multi();
    }

    fn close_all(&mut self, ctx: &egui::Context) {
        self.files.clear();
        self.pending.clear();
        self.pending_total = 0;
        self.selected = None;
        self.loaded = None;
        self.others.clear();
        self.multi.clear();
        self.xy_cache = None;
        self.generation += 1;
        self.computed_for = None;
        self.update_title(ctx);
    }

    /// Directory to pre-fill popups and dialogs with: the first open file's,
    /// else the most recent one's.
    fn default_dir(&self) -> Option<&Path> {
        self.files
            .first()
            .map(|f| f.tree.file_path.as_path())
            .or_else(|| self.recent.first().map(|p| p.as_path()))
            .and_then(Path::parent)
    }

    fn close_file(&mut self, f: usize, ctx: &egui::Context) {
        self.files.remove(f);
        self.generation += 1;
        self.computed_for = None;
        self.xy_cache = None;
        fn fix(sel: &mut Option<Sel>, loaded: &mut Option<Loaded>, f: usize) {
            if let Some((sf, _)) = sel {
                if *sf == f {
                    *sel = None;
                    *loaded = None;
                } else if *sf > f {
                    *sf -= 1;
                    if let Some(l) = loaded {
                        l.file -= 1;
                    }
                }
            }
        }
        fix(&mut self.selected, &mut self.loaded, f);
        self.others.retain(|l| l.file != f);
        for l in &mut self.others {
            if l.file > f {
                l.file -= 1;
            }
        }
        self.rebuild_multi();
        self.update_title(ctx);
    }

    fn update_title(&self, ctx: &egui::Context) {
        let title = match self.files.len() {
            0 => "NeXus Viewer".to_string(),
            1 => {
                let name = self.files[0]
                    .tree
                    .file_path
                    .file_name()
                    .map(|s| s.to_string_lossy())
                    .unwrap_or_default();
                format!("NeXus Viewer — {name}")
            }
            n => format!("NeXus Viewer — {n} files"),
        };
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(title));
    }

    /// Short name shown in the tree, tables and legends: the file name minus
    /// the .nxs.h5 extensions (e.g. "VENUS_26871").
    fn file_label(&self, f: usize) -> String {
        let p = &self.files[f].tree.file_path;
        let name = p
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| p.display().to_string());
        name.trim_end_matches(".h5").trim_end_matches(".nxs").to_string()
    }

    fn open_dialog(&mut self, start: Option<&Path>, add: bool) {
        let mut dialog = rfd::FileDialog::new()
            .add_filter("NeXus / HDF5", &["h5", "nxs", "hdf5", "nx5", "nxs.h5"])
            .add_filter("All files", &["*"]);
        // Start where asked, else in the current file's directory, or the
        // last opened one.
        if let Some(dir) = start.or_else(|| self.default_dir()) {
            dialog = dialog.set_directory(dir);
        }
        // Multi-select: Ctrl/Shift+click picks several files at once.
        if let Some(paths) = dialog.pick_files() {
            self.enqueue_opens(paths, add);
        }
    }

    /// Open what is typed in the path popup: a file directly, a directory as
    /// the starting point of the browse dialog.
    fn open_typed_path(&mut self, ctx: &egui::Context, add: bool) {
        let mut typed = self.path_input.trim().to_owned();
        if let Some(rest) = typed.strip_prefix("~/") {
            if let Ok(home) = std::env::var("HOME") {
                typed = format!("{home}/{rest}");
            }
        }
        if typed.is_empty() {
            return;
        }
        let path = PathBuf::from(&typed);
        if path.is_file() {
            self.path_popup = false;
            self.path_error = None;
            self.open_file(&path, ctx, add);
        } else if path.is_dir() {
            self.path_popup = false;
            self.path_error = None;
            self.open_dialog(Some(&path), add);
        } else {
            self.path_error = Some(format!("Not found: {}", path.display()));
        }
    }

    /// Modal with a text field to type/paste a path instead of browsing.
    fn show_path_popup(&mut self, ctx: &egui::Context) {
        if !self.path_popup {
            return;
        }
        let modal = egui::Modal::new(egui::Id::new("open_path")).show(ctx, |ui| {
            ui.set_width(620.0);
            ui.heading("Open path");
            ui.label("File path opens it directly; directory path starts the browser there.");
            ui.add_space(6.0);
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.path_input)
                    .hint_text("/SNS/VENUS/IPTS-xxxxx/nexus/…")
                    .desired_width(f32::INFINITY),
            );
            if self.path_focus {
                resp.request_focus();
                self.path_focus = false;
            }
            if resp.changed() {
                self.path_error = None;
            }
            if let Some(err) = &self.path_error {
                ui.colored_label(ui.visuals().error_fg_color, err);
            }
            ui.add_space(6.0);
            let entered = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            ui.horizontal(|ui| {
                if ui.button("Open").clicked() || entered {
                    self.open_typed_path(ctx, false);
                }
                if ui
                    .add_enabled(!self.files.is_empty(), egui::Button::new("Add"))
                    .on_hover_text("Open alongside the current files (cross-file compare)")
                    .clicked()
                {
                    self.open_typed_path(ctx, true);
                }
                if ui.button("Cancel").clicked() {
                    self.path_popup = false;
                    self.path_error = None;
                }
            });
        });
        if modal.should_close() {
            self.path_popup = false;
            self.path_error = None;
        }
    }

    /// Modal to open every file of a run-number range from one directory
    /// (the way 100 runs of an experiment get compared).
    fn show_range_popup(&mut self, ctx: &egui::Context) {
        if !self.range_popup {
            return;
        }
        let modal = egui::Modal::new(egui::Id::new("open_runs")).show(ctx, |ui| {
            ui.set_width(620.0);
            ui.heading("Open runs");
            ui.label(
                "Opens the NeXus file of every run in the list. With no directory, \
                 the runs are located automatically across all IPTS experiments.",
            );
            ui.add_space(6.0);
            let root_hint =
                format!("empty = search {}/IPTS-*/nexus", self.instrument_root().display());
            ui.label("Directory (optional — empty searches every IPTS):");
            ui.add(
                egui::TextEdit::singleline(&mut self.range_dir)
                    .hint_text(root_hint)
                    .desired_width(f32::INFINITY),
            );
            ui.add_space(4.0);
            ui.label("Run numbers — ranges and single runs, comma separated:");
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.range_input)
                    .hint_text("26871-26970, 27012")
                    .desired_width(f32::INFINITY),
            );
            if self.range_focus {
                resp.request_focus();
                self.range_focus = false;
            }
            self.update_range_matches();
            let scanning = self.index_rx.is_some()
                && self.range_dir.trim().is_empty()
                && !self.range_input.trim().is_empty();
            let file_name = |p: &PathBuf| {
                p.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
            };
            if scanning {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(format!(
                        "scanning {}/IPTS-*/nexus …",
                        self.instrument_root().display()
                    ));
                });
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
            } else if let Some(err) = &self.range_error {
                ui.colored_label(ui.visuals().error_fg_color, err);
            } else {
                ui.horizontal(|ui| {
                    match self.range_matches.len() {
                        0 => {
                            ui.label(RichText::new("no matching file").weak());
                        }
                        1 => {
                            ui.label(format!("1 file: {}", file_name(&self.range_matches[0])));
                        }
                        n => {
                            ui.label(format!(
                                "{n} files: {} … {}",
                                file_name(&self.range_matches[0]),
                                file_name(&self.range_matches[n - 1])
                            ));
                        }
                    }
                    if let Some(note) = &self.range_note {
                        ui.label(RichText::new(note).weak());
                    }
                    if self.run_index.is_some()
                        && self.range_dir.trim().is_empty()
                        && ui
                            .small_button("⟳")
                            .on_hover_text("Rescan the IPTS directories (picks up new runs)")
                            .clicked()
                    {
                        self.run_index = None;
                        self.range_key = None;
                    }
                });
            }
            ui.add_space(6.0);
            let entered = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            let any = !self.range_matches.is_empty();
            ui.horizontal(|ui| {
                if ui.add_enabled(any, egui::Button::new("Open")).clicked() || (entered && any)
                {
                    self.range_popup = false;
                    self.enqueue_opens(self.range_matches.clone(), false);
                }
                if ui
                    .add_enabled(any && !self.files.is_empty(), egui::Button::new("Add"))
                    .on_hover_text("Open alongside the current files")
                    .clicked()
                {
                    self.range_popup = false;
                    self.enqueue_opens(self.range_matches.clone(), true);
                }
                if ui.button("Cancel").clicked() {
                    self.range_popup = false;
                }
            });
        });
        if modal.should_close() {
            self.range_popup = false;
        }
    }

    /// Recompute the run-popup match list when its inputs changed.
    fn update_range_matches(&mut self) {
        // Collect a finished IPTS scan and recompute with the fresh index.
        if let Some(Ok(idx)) = self.index_rx.as_ref().map(|rx| rx.try_recv()) {
            self.run_index = Some(idx);
            self.index_rx = None;
            self.range_key = None;
        }
        let key = (self.range_dir.clone(), self.range_input.clone());
        if self.range_key.as_ref() == Some(&key) {
            return;
        }
        self.range_key = Some(key);
        self.range_matches.clear();
        self.range_error = None;
        self.range_note = None;
        if self.range_input.trim().is_empty() {
            return;
        }
        let Some(ranges) = parse_run_ranges(&self.range_input) else {
            self.range_error = Some("cannot parse the run list".into());
            return;
        };
        let mut dir = self.range_dir.trim().to_owned();
        if dir.is_empty() {
            // No directory: locate the runs across every IPTS experiment.
            let root = self.instrument_root();
            match &self.run_index {
                Some(idx) if idx.root == root => {
                    let mut hits: Vec<(u64, PathBuf)> = Vec::new();
                    for &(lo, hi) in &ranges {
                        hits.extend(idx.by_run.range(lo..=hi).map(|(&r, p)| (r, p.clone())));
                    }
                    hits.sort();
                    hits.dedup();
                    self.range_matches = hits.into_iter().map(|(_, p)| p).collect();
                    self.range_note = Some(auto_match_note(&self.range_matches, &ranges));
                }
                _ => {
                    self.start_index_scan(root);
                    // Recompute each frame until the scan lands.
                    self.range_key = None;
                }
            }
            return;
        }
        if let Some(rest) = dir.strip_prefix("~/") {
            if let Ok(home) = std::env::var("HOME") {
                dir = format!("{home}/{rest}");
            }
        }
        let dir = PathBuf::from(dir);
        if !dir.is_dir() {
            self.range_error = Some(format!("not a directory: {}", dir.display()));
            return;
        }
        self.range_matches = find_run_files(&dir, &ranges);
    }

    /// Instrument root (e.g. /SNS/VENUS) whose IPTS-* directories the run
    /// search scans: taken from the current or recent file's path, VENUS
    /// otherwise.
    fn instrument_root(&self) -> PathBuf {
        let from = |p: &Path| {
            p.ancestors()
                .find(|a| {
                    a.file_name()
                        .is_some_and(|n| n.to_string_lossy().starts_with("IPTS-"))
                })
                .and_then(Path::parent)
                .map(Path::to_path_buf)
        };
        self.files
            .first()
            .and_then(|f| from(&f.tree.file_path))
            .or_else(|| self.recent.iter().find_map(|p| from(p)))
            .unwrap_or_else(|| PathBuf::from("/SNS/VENUS"))
    }

    /// Scan `root`/IPTS-*/nexus on a background thread (tens of thousands of
    /// directory entries on a network filesystem — too slow for the UI thread).
    fn start_index_scan(&mut self, root: PathBuf) {
        if self.index_rx.is_some() {
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        self.index_rx = Some(rx);
        std::thread::spawn(move || {
            let _ = tx.send(build_run_index(&root));
        });
    }

    /// Recompute the per-file search-match arrays if the query or the set of
    /// open files changed.
    fn update_filter(&mut self) {
        let key = (self.search.clone(), self.case_sensitive, self.generation);
        if self.computed_for.as_ref() == Some(&key) {
            return;
        }
        let needle = if self.case_sensitive {
            self.search.clone()
        } else {
            self.search.to_lowercase()
        };
        self.total_matches = 0;
        for of in &mut self.files {
            let n = of.tree.nodes.len();
            of.node_match = vec![false; n];
            of.subtree_match = vec![false; n];
            of.match_count = 0;
            if !needle.is_empty() {
                for i in 1..n {
                    let node = &of.tree.nodes[i];
                    let hay = if self.case_sensitive { &node.name } else { &node.name_lower };
                    if hay.contains(&needle) {
                        of.node_match[i] = true;
                        of.match_count += 1;
                    }
                    of.subtree_match[i] = of.node_match[i];
                }
                // Children always have larger indices than their parent, so
                // one reverse pass propagates matches up to the root.
                for i in (1..n).rev() {
                    if of.subtree_match[i] {
                        let p = of.tree.nodes[i].parent;
                        of.subtree_match[p] = true;
                    }
                }
            }
            self.total_matches += of.match_count;
        }
        self.computed_for = Some(key);
    }

    fn filter_active(&self) -> bool {
        !self.search.is_empty()
    }

    fn select(&mut self, file: usize, node: usize) {
        self.selected = Some((file, node));
        self.xy_cache = None;
        self.loaded = Some(load_node(&self.files[file].tree, file, node));
        self.rebuild_multi();
        // Groups are browsed for their attributes, not remembered as PVs.
        if !self.files[file].tree.nodes[node].is_group {
            self.remember_pv(file, node);
        }
    }

    /// Ctrl+click: add a PV to (or remove it from) the compare plot.
    fn toggle_other(&mut self, file: usize, node: usize) {
        self.xy_cache = None;
        if let Some(i) = self.others.iter().position(|l| l.sel() == (file, node)) {
            self.others.remove(i);
        } else if self.selected != Some((file, node)) {
            self.others.push(load_node(&self.files[file].tree, file, node));
            self.remember_pv(file, node);
        }
    }

    /// Drop the `i`-th extra selection of the compare plot.
    fn remove_other(&mut self, i: usize) {
        if i < self.others.len() {
            self.others.remove(i);
            self.xy_cache = None;
        }
    }

    fn remember_pv(&mut self, file: usize, node: usize) {
        let path = &self.files[file].tree.nodes[node].path;
        recent::add_pv(&mut self.recent_pvs, path);
    }

    /// The dataset at `path` in the first open file that has it.
    fn find_pv(&self, path: &str) -> Option<Sel> {
        self.files.iter().enumerate().find_map(|(f, of)| {
            of.tree
                .node_by_path(path)
                .filter(|&i| !of.tree.nodes[i].is_group)
                .map(|i| (f, i))
        })
    }

    fn sel_path(&self, (f, n): Sel) -> &str {
        &self.files[f].tree.nodes[n].path
    }

    /// Whether a PV at `path` (in any open file) is part of the current plot.
    fn pv_selected(&self, path: &str) -> bool {
        self.selected.is_some_and(|s| self.sel_path(s) == path)
            || self.others.iter().any(|l| self.sel_path(l.sel()) == path)
    }

    /// Recent-PV checkbox toggled on: the PV becomes the main selection when
    /// there is none, otherwise it joins the compare plot.
    fn check_pv(&mut self, path: &str) {
        let Some((f, n)) = self.find_pv(path) else { return };
        if self.selected.is_none() {
            self.select(f, n);
        } else if !self.pv_selected(path) {
            self.toggle_other(f, n);
        }
    }

    /// Recent-PV checkbox toggled off: the PV leaves the plot. When it was
    /// the main selection, the first extra one takes its place.
    fn uncheck_pv(&mut self, path: &str) {
        self.xy_cache = None;
        self.others.retain(|l| self.files[l.file].tree.nodes[l.node].path != path);
        if self.selected.is_some_and(|s| self.sel_path(s) == path) {
            if self.others.is_empty() {
                self.selected = None;
                self.loaded = None;
                self.multi.clear();
            } else {
                let l = self.others.remove(0);
                self.selected = Some(l.sel());
                self.loaded = Some(l);
                self.rebuild_multi();
            }
        }
    }

    /// Load the dataset at the selected PV's path from every open file, for
    /// the cross-file compare view (only meaningful with 2+ files).
    fn rebuild_multi(&mut self) {
        self.multi.clear();
        let Some((sf, sn)) = self.selected else { return };
        if self.files.len() < 2 {
            return;
        }
        let path = self.files[sf].tree.nodes[sn].path.clone();
        for f in 0..self.files.len() {
            let tree = &self.files[f].tree;
            let loaded = tree
                .node_by_path(&path)
                .filter(|&i| !tree.nodes[i].is_group)
                .map(|i| load_node(tree, f, i));
            self.multi.push(loaded);
        }
    }

    /// (Re)build the x-y pairing cache when the compare selection changed
    /// (interpolating hundreds of thousands of points every frame would not
    /// keep the UI at 60 fps).
    fn ensure_xy_cache(&mut self) {
        if self.others.len() != 1 {
            return;
        }
        let (Some(a), Some(b)) = (&self.loaded, self.others.first()) else {
            return;
        };
        if !a.plottable() || !b.plottable() {
            return;
        }
        let key = (a.sel(), b.sel(), self.swap_xy);
        if self.xy_cache.as_ref().is_some_and(|c| c.key == key) {
            return;
        }
        let cache = build_xy(a, b, self.swap_xy);
        self.xy_cache = Some(cache);
    }
}

fn load_node(tree: &Tree, file: usize, idx: usize) -> Loaded {
    let node = &tree.nodes[idx];
    // For DASlogs PVs the interesting name is the group, not `value`.
    let label = if node.name == "value" {
        tree.nodes[node.parent].name.clone()
    } else {
        node.name.clone()
    };
    let mut loaded = Loaded {
        file,
        node: idx,
        value: Value::Empty,
        attrs: h5io::read_attributes(&tree.file, &node.path),
        label,
        units: None,
        y: Vec::new(),
        x: None,
        points: Vec::new(),
        x_label: "index".to_string(),
        stats: None,
    };
    loaded.units = loaded
        .attrs
        .iter()
        .find(|(k, _)| k == "units")
        .map(|(_, v)| v.clone())
        .filter(|u| !u.is_empty());
    let Ok(ds) = tree.file.dataset(&node.path) else {
        return loaded;
    };
    loaded.value = h5io::read_container(&ds);

    // Build plot data for 1-D numeric datasets. If a sibling `time` dataset
    // of the same length exists (SNS DASlogs layout), use it as the x axis.
    if let Value::Numeric { data, shape } = &loaded.value {
        loaded.stats = compute_stats(data);
        let squeezed: Vec<usize> = shape.iter().copied().filter(|&d| d > 1).collect();
        if data.len() > 1 && squeezed.len() <= 1 {
            if node.name != "time" {
                if let Some(t_idx) = tree.child_named(node.parent, "time") {
                    if let Ok(t_ds) = tree.file.dataset(&tree.nodes[t_idx].path) {
                        if let Value::Numeric { data: t, .. } = h5io::read_container(&t_ds) {
                            if t.len() == data.len() {
                                let units = h5io::read_attributes(&tree.file, &tree.nodes[t_idx].path)
                                    .into_iter()
                                    .find(|(k, _)| k == "units")
                                    .map(|(_, v)| v)
                                    .unwrap_or_else(|| "s".into());
                                loaded.x_label = format!("time ({units})");
                                loaded.x = Some(t);
                            }
                        }
                    }
                }
            }
            loaded.points = decimate(loaded.x.as_deref(), data);
            loaded.y = data.clone();
        }
    }
    loaded
}

fn compute_stats(data: &[f64]) -> Option<Stats> {
    if data.is_empty() {
        return None;
    }
    let (mut min, mut max, mut sum) = (f64::INFINITY, f64::NEG_INFINITY, 0.0);
    for &v in data {
        min = min.min(v);
        max = max.max(v);
        sum += v;
    }
    Some(Stats { n: data.len(), min, max, mean: sum / data.len() as f64 })
}

/// Turn (x, y) series into plot points, min/max-decimated so huge arrays stay
/// responsive without hiding spikes.
fn decimate(x: Option<&[f64]>, y: &[f64]) -> Vec<[f64; 2]> {
    const MAX_POINTS: usize = 20_000;
    let n = y.len();
    let get_x = |i: usize| x.map_or(i as f64, |x| x[i]);
    if n <= MAX_POINTS {
        return (0..n).map(|i| [get_x(i), y[i]]).collect();
    }
    let buckets = MAX_POINTS / 2;
    let mut out = Vec::with_capacity(buckets * 2);
    for b in 0..buckets {
        let start = b * n / buckets;
        let end = ((b + 1) * n / buckets).max(start + 1);
        let (mut i_min, mut i_max) = (start, start);
        for i in start..end {
            if y[i] < y[i_min] {
                i_min = i;
            }
            if y[i] > y[i_max] {
                i_max = i;
            }
        }
        let (first, second) = if i_min <= i_max { (i_min, i_max) } else { (i_max, i_min) };
        out.push([get_x(first), y[first]]);
        if second != first {
            out.push([get_x(second), y[second]]);
        }
    }
    out
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        // Open files dropped onto the window: Ctrl+drop adds them to the open
        // set; a plain drop replaces it (dropping several at once opens them
        // together, ready to compare).
        let (dropped, ctrl): (Vec<PathBuf>, bool) = ctx.input(|i| {
            (
                i.raw.dropped_files.iter().filter_map(|f| f.path.clone()).collect(),
                i.modifiers.ctrl || i.modifiers.command,
            )
        });
        if !dropped.is_empty() {
            self.enqueue_opens(dropped, ctrl);
        }
        self.process_pending(&ctx);

        self.top_bar(ui, &ctx);
        self.show_path_popup(&ctx);
        self.show_range_popup(&ctx);
        self.update_filter();
        self.left_tree(ui);
        self.right_panel(ui);
    }
}

impl App {
    /// Toolbar. Its three groups (file buttons, search, status) sit on one
    /// row when they fit, otherwise the status group and then the search
    /// group move onto their own rows so nothing overlaps.
    fn top_bar(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        egui::Panel::top("top").show(ui, |ui| {
            ui.add_space(4.0);
            let avail = ui.available_width();
            let gap = ui.spacing().item_spacing.x;
            let [w_files, w_search, w_status] = self.top_widths;
            let one_row = w_files + w_search + w_status + 2.0 * gap <= avail;
            let two_rows = !one_row && w_files + w_search + gap <= avail;
            let mut widths = [0.0f32; 3];
            if one_row {
                ui.horizontal(|ui| {
                    widths[0] = self.top_files(ui, ctx);
                    widths[1] = self.top_search(ui, true);
                    widths[2] = self.top_status(ui, ctx);
                });
            } else if two_rows {
                ui.horizontal(|ui| {
                    widths[0] = self.top_files(ui, ctx);
                    widths[1] = self.top_search(ui, true);
                });
                ui.horizontal(|ui| widths[2] = self.top_status(ui, ctx));
            } else {
                ui.horizontal(|ui| widths[0] = self.top_files(ui, ctx));
                ui.horizontal(|ui| widths[1] = self.top_search(ui, false));
                ui.horizontal(|ui| widths[2] = self.top_status(ui, ctx));
            }
            // Widths from this frame decide next frame's layout; redraw once
            // more if they changed (e.g. right after a resize or open).
            let changed = widths
                .iter()
                .zip(self.top_widths.iter())
                .any(|(a, b)| (a - b).abs() > 0.5);
            self.top_widths = widths;
            if changed {
                ctx.request_repaint();
            }
            ui.add_space(4.0);
        });
    }

    /// Open / Add / Runs / Path / Recent buttons. Returns the width used.
    fn top_files(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) -> f32 {
        let resp = ui.horizontal(|ui| {
            if ui.button("📂 Open…").clicked() {
                self.open_dialog(None, false);
            }
            if ui
                .button("➕ Add…")
                .on_hover_text("Open another file alongside, to compare PVs across files")
                .clicked()
            {
                self.open_dialog(None, true);
            }
            if ui
                .button("🔢 Runs…")
                .on_hover_text(
                    "Open runs by number (e.g. 26871-26970) — their NeXus\n\
                     files are located automatically across all IPTS",
                )
                .clicked()
            {
                self.range_popup = true;
                self.range_focus = true;
            }
            if ui
                .button("⌨ Path…")
                .on_hover_text("Type or paste a file / directory path")
                .clicked()
            {
                // Pre-fill with the current file's directory.
                if self.path_input.is_empty() {
                    if let Some(dir) = self.default_dir() {
                        self.path_input = dir.display().to_string();
                    }
                }
                self.path_popup = true;
                self.path_focus = true;
                self.path_error = None;
            }
            let mut reopen: Option<(PathBuf, bool)> = None;
            let mut clear_recent = false;
            ui.add_enabled_ui(!self.recent.is_empty(), |ui| {
                ui.menu_button("🕘 Recent", |ui| {
                    for p in &self.recent {
                        let name = p
                            .file_name()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_else(|| p.display().to_string());
                        let exists = p.exists();
                        let resp = ui
                            .add_enabled(exists, egui::Button::new(name))
                            .on_hover_text(format!(
                                "{}\nCtrl+click: add to the open files",
                                p.display()
                            ))
                            .on_disabled_hover_text(format!(
                                "File not found: {}",
                                p.display()
                            ));
                        if resp.clicked() {
                            let ctrl = ui.input(|i| i.modifiers.ctrl || i.modifiers.command);
                            reopen = Some((p.clone(), ctrl && !self.files.is_empty()));
                        }
                    }
                    ui.separator();
                    if ui.button("Clear list").clicked() {
                        clear_recent = true;
                    }
                })
                .response
                .on_hover_text("Reopen one of the last files");
            });
            if let Some((p, add)) = reopen {
                self.open_file(&p, ctx, add);
            }
            if clear_recent {
                recent::clear(&mut self.recent);
            }
        });
        resp.response.rect.width()
    }

    /// Search box, its toggles, match count and bulk-open progress; `sep`
    /// draws the divider from the file buttons when both share a row.
    /// Returns the width used.
    fn top_search(&mut self, ui: &mut egui::Ui, sep: bool) -> f32 {
        let resp = ui.horizontal(|ui| {
            if sep {
                ui.separator();
            }
            ui.label("🔍");
            ui.add(
                egui::TextEdit::singleline(&mut self.search)
                    .hint_text("Search PVs / keywords…")
                    .desired_width(320.0),
            );
            if ui
                .selectable_label(self.case_sensitive, RichText::new("Aa").strong())
                .on_hover_text("Case sensitive search")
                .clicked()
            {
                self.case_sensitive = !self.case_sensitive;
            }
            if ui
                .add_enabled(!self.search.is_empty(), egui::Button::new("✖"))
                .on_hover_text("Clear search")
                .clicked()
            {
                self.search.clear();
            }
            if self.filter_active() {
                let color = if self.total_matches == 0 {
                    ui.visuals().error_fg_color
                } else if ui.visuals().dark_mode {
                    Color32::LIGHT_GREEN
                } else {
                    Color32::DARK_GREEN
                };
                ui.label(RichText::new(format!("{} match(es)", self.total_matches)).color(color));
            }
            if !self.pending.is_empty() {
                ui.separator();
                ui.spinner();
                ui.label(format!(
                    "opening {}/{}…",
                    self.pending_total - self.pending.len(),
                    self.pending_total
                ));
                if ui
                    .small_button("✖")
                    .on_hover_text("Stop opening the remaining files")
                    .clicked()
                {
                    self.cancel_pending();
                }
            }
        });
        resp.response.rect.width()
    }

    /// Right-aligned status: theme / text-size toggles and the open file(s).
    /// Returns the width the group wants, counting the file path at its
    /// full length even when it had to be truncated to fit.
    fn top_status(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) -> f32 {
        let mut close_all = false;
        let mut want = 0.0f32;
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            theme::toggle_button(ui);
            zoom::toggle_button(ui);
            match self.files.len() {
                0 => {}
                1 => {
                    let tree = &self.files[0].tree;
                    let text = format!(
                        "{}   ({} groups, {} datasets)",
                        tree.file_path.display(),
                        tree.n_groups,
                        tree.n_datasets
                    );
                    let full = ui
                        .painter()
                        .layout_no_wrap(
                            text.clone(),
                            egui::TextStyle::Body.resolve(ui.style()),
                            Color32::PLACEHOLDER,
                        )
                        .size()
                        .x;
                    let resp = ui
                        .add(egui::Label::new(RichText::new(text).weak()).truncate())
                        .on_hover_text(tree.file_path.display().to_string());
                    want += (full - resp.rect.width()).max(0.0);
                }
                n => {
                    if ui
                        .small_button("✖ all")
                        .on_hover_text("Close all files")
                        .clicked()
                    {
                        close_all = true;
                    }
                    let list = self
                        .files
                        .iter()
                        .map(|f| f.tree.file_path.display().to_string())
                        .collect::<Vec<_>>()
                        .join("\n");
                    ui.label(RichText::new(format!("{n} files open")).weak())
                        .on_hover_text(list);
                }
            }
            want += ui.min_rect().width();
        });
        if close_all {
            self.close_all(ctx);
        }
        want
    }

    fn left_tree(&mut self, ui: &mut egui::Ui) {
        egui::Panel::left("tree")
            .resizable(true)
            .default_size(430.0)
            .show(ui, |ui| {
                if self.files.is_empty() {
                    ui.add_space(20.0);
                    ui.label(
                        "Open a NeXus file (📂 or 🕘 Recent above, or drag & drop).\n\n\
                         Open several (➕ Add, or drop them together) to compare\n\
                         the same PV across runs.",
                    );
                    return;
                }
                let many = self.files.len() > 1;
                ui.label(
                    RichText::new(if many {
                        "Click a PV to compare it across all open files;\n\
                         Ctrl+click more PVs to plot them together"
                    } else {
                        "Ctrl+click a second PV to plot one vs the other,\n\
                         more to overlay them all"
                    })
                    .weak()
                    .small(),
                );
                self.recent_pv_section(ui);
                ui.separator();
                let filter = self.filter_active();
                // A handful of files start expanded; a 100-run range starts
                // collapsed so the list stays scannable.
                let default_open = self.files.len() <= 4;
                let mut clicked: Option<(usize, usize, bool)> = None;
                let mut close: Option<usize> = None;
                let others: Vec<Sel> = self.others.iter().map(Loaded::sel).collect();
                egui::ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
                    for f in 0..self.files.len() {
                        let label = self.file_label(f);
                        let of = &self.files[f];
                        let id = ui.make_persistent_id(("nexus_file", of.tree.file_path.as_path()));
                        egui::collapsing_header::CollapsingState::load_with_default_open(
                            ui.ctx(),
                            id,
                            default_open,
                        )
                        .show_header(ui, |ui| {
                            if many {
                                ui.label(RichText::new(format!("[{}]", f + 1)).strong());
                            }
                            ui.label(RichText::new(&label).strong())
                                .on_hover_text(of.tree.file_path.display().to_string());
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .small_button("✖")
                                        .on_hover_text("Close this file")
                                        .clicked()
                                    {
                                        close = Some(f);
                                    }
                                },
                            );
                        })
                        .body(|ui| {
                            // Node ids repeat across files (same HDF5 paths),
                            // so scope them per file.
                            ui.push_id(of.tree.file_path.as_path(), |ui| {
                                if filter && of.match_count == 0 {
                                    ui.label(
                                        RichText::new("No PV matches the search.").weak(),
                                    );
                                    return;
                                }
                                // The file's own (root) attributes — e.g. a
                                // normalization configuration's output_folder,
                                // ipts, created — live on "/" which has no
                                // header of its own: one entry for them.
                                let root_selected = self.selected == Some((f, 0));
                                if ui
                                    .selectable_label(
                                        root_selected,
                                        RichText::new("⚙ file attributes").weak(),
                                    )
                                    .on_hover_text(
                                        "Attributes of the file's root group \
                                         (schema, provenance, output folder…)",
                                    )
                                    .clicked()
                                {
                                    clicked = Some((f, 0, false));
                                }
                                for &r in &of.tree.nodes[0].children.clone() {
                                    show_node(
                                        ui,
                                        &of.tree,
                                        f,
                                        r,
                                        filter,
                                        false,
                                        &of.node_match,
                                        &of.subtree_match,
                                        self.selected,
                                        &others,
                                        &mut clicked,
                                    );
                                }
                            });
                        });
                    }
                });
                if let Some((f, idx, ctrl)) = clicked {
                    // Clicking a group header selects its `value` dataset when
                    // it has one (typical for DASlogs PVs), else the group
                    // itself — so its attributes show (a configuration file's
                    // `export` flags, a NeXus group's NX_class…); clicking a
                    // dataset selects it directly. Ctrl+click adds a PV to
                    // the compare plot (or removes it again).
                    let tree = &self.files[f].tree;
                    let target = if tree.nodes[idx].is_group {
                        tree.child_named(idx, "value").or(Some(idx))
                    } else {
                        Some(idx)
                    };
                    if let Some(t) = target {
                        if ctrl {
                            self.toggle_other(f, t);
                        } else if self.selected != Some((f, t)) {
                            self.select(f, t);
                        }
                    }
                }
                if let Some(f) = close {
                    let ctx = ui.ctx().clone();
                    self.close_file(f, &ctx);
                }
            });
    }

    /// The last PVs plotted, as checkboxes above the tree: checking one
    /// plots it again (the first checked PV is the main selection, further
    /// ones overlay it), unchecking removes it from the plot. PVs absent
    /// from every open file are greyed out.
    fn recent_pv_section(&mut self, ui: &mut egui::Ui) {
        if self.recent_pvs.is_empty() {
            return;
        }
        let mut toggled: Option<(String, bool)> = None;
        let mut clear = false;
        let mut uncheck_all = false;
        let n_checked = self.recent_pvs.iter().filter(|p| self.pv_selected(p)).count();
        let title = if n_checked > 0 {
            format!("🕘 Recent PVs ({} checked)", n_checked)
        } else {
            format!("🕘 Recent PVs ({})", self.recent_pvs.len())
        };
        egui::CollapsingHeader::new(title)
            .id_salt("recent_pvs")
            .default_open(true)
            .show(ui, |ui| {
                ui.label(
                    RichText::new(
                        "Check any number of PVs to plot them again; the first\n\
                         checked one is the main PV, the others overlay it.",
                    )
                    .weak()
                    .small(),
                );
                egui::ScrollArea::vertical()
                    .id_salt("recent_pvs_list")
                    .max_height(180.0)
                    .auto_shrink([false, true])
                    .show(ui, |ui| {
                        for path in &self.recent_pvs {
                            let present = self.find_pv(path).is_some();
                            let mut on = self.pv_selected(path);
                            let text = RichText::new(pv_display(path)).monospace();
                            let resp = ui
                                .add_enabled(present, egui::Checkbox::new(&mut on, text))
                                .on_hover_text(path.as_str())
                                .on_disabled_hover_text(format!(
                                    "{path}\nNot in the open files"
                                ));
                            if resp.changed() {
                                toggled = Some((path.clone(), on));
                            }
                        }
                    });
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(n_checked > 0, egui::Button::new("Uncheck all").small())
                        .on_hover_text("Clear the selection")
                        .clicked()
                    {
                        uncheck_all = true;
                    }
                    if ui
                        .small_button("Clear list")
                        .on_hover_text("Forget the recent PVs")
                        .clicked()
                    {
                        clear = true;
                    }
                });
            });
        if let Some((path, on)) = toggled {
            if on {
                self.check_pv(&path);
            } else {
                self.uncheck_pv(&path);
            }
        }
        if uncheck_all {
            self.selected = None;
            self.loaded = None;
            self.others.clear();
            self.multi.clear();
            self.xy_cache = None;
        }
        if clear {
            recent::clear_pvs(&mut self.recent_pvs);
        }
    }

    fn right_panel(&mut self, ui: &mut egui::Ui) {
        self.ensure_xy_cache();
        let mut mode = self.compare_mode;
        let mut swap = self.swap_xy;
        let mut norm = self.normalize;
        let mut remove_other: Option<usize> = None;
        let mut dismiss_error = false;
        egui::CentralPanel::default().show(ui, |ui| {
            if let Some(err) = &self.error {
                ui.horizontal(|ui| {
                    ui.colored_label(ui.visuals().error_fg_color, err);
                    if ui.small_button("✖").on_hover_text("Dismiss").clicked() {
                        dismiss_error = true;
                    }
                });
                ui.separator();
            }
            if self.loaded.is_none() {
                ui.add_space(20.0);
                ui.label(
                    RichText::new(
                        "Select a dataset in the tree to view it.\n\
                         With several files open, it is compared across all of them.\n\
                         Ctrl+click a second one to plot one PV against another,\n\
                         more of them to overlay several.\n\
                         Recent PVs can be re-checked from the list above the tree.",
                    )
                    .weak(),
                );
                return;
            }
            // The plots adapt to the window, but the header rows, the
            // attribute list and the plots' minimum heights can outgrow a
            // short window (small displays, the large-text mode). The panel
            // height is measured before entering the scroll area — inside it
            // the available height is unbounded — and the minimum plot
            // heights are what make the scroll bar appear.
            let panel_h = ui.available_height();
            egui::ScrollArea::vertical()
                .id_salt("dataset_scroll")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    self.dataset_content(ui, panel_h, &mut mode, &mut swap, &mut norm, &mut remove_other);
                });
        });
        self.compare_mode = mode;
        self.swap_xy = swap;
        self.normalize = norm;
        if let Some(i) = remove_other {
            self.remove_other(i);
        }
        if dismiss_error {
            self.error = None;
        }
    }

    /// The dataset view inside the central panel's scroll area. The flag
    /// references land back on `self` after the panel closes (the borrow of
    /// the loaded dataset keeps `self` shared here).
    fn dataset_content(
        &self,
        ui: &mut egui::Ui,
        panel_h: f32,
        mode: &mut CompareMode,
        swap: &mut bool,
        norm: &mut bool,
        remove_other: &mut Option<usize>,
    ) {
        let Some(loaded) = &self.loaded else { return };
        let many = self.files.len() > 1;
        let tree = &self.files[loaded.file].tree;
        let node = &tree.nodes[loaded.node];
        let compare = !self.others.is_empty();

        ui.horizontal(|ui| {
            if compare {
                ui.label(RichText::new("[1]").strong());
                if many {
                    ui.label(RichText::new(self.file_label(loaded.file)).strong());
                }
            }
            ui.label(RichText::new(&node.path).monospace().strong());
            if ui.small_button("📋").on_hover_text("Copy path").clicked() {
                ui.ctx().copy_text(node.path.clone());
            }
        });
        for (k, l) in self.others.iter().enumerate() {
            let node_k = &self.files[l.file].tree.nodes[l.node];
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("[{}]", k + 2)).strong());
                if many {
                    ui.label(RichText::new(self.file_label(l.file)).strong());
                }
                ui.label(RichText::new(&node_k.path).monospace().strong());
                if ui.small_button("📋").on_hover_text("Copy path").clicked() {
                    ui.ctx().copy_text(node_k.path.clone());
                }
                if ui.small_button("✖").on_hover_text("Remove from the plot").clicked() {
                    *remove_other = Some(k);
                }
            });
        }

        // Compare view: one extra plottable PV pairs with the main one (x-y)
        // or overlays it; several extra ones all overlay it.
        if compare {
            let warn = ui.visuals().warn_fg_color;
            let extra: Vec<(usize, &Loaded)> = self
                .others
                .iter()
                .enumerate()
                .filter(|(_, l)| l.plottable())
                .map(|(k, l)| (k + 2, l))
                .collect();
            for (k, l) in self.others.iter().enumerate() {
                if !l.plottable() {
                    ui.colored_label(
                        warn,
                        format!("[{}] has no plottable 1-D array — left out.", k + 2),
                    );
                }
            }
            if loaded.plottable() && !extra.is_empty() {
                ui.separator();
                let pair = self.others.len() == 1;
                let overlay = !pair || *mode == CompareMode::Overlay;
                ui.horizontal(|ui| {
                    ui.label("Plot:");
                    if pair {
                        ui.selectable_value(mode, CompareMode::Xy, "1 vs 2");
                        ui.selectable_value(mode, CompareMode::Overlay, "both as y");
                    } else {
                        ui.label(format!("all {} as y", extra.len() + 1));
                    }
                    if overlay {
                        ui.checkbox(norm, "normalize [0–1]");
                    } else if ui.button("swap axes").clicked() {
                        *swap = !*swap;
                    }
                });
                if let Some(s) = &loaded.stats {
                    stats_line(ui, "[1] ", s);
                }
                for (k, l) in &extra {
                    if let Some(s) = &l.stats {
                        stats_line(ui, &format!("[{k}] "), s);
                    }
                }
                ui.add_space(4.0);
                if overlay {
                    let mut series = vec![(1, loaded)];
                    series.extend(extra.iter().copied());
                    draw_overlay(ui, &series, *norm, panel_h);
                } else {
                    let key = (loaded.sel(), extra[0].1.sel(), *swap);
                    draw_xy(ui, self.xy_cache.as_ref(), key, panel_h);
                }
                return;
            }
            if !loaded.plottable() {
                ui.colored_label(
                    warn,
                    "The 1st selection has no plottable 1-D array — showing it alone.",
                );
            }
        }

        // Cross-file view: the selected PV in every open file.
        if many && !compare {
            let labels: Vec<String> =
                (0..self.files.len()).map(|f| self.file_label(f)).collect();
            draw_multi(ui, &self.multi, &labels, norm, panel_h);
            return;
        }

        if node.is_group {
            ui.label(
                RichText::new(format!(
                    "group — {} member(s){}",
                    node.children.len(),
                    if node.path == "/" { " (file root)" } else { "" }
                ))
                .weak(),
            );
        } else {
            ui.label(
                RichText::new(format!("dtype: {}   shape: {:?}", node.dtype, node.shape))
                    .weak(),
            );
        }

        if node.is_group && loaded.attrs.is_empty() {
            ui.label(RichText::new("(no attributes)").weak());
        }
        if !loaded.attrs.is_empty() {
            egui::CollapsingHeader::new(format!("Attributes ({})", loaded.attrs.len()))
                .default_open(true)
                .show(ui, |ui| {
                    egui::Grid::new("attrs").striped(true).show(ui, |ui| {
                        for (k, v) in &loaded.attrs {
                            ui.label(RichText::new(k).monospace());
                            ui.label(RichText::new(v).monospace());
                            ui.end_row();
                        }
                    });
                });
        }
        ui.separator();

        match &loaded.value {
            Value::Empty if node.is_group => {
                ui.label(
                    RichText::new(
                        "A group holds no data of its own — its attributes are listed \
                         above; expand it in the tree for its members.",
                    )
                    .weak(),
                );
            }
            Value::Empty => {
                ui.label(RichText::new("(empty dataset)").weak());
            }
            Value::Unsupported(msg) => {
                let color = ui.visuals().error_fg_color;
                ui.colored_label(color, msg);
            }
            Value::Strings(items) => show_strings(ui, items, panel_h),
            Value::Numeric { data, shape } => {
                if data.len() == 1 {
                    let units = loaded
                        .attrs
                        .iter()
                        .find(|(k, _)| k == "units")
                        .map(|(_, v)| format!(" {v}"))
                        .unwrap_or_default();
                    ui.add_space(8.0);
                    ui.label(
                        RichText::new(format!("{}{units}", format_num(data[0])))
                            .monospace()
                            .size(26.0),
                    );
                } else if !loaded.points.is_empty() {
                    show_plot(ui, loaded, panel_h);
                } else {
                    ui.label(format!(
                        "{}-D array {:?} — plotting supports 1-D data. First values:",
                        shape.len(),
                        shape
                    ));
                    ui.label(
                        RichText::new(h5io::format_value_short(&loaded.value, 100))
                            .monospace(),
                    );
                    if let Some(s) = &loaded.stats {
                        stats_line(ui, "", s);
                    }
                }
            }
        }
    }
}

/// "26871-26970, 27012 27040-27045" → inclusive (lo, hi) ranges; None when a
/// token is not a number or a number-number range.
fn parse_run_ranges(s: &str) -> Option<Vec<(u64, u64)>> {
    let mut out = Vec::new();
    for tok in s.split(',').flat_map(str::split_whitespace) {
        if let Some((a, b)) = tok.split_once('-') {
            let (a, b): (u64, u64) = (a.trim().parse().ok()?, b.trim().parse().ok()?);
            out.push((a.min(b), a.max(b)));
        } else {
            let v = tok.parse().ok()?;
            out.push((v, v));
        }
    }
    (!out.is_empty()).then_some(out)
}

/// Order matters: ".nxs.h5" must be tried before ".h5".
const NEXUS_EXTS: [&str; 5] = [".nxs.h5", ".nxs", ".h5", ".hdf5", ".nx5"];

/// NeXus files in `dir` whose name ends in a run number falling in one of
/// `ranges` (e.g. VENUS_26871.nxs.h5), sorted by run number.
fn find_run_files(dir: &Path, ranges: &[(u64, u64)]) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut hits: Vec<(u64, PathBuf)> = Vec::new();
    for entry in rd.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(stem) = NEXUS_EXTS.iter().find_map(|e| name.strip_suffix(e)) else {
            continue;
        };
        let Some(run) = trailing_number(stem) else { continue };
        if ranges.iter().any(|&(lo, hi)| run >= lo && run <= hi) {
            hits.push((run, entry.path()));
        }
    }
    hits.sort();
    hits.into_iter().map(|(_, p)| p).collect()
}

/// Index every run found under `root`/IPTS-*/nexus. A run lives in exactly
/// one IPTS, so a flat run → file map is enough to locate it.
fn build_run_index(root: &Path) -> RunIndex {
    let mut by_run = std::collections::BTreeMap::new();
    if let Ok(rd) = std::fs::read_dir(root) {
        for entry in rd.flatten() {
            if !entry.file_name().to_string_lossy().starts_with("IPTS-") {
                continue;
            }
            let Ok(files) = std::fs::read_dir(entry.path().join("nexus")) else {
                continue;
            };
            for f in files.flatten() {
                let name = f.file_name().to_string_lossy().into_owned();
                let Some(stem) = NEXUS_EXTS.iter().find_map(|e| name.strip_suffix(e)) else {
                    continue;
                };
                if let Some(run) = trailing_number(stem) {
                    by_run.insert(run, f.path());
                }
            }
        }
    }
    RunIndex { root: root.to_path_buf(), by_run }
}

/// "from 3 IPTS" plus how many requested runs have no file, for the
/// located-anywhere mode of the runs popup.
fn auto_match_note(matches: &[PathBuf], ranges: &[(u64, u64)]) -> String {
    let ipts: std::collections::HashSet<&Path> = matches
        .iter()
        .filter_map(|p| p.parent().and_then(Path::parent))
        .collect();
    let requested = merged_run_count(ranges);
    let missing = requested.saturating_sub(matches.len() as u64);
    let mut note = format!("(from {} IPTS", ipts.len());
    if missing > 0 {
        note += &format!("; {missing} of the {requested} runs not found");
    }
    note + ")"
}

/// Number of distinct runs the (possibly overlapping) ranges ask for.
fn merged_run_count(ranges: &[(u64, u64)]) -> u64 {
    let mut sorted = ranges.to_vec();
    sorted.sort();
    let mut total = 0;
    let mut cur: Option<(u64, u64)> = None;
    for &(lo, hi) in &sorted {
        match cur {
            Some((clo, chi)) if lo <= chi => cur = Some((clo, chi.max(hi))),
            Some((clo, chi)) => {
                total += chi - clo + 1;
                cur = Some((lo, hi));
            }
            None => cur = Some((lo, hi)),
        }
    }
    if let Some((clo, chi)) = cur {
        total += chi - clo + 1;
    }
    total
}

/// The number a name ends with ("VENUS_26871" → 26871).
fn trailing_number(s: &str) -> Option<u64> {
    let b = s.as_bytes();
    let mut i = b.len();
    while i > 0 && b[i - 1].is_ascii_digit() {
        i -= 1;
    }
    s[i..].parse().ok()
}

#[cfg(test)]
mod run_range_tests {
    use super::*;

    #[test]
    fn parse_ranges() {
        assert_eq!(parse_run_ranges("26871"), Some(vec![(26871, 26871)]));
        assert_eq!(
            parse_run_ranges("26871-26970, 27012 27040-27045"),
            Some(vec![(26871, 26970), (27012, 27012), (27040, 27045)])
        );
        // Reversed bounds are normalized.
        assert_eq!(parse_run_ranges("30-10"), Some(vec![(10, 30)]));
        assert_eq!(parse_run_ranges(""), None);
        assert_eq!(parse_run_ranges("abc"), None);
        assert_eq!(parse_run_ranges("26871-"), None);
    }

    #[test]
    fn pv_display_names() {
        assert_eq!(pv_display("/entry/DASlogs/BL10:SE:Temp/value"), "BL10:SE:Temp");
        assert_eq!(
            pv_display("/entry/DASlogs/BL10:SE:Temp/average_value"),
            "BL10:SE:Temp/average_value"
        );
        assert_eq!(pv_display("/entry/title"), "entry/title");
        assert_eq!(pv_display("/title"), "title");
    }

    #[test]
    fn trailing_numbers() {
        assert_eq!(trailing_number("VENUS_26871"), Some(26871));
        assert_eq!(trailing_number("run42"), Some(42));
        assert_eq!(trailing_number("VENUS_"), None);
        assert_eq!(trailing_number(""), None);
    }

    #[test]
    fn merged_run_counts() {
        assert_eq!(merged_run_count(&[(10, 20)]), 11);
        // Overlapping and duplicate ranges count each run once.
        assert_eq!(merged_run_count(&[(10, 20), (15, 25), (15, 25)]), 16);
        assert_eq!(merged_run_count(&[(30, 30), (10, 20)]), 12);
    }

    #[test]
    fn index_locates_run_without_ipts() {
        // Run 26871 lives in IPTS-38715; the index must find it from the
        // instrument root alone.
        let idx = build_run_index(Path::new("/SNS/VENUS"));
        assert!(idx.by_run.len() > 1000);
        let p = idx.by_run.get(&26871).expect("run 26871 in the index");
        assert!(p.ends_with("IPTS-38715/nexus/VENUS_26871.nxs.h5"), "{}", p.display());
    }

    #[test]
    fn find_runs_in_sample_dir() {
        // The IPTS-38715 nexus directory holds VENUS_26858 … VENUS_26871.
        let dir = Path::new("/SNS/VENUS/IPTS-38715/nexus");
        let hits = find_run_files(dir, &[(26860, 26862), (26871, 26871)]);
        let names: Vec<String> = hits
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            [
                "VENUS_26860.nxs.h5",
                "VENUS_26861.nxs.h5",
                "VENUS_26862.nxs.h5",
                "VENUS_26871.nxs.h5"
            ]
        );
    }
}

/// Cross-file compare: value/stats of the selected PV in each open file, and
/// an overlay plot of the files where it is a 1-D series.
fn draw_multi(
    ui: &mut egui::Ui,
    multi: &[Option<Loaded>],
    labels: &[String],
    normalize: &mut bool,
    panel_h: f32,
) {
    let scalar = |l: &Loaded| match &l.value {
        Value::Numeric { data, .. } if data.len() == 1 => Some(data[0]),
        _ => None,
    };
    // Δ column reference: the first file that has the PV as a scalar.
    let reference = multi.iter().flatten().next().and_then(scalar);
    ui.separator();
    // With ~100 open files the table alone would fill the screen — scroll it
    // and keep the plot below visible.
    let remaining = (panel_h - ui.min_rect().height() - ui.spacing().item_spacing.y).max(0.0);
    let table_height = (remaining * 0.45)
        .max(100.0)
        .min(22.0 * multi.len() as f32 + 8.0);
    egui::ScrollArea::vertical()
        .id_salt("multi_table")
        .max_height(table_height)
        .auto_shrink([false, true])
        .show(ui, |ui| {
            egui::Grid::new("multi_grid").striped(true).show(ui, |ui| {
                for (i, (entry, label)) in multi.iter().zip(labels).enumerate() {
                    ui.label(RichText::new(format!("[{}] {label}", i + 1)).strong());
                    match entry {
                        None => {
                            ui.label(RichText::new("not in this file").weak());
                        }
                        Some(l) => {
                            ui.label(RichText::new(value_summary(l)).monospace());
                            if let (Some(r), Some(v)) = (reference, scalar(l)) {
                                let d = v - r;
                                if d == 0.0 {
                                    ui.label(RichText::new("=").weak());
                                } else {
                                    ui.label(
                                        RichText::new(format!("Δ = {}", format_num(d)))
                                            .monospace()
                                            .color(ui.visuals().warn_fg_color),
                                    );
                                }
                            }
                        }
                    }
                    ui.end_row();
                }
            });
        });

    // Logged PVs: every file's curve overlaid in one plot.
    let plottable: Vec<(usize, &Loaded)> = multi
        .iter()
        .enumerate()
        .filter_map(|(i, e)| e.as_ref().filter(|l| l.plottable()).map(|l| (i, l)))
        .collect();
    if !plottable.is_empty() {
        ui.add_space(4.0);
        if plottable.len() > 1 {
            ui.horizontal(|ui| {
                ui.checkbox(normalize, "normalize [0–1]");
                if plottable.len() > 20 {
                    ui.label(
                        RichText::new(format!("{} series — legend hidden", plottable.len()))
                            .weak(),
                    );
                }
            });
        }
        let mut x_labels: Vec<&str> =
            plottable.iter().map(|(_, l)| l.x_label.as_str()).collect();
        x_labels.dedup();
        let x_label = x_labels.join(" / ");
        let norm = *normalize && plottable.len() > 1;
        let y_label = if norm {
            "normalized [0–1]".to_string()
        } else {
            series_label(plottable[0].1)
        };
        // Cap the total point count so 100 overlaid runs still redraw fast.
        let per_line = (200_000 / plottable.len()).max(1_000);
        let mut plot = Plot::new("multi_plot")
            .x_axis_label(x_label)
            .y_axis_label(y_label)
            .allow_boxed_zoom(true)
            .height(plot_height(ui, panel_h));
        if plottable.len() <= 20 {
            plot = plot.legend(Legend::default());
        }
        plot.show(ui, |plot_ui| {
            for (i, l) in &plottable {
                let pts = match (norm, &l.stats) {
                    (true, Some(s)) => norm_points(&l.points, s),
                    _ => l.points.clone(),
                };
                plot_ui.line(Line::new(
                    format!("[{}] {}", i + 1, labels[*i]),
                    PlotPoints::from(stride_cap(pts, per_line)),
                ));
            }
        });
        return;
    }

    // Scalar PV in several files: plot its value against the run number
    // (falling back to the file position when names carry no number).
    let scalars: Vec<(usize, f64)> = multi
        .iter()
        .enumerate()
        .filter_map(|(i, e)| e.as_ref().and_then(&scalar).map(|v| (i, v)))
        .collect();
    if scalars.len() >= 3 {
        let runs: Vec<Option<u64>> =
            scalars.iter().map(|&(i, _)| trailing_number(&labels[i])).collect();
        let mut distinct: Vec<u64> = runs.iter().flatten().copied().collect();
        distinct.sort();
        distinct.dedup();
        let use_runs = distinct.len() == scalars.len();
        let pts: Vec<[f64; 2]> = scalars
            .iter()
            .zip(&runs)
            .map(|(&(i, v), r)| {
                let x = if use_runs { r.unwrap() as f64 } else { i as f64 + 1.0 };
                [x, v]
            })
            .collect();
        let x_label = if use_runs { "run number" } else { "file #" };
        let y_label = series_label(multi[scalars[0].0].as_ref().unwrap());
        ui.add_space(4.0);
        Plot::new("multi_scalar_plot")
            .x_axis_label(x_label)
            .y_axis_label(y_label.clone())
            .allow_boxed_zoom(true)
            .height(plot_height(ui, panel_h))
            .show(ui, |plot_ui| {
                plot_ui.line(Line::new(y_label.clone(), PlotPoints::from(pts.clone())));
                plot_ui.points(Points::new(y_label, PlotPoints::from(pts)).radius(3.0));
            });
    }
}

/// Short name of a PV for the recent list: the group name for a DASlogs
/// `value` dataset, else `parent/name`.
fn pv_display(path: &str) -> String {
    let mut parts = path.rsplit('/');
    let last = parts.next().unwrap_or(path);
    let parent = parts.next().filter(|p| !p.is_empty());
    match parent {
        Some(p) if last == "value" => p.to_string(),
        Some(p) => format!("{p}/{last}"),
        None => last.to_string(),
    }
}

/// One-line rendering of a PV for the cross-file table.
fn value_summary(l: &Loaded) -> String {
    match &l.value {
        Value::Empty => "(empty)".into(),
        Value::Unsupported(m) => format!("<{m}>"),
        Value::Strings(s) if s.len() == 1 => truncate(&s[0], 120),
        Value::Strings(s) => format!("{} strings: {}, …", s.len(), truncate(&s[0], 60)),
        Value::Numeric { data, .. } if data.len() == 1 => match &l.units {
            Some(u) => format!("{} {u}", format_num(data[0])),
            None => format_num(data[0]),
        },
        Value::Numeric { shape, .. } => match &l.stats {
            Some(s) => format!(
                "n = {}    min = {}    max = {}    mean = {}",
                s.n,
                format_num(s.min),
                format_num(s.max),
                format_num(s.mean)
            ),
            None => format!("array {shape:?}"),
        },
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(max).collect::<String>())
    }
}

fn show_strings(ui: &mut egui::Ui, items: &[String], panel_h: f32) {
    ui.add_space(8.0);
    if items.len() == 1 {
        ui.label(RichText::new(&items[0]).monospace().size(20.0));
    } else {
        ui.label(RichText::new(format!("{} strings:", items.len())).weak());
        // Nested inside the panel's scroll area, so it needs its own cap.
        let list_h = plot_height(ui, panel_h);
        egui::ScrollArea::vertical().max_height(list_h).auto_shrink([false, false]).show(ui, |ui| {
            for (i, s) in items.iter().enumerate().take(2000) {
                ui.label(RichText::new(format!("[{i}] {s}")).monospace());
            }
            if items.len() > 2000 {
                ui.label(RichText::new(format!("… {} more", items.len() - 2000)).weak());
            }
        });
    }
}

fn stats_line(ui: &mut egui::Ui, prefix: &str, s: &Stats) {
    ui.label(
        RichText::new(format!(
            "{prefix}n = {}    min = {}    max = {}    mean = {}",
            s.n,
            format_num(s.min),
            format_num(s.max),
            format_num(s.mean)
        ))
        .monospace(),
    );
}

/// "PV name (units)" label used on plot axes and legends.
fn series_label(l: &Loaded) -> String {
    match &l.units {
        Some(u) => format!("{} ({u})", l.label),
        None => l.label.clone(),
    }
}

fn show_plot(ui: &mut egui::Ui, loaded: &Loaded, panel_h: f32) {
    if let Some(s) = &loaded.stats {
        stats_line(ui, "", s);
    }
    let y_label = series_label(loaded);
    ui.add_space(4.0);
    let line = Line::new(y_label.clone(), PlotPoints::from(loaded.points.clone()));
    Plot::new("dataset_plot")
        .x_axis_label(loaded.x_label.clone())
        .y_axis_label(y_label)
        .allow_boxed_zoom(true)
        .height(plot_height(ui, panel_h))
        .show(ui, |plot_ui| {
            plot_ui.line(line);
        });
}

/// Pair the two selected PVs into (x, y) points. PVs logged at different
/// times are paired by linearly interpolating the y PV onto the x PV's time
/// grid; series without a time axis are paired by index when lengths match.
fn build_xy(a: &Loaded, b: &Loaded, swap: bool) -> XyCache {
    let key = (a.sel(), b.sel(), swap);
    let (xl, yl) = if swap { (b, a) } else { (a, b) };
    let x_label = series_label(xl);
    let y_label = series_label(yl);
    let (points, paired_by): (Vec<[f64; 2]>, String) =
        if let (Some(tx), Some(ty)) = (&xl.x, &yl.x) {
            let yi = interp_onto(tx, ty, &yl.y);
            (
                xl.y.iter().zip(yi).map(|(&x, y)| [x, y]).collect(),
                format!("\"{}\" linearly interpolated onto \"{}\"'s time grid", yl.label, xl.label),
            )
        } else if xl.y.len() == yl.y.len() {
            (
                xl.y.iter().zip(yl.y.iter()).map(|(&x, &y)| [x, y]).collect(),
                "paired by index".to_string(),
            )
        } else {
            (
                Vec::new(),
                format!(
                    "cannot pair these PVs: {} vs {} points and no common time axis",
                    xl.y.len(),
                    yl.y.len()
                ),
            )
        };
    let n_pairs = points.len();
    let points = stride_cap(points, 60_000);
    XyCache { key, points, n_pairs, paired_by, x_label, y_label }
}

/// Linear interpolation of (t, v) sampled at the (sorted) times `grid`,
/// clamped at both ends.
fn interp_onto(grid: &[f64], t: &[f64], v: &[f64]) -> Vec<f64> {
    let last = t.len() - 1;
    let mut out = Vec::with_capacity(grid.len());
    let mut j = 0usize;
    for &x in grid {
        while j + 1 <= last && t[j + 1] < x {
            j += 1;
        }
        let y = if x <= t[0] {
            v[0]
        } else if x >= t[last] {
            v[last]
        } else {
            let (t0, t1) = (t[j], t[j + 1]);
            if t1 <= t0 {
                v[j]
            } else {
                v[j] + (v[j + 1] - v[j]) * (x - t0) / (t1 - t0)
            }
        };
        out.push(y);
    }
    out
}

fn stride_cap(points: Vec<[f64; 2]>, max: usize) -> Vec<[f64; 2]> {
    if points.len() <= max {
        return points;
    }
    let step = points.len().div_ceil(max);
    points.into_iter().step_by(step).collect()
}

/// Height for a plot filling the rest of the panel: what the panel has left
/// under the content drawn so far, floored so a short window scrolls instead
/// of squeezing the plot away.
fn plot_height(ui: &egui::Ui, panel_h: f32) -> f32 {
    (panel_h - ui.min_rect().height() - ui.spacing().item_spacing.y).max(160.0)
}

fn draw_xy(ui: &mut egui::Ui, cache: Option<&XyCache>, key: (Sel, Sel, bool), panel_h: f32) {
    let Some(c) = cache.filter(|c| c.key == key) else {
        // The cache is rebuilt at the start of the next frame.
        ui.spinner();
        ui.ctx().request_repaint();
        return;
    };
    if c.points.is_empty() {
        let color = ui.visuals().warn_fg_color;
        ui.colored_label(color, &c.paired_by);
        return;
    }
    let decim = if c.points.len() < c.n_pairs { "; display decimated" } else { "" };
    ui.label(RichText::new(format!("{} pairs — {}{decim}", c.n_pairs, c.paired_by)).weak());
    let pts = Points::new(
        format!("{}  vs  {}", c.y_label, c.x_label),
        PlotPoints::from(c.points.clone()),
    )
    .radius(1.5);
    Plot::new("xy_plot")
        .x_axis_label(c.x_label.clone())
        .y_axis_label(c.y_label.clone())
        .allow_boxed_zoom(true)
        .height(plot_height(ui, panel_h))
        .show(ui, |plot_ui| {
            plot_ui.points(pts);
        });
}

/// Several PVs as y-curves over their own x axes, each labelled with its
/// selection number.
fn draw_overlay(ui: &mut egui::Ui, series: &[(usize, &Loaded)], normalize: bool, panel_h: f32) {
    let mut x_labels: Vec<&str> = series.iter().map(|(_, l)| l.x_label.as_str()).collect();
    x_labels.dedup();
    let x_label = x_labels.join(" / ");
    let mut units: Vec<&str> =
        series.iter().map(|(_, l)| l.units.as_deref().unwrap_or("")).collect();
    units.dedup();
    let y_label = if normalize {
        "normalized [0–1]".to_string()
    } else if let [u] = units[..] {
        u.to_string()
    } else {
        String::new()
    };
    // Cap the total point count so many overlaid PVs still redraw fast.
    let per_line = (200_000 / series.len().max(1)).max(1_000);
    let lines: Vec<Line> = series
        .iter()
        .map(|(n, l)| {
            let pts = match (normalize, &l.stats) {
                (true, Some(s)) => norm_points(&l.points, s),
                _ => l.points.clone(),
            };
            Line::new(
                format!("[{n}] {}", series_label(l)),
                PlotPoints::from(stride_cap(pts, per_line)),
            )
        })
        .collect();
    Plot::new("overlay_plot")
        .legend(Legend::default())
        .x_axis_label(x_label)
        .y_axis_label(y_label)
        .allow_boxed_zoom(true)
        .height(plot_height(ui, panel_h))
        .show(ui, |plot_ui| {
            for line in lines {
                plot_ui.line(line);
            }
        });
}

fn norm_points(pts: &[[f64; 2]], s: &Stats) -> Vec<[f64; 2]> {
    let d = (s.max - s.min).abs().max(f64::EPSILON);
    pts.iter().map(|p| [p[0], (p[1] - s.min) / d]).collect()
}

#[allow(clippy::too_many_arguments)]
fn show_node(
    ui: &mut egui::Ui,
    tree: &Tree,
    file: usize,
    idx: usize,
    filter: bool,
    ancestor_matched: bool,
    node_match: &[bool],
    subtree_match: &[bool],
    selected: Option<Sel>,
    others: &[Sel],
    clicked: &mut Option<(usize, usize, bool)>,
) {
    if filter && !ancestor_matched && !subtree_match[idx] {
        return;
    }
    let node = &tree.nodes[idx];
    let matched = filter && node_match[idx];
    let highlight = if ui.visuals().dark_mode {
        Color32::from_rgb(255, 210, 80)
    } else {
        Color32::from_rgb(180, 95, 0)
    };
    let name_text = |base: RichText| -> RichText {
        if matched {
            base.color(highlight).strong()
        } else {
            base
        }
    };
    if node.is_group {
        let header = egui::CollapsingHeader::new(name_text(RichText::new(&node.name).strong()))
            .id_salt(&node.path)
            .default_open(node.path == "/entry")
            .open(if filter { Some(true) } else { None });
        let resp = header.show(ui, |ui| {
            for &c in &node.children {
                show_node(ui, tree, file, c, filter, ancestor_matched || matched,
                          node_match, subtree_match, selected, others, clicked);
            }
        });
        if resp.header_response.clicked() {
            let ctrl = ui.input(|i| i.modifiers.ctrl || i.modifiers.command);
            *clicked = Some((file, idx, ctrl));
        }
    } else {
        let dims = if node.shape.is_empty() || node.shape.iter().product::<usize>() <= 1 {
            String::new()
        } else {
            format!(
                "  [{}]",
                node.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("×")
            )
        };
        let other = others.iter().position(|&s| s == (file, idx));
        let mut text = RichText::new(format!(
            "{}{dims}{}",
            node.name,
            other.map(|k| format!("  [{}]", k + 2)).unwrap_or_default()
        ))
        .monospace();
        if other.is_some() {
            text = text
                .color(if ui.visuals().dark_mode {
                    Color32::from_rgb(110, 200, 255)
                } else {
                    Color32::from_rgb(0, 90, 180)
                })
                .strong();
        }
        if ui.selectable_label(selected == Some((file, idx)), name_text(text)).clicked() {
            let ctrl = ui.input(|i| i.modifiers.ctrl || i.modifiers.command);
            *clicked = Some((file, idx, ctrl));
        }
    }
}
