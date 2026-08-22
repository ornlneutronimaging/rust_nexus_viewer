//! NeXus file viewer: tree of the file content on the left, values/plots on
//! the right, with a search bar to narrow the tree down to matching PVs.

mod h5io;
mod recent;
mod theme;

use eframe::egui;
use egui::{Color32, RichText};
use egui_plot::{Legend, Line, Plot, PlotPoints, Points};
use h5io::{format_num, Tree, Value};
use std::path::{Path, PathBuf};

fn main() -> eframe::Result {
    let arg_path = std::env::args().nth(1).map(PathBuf::from);
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
            let mut app = App::default();
            app.recent = recent::load();
            if let Some(p) = &arg_path {
                app.open_file(p, &cc.egui_ctx);
            }
            Ok(Box::new(app))
        }),
    )
}

/// Everything derived from a selected dataset.
struct Loaded {
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
    key: (usize, usize, bool),
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
    tree: Option<Tree>,
    error: Option<String>,
    search: String,
    case_sensitive: bool,
    /// (query, case_sensitive) the match arrays were computed for.
    computed_for: Option<(String, bool)>,
    node_match: Vec<bool>,
    subtree_match: Vec<bool>,
    match_count: usize,
    selected: Option<usize>,
    loaded: Option<Loaded>,
    /// Second selection (Ctrl+click) for compare plots.
    second: Option<usize>,
    loaded2: Option<Loaded>,
    compare_mode: CompareMode,
    swap_xy: bool,
    normalize: bool,
    xy_cache: Option<XyCache>,
    /// Most-recently opened files, newest first (persisted across restarts).
    recent: Vec<PathBuf>,
}

impl App {
    fn open_file(&mut self, path: &Path, ctx: &egui::Context) {
        match h5io::load(path) {
            Ok(tree) => {
                let name = path.file_name().map(|s| s.to_string_lossy()).unwrap_or_default();
                ctx.send_viewport_cmd(egui::ViewportCommand::Title(format!(
                    "NeXus Viewer — {name}"
                )));
                self.tree = Some(tree);
                self.error = None;
                self.selected = None;
                self.loaded = None;
                self.second = None;
                self.loaded2 = None;
                self.xy_cache = None;
                self.computed_for = None;
                recent::add(&mut self.recent, path);
            }
            Err(e) => self.error = Some(format!("{e:#}")),
        }
    }

    fn open_dialog(&mut self, ctx: &egui::Context) {
        let mut dialog = rfd::FileDialog::new()
            .add_filter("NeXus / HDF5", &["h5", "nxs", "hdf5", "nx5", "nxs.h5"])
            .add_filter("All files", &["*"]);
        // Start in the current file's directory, or the last opened one.
        let start_dir = self
            .tree
            .as_ref()
            .map(|t| t.file_path.as_path())
            .or_else(|| self.recent.first().map(|p| p.as_path()))
            .and_then(Path::parent);
        if let Some(dir) = start_dir {
            dialog = dialog.set_directory(dir);
        }
        if let Some(path) = dialog.pick_file() {
            self.open_file(&path, ctx);
        }
    }

    /// Recompute the search-match arrays if the query changed.
    fn update_filter(&mut self) {
        let key = (self.search.clone(), self.case_sensitive);
        if self.computed_for.as_ref() == Some(&key) {
            return;
        }
        let Some(tree) = &self.tree else { return };
        let n = tree.nodes.len();
        self.node_match = vec![false; n];
        self.subtree_match = vec![false; n];
        self.match_count = 0;
        if !self.search.is_empty() {
            let needle = if self.case_sensitive {
                self.search.clone()
            } else {
                self.search.to_lowercase()
            };
            for i in 1..n {
                let node = &tree.nodes[i];
                let hay = if self.case_sensitive { &node.name } else { &node.name_lower };
                if hay.contains(&needle) {
                    self.node_match[i] = true;
                    self.match_count += 1;
                }
                self.subtree_match[i] = self.node_match[i];
            }
            // Children always have larger indices than their parent, so one
            // reverse pass propagates matches up to the root.
            for i in (1..n).rev() {
                if self.subtree_match[i] {
                    let p = tree.nodes[i].parent;
                    self.subtree_match[p] = true;
                }
            }
        }
        self.computed_for = Some(key);
    }

    fn filter_active(&self) -> bool {
        !self.search.is_empty()
    }

    fn select(&mut self, idx: usize) {
        self.selected = Some(idx);
        self.xy_cache = None;
        if let Some(tree) = &self.tree {
            self.loaded = Some(load_node(tree, idx));
        }
    }

    /// Ctrl+click: set (or toggle off) the second PV of a compare plot.
    fn select_second(&mut self, idx: usize) {
        self.xy_cache = None;
        if self.second == Some(idx) {
            self.second = None;
            self.loaded2 = None;
        } else if let Some(tree) = &self.tree {
            self.second = Some(idx);
            self.loaded2 = Some(load_node(tree, idx));
        }
    }

    /// (Re)build the x-y pairing cache when the compare selection changed
    /// (interpolating hundreds of thousands of points every frame would not
    /// keep the UI at 60 fps).
    fn ensure_xy_cache(&mut self) {
        let (Some(a), Some(b)) = (&self.loaded, &self.loaded2) else {
            return;
        };
        if !a.plottable() || !b.plottable() {
            return;
        }
        let key = (a.node, b.node, self.swap_xy);
        if self.xy_cache.as_ref().is_some_and(|c| c.key == key) {
            return;
        }
        let cache = build_xy(a, b, self.swap_xy);
        self.xy_cache = Some(cache);
    }
}

fn load_node(tree: &Tree, idx: usize) -> Loaded {
    let node = &tree.nodes[idx];
    // For DASlogs PVs the interesting name is the group, not `value`.
    let label = if node.name == "value" {
        tree.nodes[node.parent].name.clone()
    } else {
        node.name.clone()
    };
    let mut loaded = Loaded {
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
        // Open a file dropped onto the window.
        let dropped: Option<PathBuf> = ctx.input(|i| {
            i.raw.dropped_files.first().and_then(|f| f.path.clone())
        });
        if let Some(path) = dropped {
            self.open_file(&path, &ctx);
        }

        self.top_bar(ui, &ctx);
        self.update_filter();
        self.left_tree(ui);
        self.right_panel(ui);
    }
}

impl App {
    fn top_bar(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        egui::Panel::top("top").show(ui, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if ui.button("📂 Open…").clicked() {
                    self.open_dialog(ctx);
                }
                let mut reopen: Option<PathBuf> = None;
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
                                .on_hover_text(p.display().to_string())
                                .on_disabled_hover_text(format!(
                                    "File not found: {}",
                                    p.display()
                                ));
                            if resp.clicked() {
                                reopen = Some(p.clone());
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
                if let Some(p) = reopen {
                    self.open_file(&p, ctx);
                }
                if clear_recent {
                    recent::clear(&mut self.recent);
                }
                ui.separator();
                ui.label("🔍");
                let resp = ui.add(
                    egui::TextEdit::singleline(&mut self.search)
                        .hint_text("Search PVs / keywords…")
                        .desired_width(320.0),
                );
                if resp.changed() {
                    // Filter arrays refresh in update_filter().
                }
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
                    let color = if self.match_count == 0 {
                        ui.visuals().error_fg_color
                    } else if ui.visuals().dark_mode {
                        Color32::LIGHT_GREEN
                    } else {
                        Color32::DARK_GREEN
                    };
                    ui.label(
                        RichText::new(format!("{} match(es)", self.match_count)).color(color),
                    );
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    theme::toggle_button(ui);
                    if let Some(tree) = &self.tree {
                        ui.label(
                            RichText::new(format!(
                                "{}   ({} groups, {} datasets)",
                                tree.file_path.display(),
                                tree.n_groups,
                                tree.n_datasets
                            ))
                            .weak(),
                        );
                    }
                });
            });
            ui.add_space(4.0);
        });
    }

    fn left_tree(&mut self, ui: &mut egui::Ui) {
        egui::Panel::left("tree")
            .resizable(true)
            .default_size(430.0)
            .show(ui, |ui| {
                let Some(tree) = &self.tree else {
                    ui.add_space(20.0);
                    ui.label("Open a NeXus file (📂 or 🕘 Recent above, or drag & drop).");
                    return;
                };
                ui.label(
                    RichText::new("Ctrl+click a second PV to plot one vs the other")
                        .weak()
                        .small(),
                );
                ui.separator();
                let filter = self.filter_active();
                if filter && self.match_count == 0 {
                    ui.add_space(10.0);
                    ui.label(RichText::new("No PV matches the search.").weak());
                    return;
                }
                let roots = tree.nodes[0].children.clone();
                let mut clicked: Option<(usize, bool)> = None;
                egui::ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
                    for r in roots {
                        show_node(ui, tree, r, filter, false, &self.node_match,
                                  &self.subtree_match, self.selected, self.second, &mut clicked);
                    }
                });
                if let Some((idx, ctrl)) = clicked {
                    // Clicking a group header selects its `value` dataset when
                    // it has one (typical for DASlogs PVs); clicking a dataset
                    // selects it directly. Ctrl+click picks the second PV of a
                    // compare plot.
                    let target = if tree.nodes[idx].is_group {
                        tree.child_named(idx, "value")
                    } else {
                        Some(idx)
                    };
                    if let Some(t) = target {
                        if ctrl {
                            self.select_second(t);
                        } else if self.selected != Some(t) {
                            self.select(t);
                        }
                    }
                }
            });
    }

    fn right_panel(&mut self, ui: &mut egui::Ui) {
        self.ensure_xy_cache();
        let mut mode = self.compare_mode;
        let mut swap = self.swap_xy;
        let mut norm = self.normalize;
        let mut clear_second = false;
        egui::CentralPanel::default().show(ui, |ui| {
            if let Some(err) = &self.error {
                let color = ui.visuals().error_fg_color;
                ui.colored_label(color, err);
                return;
            }
            let (Some(tree), Some(loaded)) = (&self.tree, &self.loaded) else {
                ui.add_space(20.0);
                ui.label(
                    RichText::new(
                        "Select a dataset in the tree to view it.\n\
                         Ctrl+click a second one to plot one PV against another.",
                    )
                    .weak(),
                );
                return;
            };
            let node = &tree.nodes[loaded.node];

            ui.horizontal(|ui| {
                if self.loaded2.is_some() {
                    ui.label(RichText::new("[1]").strong());
                }
                ui.label(RichText::new(&node.path).monospace().strong());
                if ui.small_button("📋").on_hover_text("Copy path").clicked() {
                    ui.ctx().copy_text(node.path.clone());
                }
            });
            if let Some(l2) = &self.loaded2 {
                let node2 = &tree.nodes[l2.node];
                ui.horizontal(|ui| {
                    ui.label(RichText::new("[2]").strong());
                    ui.label(RichText::new(&node2.path).monospace().strong());
                    if ui.small_button("📋").on_hover_text("Copy path").clicked() {
                        ui.ctx().copy_text(node2.path.clone());
                    }
                    if ui.small_button("✖").on_hover_text("Remove second selection").clicked() {
                        clear_second = true;
                    }
                });
            }

            // Compare view when a second plottable PV is selected.
            if let Some(l2) = &self.loaded2 {
                if loaded.plottable() && l2.plottable() {
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label("Plot:");
                        ui.selectable_value(&mut mode, CompareMode::Xy, "1 vs 2");
                        ui.selectable_value(&mut mode, CompareMode::Overlay, "both as y");
                        match mode {
                            CompareMode::Xy => {
                                if ui.button("swap axes").clicked() {
                                    swap = !swap;
                                }
                            }
                            CompareMode::Overlay => {
                                ui.checkbox(&mut norm, "normalize [0–1]");
                            }
                        }
                    });
                    if let Some(s) = &loaded.stats {
                        stats_line(ui, "[1] ", s);
                    }
                    if let Some(s) = &l2.stats {
                        stats_line(ui, "[2] ", s);
                    }
                    ui.add_space(4.0);
                    match mode {
                        CompareMode::Xy => {
                            let key = (loaded.node, l2.node, swap);
                            draw_xy(ui, self.xy_cache.as_ref(), key);
                        }
                        CompareMode::Overlay => draw_overlay(ui, loaded, l2, norm),
                    }
                    return;
                }
                let color = ui.visuals().warn_fg_color;
                ui.colored_label(
                    color,
                    "The 2nd selection has no plottable 1-D array — showing the 1st only.",
                );
            }
            ui.label(
                RichText::new(format!("dtype: {}   shape: {:?}", node.dtype, node.shape)).weak(),
            );

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
                Value::Empty => {
                    ui.label(RichText::new("(empty dataset)").weak());
                }
                Value::Unsupported(msg) => {
                    let color = ui.visuals().error_fg_color;
                    ui.colored_label(color, msg);
                }
                Value::Strings(items) => show_strings(ui, items),
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
                        show_plot(ui, loaded);
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
        });
        self.compare_mode = mode;
        self.swap_xy = swap;
        self.normalize = norm;
        if clear_second {
            self.second = None;
            self.loaded2 = None;
            self.xy_cache = None;
        }
    }
}

fn show_strings(ui: &mut egui::Ui, items: &[String]) {
    ui.add_space(8.0);
    if items.len() == 1 {
        ui.label(RichText::new(&items[0]).monospace().size(20.0));
    } else {
        ui.label(RichText::new(format!("{} strings:", items.len())).weak());
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
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

fn show_plot(ui: &mut egui::Ui, loaded: &Loaded) {
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
        .show(ui, |plot_ui| {
            plot_ui.line(line);
        });
}

/// Pair the two selected PVs into (x, y) points. PVs logged at different
/// times are paired by linearly interpolating the y PV onto the x PV's time
/// grid; series without a time axis are paired by index when lengths match.
fn build_xy(a: &Loaded, b: &Loaded, swap: bool) -> XyCache {
    let key = (a.node, b.node, swap);
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

fn draw_xy(ui: &mut egui::Ui, cache: Option<&XyCache>, key: (usize, usize, bool)) {
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
        .show(ui, |plot_ui| {
            plot_ui.points(pts);
        });
}

fn draw_overlay(ui: &mut egui::Ui, a: &Loaded, b: &Loaded, normalize: bool) {
    let x_label = if a.x_label == b.x_label {
        a.x_label.clone()
    } else {
        format!("{} / {}", a.x_label, b.x_label)
    };
    let y_label = if normalize {
        "normalized [0–1]".to_string()
    } else {
        match (&a.units, &b.units) {
            (Some(u), Some(v)) if u == v => u.clone(),
            _ => String::new(),
        }
    };
    let series = |l: &Loaded, n: &str| -> Line {
        let pts = match (normalize, &l.stats) {
            (true, Some(s)) => norm_points(&l.points, s),
            _ => l.points.clone(),
        };
        Line::new(format!("{n} {}", series_label(l)), PlotPoints::from(pts))
    };
    let la = series(a, "[1]");
    let lb = series(b, "[2]");
    Plot::new("overlay_plot")
        .legend(Legend::default())
        .x_axis_label(x_label)
        .y_axis_label(y_label)
        .allow_boxed_zoom(true)
        .show(ui, |plot_ui| {
            plot_ui.line(la);
            plot_ui.line(lb);
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
    idx: usize,
    filter: bool,
    ancestor_matched: bool,
    node_match: &[bool],
    subtree_match: &[bool],
    selected: Option<usize>,
    second: Option<usize>,
    clicked: &mut Option<(usize, bool)>,
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
                show_node(ui, tree, c, filter, ancestor_matched || matched,
                          node_match, subtree_match, selected, second, clicked);
            }
        });
        if resp.header_response.clicked() {
            let ctrl = ui.input(|i| i.modifiers.ctrl || i.modifiers.command);
            *clicked = Some((idx, ctrl));
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
        let is_second = second == Some(idx);
        let mut text = RichText::new(format!(
            "{}{dims}{}",
            node.name,
            if is_second { "  [2]" } else { "" }
        ))
        .monospace();
        if is_second {
            text = text
                .color(if ui.visuals().dark_mode {
                    Color32::from_rgb(110, 200, 255)
                } else {
                    Color32::from_rgb(0, 90, 180)
                })
                .strong();
        }
        if ui.selectable_label(selected == Some(idx), name_text(text)).clicked() {
            let ctrl = ui.input(|i| i.modifiers.ctrl || i.modifiers.command);
            *clicked = Some((idx, ctrl));
        }
    }
}
