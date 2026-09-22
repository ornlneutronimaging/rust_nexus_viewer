//! The last few NeXus files opened and the last few PVs plotted, persisted
//! across restarts.
//!
//! Both lists live under `~/.config/venus_rust_tools/`, one entry per line,
//! most recent first: `nexus_viewer_recent` holds absolute file paths and
//! `nexus_viewer_recent_pvs` holds HDF5 dataset paths (the PV's `value`
//! dataset, e.g. `/entry/DASlogs/BL10:SE:Temp/value`). Persistence is best
//! effort, like the theme preference: an unwritable home directory only
//! loses the lists, it never errors.

use std::path::{Path, PathBuf};

pub const MAX: usize = 5;
pub const MAX_PVS: usize = 20;

const FILES: &str = "nexus_viewer_recent";
const PVS: &str = "nexus_viewer_recent_pvs";

/// A list file, under `$XDG_CONFIG_HOME` (or `~/.config`).
fn list_path(name: &str) -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
        .or_else(|| {
            std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config"))
        })?;
    Some(base.join("venus_rust_tools").join(name))
}

fn load_lines(name: &str, max: usize) -> Vec<String> {
    let Some(text) = list_path(name).and_then(|p| std::fs::read_to_string(p).ok()) else {
        return Vec::new();
    };
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .take(max)
        .collect()
}

fn save_lines(name: &str, lines: impl Iterator<Item = String>) {
    let Some(path) = list_path(name) else { return };
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut text = String::new();
    for l in lines {
        text.push_str(&l);
        text.push('\n');
    }
    let _ = std::fs::write(path, text);
}

pub fn load() -> Vec<PathBuf> {
    load_lines(FILES, MAX).into_iter().map(PathBuf::from).collect()
}

/// Move `path` to the front of the list (deduplicated, capped at [`MAX`])
/// and persist it. Canonicalized so the same file reached through different
/// mounts (/SNS vs /gpfs) or relative paths only appears once.
pub fn add(list: &mut Vec<PathBuf>, path: &Path) {
    let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    list.retain(|p| p != &path);
    list.insert(0, path);
    list.truncate(MAX);
    save(list);
}

pub fn clear(list: &mut Vec<PathBuf>) {
    list.clear();
    save(list);
}

fn save(list: &[PathBuf]) {
    save_lines(FILES, list.iter().map(|p| p.to_string_lossy().into_owned()));
}

pub fn load_pvs() -> Vec<String> {
    load_lines(PVS, MAX_PVS)
}

/// Move the dataset path to the front of the PV list (deduplicated, capped
/// at [`MAX_PVS`]) and persist it.
pub fn add_pv(list: &mut Vec<String>, path: &str) {
    list.retain(|p| p != path);
    list.insert(0, path.to_string());
    list.truncate(MAX_PVS);
    save_lines(PVS, list.iter().cloned());
}

pub fn clear_pvs(list: &mut Vec<String>) {
    list.clear();
    save_lines(PVS, std::iter::empty());
}
