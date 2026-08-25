//! Loading a NeXus/HDF5 file into a browsable tree, plus dataset/attribute reading.

use anyhow::{Context, Result};
use hdf5_metno as h5;
use h5::types::TypeDescriptor;
use std::path::{Path, PathBuf};

/// Refuse to load datasets bigger than this many elements (safety valve).
const MAX_ELEMENTS: usize = 50_000_000;

pub struct Node {
    pub name: String,
    pub name_lower: String,
    pub path: String,
    pub parent: usize,
    pub is_group: bool,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub children: Vec<usize>,
}

pub struct Tree {
    pub file: h5::File,
    pub file_path: PathBuf,
    /// Arena of nodes; index 0 is the root group "/". A parent always has a
    /// smaller index than its children.
    pub nodes: Vec<Node>,
    pub n_groups: usize,
    pub n_datasets: usize,
}

impl Tree {
    pub fn child_named(&self, group_idx: usize, name: &str) -> Option<usize> {
        self.nodes[group_idx]
            .children
            .iter()
            .copied()
            .find(|&c| self.nodes[c].name == name)
    }

    /// Node with exactly this HDF5 path, if the file has one (used to find
    /// the same PV in other open files).
    pub fn node_by_path(&self, path: &str) -> Option<usize> {
        self.nodes.iter().position(|n| n.path == path)
    }
}

pub fn load(path: &Path) -> Result<Tree> {
    let file = h5::File::open(path)
        .with_context(|| format!("cannot open {} as HDF5", path.display()))?;
    let mut nodes = vec![Node {
        name: "/".into(),
        name_lower: "/".into(),
        path: "/".into(),
        parent: 0,
        is_group: true,
        shape: vec![],
        dtype: String::new(),
        children: vec![],
    }];
    walk(&file, 0, &mut nodes)?;
    let n_groups = nodes.iter().filter(|n| n.is_group).count() - 1; // minus root
    let n_datasets = nodes.len() - n_groups - 1;
    Ok(Tree { file, file_path: path.to_path_buf(), nodes, n_groups, n_datasets })
}

fn walk(group: &h5::Group, parent: usize, nodes: &mut Vec<Node>) -> Result<()> {
    let mut names = group.member_names().unwrap_or_default();
    names.sort_by_key(|n| n.to_lowercase());
    for name in names {
        let parent_path = &nodes[parent].path;
        let path = if parent_path == "/" {
            format!("/{name}")
        } else {
            format!("{parent_path}/{name}")
        };
        if let Ok(sub) = group.group(&name) {
            let idx = nodes.len();
            nodes.push(Node {
                name_lower: name.to_lowercase(),
                name,
                path,
                parent,
                is_group: true,
                shape: vec![],
                dtype: String::new(),
                children: vec![],
            });
            nodes[parent].children.push(idx);
            walk(&sub, idx, nodes)?;
        } else if let Ok(ds) = group.dataset(&name) {
            let dtype = ds
                .dtype()
                .and_then(|d| d.to_descriptor())
                .map(|td| dtype_name(&td))
                .unwrap_or_else(|_| "?".into());
            let idx = nodes.len();
            nodes.push(Node {
                name_lower: name.to_lowercase(),
                name,
                path,
                parent,
                is_group: false,
                shape: ds.shape(),
                dtype,
                children: vec![],
            });
            nodes[parent].children.push(idx);
        }
    }
    Ok(())
}

pub fn dtype_name(td: &TypeDescriptor) -> String {
    use TypeDescriptor::*;
    match td {
        Integer(sz) => format!("int{}", 8 * *sz as usize),
        Unsigned(sz) => format!("uint{}", 8 * *sz as usize),
        Float(sz) => format!("float{}", 8 * *sz as usize),
        Boolean => "bool".into(),
        Enum(_) => "enum".into(),
        Compound(_) => "compound".into(),
        FixedArray(inner, n) => format!("{}[{n}]", dtype_name(inner)),
        FixedAscii(n) | FixedUnicode(n) => format!("str[{n}]"),
        VarLenAscii | VarLenUnicode => "str".into(),
        VarLenArray(inner) => format!("vlen<{}>", dtype_name(inner)),
        _ => "other".into(),
    }
}

fn is_string(td: &TypeDescriptor) -> bool {
    matches!(
        td,
        TypeDescriptor::FixedAscii(_)
            | TypeDescriptor::FixedUnicode(_)
            | TypeDescriptor::VarLenAscii
            | TypeDescriptor::VarLenUnicode
    )
}

fn is_numeric(td: &TypeDescriptor) -> bool {
    matches!(
        td,
        TypeDescriptor::Integer(_)
            | TypeDescriptor::Unsigned(_)
            | TypeDescriptor::Float(_)
            | TypeDescriptor::Boolean
            | TypeDescriptor::Enum(_)
    )
}

pub enum Value {
    Empty,
    Strings(Vec<String>),
    Numeric { data: Vec<f64>, shape: Vec<usize> },
    Unsupported(String),
}

/// Read the full content of a dataset or attribute.
pub fn read_container(c: &h5::Container) -> Value {
    let size = c.size();
    if size == 0 {
        return Value::Empty;
    }
    if size > MAX_ELEMENTS {
        return Value::Unsupported(format!(
            "dataset has {size} elements — too large to load"
        ));
    }
    let td = match c.dtype().and_then(|d| d.to_descriptor()) {
        Ok(td) => td,
        Err(e) => return Value::Unsupported(format!("cannot read datatype: {e}")),
    };
    if is_string(&td) {
        match read_strings(c, &td) {
            Ok(v) => Value::Strings(v),
            Err(e) => Value::Unsupported(format!("string read failed: {e}")),
        }
    } else if is_numeric(&td) {
        match c.read_raw::<f64>() {
            Ok(data) => Value::Numeric { data, shape: c.shape() },
            Err(e) => Value::Unsupported(format!("numeric read failed: {e}")),
        }
    } else {
        Value::Unsupported(format!("unsupported datatype: {}", dtype_name(&td)))
    }
}

/// Fixed-length strings longer than this are not supported (they would blow
/// up the read buffer, since HDF5 only converts fixed -> fixed strings).
const MAX_FIXED_STR: usize = 4096;

fn read_strings(c: &h5::Container, td: &TypeDescriptor) -> Result<Vec<String>> {
    use TypeDescriptor::*;
    // HDF5 has no fixed -> variable-length string conversion path, so fixed
    // strings are read into a generous fixed-size buffer instead.
    match td {
        VarLenAscii => Ok(c.read_raw::<h5::types::VarLenAscii>()?.iter().map(|s| s.to_string()).collect()),
        VarLenUnicode => Ok(c.read_raw::<h5::types::VarLenUnicode>()?.iter().map(|s| s.to_string()).collect()),
        FixedAscii(n) if *n <= MAX_FIXED_STR => {
            Ok(c.read_raw::<h5::types::FixedAscii<MAX_FIXED_STR>>()?
                .iter()
                .map(|s| s.to_string())
                .collect())
        }
        FixedUnicode(n) if *n <= MAX_FIXED_STR => {
            Ok(c.read_raw::<h5::types::FixedUnicode<MAX_FIXED_STR>>()?
                .iter()
                .map(|s| s.to_string())
                .collect())
        }
        _ => anyhow::bail!("string too long ({} bytes)", dtype_name(td)),
    }
}

/// All attributes of the object at `path`, with values rendered as short text.
pub fn read_attributes(file: &h5::File, path: &str) -> Vec<(String, String)> {
    // `path` may name a group or a dataset; both deref to Location.
    if let Ok(g) = file.group(path) {
        attributes_of(&g)
    } else if let Ok(d) = file.dataset(path) {
        attributes_of(&d)
    } else {
        Vec::new()
    }
}

fn attributes_of(loc: &h5::Location) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let Ok(names) = loc.attr_names() else { return out };
    for name in names {
        let text = match loc.attr(&name) {
            Ok(attr) => format_value_short(&read_container(&attr), 16),
            Err(e) => format!("<{e}>"),
        };
        out.push((name, text));
    }
    out
}

/// Compact one-line rendering, used for attributes.
pub fn format_value_short(v: &Value, max_items: usize) -> String {
    match v {
        Value::Empty => "(empty)".into(),
        Value::Strings(s) => s
            .iter()
            .take(max_items)
            .cloned()
            .collect::<Vec<_>>()
            .join(", "),
        Value::Numeric { data, .. } => {
            let mut txt = data
                .iter()
                .take(max_items)
                .map(|x| format_num(*x))
                .collect::<Vec<_>>()
                .join(", ");
            if data.len() > max_items {
                txt.push_str(", …");
            }
            txt
        }
        Value::Unsupported(msg) => format!("<{msg}>"),
    }
}

pub fn format_num(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else if v != 0.0 && (v.abs() >= 1e6 || v.abs() < 1e-4) {
        format!("{v:.6e}")
    } else {
        format!("{v:.6}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "/SNS/VENUS/IPTS-38715/nexus/VENUS_26871.nxs.h5";

    #[test]
    fn load_sample_file() {
        let tree = load(Path::new(SAMPLE)).expect("load sample");
        assert!(tree.nodes.len() > 100, "expected many nodes, got {}", tree.nodes.len());
        // A known PV group must be present.
        assert!(tree.nodes.iter().any(|n| n.path == "/entry/DASlogs/AcqFreq/value"));

        // Fixed-length string scalar.
        let ds = tree.file.dataset("/entry/title").unwrap();
        match read_container(&ds) {
            Value::Strings(s) => {
                println!("title = {s:?}");
                assert_eq!(s.len(), 1);
            }
            _ => panic!("title should read as strings"),
        }

        // u32 array -> f64 conversion.
        let ds = tree.file.dataset("/entry/DASlogs/AcqFreq/value").unwrap();
        match read_container(&ds) {
            Value::Numeric { data, .. } => println!("AcqFreq value = {data:?}"),
            Value::Unsupported(m) => panic!("AcqFreq value unsupported: {m}"),
            _ => panic!("AcqFreq value should be numeric"),
        }

        // f64 time array + attributes.
        let ds = tree.file.dataset("/entry/DASlogs/AcqFreq/time").unwrap();
        assert!(matches!(read_container(&ds), Value::Numeric { .. }));
        let attrs = read_attributes(&tree.file, "/entry/DASlogs/AcqFreq/time");
        println!("time attrs = {attrs:?}");
        assert!(attrs.iter().any(|(k, _)| k == "units"));

        // A larger PV series.
        let ds = tree.file.dataset("/entry/DASlogs/proton_charge/value");
        if let Ok(ds) = ds {
            match read_container(&ds) {
                Value::Numeric { data, .. } => println!("proton_charge n = {}", data.len()),
                Value::Unsupported(m) => panic!("proton_charge unsupported: {m}"),
                _ => {}
            }
        }
    }
}
