//! Exporting datasets to plain files: 2-D arrays as 32-bit float grey TIFF
//! images (ImageJ / tifffile readable), 1-D arrays as CSV.

use anyhow::{Context, Result};
use std::io::{BufWriter, Write};
use std::path::Path;

/// Write a row-major `rows × cols` array as one 32-bit float TIFF image.
pub fn write_tiff(path: &Path, rows: usize, cols: usize, data: &[f64]) -> Result<()> {
    anyhow::ensure!(
        rows * cols == data.len(),
        "shape {rows}×{cols} does not match {} values",
        data.len()
    );
    let (w, h) = (u32::try_from(cols)?, u32::try_from(rows)?);
    let pixels: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let file = std::fs::File::create(path)
        .with_context(|| format!("cannot create {}", path.display()))?;
    let mut enc = tiff::encoder::TiffEncoder::new(BufWriter::new(file))?;
    enc.write_image::<tiff::encoder::colortype::Gray32Float>(w, h, &pixels)
        .with_context(|| format!("cannot write {}", path.display()))?;
    Ok(())
}

/// Write columns of equal length as CSV with the given header line. Values
/// are written with full precision (shortest round-trip representation).
pub fn write_csv(path: &Path, header: &[&str], columns: &[&[f64]]) -> Result<()> {
    let n = columns.first().map_or(0, |c| c.len());
    anyhow::ensure!(
        columns.iter().all(|c| c.len() == n),
        "columns have different lengths"
    );
    let file = std::fs::File::create(path)
        .with_context(|| format!("cannot create {}", path.display()))?;
    let mut out = BufWriter::new(file);
    writeln!(out, "{}", header.join(","))?;
    for i in 0..n {
        let row: Vec<String> = columns.iter().map(|c| c[i].to_string()).collect();
        writeln!(out, "{}", row.join(","))?;
    }
    out.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiff_round_trips() {
        let dir = std::env::temp_dir().join(format!("nexus_viewer_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.tiff");
        let data: Vec<f64> = vec![1.5, -2.0, f64::NAN, 4.0, 5.0, 6.0];
        write_tiff(&path, 2, 3, &data).unwrap();
        let mut dec = tiff::decoder::Decoder::new(std::fs::File::open(&path).unwrap()).unwrap();
        assert_eq!(dec.dimensions().unwrap(), (3, 2));
        match dec.read_image().unwrap() {
            tiff::decoder::DecodingResult::F32(v) => {
                assert_eq!(v.len(), 6);
                assert_eq!(v[0], 1.5);
                assert_eq!(v[1], -2.0);
                assert!(v[2].is_nan());
                assert_eq!(v[5], 6.0);
            }
            other => panic!("unexpected sample type {other:?}"),
        }
        assert!(write_tiff(&path, 2, 2, &data).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn csv_has_header_and_rows() {
        let dir = std::env::temp_dir().join(format!("nexus_viewer_csv_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.csv");
        write_csv(&path, &["index", "angles (degree)"], &[&[0.0, 1.0], &[0.1, 180.0]]).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text, "index,angles (degree)\n0,0.1\n1,180\n");
        assert!(write_csv(&path, &["a", "b"], &[&[1.0], &[1.0, 2.0]]).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
