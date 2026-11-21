//! Decode strided attributes and sparse overlays, including zero-based morph targets.
use super::*;

fn component_size(kind: usize) -> Result<usize> {
    match kind {
        5120 | 5121 => Ok(1),
        5122 | 5123 => Ok(2),
        5125 | 5126 => Ok(4),
        _ => Err(err("Unsupported glTF accessor component type.")),
    }
}

fn scalar(bytes: &[u8], kind: usize, normalized: bool) -> Result<f64> {
    let (v, max) = match kind {
        5120 => ((bytes[0] as i8) as f64, 127.),
        5121 => (bytes[0] as f64, 255.),
        5122 => (i16::from_le_bytes([bytes[0], bytes[1]]) as f64, 32767.),
        5123 => (u16::from_le_bytes([bytes[0], bytes[1]]) as f64, 65535.),
        5125 => (le32(bytes) as f64, 4294967295.),
        _ => (f32::from_bits(le32(bytes)) as f64, 1.),
    };
    if !v.is_finite() {
        return Err(err("Non-finite glTF attribute."));
    }
    Ok(if normalized { (v / max).max(-1.) } else { v })
}

impl Document<'_> {
    fn rows<'a>(
        &'a self,
        spec: &Value,
        rows: usize,
        width: usize,
        strided: bool,
    ) -> Result<impl Iterator<Item = &'a [u8]>> {
        let (data, view) = self.view(&spec["bufferView"])?;
        let offset = optional_index(&spec["byteOffset"], 0)?;
        let stride = if strided {
            optional_index(&view["byteStride"], width)?
        } else {
            width
        };
        let end = rows
            .checked_sub(1)
            .and_then(|n| n.checked_mul(stride))
            .and_then(|n| n.checked_add(offset))
            .and_then(|n| n.checked_add(width));
        if stride < width || end.is_none_or(|n| n > data.len()) {
            return Err(err("Invalid glTF accessor bounds or count."));
        }
        Ok((0..rows).map(move |i| &data[offset + i * stride..offset + i * stride + width]))
    }

    pub(super) fn accessor<const N: usize>(
        &self,
        id: &Value,
        min_channels: usize,
        count: Option<usize>,
    ) -> Result<Vec<[f64; N]>> {
        let spec = get(&self.json["accessors"], id)?;
        let n = match spec["type"].as_str() {
            Some("SCALAR") => 1,
            Some("VEC2") => 2,
            Some("VEC3") => 3,
            Some("VEC4") => 4,
            Some("MAT4") => 16,
            _ => 0,
        };
        if !(min_channels..=N).contains(&n) {
            return Err(err("Unsupported glTF accessor type."));
        }
        let kind = index(&spec["componentType"])?;
        let size = component_size(kind)?;
        let normalized = spec["normalized"].as_bool().unwrap_or(false);
        let rows = index(&spec["count"])?;
        if rows == 0 || count.is_some_and(|c| c != rows) {
            return Err(err("Invalid glTF accessor bounds or count."));
        }
        let decode = |bytes: &[u8]| -> Result<[f64; N]> {
            let mut row = [1.; N]; // implicit alpha for RGB colors
            for (dst, src) in row.iter_mut().zip(bytes.chunks_exact(size)) {
                *dst = scalar(src, kind, normalized)?;
            }
            Ok(row)
        };
        let mut out = if spec["bufferView"].is_null() {
            if rows > (256 << 20) / std::mem::size_of::<[f64; N]>() {
                return Err(err("glTF zero-initialized accessor is too large."));
            }
            let mut zero = [1.; N];
            zero[..n].fill(0.);
            vec![zero; rows]
        } else {
            self.rows(spec, rows, size * n, true)?
                .map(decode)
                .collect::<Result<Vec<_>>>()?
        };
        if let Some(sparse) = spec.get("sparse") {
            let patches = index(&sparse["count"])?;
            let inds = &sparse["indices"];
            let index_type = index(&inds["componentType"])?;
            if patches == 0 || patches > rows || !matches!(index_type, 5121 | 5123 | 5125) {
                return Err(err("Invalid glTF sparse accessor indices."));
            }
            let indices = self.rows(inds, patches, component_size(index_type)?, false)?;
            let vals = self.rows(&sparse["values"], patches, size * n, false)?;
            let mut prior = None;
            for (ind, v) in indices.zip(vals) {
                let i = scalar(ind, index_type, false)? as usize;
                if i >= rows || prior.is_some_and(|p| p >= i) {
                    return Err(err("Invalid glTF sparse accessor indices."));
                }
                out[i] = decode(v)?;
                prior = Some(i);
            }
        }
        Ok(out)
    }
}
