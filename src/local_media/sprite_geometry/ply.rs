//! Vertex-only Gaussian splats. Read the declared properties in file order,
//! including exporters' extra SH coefficients, without allocating each record.
use super::{err, gltf::trs, Splat};
use crate::Result;
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq)]
enum Scalar {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl Scalar {
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "char" | "int8" => Self::I8,
            "uchar" | "uint8" => Self::U8,
            "short" | "int16" => Self::I16,
            "ushort" | "uint16" => Self::U16,
            "int" | "int32" => Self::I32,
            "uint" | "uint32" => Self::U32,
            "float" | "float32" => Self::F32,
            "double" | "float64" => Self::F64,
            _ => return None,
        })
    }
    fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::F64 => 8,
            _ => 4,
        }
    }
    fn read(self, raw: &[u8]) -> f64 {
        // Header and exact record size were checked before enumerating records.
        match self {
            Self::I8 => raw[0] as i8 as f64,
            Self::U8 => raw[0] as f64,
            Self::I16 => i16::from_le_bytes(raw[..2].try_into().unwrap()) as f64,
            Self::U16 => u16::from_le_bytes(raw[..2].try_into().unwrap()) as f64,
            Self::I32 => i32::from_le_bytes(raw[..4].try_into().unwrap()) as f64,
            Self::U32 => u32::from_le_bytes(raw[..4].try_into().unwrap()) as f64,
            Self::F32 => f32::from_le_bytes(raw[..4].try_into().unwrap()) as f64,
            Self::F64 => f64::from_le_bytes(raw[..8].try_into().unwrap()),
        }
    }
}

#[derive(Clone, Copy)]
struct Property {
    kind: Scalar,
    ordinal: usize,
    offset: usize,
}

pub(super) fn parse(raw: &[u8]) -> Result<Vec<Splat>> {
    let mut header_len = 0;
    let mut terminated = false;
    for line in raw[..raw.len().min(65536)].split_inclusive(|&b| b == b'\n') {
        header_len += line.len();
        if line == b"end_header\n" || line == b"end_header\r\n" {
            terminated = true;
            break;
        }
    }
    if !terminated {
        return Err(err("Missing PLY header terminator."));
    }
    let header = std::str::from_utf8(&raw[..header_len]).map_err(|_| err("Invalid PLY header."))?;
    let mut binary = None;
    let mut count = None;
    let mut element = "";
    let mut stride = 0;
    let mut props = HashMap::new();
    for line in header.lines() {
        let mut words = line.split_ascii_whitespace();
        match words.next() {
            Some("format") => {
                let format = words.next();
                if words.next() != Some("1.0")
                    || !matches!(format, Some("ascii" | "binary_little_endian"))
                {
                    return Err(err(
                        "Supported Gaussian PLY formats are ASCII and binary little-endian 1.0.",
                    ));
                }
                binary = Some(format == Some("binary_little_endian"));
            }
            Some("element") => {
                element = words.next().unwrap_or("");
                let length = words
                    .next()
                    .and_then(|s| s.parse::<usize>().ok())
                    .ok_or_else(|| err("Invalid Gaussian PLY element count."))?;
                if element == "vertex" {
                    count = Some(length);
                } else if length != 0 {
                    return Err(err("Expected an uncompressed vertex-only Gaussian PLY."));
                }
            }
            Some("property") if element == "vertex" => {
                let kind = words
                    .next()
                    .and_then(Scalar::parse)
                    .ok_or_else(|| err("Unsupported Gaussian PLY property encoding."))?;
                let name = words
                    .next()
                    .ok_or_else(|| err("Missing Gaussian PLY property name."))?;
                if words.next().is_some() || props.contains_key(name) {
                    return Err(err("Unsupported Gaussian PLY property encoding."));
                }
                props.insert(
                    name,
                    Property {
                        kind,
                        ordinal: props.len(),
                        offset: stride,
                    },
                );
                stride += kind.size();
            }
            _ => {}
        }
    }
    let invalid = || {
        err("Expected Gaussian PLY positions, log scales, quaternion, opacity logit, and DC or RGB colors. Ordinary point clouds are unsupported.")
    };
    let binary = binary.ok_or_else(invalid)?;
    let count = count
        .filter(|v| *v > 0)
        .ok_or_else(|| err("Invalid Gaussian PLY vertex count."))?;
    let dc = ["f_dc_0", "f_dc_1", "f_dc_2"]
        .iter()
        .all(|s| props.contains_key(s));
    let rgb = if dc {
        ["f_dc_0", "f_dc_1", "f_dc_2"]
    } else {
        ["red", "green", "blue"]
    };
    let names = [
        "x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
        "opacity", rgb[0], rgb[1], rgb[2],
    ];
    let fields: Vec<_> = names
        .iter()
        .map(|s| props.get(s).copied().ok_or_else(invalid))
        .collect::<Result<_>>()?;
    if !dc && fields[11..].iter().any(|p| p.kind != Scalar::U8) {
        return Err(err("Gaussian RGB properties must be unsigned bytes, or supply floating-point DC coefficients."));
    }
    let data = &raw[header_len..];
    let bad_length = || err("Gaussian PLY payload does not match its declared vertex count.");
    let mut splats = Vec::new();
    if binary {
        if count.checked_mul(stride) != Some(data.len()) {
            return Err(bad_length());
        }
        for row in data.chunks_exact(stride) {
            let values = std::array::from_fn(|i| fields[i].kind.read(&row[fields[i].offset..]));
            splats.push(splat(values, dc)?);
        }
    } else {
        let rows = std::str::from_utf8(data).map_err(|_| err("Invalid ASCII PLY vertex."))?;
        let mut columns = Vec::with_capacity(props.len());
        for row in rows.trim().lines() {
            columns.clear();
            columns.extend(row.split_ascii_whitespace());
            if columns.len() != props.len() {
                return Err(err("Invalid ASCII PLY vertex."));
            }
            let mut values = [0.; 14];
            for (i, p) in fields.iter().enumerate() {
                values[i] = columns[p.ordinal]
                    .parse()
                    .map_err(|_| err("Invalid ASCII PLY vertex."))?;
            }
            splats.push(splat(values, dc)?);
        }
        if splats.len() != count {
            return Err(bad_length());
        }
    }
    Ok(splats)
}

fn splat(values: [f64; 14], dc: bool) -> Result<Splat> {
    if !values.iter().all(|v| v.is_finite()) {
        return Err(err("Non-finite Gaussian PLY attribute."));
    }
    let center = [values[0], values[1], values[2]];
    let scales = [values[3].exp(), values[4].exp(), values[5].exp()];
    let norm = values[6..10]
        .iter()
        .fold(0_f64, |length, x| length.hypot(*x));
    if norm == 0. || !norm.is_finite() || scales.iter().any(|v| !v.is_finite() || *v <= 0.) {
        return Err(err("Invalid Gaussian scale or rotation."));
    }
    let rotation = trs(
        [0.; 3],
        [
            values[7] / norm,
            values[8] / norm,
            values[9] / norm,
            values[6] / norm,
        ],
        [1.; 3],
    );
    let extent = std::array::from_fn(|k| {
        3. * (rotation[k] * scales[0])
            .hypot(rotation[k + 4] * scales[1])
            .hypot(rotation[k + 8] * scales[2])
    });
    let color = std::array::from_fn(|k| {
        if dc {
            255. * (0.5 + 0.2820947918 * values[k + 11])
        } else {
            values[k + 11]
        }
    });
    Ok(Splat {
        center,
        scales,
        rotation,
        extent,
        color,
        alpha: 1. / (1. + (-values[10]).exp()),
    })
}
