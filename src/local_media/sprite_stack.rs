//! Render horizontal stacks of square sprite tiles with an orthographic camera.
use super::sprite_codecs::{decode_image, encode_turntable};
use super::sprite_geometry::slice_model;
use super::{
    bytes_to_data_url, done, encode_png_rgba, err, input_text, js_number, url_bytes, PngImage,
};
use crate::{Node, NodeOutcome, Result};
use serde_json::{json, Map, Value};
use std::collections::HashMap;

pub(super) fn number(
    fields: &Map<String, Value>,
    key: &str,
    default: f64,
    min: f64,
    max: f64,
    integer: bool,
) -> Result<f64> {
    let field = fields.get(key);
    let value = match field {
        None | Some(Value::Null) => default,
        Some(Value::String(s)) if s.is_empty() => default,
        _ => js_number(field).unwrap_or(f64::NAN),
    };
    if !(min..=max).contains(&value) || (integer && value.fract() != 0.0) {
        let kind = if integer { "an integer " } else { "" };
        return Err(err(format!("{key} must be {kind}between {min} and {max}.")));
    }
    Ok(value)
}

pub(super) fn flag(fields: &Map<String, Value>, key: &str, default: bool) -> bool {
    match fields.get(key) {
        None | Some(Value::Null) => default,
        Some(v) => crate::flag_is_true(Some(v)),
    }
}

/// Nontransparent bounds in local tile coordinates, recentered around the rotation origin.
struct Layer {
    index: usize,
    x: [f64; 2],
    y: [f64; 2],
}

struct View<'a> {
    sheet: &'a PngImage,
    layers: Vec<Layer>,
    layer_count: usize,
    tile: usize,
    width: u32,
    height: u32,
    scale: f64,
    squash: f64,
    rise: f64,
}

impl<'a> View<'a> {
    fn new(sheet: &'a PngImage, fields: &Map<String, Value>) -> Result<Self> {
        let tile = sheet.height as usize;
        if !(16..=128).contains(&tile)
            || sheet.width as usize % tile != 0
            || !(8..=128).contains(&(sheet.width as usize / tile))
        {
            return Err(err(
                "Expected a horizontal Slice 3D sheet with 8–128 square tiles of 16–128 pixels.",
            ));
        }
        let layer_count = sheet.width as usize / tile;
        let pitch = number(fields, "pitch", 30., 0., 90., false)? * std::f64::consts::PI / 180.;
        let scale = number(fields, "scale", 3., 1., 8., false)?;
        let spacing =
            number(fields, "spacing", 1., 0., 4., false)? * (tile - 2) as f64 / layer_count as f64;
        let rise = spacing * pitch.cos();
        let squash = pitch.sin().max(1. / tile as f64);
        let floor = tile as f64 * std::f64::consts::SQRT_2;
        let even_size = |extent: f64| 2 * ((extent * scale + 4.) / 2.).ceil() as u32;
        let width = even_size(floor);
        let height = even_size(floor * squash + rise * (layer_count - 1) as f64);
        let mut layers = Vec::with_capacity(layer_count);
        for index in 0..layer_count {
            let (mut left, mut top, mut right, mut bottom) = (tile, tile, 0, 0);
            for y in 0..tile {
                let start = 4 * (y * sheet.width as usize + index * tile);
                let row = &sheet.rgba[start..start + 4 * tile];
                for (x, pixel) in row.chunks_exact(4).enumerate() {
                    if pixel[3] != 0 {
                        left = left.min(x);
                        top = top.min(y);
                        right = right.max(x + 1);
                        bottom = bottom.max(y + 1);
                    }
                }
            }
            if left < right {
                let half = tile as f64 / 2.;
                layers.push(Layer {
                    index,
                    x: [left as f64 - half, right as f64 - half],
                    y: [top as f64 - half, bottom as f64 - half],
                });
            }
        }
        Ok(Self {
            sheet,
            layers,
            layer_count,
            tile,
            width,
            height,
            scale,
            squash,
            rise,
        })
    }

    fn frame(&self, yaw: f64) -> PngImage {
        let (w, h) = (self.width as usize, self.height as usize);
        let mut rgba = vec![0; w * h * 4];
        let (sin, cos) = (yaw * std::f64::consts::PI / 180.).sin_cos();
        let center_x = w as f64 / 2.;
        let tile_center = self.tile as f64 / 2.;
        // The x terms of the inverse rotation are shared by every row and layer.
        let column_terms: Vec<_> = (0..w)
            .map(|x| {
                let dx = (x as f64 + 0.5 - center_x) / self.scale;
                (cos * dx, -sin * dx)
            })
            .collect();
        for layer in &self.layers {
            let center_y = h as f64 / 2.
                + ((self.layer_count - 1) as f64 / 2. - layer.index as f64)
                    * self.rise
                    * self.scale;
            // A transparent layer or a narrow silhouette should cost only what it covers.
            let (mut left, mut top, mut right, mut bottom) = (w as f64, h as f64, 0.0_f64, 0.0_f64);
            for u in layer.x {
                for v in layer.y {
                    let x = center_x + (cos * u - sin * v) * self.scale;
                    let y = center_y + (sin * u + cos * v) * self.scale * self.squash;
                    left = left.min(x);
                    right = right.max(x);
                    top = top.min(y);
                    bottom = bottom.max(y);
                }
            }
            let x0 = (left.floor() as usize).min(w);
            let x1 = (right.ceil() as usize).min(w);
            let y0 = (top.floor() as usize).min(h);
            let y1 = (bottom.ceil() as usize).min(h);
            for y in y0..y1 {
                let dy = (y as f64 + 0.5 - center_y) / (self.scale * self.squash);
                let (du, dv) = (sin * dy, cos * dy);
                let row = &mut rgba[(y * w + x0) * 4..(y * w + x1) * 4];
                for (&(cx, sx), dst) in column_terms[x0..x1].iter().zip(row.chunks_exact_mut(4)) {
                    let u = (cx + du + tile_center).floor() as isize;
                    let v = (sx + dv + tile_center).floor() as isize;
                    if !(0..self.tile as isize).contains(&u)
                        || !(0..self.tile as isize).contains(&v)
                    {
                        continue;
                    }
                    let offset = (v as usize * self.sheet.width as usize
                        + layer.index * self.tile
                        + u as usize)
                        * 4;
                    let src = &self.sheet.rgba[offset..offset + 4];
                    match src[3] {
                        0 => {}
                        255 => dst.copy_from_slice(src),
                        _ if dst[3] == 0 => dst.copy_from_slice(src),
                        a => {
                            let alpha = f64::from(a);
                            let under = f64::from(dst[3]) * (1. - alpha / 255.);
                            let total = alpha + under;
                            // Uint8ClampedArray rounds half-integers to even in the JS reference.
                            for k in 0..3 {
                                dst[k] = ((f64::from(src[k]) * alpha + f64::from(dst[k]) * under)
                                    / total)
                                    .round_ties_even()
                                    as u8;
                            }
                            dst[3] = total.round_ties_even() as u8;
                        }
                    }
                }
            }
        }
        PngImage {
            width: self.width,
            height: self.height,
            rgba,
        }
    }
}

fn image_output(img: &PngImage) -> Value {
    Value::String(bytes_to_data_url(&encode_png_rgba(img), Some("image/png")))
}

pub fn run_sprite(node: &Node, inputs: &HashMap<String, Value>) -> Result<NodeOutcome> {
    let port = if node.ty == "slice-3d" {
        "model"
    } else {
        "sheet"
    };
    let url = input_text(inputs, port)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| err(format!("Wire {port} into {}.", node.ty)))?;
    let source = url_bytes(&url)?;
    let fields = &node.fields;
    let mut ports = Map::new();
    if node.ty == "slice-3d" {
        let palette_count = number(fields, "palette", 16., 0., 64., true)?;
        let palette = if palette_count != 0. {
            input_text(inputs, "paletteImage")
                .filter(|url| !url.is_empty())
                .map(|url| decode_image(&url_bytes(&url)?))
                .transpose()?
        } else {
            None
        };
        let (sheet, layout) = slice_model(&source, fields, palette.as_ref(), &decode_image)?;
        let view = View::new(&sheet, &Map::new())?;
        ports.insert("image".into(), image_output(&view.frame(35.)));
        ports.insert("layout".into(), Value::String(layout.to_string()));
        ports.insert("sheet".into(), image_output(&sheet));
    } else {
        let sheet = decode_image(&source)?;
        let yaw = number(fields, "yaw", 35., -3600., 3600., false)?;
        let view = View::new(&sheet, fields)?;
        ports.insert("image".into(), image_output(&view.frame(yaw)));
        if flag(fields, "turntable", false) {
            let fps = number(fields, "fps", 24., 1., 60., true)?;
            let duration = number(fields, "duration", 3., 0.25, 30., false)?;
            let start = number(fields, "yawStart", 0., -3600., 3600., false)?;
            let end = number(fields, "yawEnd", 360., -3600., 3600., false)?;
            let count = ((fps * duration).round() as u32).max(2);
            let bytes = encode_turntable(view.width, view.height, count, duration, |i| {
                view.frame(start + (end - start) * f64::from(i) / f64::from(count - 1))
            })?;
            ports.insert(
                "video".into(),
                json!(bytes_to_data_url(&bytes, Some("video/mp4"))),
            );
        }
    }
    Ok(done(ports, Some(0.0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotated_translucent_layers_match_the_reference_pixels() {
        let mut sheet = PngImage {
            width: 128,
            height: 16,
            rgba: vec![0; 128 * 16 * 4],
        };
        for layer in 0..8 {
            for y in 1..10 {
                for x in 3..9 {
                    if layer == 2 || (x + y + layer) % 3 == 0 {
                        continue;
                    }
                    let offset = (y * 128 + layer * 16 + x) * 4;
                    sheet.rgba[offset..offset + 4].copy_from_slice(&[
                        (x * 20) as u8,
                        (y * 20) as u8,
                        (layer * 32) as u8,
                        [0, 128, 192, 255][(layer + x + y) % 4],
                    ]);
                }
            }
        }
        // Golden values from the Node software renderer. These sample a fully
        // collapsed floor, overlapping layers, and a fractional off-axis view.
        for (yaw, pitch, spacing, scale, expected) in [
            (0., 0., 1., 2., (50, 32, 0x04863c25u32)),
            (35., 30., 1., 3., (72, 70, 0x8c5c12b4)),
            (175., 90., 0., 1., (28, 28, 0x27d4cf58)),
            (-283., 42.75, 0.33, 2.5, (62, 50, 0x51a6de65)),
        ] {
            let fields = json!({"pitch":pitch,"spacing":spacing,"scale":scale});
            let view = View::new(&sheet, fields.as_object().unwrap()).unwrap();
            let output = view.frame(yaw);
            let hash = output.rgba.iter().fold(2166136261u32, |h, &b| {
                (h ^ u32::from(b)).wrapping_mul(16777619)
            });
            assert_eq!((output.width, output.height, hash), expected, "yaw={yaw}");
        }
    }
}
