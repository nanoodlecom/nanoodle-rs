//! Read the active, static triangle scene of a self-contained glTF 2 asset.
use super::{err, AlphaMode, Mesh, PngImage, Texture};
use crate::local_media::{data_url_bytes, sniff_mime};
use crate::Result;
use serde_json::Value;
use std::borrow::Cow;
use std::collections::HashMap;
use std::rc::Rc;

type Matrix = [f64; 16];
const IDENTITY: Matrix = [
    1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
];

pub(super) fn trs(t: [f64; 3], r: [f64; 4], s: [f64; 3]) -> Matrix {
    let [x, y, z, w] = r;
    let [sx, sy, sz] = s;
    let (x2, y2, z2) = (x + x, y + y, z + z);
    let (xx, xy, xz) = (x * x2, x * y2, x * z2);
    let (yy, yz, zz) = (y * y2, y * z2, z * z2);
    let (wx, wy, wz) = (w * x2, w * y2, w * z2);
    [
        (1. - (yy + zz)) * sx,
        (xy + wz) * sx,
        (xz - wy) * sx,
        0.,
        (xy - wz) * sy,
        (1. - (xx + zz)) * sy,
        (yz + wx) * sy,
        0.,
        (xz + wy) * sz,
        (yz - wx) * sz,
        (1. - (xx + yy)) * sz,
        0.,
        t[0],
        t[1],
        t[2],
        1.,
    ]
}

fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
    std::array::from_fn(|i| {
        let r = i % 4;
        let c = i - r;
        a[r] * b[c] + a[4 + r] * b[c + 1] + a[8 + r] * b[c + 2] + a[12 + r] * b[c + 3]
    })
}

fn numbers<const N: usize>(value: &Value, default: [f64; N], message: &str) -> Result<[f64; N]> {
    if value.is_null() {
        return Ok(default);
    }
    let items = value
        .as_array()
        .filter(|a| a.len() == N)
        .ok_or_else(|| err(message))?;
    let mut out = default;
    for (dst, item) in out.iter_mut().zip(items) {
        *dst = item
            .as_f64()
            .filter(|x| x.is_finite())
            .ok_or_else(|| err(message))?;
    }
    Ok(out)
}

fn unsigned(value: &Value) -> Option<usize> {
    usize::try_from(value.as_u64()?).ok()
}

fn index(value: &Value) -> Result<usize> {
    unsigned(value).ok_or_else(|| err("Invalid glTF index or byte count."))
}

fn get<'a>(array: &'a Value, i: &Value) -> Result<&'a Value> {
    array
        .get(index(i)?)
        .ok_or_else(|| err("Invalid glTF index."))
}

fn optional_index(value: &Value, fallback: usize) -> Result<usize> {
    if value.is_null() {
        Ok(fallback)
    } else {
        index(value)
    }
}

fn items(value: &Value) -> Result<&[Value]> {
    if value.is_null() {
        return Ok(&[]);
    }
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| err("Invalid glTF array."))
}

fn le32(raw: &[u8]) -> u32 {
    u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])
}

fn unpack(raw: &[u8]) -> Result<(Value, Option<&[u8]>)> {
    if !raw.starts_with(b"glTF") {
        let text = raw.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(raw);
        return serde_json::from_slice(text)
            .map(|doc| (doc, None))
            .map_err(|_| err("Expected a GLB, a self-contained glTF 2 mesh, or a Gaussian PLY."));
    }
    if raw.len() < 20 || le32(&raw[4..]) != 2 || le32(&raw[8..]) as usize != raw.len() {
        return Err(err("Invalid GLB v2 header or length."));
    }
    let mut remaining = &raw[12..];
    let mut json = None;
    let mut bin = None;
    while !remaining.is_empty() {
        let (header, body) = remaining
            .split_at_checked(8)
            .ok_or_else(|| err("Truncated GLB chunk header."))?;
        let (chunk, tail) = body
            .split_at_checked(le32(header) as usize)
            .ok_or_else(|| err("Truncated GLB chunk."))?;
        match &header[4..] {
            b"JSON" => json = Some(chunk),
            b"BIN\0" => bin = Some(chunk),
            _ => {}
        }
        remaining = tail;
    }
    let document = serde_json::from_slice(json.ok_or_else(|| err("Missing GLB JSON chunk."))?)
        .map_err(|_| err("Invalid GLB JSON chunk."))?;
    Ok((document, bin))
}

fn embedded(uri: &Value) -> Result<Vec<u8>> {
    let uri = uri
        .as_str()
        .filter(|s| s.starts_with("data:"))
        .ok_or_else(|| {
            err("External glTF buffers and images require a self-contained .gltf or .glb export.")
        })?;
    Ok(data_url_bytes(uri)?.0)
}

struct DecodedImage {
    pixels: Rc<PngImage>,
    is_webp: bool,
}

struct Document<'a> {
    json: Value,
    buffers: Vec<Cow<'a, [u8]>>,
}

impl<'a> Document<'a> {
    fn open(raw: &'a [u8]) -> Result<Self> {
        let (json, bin) = unpack(raw)?;
        if json["asset"]["version"] != "2.0" {
            return Err(err("Expected glTF 2.0."));
        }
        let unsupported: Vec<_> = items(&json["extensionsRequired"])?
            .iter()
            .map(|name| name.as_str().unwrap_or("?"))
            .filter(|&name| name != "EXT_texture_webp")
            .collect();
        if !unsupported.is_empty() {
            return Err(err(format!(
                "Unsupported required glTF extensions: {}",
                unsupported.join(", ")
            )));
        }
        let mut buffers = Vec::new();
        for (i, spec) in items(&json["buffers"])?.iter().enumerate() {
            let data = if !spec["uri"].is_null() {
                Cow::Owned(embedded(&spec["uri"])?)
            } else {
                Cow::Borrowed(
                    bin.filter(|_| i == 0)
                        .ok_or_else(|| err("Missing glTF buffer."))?,
                )
            };
            if index(&spec["byteLength"])? > data.len() {
                return Err(err("Missing or truncated glTF buffer."));
            }
            buffers.push(data);
        }
        Ok(Self { json, buffers })
    }

    fn view(&self, id: &Value) -> Result<(&[u8], &Value)> {
        let view = get(&self.json["bufferViews"], id)?;
        if !view["extensions"]["EXT_meshopt_compression"].is_null() {
            return Err(err("Meshopt-compressed buffers are unsupported."));
        }
        let buffer = self
            .buffers
            .get(index(&view["buffer"])?)
            .ok_or_else(|| err("Invalid glTF buffer view."))?;
        let start = optional_index(&view["byteOffset"], 0)?;
        let size = index(&view["byteLength"])?;
        let portion = start
            .checked_add(size)
            .and_then(|end| buffer.get(start..end))
            .ok_or_else(|| err("Invalid glTF buffer view."))?;
        Ok((portion, view))
    }

    // Decode directly into the final vector width. RGB attributes get an
    // implicit alpha of one; the remaining supported shapes fill every slot.
    fn accessor<const N: usize>(
        &self,
        id: &Value,
        min_channels: usize,
        count: Option<usize>,
    ) -> Result<Vec<[f64; N]>> {
        let a = get(&self.json["accessors"], id)?;
        let n = match a["type"].as_str() {
            Some("SCALAR") => 1,
            Some("VEC2") => 2,
            Some("VEC3") => 3,
            Some("VEC4") => 4,
            _ => 0,
        };
        let kind = index(&a["componentType"])?;
        let (size, max) = match kind {
            5120 => (1, 127.),
            5121 => (1, 255.),
            5122 => (2, 32767.),
            5123 => (2, 65535.),
            5125 => (4, 4294967295.),
            5126 => (4, 1.),
            _ => return Err(err("Unsupported glTF accessor component type.")),
        };
        if !(min_channels..=N).contains(&n) || !a["sparse"].is_null() || a["bufferView"].is_null() {
            return Err(err("Unsupported glTF accessor type or sparse encoding."));
        }
        let (buf, view) = self.view(&a["bufferView"])?;
        let width = n * size;
        let stride = optional_index(&view["byteStride"], width)?;
        let offset = optional_index(&a["byteOffset"], 0)?;
        let rows = index(&a["count"])?;
        let end = rows
            .checked_sub(1)
            .and_then(|n| n.checked_mul(stride))
            .and_then(|n| n.checked_add(offset))
            .and_then(|n| n.checked_add(width));
        if stride < width || count.is_some_and(|c| rows != c) || end.is_none_or(|e| e > buf.len()) {
            return Err(err("Invalid glTF accessor bounds or count."));
        }
        let normalized = a["normalized"].as_bool().unwrap_or(false);
        let mut values = Vec::with_capacity(rows);
        for row in 0..rows {
            let start = offset + row * stride;
            let mut values_row = [1.; N];
            for (out, bytes) in values_row
                .iter_mut()
                .zip(buf[start..start + width].chunks_exact(size))
            {
                let value = match kind {
                    5120 => (bytes[0] as i8) as f64,
                    5121 => bytes[0] as f64,
                    5122 => i16::from_le_bytes([bytes[0], bytes[1]]) as f64,
                    5123 => u16::from_le_bytes([bytes[0], bytes[1]]) as f64,
                    5125 => le32(bytes) as f64,
                    _ => f32::from_bits(le32(bytes)) as f64,
                };
                if !value.is_finite() {
                    return Err(err("Non-finite glTF attribute."));
                }
                *out = if normalized {
                    (value / max).max(-1.)
                } else {
                    value
                };
            }
            values.push(values_row);
        }
        Ok(values)
    }

    fn texture(
        &self,
        info: &Value,
        cache: &mut HashMap<usize, DecodedImage>,
        decode: &dyn Fn(&[u8]) -> Result<PngImage>,
    ) -> Result<Texture> {
        if !info["extensions"]["KHR_texture_transform"].is_null() {
            return Err(err("Unsupported base-color texture encoding or transform."));
        }
        let texture = get(&self.json["textures"], &info["index"])?;
        let webp = texture["extensions"].get("EXT_texture_webp");
        let source = match webp {
            Some(ext) => unsigned(&ext["source"])
                .filter(|&i| self.json["images"].get(i).is_some())
                .ok_or_else(|| err("Invalid EXT_texture_webp image source."))?,
            None => index(&texture["source"])?,
        };
        let img = self.json["images"]
            .get(source)
            .ok_or_else(|| err("Missing glTF texture image."))?;
        if webp.is_some() && !img["bufferView"].is_null() && img["mimeType"] != "image/webp" {
            return Err(err("EXT_texture_webp bufferView requires image/webp."));
        }
        let invalid_webp = || err("EXT_texture_webp source must contain a WebP image.");
        let decoded = match cache.entry(source) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let data = if !img["uri"].is_null() {
                    Cow::Owned(embedded(&img["uri"])?)
                } else {
                    Cow::Borrowed(self.view(&img["bufferView"])?.0)
                };
                let is_webp = sniff_mime(&data) == "image/webp";
                if webp.is_some() && !is_webp {
                    return Err(invalid_webp());
                }
                let pixels = decode(&data)?;
                let n = (pixels.width as usize)
                    .checked_mul(pixels.height as usize)
                    .and_then(|n| n.checked_mul(4));
                if n != Some(pixels.rgba.len()) || pixels.rgba.is_empty() {
                    return Err(err("Invalid texture dimensions."));
                }
                entry.insert(DecodedImage {
                    pixels: Rc::new(pixels),
                    is_webp,
                })
            }
        };
        // A core texture may already have decoded this index as a PNG or JPEG.
        if webp.is_some() && !decoded.is_webp {
            return Err(invalid_webp());
        }
        let image = Rc::clone(&decoded.pixels);
        let sampler = if texture["sampler"].is_null() {
            &Value::Null
        } else {
            get(&self.json["samplers"], &texture["sampler"])?
        };
        let wrap = |axis: &str| {
            sampler[axis]
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .unwrap_or(10497)
        };
        Ok(Texture {
            image,
            wrap_s: wrap("wrapS"),
            wrap_t: wrap("wrapT"),
        })
    }

    fn primitive(
        &self,
        p: &Value,
        world: &Matrix,
        cache: &mut HashMap<usize, DecodedImage>,
        decode: &dyn Fn(&[u8]) -> Result<PngImage>,
    ) -> Result<Mesh> {
        let mode = optional_index(&p["mode"], 4)?;
        if !(4..=6).contains(&mode) {
            return Err(err("Slice 3D requires triangles. For a point-cloud preview, supply the original Gaussian PLY."));
        }
        if !items(&p["targets"])?.is_empty()
            || !p["extensions"]["KHR_draco_mesh_compression"].is_null()
        {
            return Err(err(
                "Morph targets and Draco-compressed meshes are unsupported.",
            ));
        }
        let attrs = &p["attributes"];
        let mut positions = self.accessor::<3>(&attrs["POSITION"], 3, None)?;
        for point in &mut positions {
            let [x, y, z] = *point;
            *point = std::array::from_fn(|i| {
                world[i] * x + world[i + 4] * y + world[i + 8] * z + world[i + 12]
            });
            if !point.iter().all(|v| v.is_finite()) {
                return Err(err("Invalid glTF scene transform."));
            }
        }
        let count = positions.len();
        let indices = if p["indices"].is_null() {
            None
        } else {
            let desc = get(&self.json["accessors"], &p["indices"])?;
            if !matches!(desc["componentType"].as_u64(), Some(5121 | 5123 | 5125))
                || desc["normalized"] == true
            {
                return Err(err("glTF triangle indices must be unsigned integers."));
            }
            let indices = self
                .accessor::<1>(&p["indices"], 1, None)?
                .into_iter()
                .map(|[v]| v as usize)
                .collect::<Vec<_>>();
            if indices.iter().any(|&i| i >= count) {
                return Err(err("Invalid triangle index."));
            }
            Some(indices)
        };
        let length = indices.as_ref().map_or(count, Vec::len);
        if length < 3 || (mode == 4 && length % 3 != 0) {
            return Err(err("Incomplete triangle primitive."));
        }
        let mat = if p["material"].is_null() {
            &Value::Null
        } else {
            get(&self.json["materials"], &p["material"])?
        };
        let pbr = &mat["pbrMetallicRoughness"];
        let factor = numbers(
            &pbr["baseColorFactor"],
            [1.; 4],
            "Invalid glTF base color factor.",
        )?;
        if factor.iter().any(|v| !(0.0..=1.0).contains(v)) {
            return Err(err("Invalid glTF base color factor."));
        }
        let alpha_mode = match mat["alphaMode"].as_str() {
            None if mat["alphaMode"].is_null() => AlphaMode::Opaque,
            Some("OPAQUE") => AlphaMode::Opaque,
            Some("BLEND") => AlphaMode::Blend,
            Some("MASK") => AlphaMode::Mask,
            _ => return Err(err("Invalid glTF alpha mode.")),
        };
        let cutoff = if mat["alphaCutoff"].is_null() {
            0.5
        } else {
            mat["alphaCutoff"]
                .as_f64()
                .ok_or_else(|| err("Invalid glTF alpha cutoff."))?
        };
        let colors = if attrs["COLOR_0"].is_null() {
            None
        } else {
            Some(self.accessor::<4>(&attrs["COLOR_0"], 3, Some(count))?)
        };
        let color_channels = if colors.is_some()
            && get(&self.json["accessors"], &attrs["COLOR_0"])?["type"] == "VEC3"
        {
            3
        } else {
            4
        };
        let texture_info = &pbr["baseColorTexture"];
        let (uvs, texture) = if texture_info.is_null() {
            (None, None)
        } else {
            let coord = format!("TEXCOORD_{}", optional_index(&texture_info["texCoord"], 0)?);
            (
                Some(self.accessor::<2>(&attrs[coord], 2, Some(count))?),
                Some(self.texture(texture_info, cache, decode)?),
            )
        };
        Ok(Mesh {
            positions,
            indices,
            mode: mode as u32,
            colors,
            color_channels,
            uvs,
            texture,
            factor,
            alpha_mode,
            cutoff,
        })
    }
}

fn node_matrix(node: &Value) -> Result<Matrix> {
    let msg = "Invalid glTF scene transform.";
    if !node["matrix"].is_null() {
        return numbers(&node["matrix"], IDENTITY, msg);
    }
    Ok(trs(
        numbers(&node["translation"], [0.; 3], msg)?,
        numbers(&node["rotation"], [0., 0., 0., 1.], msg)?,
        numbers(&node["scale"], [1.; 3], msg)?,
    ))
}

pub(super) fn parse(raw: &[u8], decode: &dyn Fn(&[u8]) -> Result<PngImage>) -> Result<Vec<Mesh>> {
    let doc = Document::open(raw)?;
    let mut meshes = Vec::new();
    let mut textures = HashMap::new();
    let mesh_defs = items(&doc.json["meshes"])?;
    let mut emit = |i: usize, world: &Matrix| -> Result<()> {
        let mesh = mesh_defs.get(i).ok_or_else(|| err("Missing glTF mesh."))?;
        for p in items(&mesh["primitives"])? {
            meshes.push(doc.primitive(p, world, &mut textures, decode)?);
        }
        Ok(())
    };
    let nodes = items(&doc.json["nodes"])?;
    if nodes.is_empty() {
        for i in 0..mesh_defs.len() {
            emit(i, &IDENTITY)?;
        }
    } else {
        let scene_id = optional_index(&doc.json["scene"], 0)?;
        let roots = doc.json["scenes"]
            .get(scene_id)
            .and_then(|v| v.get("nodes"));
        let roots = if let Some(roots) = roots {
            items(roots)?
                .iter()
                .map(index)
                .collect::<Result<Vec<_>>>()?
        } else {
            let mut child = vec![false; nodes.len()];
            for node in nodes {
                for v in items(&node["children"])? {
                    *child
                        .get_mut(index(v)?)
                        .ok_or_else(|| err("Invalid glTF child node."))? = true;
                }
            }
            (0..nodes.len()).filter(|&i| !child[i]).collect()
        };
        // glTF scene nodes form disjoint trees. An explicit stack also handles
        // deeply nested scenes without growing the native call stack.
        let mut seen = vec![false; nodes.len()];
        let mut pending: Vec<_> = roots.into_iter().rev().map(|i| (i, IDENTITY)).collect();
        while let Some((i, parent)) = pending.pop() {
            let node = nodes
                .get(i)
                .ok_or_else(|| err("Invalid or cyclic glTF node tree."))?;
            if std::mem::replace(&mut seen[i], true) {
                return Err(err("Invalid or cyclic glTF node tree."));
            }
            if !node["skin"].is_null() {
                return Err(err("Skinned meshes require a static mesh export."));
            }
            let world = multiply(&parent, &node_matrix(node)?);
            if !node["mesh"].is_null() {
                emit(index(&node["mesh"])?, &world)?;
            }
            for child in items(&node["children"])?.iter().rev() {
                pending.push((index(child)?, world));
            }
        }
    }
    if meshes.is_empty() {
        return Err(err("No triangles in the selected glTF scene."));
    }
    Ok(meshes)
}

#[cfg(test)]
mod webp_tests {
    use super::*;
    use serde_json::json;
    use std::cell::Cell;

    const WEBP: &str =
        "data:image/webp;base64,UklGRh4AAABXRUJQVlA4TBEAAAAvAAAAAAdQkE40pf+BiOh/AAA=";
    const FALLBACK: &str = "data:image/png;base64,iVBORw0KGgo=";

    // Tag the selected bytes so these tests exercise the parser independently
    // of the format-specific pixel decoders.
    fn identify(bytes: &[u8]) -> Result<PngImage> {
        Ok(PngImage {
            width: 1,
            height: 1,
            rgba: vec![bytes[0], 0, 0, 255],
        })
    }

    #[test]
    fn webp_source_overrides_fallback_and_reuses_its_decoded_image() {
        let spec = json!({
            "asset":{"version":"2.0"},
            "extensionsRequired":["EXT_texture_webp"],
            "images":[{"uri":FALLBACK},{"uri":WEBP}],
            "textures":[
                {"source":0, "extensions":{"EXT_texture_webp":{"source":1}}},
                {"extensions":{"EXT_texture_webp":{"source":1}}},
                {"source":0}
            ]
        })
        .to_string();
        let doc = Document::open(spec.as_bytes()).unwrap();
        let calls = Cell::new(0);
        let decode = |raw: &[u8]| {
            calls.set(calls.get() + 1);
            identify(raw)
        };
        let mut cache = HashMap::new();
        for (index, tag) in [(0, b'R'), (1, b'R'), (2, 0x89)] {
            let result = doc
                .texture(&json!({"index":index}), &mut cache, &decode)
                .unwrap();
            assert_eq!(result.image.rgba[0], tag);
        }
        assert_eq!(calls.get(), 2);
        let unsupported=json!({"asset":{"version":"2.0"},"extensionsRequired":["EXT_texture_webp","EXT_meshopt_compression"]}).to_string();
        let error = Document::open(unsupported.as_bytes()).err().unwrap();
        assert_eq!(
            error.0,
            "Unsupported required glTF extensions: EXT_meshopt_compression"
        );
    }

    #[test]
    fn invalid_override_never_selects_a_core_or_cached_non_webp_image() {
        let mut doc = Document {
            json: json!({
                "images":[{"uri":FALLBACK}],
                "textures":[{"source":0},{"source":0,"extensions":{"EXT_texture_webp":{"source":0}}}]
            }),
            buffers: vec![],
        };
        let mut cache = HashMap::new();
        doc.texture(&json!({"index":0}), &mut cache, &identify)
            .unwrap();
        let info = json!({"index":1});
        assert!(doc
            .texture(&info, &mut cache, &identify)
            .err()
            .unwrap()
            .0
            .contains("must contain a WebP"));
        for bad in [
            Value::Null,
            json!({}),
            json!({"source":-1}),
            json!({"source":0.5}),
            json!({"source":"0"}),
            json!({"source":1}),
        ] {
            doc.json["textures"][1]["extensions"]["EXT_texture_webp"] = bad;
            assert!(doc
                .texture(&info, &mut cache, &identify)
                .err()
                .unwrap()
                .0
                .contains("image source"));
        }
    }

    #[test]
    fn embedded_buffer_view_checks_declared_webp_mime_and_signature() {
        let raw = data_url_bytes(WEBP).unwrap().0;
        let mut doc = Document {
            json: json!({
                "images":[{"bufferView":0,"mimeType":"image/png"}],
                "textures":[{"extensions":{"EXT_texture_webp":{"source":0}}}],
                "bufferViews":[{"buffer":0,"byteLength":raw.len()}]
            }),
            buffers: vec![Cow::Owned(raw)],
        };
        let info = json!({"index":0});
        let mut cache = HashMap::new();
        assert!(doc
            .texture(&info, &mut cache, &identify)
            .err()
            .unwrap()
            .0
            .contains("requires image/webp"));
        doc.json["images"][0]["mimeType"] = json!("image/webp");
        assert_eq!(
            doc.texture(&info, &mut cache, &identify)
                .unwrap()
                .image
                .rgba[0],
            b'R'
        );
        cache.clear();
        doc.buffers[0].to_mut()[8] = b'X';
        assert!(doc
            .texture(&info, &mut cache, &identify)
            .err()
            .unwrap()
            .0
            .contains("must contain a WebP"));
    }
}
