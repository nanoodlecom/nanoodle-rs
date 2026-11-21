//! Slice posed triangles and oriented Gaussian splats onto a shared world grid.
//! The PNG strip orders square tiles from ground to top, left to right.
use super::{
    err,
    sprite_sequence::SequenceWriter,
    sprite_stack::{flag, number},
    PngImage,
};
use crate::Result;
use serde_json::{json, Map, Value};
use std::rc::Rc;
mod gltf;
mod palette;
mod ply;

type Point = [f64; 3];
#[derive(Clone)]
struct Texture {
    image: Rc<PngImage>,
    wrap_s: u32,
    wrap_t: u32,
}
#[derive(Clone)]
enum AlphaMode {
    Opaque,
    Mask,
    Blend,
}
#[derive(Clone)]
struct Mesh {
    positions: Vec<Point>,
    indices: Option<Rc<Vec<usize>>>,
    mode: u32,
    colors: Option<Rc<Vec<[f64; 4]>>>,
    color_channels: usize,
    uvs: Option<Rc<Vec<[f64; 2]>>>,
    texture: Option<Texture>,
    factor: [f64; 4],
    alpha_mode: AlphaMode,
    cutoff: f64,
}
struct Splat {
    center: Point,
    scales: Point,
    rotation: [f64; 16],
    extent: Point,
    color: Point,
    alpha: f64,
}
enum Model {
    Mesh(Vec<Mesh>),
    Splat(Vec<Splat>),
}

struct Bounds {
    min: Point,
    max: Point,
}

impl Bounds {
    fn include(&mut self, model: &Model) {
        let mut grow = |p: Point| {
            for k in 0..3 {
                self.min[k] = self.min[k].min(p[k]);
                self.max[k] = self.max[k].max(p[k]);
            }
        };
        match model {
            Model::Mesh(meshes) => {
                for m in meshes {
                    for &p in &m.positions {
                        grow(p);
                    }
                }
            }
            Model::Splat(splats) => {
                for s in splats {
                    grow(std::array::from_fn(|k| s.center[k] - s.extent[k]));
                    grow(std::array::from_fn(|k| s.center[k] + s.extent[k]));
                }
            }
        }
    }
}

struct Grid {
    axis: [usize; 3],
    sign: f64,
    size: usize,
    layers: usize,
    min: Point,
    max: Point,
    cu: f64,
    cv: f64,
    unit: f64,
    step: f64,
}
impl Grid {
    fn new(bounds: Bounds, axis: &str, size: usize, layers: usize) -> Result<Self> {
        let (axis, sign) = match axis {
            "x" => ([1, 2, 0], -1.),
            "y" => ([0, 2, 1], 1.),
            "z" => ([0, 1, 2], -1.),
            _ => return Err(err("Up axis must be auto, x, y or z.")),
        };
        let Bounds { min, max } = bounds;
        let span = (0..3).map(|k| max[k] - min[k]).fold(0., f64::max);
        if !span.is_finite() || span <= 0. {
            return Err(err("Model bounds have no usable extent."));
        }
        let [u, v, _] = axis;
        Ok(Self {
            axis,
            sign,
            min,
            max,
            cu: (max[u] + min[u]) / 2.,
            cv: (max[v] + min[v]) / 2.,
            unit: span / (size - 2) as f64,
            step: span / layers as f64,
            size,
            layers,
        })
    }
    fn project(&self, p: Point) -> Point {
        let [u, v, h] = self.axis;
        [
            (p[u] - self.cu) / self.unit + self.size as f64 / 2.,
            self.sign * (p[v] - self.cv) / self.unit + self.size as f64 / 2.,
            (p[h] - self.min[h]) / self.step,
        ]
    }
    fn world(&self, x: f64, y: f64, l: f64) -> Point {
        let mut p = [0.; 3];
        let [u, v, h] = self.axis;
        p[u] = (x - self.size as f64 / 2.) * self.unit + self.cu;
        p[v] = self.sign * (y - self.size as f64 / 2.) * self.unit + self.cv;
        p[h] = self.min[h] + l * self.step;
        p
    }
    fn pixel(&self, cell: [usize; 3]) -> usize {
        (cell[1] * self.layers + cell[2]) * self.size + cell[0]
    }
}

fn wrap(t: f64, mode: u32) -> f64 {
    match mode {
        33071 => t.clamp(0., 1.),
        33648 => 1. - (((t % 2. + 2.) % 2.) - 1.).abs(),
        _ => t - t.floor(),
    }
}
fn linear(x: f64) -> f64 {
    if x < 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}
fn srgb(x: f64) -> f64 {
    if x < 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1. / 2.4) - 0.055
    }
}
fn byte(x: f64) -> u8 {
    x.round_ties_even() as u8
}

#[derive(Clone, Copy)]
struct Sample {
    rgba: [u8; 4],
    visible: bool,
}
impl Mesh {
    fn sample(&self, ids: [usize; 3], w: Point) -> Sample {
        let mut color = self.factor;
        if let Some(attr) = &self.colors {
            for k in 0..self.color_channels {
                color[k] *=
                    w[0] * attr[ids[0]][k] + w[1] * attr[ids[1]][k] + w[2] * attr[ids[2]][k];
            }
        }
        if let (Some(texture), Some(attr)) = (&self.texture, &self.uvs) {
            let image = &texture.image;
            let uv =
                |k: usize| w[0] * attr[ids[0]][k] + w[1] * attr[ids[1]][k] + w[2] * attr[ids[2]][k];
            let x = ((wrap(uv(0), texture.wrap_s) * image.width as f64) as usize)
                .min(image.width as usize - 1);
            let y = ((wrap(uv(1), texture.wrap_t) * image.height as f64) as usize)
                .min(image.height as usize - 1);
            let px = &image.rgba[(y * image.width as usize + x) * 4..];
            for k in 0..3 {
                color[k] *= linear(px[k] as f64 / 255.);
            }
            color[3] *= px[3] as f64 / 255.;
        }
        let alpha = match self.alpha_mode {
            AlphaMode::Opaque => 1.,
            AlphaMode::Mask => {
                if color[3] >= self.cutoff {
                    1.
                } else {
                    0.
                }
            }
            AlphaMode::Blend => color[3],
        };
        Sample {
            rgba: [
                byte(srgb(color[0]) * 255.),
                byte(srgb(color[1]) * 255.),
                byte(srgb(color[2]) * 255.),
                byte(alpha * 255.),
            ],
            visible: alpha != 0.,
        }
    }
}
fn put(img: &mut PngImage, grid: &Grid, cell: [usize; 3], c: Sample, only_empty: bool) {
    let i = grid.pixel(cell) * 4;
    if c.visible && (!only_empty || img.rgba[i + 3] == 0) {
        img.rgba[i..i + 4].copy_from_slice(&c.rgba);
    }
}

#[derive(Clone, Copy)]
struct Hit {
    depth: f64,
    winding: i32,
    sample: Sample,
}
const PLANES: [[usize; 3]; 3] = [[0, 1, 2], [1, 2, 0], [2, 0, 1]];

fn area([p, q, r]: [Point; 3], a: usize, b: usize) -> f64 {
    (q[a] - p[a]) * (r[b] - p[b]) - (q[b] - p[b]) * (r[a] - p[a])
}
fn scan(
    [p, q, r]: [Point; 3],
    a: usize,
    b: usize,
    width: usize,
    height: usize,
    mut visit: impl FnMut(usize, usize, Point, i32),
) {
    let dx = q[a] - p[a];
    let dy = q[b] - p[b];
    let ex = r[a] - p[a];
    let ey = r[b] - p[b];
    let det = dx * ey - dy * ex;
    if det.abs() < 1e-12 {
        return;
    }
    let span = |k: usize, limit: usize| {
        let start = (p[k].min(q[k]).min(r[k]) - 0.5).ceil() as usize;
        let end = ((p[k].max(q[k]).max(r[k]) - 0.5).floor() + 1.) as usize;
        start..end.min(limit)
    };
    let xs = span(a, width);
    for y in span(b, height) {
        for x in xs.clone() {
            let px = x as f64 + 0.5 - p[a];
            let py = y as f64 + 0.5 - p[b];
            let v = (px * ey - py * ex) / det;
            let w = (dx * py - dy * px) / det;
            let u = 1. - v - w;
            if u.min(v).min(w) >= -1e-8 {
                visit(x, y, [u, v, w], if det > 0. { 1 } else { -1 });
            }
        }
    }
}

fn paint_mesh(grid: &Grid, meshes: &[Mesh], img: &mut PngImage) {
    let dims = [grid.size, grid.size, grid.layers];
    let mut rays = vec![Vec::<Hit>::new(); grid.size * grid.size];
    for m in meshes {
        let points: Vec<_> = m.positions.iter().map(|&p| grid.project(p)).collect();
        let count = m
            .indices
            .as_ref()
            .map_or(points.len(), |indices| indices.len());
        // A flat untextured material gives one color for every face sample.
        let solid = if m.colors.is_none() && m.texture.is_none() {
            Some(m.sample([0; 3], [1., 0., 0.]))
        } else {
            None
        };
        for n in (0..count - 2).step_by(if m.mode == 4 { 3 } else { 1 }) {
            let corners = match m.mode {
                6 => [0, n + 1, n + 2],
                5 if n % 2 == 1 => [n + 1, n, n + 2],
                _ => [n, n + 1, n + 2],
            };
            let ids = corners.map(|k| m.indices.as_ref().map_or(k, |idx| idx[k]));
            let verts = ids.map(|i| points[i]);
            let areas = PLANES.map(|[a, b, _]| area(verts, a, b).abs());
            let mut dominant = 0;
            for k in 1..3 {
                if areas[k] > areas[dominant] {
                    dominant = k;
                }
            }
            if areas[dominant] == 0. {
                continue;
            }
            let mut painted = false;
            for (index, &plane) in [dominant, 0].iter().enumerate() {
                if index == 1 && dominant == 0 {
                    break;
                }
                let [a, b, c] = PLANES[plane];
                scan(verts, a, b, dims[a], dims[b], |x, y, w, winding| {
                    let depth = w[0] * verts[0][c] + w[1] * verts[1][c] + w[2] * verts[2][c];
                    let sample = solid.unwrap_or_else(|| m.sample(ids, w));
                    if plane == dominant {
                        let mut cell = [0; 3];
                        cell[a] = x;
                        cell[b] = y;
                        cell[c] = (depth.floor() as usize).min(dims[c] - 1);
                        put(img, grid, cell, sample, false);
                        painted = true;
                    }
                    if plane == 0 {
                        rays[y * grid.size + x].push(Hit {
                            depth,
                            winding,
                            sample,
                        });
                    }
                });
            }
            if !painted {
                let cell = std::array::from_fn(|k| {
                    (((verts[0][k] + verts[1][k] + verts[2][k]) / 3.).floor() as usize)
                        .min(dims[k] - 1)
                });
                put(
                    img,
                    grid,
                    cell,
                    solid.unwrap_or_else(|| m.sample(ids, [1. / 3.; 3])),
                    false,
                );
            }
        }
    }
    for (i, ray) in rays.iter_mut().enumerate().filter(|(_, r)| r.len() > 1) {
        ray.sort_by(|a, b| {
            a.depth
                .partial_cmp(&b.depth)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // Collapse coincident face edges before counting signed crossings.
        let mut used = 0;
        for n in 0..ray.len() {
            let hit = ray[n];
            if used > 0 && (ray[used - 1].depth - hit.depth).abs() < 1e-6 {
                ray[used - 1].winding += hit.winding;
            } else {
                ray[used] = hit;
                used += 1;
            }
        }
        let hits = &ray[..used];
        if hits.iter().map(|hit| hit.winding.signum()).sum::<i32>() != 0 {
            continue;
        }
        let mut winding = 0;
        for interval in hits.windows(2) {
            let start = interval[0];
            let end = interval[1];
            winding += start.winding.signum();
            if winding == 0 {
                continue;
            }
            let first = (start.depth - 0.5).ceil() as usize;
            let last = ((end.depth - 0.5).ceil() as usize).min(grid.layers);
            for layer in first..last {
                let color = if (layer as f64 + 0.5) < (start.depth + end.depth) / 2. {
                    start.sample
                } else {
                    end.sample
                };
                put(
                    img,
                    grid,
                    [i % grid.size, i / grid.size, layer],
                    color,
                    true,
                );
            }
        }
    }
}

fn paint_splats(grid: &Grid, splats: &[Splat], img: &mut PngImage) {
    let dims = [grid.size, grid.size, grid.layers];
    let mut weights = vec![0_f32; img.rgba.len() / 4];
    for s in splats {
        let center = grid.project(s.center);
        let [u, v, h] = grid.axis;
        let radius = [
            s.extent[u] / grid.unit,
            s.extent[v] / grid.unit,
            s.extent[h] / grid.step,
        ];
        let mut lo = [0; 3];
        let mut hi = [0; 3];
        let mut peak = [0; 3];
        for k in 0..3 {
            lo[k] = (center[k] - radius[k]).floor() as usize;
            hi[k] = ((center[k] + radius[k]).floor() as usize).min(dims[k] - 1);
            peak[k] = (center[k].floor() as usize).min(dims[k] - 1);
        }
        for l in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    let at_center = [x, y, l] == peak;
                    let mut distance = 0.;
                    if !at_center {
                        let world = grid.world(x as f64 + 0.5, y as f64 + 0.5, l as f64 + 0.5);
                        let d: Point = std::array::from_fn(|k| world[k] - s.center[k]);
                        for k in 0..3 {
                            let r = &s.rotation[k * 4..];
                            distance +=
                                ((d[0] * r[0] + d[1] * r[1] + d[2] * r[2]) / s.scales[k]).powi(2);
                        }
                        if distance > 9. {
                            continue;
                        }
                    }
                    let alpha = s.alpha * (-0.5 * distance).exp();
                    if alpha < 0.5 / 255. {
                        continue;
                    }
                    let i = grid.pixel([x, y, l]);
                    let px = &mut img.rgba[i * 4..i * 4 + 4];
                    let old = weights[i] as f64;
                    let sum = old + alpha;
                    for k in 0..3 {
                        px[k] = byte((px[k] as f64 * old + s.color[k] * alpha) / sum);
                    }
                    px[3] = byte(px[3] as f64 + alpha * (255. - px[3] as f64));
                    weights[i] = sum as f32;
                }
            }
        }
    }
}

pub(super) struct SliceSettings {
    size: usize,
    layers: usize,
    pub(super) colors: usize,
    outline: bool,
}

impl SliceSettings {
    pub(super) fn new(fields: &Map<String, Value>) -> Result<Self> {
        let character = fields.get("preset").and_then(Value::as_str) == Some("character");
        Ok(Self {
            size: number(
                fields,
                "resolution",
                if character { 128. } else { 48. },
                16.,
                128.,
                true,
            )? as usize,
            layers: number(
                fields,
                "slices",
                if character { 80. } else { 48. },
                8.,
                128.,
                true,
            )? as usize,
            colors: number(
                fields,
                "palette",
                if character { 0. } else { 16. },
                0.,
                64.,
                true,
            )? as usize,
            outline: flag(fields, "outline", true),
        })
    }
}

fn rasterize(
    model: &Model,
    grid: &Grid,
    settings: &SliceSettings,
    palette: Option<&PngImage>,
) -> Result<PngImage> {
    let width = grid.size * grid.layers;
    let mut img = PngImage {
        width: width as u32,
        height: grid.size as u32,
        rgba: vec![0; width * grid.size * 4],
    };
    match model {
        Model::Mesh(meshes) => paint_mesh(grid, meshes, &mut img),
        Model::Splat(splats) => paint_splats(grid, splats, &mut img),
    }
    palette::finish(
        &mut img,
        grid.size,
        settings.colors,
        settings.outline,
        palette,
    )?;
    Ok(img)
}

pub(super) struct Sliced {
    pub(super) first: PngImage,
    pub(super) animation: Option<Vec<u8>>,
    pub(super) layout: Value,
}

pub(super) fn slice_model(
    raw: &[u8],
    fields: &Map<String, Value>,
    settings: &SliceSettings,
    palette: Option<&PngImage>,
    decode: &dyn Fn(&[u8]) -> Result<PngImage>,
) -> Result<Sliced> {
    let count = number(fields, "frames", 1., 1., 64., true)? as usize;
    let clip = fields.get("clip").and_then(Value::as_str).unwrap_or("");
    let (first_pose, scene, source, default_axis) = if raw.starts_with(b"ply") {
        if count > 1 || !clip.is_empty() {
            return Err(err(
                "Gaussian PLY has no animation clips. Use frames=1 and leave clip blank.",
            ));
        }
        (Model::Splat(ply::parse(raw)?), None, "splat", "z")
    } else {
        let scene = gltf::Scene::open(raw, decode, clip, count)?;
        (Model::Mesh(scene.pose(0)?), Some(scene), "mesh", "y")
    };
    let axis = match fields.get("upAxis").and_then(Value::as_str) {
        None | Some("auto" | "") => default_axis,
        Some(axis) => axis,
    };
    // One bounding pass fixes the ground plane for every pose without storing the entire clip.
    let mut bounds = Bounds {
        min: [f64::INFINITY; 3],
        max: [f64::NEG_INFINITY; 3],
    };
    bounds.include(&first_pose);
    if let Some(scene) = &scene {
        for i in 1..count {
            bounds.include(&Model::Mesh(scene.pose(i)?));
        }
    }
    let grid = Grid::new(bounds, axis, settings.size, settings.layers)?;
    let mut layout = json!({"version":1,"order":"bottom-to-top","columns":grid.layers,"rows":1,"tileWidth":grid.size,"tileHeight":grid.size,"upAxis":axis,"source":source,"bounds":{"min":grid.min,"max":grid.max},"pixelSize":grid.unit,"layerHeight":grid.step});
    let first = rasterize(&first_pose, &grid, settings, palette)?;
    drop(first_pose);
    let animation = if count > 1 {
        let scene = scene
            .as_ref()
            .ok_or_else(|| err("No animation clip. Use frames=1."))?;
        let (name, start, duration) = scene
            .clip()
            .ok_or_else(|| err("No animation clip. Use frames=1."))?;
        let times: Vec<_> = (0..count)
            .map(|i| start + duration * i as f64 / count as f64)
            .collect();
        layout["animation"] = json!({"clip":name,"frames":count,"duration":duration,"times":times});
        let mut writer = SequenceWriter::new(count, duration)?;
        writer.push(&first)?;
        for i in 1..count {
            writer.push(&rasterize(
                &Model::Mesh(scene.pose(i)?),
                &grid,
                settings,
                palette,
            )?)?;
        }
        Some(writer.finish()?)
    } else {
        None
    };
    Ok(Sliced {
        first,
        animation,
        layout,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn character_preset_respects_explicit_and_legacy_values() {
        let fields = json!({"preset":"character"});
        let character = SliceSettings::new(fields.as_object().unwrap()).unwrap();
        assert_eq!(
            (
                character.size,
                character.layers,
                character.colors,
                character.outline
            ),
            (128, 80, 0, true)
        );
        let fields =
            json!({"preset":"character","resolution":24,"slices":12,"palette":8,"outline":false});
        let custom = SliceSettings::new(fields.as_object().unwrap()).unwrap();
        assert_eq!(
            (custom.size, custom.layers, custom.colors, custom.outline),
            (24, 12, 8, false)
        );
        let original = SliceSettings::new(&Map::new()).unwrap();
        assert_eq!(
            (
                original.size,
                original.layers,
                original.colors,
                original.outline
            ),
            (48, 48, 16, true)
        );
    }

    #[test]
    fn transformed_triangle_stays_in_its_ground_tile() {
        let positions = [0_f32, 0., 0., 2., 0., 0., 0., 0., 2.];
        let bytes: Vec<_> = positions.into_iter().flat_map(f32::to_le_bytes).collect();
        let mesh = json!({
            "asset":{"version":"2.0"},
            "buffers":[{"byteLength":bytes.len(),"uri":super::super::bytes_to_data_url(&bytes,None)}],
            "bufferViews":[{"buffer":0,"byteLength":bytes.len()}],
            "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"}],
            "materials":[{"pbrMetallicRoughness":{"baseColorFactor":[1,0,0,1]}}],
            "meshes":[{"primitives":[{"attributes":{"POSITION":0},"material":0}]}],
            "nodes":[{"translation":[3,4,5],"mesh":0}]
        }).to_string();
        let fields = json!({"resolution":16, "slices":8,"outline":true,"palette":0});
        let fields = fields.as_object().unwrap();
        let Sliced {
            first: image,
            layout,
            ..
        } = slice_model(
            mesh.as_bytes(),
            fields,
            &SliceSettings::new(fields).unwrap(),
            None,
            &|_| panic!("untextured primitive"),
        )
        .unwrap();
        assert_eq!((image.width, image.height), (128, 16));
        assert_eq!(layout["upAxis"], "y");
        assert_eq!(layout["bounds"]["min"][1].as_f64(), Some(4.));
        let pixel = |x: usize, y: usize| &image.rgba[(y * 128 + x) * 4..(y * 128 + x) * 4 + 4];
        assert_eq!(pixel(1, 1), &[255, 0, 0, 255]); // interior
        assert_eq!(pixel(0, 1), &[24, 24, 24, 255]); // outline
        assert_eq!(pixel(13, 13), &[0, 0, 0, 0]); // beyond the diagonal
        for y in 0..16 {
            assert!(image.rgba[(y * 128 + 16) * 4..(y + 1) * 128 * 4]
                .iter()
                .all(|&v| v == 0));
        }
    }
}
