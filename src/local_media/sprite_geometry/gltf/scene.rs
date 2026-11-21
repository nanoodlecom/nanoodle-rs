//! Load rest geometry once, then transform each sampled pose into world space.
use super::{
    animation::{Clip, Pose},
    *,
};
use crate::local_media::sprite_geometry::Point;
use serde_json::json;

fn transform(matrix: &Matrix, [x, y, z]: Point) -> Point {
    std::array::from_fn(|i| matrix[i] * x + matrix[i + 4] * y + matrix[i + 8] * z + matrix[i + 12])
}

struct Skin {
    joints: Vec<usize>,
    inverse_binds: Vec<Matrix>,
}

impl Skin {
    fn load(doc: &Document<'_>, id: &Value, active: &[bool]) -> Result<Self> {
        let spec = get(&doc.json["skins"], id)?;
        let joints = items(&spec["joints"])?
            .iter()
            .map(index)
            .collect::<Result<Vec<_>>>()?;
        if joints.is_empty()
            || joints
                .iter()
                .any(|&j| !active.get(j).copied().unwrap_or(false))
        {
            return Err(err("Missing joint node in active glTF skin."));
        }
        let inverse_binds = if spec["inverseBindMatrices"].is_null() {
            vec![IDENTITY; joints.len()]
        } else {
            let accessor = get(&doc.json["accessors"], &spec["inverseBindMatrices"])?;
            if accessor["componentType"] != 5126 {
                return Err(err("glTF inverse bind matrices must use floats."));
            }
            let matrices = doc.accessor::<16>(&spec["inverseBindMatrices"], 16, None)?;
            if matrices.len() < joints.len() {
                return Err(err("Missing glTF inverse bind matrices."));
            }
            matrices
        };
        Ok(Self {
            joints,
            inverse_binds,
        })
    }

    fn palette(&self, worlds: &[Matrix]) -> Vec<Matrix> {
        self.joints
            .iter()
            .zip(&self.inverse_binds)
            .map(|(&node, inverse)| multiply(&worlds[node], inverse))
            .collect()
    }
}

struct Influences {
    joints: Vec<[usize; 4]>,
    weights: Vec<[f64; 4]>,
}

fn influences(
    doc: &Document<'_>,
    attrs: &Value,
    vertices: usize,
    joints: usize,
) -> Result<Vec<Influences>> {
    let attrs = attrs
        .as_object()
        .ok_or_else(|| err("Invalid glTF attributes."))?;
    let sets = attrs.keys().filter(|k| k.starts_with("JOINTS_")).count();
    if sets == 0 || attrs.keys().filter(|k| k.starts_with("WEIGHTS_")).count() != sets {
        return Err(err(
            "glTF skin requires paired JOINTS and WEIGHTS attributes.",
        ));
    }
    let mut out = Vec::new();
    let mut totals = vec![0.; vertices];
    for i in 0..sets {
        let joint_id = attrs
            .get(&format!("JOINTS_{i}"))
            .ok_or_else(|| err("Missing glTF JOINTS attribute."))?;
        let weight_id = attrs
            .get(&format!("WEIGHTS_{i}"))
            .ok_or_else(|| err("Missing glTF WEIGHTS attribute."))?;
        let spec = get(&doc.json["accessors"], joint_id)?;
        if !matches!(spec["componentType"].as_u64(), Some(5121 | 5123))
            || spec["normalized"] == true
        {
            return Err(err("glTF joint indices must be unsigned integers."));
        }
        let weight_spec = get(&doc.json["accessors"], weight_id)?;
        if weight_spec["componentType"] != 5126
            && !(matches!(weight_spec["componentType"].as_u64(), Some(5121 | 5123))
                && weight_spec["normalized"] == true)
        {
            return Err(err(
                "glTF weights must be floats or normalized unsigned bytes or shorts.",
            ));
        }
        let indices = doc
            .accessor::<4>(joint_id, 4, Some(vertices))?
            .into_iter()
            .map(|row| row.map(|v| v as usize))
            .collect::<Vec<_>>();
        let weights = doc.accessor::<4>(weight_id, 4, Some(vertices))?;
        for (vertex, ws) in weights.iter().enumerate() {
            for (&joint, &weight) in indices[vertex].iter().zip(ws) {
                if weight < 0. || joint >= joints {
                    return Err(err("Invalid glTF skin joint index or weight."));
                }
                totals[vertex] += weight;
            }
        }
        out.push(Influences {
            joints: indices,
            weights,
        });
    }
    if totals.iter().any(|&sum| sum <= 0. || !sum.is_finite()) {
        return Err(err(
            "glTF skin vertex has zero or invalid total joint weight.",
        ));
    }
    for set in &mut out {
        for (weights, &total) in set.weights.iter_mut().zip(&totals) {
            for w in weights {
                *w /= total;
            }
        }
    }
    Ok(out)
}

struct Primitive {
    node: usize,
    skin: Option<usize>,
    mesh: Mesh,
    morphs: Vec<Option<Vec<Point>>>,
    influences: Vec<Influences>,
}

impl Primitive {
    fn posed(
        &self,
        weights: &[f64],
        world: &Matrix,
        skins: &HashMap<usize, Vec<Matrix>>,
    ) -> Result<Mesh> {
        // Mesh clones only the mutable positions. Materials, UVs, colors and
        // indices stay shared across samples.
        let mut mesh = self.mesh.clone();
        for (&weight, target) in weights.iter().zip(&self.morphs) {
            if weight == 0. {
                continue;
            }
            if let Some(deltas) = target {
                for (point, delta) in mesh.positions.iter_mut().zip(deltas) {
                    for i in 0..3 {
                        point[i] += weight * delta[i];
                    }
                }
            }
        }
        let palette = self.skin.map(|id| &skins[&id]);
        for (vertex, point) in mesh.positions.iter_mut().enumerate() {
            *point = if let Some(palette) = palette {
                let mut result = [0.; 3];
                for set in &self.influences {
                    for (&joint, weight) in set.joints[vertex].iter().zip(set.weights[vertex]) {
                        if weight == 0. {
                            continue;
                        }
                        let component = transform(&palette[joint], *point);
                        for i in 0..3 {
                            result[i] += weight * component[i];
                        }
                    }
                }
                result
            } else {
                transform(world, *point)
            };
            if !point.iter().all(|v| v.is_finite()) {
                return Err(err("Invalid glTF scene transform."));
            }
        }
        Ok(mesh)
    }
}

// glTF scenes are disjoint trees. Parent-first traversal also ensures joint
// transforms exist before skinning, regardless of the mesh's traversal order.
fn active_order(nodes: &[Value], scene: Option<&Value>) -> Result<Vec<(usize, Option<usize>)>> {
    let roots = if let Some(scene) = scene {
        items(&scene["nodes"])?
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
    let mut seen = vec![false; nodes.len()];
    let mut stack: Vec<_> = roots.into_iter().rev().map(|i| (i, None)).collect();
    let mut order = Vec::new();
    while let Some((i, parent)) = stack.pop() {
        let used = seen.get_mut(i).ok_or_else(|| err("Invalid glTF node."))?;
        if std::mem::replace(used, true) {
            return Err(err("Invalid or cyclic glTF node tree."));
        }
        order.push((i, parent));
        for child in items(&nodes[i]["children"])?.iter().rev() {
            stack.push((index(child)?, Some(i)));
        }
    }
    Ok(order)
}

pub(in super::super) struct Scene {
    frames: usize,
    clip: Option<Clip>,
    defaults: Vec<Option<Pose>>,
    order: Vec<(usize, Option<usize>)>,
    skins: HashMap<usize, Skin>,
    prims: Vec<Primitive>,
}

impl Scene {
    pub(in super::super) fn open(
        raw: &[u8],
        decode: &dyn Fn(&[u8]) -> Result<PngImage>,
        requested: &str,
        frames: usize,
    ) -> Result<Self> {
        if !(1..=64).contains(&frames) {
            return Err(err("Slice 3D frames must be in 1..64."));
        }
        let doc = Document::open(raw)?;
        let clip = Clip::select(&doc, requested, frames)?;
        let mesh_defs = items(&doc.json["meshes"])?;
        // Bare mesh libraries were historically sliced as a single identity scene.
        let mut nodes = items(&doc.json["nodes"])?;
        let synthetic;
        let scene = if nodes.is_empty() {
            synthetic = (0..mesh_defs.len())
                .map(|i| json!({"mesh":i}))
                .collect::<Vec<_>>();
            nodes = &synthetic;
            None
        } else {
            let selected = optional_index(&doc.json["scene"], 0)?;
            let scenes = items(&doc.json["scenes"])?;
            if scenes.is_empty() {
                None
            } else {
                Some(
                    scenes
                        .get(selected)
                        .ok_or_else(|| err("Invalid glTF scene index."))?,
                )
            }
        };
        let order = active_order(nodes, scene)?;
        let mut active = vec![false; nodes.len()];
        for &(node, _) in &order {
            active[node] = true;
        }
        let mut defaults = vec![None; nodes.len()];
        let mut skins = HashMap::new();
        let mut textures = HashMap::new();
        let mut prims = Vec::new();
        for &(i, _) in &order {
            let node = &nodes[i];
            let mesh = if node["mesh"].is_null() {
                &Value::Null
            } else {
                get(&doc.json["meshes"], &node["mesh"])?
            };
            let pose = Pose::new(node, mesh)?;
            let skin_id = if node["skin"].is_null() {
                None
            } else {
                let id = index(&node["skin"])?;
                if let std::collections::hash_map::Entry::Vacant(entry) = skins.entry(id) {
                    entry.insert(Skin::load(&doc, &node["skin"], &active)?);
                }
                Some(id)
            };
            for p in items(&mesh["primitives"])? {
                let base = doc.primitive(p, &mut textures, decode)?;
                let count = base.positions.len();
                let morphs = items(&p["targets"])?
                    .iter()
                    .map(|t| {
                        t.get("POSITION")
                            .map(|a| doc.accessor::<3>(a, 3, Some(count)))
                            .transpose()
                    })
                    .collect::<Result<Vec<_>>>()?;
                if morphs.len() != pose.weights.len() {
                    return Err(err("Inconsistent glTF morph target count."));
                }
                let influences = if let Some(id) = skin_id {
                    influences(&doc, &p["attributes"], count, skins[&id].joints.len())?
                } else {
                    Vec::new()
                };
                prims.push(Primitive {
                    node: i,
                    skin: skin_id,
                    mesh: base,
                    morphs,
                    influences,
                });
            }
            defaults[i] = Some(pose);
        }
        if prims.is_empty() {
            return Err(err("No triangles in the selected glTF scene."));
        }
        Ok(Self {
            frames,
            clip,
            defaults,
            order,
            skins,
            prims,
        })
    }

    pub(in super::super) fn clip(&self) -> Option<(&str, f64, f64)> {
        self.clip
            .as_ref()
            .map(|c| (c.name.as_str(), c.start, c.duration))
    }

    pub(in super::super) fn pose(&self, f: usize) -> Result<Vec<Mesh>> {
        if f >= self.frames {
            return Err(err("Invalid glTF pose index."));
        }
        let mut worlds = vec![IDENTITY; self.defaults.len()];
        let mut poses = self.defaults.clone();
        if let Some(clip) = &self.clip {
            clip.apply(
                clip.start + clip.duration * f as f64 / self.frames as f64,
                &mut poses,
            )?;
        }
        for &(i, parent) in &self.order {
            let local = poses[i].as_ref().unwrap().local();
            worlds[i] = parent.map_or(local, |p| multiply(&worlds[p], &local));
        }
        let palettes = self
            .skins
            .iter()
            .map(|(&id, skin)| (id, skin.palette(&worlds)))
            .collect();
        self.prims
            .iter()
            .map(|p| {
                p.posed(
                    &poses[p.node].as_ref().unwrap().weights,
                    &worlds[p.node],
                    &palettes,
                )
            })
            .collect()
    }
}
