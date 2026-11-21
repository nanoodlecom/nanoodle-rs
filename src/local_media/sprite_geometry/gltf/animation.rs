//! Evaluate authored glTF keyframes in local node space.
use super::*;
use std::collections::HashSet;

#[derive(Clone)]
pub(super) struct Pose {
    matrix: Option<Matrix>,
    translation: [f64; 3],
    rotation: [f64; 4],
    scale: [f64; 3],
    pub weights: Vec<f64>,
}

impl Pose {
    pub(super) fn new(node: &Value, mesh: &Value) -> Result<Self> {
        let message = "Invalid glTF scene transform.";
        let matrix = if node["matrix"].is_null() {
            None
        } else {
            Some(numbers(&node["matrix"], IDENTITY, message)?)
        };
        let mut rotation = numbers(&node["rotation"], [0., 0., 0., 1.], message)?;
        normalize(&mut rotation)?;
        let weights = node.get("weights").unwrap_or(&mesh["weights"]);
        let count = items(&mesh["primitives"][0]["targets"])?.len();
        let weights = if weights.is_null() {
            vec![0.; count]
        } else {
            items(weights)?
                .iter()
                .map(|v| {
                    v.as_f64()
                        .filter(|f| f.is_finite())
                        .ok_or_else(|| err("Invalid glTF morph weights."))
                })
                .collect::<Result<Vec<_>>>()?
        };
        if weights.len() != count {
            return Err(err("Invalid glTF morph weight count."));
        }
        Ok(Self {
            matrix,
            translation: numbers(&node["translation"], [0.; 3], message)?,
            rotation,
            scale: numbers(&node["scale"], [1.; 3], message)?,
            weights,
        })
    }

    pub(super) fn local(&self) -> Matrix {
        self.matrix
            .unwrap_or_else(|| trs(self.translation, self.rotation, self.scale))
    }
}

fn normalize(q: &mut [f64]) -> Result<()> {
    let length = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    if length == 0. || !length.is_finite() {
        return Err(err("Invalid zero or non-finite glTF rotation."));
    }
    for v in q {
        *v /= length;
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Path {
    Translation,
    Rotation,
    Scale,
    Weights,
}

#[derive(PartialEq)]
enum Interpolation {
    Step,
    Linear,
    Cubic,
}

struct Track {
    path: Path,
    node: usize,
    width: usize,
    interpolation: Interpolation,
    times: Vec<f64>,
    values: Vec<f64>,
}

impl Track {
    fn read(doc: &Document<'_>, anim: &Value, channel: &Value) -> Result<Self> {
        let target = &channel["target"];
        let node = index(&target["node"])?;
        let node_spec = get(&doc.json["nodes"], &target["node"])?;
        let path = match target["path"].as_str() {
            Some("translation") => Path::Translation,
            Some("rotation") => Path::Rotation,
            Some("scale") => Path::Scale,
            Some("weights") => Path::Weights,
            _ => return Err(err("Unsupported glTF animation target.")),
        };
        if path != Path::Weights && !node_spec["matrix"].is_null() {
            return Err(err("glTF animation cannot target TRS on a matrix node."));
        }
        let sampler = get(&anim["samplers"], &channel["sampler"])?;
        let interpolation = match sampler
            .get("interpolation")
            .and_then(Value::as_str)
            .unwrap_or("LINEAR")
        {
            "STEP" => Interpolation::Step,
            "LINEAR" => Interpolation::Linear,
            "CUBICSPLINE" => Interpolation::Cubic,
            _ => return Err(err("Unsupported glTF animation interpolation.")),
        };
        if get(&doc.json["accessors"], &sampler["input"])?["componentType"] != 5126 {
            return Err(err("glTF animation input times must use floats."));
        }
        let times: Vec<_> = doc
            .accessor::<1>(&sampler["input"], 1, None)?
            .into_iter()
            .map(|[v]| v)
            .collect();
        if times[0] < 0. || times.windows(2).any(|t| t[0] >= t[1]) {
            return Err(err("glTF animation key times must be strictly increasing."));
        }
        if interpolation == Interpolation::Cubic && times.len() < 2 {
            return Err(err("CUBICSPLINE animation requires at least two keys."));
        }
        let keys = times.len()
            * if interpolation == Interpolation::Cubic {
                3
            } else {
                1
            };
        let output = &sampler["output"];
        let (width, mut values): (usize, Vec<f64>) = match path {
            Path::Translation | Path::Scale => (
                3,
                doc.accessor::<3>(output, 3, Some(keys))?
                    .into_iter()
                    .flatten()
                    .collect(),
            ),
            Path::Rotation => (
                4,
                doc.accessor::<4>(output, 4, Some(keys))?
                    .into_iter()
                    .flatten()
                    .collect(),
            ),
            Path::Weights => {
                let mesh = get(&doc.json["meshes"], &node_spec["mesh"])?;
                let width = items(&mesh["primitives"][0]["targets"])?.len();
                let count = keys
                    .checked_mul(width)
                    .filter(|v| *v > 0)
                    .ok_or_else(|| err("Invalid glTF morph animation width."))?;
                (
                    width,
                    doc.accessor::<1>(output, 1, Some(count))?
                        .into_iter()
                        .flatten()
                        .collect(),
                )
            }
        };
        if path == Path::Rotation && interpolation == Interpolation::Linear {
            for key in values.chunks_exact_mut(4) {
                normalize(key)?;
            }
        }
        Ok(Self {
            path,
            node,
            width,
            interpolation,
            times,
            values,
        })
    }

    fn values_at(&self, key: usize, cubic_slot: usize) -> &[f64] {
        let offset = if self.interpolation == Interpolation::Cubic {
            key * 3 + cubic_slot
        } else {
            key
        } * self.width;
        &self.values[offset..offset + self.width]
    }

    fn sample(&self, time: f64, pose: &mut Pose) -> Result<()> {
        let dst: &mut [f64] = match self.path {
            Path::Translation => &mut pose.translation,
            Path::Rotation => &mut pose.rotation,
            Path::Scale => &mut pose.scale,
            Path::Weights => &mut pose.weights,
        };
        let upper = self.times.partition_point(|&key| key <= time);
        let lower = upper.saturating_sub(1);
        let a = self.values_at(lower, 1);
        if upper == 0
            || upper == self.times.len()
            || time == self.times[lower]
            || self.interpolation == Interpolation::Step
        {
            dst.copy_from_slice(a);
        } else {
            let dt = self.times[upper] - self.times[lower];
            let u = (time - self.times[lower]) / dt;
            let b = self.values_at(upper, 1);
            match self.interpolation {
                Interpolation::Cubic => {
                    let out_tangent = self.values_at(lower, 2);
                    let in_tangent = self.values_at(upper, 0);
                    let (u2, u3) = (u * u, u * u * u);
                    for i in 0..self.width {
                        dst[i] = (2. * u3 - 3. * u2 + 1.) * a[i]
                            + (u3 - 2. * u2 + u) * dt * out_tangent[i]
                            + (-2. * u3 + 3. * u2) * b[i]
                            + (u3 - u2) * dt * in_tangent[i];
                    }
                }
                Interpolation::Linear if self.path == Path::Rotation => {
                    // glTF rotation keys are unit quaternions. Shortest-arc slerp
                    // negates the second key when its representation is antipodal.
                    let dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
                    let sign = if dot < 0. { -1. } else { 1. };
                    let theta = dot.abs().min(1.).acos();
                    let (left, right) = if theta <= 1e-5 {
                        (1. - u, u)
                    } else {
                        (
                            ((1. - u) * theta).sin() / theta.sin(),
                            (u * theta).sin() / theta.sin(),
                        )
                    };
                    for i in 0..4 {
                        dst[i] = left * a[i] + sign * right * b[i];
                    }
                }
                _ => {
                    for i in 0..self.width {
                        dst[i] = a[i] + u * (b[i] - a[i]);
                    }
                }
            }
        }
        if self.path == Path::Rotation {
            normalize(dst)?;
        }
        Ok(())
    }
}

pub(super) struct Clip {
    pub name: String,
    pub start: f64,
    pub duration: f64,
    tracks: Vec<Track>,
}

impl Clip {
    pub(super) fn select(
        doc: &Document<'_>,
        requested: &str,
        frames: usize,
    ) -> Result<Option<Self>> {
        if requested.is_empty() && frames == 1 {
            return Ok(None);
        }
        let animations = items(&doc.json["animations"])?;
        let names: Vec<_> = animations
            .iter()
            .enumerate()
            .map(|(i, v)| {
                v["name"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("Animation {}", i + 1))
            })
            .collect();
        if names.is_empty() {
            return Err(err(
                "No animation clips in this glTF. Use frames=1 and a blank clip for its rest pose.",
            ));
        }
        let i = if requested.is_empty() {
            0
        } else {
            names.iter().position(|s| s == requested).ok_or_else(|| {
                err(format!(
                    "Unknown glTF animation clip '{requested}'. Available clips: {}.",
                    names.join(", ")
                ))
            })?
        };
        let anim = &animations[i];
        let mut tracks = Vec::new();
        let mut seen = HashSet::new();
        let mut start = f64::INFINITY;
        let mut end = f64::NEG_INFINITY;
        for channel in items(&anim["channels"])? {
            if channel["target"]["node"].is_null() {
                continue;
            }
            let track = Track::read(doc, anim, channel)?;
            if !seen.insert((track.node, track.path)) {
                return Err(err("Duplicate glTF animation target."));
            }
            start = start.min(track.times[0]);
            end = end.max(*track.times.last().unwrap());
            tracks.push(track);
        }
        if tracks.is_empty() {
            return Err(err(
                "Selected glTF clip has no supported node animation channels.",
            ));
        }
        if frames > 1 && end <= start {
            return Err(err(
                "glTF clip has zero duration. Use frames=1 to sample its first pose.",
            ));
        }
        Ok(Some(Self {
            name: names[i].clone(),
            start,
            duration: end - start,
            tracks,
        }))
    }

    pub(super) fn apply(&self, time: f64, poses: &mut [Option<Pose>]) -> Result<()> {
        for track in &self.tracks {
            if let Some(pose) = &mut poses[track.node] {
                track.sample(time, pose)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rotation(interpolation: Interpolation, values: Vec<f64>) -> Track {
        Track {
            path: Path::Rotation,
            node: 0,
            width: 4,
            interpolation,
            times: vec![0., 2.],
            values,
        }
    }

    #[test]
    fn linear_rotation_takes_shortest_path_even_for_negative_quaternions() {
        let s = 0.5_f64.sqrt();
        let track = rotation(Interpolation::Linear, vec![0., 0., 0., 1., 0., 0., -s, -s]);
        let mut pose = Pose::new(&Value::Null, &Value::Null).unwrap();
        track.sample(1., &mut pose).unwrap();
        let m = pose.local();
        assert!((m[0] - s).abs() < 1e-12 && (m[1] - s).abs() < 1e-12);
        track.sample(-1., &mut pose).unwrap();
        assert_eq!(pose.rotation, [0., 0., 0., 1.]);
    }

    #[test]
    fn cubic_rotation_scales_tangents_by_key_interval_and_renormalizes() {
        let track = rotation(
            Interpolation::Cubic,
            vec![
                0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
                0., 0., 0.,
            ],
        );
        let mut pose = Pose::new(&Value::Null, &Value::Null).unwrap();
        track.sample(1., &mut pose).unwrap();
        let expected_norm = 1.25_f64.sqrt();
        assert!((pose.rotation[2] - 0.5 / expected_norm).abs() < 1e-12);
        assert!((pose.rotation[3] - 1. / expected_norm).abs() < 1e-12);
    }
}
