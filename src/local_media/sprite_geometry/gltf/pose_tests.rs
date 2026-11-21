use super::*;
use crate::local_media::bytes_to_data_url;
use serde_json::json;

type Samples = (Vec<Vec<Mesh>>, Option<(String, f64, f64)>);

fn parse_poses(
    raw: &[u8],
    decode: &dyn Fn(&[u8]) -> Result<PngImage>,
    clip: &str,
    frames: usize,
) -> Result<Samples> {
    let scene = Scene::open(raw, decode, clip, frames)?;
    Ok((
        (0..frames).map(|i| scene.pose(i)).collect::<Result<_>>()?,
        scene.clip().map(|(n, t, d)| (n.to_owned(), t, d)),
    ))
}

fn unused_image(_: &[u8]) -> Result<PngImage> {
    panic!("untextured fixture tried to decode an image")
}

struct Fixture {
    views: Vec<Value>,
    accessors: Vec<Value>,
    bytes: Vec<u8>,
}

impl Fixture {
    fn view(&mut self, raw: &[u8]) -> usize {
        while self.bytes.len() % 4 != 0 {
            self.bytes.push(0);
        }
        let index = self.views.len();
        self.views
            .push(json!({"buffer":0, "byteOffset":self.bytes.len(), "byteLength":raw.len()}));
        self.bytes.extend(raw);
        index
    }
    fn add(&mut self, accessor: Value) -> usize {
        let id = self.accessors.len();
        self.accessors.push(accessor);
        id
    }
    fn floats(&mut self, kind: &str, n: usize, raw: &[f32]) -> usize {
        let bytes = raw.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        let view = self.view(&bytes);
        self.add(json!({"bufferView":view,"count":n,"type":kind,"componentType":5126}))
    }
    fn bytes4(&mut self, row: [u8; 4], norm: bool) -> usize {
        let view = self.view(&row.repeat(3));
        self.add(json!({"bufferView":view,"count":3,"type":"VEC4","componentType":5121,"normalized":norm}))
    }
}

fn morph_scene() -> Value {
    let mut f = Fixture {
        views: vec![],
        accessors: vec![],
        bytes: vec![],
    };
    let pos = f.floats("VEC3", 3, &[0., 0., 0., 1., 1., 0., 1., 0., 0.]);
    let j0 = f.bytes4([0, 0, 0, 0], false);
    let j1 = f.bytes4([1, 0, 0, 0], false);
    let weights = f.bytes4([96, 0, 0, 0], true);
    let sparse_index = f.view(&[1]);
    let delta = f.view(
        &[2_f32, 0., 0.]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let morph = f.add(json!({
        "type":"VEC3","componentType":5126,"count":3,"sparse":{
            "count":1,"indices":{"componentType":5121,"bufferView":sparse_index},
            "values":{"bufferView":delta}
        }
    }));
    let times = f.floats("SCALAR", 2, &[1., 3.]);
    let values = f.floats("SCALAR", 2, &[0., 1.]);
    json!({
        "asset":{"version":"2.0"},"scenes":[{"nodes":[0]}],"scene":0,
        "nodes":[{"translation":[0,0,5],"children":[1,2,3]},
            {"translation":[50,0,0],"mesh":0,"skin":0,"weights":[0.5]},
            {"scale":[2,1,1]},{"translation":[4,0,0]}],
        "meshes":[{"weights":[0.25],"primitives":[{"attributes":{
            "POSITION":pos,"JOINTS_0":j0,"JOINTS_1":j1,"WEIGHTS_0":weights,"WEIGHTS_1":weights
        },"targets":[{"POSITION":morph}]}]}],
        "skins":[{"joints":[2,3]}],
        "animations":[{"channels":[{"sampler":0,"target":{"node":1,"path":"weights"}}],
            "samplers":[{"input":times,"output":values}]}],
        "buffers":[{"uri":bytes_to_data_url(&f.bytes,Some("application/octet-stream")),
                    "byteLength":f.bytes.len()}],"bufferViews":f.views,"accessors":f.accessors
    })
}

#[test]
fn sparse_morph_precedes_both_joint_sets_and_uses_instance_weights() {
    let mut scene = morph_scene();
    let poses = |scene: &Value, clip, frames| {
        parse_poses(
            &serde_json::to_vec(scene).unwrap(),
            &unused_image,
            clip,
            frames,
        )
        .unwrap()
    };
    let (rest, selection) = poses(&scene, "", 1);
    assert_eq!(selection, None);
    assert_eq!(
        rest[0][0].positions,
        [[2., 0., 5.], [5., 1., 5.], [3.5, 0., 5.]]
    );
    let (frames, selection) = poses(&scene, "Animation 1", 2);
    assert_eq!(selection, Some(("Animation 1".to_owned(), 1., 2.)));
    assert_eq!(frames[0][0].positions[1], [3.5, 1., 5.]);
    assert_eq!(frames[1][0].positions[1], [5., 1., 5.]);
    scene["nodes"][1].as_object_mut().unwrap().remove("weights");
    assert_eq!(poses(&scene, "", 1).0[0][0].positions[1], [4.25, 1., 5.]);
}

#[test]
fn rejects_missing_clip_nonincreasing_time_and_unpaired_joint_set() {
    let scene = morph_scene();
    let failure = |v: &Value, clip, n| {
        parse_poses(&serde_json::to_vec(v).unwrap(), &unused_image, clip, n)
            .err()
            .expect("expected validation error")
            .0
    };
    assert!(failure(&scene, "not an animation", 2).contains("Animation 1"));
    let mut bad = scene.clone();
    bad["animations"][0]["samplers"][0]["input"] =
        bad["animations"][0]["samplers"][0]["output"].clone();
    // Make the time keys both zero.
    let accessor = bad["animations"][0]["samplers"][0]["input"]
        .as_u64()
        .unwrap() as usize;
    bad["accessors"][accessor]
        .as_object_mut()
        .unwrap()
        .remove("bufferView");
    assert!(failure(&bad, "", 2).contains("strictly increasing"));
    let mut bad = scene;
    bad["meshes"][0]["primitives"][0]["attributes"]
        .as_object_mut()
        .unwrap()
        .remove("WEIGHTS_1");
    assert!(failure(&bad, "", 1).contains("paired"));
}
