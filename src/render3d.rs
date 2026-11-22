//! render3d — local 3D rendering node (Edge Styles S2), Rust twin of
//! nanoodle-js `render3d.mjs`.
//!
//! ONE scene implementation (render3d-assets/scene.html, three.js) renders
//! a GLB with the node's fields: camera azimuth/elevation/distance, a
//! clip plane (slice-3D), explode over the GLB's scene-graph parts, part
//! isolation, animation stepping, turntable. The JS executor drives the
//! scene over CDP; this twin uses Chromium's own headless screenshot with
//! a virtual-time budget and the scene's auto mode (fields in the URL
//! fragment) — no new dependencies (the crate carries serde_json + ureq
//! only). Chromium discovery mirrors the ffmpeg contract: NANOODLE_CHROME
//! / CHROME_PATH, the Playwright browser cache, then PATH. Video mode
//! assembles frames with ffmpeg, same as the other local nodes.

use crate::{input_text, Ctx, NanoodleError, Node, NodeOutcome, Result};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const SCENE_HTML: &str = include_str!("render3d-assets/scene.html");
const THREE_JS: &[u8] = include_bytes!("render3d-assets/three.module.js");
const GLTF_LOADER: &[u8] = include_bytes!("render3d-assets/addons/loaders/GLTFLoader.js");
const ROOM_ENV: &[u8] = include_bytes!("render3d-assets/addons/environments/RoomEnvironment.js");
const BG_UTILS: &[u8] = include_bytes!("render3d-assets/addons/utils/BufferGeometryUtils.js");

fn err(msg: impl Into<String>) -> NanoodleError {
    NanoodleError(msg.into())
}

fn b64_encode(bytes: &[u8]) -> String {
    const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

fn field_f64(node: &Node, key: &str, def: f64) -> f64 {
    match node.fields.get(key) {
        Some(Value::Number(n)) => n.as_f64().unwrap_or(def),
        Some(Value::String(s)) => s.parse::<f64>().unwrap_or(def),
        _ => def,
    }
}

fn field_str<'a>(node: &'a Node, key: &str) -> Option<String> {
    match node.fields.get(key) {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Number(n)) => Some(n.to_string()),
        Some(Value::Bool(b)) => Some(b.to_string()),
        _ => None,
    }
}

/// Node fields → scene fields. Mirror of JS `render3dSceneFields`.
pub fn scene_fields(node: &Node) -> Value {
    let mut f = Map::new();
    f.insert("azimuth".into(), json!(field_f64(node, "azimuth", 35.0)));
    f.insert("elevation".into(), json!(field_f64(node, "elevation", 18.0)));
    f.insert("distance".into(), json!(field_f64(node, "distance", 1.9)));
    f.insert("width".into(), json!((field_f64(node, "width", 800.0) as i64).max(16)));
    f.insert("height".into(), json!((field_f64(node, "height", 600.0) as i64).max(16)));
    f.insert("background".into(), json!(field_str(node, "background").filter(|s| !s.is_empty()).unwrap_or_else(|| "dark".to_string())));
    f.insert("explode".into(), json!(field_f64(node, "explode", 0.0)));
    if let Some(axis) = field_str(node, "clipAxis").filter(|s| !s.is_empty() && s != "none") {
        f.insert("clipAxis".into(), json!(axis));
        f.insert("clipPos".into(), json!(field_f64(node, "clipPos", 0.0)));
    }
    if let Some(raw) = field_str(node, "isolate").map(|s| s.trim().to_string()).filter(|s| !s.is_empty()) {
        if let Ok(n) = raw.parse::<i64>() {
            f.insert("isolate".into(), json!(n));
        } else {
            f.insert("isolate".into(), json!(raw));
        }
    }
    if let Some(raw) = field_str(node, "animClip").filter(|s| !s.is_empty()) {
        if let Ok(n) = raw.trim().parse::<i64>() {
            f.insert("animClip".into(), json!(n));
        } else {
            f.insert("animClip".into(), json!(raw));
        }
    }
    f.insert("animTime".into(), json!(field_f64(node, "animTime", 0.0)));
    Value::Object(f)
}

pub fn find_chromium() -> Option<PathBuf> {
    for var in ["NANOODLE_CHROME", "CHROME_PATH"] {
        if let Ok(p) = std::env::var(var) {
            let pb = PathBuf::from(p);
            if pb.exists() {
                return Some(pb);
            }
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        let cache = PathBuf::from(home).join(".cache/ms-playwright");
        if let Ok(rd) = std::fs::read_dir(&cache) {
            let mut dirs: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
            dirs.sort();
            for d in dirs.iter().rev() {
                for sub in ["chrome-headless-shell-linux64/chrome-headless-shell", "chrome-linux64/chrome", "chrome-mac/Chromium.app/Contents/MacOS/Chromium"] {
                    let p = d.join(sub);
                    if p.exists() {
                        return Some(p);
                    }
                }
            }
        }
    }
    if let Ok(path) = std::env::var("PATH") {
        for dir in path.split(':') {
            for name in ["chromium", "chromium-browser", "google-chrome", "google-chrome-stable"] {
                let p = PathBuf::from(dir).join(name);
                if p.exists() {
                    return Some(p);
                }
            }
        }
    }
    None
}

fn no_chromium_error() -> NanoodleError {
    err("render3d needs a Chromium binary (not found). Set NANOODLE_CHROME to one, \
         install Playwright's Chromium (npx playwright install chromium), or run this graph in the nanoodle browser app.")
}

/// Serve the scene assets + the GLB on 127.0.0.1 until `stop` is set.
fn serve_scene(glb: Vec<u8>, stop: Arc<AtomicBool>) -> Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| err(format!("render3d: bind failed: {e}")))?;
    let port = listener.local_addr().map_err(|e| err(format!("render3d: {e}")))?.port();
    listener.set_nonblocking(true).ok();
    std::thread::spawn(move || {
        while !stop.load(Ordering::Relaxed) {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    let mut buf = [0u8; 4096];
                    let n = stream.read(&mut buf).unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]);
                    let path = req.split_whitespace().nth(1).unwrap_or("/");
                    let (ctype, body): (&str, &[u8]) = match path {
                        "/model.glb" => ("model/gltf-binary", glb.as_slice()),
                        "/three.module.js" => ("text/javascript", THREE_JS),
                        "/addons/loaders/GLTFLoader.js" => ("text/javascript", GLTF_LOADER),
                        "/addons/environments/RoomEnvironment.js" => ("text/javascript", ROOM_ENV),
                        "/addons/utils/BufferGeometryUtils.js" => ("text/javascript", BG_UTILS),
                        _ => ("text/html", SCENE_HTML.as_bytes()),
                    };
                    let head = format!("HTTP/1.1 200 OK\r\ncontent-type: {ctype}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n", body.len());
                    let _ = stream.write_all(head.as_bytes());
                    let _ = stream.write_all(body);
                    let _ = stream.flush();
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(_) => break,
            }
        }
    });
    Ok(port)
}

fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 2);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn run_with_timeout(mut cmd: Command, timeout: Duration) -> Result<std::process::ExitStatus> {
    let mut child = cmd.stdout(Stdio::null()).stderr(Stdio::null()).spawn()
        .map_err(|e| err(format!("render3d: could not launch Chromium: {e}")))?;
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    return Err(err("render3d: Chromium timed out"));
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => return Err(err(format!("render3d: {e}"))),
        }
    }
}

fn screenshot(exe: &PathBuf, port: u16, fields: &Value, out: &PathBuf) -> Result<()> {
    let w = fields.get("width").and_then(|v| v.as_i64()).unwrap_or(800);
    let h = fields.get("height").and_then(|v| v.as_i64()).unwrap_or(600);
    let frag = percent_encode(&fields.to_string());
    let url = format!("http://127.0.0.1:{port}/scene.html#f={frag}");
    let mut cmd = Command::new(exe);
    cmd.args([
        "--headless", "--no-sandbox", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
        "--hide-scrollbars", "--virtual-time-budget=20000",
        &format!("--window-size={w},{h}"),
        &format!("--screenshot={}", out.display()),
        &url,
    ]);
    let status = run_with_timeout(cmd, Duration::from_secs(90))?;
    if !status.success() || !out.exists() {
        return Err(err("render3d: Chromium screenshot failed"));
    }
    Ok(())
}

fn model_bytes(src: &str) -> Result<Vec<u8>> {
    if let Some(rest) = src.strip_prefix("data:") {
        let (meta, body) = rest.split_once(',').ok_or_else(|| err("render3d: malformed data: URL"))?;
        if meta.contains(";base64") {
            // Reuse the crate's decoder via audio's sibling in local_media style:
            return decode_b64(body);
        }
        return Ok(body.as_bytes().to_vec());
    }
    let mut resp = ureq::get(src).call().map_err(|e| err(format!("render3d: could not download the model: {e}")))?;
    let bytes = resp.body_mut().read_to_vec().map_err(|e| err(format!("render3d: could not read the model: {e}")))?;
    Ok(bytes)
}

fn decode_b64(s: &str) -> Result<Vec<u8>> {
    let mut table = [255u8; 256];
    for (i, c) in b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".iter().enumerate() {
        table[*c as usize] = i as u8;
    }
    let clean: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    let mut out = Vec::with_capacity(clean.len() * 3 / 4);
    for chunk in clean.chunks(4) {
        let mut vals = [0u8; 4];
        let mut pad = 0;
        for (i, c) in chunk.iter().enumerate() {
            if *c == b'=' {
                pad += 1;
            } else {
                vals[i] = table[*c as usize];
                if vals[i] == 255 {
                    return Err(err("render3d: invalid base64 in data: URL"));
                }
            }
        }
        let n = ((vals[0] as u32) << 18) | ((vals[1] as u32) << 12) | ((vals[2] as u32) << 6) | vals[3] as u32;
        out.push((n >> 16) as u8);
        if pad < 2 {
            out.push((n >> 8) as u8);
        }
        if pad < 1 {
            out.push(n as u8);
        }
    }
    Ok(out)
}

pub fn run_render3d(node: &Node, inputs: &HashMap<String, Value>, _ctx: &Ctx) -> Result<NodeOutcome> {
    let src = match input_text(inputs, "model") {
        Some(s) if !s.is_empty() => s,
        _ => return Err(err("render3d: connect a model first")),
    };
    let exe = find_chromium().ok_or_else(no_chromium_error)?;
    let glb = model_bytes(&src)?;
    let base = scene_fields(node);
    let stop = Arc::new(AtomicBool::new(false));
    let port = serve_scene(glb, stop.clone())?;
    let mode = field_str(node, "mode").unwrap_or_else(|| "image".to_string());
    let result = render_with(node, &exe, port, &base, &mode);
    stop.store(true, Ordering::Relaxed);
    let (port_name, data_url) = result?;
    let mut m = Map::new();
    m.insert(port_name.to_string(), json!(data_url));
    Ok(NodeOutcome::Done(m, Some(0.0)))
}

fn render_with(node: &Node, exe: &PathBuf, port: u16, base: &Value, mode: &str) -> Result<(&'static str, String)> {
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("nanoodle-render3d-{}-{stamp}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| err(format!("render3d: {e}")))?;
    let cleanup = |dir: &PathBuf| {
        let _ = std::fs::remove_dir_all(dir);
    };
    if mode != "video" {
        let out = dir.join("frame.png");
        let r = screenshot(exe, port, base, &out)
            .and_then(|_| std::fs::read(&out).map_err(|e| err(format!("render3d: {e}"))))
            .map(|png| ("image", format!("data:image/png;base64,{}", b64_encode(&png))));
        cleanup(&dir);
        return r;
    }
    // Video: sweep turntable / clip timeline / slice plane across frames.
    let fps = field_f64(node, "fps", 24.0).max(1.0);
    let duration = field_f64(node, "duration", 2.0).max(0.1);
    let frames = ((fps * duration).round() as i64).max(2);
    let turntable = if node.fields.contains_key("turntable") { field_f64(node, "turntable", 360.0) } else { 360.0 };
    let has_clip_sweep = match node.fields.get("clipSweep") {
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => s == "true",
        _ => false,
    };
    let has_anim = field_str(node, "animClip").map(|s| !s.is_empty()).unwrap_or(false);
    let base_az = base.get("azimuth").and_then(|v| v.as_f64()).unwrap_or(35.0);
    for i in 0..frames {
        let t = i as f64 / (frames - 1) as f64;
        let mut f = base.clone();
        let obj = f.as_object_mut().expect("scene fields object");
        if turntable != 0.0 {
            obj.insert("azimuth".into(), json!(base_az + turntable * t));
        }
        if has_anim {
            obj.insert("animTime".into(), json!(duration * t));
        }
        if has_clip_sweep {
            obj.insert("clipPos".into(), json!(1.05 - 2.1 * t));
        }
        let out = dir.join(format!("frame-{i:03}.png"));
        if let Err(e) = screenshot(exe, port, &f, &out) {
            cleanup(&dir);
            return Err(e);
        }
    }
    let out_mp4 = dir.join("out.mp4");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-framerate", &format!("{fps}"), "-i",
               &dir.join("frame-%03d.png").display().to_string(),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
               &out_mp4.display().to_string()])
        .stdout(Stdio::null()).stderr(Stdio::null())
        .status();
    let r = match status {
        Ok(s) if s.success() => std::fs::read(&out_mp4)
            .map(|mp4| ("video", format!("data:video/mp4;base64,{}", b64_encode(&mp4))))
            .map_err(|e| err(format!("render3d: {e}"))),
        Ok(_) => Err(err("render3d: ffmpeg failed to assemble the video")),
        Err(_) => Err(err("render3d video mode needs ffmpeg on PATH (not found). Install ffmpeg, or use image mode.")),
    };
    cleanup(&dir);
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_fields_defaults_and_mapping() {
        let mut fields = Map::new();
        fields.insert("clipAxis".into(), json!("y"));
        fields.insert("clipPos".into(), json!("0.5"));
        fields.insert("isolate".into(), json!("2"));
        let node = Node { id: "r".into(), ty: "render3d".into(), name: None, fields, unknown: false };
        let f = scene_fields(&node);
        assert_eq!(f["azimuth"], json!(35.0));
        assert_eq!(f["clipAxis"], json!("y"));
        assert_eq!(f["clipPos"], json!(0.5));
        assert_eq!(f["isolate"], json!(2));
        let empty = Node { id: "r".into(), ty: "render3d".into(), name: None, fields: Map::new(), unknown: false };
        let d = scene_fields(&empty);
        assert!(d.get("clipAxis").is_none());
        assert!(d.get("isolate").is_none());
    }

    #[test]
    fn b64_roundtrip_edges() {
        assert_eq!(b64_encode(b""), "");
        assert_eq!(b64_encode(b"f"), "Zg==");
        assert_eq!(b64_encode(b"fo"), "Zm8=");
        assert_eq!(b64_encode(b"foo"), "Zm9v");
        assert_eq!(decode_b64("Zm9v").unwrap(), b"foo");
    }

    /// The two-cube GLB from the JS suite (red cube + blue cube, two
    /// scene-graph parts), assembled byte-wise.
    fn two_cube_glb() -> Vec<u8> {
        let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
            ([1.0, 0.0, 0.0], [[1.0, -1.0, 1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 1.0, 1.0]]),
            ([-1.0, 0.0, 0.0], [[-1.0, -1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, 1.0], [-1.0, 1.0, -1.0]]),
            ([0.0, 1.0, 0.0], [[-1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0]]),
            ([0.0, -1.0, 0.0], [[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [-1.0, -1.0, 1.0]]),
            ([0.0, 0.0, 1.0], [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]]),
            ([0.0, 0.0, -1.0], [[1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [1.0, 1.0, -1.0]]),
        ];
        let mut pos: Vec<f32> = Vec::new();
        let mut nor: Vec<f32> = Vec::new();
        let mut idx: Vec<u16> = Vec::new();
        for (fi, (n, corners)) in faces.iter().enumerate() {
            for c in corners {
                pos.extend([c[0] * 0.5, c[1] * 0.5, c[2] * 0.5]);
                nor.extend(*n);
            }
            let b = (fi * 4) as u16;
            idx.extend([b, b + 1, b + 2, b, b + 2, b + 3]);
        }
        let mut bin: Vec<u8> = Vec::new();
        for v in pos.iter().chain(nor.iter()) {
            bin.extend(v.to_le_bytes());
        }
        let pos_len = pos.len() * 4;
        let nor_len = nor.len() * 4;
        for v in &idx {
            bin.extend(v.to_le_bytes());
        }
        let gltf = json!({
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0, 1]}],
            "nodes": [
                {"mesh": 0, "translation": [-0.75, 0, 0], "name": "cube_red"},
                {"mesh": 1, "translation": [0.75, 0, 0], "name": "cube_blue"}
            ],
            "meshes": [
                {"name": "red", "primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "material": 0}]},
                {"name": "blue", "primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "material": 1}]}
            ],
            "materials": [
                {"name": "red", "pbrMetallicRoughness": {"baseColorFactor": [1, 0.1, 0.1, 1], "metallicFactor": 0, "roughnessFactor": 0.8}, "doubleSided": true},
                {"name": "blue", "pbrMetallicRoughness": {"baseColorFactor": [0.1, 0.2, 1, 1], "metallicFactor": 0, "roughnessFactor": 0.8}, "doubleSided": true}
            ],
            "buffers": [{"byteLength": bin.len()}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": pos_len},
                {"buffer": 0, "byteOffset": pos_len, "byteLength": nor_len},
                {"buffer": 0, "byteOffset": pos_len + nor_len, "byteLength": idx.len() * 2}
            ],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 24, "type": "VEC3", "min": [-0.5, -0.5, -0.5], "max": [0.5, 0.5, 0.5]},
                {"bufferView": 1, "componentType": 5126, "count": 24, "type": "VEC3"},
                {"bufferView": 2, "componentType": 5123, "count": 36, "type": "SCALAR"}
            ]
        });
        let mut j = serde_json::to_vec(&gltf).unwrap();
        while j.len() % 4 != 0 {
            j.push(b' ');
        }
        while bin.len() % 4 != 0 {
            bin.push(0);
        }
        let mut out: Vec<u8> = Vec::new();
        out.extend(0x46546c67u32.to_le_bytes());
        out.extend(2u32.to_le_bytes());
        out.extend(((12 + 8 + j.len() + 8 + bin.len()) as u32).to_le_bytes());
        out.extend((j.len() as u32).to_le_bytes());
        out.extend(b"JSON");
        out.extend(&j);
        out.extend((bin.len() as u32).to_le_bytes());
        out.extend(b"BIN\0");
        out.extend(&bin);
        out
    }

    fn render(fields: Map<String, Value>) -> Option<Vec<u8>> {
        if find_chromium().is_none() {
            return None; // skip: no chromium discoverable
        }
        let glb_url = format!("data:model/gltf-binary;base64,{}", b64_encode(&two_cube_glb()));
        let mut inputs = HashMap::new();
        inputs.insert("model".to_string(), json!(glb_url));
        let node = Node { id: "r".into(), ty: "render3d".into(), name: None, fields, unknown: false };
        let ctx = Ctx::new("https://nano-gpt.com/api", "fixture");
        match run_render3d(&node, &inputs, &ctx).expect("render3d run") {
            NodeOutcome::Done(m, _) => {
                let url = m.get("image").and_then(|v| v.as_str()).expect("image port").to_string();
                Some(decode_b64(url.split_once(',').unwrap().1).unwrap())
            }
            _ => panic!("unexpected gated outcome"),
        }
    }

    fn png_dims(png: &[u8]) -> (u32, u32) {
        assert_eq!(&png[0..8], b"\x89PNG\r\n\x1a\n");
        (u32::from_be_bytes(png[16..20].try_into().unwrap()), u32::from_be_bytes(png[20..24].try_into().unwrap()))
    }

    /// Count red/blue-dominant pixels by decoding the PNG through ffmpeg
    /// (the crate's own heavy fallback) to raw RGBA.
    fn color_counts(png: &[u8]) -> Option<(usize, usize)> {
        let dir = std::env::temp_dir().join(format!("nanoodle-rs-r3d-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).ok()?;
        let p = dir.join("f.png");
        std::fs::write(&p, png).ok()?;
        let raw = Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-i", &p.display().to_string(), "-f", "rawvideo", "-pix_fmt", "rgba", &dir.join("f.raw").display().to_string()])
            .status()
            .ok()?;
        if !raw.success() {
            return None;
        }
        let bytes = std::fs::read(dir.join("f.raw")).ok()?;
        let (mut red, mut blue) = (0usize, 0usize);
        for px in bytes.chunks_exact(4) {
            let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
            if r > g + 40 && r > b + 40 {
                red += 1;
            }
            if b > r + 30 && b > g + 15 {
                blue += 1;
            }
        }
        Some((red, blue))
    }

    #[test]
    fn render3d_renders_parts_and_clip_cuts() {
        let mut fields = Map::new();
        fields.insert("width".into(), json!("320"));
        fields.insert("height".into(), json!("240"));
        fields.insert("azimuth".into(), json!("0"));
        fields.insert("elevation".into(), json!("0"));
        let Some(png) = render(fields.clone()) else { return };
        assert_eq!(png_dims(&png), (320, 240));
        let (red, blue) = color_counts(&png).expect("ffmpeg decode");
        assert!(red > 500, "red cube visible: {red}");
        assert!(blue > 500, "blue cube visible: {blue}");

        let mut clip_fields = fields;
        clip_fields.insert("clipAxis".into(), json!("x"));
        clip_fields.insert("clipPos".into(), json!("0"));
        let clipped = render(clip_fields).expect("chromium present");
        let (cr, cb) = color_counts(&clipped).expect("ffmpeg decode");
        assert!(cr + cb < (red + blue) * 3 / 4, "clip cuts: {} vs {}", cr + cb, red + blue);
    }
}
