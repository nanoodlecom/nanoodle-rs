//! Video family: tvideo / ivideo / vedit / model3d.
//! Mirrors nanoodle-js client.video (submit POST /api/generate-video, poll
//! GET /api/video/status?requestId=…) + the nodes.mjs runners, including the
//! catalog-gated videoDims (no catalog → resolution omitted, aspect →
//! aspect_ratio, duration → duration; vedit never soft-sends dims).
//!
//! Deliberate simplifications vs JS (documented in the workstream report):
//! - fitImage is identity here: JS only rewrites data: URLs over the ~4 MB
//!   inline budget (local-media territory); smaller media passes unchanged.
//! - Poll cadence is 10 ms with a 120 s cap (the Rust CLI has no knob flags);
//!   JS defaults are 5 s / no deadline (25 min for 3D).

use crate::{cat_item, input_text, poll_loop, value_text, Ctx, NanoodleError, Node, NodeError, NodeOutcome, PollStep, Result};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::time::Duration;

const POLL_INTERVAL: Duration = Duration::from_millis(10);
const POLL_CAP: Duration = Duration::from_secs(120);

// ---------------------------------------------------------------- helpers

/// JS mdl(): the node's model, verbatim; empty is an error before any request.
fn mdl(node: &Node) -> Result<String> {
    let m = node.field_str("model").unwrap_or_default();
    let m = m.trim().to_string();
    if m.is_empty() {
        return Err(NanoodleError(format!("pick a model first (node {})", node.id)));
    }
    Ok(m)
}

/// JS promptOf(n, inp) without an error message: wired prompt wins over the
/// field, result trimmed ("" when neither is set).
fn prompt_raw(node: &Node, inputs: &HashMap<String, Value>) -> String {
    let raw = match inputs.get("prompt") {
        Some(v) => value_text(v),
        None => node.field_str("prompt").unwrap_or_default(),
    };
    raw.trim().to_string()
}

/// A raw field value worth sending: present, not null, not "" (JS checks).
fn field_raw(node: &Node, key: &str) -> Option<Value> {
    match node.fields.get(key) {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) if s.is_empty() => None,
        Some(v) => Some(v.clone()),
    }
}

fn model_opts(node: &Node) -> Map<String, Value> {
    match node.fields.get("modelOpts") {
        Some(Value::Object(m)) => m.clone(),
        _ => Map::new(),
    }
}

/// JS videoDims, catalog included: with a catalog, the model's declared
/// param names win (aspect → aspect_ratio | orientation | resolution_ratio,
/// duration → duration | seconds) and blank fields backfill from the catalog
/// default. Resolution forwards only when the model declares it. Without a
/// catalog (or a catalog miss) the param table is empty: resolution is
/// omitted, aspect takes the most-common spelling, and tvideo/ivideo
/// soft-send aspect+duration while vedit/lipsync forward only declared dims.
fn video_dims(ctx: &Ctx, node: &Node) -> Map<String, Value> {
    let mut out = Map::new();
    let empty = Map::new();
    let p = ctx
        .catalog
        .as_ref()
        .and_then(|c| cat_item(c, "video", &node.field_str("model").unwrap_or_default()))
        .and_then(|m| m.pointer("/supported_parameters/parameters"))
        .and_then(|v| v.as_object())
        .unwrap_or(&empty);
    let (a_param, a_wire) = if p.contains_key("aspect_ratio") {
        (p.get("aspect_ratio"), "aspect_ratio")
    } else if p.contains_key("orientation") {
        (p.get("orientation"), "orientation")
    } else if p.contains_key("resolution_ratio") {
        (p.get("resolution_ratio"), "resolution_ratio")
    } else {
        (None, "aspect_ratio")
    };
    let (d_param, d_wire) = if p.contains_key("duration") {
        (p.get("duration"), "duration")
    } else if p.contains_key("seconds") {
        (p.get("seconds"), "seconds")
    } else {
        (None, "duration")
    };
    let soft = node.ty != "vedit" && node.ty != "lipsync";
    // JS backfill: only a non-null, non-empty catalog default fills a blank.
    let default_of = |param: Option<&Value>| -> Option<Value> {
        param
            .and_then(|x| x.get("default"))
            .filter(|v| !v.is_null() && v.as_str() != Some(""))
            .cloned()
    };
    let mut asp = field_raw(node, "aspect");
    let mut dur = field_raw(node, "duration");
    if asp.is_none() {
        asp = default_of(a_param);
    }
    if dur.is_none() {
        dur = default_of(d_param);
    }
    if p.contains_key("resolution") {
        if let Some(r) = field_raw(node, "resolution") {
            out.insert("resolution".into(), r);
        }
    }
    if a_param.is_some() || soft {
        if let Some(a) = asp {
            out.insert(a_wire.into(), a);
        }
    }
    if d_param.is_some() || soft {
        if let Some(d) = dur {
            out.insert(d_wire.into(), d);
        }
    }
    out
}

/// Ports matching ^ref\d*$ in port order (JS collectPorts: trailing-number
/// index, default 1), values non-empty.
fn collect_ref_ports(inputs: &HashMap<String, Value>) -> Vec<String> {
    let mut ports: Vec<(u32, String)> = Vec::new();
    for (k, v) in inputs.iter() {
        if let Some(rest) = k.strip_prefix("ref") {
            if rest.is_empty() || rest.bytes().all(|b| b.is_ascii_digit()) {
                let s = value_text(v);
                if !s.is_empty() {
                    ports.push((rest.parse().unwrap_or(1), s));
                }
            }
        }
    }
    ports.sort_by_key(|(n, _)| *n);
    ports.into_iter().map(|(_, s)| s).collect()
}

/// JS refMaxFor(model, pricing=None): family caps, default 4.
fn ref_max_for(model: &str) -> usize {
    let id = model.to_ascii_lowercase();
    if id.contains("seedance") {
        return 9;
    }
    if id.contains("minimax-h3") {
        return 9;
    }
    if id.contains("gemini-omni-flash/v1.2") {
        return 6;
    }
    if id.contains("luma") || id.contains("ray") {
        return 4;
    }
    4
}

// ---- LoRA (JS loraParams chain, video kind) ----

fn lora_family(model: &str) -> Option<&'static str> {
    let m = model.to_ascii_lowercase();
    if m.contains("spicy") {
        return None;
    }
    if m.contains("p-h3-image") {
        return Some("ph3");
    }
    if m.contains("p-image") {
        return Some("pimage");
    }
    if m.contains("klein") {
        return Some("flux2klein");
    }
    if m.contains("flux-2") {
        return Some("flux2dev");
    }
    if m.contains("z-image") {
        return Some("zimage");
    }
    if m.contains("ltx") {
        return Some("ltx");
    }
    if m.contains("minimax-h3") {
        return Some("h3");
    }
    if m.contains("anima") && m.contains("lora") {
        return Some("anima");
    }
    if m.contains("qwen-image-2.1") && m.contains("lora") {
        return Some("qwen21");
    }
    if m.contains("lora") {
        return Some(if m.contains("krea") { "krea" } else { "flux" });
    }
    None
}

fn lora_cap(family: &str) -> usize {
    match family {
        "flux2dev" => 4,
        "flux2klein" | "zimage" | "ltx" | "krea" | "h3" | "anima" | "qwen21" | "ph3" => 3,
        _ => 1,
    }
}

/// JS normalizeLoraUrl: CivitAI refused, HF blob→resolve + .safetensors check,
/// bare repo ids refused, https required.
fn normalize_lora_url(raw: &str) -> Result<String> {
    let mut u = raw.trim().to_string();
    if u.is_empty() {
        return Ok(u);
    }
    let low = u.to_ascii_lowercase();
    if low.contains("civitai.com") || low.contains("civitai.red") || low.contains("civit.ai") {
        return Err(NanoodleError(
            "CivitAI links can't be fetched directly — download the .safetensors and re-host it (e.g. on HuggingFace), then paste that URL.".into(),
        ));
    }
    if low.contains("huggingface.co/") {
        u = u.replace("/blob/", "/resolve/");
        let ok = match u.find("/resolve/") {
            Some(i) => {
                let rest = &u[i + "/resolve/".len()..];
                match rest.find(".safetensors") {
                    Some(j) => {
                        let after = &rest[j + ".safetensors".len()..];
                        after.is_empty() || after.starts_with('?')
                    }
                    None => false,
                }
            }
            None => false,
        };
        if !ok {
            return Err(NanoodleError(
                "Link the .safetensors file on HuggingFace: open it and use Copy download link (…/resolve/main/your-lora.safetensors).".into(),
            ));
        }
        return Ok(u);
    }
    let bare_repo = {
        let parts: Vec<&str> = u.split('/').collect();
        parts.len() == 2
            && parts
                .iter()
                .all(|p| !p.is_empty() && p.chars().all(|c| c.is_alphanumeric() || c == '.' || c == '_' || c == '-'))
    };
    if bare_repo {
        return Err(NanoodleError(
            "That looks like a HuggingFace repo id — open the .safetensors file and copy its download link (…/resolve/main/your-lora.safetensors).".into(),
        ));
    }
    if !(low.starts_with("https://") || low.starts_with("http://")) {
        return Err(NanoodleError(
            "LoRA must be a direct https URL to a .safetensors file (HuggingFace or any host).".into(),
        ));
    }
    Ok(u)
}

fn num_json(f: f64) -> Value {
    if f.is_finite() && f == f.trunc() && f.abs() < 9.0e15 {
        json!(f as i64)
    } else {
        json!(f)
    }
}

/// JS loraParams(n) for a video node: {} unless the model takes LoRAs and at
/// least one row carries a URL.
fn lora_params(node: &Node, model: &str) -> Result<Map<String, Value>> {
    let mut out = Map::new();
    let family = match lora_family(model) {
        Some(f) if !model.is_empty() => f,
        _ => return Ok(out),
    };
    // nodeLoras: fields.loras rows, else the legacy single-slot fields.
    let mut rows: Vec<(String, f64)> = Vec::new();
    if let Some(Value::Array(arr)) = node.fields.get("loras") {
        for r in arr {
            if let Value::Object(o) = r {
                let url = o
                    .get("url")
                    .map(value_text)
                    .unwrap_or_default();
                if url.trim().is_empty() {
                    continue;
                }
                let scale = match o.get("strength") {
                    None | Some(Value::Null) => 1.0,
                    Some(Value::String(s)) if s.is_empty() => 1.0,
                    Some(Value::Number(n)) => n.as_f64().unwrap_or(f64::NAN),
                    Some(Value::String(s)) => s.trim().parse::<f64>().unwrap_or(f64::NAN),
                    _ => f64::NAN,
                };
                rows.push((url, scale));
            }
        }
    } else {
        let url = node.field_str("loraUrl").unwrap_or_default();
        let strength = node.field_str("loraStrength").unwrap_or_default();
        if !url.trim().is_empty() || !strength.is_empty() {
            if !url.trim().is_empty() {
                let scale = if strength.is_empty() {
                    1.0
                } else {
                    strength.trim().parse::<f64>().unwrap_or(f64::NAN)
                };
                rows.push((url, scale));
            }
        }
    }
    if rows.is_empty() {
        return Ok(out);
    }
    let sc = |v: f64| if v.is_nan() { 1.0 } else { v };
    let items: Vec<(String, f64)> = rows
        .into_iter()
        .take(lora_cap(family))
        .map(|(u, s)| Ok((normalize_lora_url(&u)?, sc(s))))
        .collect::<Result<Vec<_>>>()?;
    match family {
        "pimage" => {
            out.insert("lora_weights".into(), json!(items[0].0));
            out.insert("lora_scale".into(), num_json(items[0].1));
        }
        "flux2dev" | "flux2klein" | "zimage" | "ltx" | "krea" | "h3" | "anima" | "qwen21"
        | "ph3" => {
            for (i, (u, s)) in items.iter().enumerate() {
                out.insert(format!("lora_url_{}", i + 1), json!(u));
                out.insert(format!("lora_scale_{}", i + 1), num_json(*s));
            }
        }
        _ => {
            if items.len() == 1 {
                out.insert("lora_url".into(), json!(items[0].0));
                out.insert("lora_strength".into(), num_json(items[0].1));
            } else {
                let arr: Vec<Value> = items
                    .iter()
                    .map(|(u, s)| json!({ "path": u, "scale": num_json(*s) }))
                    .collect();
                out.insert("loras".into(), Value::Array(arr));
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------- submit + poll

struct VideoJob {
    model: String,
    prompt: String,
    dims: Map<String, Value>,
    lora: Map<String, Value>,
    extra: Map<String, Value>,
    image_data_url: Option<String>,
    last_image: Option<String>,
    sources: Map<String, Value>,
    refs: Vec<String>,
    is3d: bool,
    want_assets: bool,
}

fn url_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

/// First non-empty string at a JSON pointer-ish path (JS `a || b` chains skip
/// empty strings and nulls alike).
fn js_str(v: Option<&Value>) -> Option<String> {
    match v {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    }
}

/// (s.data && s.data.output) || s.output || {} — nulls fall through, as in JS.
fn status_output(s: &Value) -> Value {
    if let Some(d) = s.get("data") {
        if let Some(o) = d.get("output") {
            if !o.is_null() {
                return o.clone();
            }
        }
    }
    s.get("output").cloned().unwrap_or_else(|| json!({}))
}

/// JS model3dStatusUrl(s, model): a completed 3D job must yield a real model
/// URL (kind "3d" or format "glb"); a bare video URL is not a model.
fn model3d_status_url(s: &Value, model: &str) -> String {
    let out = status_output(s);
    let kind = out
        .get("kind")
        .map(value_text)
        .unwrap_or_default()
        .to_ascii_lowercase();
    let format = out
        .get("format")
        .map(value_text)
        .unwrap_or_default()
        .to_ascii_lowercase();
    if kind != "3d" && format != "glb" {
        return String::new();
    }
    if model == "speridlabs/idol" {
        return js_str(out.get("model_mesh").and_then(|m| m.get("url"))).unwrap_or_default();
    }
    let from_list = match out.get("videoUrls") {
        Some(Value::Array(list)) => match list.first() {
            Some(Value::String(u)) => {
                if u.is_empty() {
                    None
                } else {
                    Some(u.clone())
                }
            }
            Some(Value::Object(o)) => js_str(o.get("url")),
            _ => None,
        },
        _ => None,
    };
    js_str(out.get("model_url"))
        .or_else(|| js_str(out.get("model").and_then(|m| m.get("url"))))
        .or_else(|| js_str(out.get("video").and_then(|m| m.get("url"))))
        .or(from_list)
        .unwrap_or_default()
}

/// S1: one labeled companion asset from a completed job's payload.
pub struct StatusAsset {
    pub label: String,
    pub url: String,
    pub kind: String,
}

fn asset_kind(content_type: &str, url: &str) -> String {
    let ct = content_type.to_ascii_lowercase();
    if ct.starts_with("image/") {
        return "image".into();
    }
    if ct.starts_with("video/") || ct == "video/quicktime" {
        return "video".into();
    }
    if ct.starts_with("audio/") {
        return "audio".into();
    }
    if ct.contains("gltf") || ct.contains("model/") {
        return "model3d".into();
    }
    let path = url.split(['?', '#']).next().unwrap_or("").to_ascii_lowercase();
    let ext = path.rsplit('.').next().unwrap_or("");
    match ext {
        "glb" | "gltf" | "fbx" | "obj" | "stl" | "ply" | "usdz" | "3mf" => "model3d".into(),
        "png" | "jpg" | "jpeg" | "webp" | "gif" | "avif" => "image".into(),
        "mp4" | "mov" | "webm" | "mkv" => "video".into(),
        "mp3" | "wav" | "ogg" | "flac" | "m4a" => "audio".into(),
        _ => "file".into(),
    }
}

/// JS parseStatusAssets: normalize output.model3d_assets[] +
/// output.processing_assets[] to [{label, url, kind}].
pub fn parse_status_assets(s: &Value) -> Vec<StatusAsset> {
    let out = status_output(s);
    let mut assets = Vec::new();
    for key in ["model3d_assets", "processing_assets"] {
        if let Some(Value::Array(list)) = out.get(key) {
            for a in list {
                let url = match js_str(a.get("url")) {
                    Some(u) => u,
                    None => continue,
                };
                let label = js_str(a.get("label"))
                    .or_else(|| js_str(a.get("filename")))
                    .unwrap_or_default();
                let ct = a.get("contentType").map(value_text).unwrap_or_default();
                let kind = asset_kind(&ct, &url);
                assets.push(StatusAsset { label, url, kind });
            }
        }
    }
    assets
}

/// The assets JSON in the JS/Py shape — fixed key order (label, url, kind),
/// no spaces — built by hand because serde_json::Map sorts its keys.
fn assets_json(assets: &[StatusAsset]) -> String {
    let parts: Vec<String> = assets
        .iter()
        .map(|a| {
            format!(
                "{{\"label\":{},\"url\":{},\"kind\":{}}}",
                serde_json::to_string(&a.label).unwrap_or_default(),
                serde_json::to_string(&a.url).unwrap_or_default(),
                serde_json::to_string(&a.kind).unwrap_or_default()
            )
        })
        .collect();
    format!("[{}]", parts.join(","))
}

/// JS assetOutputs: role ports for companion assets. The `assets` port
/// always carries the full list as JSON; canonical roles surface as URLs.
/// First match wins per role; the primary never reappears as model2..N.
pub fn asset_outputs(assets: &[StatusAsset], primary_url: &str, is3d: bool) -> Map<String, Value> {
    let mut out = Map::new();
    if assets.is_empty() {
        return out;
    }
    out.insert("assets".into(), json!(assets_json(assets)));
    let mut taken: Vec<bool> = vec![false; assets.len()];
    let mut claim = |role: &str, out: &mut Map<String, Value>, test: &dyn Fn(&StatusAsset) -> bool| {
        for (i, a) in assets.iter().enumerate() {
            if !taken[i] && test(a) {
                taken[i] = true;
                out.insert(role.into(), json!(a.url));
                return;
            }
        }
    };
    if is3d {
        claim("depth", &mut out, &|a: &StatusAsset| a.kind == "image" && a.label.to_ascii_lowercase().contains("depth"));
        claim("normals", &mut out, &|a: &StatusAsset| a.kind == "image" && a.label.to_ascii_lowercase().contains("normal"));
        claim("mask", &mut out, &|a: &StatusAsset| a.kind == "image" && a.label.to_ascii_lowercase().contains("mask"));
        claim("points", &mut out, &|a: &StatusAsset| {
            a.label.to_ascii_lowercase().contains("point")
                || a.url.to_ascii_lowercase().split(['?', '#']).next().unwrap_or("").ends_with(".ply")
        });
        let mut n = 2;
        for (i, a) in assets.iter().enumerate() {
            if a.kind != "model3d" || taken[i] || a.url == primary_url {
                continue;
            }
            taken[i] = true;
            out.insert(format!("model{}", n), json!(a.url));
            n += 1;
        }
    } else {
        claim("matte", &mut out, &|a: &StatusAsset| a.kind == "video" && a.label.to_ascii_lowercase().contains("matte"));
        claim("cutout", &mut out, &|a: &StatusAsset| {
            a.kind == "video"
                && (a.label.to_ascii_lowercase().contains("cutout") || a.label.to_ascii_lowercase().contains("alpha"))
        });
    }
    out
}

/// JS completed-video URL: (out.video && out.video.url) || out.url ||
/// (Array.isArray(out.video) ? out.video[0].url : null).
fn video_status_url(out: &Value) -> Option<String> {
    if let Some(v) = out.get("video") {
        if let Value::Object(o) = v {
            if let Some(u) = js_str(o.get("url")) {
                return Some(u);
            }
        }
    }
    if let Some(u) = js_str(out.get("url")) {
        return Some(u);
    }
    if let Some(Value::Array(list)) = out.get("video") {
        if let Some(Value::Object(o)) = list.first() {
            return js_str(o.get("url"));
        }
    }
    None
}

/// HTTP status hidden in a Ctx error, if any (http_error formats generic
/// failures as "{status}: {body}" and auth failures as "(HTTP {status})").
fn error_status(msg: &str) -> Option<u16> {
    let head: String = msg.chars().take(3).collect();
    if msg.len() > 3
        && head.len() == 3
        && head.bytes().all(|b| b.is_ascii_digit())
        && msg[3..].starts_with(": ")
    {
        return head.parse().ok();
    }
    if let Some(i) = msg.find("(HTTP ") {
        let rest = &msg[i + 6..];
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        return digits.parse().ok();
    }
    None
}

/// Submit + poll, mirroring JS client.video. Cost: JS fires onCost once, on
/// the submit response — status responses never add cost.
fn gen_video(ctx: &Ctx, job: VideoJob) -> std::result::Result<(String, Option<f64>, Map<String, Value>), NodeError> {
    let mut body = Map::new();
    body.insert("model".into(), json!(job.model));
    // 3D omits an empty prompt; video always sends it, blank included.
    if !(job.is3d && job.prompt.trim().is_empty()) {
        body.insert("prompt".into(), json!(job.prompt));
    }
    for (k, v) in &job.dims {
        body.insert(k.clone(), v.clone());
    }
    if let Some(u) = &job.image_data_url {
        body.insert("imageDataUrl".into(), json!(u));
    }
    if let Some(u) = &job.last_image {
        body.insert("last_image".into(), json!(u));
    }
    for k in ["videoUrl", "videoDataUrl", "audioUrl", "audioDataUrl"] {
        if let Some(v) = job.sources.get(k) {
            body.insert(k.into(), v.clone());
        }
    }
    for (k, v) in &job.lora {
        body.insert(k.clone(), v.clone());
    }
    for (k, v) in &job.extra {
        body.insert(k.clone(), v.clone());
    }
    // node-owned dims win over stale modelOpts copies
    for (k, v) in &job.dims {
        body.insert(k.clone(), v.clone());
    }
    if !job.refs.is_empty() {
        // catalog-free modelRefSpec: most-common spelling, family cap
        let cap = ref_max_for(&job.model);
        let refs: Vec<Value> = job
            .refs
            .iter()
            .take(cap)
            .map(|u| json!(u))
            .collect();
        body.insert("reference_images".into(), Value::Array(refs));
    }

    let (j, cost) = ctx.post_json("/api/generate-video", &Value::Object(body))?;
    let run_id = js_str(j.get("runId"))
        .or_else(|| js_str(j.get("id")))
        .or_else(|| {
            // JS: j.runId || j.id, then String() in the query — numbers ride too
            for k in ["runId", "id"] {
                if let Some(Value::Number(n)) = j.get(k) {
                    return Some(n.to_string());
                }
            }
            None
        })
        .ok_or_else(|| NodeError { message: "no runId returned".into(), cost })?;

    let label = if job.is3d { "3D" } else { "video" };
    let path = format!("/api/video/status?requestId={}", url_encode(&run_id));
    let timeout = format!(
        "{} timed out ({}s) — the job may still be running on NanoGPT's side",
        label,
        POLL_CAP.as_secs()
    );
    let (url, extras) = poll_loop(POLL_INTERVAL, POLL_CAP, timeout, || {
        let s = match ctx.get_json(&path) {
            Ok((s, _)) => s,
            Err(e) => {
                // JS: a 4xx poll response (except 408/429) is terminal; any
                // other failure is transient — stay in Pending until the cap.
                if let Some(st) = error_status(&e.0) {
                    if (400..500).contains(&st) && st != 408 && st != 429 {
                        return Err(e);
                    }
                }
                return Ok(PollStep::Pending);
            }
        };
        let data = s.get("data");
        let st = data
            .and_then(|d| d.get("status"))
            .map(value_text)
            .filter(|x| !x.is_empty())
            .or_else(|| s.get("status").map(value_text))
            .unwrap_or_default()
            .to_ascii_uppercase();
        if st == "COMPLETED" || st == "SUCCEEDED" {
            // S1: jobs that ask also collect the payload's labeled companion
            // assets as role outputs; everyone else gets an empty map.
            let extras = |url: &str| {
                if job.want_assets {
                    asset_outputs(&parse_status_assets(&s), url, job.is3d)
                } else {
                    Map::new()
                }
            };
            if job.is3d {
                let url = model3d_status_url(&s, &job.model);
                if url.is_empty() {
                    return Ok(PollStep::Failed("completed but no model url".into()));
                }
                let ex = extras(&url);
                return Ok(PollStep::Done((url, ex)));
            }
            let out = status_output(&s);
            return Ok(match video_status_url(&out) {
                Some(url) => {
                    let ex = extras(&url);
                    PollStep::Done((url, ex))
                }
                None => PollStep::Failed("completed but no video url".into()),
            });
        }
        if ["FAILED", "ERROR", "CANCELED", "CANCELLED", "CONTENT_POLICY_VIOLATION"]
            .contains(&st.as_str())
        {
            let why = data
                .and_then(|d| d.get("error"))
                .map(value_text)
                .filter(|x| !x.is_empty())
                .unwrap_or_else(|| st.clone());
            return Ok(PollStep::Failed(format!("{} failed: {}", label, why)));
        }
        Ok(PollStep::Pending)
    })
    .map_err(|e| NodeError { message: e.0, cost })?;
    Ok((url, cost, extras))
}

fn done(key: &str, url: String, extras: Map<String, Value>, cost: Option<f64>) -> NodeOutcome {
    let mut out = extras;
    out.insert(key.into(), json!(url));
    NodeOutcome::Done(out, cost)
}

// ---------------------------------------------------------------- runners

pub fn run_tvideo(node: &Node, inputs: &HashMap<String, Value>, ctx: &Ctx) -> std::result::Result<NodeOutcome, NodeError> {
    let prompt = prompt_raw(node, inputs);
    if prompt.is_empty() {
        return Err(NanoodleError("no prompt".into()).into());
    }
    let model = mdl(node)?;
    let (url, cost, extras) = gen_video(
        ctx,
        VideoJob {
            dims: video_dims(ctx, node),
            lora: lora_params(node, &model)?,
            extra: model_opts(node),
            image_data_url: None,
            last_image: None,
            sources: Map::new(),
            refs: collect_ref_ports(inputs),
            model,
            prompt,
            is3d: false,
            want_assets: true,
        },
    )?;
    Ok(done("video", url, extras, cost))
}

pub fn run_ivideo(node: &Node, inputs: &HashMap<String, Value>, ctx: &Ctx) -> std::result::Result<NodeOutcome, NodeError> {
    let image = input_text(inputs, "image")
        .ok_or_else(|| NanoodleError("no image input".into()))?;
    let model = mdl(node)?;
    let (url, cost, extras) = gen_video(
        ctx,
        VideoJob {
            dims: video_dims(ctx, node),
            lora: lora_params(node, &model)?,
            extra: model_opts(node),
            image_data_url: Some(image),
            last_image: input_text(inputs, "endframe"),
            sources: Map::new(),
            refs: collect_ref_ports(inputs),
            model,
            prompt: prompt_raw(node, inputs),
            is3d: false,
            want_assets: true,
        },
    )?;
    Ok(done("video", url, extras, cost))
}

pub fn run_vedit(node: &Node, inputs: &HashMap<String, Value>, ctx: &Ctx) -> std::result::Result<NodeOutcome, NodeError> {
    let video = input_text(inputs, "video")
        .ok_or_else(|| NanoodleError("no video input".into()))?;
    let model = mdl(node)?;
    // JS videoSourceOpts: https → videoUrl, anything else → videoDataUrl.
    let low = video.to_ascii_lowercase();
    let mut sources = Map::new();
    if low.starts_with("https:") || low.starts_with("http:") {
        sources.insert("videoUrl".into(), json!(video));
    } else {
        sources.insert("videoDataUrl".into(), json!(video));
    }
    let (url, cost, extras) = gen_video(
        ctx,
        VideoJob {
            dims: video_dims(ctx, node),
            lora: lora_params(node, &model)?,
            extra: model_opts(node),
            // character still (Wan Animate family): catalog miss → send the wire
            image_data_url: input_text(inputs, "image"),
            last_image: None,
            sources,
            refs: collect_ref_ports(inputs),
            model,
            prompt: prompt_raw(node, inputs),
            is3d: false,
            want_assets: true,
        },
    )?;
    Ok(done("video", url, extras, cost))
}

/// JS model3dInputMods without a catalog: the two known defaults, everything
/// else (incl. catalog miss) accepts image + text.
/// JS model3dInputMods: (image, text, mesh) from the catalog entry's
/// modalities (or architecture.input_modalities); "3d" means the model
/// processes an existing mesh. Catalog miss → the two hardcoded defaults,
/// then permissive image+text.
fn model3d_input_mods(catalog: Option<&Value>, model: &str) -> (bool, bool, bool) {
    if let Some(it) = catalog.and_then(|c| cat_item(c, "model3d", model)) {
        let raw = it
            .get("modalities")
            .and_then(|v| v.as_array())
            .or_else(|| {
                it.get("architecture")
                    .and_then(|a| a.get("input_modalities"))
                    .and_then(|v| v.as_array())
            });
        if let Some(mods) = raw {
            let has = |k: &str| mods.iter().any(|x| x.as_str() == Some(k));
            let (im, tx, me) = (has("image"), has("text"), has("3d"));
            if im || tx || me {
                return (im, tx, me);
            }
        }
    }
    match model {
        "tripo3d/v2.5" => (true, false, false),
        _ => (true, true, false),
    }
}

pub fn run_model3d(node: &Node, inputs: &HashMap<String, Value>, ctx: &Ctx) -> std::result::Result<NodeOutcome, NodeError> {
    let model_field = node.field_str("model").unwrap_or_default();
    let model = if model_field.trim().is_empty() {
        "tripo3d/v2.5".to_string()
    } else {
        model_field.trim().to_string()
    };
    let (takes_image, takes_text, takes_mesh) =
        model3d_input_mods(ctx.catalog.as_ref().map(|v| v), &model);
    let prompt = prompt_raw(node, inputs);
    let image = if takes_image {
        input_text(inputs, "image").unwrap_or_default()
    } else {
        String::new()
    };
    let mesh = if takes_mesh {
        input_text(inputs, "model").unwrap_or_default()
    } else {
        String::new()
    };
    if takes_mesh && mesh.is_empty() {
        return Err(NanoodleError("Connect a model first.".into()).into());
    }
    if !takes_mesh && takes_image && !takes_text && image.is_empty() {
        return Err(NanoodleError("Connect an image first.".into()).into());
    }
    if !takes_mesh && takes_text && !takes_image && prompt.is_empty() {
        return Err(NanoodleError("Add a prompt first.".into()).into());
    }
    if !takes_mesh && takes_image && takes_text && image.is_empty() && prompt.is_empty() {
        return Err(NanoodleError("Add a prompt or connect an image first.".into()).into());
    }
    let mut extra = model_opts(node);
    if !mesh.is_empty() {
        extra.insert("model_url".into(), json!(mesh)); // the wire wins over a static modelOpts copy
    }
    let (url, cost, extras) = gen_video(
        ctx,
        VideoJob {
            dims: Map::new(),
            lora: Map::new(),
            extra,
            image_data_url: if image.is_empty() { None } else { Some(image) },
            last_image: None,
            sources: Map::new(),
            refs: Vec::new(),
            model,
            prompt: if takes_text { prompt } else { String::new() },
            is3d: true,
            want_assets: true,
        },
    )?;
    Ok(done("model", url, extras, cost))
}

#[cfg(test)]
mod tests {
    use super::*;

    // S1 fixtures: trimmed copies of REAL 2026-11-21 NanoGPT payloads
    // (Edge Styles Lab probes: moge3, alphagen, convert) — same fixtures as
    // the JS assets-surfacing and Py test_assets_surfacing suites.
    fn moge() -> Value {
        json!({"data": {"status": "COMPLETED", "output": {
            "kind": "3d", "format": "glb",
            "model_url": "https://v3b.fal.media/files/b/0ab36d09/GLB_mesh.glb",
            "model3d_assets": [
                {"label": "Textured GLB", "filename": "mesh.glb", "url": "https://v3b.fal.media/files/b/0ab36d09/GLB_mesh.glb", "contentType": "model/gltf-binary"},
                {"label": "Colored point cloud PLY", "filename": "pointcloud.ply", "url": "https://v3b.fal.media/files/b/0ab36d09/PC_pointcloud.ply", "contentType": "application/octet-stream"},
                {"label": "Colorized depth map PNG", "filename": "depth.png", "url": "https://v3b.fal.media/files/b/0ab36d09/DEPTH.png", "contentType": "image/png"},
                {"label": "Surface normals PNG", "filename": "normals.png", "url": "https://v3b.fal.media/files/b/0ab36d09/NORMALS.png", "contentType": "image/png"},
                {"label": "Validity mask PNG", "filename": "mask.png", "url": "https://v3b.fal.media/files/b/0ab36d09/MASK.png", "contentType": "image/png"}
            ]}}})
    }

    fn alphagen() -> Value {
        json!({"data": {"status": "COMPLETED", "output": {
            "video": {"url": "https://nano-gpt.com/v1/media/content?rid=abc"},
            "processing_assets": [
                {"label": "Alpha matte MP4", "filename": "matte.mp4", "url": "https://v3b.fal.media/files/b/0ab36d6e/MATTE_matte.mp4", "contentType": "video/mp4"},
                {"label": "Straight-alpha ProRes 4444 MOV", "filename": "cutout.mov", "url": "https://v3b.fal.media/files/b/0ab36d6e/CUT_cutout.mov", "contentType": "video/quicktime"}
            ]}}})
    }

    #[test]
    fn parse_moge_assets_normalizes_kinds() {
        let assets = parse_status_assets(&moge());
        assert_eq!(assets.len(), 5);
        let kinds: Vec<&str> = assets.iter().map(|a| a.kind.as_str()).collect();
        assert_eq!(kinds, ["model3d", "model3d", "image", "image", "image"]);
        assert_eq!(assets[1].label, "Colored point cloud PLY");
        assert!(parse_status_assets(&json!({"data": {"output": {}}})).is_empty());
    }

    #[test]
    fn moge_roles_land_and_primary_never_duplicates() {
        let primary = "https://v3b.fal.media/files/b/0ab36d09/GLB_mesh.glb";
        let out = asset_outputs(&parse_status_assets(&moge()), primary, true);
        assert_eq!(out["depth"], json!("https://v3b.fal.media/files/b/0ab36d09/DEPTH.png"));
        assert_eq!(out["normals"], json!("https://v3b.fal.media/files/b/0ab36d09/NORMALS.png"));
        assert_eq!(out["mask"], json!("https://v3b.fal.media/files/b/0ab36d09/MASK.png"));
        assert_eq!(out["points"], json!("https://v3b.fal.media/files/b/0ab36d09/PC_pointcloud.ply"));
        assert!(!out.contains_key("model2"));
        let parsed: Value = serde_json::from_str(out["assets"].as_str().unwrap()).unwrap();
        assert_eq!(parsed.as_array().unwrap().len(), 5);
        // JS/Py byte shape: label,url,kind order, no spaces
        assert!(out["assets"].as_str().unwrap().starts_with("[{\"label\":"));
    }

    #[test]
    fn alphagen_matte_and_cutout() {
        let out = asset_outputs(
            &parse_status_assets(&alphagen()),
            "https://nano-gpt.com/v1/media/content?rid=abc",
            false,
        );
        assert_eq!(out["matte"], json!("https://v3b.fal.media/files/b/0ab36d6e/MATTE_matte.mp4"));
        assert_eq!(out["cutout"], json!("https://v3b.fal.media/files/b/0ab36d6e/CUT_cutout.mov"));
        assert!(asset_outputs(&[], "x", false).is_empty());
    }
}
