//! Rust executor for nanoodle graphs.
//!
//! Slice 1: loader (materialize) + local nodes (text / choice / join / comment /
//! upload family). Semantics mirror nanoodle-js `graph.mjs` / `nodes.mjs`
//! exactly; the shared conformance suite in `../executor-core/conformance/`
//! is the contract. See `../executor-core/SPEC.md` and `RUST-BRIEF.md`.

use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;
use std::time::{Duration, Instant};

pub mod audio;
pub mod endpoint;
pub mod local_media;
pub mod render3d;
pub mod video;

#[cfg(test)]
mod provider_tests;

// ---------------------------------------------------------------- errors

#[derive(Debug)]
pub struct NanoodleError(pub String);

/// A failed run, mirroring JS RunError: the message plus the partial
/// result (records, outputs, cost, errors) produced before the throw.
/// Pre-run validation failures carry no result.
pub struct RunFailure {
    pub message: String,
    pub result: Option<RunResult>,
}

impl From<NanoodleError> for RunFailure {
    fn from(e: NanoodleError) -> Self {
        RunFailure { message: e.0, result: None }
    }
}

/// A node failure that may already have billed: JS fires onCost at submit
/// time, so a job that fails while polling still counts its submit cost
/// (and a decide "pick" whose second send fails keeps the first's cost).
/// Plain NanoodleErrors convert with no cost attached.
pub struct NodeError {
    pub message: String,
    pub cost: Option<f64>,
}

impl From<NanoodleError> for NodeError {
    fn from(e: NanoodleError) -> Self {
        NodeError { message: e.0, cost: None }
    }
}

impl fmt::Display for NanoodleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for NanoodleError {}

pub type Result<T> = std::result::Result<T, NanoodleError>;
type NodeResult<T> = std::result::Result<T, NodeError>;

// ---------------------------------------------------------------- registry

/// (title, static inputs, output ports) per node type — mirrors JS NODE_TYPES.
fn node_type(ty: &str) -> Option<(&'static str, &'static [&'static str], &'static [&'static str])> {
    Some(match ty {
        "text" => ("Text", &[][..], &["text"][..]),
        "upload" => ("Image input", &[][..], &["image"][..]),
        "aupload" => ("Audio input", &[][..], &["audio"][..]),
        "vupload" => ("Video input", &[][..], &["video"][..]),
        "mupload" => ("3D input", &[][..], &["model"][..]),
        "choice" => ("Choice", &[][..], &["text"][..]),
        "join" => ("Join", &["a", "b"][..], &["text"][..]),
        "llm" => ("LLM", &[][..], &["text"][..]),
        "image" => ("Image", &[][..], &["image"][..]),
        "edit" => ("Edit", &[][..], &["image"][..]),
        "inpaint" => ("Inpaint", &["image", "mask"][..], &["image"][..]),
        "slice-3d" => ("Slice 3D", &["model", "paletteImage"][..], &["sheet", "layout", "image"][..]),
        "sprite-stack-render" => ("Sprite stack render", &["sheet"][..], &["image", "video"][..]),
        "resize" => ("Resize / crop", &["image"][..], &["image"][..]),
        "vision" => ("Vision", &["image"][..], &["text"][..]),
        "decide" => ("Decide", &["text"][..], &["text", "image"][..]),
        "tvideo" => ("Text→Video", &[][..], &["video"][..]),
        "ivideo" => ("Image→Video", &["image"][..], &["video"][..]),
        "model3d" => ("3D model", &["image", "model"][..], &["model"][..]),
        "vedit" => ("Video edit", &["video"][..], &["video"][..]),
        // Frame extraction emits numbered ports. The registry supplies the primary sink port.
        "vframes" => ("Video → frames", &["video"][..], &["frame1"][..]),
        "render3d" => ("Render 3D", &["model"][..], &["image", "video"][..]),
        "combine" => ("Combine videos", &[][..], &["video"][..]),
        "soundtrack" => ("Soundtrack", &["video", "audio"][..], &["video"][..]),
        "lipsync" => ("Avatar / lipsync", &["image", "audio"][..], &["video"][..]),
        "music" => ("Music", &[][..], &["audio"][..]),
        "remix" => ("Remix audio", &["audio"][..], &["audio"][..]),
        "tts" => ("Speech", &[][..], &["audio"][..]),
        "trim" => ("Trim audio", &["audio"][..], &["audio"][..]),
        "extractaudio" => ("Extract audio", &["video"][..], &["audio"][..]),
        "transcribe" => ("Transcribe", &["audio"][..], &["text"][..]),
        "cleanvoice" => ("Clean voice", &["audio", "video"][..], &["audio"][..]),
        "endpoint" => ("Custom endpoint", &["text", "image", "audio", "video"][..], &["text", "image", "audio", "video"][..]),
        "comment" => ("Comment", &[][..], &[][..]),
        _ => return None,
    })
}

pub const NODE_TYPE_NAMES: &[&str] = &[
    "text", "upload", "aupload", "vupload", "mupload", "choice", "join", "llm",
    "image", "edit", "inpaint", "resize", "slice-3d", "sprite-stack-render", "vision", "decide", "tvideo", "ivideo",
    "model3d", "vedit", "vframes", "render3d", "combine", "soundtrack", "lipsync", "music",
    "remix", "tts", "trim", "extractaudio", "transcribe", "cleanvoice",
    "endpoint", "comment",
];

fn dynamic_input_name(port: &str) -> bool {
    port == "audio" || port == "endframe"
}

/// ^img\d*$ | ^image\d*$ | ^vid\d*$ | ^clip\d*$ | ^ref\d*$  (JS DYNAMIC_INPUT_RES).
fn dynamic_input_re(port: &str) -> bool {
    ["img", "image", "vid", "clip", "ref"].iter().any(|prefix| numbered_input_port(port, prefix))
}

fn numbered_input_port(port: &str, prefix: &str) -> bool {
    port.strip_prefix(prefix).is_some_and(|suffix| suffix.bytes().all(|b| b.is_ascii_digit()))
}

pub fn is_input_port(node: &Node, port: &str) -> bool {
    if let Some((_, inputs, _)) = node_type(&node.ty) {
        if inputs.contains(&port) {
            return true;
        }
    }
    dynamic_input_name(port) || dynamic_input_re(port)
}

// ---------------------------------------------------------------- graph

#[derive(Debug, Clone)]
pub struct Node {
    pub id: String,
    pub ty: String,
    pub name: Option<String>,
    pub fields: Map<String, Value>,
    pub unknown: bool,
}

impl Node {
    pub fn display_name(&self) -> String {
        if let Some(n) = &self.name {
            let t = n.trim();
            if !t.is_empty() {
                return t.to_string();
            }
        }
        node_type(&self.ty).map(|(title, _, _)| title.to_string()).unwrap_or_else(|| self.ty.clone())
    }
    pub fn field_str(&self, key: &str) -> Option<String> {
        match self.fields.get(key) {
            Some(Value::String(s)) => Some(s.clone()),
            Some(Value::Number(n)) => Some(n.to_string()),
            Some(Value::Bool(b)) => Some(b.to_string()),
            _ => None,
        }
    }
    pub fn optional(&self) -> bool {
        match self.fields.get("optional") {
            Some(Value::Bool(true)) => true,
            Some(Value::String(s)) => s == "true",
            _ => false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Link {
    pub from_node: String,
    pub from_port: String,
    pub to_node: String,
    pub to_port: String,
}

impl fmt::Display for Link {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "wire {}.{} → {}.{}", self.from_node, self.from_port, self.to_node, self.to_port)
    }
}

fn undeclared_source_port(source: &Node, link: &Link) -> Option<String> {
    let (_, _, ports) = node_type(&source.ty)?;
    let port = link.from_port.as_str();
    let possible = ports.contains(&port)
        || match source.ty.as_str() {
            "vframes" => local_media::frame_output_index(port).is_some(),
            // Branch mode is checked separately by BranchPlan.
            "decide" => matches!(port, "yes" | "no" | "decision"),
            "image" | "endpoint" => port == "images",
            "edit" | "inpaint" => matches!(port, "images" | "layers"),
            _ => false,
        };
    (!possible).then(|| format!(
        "{link} uses undeclared source port {port:?} on {}", source.display_name()
    ))
}

/// Additional handles and wired fields beyond a node's static ports and workflow inputs.
/// An authored setting or model can change at run time, so recognize all supported modes.
fn extra_wire_targets(ty: &str, port: &str) -> bool {
    let dynamic = match ty {
        "llm" | "decide" => numbered_input_port(port, "img"),
        "edit" => numbered_input_port(port, "image"),
        "tvideo" | "ivideo" | "vedit" => numbered_input_port(port, "ref"),
        "combine" => numbered_input_port(port, "clip") || numbered_input_port(port, "vid"),
        _ => false,
    };
    if dynamic {
        return true;
    }
    let fields: &[&str] = match ty {
        "choice" => &["selected", "options"],
        "join" => &["sep"],
        "llm" => &["audio", "video", "model", "temperature", "maxTokens", "format", "reasoningEffort", "showThinking"],
        "vision" => &["model", "q"],
        "decide" => &["model", "mode", "question", "options", "levels", "gate"],
        "image" | "edit" | "inpaint" => {
            if ty == "image" && matches!(port, "variations" | "sel" | "customCivitaiAir") {
                return true;
            }
            &["prompt", "model", "size", "seed", "modelOpts", "loras", "loraUrl", "loraStrength"]
        }
        "tvideo" | "ivideo" | "vedit" => {
            if (ty == "ivideo" && port == "endframe") || (ty == "vedit" && port == "image") {
                return true;
            }
            &["prompt", "model", "resolution", "aspect", "duration", "modelOpts", "loras", "loraUrl", "loraStrength"]
        }
        "model3d" => &["modelOpts"],
        "lipsync" => &["prompt", "model", "resolution", "aspect", "duration", "modelOpts"],
        "music" => &["model", "lyrics", "instrumental", "duration", "negative_prompt", "seed", "response_format", "extraJson"],
        "remix" => &["model", "lyrics", "duration", "response_format", "extraJson"],
        "tts" => &["model", "voice", "speed", "instructions", "response_format", "extraJson"],
        "transcribe" => &["model", "language"],
        "cleanvoice" => &["model", "url"],
        "endpoint" => &["url", "mode", "model", "auth", "system", "size"],
        "slice-3d" => &["preset", "clip", "frames", "slices", "resolution", "palette", "outline", "upAxis"],
        "sprite-stack-render" => &["yaw", "pitch", "spacing", "scale", "turntable", "duration", "fps", "yawStart", "yawEnd"],
        "resize" => &["mode", "width", "height"],
        "vframes" => &["dir", "frames", "gap", "stride", "startSec", "endSec"],
        "render3d" => &[
            "mode", "azimuth", "elevation", "distance", "width", "height", "background", "clipAxis", "clipPos",
            "clipSweep", "explode", "isolate", "animClip", "animTime", "turntable", "duration", "fps",
        ],
        "combine" => &["dedup"],
        "soundtrack" => &["loop"],
        "trim" | "extractaudio" => &["start", "length"],
        _ => &[],
    };
    fields.contains(&port)
}

fn undeclared_target_port(target: &Node, link: &Link) -> Option<String> {
    let (_, inputs, _) = node_type(&target.ty)?;
    let port = link.to_port.as_str();
    let possible = inputs.contains(&port)
        || input_specs(&target.ty).iter().any(|(field, ..)| *field == port)
        || extra_wire_targets(&target.ty, port);
    (!possible).then(|| format!(
        "{link} uses undeclared target port {port:?} on {}", target.display_name()
    ))
}

fn wire_issue(by_id: &HashMap<&str, &Node>, link: &Link) -> Option<String> {
    let source_id = link.from_node.as_str();
    let target_id = link.to_node.as_str();
    let Some(source) = by_id.get(source_id).filter(|_| !source_id.is_empty()) else {
        return Some(format!("{link} references missing source node {source_id:?}"));
    };
    let Some(target) = by_id.get(target_id).filter(|_| !target_id.is_empty()) else {
        return Some(format!("{link} references missing target node {target_id:?}"));
    };
    undeclared_source_port(source, link).or_else(|| undeclared_target_port(target, link))
}

fn has_repeated_input(links: &[Link]) -> bool {
    let mut seen = HashSet::new();
    links.iter().any(|link| !seen.insert((link.to_node.as_str(), link.to_port.as_str())))
}

fn competing_input_issues<'a>(links: impl IntoIterator<Item = &'a Link>) -> Vec<String> {
    let mut inputs: BTreeMap<_, BTreeSet<_>> = BTreeMap::new();
    for link in links {
        inputs.entry((link.to_node.as_str(), link.to_port.as_str()))
            .or_default().insert((link.from_node.as_str(), link.from_port.as_str()));
    }
    inputs.into_iter().filter(|(_, sources)| sources.len() > 1)
        .map(|((node, port), sources)| {
            let mut sources: Vec<_> = sources.into_iter().map(|(node, port)| format!("{node}.{port}")).collect();
            sources.sort_unstable();
            format!("input {node}.{port} has competing wires from {}. Leave one active wire for this input.", sources.join(", "))
        }).collect()
}

#[derive(Debug)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub links: Vec<Link>,
    pub warnings: Vec<String>,
}

const MEDIA_FIELD_KEYS: [&str; 4] = ["image", "mask", "audio", "video"];

fn is_media_url(s: &str) -> bool {
    let t = s.trim().to_ascii_lowercase();
    t.starts_with("data:") || t.starts_with("http:") || t.starts_with("https:")
}

fn scrub_media_placeholders(n: &mut Node, warnings: &mut Vec<String>) {
    let mut keys: Vec<&str> = MEDIA_FIELD_KEYS.to_vec();
    if n.ty == "mupload" {
        keys.push("model");
    }
    for k in keys {
        let v = match n.fields.get(k) {
            None | Some(Value::Null) => continue,
            Some(Value::String(s)) if s.is_empty() || is_media_url(s) => continue,
            Some(Value::String(s)) if s.strip_prefix("examples/").is_some_and(|path| {
                !path.is_empty() && !path.contains("..") &&
                    path.bytes().all(|b| b.is_ascii_alphanumeric() || b"_./-".contains(&b))
            }) => {
                let url = format!("https://nanoodle.com/{s}");
                n.fields.insert(k.to_owned(), Value::String(url));
                continue;
            }
            Some(v) => v.clone(),
        };
        let shown = match &v {
            Value::String(s) if s.chars().count() > 60 => {
                format!("\"{}…\"", s.chars().take(57).collect::<String>())
            }
            Value::String(s) => format!("\"{}\"", s),
            Value::Array(_) => "an array".to_string(),
            Value::Object(_) => "an object".to_string(),
            other => format!("a {}", other_type(other)),
        };
        n.fields.insert(k.to_string(), Value::String(String::new()));
        warnings.push(format!(
            "node {} ({}): fields.{} held {} — not a data: or http(s) URL, so it was treated as empty.",
            n.id, n.display_name(), k, shown
        ));
    }
}

fn other_type(v: &Value) -> &'static str {
    match v {
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        _ => "value",
    }
}

/// Load raw parsed graph JSON into an executable graph. Mirrors JS materialize.
pub fn materialize(data: &Value) -> Result<Graph> {
    let obj = data.as_object().ok_or_else(|| {
        NanoodleError("not a nanoodle graph: expected an object with a nodes array".into())
    })?;
    let raw_nodes = obj.get("nodes").and_then(|v| v.as_array()).ok_or_else(|| {
        NanoodleError("not a nanoodle graph: expected an object with a nodes array".into())
    })?;
    let mut warnings = Vec::new();
    let mut nodes: Vec<Node> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for (index, raw) in raw_nodes.iter().enumerate() {
        let position = index + 1;
        let raw = raw.as_object().ok_or_else(|| {
            NanoodleError(format!("node at position {position} must be an object"))
        })?;
        let id = match raw.get("id") {
            Some(Value::String(id)) if !id.trim().is_empty() => id.clone(),
            Some(Value::Number(id)) if id.as_f64().is_some_and(f64::is_finite) => id.to_string(),
            _ => return Err(NanoodleError(format!(
                "node at position {position} has a missing or invalid id (expected a non-empty string or finite number)"
            ))),
        };
        let raw_type = raw.get("type").and_then(Value::as_str)
            .filter(|ty| !ty.trim().is_empty()).ok_or_else(|| NanoodleError(format!(
                "node at position {position} has a missing or invalid type (expected a non-empty string)"
            )))?;
        if !seen.insert(id.clone()) {
            return Err(NanoodleError(format!(
                "duplicate node id \"{}\" — every node must have a unique id",
                id
            )));
        }
        let branching = raw_type == "decide-branch";
        let ty = match raw_type {
            "audio" => "tts",
            "decide-branch" => "decide",
            _ => raw_type,
        }.to_owned();
        let mut n = Node {
            id,
            name: raw.get("name").and_then(|v| v.as_str()).map(|s| s.to_string()),
            fields: raw.get("fields").and_then(|v| v.as_object()).cloned().unwrap_or_default(),
            unknown: false,
            ty,
        };
        if branching {
            n.fields.insert("branch".into(), Value::Bool(true));
        }
        if node_type(&n.ty).is_none() {
            n.unknown = true;
            warnings.push(format!(
                "unknown node type \"{}\" (node {}) — kept, but running this workflow will fail; you may need a newer nanoodle library",
                raw_type,
                n.id
            ));
        } else {
            scrub_media_placeholders(&mut n, &mut warnings);
        }
        nodes.push(n);
    }
    let by_id: HashMap<&str, &Node> = nodes.iter().map(|n| (n.id.as_str(), n)).collect();
    let mut links = Vec::new();
    if let Some(raw_links) = obj.get("links").and_then(|v| v.as_array()) {
        for l in raw_links {
            let f = l.get("from").unwrap_or(&Value::Null);
            let t = l.get("to").unwrap_or(&Value::Null);
            let (fn_, fp) = (node_id_str(f.get("node")), port_str(f.get("port")));
            let (tn, mut tp) = (node_id_str(t.get("node")), port_str(t.get("port")));
            if tp == "text" && by_id.get(tn.as_str()).is_some_and(|target| matches!(target.ty.as_str(), "music" | "tts")) {
                tp = "prompt".to_string();
            }
            let link = Link { from_node: fn_, from_port: fp, to_node: tn, to_port: tp };
            if let Some(warning) = wire_issue(&by_id, &link) {
                warnings.push(warning);
            }
            links.push(link);
        }
    }
    let mut graph = Graph { nodes, links, warnings };
    if has_repeated_input(&graph.links) {
        // Keep all links. A user can inspect and repair the target before the next run.
        let by_id: HashMap<_, _> = graph.nodes.iter().map(|node| (node.id.as_str(), node)).collect();
        if let Ok(order) = topo_sort(&graph) {
            if let Ok(branches) = BranchPlan::new(&graph, &order, &by_id) {
                let issues = preflight_competing_inputs(&graph, &order, &by_id, &branches);
                graph.warnings.extend(issues);
            }
        }
    }
    Ok(graph)
}

fn node_id_str(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.clone(),
        Some(other) if !other.is_null() => other.to_string(),
        _ => String::new(),
    }
}

fn port_str(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.clone(),
        Some(other) if !other.is_null() => other.to_string(),
        _ => String::new(),
    }
}

/// Kahn topological sort (stable: graph order breaks ties). Errors name the cycle.
pub fn topo_sort(graph: &Graph) -> Result<Vec<String>> {
    let mut indeg: HashMap<&str, usize> = graph.nodes.iter().map(|n| (n.id.as_str(), 0)).collect();
    let mut out_adj: HashMap<&str, Vec<&str>> = graph.nodes.iter().map(|n| (n.id.as_str(), Vec::new())).collect();
    for l in &graph.links {
        // Inspection can sort the known nodes. Execution separately rejects dangling wires.
        let from = l.from_node.as_str();
        let to = l.to_node.as_str();
        if !from.is_empty() && !to.is_empty() {
            if let (Some(targets), Some(degree)) = (out_adj.get_mut(from), indeg.get_mut(to)) {
                targets.push(to);
                *degree += 1;
            }
        }
    }
    let mut order = Vec::new();
    let mut ready: Vec<&str> = graph
        .nodes
        .iter()
        .filter(|n| indeg[n.id.as_str()] == 0)
        .map(|n| n.id.as_str())
        .collect();
    let mut qi = 0;
    while qi < ready.len() {
        let id = ready[qi];
        qi += 1;
        order.push(id.to_string());
        if let Some(targets) = out_adj.get(id) {
            for t in targets.clone() {
                let d = indeg.get_mut(t).unwrap();
                *d -= 1;
                if *d == 0 {
                    ready.push(t);
                }
            }
        }
    }
    if order.len() != graph.nodes.len() {
        let in_order: HashSet<&str> = order.iter().map(|s| s.as_str()).collect();
        let cyc: Vec<&str> = graph
            .nodes
            .iter()
            .map(|n| n.id.as_str())
            .filter(|id| !in_order.contains(id))
            .collect();
        return Err(NanoodleError(format!("graph has a cycle involving: {}", cyc.join(", "))));
    }
    Ok(order)
}

// ---------------------------------------------------------------- http ctx

/// One classification of an async job's status response. Every NanoGPT
/// long-running job (video, 3D, audio, lipsync) is the same implicit state
/// machine — submit → poll → (Pending)* → Done | Failed — with transient poll
/// errors (network, 5xx, 408, 429) staying in Pending and terminal 4xx
/// failing fast. The families used to hand-roll this loop three times; now
/// each classifier handles a single successful response and the shared loop
/// owns the cadence and deadline.
pub enum PollStep<T> {
    Pending,
    Done(T),
    Failed(String),
}

/// Drive a PollStep state machine to its terminal state.
pub fn poll_loop<T>(
    interval: Duration,
    cap: Duration,
    timeout: String,
    mut step: impl FnMut() -> Result<PollStep<T>>,
) -> Result<T> {
    poll_with_remaining(interval, cap, timeout, |_| step())
}

fn poll_with_remaining<T>(
    interval: Duration,
    cap: Duration,
    timeout: String,
    mut step: impl FnMut(Duration) -> Result<PollStep<T>>,
) -> Result<T> {
    let start = Instant::now();
    loop {
        std::thread::sleep(interval.min(cap.saturating_sub(start.elapsed())));
        let remaining = cap.saturating_sub(start.elapsed());
        if remaining.is_zero() {
            return Err(NanoodleError(timeout));
        }
        let update = step(remaining);
        if start.elapsed() >= cap {
            return Err(NanoodleError(timeout));
        }
        match update? {
            PollStep::Pending => continue,
            PollStep::Done(v) => return Ok(v),
            PollStep::Failed(m) => return Err(NanoodleError(m)),
        }
    }
}

/// Look up a model in an opt-in catalog (JS catalog.mjs catItem): the raw
/// arrays the NanoGPT catalog endpoints return, keyed chat/image/video/audio.
/// Permissive: no catalog, or a model absent from it, changes nothing.
pub fn cat_item<'a>(catalog: &'a Value, kind: &str, id: &str) -> Option<&'a Value> {
    catalog
        .get(kind)?
        .as_array()?
        .iter()
        .find(|m| m.get("id").and_then(|v| v.as_str()) == Some(id))
}

/// Execution context holding the NanoGPT URL, key and pooled HTTP agent.
pub struct Ctx {
    pub base_url: String,
    pub api_key: String,
    /// Opt-in model catalog ({chat, image, video, audio} raw arrays), never
    /// fetched by the library — the caller supplies it, like the JS SDK.
    pub catalog: Option<Value>,
    agent: ureq::Agent,
}

impl Ctx {
    pub fn new(base_url: &str, api_key: &str) -> Ctx {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .build()
            .into();
        Ctx {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            catalog: None,
            agent,
        }
    }

    fn post_response(&self, path: &str, body: &Value) -> Result<ureq::http::Response<ureq::Body>> {
        let url = format!("{}{}", self.base_url, path);
        self.agent
            .post(&url)
            .header("Authorization", &format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .send_json(body)
            .map_err(|e| NanoodleError(format!("request failed: {}", e)))
    }

    fn get_response(
        &self,
        path: &str,
        timeout: Option<Duration>,
    ) -> Result<ureq::http::Response<ureq::Body>> {
        let url = format!("{}{}", self.base_url, path);
        self.agent
            .get(&url)
            .header("Authorization", &format!("Bearer {}", self.api_key))
            .config()
            .timeout_global(timeout)
            .build()
            .call()
            .map_err(|e| NanoodleError(format!("request failed: {}", e)))
    }

    /// POST JSON, mirroring JS client.httpError + costWithHeaders behavior.
    /// Returns (parsed body, cost USD if known).
    pub fn post_json(&self, path: &str, body: &Value) -> Result<(Value, Option<f64>)> {
        self.post_billed_json(path, body).map_err(|e| NanoodleError(e.message))
    }

    fn post_billed_json(&self, path: &str, body: &Value) -> NodeResult<(Value, Option<f64>)> {
        read_billed_json(self.post_response(path, body)?)
    }

    /// GET JSON. Returns (parsed body, cost if known).
    pub fn get_json(&self, path: &str) -> Result<(Value, Option<f64>)> {
        let mut resp = self.get_response(path, None)?;
        let status = resp.status().as_u16();
        let text = resp
            .body_mut()
            .read_to_string()
            .map_err(|e| NanoodleError(format!("response read failed: {}", e)))?;
        if !(200..300).contains(&status) {
            return Err(http_error(status, &text));
        }
        let j: Value = serde_json::from_str(&text)
            .map_err(|e| NanoodleError(format!("invalid JSON response: {}", e)))?;
        let cost = cost_of(&j, None);
        Ok((j, cost))
    }

    fn poll_json<T>(
        &self,
        path: &str,
        interval: Duration,
        cap: Duration,
        timeout: String,
        mut classify: impl FnMut(Value) -> PollStep<T>,
    ) -> Result<T> {
        poll_with_remaining(interval, cap, timeout, |remaining| {
            let Ok(mut response) = self.get_response(path, Some(remaining)) else {
                return Ok(PollStep::Pending);
            };
            let status = response.status().as_u16();
            match status {
                200..=299 => Ok(response
                    .body_mut()
                    .read_json::<Value>()
                    .map(&mut classify)
                    .unwrap_or(PollStep::Pending)),
                // The HTTP status alone explains these failures, even if the
                // server leaves the error body incomplete.
                401 | 402 | 403 => Err(http_error(status, "")),
                400..=499 if !matches!(status, 408 | 429) => {
                    let text = response.body_mut().read_to_string().unwrap_or_default();
                    Err(http_error(status, &text))
                }
                _ => Ok(PollStep::Pending),
            }
        })
    }

    /// POST raw bytes with an explicit content type (TTS/music return audio
    /// bytes; callers that need them use this). Returns (status, content-type,
    /// bytes, cost). Binary responses report price in the headers.
    pub fn post_bytes(&self, path: &str, body: &Value) -> Result<(u16, String, Vec<u8>, Option<f64>)> {
        self.post_billed_bytes(path, body).map_err(|e| NanoodleError(e.message))
    }

    fn post_billed_bytes(
        &self,
        path: &str,
        body: &Value,
    ) -> NodeResult<(u16, String, Vec<u8>, Option<f64>)> {
        read_provider_body(self.post_response(path, body)?)
    }
}

/// Read price headers before consuming a body that might fail mid-stream.
fn read_provider_body(
    mut response: ureq::http::Response<ureq::Body>,
) -> NodeResult<(u16, String, Vec<u8>, Option<f64>)> {
    let status = response.status().as_u16();
    let headers = response.headers();
    let content_type = headers
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_string();
    let header_cost = headers
        .get("x-nanogpt-pricing")
        .and_then(|value| serde_json::from_slice(value.as_bytes()).ok())
        .and_then(|pricing| pricing_cost(&pricing))
        .or_else(|| {
            ["x-cost", "x-nano-cost"].iter().find_map(|key| {
                let amount = headers.get(*key)?.to_str().ok()?.trim().parse::<f64>().ok()?;
                amount.is_finite().then_some(amount)
            })
        });
    let bytes = response.body_mut().read_to_vec().map_err(|error| NodeError {
        message: format!("response read failed: {error}"),
        cost: if (200..300).contains(&status) { header_cost } else { None },
    })?;
    Ok((status, content_type, bytes, header_cost))
}

fn read_billed_json(response: ureq::http::Response<ureq::Body>) -> NodeResult<(Value, Option<f64>)> {
    let (status, _, bytes, header_cost) = read_provider_body(response)?;
    if !(200..300).contains(&status) {
        return Err(http_error(status, &String::from_utf8_lossy(&bytes)).into());
    }
    decode_billed_json(&bytes, header_cost)
}

fn decode_billed_json(bytes: &[u8], header_cost: Option<f64>) -> NodeResult<(Value, Option<f64>)> {
    let json = serde_json::from_slice(bytes).map_err(|error| NodeError {
        message: format!("invalid JSON response: {error}"),
        cost: header_cost,
    })?;
    let cost = cost_of(&json, header_cost);
    Ok((json, cost))
}

fn http_error(status: u16, body: &str) -> NanoodleError {
    if status == 401 || status == 403 {
        return NanoodleError(format!(
            "API key rejected (HTTP {}) — check your NanoGPT key / NANOGPT_API_KEY",
            status
        ));
    }
    let low = body.to_ascii_lowercase();
    if status == 402
        || ["insufficient", "balance", "funds", "not enough", "payment required"]
            .iter()
            .any(|k| low.contains(k))
    {
        return NanoodleError(
            "out of balance — this run needs more credit. Top up at nano-gpt.com, then run again."
                .into(),
        );
    }
    NanoodleError(format!("{}: {}", status, body.chars().take(160).collect::<String>()))
}

fn pricing_cost(pricing: &Value) -> Option<f64> {
    ["costUsd", "cost", "amount"].iter()
        .find_map(|key| pricing.get(key).and_then(Value::as_f64))
}

/// USD cost priority (JS costFromJson): j.cost (>0) → x_nanogpt_pricing.
/// (costUsd|cost|amount) → metadata.cost → j.cost even when 0 → header pricing.
fn cost_of(j: &Value, header_cost: Option<f64>) -> Option<f64> {
    let num = |v: &Value| v.as_f64();
    if let Some(c) = j.get("cost").and_then(num) {
        if c > 0.0 {
            return Some(c);
        }
    }
    if let Some(p) = j.get("x_nanogpt_pricing") {
        if let Some(c) = pricing_cost(p) {
            return Some(c);
        }
    }
    if let Some(c) = j.get("metadata").and_then(|m| m.get("cost")).and_then(num) {
        return Some(c);
    }
    if let Some(c) = j.get("cost").and_then(num) {
        return Some(c);
    }
    header_cost
}

// ---------------------------------------------------------------- engine

#[derive(Debug, Default)]
pub struct NodeRecord {
    pub status: String, // done | error | gated | skipped
    pub out: Map<String, Value>,
    pub error: Option<String>,
    pub cost: Option<f64>,
}

#[derive(Debug, Default)]
pub struct RunResult {
    pub outputs: Map<String, Value>,
    pub nodes: HashMap<String, NodeRecord>,
    pub errors: Vec<String>,
    pub cost_usd: f64,
}

pub fn value_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

pub enum NodeOutcome {
    Done(Map<String, Value>, Option<f64>),
    /// Decide yes/no gate answered no: the node ran and billed, `out` holds its
    /// answer, downstream is skipped (JS status "gated").
    Gated(Map<String, Value>, Option<f64>),
}

fn run_node(
    node: &Node,
    inputs: &HashMap<String, Value>,
    ctx: &Ctx,
) -> std::result::Result<NodeOutcome, NodeError> {
    match node.ty.as_str() {
        "llm" => run_llm(node, inputs, ctx),
        "vision" => run_vision(node, inputs, ctx),
        "decide" => run_decide(node, inputs, ctx),
        "image" => run_image(node, inputs, ctx),
        "edit" => run_edit(node, inputs, ctx),
        "tvideo" => video::run_tvideo(node, inputs, ctx),
        "ivideo" => video::run_ivideo(node, inputs, ctx),
        "vedit" => video::run_vedit(node, inputs, ctx),
        "model3d" => video::run_model3d(node, inputs, ctx),
        "music" => audio::run_music(node, inputs, ctx),
        "tts" => audio::run_tts(node, inputs, ctx),
        "remix" => audio::run_remix(node, inputs, ctx),
        "transcribe" => audio::run_billed_transcribe(node, inputs, ctx),
        "cleanvoice" => audio::run_cleanvoice(node, inputs, ctx),
        "endpoint" => endpoint::run_endpoint(node, inputs, ctx).map_err(Into::into),
        "slice-3d" | "sprite-stack-render" => local_media::run_sprite(node, inputs).map_err(Into::into),
        "resize" => local_media::run_resize(node, inputs, ctx).map_err(Into::into),
        "vframes" => local_media::run_vframes(node, inputs, ctx).map_err(Into::into),
        "render3d" => render3d::run_render3d(node, inputs, ctx).map_err(Into::into),
        "combine" => local_media::run_combine(node, inputs, ctx).map_err(Into::into),
        "soundtrack" => local_media::run_soundtrack(node, inputs, ctx).map_err(Into::into),
        "trim" => local_media::run_trim(node, inputs, ctx).map_err(Into::into),
        "extractaudio" => local_media::run_extractaudio(node, inputs, ctx).map_err(Into::into),
        "inpaint" => local_media::run_billed_inpaint(node, inputs, ctx),
        "lipsync" => local_media::run_lipsync(node, inputs, ctx),
        _ => run_local_node(node, inputs)
            .map(|out| NodeOutcome::Done(out, Some(0.0)))
            .map_err(Into::into),
    }
}

pub fn input_text(inputs: &HashMap<String, Value>, port: &str) -> Option<String> {
    inputs.get(port).map(value_text).filter(|s| !s.is_empty())
}

/// Wired image ports (img1, img2, …) in numeric order — JS collectPorts(IMG_PORT_RE).
fn collect_img_ports(inputs: &HashMap<String, Value>) -> Vec<String> {
    let mut ports: Vec<(u32, &String)> = inputs
        .iter()
        .filter_map(|(k, v)| {
            let rest = k.strip_prefix("img")?;
            if rest.is_empty() || rest.bytes().all(|b| b.is_ascii_digit()) {
                if let Value::String(s) = v {
                    if !s.is_empty() {
                        return Some((rest.parse().unwrap_or(1), s));
                    }
                }
            }
            None
        })
        .collect();
    ports.sort_by_key(|(n, _)| *n);
    ports.into_iter().map(|(_, s)| s.clone()).collect()
}

pub fn model_of(node: &Node) -> String {
    node.field_str("model").unwrap_or_default()
}

/// JS mdl(): the model field, verbatim; empty is a hard error before any
/// request. Every model-driven runner except model3d (defaults to Tripo)
/// and cleanvoice (has a default) goes through here.
pub fn mdl(node: &Node) -> Result<String> {
    let m = model_of(node).trim().to_string();
    if m.is_empty() {
        return Err(NanoodleError(format!("pick a model first (node {})", node.id)));
    }
    Ok(m)
}

/// POST /api/v1/chat/completions — mirrors JS client.chat.
fn chat(
    ctx: &Ctx,
    messages: Value,
    model: &str,
    temperature: Option<f64>,
    response_format: bool,
    reasoning_effort: Option<&str>,
    show_thinking: bool,
    max_tokens: Option<i64>,
) -> NodeResult<(String, Option<f64>)> {
    let mut body = json!({ "model": model, "messages": messages });
    // max_tokens only when the caller explicitly opted in (D1 ruling:
    // default-off, honored exactly when set — mirrors JS llmOpts / Py _run_llm).
    if let Some(mt) = max_tokens {
        body["max_tokens"] = json!(mt);
    }
    if model != "google/gemini-4-carbon" {
        body["temperature"] = json!(temperature.unwrap_or(0.8));
    }
    if response_format {
        body["response_format"] = json!({ "type": "json_object" });
    }
    if model != "google/gemini-4-carbon" {
        if let Some(e) = reasoning_effort {
            body["reasoning_effort"] = json!(e);
        }
    }
    let (j, cost) = ctx.post_billed_json("/api/v1/chat/completions", &body)?;
    let msg = j
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .cloned()
        .unwrap_or(Value::Null);
    let content = msg.get("content").cloned().unwrap_or(Value::Null);
    let mut out = match &content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .map(|p| p.get("text").and_then(|t| t.as_str()).unwrap_or(""))
            .collect::<Vec<_>>()
            .join(""),
        _ => return Err(NodeError { message: "no text in response".into(), cost }),
    };
    if show_thinking {
        if let Some(r) = msg.get("reasoning").and_then(|r| r.as_str()) {
            out = format!("```thinking\n{}\n```\n\n{}", r, out);
        }
    }
    Ok((out, cost))
}

fn run_llm(
    node: &Node,
    inputs: &HashMap<String, Value>,
    ctx: &Ctx,
) -> NodeResult<NodeOutcome> {
    let prompt = prompt_of(node, inputs)
        .ok_or_else(|| NanoodleError("no prompt".into()))?;
    let imgs = collect_img_ports(inputs);
    // IO def backfill: an empty system field means the documented default.
    let system = node.field_str("system").unwrap_or_default();
    let system = if system.trim().is_empty() {
        "You are a helpful, concise assistant.".to_string()
    } else {
        system.trim().to_string()
    };
    let mut messages = vec![json!({ "role": "system", "content": system })];
    if imgs.is_empty() {
        messages.push(json!({ "role": "user", "content": prompt }));
    } else {
        let mut content = vec![json!({ "type": "text", "text": prompt })];
        for u in &imgs {
            content.push(json!({ "type": "image_url", "image_url": { "url": u } }));
        }
        messages.push(json!({ "role": "user", "content": content }));
    }
    let temperature = node.field_str("temperature").and_then(|s| s.parse::<f64>().ok());
    let response_format = node.field_str("format").as_deref() == Some("JSON");
    let effort = node.field_str("reasoningEffort");
    let effort = effort.as_deref().filter(|e| !e.is_empty() && *e != "default");
    let show_thinking = matches!(node.fields.get("showThinking"), Some(Value::Bool(true)))
        || node.field_str("showThinking").as_deref() == Some("true");
    // D1 ruling (2026-11-18): maxTokens opt-in — truthy + numeric field →
    // max_tokens = trunc(+v); absent/non-numeric → nothing sent.
    let max_tokens = match node.fields.get("maxTokens") {
        Some(Value::String(s)) if !s.is_empty() => {
            s.parse::<f64>().ok().map(|v| v.trunc() as i64)
        }
        Some(Value::Number(n)) => n.as_f64().filter(|v| *v != 0.0).map(|v| v.trunc() as i64),
        _ => None,
    };
    let (text, cost) = chat(
        ctx,
        Value::Array(messages),
        &mdl(node)?,
        temperature,
        response_format,
        effort,
        show_thinking,
        max_tokens,
    )?;
    let mut out = Map::new();
    out.insert("text".into(), json!(text));
    Ok(NodeOutcome::Done(out, cost))
}

fn run_vision(
    node: &Node,
    inputs: &HashMap<String, Value>,
    ctx: &Ctx,
) -> NodeResult<NodeOutcome> {
    let img = input_text(inputs, "image")
        .ok_or_else(|| NanoodleError("no image input".into()))?;
    let q = node.field_str("q").unwrap_or_default();
    let q = if q.trim().is_empty() { "Describe this image." } else { q.trim() };
    let messages = json!([{
        "role": "user",
        "content": [
            { "type": "text", "text": q },
            { "type": "image_url", "image_url": { "url": img } }
        ]
    }]);
    let (text, cost) = chat(ctx, messages, &mdl(node)?, None, false, None, false, None)?;
    let mut out = Map::new();
    out.insert("text".into(), json!(text));
    Ok(NodeOutcome::Done(out, cost))
}

// ---- decide (mirrors JS decide.mjs; image fitting is identity in this slice —
// the conformance corpus uses small data: URLs that JS also passes unchanged)

fn decide_lines(s: &str) -> Vec<String> {
    s.split('\n').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()
}

fn branch_decider(node: &Node) -> bool {
    node.ty == "decide" && node.fields.get("branch").and_then(Value::as_bool) == Some(true)
}

fn run_decide(
    node: &Node,
    inputs: &HashMap<String, Value>,
    ctx: &Ctx,
) -> std::result::Result<NodeOutcome, NodeError> {
    let f = &node.fields;
    let mode_raw = node.field_str("mode").unwrap_or_default();
    let branching = branch_decider(node);
    let mode = if branching {
        "yesno"
    } else {
        match mode_raw.as_str() {
            "choose" | "score" | "yesno" => mode_raw.as_str(),
            _ => "pick",
        }
    };
    let text = input_text(inputs, "text").unwrap_or_default().trim().to_string();
    let imgs = collect_img_ports(inputs);
    let question = node.field_str("question").unwrap_or_default();
    let question = if question.trim().is_empty() {
        match mode {
            "pick" => "Which image best matches the brief?",
            "choose" => "Which label fits best?",
            "score" => "How good is it?",
            _ => "Is it good enough to use?",
        }
        .to_string()
    } else {
        question.trim().to_string()
    };
    // question object
    let q: Value = match mode {
        "pick" => {
            if imgs.len() < 2 {
                return Err(NodeError::from(NanoodleError(
                    "pick needs at least two images — wire 2 or more into the image ports".into(),
                )));
            }
            let mut criteria = Map::new();
            for i in 1..=imgs.len() {
                criteria.insert(format!("image_{}", i), json!(format!("image {}", i)));
            }
            json!({ "type": "choice", "instructions": question, "criteria": criteria })
        }
        "choose" => {
            let mut labels: Vec<String> = Vec::new();
            for l in decide_lines(&node.field_str("options").unwrap_or_default()) {
                if !labels.contains(&l) {
                    labels.push(l);
                }
            }
            if labels.len() < 2 {
                return Err(NodeError::from(NanoodleError(
                    "add at least two labels to choose from (one per line)".into(),
                )));
            }
            if labels.len() > 255 {
                return Err(NodeError::from(NanoodleError("too many labels — 255 at most".into())));
            }
            let mut criteria = Map::new();
            for l in &labels {
                criteria.insert(l.clone(), Value::Null);
            }
            json!({ "type": "choice", "instructions": question, "criteria": criteria })
        }
        "score" => {
            let levels_raw = node.field_str("levels").unwrap_or_default();
            let levels = if levels_raw.trim().is_empty() {
                decide_lines("poor\nokay\ngood\ngreat")
            } else {
                decide_lines(&levels_raw)
            };
            if levels.len() < 2 || levels.len() > 10 {
                return Err(NodeError::from(NanoodleError(
                    "the scale needs 2 to 10 levels, worst first (one per line)".into(),
                )));
            }
            json!({ "type": "score", "instructions": question, "criteria": levels })
        }
        _ => json!({ "type": "noul", "instructions": question }),
    };
    if text.is_empty() && imgs.is_empty() && node.field_str("question").unwrap_or_default().trim().is_empty() {
        return Err(NodeError::from(NanoodleError("nothing to judge — wire text or an image into Decide".into())));
    }
    let model = mdl(node)?;
    // Image limits + fitting (JS decideRun): no catalog row → permissive
    // launch limits; a catalog row WITH decision_input decides — no
    // image_input means the model can't see images at all.
    let sent: Vec<String> = if imgs.is_empty() {
        Vec::new()
    } else {
        let item = ctx
            .catalog
            .as_ref()
            .and_then(|c| cat_item(c, "chat", &model));
        let di = item
            .as_ref()
            .and_then(|it| it.get("decision_input"))
            .filter(|v| !v.is_null());
        let lim: Option<(usize, i64, usize)> = match di {
            None => Some((4, 512, 240000)),
            Some(di) => {
                let sees = matches!(di.get("image_input"), Some(v) if !matches!(v, Value::Bool(false)));
                if !sees {
                    None
                } else {
                    let l = di.get("image_limits");
                    let get = |k: &str, dflt: i64| -> i64 {
                        l.and_then(|l| l.get(k))
                            .and_then(|v| v.as_f64())
                            .filter(|n| *n > 0.0)
                            .map(|n| n as i64)
                            .unwrap_or(dflt)
                    };
                    Some((get("maxImages", 4) as usize, get("maxDimension", 512), get("maxEncodedBytes", 240000) as usize))
                }
            }
        };
        let (max_images, max_dim, max_bytes) = match lim {
            None => {
                return Err(NodeError::from(NanoodleError(
                    "this decision model can't see images — pick one that can (e.g. PPLX Decider or Clef)".into(),
                )))
            }
            Some(l) => l,
        };
        if imgs.len() > max_images {
            return Err(NodeError::from(NanoodleError(format!(
                "this decision model takes at most {} images — unwire the extras",
                max_images
            ))));
        }
        let budget = (max_bytes as f64 * 0.95 / imgs.len() as f64).floor() as usize;
        let mut fitted = Vec::with_capacity(imgs.len());
        for u in &imgs {
            fitted.push(local_media::fit_image_jpeg(u, max_dim, budget)?);
        }
        fitted
    };
    let state_for = |ordered: &[String]| -> Value {
        if ordered.is_empty() {
            return json!(text);
        }
        let mut parts: Vec<Value> = Vec::new();
        if !text.is_empty() {
            parts.push(json!(text));
        }
        for (i, u) in ordered.iter().enumerate() {
            if ordered.len() > 1 {
                parts.push(json!(format!("image_{}:", i + 1)));
            }
            parts.push(json!({ "type": "image_url", "image_url": { "url": u } }));
        }
        Value::Array(parts)
    };
    let send = |state: Value| -> NodeResult<(Value, Option<f64>)> {
        let body = json!({ "model": model, "state": state, "questions": { "answer": q } });
        let (j, c) = ctx.post_billed_json("/api/v1/decisions", &body)?;
        // Standard cost extraction first (same as every other family); the
        // decisions API also reports usage.cost — keep it as a fallback.
        // (Was: usage.cost only, which silently dropped top-level cost.)
        let cost = c.or_else(|| {
            j.get("usage").and_then(|u| u.get("cost")).and_then(|v| v.as_f64())
        });
        Ok((j, cost))
    };
    let _ = f;
    let (ans, cost, out_text, out_image) = match mode {
        "pick" => {
            // ask in order and reversed, average the probabilities back (JS decideRun)
            let fwd: Vec<String> = sent.clone();
            let rev: Vec<String> = sent.iter().rev().cloned().collect();
            let (j1, c1) = send(state_for(&fwd))?;
            // A failed second send still billed the first (JS onCost per send).
            let (j2, c2) = send(state_for(&rev))
                .map_err(|mut e| {
                    if let Some(first_cost) = c1 {
                        e.cost = Some(first_cost + e.cost.unwrap_or_default());
                    }
                    e
                })?;
            let probs_of = |j: &Value| -> Map<String, Value> {
                j.pointer("/answers/answer/probabilities")
                    .and_then(|p| p.as_object())
                    .cloned()
                    .unwrap_or_default()
            };
            let p1 = probs_of(&j1);
            let p2 = probs_of(&j2);
            let n = imgs.len();
            let mut sums = vec![0.0f64; n];
            for (k, orig) in (0..n).enumerate() {
                sums[orig] += p1.get(&format!("image_{}", k + 1)).and_then(|v| v.as_f64()).unwrap_or(0.0) / 2.0;
            }
            for (k, orig) in (0..n).rev().enumerate() {
                sums[orig] += p2.get(&format!("image_{}", k + 1)).and_then(|v| v.as_f64()).unwrap_or(0.0) / 2.0;
            }
            let mut best = 0usize;
            for i in 1..n {
                if sums[i] > sums[best] {
                    best = i;
                }
            }
            let cost = match (c1, c2) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            };
            (
                j1,
                cost,
                format!("image {}", best + 1),
                imgs.get(best).cloned().unwrap_or_default(),
            )
        }
        _ => {
            let (j, c) = send(state_for(&sent))?;
            if branching {
                let answer = j.pointer("/answers/answer").unwrap_or(&Value::Null);
                let probability = answer.get("noul").and_then(Value::as_f64)
                    .filter(|p| (0.0..=1.0).contains(p));
                let Some(yes) = probability.filter(|_| answer.get("type").and_then(Value::as_str) == Some("noul")) else {
                    return Err(NodeError {
                        message: "Decide needs a valid yes/no answer to select a branch".into(),
                        cost: c,
                    });
                };
                let branch = if yes >= 0.5 { "yes" } else { "no" };
                let mut out = Map::new();
                out.insert("text".into(), json!(branch));
                out.insert("image".into(), json!(imgs.first().cloned().unwrap_or_default()));
                out.insert("decision".into(), json!({ "branch": branch, "mode": "yesno", "yes": yes }));
                out.insert(branch.into(), json!(if text.is_empty() { branch } else { &text }));
                return Ok(NodeOutcome::Done(out, c));
            }
            let ans = j
                .pointer("/answers/answer")
                .cloned()
                .ok_or_else(|| NodeError { message: "the decision model returned no answer".into(), cost: c })?;
            let (t, im) = match mode {
                "choose" => (
                    ans.get("choice").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    imgs.first().cloned().unwrap_or_default(),
                ),
                "score" => {
                    let sc = ans.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0) + 1.0;
                    let rounded = (sc * 100.0).round() / 100.0;
                    let t = if rounded == rounded.trunc() {
                        format!("{}", rounded as i64)
                    } else {
                        format!("{}", rounded)
                    };
                    (t, imgs.first().cloned().unwrap_or_default())
                }
                _ => {
                    // yesno
                    let py = ans.get("noul").and_then(|v| v.as_f64()).unwrap_or(0.0).clamp(0.0, 1.0);
                    let t = if py >= 0.5 { "yes" } else { "no" }.to_string();
                    let mut out = Map::new();
                    out.insert("text".into(), json!(t));
                    out.insert("image".into(), json!(imgs.first().cloned().unwrap_or_default()));
                    let gate = matches!(node.fields.get("gate"), Some(Value::Bool(true)))
                        || node.field_str("gate").as_deref() == Some("true");
                    if gate && py < 0.5 {
                        return Ok(NodeOutcome::Gated(out, c));
                    }
                    return Ok(NodeOutcome::Done(out, c));
                }
            };
            (j, c, t, im)
        }
    };
    let _ = ans;
    let mut out = Map::new();
    out.insert("text".into(), json!(out_text));
    out.insert("image".into(), json!(out_image));
    Ok(NodeOutcome::Done(out, cost))
}

// ---- image family (POST /v1/images/generations — note: not /api/v1)

fn b64_image_mime(b64: &str) -> &'static str {
    // tiny base64 prefix sniff (JS b64ImageMime): decode the first bytes' worth.
    let head: String = b64.chars().take(24).collect();
    let mut bytes = Vec::new();
    let mut acc: u32 = 0;
    let mut nbits = 0;
    for c in head.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => break,
        };
        acc = (acc << 6) | v as u32;
        nbits += 6;
        if nbits >= 8 {
            nbits -= 8;
            bytes.push((acc >> nbits) as u8);
        }
        if bytes.len() >= 12 {
            break;
        }
    }
    if bytes.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
        "image/png"
    } else if bytes.starts_with(&[0xFF, 0xD8]) {
        "image/jpeg"
    } else if bytes.starts_with(b"GIF") {
        "image/gif"
    } else if bytes.starts_with(b"RIFF") && bytes.len() >= 12 && &bytes[8..12] == b"WEBP" {
        "image/webp"
    } else {
        "image/png"
    }
}

/// fields.modelOpts passthrough + numeric seed (JS imgExtra, minus the
/// custom-civitai AIR handling — deferred, noted in RUST-BRIEF slice notes).
fn img_extra(node: &Node) -> Map<String, Value> {
    let mut e = Map::new();
    if let Some(Value::Object(mo)) = node.fields.get("modelOpts") {
        for (k, v) in mo {
            e.insert(k.clone(), v.clone());
        }
    }
    if let Some(s) = node.fields.get("seed") {
        match s {
            Value::Number(n) => {
                e.insert("seed".into(), json!(n));
            }
            Value::String(s) if !s.trim().is_empty() => {
                if let Ok(n) = s.trim().parse::<f64>() {
                    e.insert("seed".into(), json!(n));
                }
            }
            _ => {}
        }
    }
    e
}

fn gen_image(
    ctx: &Ctx,
    model: &str,
    prompt: Option<&str>,
    size: &str,
    image_data_url: Option<Value>,
    mask_data_url: Option<&str>,
    extra: Map<String, Value>,
    n: i64,
) -> NodeResult<(Vec<String>, Option<f64>)> {
    let (urls, _layers, cost) =
        gen_image_full(ctx, model, prompt, size, image_data_url, mask_data_url, extra, n)?;
    Ok((urls, cost))
}

fn gen_image_full(
    ctx: &Ctx,
    model: &str,
    prompt: Option<&str>,
    size: &str,
    image_data_url: Option<Value>,
    mask_data_url: Option<&str>,
    extra: Map<String, Value>,
    n: i64,
) -> NodeResult<(Vec<String>, Option<Vec<Value>>, Option<f64>)> {
    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert("size".into(), json!(if size.is_empty() { "1024x1024" } else { size }));
    body.insert("n".into(), json!(n));
    body.insert("response_format".into(), json!("b64_json"));
    if let Some(p) = prompt {
        if !p.is_empty() {
            body.insert("prompt".into(), json!(p));
        }
    }
    if let Some(u) = image_data_url {
        body.insert("imageDataUrl".into(), u);
    }
    if let Some(m) = mask_data_url {
        body.insert("maskDataUrl".into(), json!(m));
    }
    for (k, v) in extra {
        body.insert(k, v);
    }
    let (j, cost) = ctx.post_billed_json("/v1/images/generations", &Value::Object(body))?;
    let mut urls = Vec::new();
    let mut layers: Vec<Value> = Vec::new();
    if let Some(data) = j.get("data").and_then(|d| d.as_array()) {
        for d in data {
            let url = if let Some(b) = d.get("b64_json").and_then(|v| v.as_str()) {
                Some(format!("data:{};base64,{}", b64_image_mime(b), b))
            } else {
                d.get("url").and_then(|v| v.as_str()).map(|u| u.to_string())
            };
            if let Some(u) = url {
                urls.push(u);
                layers.push(d.get("layer").cloned().unwrap_or(Value::Null));
            }
        }
    }
    if urls.is_empty() {
        return Err(NodeError { message: "no image in response".into(), cost });
    }
    let layers = if layers.iter().any(|l| !l.is_null()) { Some(layers) } else { None };
    Ok((urls, layers, cost))
}

/// The layers-aware entry point (gen_image_full): also returns the per-item
/// `layer` metadata aligned by index with the urls (layer-decomposition
/// models return {z_index, name, description, bounding_box} per item), or
/// None when no item carries any.
fn gen_image_items(
    ctx: &Ctx,
    model: &str,
    prompt: Option<&str>,
    size: &str,
    image_data_url: Option<Value>,
    mask_data_url: Option<&str>,
    extra: Map<String, Value>,
    n: i64,
) -> NodeResult<(Vec<String>, Option<Vec<Value>>, Option<f64>)> {
    gen_image_full(ctx, model, prompt, size, image_data_url, mask_data_url, extra, n)
}

/// JS promptOf: a wired prompt wins over the field EVEN when empty (no
/// fallback), the result is trimmed, and empty-after-trim is "no prompt"
/// (None here; throwing runners map it to their error).
pub fn prompt_of(node: &Node, inputs: &HashMap<String, Value>) -> Option<String> {
    let raw = match inputs.get("prompt") {
        Some(v) if !v.is_null() => value_text(v),
        _ => match node.fields.get("prompt") {
            Some(v) if !v.is_null() => value_text(v),
            _ => return None,
        },
    };
    let p = raw.trim().to_string();
    if p.is_empty() {
        None
    } else {
        Some(p)
    }
}

/// edit reference ports: image, image2, image3, … in that order (EDIT_IMG_RE).
fn collect_edit_ports(inputs: &HashMap<String, Value>) -> Vec<String> {
    let mut ports: Vec<(u32, &String)> = Vec::new();
    for (k, v) in inputs.iter() {
        let ord = if k == "image" {
            Some(1)
        } else if let Some(rest) = k.strip_prefix("image") {
            if !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()) {
                rest.parse::<u32>().ok()
            } else {
                None
            }
        } else {
            None
        };
        if let (Some(o), Value::String(s)) = (ord, v) {
            if !s.is_empty() {
                ports.push((o, s));
            }
        }
    }
    ports.sort_by_key(|(o, _)| *o);
    ports.into_iter().map(|(_, s)| s.clone()).collect()
}

fn run_image(
    node: &Node,
    inputs: &HashMap<String, Value>,
    ctx: &Ctx,
) -> NodeResult<NodeOutcome> {
    let prompt = prompt_of(node, inputs)
        .ok_or_else(|| NanoodleError("no prompt".into()))?;
    let want: i64 = node
        .field_str("variations")
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(1)
        .max(1);
    let size = node.field_str("size").unwrap_or_default();
    let (urls, cost) = gen_image(
        ctx,
        &mdl(node)?,
        Some(&prompt),
        &size,
        None,
        None,
        img_extra(node),
        want,
    )?;
    let sel = node
        .field_str("sel")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0)
        .min(urls.len() - 1);
    let mut out = Map::new();
    out.insert("image".into(), json!(urls[sel]));
    out.insert("images".into(), json!(urls));
    Ok(NodeOutcome::Done(out, cost))
}

fn run_edit(
    node: &Node,
    inputs: &HashMap<String, Value>,
    ctx: &Ctx,
) -> NodeResult<NodeOutcome> {
    let imgs = collect_edit_ports(inputs);
    if imgs.is_empty() {
        return Err(NanoodleError("no image input".into()).into());
    }
    let prompt = prompt_of(node, inputs).unwrap_or_default();
    if prompt.is_empty() {
        return Err(NanoodleError("no edit instruction".into()).into());
    }
    let src = if imgs.len() > 1 {
        json!(imgs)
    } else {
        json!(imgs[0])
    };
    let size = node.field_str("size").unwrap_or_default();
    let (urls, layers, cost) = gen_image_items(
        ctx,
        &mdl(node)?,
        Some(&prompt),
        &size,
        Some(src),
        None,
        img_extra(node),
        1,
    )?;
    Ok(NodeOutcome::Done(multi_image_out(urls, layers), cost))
}

/// edit/inpaint surface EVERY returned image: `image` = the first (the
/// standing single-output contract), `images` = all of them in response
/// order, and `layers` = the per-item layer metadata aligned by index when
/// any item carries it.
pub(crate) fn multi_image_out(urls: Vec<String>, layers: Option<Vec<Value>>) -> Map<String, Value> {
    let mut out = Map::new();
    out.insert("image".into(), json!(urls[0]));
    out.insert("images".into(), json!(urls));
    if let Some(l) = layers {
        out.insert("layers".into(), Value::Array(l));
    }
    out
}

fn run_local_node(
    node: &Node,
    inputs: &HashMap<String, Value>,
) -> std::result::Result<Map<String, Value>, NanoodleError> {
    let mut out = Map::new();
    match node.ty.as_str() {
        "text" => {
            out.insert("text".into(), json!(node.field_str("text").unwrap_or_default()));
        }
        "upload" | "aupload" | "vupload" | "mupload" => {
            let (key, label) = match node.ty.as_str() {
                "upload" => ("image", "no image — this Image input has no image"),
                "aupload" => ("audio", "no audio — this Audio input has no clip"),
                "vupload" => ("video", "no video — this Video input has no clip"),
                _ => ("model", "no 3D file — drop a .glb first"),
            };
            let v = node.field_str(key).unwrap_or_default();
            if v.is_empty() {
                if node.optional() {
                    out.insert(key.into(), json!(""));
                } else {
                    return Err(NanoodleError(label.into()));
                }
            } else {
                out.insert(key.into(), json!(v));
            }
        }
        "choice" => {
            let opts: Vec<String> = node
                .field_str("options")
                .unwrap_or_default()
                .split('\n')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            let sel = node.field_str("selected");
            let val = match &sel {
                Some(s) if opts.iter().any(|o| o == s) => s.clone(),
                _ => opts.first().cloned().unwrap_or_default(),
            };
            if val.is_empty() {
                return Err(NanoodleError("no options — this Choice has no options to pick from".into()));
            }
            out.insert("text".into(), json!(val));
        }
        "join" => {
            let sep_raw = node.field_str("sep").unwrap_or_else(|| " ".to_string());
            let sep = sep_raw.replace("\\n", "\n");
            let parts: Vec<String> = ["a", "b"]
                .iter()
                .filter_map(|p| inputs.get(*p))
                .map(value_text)
                .filter(|s| !s.is_empty())
                .collect();
            out.insert("text".into(), json!(parts.join(&sep)));
        }
        other => {
            return Err(NanoodleError(format!(
                "node type '{}' is not implemented in nanoodle-rs slice 1",
                other
            )));
        }
    }
    Ok(out)
}

/// What a node's step needs, gathered from the run state under a brief lock:
/// the scheduler hands this to a worker, which executes it lock-free.
enum Gathered {
    /// comment nodes never run.
    Comment,
    /// An upstream failure or competing active wires prevent execution.
    InputError(String),
    /// An upstream gated shut (or was gate-skipped): skipped, unbilled.
    Blocked,
    /// Inputs + the effective node (field overrides applied).
    Ready { node: Node, inputs: HashMap<String, Value> },
}

/// Gather a node's inputs + field overrides from links (JS: every wired
/// value is visible in `inp`; non-input ports additionally override the
/// field). Only the single wired Value is cloned per link.
fn gather_step<'a>(
    node: &Node,
    graph: &Graph,
    outputs_by_node: &HashMap<String, Map<String, Value>>,
    records: &HashMap<String, NodeRecord>,
    links: impl Iterator<Item = &'a Link> + Clone,
    frame_floor: Option<i64>,
) -> Gathered {
    if node.ty == "comment" {
        return Gathered::Comment;
    }
    let mut inputs: HashMap<String, Value> = HashMap::new();
    let mut effective: Option<Node> = None;
    let mut failed: Option<String> = None;
    let mut skipped_err: Vec<String> = Vec::new();
    let mut skipped_in = 0usize;
    let mut gate_blocked = false;
    let mut inbound = 0usize;
    let mut repeated_input = false;
    let emitted_value = |link: &Link| outputs_by_node
        .get(&link.from_node)
        .and_then(|out| out.get(&link.from_port))
        .filter(|value| !value.is_null());
    for l in links.clone() {
        inbound += 1;
        if let Some(rec) = records.get(&l.from_node) {
            // display name only matters for failure bookkeeping — stay lazy
            // on the hot path (every link of every node passes through here)
            let src_name = || {
                graph
                    .nodes
                    .iter()
                    .find(|n| n.id == l.from_node)
                    .map(|n| n.display_name())
                    .unwrap_or_else(|| l.from_node.clone())
            };
            match rec.status.as_str() {
                // a real failure upstream wins over a gate (JS workflow.mjs)
                "error" => {
                    if failed.is_none() {
                        failed = Some(src_name());
                    }
                }
                // onError:"skip" records carry their error under "skipped"
                "skipped" if rec.error.is_some() => {
                    skipped_in += 1;
                    let name = src_name();
                    if !skipped_err.contains(&name) {
                        skipped_err.push(name);
                    }
                }
                "skipped" | "gated" => gate_blocked = true,
                _ => {}
            }
        }
        let Some(v) = emitted_value(l).cloned() else { continue; };
        repeated_input |= inputs.insert(l.to_port.clone(), v.clone()).is_some();
        // The endpoint resolver needs the authored base URL and mode even
        // when a wire supplies a relative path or leaves its mode blank.
        let endpoint_route = node.ty == "endpoint" && matches!(l.to_port.as_str(), "url" | "mode");
        if !is_input_port(node, &l.to_port)
            && !endpoint_route
            && !(node.ty == "decide" && l.to_port == "branch")
        {
            effective
                .get_or_insert_with(|| node.clone())
                .fields
                .insert(l.to_port.clone(), v);
        }
    }
    if let Some(name) = failed {
        return Gathered::InputError(format!("upstream failed: {name}"));
    }
    if inbound > 0 && skipped_in == inbound {
        return Gathered::InputError(format!("all inputs were skipped after failing: {}", skipped_err.join(", ")));
    }
    if gate_blocked {
        return Gathered::Blocked;
    }
    if repeated_input {
        let emitted = links.filter(|link| emitted_value(link).is_some());
        if let Some(issue) = competing_input_issues(emitted).into_iter().next() {
            return Gathered::InputError(issue);
        }
    }
    let mut effective = effective.unwrap_or_else(|| node.clone());
    // Outgoing frame wires set a minimum on the resolved count, including a wired count.
    if let Some(floor) = frame_floor.filter(|&floor| floor > local_media::vframes_count(&effective)) {
        effective.fields.insert("frames".into(), json!(floor));
    }
    Gathered::Ready { node: effective, inputs }
}

/// Execute a gathered step: returns (record, out, blocks_downstream, error).
fn execute_step(g: Gathered, ctx: &Ctx) -> (NodeRecord, Map<String, Value>, bool, Option<String>) {
    match g {
        Gathered::Comment => (
            NodeRecord { status: "skipped".into(), out: Map::new(), error: None, cost: None },
            Map::new(),
            false,
            None,
        ),
        Gathered::Blocked => (
            NodeRecord { status: "skipped".into(), out: Map::new(), error: None, cost: None },
            Map::new(),
            true,
            None,
        ),
        Gathered::InputError(msg) => {
            (
                NodeRecord { status: "error".into(), out: Map::new(), error: Some(msg.clone()), cost: None },
                Map::new(),
                true,
                Some(msg),
            )
        }
        Gathered::Ready { node, inputs } => match run_node(&node, &inputs, ctx) {
            Ok(NodeOutcome::Done(out, cost)) => (
                NodeRecord { status: "done".into(), out: out.clone(), error: None, cost },
                out,
                false,
                None,
            ),
            Ok(NodeOutcome::Gated(out, cost)) => (
                NodeRecord { status: "gated".into(), out: out.clone(), error: None, cost },
                out,
                true,
                None,
            ),
            Err(e) => {
                // fields.onError:"skip" — the failure is recorded (and counted
                // in errors) but the node lands as skipped; dependents see
                // empty ports and run on whatever else they have (JS).
                if node.field_str("onError").as_deref() == Some("skip") && !branch_decider(&node) {
                    (
                        NodeRecord { status: "skipped".into(), out: Map::new(), error: Some(e.message.clone()), cost: e.cost },
                        Map::new(),
                        false,
                        Some(e.message),
                    )
                } else {
                    (
                        NodeRecord { status: "error".into(), out: Map::new(), error: Some(e.message.clone()), cost: e.cost },
                        Map::new(),
                        true,
                        Some(e.message),
                    )
                }
            }
        },
    }
}

/// Shared scheduler state: workers gather a ready node's inputs under the
/// lock, execute lock-free, then record and release dependents.
struct SchedState {
    outputs_by_node: HashMap<String, Map<String, Value>>,
    records: HashMap<String, NodeRecord>,
    errors: Vec<String>,
    cost: f64,
    pending: HashMap<String, usize>,
    ready: std::collections::VecDeque<String>,
    recorded: usize,
    branch_skipped: HashSet<String>,
}

/// Shared routing rules with the JS and Python executors. Explicit Yes/No
/// wires open branches. A join needs an active routed input. Ordinary inputs
/// do not open an unselected branch.
struct BranchPlan<'a> {
    direct: HashSet<usize>,
    selectors: HashMap<&'a str, Vec<usize>>,
}

impl<'a> BranchPlan<'a> {
    fn new(graph: &'a Graph, order: &[String], by_id: &HashMap<&str, &'a Node>) -> Result<Self> {
        let mut direct = HashSet::new();
        for (idx, link) in graph.links.iter().enumerate() {
            let source = by_id.get(link.from_node.as_str());
            if let Some(node) = source.filter(|node| {
                node.ty == "decide" && matches!(link.from_port.as_str(), "yes" | "no")
            }) {
                if !branch_decider(node) {
                    return Err(NanoodleError(format!(
                        "node {}: enable branching on Decide to wire {}",
                        node.id, link.from_port
                    )));
                }
                direct.insert(idx);
            }
        }
        let mut selectors = HashMap::new();
        if !direct.is_empty() {
            let mut inbound: HashMap<&str, Vec<usize>> = HashMap::new();
            for (idx, link) in graph.links.iter().enumerate() {
                inbound.entry(&link.to_node).or_default().push(idx);
            }
            for id in order {
                let links = inbound.get(id.as_str()).map(Vec::as_slice).unwrap_or(&[]);
                let mut controls: Vec<_> = links.iter().copied()
                    .filter(|idx| direct.contains(idx))
                    .collect();
                if controls.is_empty() {
                    controls.extend(links.iter().copied().filter(|idx| {
                        selectors.contains_key(graph.links[*idx].from_node.as_str())
                    }));
                }
                if !controls.is_empty() {
                    selectors.insert(by_id[id.as_str()].id.as_str(), controls);
                }
            }
        }
        Ok(Self { direct, selectors })
    }

    fn closed(&self, idx: usize, link: &Link, state: &SchedState) -> bool {
        if state.branch_skipped.contains(&link.from_node) {
            return true;
        }
        if !self.direct.contains(&idx) {
            return false;
        }
        let selected = state.outputs_by_node.get(&link.from_node)
            .and_then(|out| out.get("decision"))
            .and_then(|d| d.get("branch"))
            .and_then(Value::as_str);
        matches!(selected, Some("yes" | "no")) && selected != Some(link.from_port.as_str())
    }
}

fn preflight_competing_inputs(
    graph: &Graph,
    order: &[String],
    by_id: &HashMap<&str, &Node>,
    branches: &BranchPlan<'_>,
) -> Vec<String> {
    // A legacy Decide gate can skip its downstream regardless of which ports they receive.
    let wired: HashSet<_> = graph.links.iter().filter(|link| matches!(link.to_port.as_str(), "mode" | "gate"))
        .map(|link| (link.to_node.as_str(), link.to_port.as_str())).collect();
    let gates: HashSet<_> = graph.nodes.iter().filter(|node| {
        let id = node.id.as_str();
        node.ty == "decide" && !branch_decider(node)
            && (flag_is_true(node.fields.get("gate")) || wired.contains(&(id, "gate")))
            && (node.field_str("mode").as_deref() == Some("yesno") || wired.contains(&(id, "mode")))
    }).map(|node| node.id.as_str()).collect();
    let mut conditional: HashSet<_> = branches.selectors.keys().copied().collect();
    if !gates.is_empty() {
        let mut outgoing: HashMap<_, Vec<_>> = HashMap::new();
        for link in &graph.links {
            outgoing.entry(link.from_node.as_str()).or_default().push(link.to_node.as_str());
        }
        for id in order.iter().map(String::as_str) {
            if gates.contains(id) || conditional.contains(id) {
                conditional.extend(outgoing.get(id).into_iter().flatten().copied());
            }
        }
    }

    competing_input_issues(graph.links.iter().filter(|link| {
        if conditional.contains(link.to_node.as_str()) || wire_issue(by_id, link).is_some() {
            return false;
        }
        let source = by_id[link.from_node.as_str()];
        if source.fields.get("onError").and_then(Value::as_str) == Some("skip") {
            return false;
        }
        // Some declared ports only emit in particular modes or when enough media exists.
        match source.ty.as_str() {
            "endpoint" | "render3d" => false,
            "vframes" => link.from_port == "frame1",
            "sprite-stack-render" => link.from_port == "image",
            "edit" | "inpaint" => link.from_port != "layers",
            "decide" => matches!(link.from_port.as_str(), "text" | "image"),
            _ => true,
        }
    }))
}

/// Run a materialized graph; unknown types fail fast exactly like JS run().
/// Scheduler: a persistent worker pool (≤32, main thread included). A node
/// becomes ready when every node wired into it has a record, and it starts
/// the moment a worker frees — finer-grained than wave barriers, same
/// observable semantics (the JS engine's allSettled shape): a node only ever
/// reads its own upstreams' final outputs, so statuses, gating and cost are
/// identical to a sequential topo walk; only the wall clock changes.
/// JS io.mjs uniqueKey: first free of base, "base 2", "base 3", … compared
/// case-insensitively against used keys and a reserved set (for outputs: the
/// other sinks' node ids, so a display name can never shadow an id key).
/// `reserved` holds ALL sinks' ids and `res_exempt` is the current sink's
/// own id — exempting it inline keeps this O(1) per check instead of
/// rebuilding a per-sink set (which made the epilogue O(sinks^2)).
/// `hints` caches the next suffix to try per base: keys are only ever
/// added to `used` and `reserved` is fixed during a derivation, so every
/// suffix below the hint is still blocked — probing from the hint finds
/// exactly the key a from-2 probe would, in amortized O(1). (Without it,
/// N same-named sinks cost O(N^2) probes: "Join", "Join 2", … "Join N".)
fn unique_key(
    base: &str,
    used: &mut HashSet<String>,
    reserved: &HashSet<String>,
    res_exempt: &str,
    hints: &mut HashMap<String, usize>,
) -> String {
    // Probe order mirrors JS exactly: base, then "base 2", "base 3", …
    // The hint only skips probes that are provably still blocked: a
    // suffixed key was handed out before (hint > 2) only after the base
    // itself was found blocked, and blocked keys never free up.
    let mut suffix = hints.get(base).copied().unwrap_or(2).max(2);
    let mut key = if suffix > 2 {
        format!("{} {}", base, suffix)
    } else {
        base.to_string()
    };
    loop {
        let lk = key.to_lowercase();
        let blocked = used.contains(&lk) || (lk != res_exempt && reserved.contains(&lk));
        if !blocked {
            break;
        }
        key = format!("{} {}", base, suffix);
        suffix += 1;
    }
    used.insert(key.to_lowercase());
    hints.insert(base.to_string(), suffix);
    key
}

/// One derived user input (JS io.mjs deriveInputs).
struct InputEntry {
    key: String,
    node_id: String,
    field: String,
    optional: bool,
    def: Option<Value>,
}

/// (field, label, optional, spec default) per type — JS INPUT_SPECS.
fn input_specs(ty: &str) -> &'static [(&'static str, &'static str, bool, Option<&'static str>)] {
    match ty {
        "text" => &[("text", "Text", false, None)],
        "upload" => &[("image", "Image", false, None)],
        "aupload" => &[("audio", "Audio", false, None)],
        "vupload" => &[("video", "Video", false, None)],
        "mupload" => &[("model", "3D file", false, None)],
        "llm" => &[
            ("prompt", "Prompt", false, None),
            ("system", "System prompt", true, Some("You are a helpful, concise assistant.")),
        ],
        "image" => &[("prompt", "Image prompt", false, None)],
        "tvideo" => &[("prompt", "Video prompt", false, None)],
        "music" => &[("prompt", "Style / prompt", false, None)],
        "remix" => &[("prompt", "Style / direction", false, None)],
        "tts" => &[("prompt", "Text to speak", false, None)],
        "model3d" => &[("prompt", "3D prompt", true, None)],
        "endpoint" => &[("prompt", "Prompt", false, None)],
        _ => &[],
    }
}

fn flag_is_true(v: Option<&Value>) -> bool {
    match v {
        Some(Value::Bool(true)) => true,
        Some(Value::String(s)) => s == "true",
        _ => false,
    }
}

/// JS io.mjs deriveInputs: INPUT_SPECS fields not fed by a wire, plus the
/// inpaint / choice special cases, each with a unique friendly key.
fn derive_inputs(graph: &Graph) -> Vec<InputEntry> {
    let fed = |id: &str, port: &str| {
        graph.links.iter().any(|l| l.to_node == id && l.to_port == port)
    };
    struct Raw {
        node_id: String,
        field: String,
        label: String,
        optional: bool,
        def: Option<Value>,
    }
    let mut raw: Vec<Raw> = Vec::new();
    let mut mk = |raw: &mut Vec<Raw>, n: &Node, field: &str, label: &str, optional: bool, spec_def: Option<&str>| {
        let cur = n.fields.get(field);
        let def = match cur {
            Some(v) if !matches!(v, Value::Null) && !(matches!(v, Value::String(s) if s.is_empty())) => Some(v.clone()),
            _ => spec_def.map(|s| Value::String(s.to_string())),
        };
        raw.push(Raw {
            node_id: n.id.clone(),
            field: field.to_string(),
            label: label.to_string(),
            optional: optional || flag_is_true(n.fields.get("optional")),
            def,
        });
    };
    for n in &graph.nodes {
        if n.unknown || flag_is_true(n.fields.get("locked")) {
            continue;
        }
        if n.ty == "inpaint" {
            if !fed(&n.id, "prompt") {
                mk(&mut raw, n, "prompt", "What to paint in", false, None);
            }
            let img_fed = fed(&n.id, "image");
            let mask_fed = fed(&n.id, "mask");
            if !img_fed {
                mk(&mut raw, n, "image", if mask_fed { "Image" } else { "Image — the picture to repaint" }, false, None);
            }
            if !mask_fed {
                mk(&mut raw, n, "mask", "Mask (white = repaint)", false, None);
            }
            continue;
        }
        if n.ty == "choice" {
            let options: Vec<String> = n
                .field_str("options")
                .unwrap_or_default()
                .split('\n')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            let before = raw.len();
            mk(&mut raw, n, "selected", "Choice", false, None);
            let e = &mut raw[before];
            let def_ok = matches!(&e.def, Some(Value::String(s)) if options.contains(s));
            if !def_ok {
                e.def = options.first().map(|s| Value::String(s.clone()));
            }
            continue;
        }
        for (field, label, optional, spec_def) in input_specs(&n.ty) {
            let wired = fed(&n.id, field)
                || (n.ty == "endpoint" && *field == "prompt" && fed(&n.id, "text"));
            if !wired {
                mk(&mut raw, n, field, label, *optional, *spec_def);
            }
        }
    }
    // Friendly keys: the node's custom name labels its input when it
    // contributes exactly one required input, or exactly one input at all.
    // Aggregates are precomputed in one pass — a per-entry rescan of `raw`
    // made this O(entries^2), which dominated wide graphs.
    let mut counts: HashMap<&str, (usize, usize, Option<&str>)> = HashMap::new();
    for e in &raw {
        let c = counts.entry(e.node_id.as_str()).or_insert((0, 0, None));
        c.0 += 1;
        if !e.optional {
            c.1 += 1;
            c.2 = Some(e.field.as_str());
        }
    }
    let names: HashMap<&str, &str> = graph
        .nodes
        .iter()
        .map(|n| (n.id.as_str(), n.name.as_deref().unwrap_or("")))
        .collect();
    let mut used: HashSet<String> = HashSet::new();
    let mut key_hints: HashMap<String, usize> = HashMap::new();
    let mut entries = Vec::new();
    for e in &raw {
        let (total, req, req_field) = counts[e.node_id.as_str()];
        let names_it = (req == 1 && req_field == Some(e.field.as_str()))
            || (req == 0 && total == 1);
        let custom = names.get(e.node_id.as_str()).copied().unwrap_or("");
        let base = if !custom.trim().is_empty() && names_it { custom.trim().to_string() } else { e.label.clone() };
        let key = unique_key(&base, &mut used, &HashSet::new(), "", &mut key_hints);
        entries.push(InputEntry {
            key,
            node_id: e.node_id.clone(),
            field: e.field.clone(),
            optional: e.optional,
            def: e.def.clone(),
        });
    }
    entries
}

pub fn run(graph: &Graph, ctx: &Ctx) -> std::result::Result<RunResult, RunFailure> {
    if let Some(n) = graph.nodes.iter().find(|n| n.unknown) {
        return Err(NanoodleError(format!(
            "node {}: unknown node type '{}' — this graph needs a newer nanoodle library",
            n.id, n.ty
        ))
        .into());
    }
    let by_id: HashMap<&str, &Node> = graph.nodes.iter().map(|n| (n.id.as_str(), n)).collect();
    // Validate the current links, including graphs built or edited through the Rust API.
    for link in &graph.links {
        if let Some(message) = wire_issue(&by_id, link) {
            return Err(NanoodleError(message).into());
        }
    }
    // These LLM media handles are valid in the editor, but the Rust LLM runner
    // has no audio or video payload support. A text-only request would lose the wire.
    if let Some(link) = graph.links.iter().find(|link| {
        matches!(link.to_port.as_str(), "audio" | "video")
            && by_id.get(link.to_node.as_str()).is_some_and(|node| node.ty == "llm")
    }) {
        return Err(NanoodleError(format!(
            "{link} uses unsupported LLM input {:?} in the Rust executor", link.to_port
        )).into());
    }
    let order = topo_sort(graph)?; // cycle check (names the cyclic nodes)
    let branches = BranchPlan::new(graph, &order, &by_id)?;
    if has_repeated_input(&graph.links) {
        if let Some(issue) = preflight_competing_inputs(graph, &order, &by_id, &branches).into_iter().next() {
            return Err(NanoodleError(issue).into());
        }
    }
    let total = graph.nodes.len();

    // Required-input validation (JS workflow.run): a derived input whose
    // field is empty and has no default fails the run before anything runs.
    // (No caller inputs in this entry point — explicit set is empty, and
    // field/default backfill is already mirrored inside the node runners.)
    for e in derive_inputs(graph) {
        if e.optional {
            continue;
        }
        let v = by_id[e.node_id.as_str()].fields.get(&e.field);
        let empty = match v {
            None | Some(Value::Null) => true,
            Some(Value::String(s)) => s.trim().is_empty(),
            Some(_) => false,
        };
        if !empty {
            continue;
        }
        let def_fills = matches!(&e.def, Some(Value::String(s)) if !s.is_empty())
            || matches!(&e.def, Some(x) if !matches!(x, Value::Null));
        if !def_fills {
            return Err(NanoodleError(format!(
                "missing required input \"{}\" ({}.{})",
                e.key, e.node_id, e.field
            ))
            .into());
        }
    }

    // Readiness bookkeeping: count each node's distinct upstream nodes once;
    // a node becomes ready when that count drains to zero. (The first cut
    // rescanned nodes × links on every wave — O(waves · nodes · links), which
    // made a 500-node local chain cost ~200 ms of pure scheduling.)
    let mut pending: HashMap<String, usize> =
        graph.nodes.iter().map(|n| (n.id.clone(), 0)).collect();
    let mut dependents: HashMap<String, Vec<String>> = HashMap::new();
    let mut frame_floors: HashMap<&str, i64> = HashMap::new();
    {
        let mut seen: HashSet<(&str, &str)> = HashSet::new();
        for l in &graph.links {
            let source = l.from_node.as_str();
            if by_id.get(source).is_some_and(|n| n.ty == "vframes") {
                if let Some(floor) = local_media::frame_output_index(&l.from_port) {
                    let count = frame_floors.entry(source).or_insert(1);
                    *count = (*count).max(floor);
                }
            }
            if seen.insert((source, l.to_node.as_str())) {
                if let Some(p) = pending.get_mut(&l.to_node) {
                    *p += 1;
                }
                dependents
                    .entry(l.from_node.clone())
                    .or_default()
                    .push(l.to_node.clone());
            }
        }
    }
    let ready: std::collections::VecDeque<String> = graph
        .nodes
        .iter()
        .filter(|n| pending[&n.id] == 0)
        .map(|n| n.id.clone())
        .collect();

    let state = std::sync::Mutex::new(SchedState {
        outputs_by_node: HashMap::new(),
        records: HashMap::new(),
        errors: Vec::new(),
        cost: 0.0,
        pending,
        ready,
        recorded: 0,
        branch_skipped: HashSet::new(),
    });
    let cvar = std::sync::Condvar::new();
    let dependents_ref = &dependents;
    let by_id_ref = &by_id;
    let frame_floors_ref = &frame_floors;
    let branches_ref = &branches;
    let state_ref = &state;
    let cvar_ref = &cvar;
    let worker = move || loop {
        let (id, gathered) = {
            let mut st = state_ref.lock().unwrap();
            loop {
                if st.recorded == total {
                    return;
                }
                if let Some(id) = st.ready.pop_front() {
                    let inactive = branches_ref.selectors.get(id.as_str()).is_some_and(|controls| {
                        controls.iter().all(|idx| branches_ref.closed(*idx, &graph.links[*idx], &st))
                    });
                    let g = if inactive {
                        st.branch_skipped.insert(id.clone());
                        Gathered::Blocked
                    } else {
                        let links = graph.links.iter().enumerate()
                            .filter(|(idx, link)| link.to_node == id && !branches_ref.closed(*idx, link, &st))
                            .map(|(_, link)| link);
                        gather_step(
                            by_id_ref[id.as_str()], graph, &st.outputs_by_node, &st.records, links,
                            frame_floors_ref.get(id.as_str()).copied(),
                        )
                    };
                    break (id, g);
                }
                st = cvar_ref.wait(st).unwrap();
            }
        };
        let (rec, out, is_blocked, err) = execute_step(gathered, ctx);
        let (released, finished) = {
            let mut st = state_ref.lock().unwrap();
            if let Some(c) = rec.cost {
                st.cost += c;
            }
            if let Some(e) = err {
                st.errors.push(format!("node {}: {}", id, e));
            }
            let _ = is_blocked; // downstream semantics now derive from records in gather_step
            st.outputs_by_node.insert(id.clone(), out);
            st.records.insert(id.clone(), rec);
            st.recorded += 1;
            let mut released = 0usize;
            if let Some(deps) = dependents_ref.get(&id) {
                for d in deps {
                    if let Some(p) = st.pending.get_mut(d) {
                        *p -= 1;
                        if *p == 0 {
                            st.ready.push_back(d.clone());
                            released += 1;
                        }
                    }
                }
            }
            (released, st.recorded == total)
        };
        // Wake one waiter per newly-ready node (notify_all on every record
        // made chains pay a 32-thread thundering herd per step); the final
        // record wakes everyone so they can exit.
        if finished {
            cvar_ref.notify_all();
        } else {
            for _ in 0..released {
                cvar_ref.notify_one();
            }
        }
    };

    const MAX_PARALLEL: usize = 32;
    let workers = MAX_PARALLEL.min(total).max(1);
    if workers <= 1 {
        worker();
    } else {
        std::thread::scope(|s| {
            for _ in 1..workers {
                s.spawn(worker);
            }
            worker();
        });
    }

    let st = state.into_inner().unwrap();
    let mut result = RunResult::default();
    result.nodes = st.records;
    result.errors = st.errors;
    result.cost_usd = st.cost;

    let has_outgoing: HashSet<&str> = graph.links.iter().map(|l| l.from_node.as_str()).collect();

    // outputs: sink nodes (no outgoing links, type has outputs) keyed by
    // display name AND node id, primary output port only. Keys are assigned
    // for ALL sinks in topo order (JS deriveOutputs) — a sink that errored
    // still consumes its key, so a surviving same-named sink keeps its
    // suffixed key — and only sinks that produced the port emit a value.
    let sinks: Vec<&Node> = order
        .iter()
        .map(|id| by_id[id.as_str()])
        .filter(|n| {
            !has_outgoing.contains(n.id.as_str())
                && !node_type(&n.ty).map(|t| t.2.is_empty()).unwrap_or(true)
        })
        .collect();
    let sink_ids: HashSet<String> = sinks.iter().map(|n| n.id.to_lowercase()).collect();
    let mut used: HashSet<String> = HashSet::new();
    let mut key_hints: HashMap<String, usize> = HashMap::new();
    let mut error_skips = Vec::new();
    for n in &sinks {
        let key = unique_key(&n.display_name(), &mut used, &sink_ids, &n.id.to_lowercase(), &mut key_hints);
        let rec = match result.nodes.get(&n.id) {
            Some(r) if r.status == "done" || (r.status == "gated" && !r.out.is_empty()) => r,
            Some(r) if r.status == "skipped" => {
                if let Some(error) = &r.error {
                    error_skips.push((key, error));
                }
                continue;
            }
            _ => continue,
        };
        let (_, _, ports) = node_type(&n.ty).unwrap_or(("", &[], &[]));
        let value = if matches!(n.ty.as_str(), "endpoint" | "render3d") {
            // These runners emit the port for the resolved mode. The endpoint's
            // mode can also come from a wire or route prefix.
            ports.iter().find_map(|port| rec.out.get(*port))
        } else {
            rec.out.get(ports[0])
        };
        if let Some(v) = value {
            result.outputs.insert(key, v.clone());
            result.outputs.insert(n.id.clone(), v.clone());
        }
    }

    // JS run() throws RunError when any sink (output) node errored — the
    // error CARRIES the result (records, outputs, cost), like JS/Py.
    // Non-sink errors are collected in result.errors and never throw.
    for n in &graph.nodes {
        if has_outgoing.contains(n.id.as_str()) || n.ty == "comment" {
            continue;
        }
        if let Some(rec) = result.nodes.get(&n.id) {
            if rec.status == "error" {
                return Err(RunFailure {
                    message: format!(
                        "run failed — \"{}\": {}",
                        n.display_name(),
                        rec.error.clone().unwrap_or_default()
                    ),
                    result: Some(result),
                });
            }
        }
    }
    if !sinks.is_empty() && error_skips.len() == sinks.len() {
        let detail = error_skips
            .into_iter()
            .map(|(key, error)| format!("\"{key}\": {error}"))
            .collect::<Vec<_>>()
            .join("; ");
        return Err(RunFailure {
            message: format!("run failed — every output was skipped after failing: {detail}"),
            result: Some(result),
        });
    }
    Ok(result)
}
