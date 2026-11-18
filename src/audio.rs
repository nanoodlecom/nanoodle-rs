//! Audio family: music / tts / remix / transcribe / cleanvoice.
//! Mirrors nanoodle-js client.mjs `audio()` / `transcribe()` and the nodes.mjs
//! runners + `audioParams` (no-catalog path: every gate passes, exactly like
//! an offline export), plus cleanvoice.mjs.
//!
//! POST /api/v1/audio/speech serves music + tts + remix + cleanvoice; the
//! answer is either JSON ({ url } / { runId } → poll /api/tts/status) or raw
//! audio bytes, which become a data: URL. Transcription is a hand-built
//! multipart POST /api/v1/audio/transcriptions (form field MUST be "file");
//! ureq's multipart feature is not enabled in this crate, and Ctx exposes no
//! raw-body POST, so transcribe (like endpoint) drives its own ureq Agent
//! against ctx.base_url with ctx.api_key.

use crate::{input_text, model_of, poll_loop, value_text, Ctx, NanoodleError, Node, NodeError, NodeOutcome, PollStep, Result};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::time::Duration;

const MEDIA_INLINE_MAX: f64 = 4.4 * 1024.0 * 1024.0;
const SPEECH_PATH: &str = "/api/v1/audio/speech";

// ---------------------------------------------------------------- helpers

/// JS mdl(): the model field, verbatim; empty is a hard error.
fn mdl(n: &Node) -> Result<String> {
    let m = model_of(n);
    let m = m.trim();
    if m.is_empty() {
        return Err(NanoodleError(format!("pick a model first (node {})", n.id)));
    }
    Ok(m.to_string())
}

/// JS promptOf(n, inp, errMsg): wired prompt wins, else fields.prompt; trimmed.
fn prompt_of_js(n: &Node, inputs: &HashMap<String, Value>, err: &str) -> Result<String> {
    let raw = match inputs.get("prompt") {
        Some(v) if !v.is_null() => value_text(v),
        _ => match n.fields.get("prompt") {
            Some(v) if !v.is_null() => value_text(v),
            _ => String::new(),
        },
    };
    let p = raw.trim().to_string();
    if p.is_empty() {
        return Err(NanoodleError(err.into()));
    }
    Ok(p)
}

fn non_empty(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(v) => !value_text(v).trim().is_empty(),
    }
}

/// JS Number(v): numbers pass through, strings parse, anything else is NaN.
fn num_of(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => {
            let t = s.trim();
            if t.is_empty() {
                Some(0.0) // JS Number("") === 0; callers gate on non_empty first
            } else {
                t.parse::<f64>().ok()
            }
        }
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn num_field(n: &Node, key: &str) -> Option<f64> {
    n.fields.get(key).and_then(num_of)
}

/// JS httpError (client.mjs) — replicated because lib.rs keeps its copy private.
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

/// Cost from a JSON body only (same priority as lib.rs cost_of minus the
/// header fallback, which Ctx::post_bytes does not expose).
fn cost_of_body(j: &Value) -> Option<f64> {
    if let Some(c) = j.get("cost").and_then(|v| v.as_f64()) {
        if c > 0.0 {
            return Some(c);
        }
    }
    if let Some(p) = j.get("x_nanogpt_pricing") {
        for k in ["costUsd", "cost", "amount"] {
            if let Some(c) = p.get(k).and_then(|v| v.as_f64()) {
                return Some(c);
            }
        }
    }
    if let Some(c) = j.get("metadata").and_then(|m| m.get("cost")).and_then(|v| v.as_f64()) {
        return Some(c);
    }
    j.get("cost").and_then(|v| v.as_f64())
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn b64_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity((bytes.len() + 2) / 3 * 4);
    for chunk in bytes.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        s.push(B64[(n >> 18) as usize & 63] as char);
        s.push(B64[(n >> 12) as usize & 63] as char);
        s.push(if chunk.len() > 1 { B64[(n >> 6) as usize & 63] as char } else { '=' });
        s.push(if chunk.len() > 2 { B64[n as usize & 63] as char } else { '=' });
    }
    s
}

fn b64_decode(s: &str) -> Result<Vec<u8>> {
    let mut table = [255u8; 256];
    for (i, c) in B64.iter().enumerate() {
        table[*c as usize] = i as u8;
    }
    let clean: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    let mut out = Vec::with_capacity(clean.len() / 4 * 3 + 3);
    for chunk in clean.chunks(4) {
        let mut vals = [0u8; 4];
        let mut pad = 0;
        for (i, c) in chunk.iter().enumerate() {
            if *c == b'=' {
                pad += 1;
                vals[i] = 0;
            } else {
                let v = table[*c as usize];
                if v == 255 {
                    return Err(NanoodleError("invalid base64 in data: URL".into()));
                }
                vals[i] = v;
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

fn bytes_to_data_url(bytes: &[u8], mime: &str) -> String {
    format!("data:{};base64,{}", mime, b64_encode(bytes))
}

/// An agent configured exactly like Ctx's (used where Ctx has no suitable
/// helper: multipart transcribe, downloads).
fn raw_agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .http_status_as_error(false)
        .build()
        .into()
}

// ------------------------------------------------------- audioParams (nodes.mjs)

/// JS audioParams with no catalog (sp = null): duration/voice gates pass and
/// format follows the pre-catalog rule (response_format unless mp3).
fn audio_params(n: &Node) -> Result<Map<String, Value>> {
    let f = &n.fields;
    let mut body = Map::new();
    let put_format = |body: &mut Map<String, Value>| {
        if let Some(fmt) = f.get("response_format") {
            if non_empty(Some(fmt)) {
                let fmt = value_text(fmt);
                if fmt != "mp3" {
                    body.insert("response_format".into(), json!(fmt));
                }
            }
        }
    };
    match n.ty.as_str() {
        "music" => {
            if non_empty(f.get("lyrics")) {
                body.insert("lyrics".into(), f.get("lyrics").cloned().unwrap_or(Value::Null));
            }
            if matches!(f.get("instrumental"), Some(Value::Bool(true)))
                || f.get("instrumental").and_then(|v| v.as_str()) == Some("true")
            {
                body.insert("instrumental".into(), json!(true));
            }
            if non_empty(f.get("duration")) {
                if let Some(d) = num_field(n, "duration") {
                    body.insert("duration".into(), json!(d));
                }
            }
            if non_empty(f.get("negative_prompt")) {
                body.insert(
                    "negative_prompt".into(),
                    f.get("negative_prompt").cloned().unwrap_or(Value::Null),
                );
            }
            if non_empty(f.get("seed")) {
                if let Some(s) = num_field(n, "seed") {
                    body.insert("seed".into(), json!(s));
                }
            }
            put_format(&mut body);
        }
        "tts" => {
            if non_empty(f.get("voice")) {
                body.insert("voice".into(), f.get("voice").cloned().unwrap_or(Value::Null));
            }
            if non_empty(f.get("speed")) {
                if let Some(s) = num_field(n, "speed") {
                    if s != 1.0 {
                        body.insert("speed".into(), json!(s));
                    }
                }
            }
            if non_empty(f.get("instructions")) {
                body.insert(
                    "instructions".into(),
                    f.get("instructions").cloned().unwrap_or(Value::Null),
                );
            }
            put_format(&mut body);
        }
        "remix" => {
            if non_empty(f.get("lyrics")) {
                body.insert("lyrics".into(), f.get("lyrics").cloned().unwrap_or(Value::Null));
            }
            if non_empty(f.get("duration")) {
                if let Some(d) = num_field(n, "duration") {
                    body.insert("duration".into(), json!(d));
                }
            }
            put_format(&mut body);
        }
        _ => {}
    }
    // fields.extraJson merges verbatim last (invalid JSON is a hard error).
    if let Some(ej) = f.get("extraJson") {
        if !value_text(ej).trim().is_empty() {
            let parsed: Value = serde_json::from_str(&value_text(ej))
                .map_err(|_| NanoodleError("advanced params: invalid JSON in extraJson".into()))?;
            if let Value::Object(m) = parsed {
                for (k, v) in m {
                    body.insert(k, v);
                }
            }
        }
    }
    // Surface-one-track clamp for music/remix, re-enforced after extraJson.
    if n.ty == "music" || n.ty == "remix" {
        let drop: Vec<String> = body
            .keys()
            .filter(|k| {
                let lk = k.to_ascii_lowercase();
                matches!(
                    lk.as_str(),
                    "number_of_songs" | "n" | "num_songs" | "song_count" | "generation_count"
                        | "generation_count_parameter"
                ) || lk.contains("generation_count")
                    || lk.contains("numsongs")
                    || lk.contains("num_songs")
                    || lk.contains("songcount")
                    || lk.contains("song_count")
            })
            .cloned()
            .collect();
        for k in drop {
            body.remove(&k);
        }
    }
    Ok(body)
}

// ------------------------------------------------------- client.audio (client.mjs)

/// AUDIO_PROMPT_MODEL_RE (client.mjs), hand-matched:
/// (?:^|/)(prompt-to-song|generate-bgm)$ | (?:^|/)mureka-ai/[^/]+/generate-song$ | ^minimax/music-3$
fn is_prompt_model(model: &str) -> bool {
    let m = model.trim().to_ascii_lowercase();
    if m == "minimax/music-3" {
        return true;
    }
    for seg in ["prompt-to-song", "generate-bgm"] {
        if m == seg || m.ends_with(&format!("/{}", seg)) {
            return true;
        }
    }
    // mureka-ai/<one segment>/generate-song anchored at the end, with the
    // "mureka-ai" segment at the start or right after a '/'.
    let mut idx = 0;
    while let Some(pos) = m[idx..].find("mureka-ai/") {
        let at = idx + pos;
        if at == 0 || m.as_bytes()[at - 1] == b'/' {
            let rest = &m[at + "mureka-ai/".len()..];
            if let Some(stripped) = rest.strip_suffix("/generate-song") {
                if !stripped.is_empty() && !stripped.contains('/') {
                    return true;
                }
            }
        }
        idx = at + 1;
    }
    false
}

/// AUDIO_STYLE_MODEL_RE: /yue2-3b\/(?:text|music)-to-music$/i (unanchored start).
fn is_style_model(model: &str) -> bool {
    let m = model.trim().to_ascii_lowercase();
    m.ends_with("yue2-3b/text-to-music") || m.ends_with("yue2-3b/music-to-music")
}

fn audio_mime_for(fmt: Option<&str>) -> &'static str {
    match fmt {
        Some("opus") => "audio/ogg",
        Some("aac") => "audio/aac",
        Some("flac") => "audio/flac",
        Some("wav") | Some("pcm") => "audio/wav",
        _ => "audio/mpeg", // mp3 + AUDIO_MIME fallback
    }
}

/// application/x-www-form-urlencoded component encoding (URLSearchParams).
fn form_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'*' | b'-' | b'.' | b'_' => {
                out.push(b as char)
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

/// JS client._pollAudio: GET /api/tts/status until completed / failed / timeout.
fn poll_audio(ctx: &Ctx, model: &str, j: &Value) -> Result<String> {
    let run_id = match j.get("runId").or_else(|| j.get("id")) {
        Some(Value::String(s)) => s.clone(),
        Some(v) => value_text(v),
        None => return Err(NanoodleError("no audio url in response".into())),
    };
    let mut qs = format!("runId={}&model={}", form_encode(&run_id), form_encode(model));
    if let Some(c) = j.get("cost") {
        if !c.is_null() {
            qs.push_str(&format!("&cost={}", form_encode(&value_text(c))));
        }
    }
    if let Some(p) = j.get("paymentSource") {
        qs.push_str(&format!("&paymentSource={}", form_encode(&value_text(p))));
    }
    if let Some(a) = j.get("isApiRequest") {
        if !a.is_null() {
            qs.push_str(&format!("&isApiRequest={}", form_encode(&value_text(a))));
        }
    }
    let path = format!("/api/tts/status?{}", qs);
    let cap = Duration::from_millis(300_000); // JS client default timeouts.audio
    poll_loop(
        Duration::from_millis(50),
        cap,
        "audio timed out (300s) — the job may still be running on NanoGPT's side".into(),
        || {
            let s = match ctx.get_json(&path) {
                Ok((s, _)) => s,
                Err(e) => {
                    // JS: transient poll failures keep polling; 4xx (except
                    // 408/429) throw. get_json flattens status into the message.
                    let msg = e.0.clone();
                    let status: Option<u32> = msg
                        .split(':')
                        .next()
                        .and_then(|p| p.trim().parse::<u32>().ok());
                    match status {
                        Some(st) if (400..500).contains(&st) && st != 408 && st != 429 => {
                            return Err(e)
                        }
                        _ => return Ok(PollStep::Pending),
                    }
                }
            };
            let st = s
                .get("status")
                .map(value_text)
                .unwrap_or_default()
                .to_ascii_lowercase();
            Ok(match st.as_str() {
                "completed" | "succeeded" => {
                    let url = s
                        .get("audioUrl")
                        .or_else(|| s.get("url"))
                        .or_else(|| s.pointer("/data/audioUrl"))
                        .or_else(|| s.pointer("/data/url"))
                        .filter(|v| !v.is_null())
                        .map(value_text)
                        .filter(|u| !u.is_empty());
                    match url {
                        Some(u) => PollStep::Done(u),
                        None => PollStep::Failed("completed but no audio url".into()),
                    }
                }
                "error" | "failed" | "canceled" | "cancelled" | "content_policy_violation" => {
                    let why = s
                        .get("error")
                        .or_else(|| s.get("message"))
                        .map(value_text)
                        .filter(|m| !m.is_empty())
                        .unwrap_or(st.clone());
                    PollStep::Failed(format!("audio failed: {}", why))
                }
                _ => PollStep::Pending,
            })
        },
    )
}

/// JS client.audio(): POST /api/v1/audio/speech for music + tts + remix.
/// Returns (audio URL or data: URL, cost if known).
fn gen_audio(
    ctx: &Ctx,
    model: &str,
    input: &str,
    extra: Map<String, Value>,
) -> std::result::Result<(String, Option<f64>), NodeError> {
    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert("input".into(), json!(input));
    for (k, v) in extra {
        body.insert(k, v);
    }
    if is_prompt_model(model) {
        let blank = match body.get("prompt") {
            None | Some(Value::Null) => true,
            Some(v) => value_text(v).trim().is_empty(),
        };
        if blank {
            body.insert("prompt".into(), json!(input));
        }
        body.remove("input");
    } else if is_style_model(model) {
        let blank = match body.get("style") {
            None | Some(Value::Null) => true,
            Some(v) => value_text(v).trim().is_empty(),
        };
        if blank {
            body.insert("style".into(), json!(input));
        }
        body.remove("input");
        body.remove("prompt");
    }
    let (status, ctype, bytes, header_cost) =
        ctx.post_bytes(SPEECH_PATH, &Value::Object(body.clone()))?;
    if !(200..300).contains(&status) {
        return Err(http_error(status, &String::from_utf8_lossy(&bytes)).into());
    }
    if ctype.to_ascii_lowercase().contains("application/json") {
        let j: Value = serde_json::from_slice(&bytes)
            .map_err(|e| NanoodleError(format!("invalid JSON response: {}", e)))?;
        let cost = cost_of_body(&j);
        let url = j
            .get("url")
            .or_else(|| j.get("audioUrl"))
            .or_else(|| j.pointer("/data/url"))
            .or_else(|| j.pointer("/data/audioUrl"))
            .filter(|v| !v.is_null())
            .map(value_text)
            .filter(|u| !u.is_empty());
        if let Some(u) = url {
            return Ok((u, cost));
        }
        if j.get("runId").is_some() || j.get("id").is_some() {
            // A failed poll still billed the submit (JS onCost at submit).
            let u = poll_audio(ctx, model, &j)
                .map_err(|e| NodeError { message: e.0, cost })?;
            return Ok((u, cost));
        }
        return Err(NodeError { message: "no audio url in response".into(), cost });
    }
    // Binary body → data: URL; generic content-types take the requested format.
    let mut mime = ctype.split(';').next().unwrap_or("").trim().to_ascii_lowercase();
    if mime.is_empty() || mime == "application/octet-stream" || mime == "binary/octet-stream" {
        let fmt = body
            .get("output_format")
            .or_else(|| body.get("response_format"))
            .map(value_text);
        mime = audio_mime_for(fmt.as_deref()).to_string();
    }
    Ok((bytes_to_data_url(&bytes, &mime), header_cost))
}

fn done_audio(url: String, cost: Option<f64>) -> NodeOutcome {
    let mut out = Map::new();
    out.insert("audio".into(), json!(url));
    NodeOutcome::Done(out, cost)
}

// ---------------------------------------------------------------- runners

pub fn run_music(n: &Node, i: &HashMap<String, Value>, c: &Ctx) -> std::result::Result<NodeOutcome, NodeError> {
    let text = prompt_of_js(n, i, "no prompt — describe the track")?;
    let model = mdl(n)?;
    let (url, cost) = gen_audio(c, &model, &text, audio_params(n)?)?;
    Ok(done_audio(url, cost))
}

pub fn run_tts(n: &Node, i: &HashMap<String, Value>, c: &Ctx) -> std::result::Result<NodeOutcome, NodeError> {
    let text = prompt_of_js(n, i, "no text — give the Speech node something to say")?;
    let model = mdl(n)?;
    let (url, cost) = gen_audio(c, &model, &text, audio_params(n)?)?;
    Ok(done_audio(url, cost))
}

pub fn run_remix(n: &Node, i: &HashMap<String, Value>, c: &Ctx) -> std::result::Result<NodeOutcome, NodeError> {
    let audio = input_text(i, "audio")
        .ok_or_else(|| NanoodleError("no audio — wire a source track into the audio port".into()))?;
    let text = prompt_of_js(n, i, "no prompt — describe the cover / extension first")?;
    let model = mdl(n)?;
    let mut params = audio_params(n)?;
    // https source rides as-is; a data: source is inlined under the size cap.
    if !audio.to_ascii_lowercase().starts_with("https:") && !audio.to_ascii_lowercase().starts_with("http:") {
        if (audio.len() as f64) > MEDIA_INLINE_MAX {
            return Err(NanoodleError(
                "source audio is too large to inline (~4 MB send limit) — use a shorter clip"
                    .into(),
            ).into());
        }
    }
    params.insert("audio".into(), json!(audio));
    let (url, cost) = gen_audio(c, &model, &text, params)?;
    Ok(done_audio(url, cost))
}

// ------------------------------------------------------------- transcribe

/// JS media.dataUrlBytes: split a data: URL into (bytes, mime).
fn data_url_bytes(url: &str) -> Result<(Vec<u8>, String)> {
    let comma = url
        .find(',')
        .ok_or_else(|| NanoodleError("not a data: URL".into()))?;
    if !url.to_ascii_lowercase().starts_with("data:") {
        return Err(NanoodleError("not a data: URL".into()));
    }
    let head = &url[5..comma];
    let mime = head.split(';').next().unwrap_or("");
    let mime = if mime.is_empty() { "application/octet-stream" } else { mime };
    let body = &url[comma + 1..];
    let head_check = format!("{};", head);
    let is_b64 = head.to_ascii_lowercase().ends_with(";base64")
        || head_check.to_ascii_lowercase().contains(";base64;");
    let bytes = if is_b64 {
        b64_decode(body)?
    } else {
        percent_decode(body)
    };
    Ok((bytes, mime.to_string()))
}

fn percent_decode(s: &str) -> Vec<u8> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut k = 0;
    while k < b.len() {
        if b[k] == b'%' && k + 2 < b.len() {
            let hex = std::str::from_utf8(&b[k + 1..k + 3]).unwrap_or("");
            if let Ok(v) = u8::from_str_radix(hex, 16) {
                out.push(v);
                k += 3;
                continue;
            }
        }
        out.push(b[k]);
        k += 1;
    }
    out
}

/// JS client.transcribe(): multipart POST, audio form field named "file".
fn transcribe(
    ctx: &Ctx,
    model: &str,
    audio_url: &str,
    language: &str,
) -> Result<(String, Option<f64>)> {
    let (bytes, mime) = if audio_url.to_ascii_lowercase().starts_with("data:") {
        data_url_bytes(audio_url)?
    } else {
        // downloadMedia: plain GET, no auth headers, up to 3 attempts.
        let agent = raw_agent();
        let mut last_err = String::new();
        let mut got: Option<(Vec<u8>, String)> = None;
        for _ in 0..3 {
            match agent.get(audio_url).call() {
                Ok(mut resp) => {
                    let status = resp.status().as_u16();
                    if !(200..300).contains(&status) {
                        return Err(NanoodleError(format!(
                            "couldn't download the audio to transcribe ({})",
                            status
                        )));
                    }
                    let ct = resp
                        .headers()
                        .get("content-type")
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or("audio/mpeg")
                        .split(';')
                        .next()
                        .unwrap_or("audio/mpeg")
                        .to_string();
                    let data = resp
                        .body_mut()
                        .read_to_vec()
                        .map_err(|e| NanoodleError(format!("response read failed: {}", e)))?;
                    got = Some((data, ct));
                    break;
                }
                Err(e) => last_err = format!("{}", e),
            }
        }
        match got {
            Some(g) => g,
            None => {
                return Err(NanoodleError(format!(
                    "couldn't download the audio to transcribe: {}",
                    last_err
                )))
            }
        }
    };
    if (bytes.len() as f64) > 3.5 * 1024.0 * 1024.0 {
        return Err(NanoodleError(
            "this clip is too big to transcribe directly (~3.5 MB max) — nanoodle sends audio inline; use a shorter clip"
                .into(),
        ));
    }
    let ext = {
        let part = mime.split('/').nth(1).unwrap_or("mp3");
        let part = part.split(';').next().unwrap_or("mp3");
        if part.is_empty() { "mp3" } else { part }.to_string()
    };
    let file_mime = if mime.is_empty() { "audio/mpeg".to_string() } else { mime.clone() };
    let boundary = format!(
        "----nanoodleRsBoundary{:x}{:x}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let mut body: Vec<u8> = Vec::new();
    body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body.extend_from_slice(
        format!(
            "Content-Disposition: form-data; name=\"file\"; filename=\"audio.{}\"\r\n",
            ext
        )
        .as_bytes(),
    );
    body.extend_from_slice(format!("Content-Type: {}\r\n\r\n", file_mime).as_bytes());
    body.extend_from_slice(&bytes);
    body.extend_from_slice(b"\r\n");
    body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body.extend_from_slice(b"Content-Disposition: form-data; name=\"model\"\r\n\r\n");
    body.extend_from_slice(model.as_bytes());
    body.extend_from_slice(b"\r\n");
    if !language.is_empty() {
        body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        body.extend_from_slice(b"Content-Disposition: form-data; name=\"language\"\r\n\r\n");
        body.extend_from_slice(language.as_bytes());
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{}--\r\n", boundary).as_bytes());

    let url = format!("{}/api/v1/audio/transcriptions", ctx.base_url);
    let agent = raw_agent();
    let mut resp = agent
        .post(&url)
        .header("Authorization", &format!("Bearer {}", ctx.api_key))
        .header("Content-Type", &format!("multipart/form-data; boundary={}", boundary))
        .send(body.as_slice())
        .map_err(|e| NanoodleError(format!("request failed: {}", e)))?;
    let status = resp.status().as_u16();
    let pricing = resp
        .headers()
        .get("x-nanogpt-pricing")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let text = resp
        .body_mut()
        .read_to_string()
        .map_err(|e| NanoodleError(format!("response read failed: {}", e)))?;
    if !(200..300).contains(&status) {
        return Err(http_error(status, &text));
    }
    let j: Value = serde_json::from_str(&text)
        .map_err(|e| NanoodleError(format!("invalid JSON response: {}", e)))?;
    let mut cost = cost_of_body(&j);
    if cost.is_none() {
        if let Some(h) = pricing.as_deref() {
            if let Ok(p) = serde_json::from_str::<Value>(h) {
                for k in ["costUsd", "cost", "amount"] {
                    if let Some(c) = p.get(k).and_then(|v| v.as_f64()) {
                        cost = Some(c);
                        break;
                    }
                }
            }
        }
    }
    let txt = j
        .get("transcription")
        .or_else(|| j.get("text"))
        .or_else(|| j.pointer("/data/transcription"))
        .or_else(|| j.pointer("/data/text"))
        .filter(|v| !v.is_null())
        .map(value_text);
    match txt {
        Some(t) => Ok((t, cost)),
        None => Err(NanoodleError("no transcription in response".into())),
    }
}

pub fn run_transcribe(n: &Node, i: &HashMap<String, Value>, c: &Ctx) -> Result<NodeOutcome> {
    let audio = input_text(i, "audio")
        .ok_or_else(|| NanoodleError("no audio input".into()))?;
    let model = mdl(n)?;
    // JS runner: (fields.language || "auto").trim() — "auto" rides along as a
    // real form field because the client appends any non-empty language.
    let language = match n.fields.get("language") {
        Some(v) if !v.is_null() => value_text(v),
        _ => "auto".to_string(),
    };
    let language = language.trim().to_string();
    let (text, cost) = transcribe(c, &model, &audio, &language)?;
    let mut out = Map::new();
    out.insert("text".into(), json!(text));
    Ok(NodeOutcome::Done(out, cost))
}

// ------------------------------------------------------------- cleanvoice

const CLEANVOICE_DEFAULT_MODEL: &str = "elevenlabs/audio-isolation";
const CLEANVOICE_FALLBACK_SECS: f64 = 60.0;

/// cleanvoice.mjs cleanVoiceError: NanoGPT's 400s reworded into the fix.
fn cleanvoice_error(e: NodeError) -> NodeError {
    let m = e.message.clone();
    let low = m.to_ascii_lowercase();
    let mapped = if low.contains("verify the source duration") || low.contains("verify the source audio duration") {
        "NanoGPT couldn't read this file's length — use MP3, WAV, M4A/AAC or MP4 (OGG isn't accepted)."
    } else if low.contains("unable to download") {
        "NanoGPT couldn't download the file — the link must be public (no sign-in, not expired)."
    } else if low.contains("public http source")
        || low.contains("public https source")
        || low.contains("public http(s) source")
    {
        "Clean voice needs a public https link — clips uploaded in the browser can't reach it."
    } else {
        return e;
    };
    NodeError { message: mapped.into(), cost: e.cost }
}

pub fn run_cleanvoice(n: &Node, i: &HashMap<String, Value>, c: &Ctx) -> std::result::Result<NodeOutcome, NodeError> {
    // cleanVoiceSource: wired audio, else wired video, else the pasted link.
    let str_in = |key: &str| -> String {
        match i.get(key) {
            Some(Value::String(s)) => s.trim().to_string(),
            _ => String::new(),
        }
    };
    let a = str_in("audio");
    let v = str_in("video");
    let link = n.field_str("url").unwrap_or_default().trim().to_string();
    let url = if !a.is_empty() {
        a.clone()
    } else if !v.is_empty() {
        v.clone()
    } else {
        link.clone()
    };
    if url.is_empty() {
        return Err(NanoodleError(
            "no audio — wire audio or a video in, or paste a public link to the recording".into(),
        ).into());
    }
    let low = url.to_ascii_lowercase();
    if !(low.starts_with("http://") || low.starts_with("https://")) {
        return Err(NanoodleError(
            "Clean voice needs a hosted file — NanoGPT downloads it from a public link, and a clip uploaded or made in your browser has none. Wire a generated video or track, or paste a public https link to the recording."
                .into(),
        ).into());
    }
    let model = {
        let m = n.field_str("model").unwrap_or_default();
        let m = m.trim();
        if m.is_empty() {
            CLEANVOICE_DEFAULT_MODEL.to_string()
        } else {
            m.to_string()
        }
    };
    // Node hosts can't read media metadata, so the duration quote is always
    // the 60 s fallback (the server re-measures before billing).
    let mut extra = Map::new();
    extra.insert("audio".into(), json!(url));
    extra.insert("duration".into(), json!(CLEANVOICE_FALLBACK_SECS));
    match gen_audio(c, &model, "", extra) {
        Ok((url, cost)) => Ok(done_audio(url, cost)),
        Err(e) => Err(cleanvoice_error(e)),
    }
}
