//! Custom endpoint node — mirror of nanoodle-js endpoint.mjs.
//!
//! POSTs a NanoGPT-shaped body to a URL the graph names. The request goes
//! straight to that URL: never the NanoGPT client, never the API key (custom
//! auth is only fields.auth). http is allowed for loopback and LAN hosts; a
//! public host must be https.
//!
//! The crate has no `url` crate, so URL parsing/validation below is a small
//! hand-rolled equivalent of the pieces endpoint.mjs uses (scheme, userinfo,
//! hostname, pathname/search replacement for path joining).

use crate::{value_text, Ctx, NanoodleError, Node, NodeOutcome, Result};
use serde_json::{json, Map, Value};
use std::collections::HashMap;

const ENDPOINT_DEF_URL: &str = "http://127.0.0.1:8787/v1/chat/completions";
const ENDPOINT_MODES: [&str; 5] = ["chat", "image", "video", "audio", "json"];

// ---------------------------------------------------------------- URL bits

struct ParsedUrl {
    scheme: String,    // lowercased, no "://"
    authority: String, // verbatim (userinfo@host:port)
    host: String,      // hostname, lowercased, brackets kept for IPv6
    has_userinfo: bool,
    path: String,      // may be ""
    query: Option<String>,
}

fn parse_url(raw: &str) -> Option<ParsedUrl> {
    let s = raw.trim();
    let scheme_end = s.find("://")?;
    let scheme_raw = &s[..scheme_end];
    if scheme_raw.is_empty()
        || !scheme_raw
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
        || !scheme_raw.chars().next().map(|c| c.is_ascii_alphabetic()).unwrap_or(false)
    {
        return None;
    }
    let rest = &s[scheme_end + 3..];
    let auth_end = rest
        .find(|c| c == '/' || c == '?' || c == '#')
        .unwrap_or(rest.len());
    let authority = &rest[..auth_end];
    if authority.is_empty() {
        return None;
    }
    let tail = &rest[auth_end..];
    let (path, query) = match tail.find('?') {
        Some(q) => {
            let after = &tail[q + 1..];
            let qend = after.find('#').unwrap_or(after.len());
            (tail[..q].to_string(), Some(after[..qend].to_string()))
        }
        None => {
            let pend = tail.find('#').unwrap_or(tail.len());
            (tail[..pend].to_string(), None)
        }
    };
    let (has_userinfo, hostport) = match authority.rfind('@') {
        Some(at) => (true, &authority[at + 1..]),
        None => (false, authority),
    };
    let host = if hostport.starts_with('[') {
        match hostport.find(']') {
            Some(end) => hostport[..=end].to_string(),
            None => return None,
        }
    } else {
        hostport.split(':').next().unwrap_or("").to_string()
    };
    if host.is_empty() {
        return None;
    }
    Some(ParsedUrl {
        scheme: scheme_raw.to_ascii_lowercase(),
        authority: authority.to_string(),
        host: host.to_ascii_lowercase(),
        has_userinfo,
        path,
        query,
    })
}

fn is_loopback_host(host: &str) -> bool {
    let h = host.trim_start_matches('[').trim_end_matches(']');
    h == "localhost" || h == "127.0.0.1" || h == "::1"
}

fn is_private_ipv4(host: &str) -> bool {
    let parts: Vec<&str> = host.split('.').collect();
    if parts.len() != 4 {
        return false;
    }
    let mut nums = [0u32; 4];
    for (i, p) in parts.iter().enumerate() {
        if p.is_empty() || !p.bytes().all(|b| b.is_ascii_digit()) {
            return false;
        }
        match p.parse::<u32>() {
            Ok(v) if v <= 255 => nums[i] = v,
            _ => return false,
        }
    }
    let (a, b) = (nums[0], nums[1]);
    a == 10 || (a == 192 && b == 168) || (a == 172 && (16..=31).contains(&b)) || (a == 169 && b == 254)
}

fn host_is_local(host: &str) -> bool {
    is_loopback_host(host) || is_private_ipv4(host) || host.to_ascii_lowercase().ends_with(".local")
}

/// endpointUrlOk: Ok(()) or the refusal message.
fn endpoint_url_ok(url: &str) -> Result<()> {
    let s = url.trim();
    if s.is_empty() {
        return Err(NanoodleError("URL required — set the custom endpoint URL".into()));
    }
    let u = match parse_url(s) {
        Some(u) => u,
        None => {
            return Err(NanoodleError(
                "that URL isn’t allowed — use http://localhost, 127.0.0.1, a LAN host, or https"
                    .into(),
            ))
        }
    };
    if u.scheme != "http" && u.scheme != "https" {
        return Err(NanoodleError(
            "that URL isn’t allowed — use http://localhost, 127.0.0.1, a LAN host, or https".into(),
        ));
    }
    if u.has_userinfo {
        return Err(NanoodleError(
            "that URL isn’t allowed — don’t put credentials in the URL; use the Authorization field"
                .into(),
        ));
    }
    if u.scheme == "https" {
        return Ok(());
    }
    if host_is_local(&u.host) {
        return Ok(());
    }
    Err(NanoodleError(
        "that URL isn’t allowed — http is only for localhost, 127.0.0.1, or a LAN host; use https for a public host"
            .into(),
    ))
}

// ------------------------------------------------------------- route parsing

#[derive(Default, Clone)]
struct Route {
    mode: Option<String>,
    url: Option<String>,
    path: Option<String>,
}

fn is_path(s: &str) -> bool {
    let raw = s.trim();
    if raw.is_empty() || !raw.starts_with('/') || raw.len() < 2 {
        return false;
    }
    if raw.chars().any(|c| c.is_whitespace() || c == '·' || c == '|' || c == '•') {
        return false;
    }
    // not a scheme:// URL (a leading "/" already excludes it, kept for parity)
    let low = raw.to_ascii_lowercase();
    if low.contains("://") {
        return false;
    }
    true
}

fn join_path(base: &str, path: &str) -> Option<String> {
    let u = parse_url(base.trim())?;
    let mut p = path.trim().to_string();
    if !p.starts_with('/') {
        p = format!("/{}", p);
    }
    let (pathname, search) = match p.find('?') {
        Some(q) => (p[..q].to_string(), Some(p[q + 1..].to_string())),
        None => (p, None),
    };
    let mut out = format!("{}://{}{}", u.scheme, u.authority, pathname);
    if let Some(s) = search {
        out.push('?');
        out.push_str(&s);
    }
    Some(out)
}

/// endpointParseRoute: pull a mode and/or URL (or bare path) out of a string
/// like "chat · http://host/v1/chat" or "/v1/images".
fn parse_route(s: &str) -> Route {
    let raw = s.trim();
    let mut out = Route::default();
    if raw.is_empty() {
        return out;
    }
    let low = raw.to_ascii_lowercase();
    if ENDPOINT_MODES.contains(&low.as_str()) {
        out.mode = Some(low);
        return out;
    }
    let mut mode = "";
    let mut rest = raw.to_string();
    for prefix in ENDPOINT_MODES {
        if !low.starts_with(prefix) {
            continue;
        }
        let after = &raw[prefix.len()..];
        let ch = after.chars().next();
        let is_sep = matches!(ch, Some('·') | Some('|') | Some(':') | Some('•') | Some('-'))
            || ch.map(|c| c.is_whitespace()).unwrap_or(false);
        if is_sep {
            mode = prefix;
            rest = after
                .trim_start_matches(|c: char| {
                    c.is_whitespace() || matches!(c, '·' | '|' | '•' | ':' | '-')
                })
                .to_string();
            break;
        }
    }
    // strip a trailing " (…)" annotation
    let rest_trim = rest.trim_end().to_string();
    if rest_trim.ends_with(')') {
        if let Some(open) = rest_trim.rfind('(') {
            let inner = &rest_trim[open + 1..rest_trim.len() - 1];
            if !inner.contains(')') && open > 0 && rest_trim.as_bytes()[open - 1].is_ascii_whitespace() {
                rest = rest_trim[..open].trim_end().to_string();
            }
        }
    }
    let rest = rest.trim().to_string();
    // first http(s) URL in the text, trailing punctuation stripped
    let url = find_url(&rest);
    if !mode.is_empty() {
        out.mode = Some(mode.to_string());
    }
    if let Some(u) = url {
        out.url = Some(u);
    } else if is_path(&rest) {
        out.path = Some(rest.clone());
    } else if mode.is_empty() && is_path(raw) {
        out.path = Some(raw.to_string());
    }
    out
}

fn find_url(s: &str) -> Option<String> {
    let low = s.to_ascii_lowercase();
    let mut start: Option<usize> = None;
    for pat in ["https://", "http://"] {
        if let Some(p) = low.find(pat) {
            start = Some(match start {
                None => p,
                Some(cur) => cur.min(p),
            });
        }
    }
    let start = start?;
    let mut end = s.len();
    for (i, c) in s.char_indices() {
        if i > start && (c.is_whitespace() || c == ')') {
            end = i;
            break;
        }
    }
    let mut u = &s[start..end];
    while u.ends_with(['.', ',', ';']) {
        u = &u[..u.len() - 1];
    }
    if u.is_empty() {
        None
    } else {
        Some(u.to_string())
    }
}

fn value_as_string(v: Option<&Value>) -> String {
    match v {
        Some(val) if !val.is_null() => value_text(val),
        _ => String::new(),
    }
}

/// endpointResolveTarget: wired url/mode beat fields; a bare path joins onto
/// the field URL; the default URL applies only when the graph has no url key.
fn resolve_target(n: &Node, inp: &HashMap<String, Value>) -> (String, String) {
    let has_field_url = n.fields.contains_key("url");
    let field_url = if has_field_url {
        value_as_string(n.fields.get("url"))
    } else {
        ENDPOINT_DEF_URL.to_string()
    };
    let inp_url = inp.get("url");
    let url_raw = match inp_url {
        Some(v) if !v.is_null() && !value_text(v).trim().is_empty() => value_text(v),
        _ => field_url.clone(),
    };
    let inp_mode = inp.get("mode");
    let mode_raw = match inp_mode {
        Some(v) if !v.is_null() && !value_text(v).trim().is_empty() => value_text(v),
        _ => value_as_string(n.fields.get("mode")),
    };
    let parsed_url = parse_route(&url_raw);
    let parsed_mode = parse_route(&mode_raw);
    let parsed_field = parse_route(&field_url);
    let path = parsed_url.path.clone().or_else(|| parsed_mode.path.clone());
    let mut resolved = parsed_url.clone();
    if let Some(p) = &path {
        if resolved.url.is_none() {
            let base = parsed_field.url.clone().unwrap_or_else(|| field_url.trim().to_string());
            if !base.is_empty() && !is_path(&base) {
                if let Some(joined) = join_path(&base, p) {
                    resolved = Route {
                        mode: parsed_url.mode.clone(),
                        url: Some(joined),
                        path: None,
                    };
                }
            }
        }
    }
    let mut url = resolved
        .url
        .clone()
        .unwrap_or_else(|| url_raw.trim().to_string());
    let mode = if resolved.url.is_some() && resolved.mode.is_some() {
        resolved.mode.clone().unwrap()
    } else if parsed_mode.url.is_some() && parsed_mode.mode.is_some() {
        url = parsed_mode.url.clone().unwrap();
        parsed_mode.mode.clone().unwrap()
    } else {
        parsed_mode
            .mode
            .clone()
            .or_else(|| resolved.mode.clone())
            .or_else(|| {
                let f = value_as_string(n.fields.get("mode"));
                if f.is_empty() {
                    None
                } else {
                    Some(f)
                }
            })
            .unwrap_or_else(|| "chat".to_string())
    };
    let mode_low = mode.to_ascii_lowercase();
    let mode = if ENDPOINT_MODES.contains(&mode_low.as_str()) {
        mode_low
    } else {
        "chat".to_string()
    };
    (url, mode)
}

// ------------------------------------------------------------ request bodies

fn endpoint_model(n: &Node) -> String {
    let m = n.field_str("model").unwrap_or_default();
    let m = m.trim();
    if m.is_empty() {
        "local".to_string()
    } else {
        m.to_string()
    }
}

fn endpoint_prompt(n: &Node, inp: &HashMap<String, Value>) -> String {
    let v = match inp.get("prompt") {
        Some(v) if !v.is_null() => Some(value_text(v)),
        _ => match inp.get("text") {
            Some(v) if !v.is_null() => Some(value_text(v)),
            _ => match n.fields.get("prompt") {
                Some(v) if !v.is_null() => Some(value_text(v)),
                _ => None,
            },
        },
    };
    v.unwrap_or_default().trim().to_string()
}

fn request_body(mode: &str, n: &Node, inp: &HashMap<String, Value>) -> Value {
    let model = endpoint_model(n);
    match mode {
        "chat" => {
            let prompt = endpoint_prompt(n, inp);
            let mut messages: Vec<Value> = Vec::new();
            let sys = n.field_str("system").unwrap_or_default();
            let sys = sys.trim();
            if !sys.is_empty() {
                messages.push(json!({ "role": "system", "content": sys }));
            }
            let imgs: Vec<Value> = match inp.get("image") {
                Some(Value::Array(a)) => a.clone(),
                Some(v) if !v.is_null() => vec![v.clone()],
                _ => Vec::new(),
            };
            let aud = match inp.get("audio") {
                Some(v) if !v.is_null() => {
                    let s = value_text(v);
                    let data = strip_data_prefix(&s);
                    Some(json!({ "type": "input_audio", "input_audio": { "data": data, "format": "wav" } }))
                }
                _ => None,
            };
            if !imgs.is_empty() || aud.is_some() {
                let mut parts: Vec<Value> = vec![json!({ "type": "text", "text": prompt })];
                for url in &imgs {
                    parts.push(json!({ "type": "image_url", "image_url": { "url": url } }));
                }
                if let Some(a) = aud {
                    parts.push(a);
                }
                messages.push(json!({ "role": "user", "content": parts }));
            } else {
                messages.push(json!({ "role": "user", "content": prompt }));
            }
            json!({ "model": model, "messages": messages, "temperature": 0.8 })
        }
        "image" => {
            let mut ib = Map::new();
            ib.insert("model".into(), json!(model));
            let size = n.field_str("size").unwrap_or_default();
            ib.insert(
                "size".into(),
                json!(if size.is_empty() { "1024x1024".to_string() } else { size }),
            );
            ib.insert("n".into(), json!(1));
            ib.insert("response_format".into(), json!("b64_json"));
            let ip = endpoint_prompt(n, inp);
            if !ip.is_empty() {
                ib.insert("prompt".into(), json!(ip));
            }
            if let Some(v) = inp.get("image") {
                if !v.is_null() {
                    ib.insert("imageDataUrl".into(), v.clone());
                }
            }
            Value::Object(ib)
        }
        "video" => {
            let mut vb = Map::new();
            vb.insert("model".into(), json!(model));
            vb.insert("prompt".into(), json!(endpoint_prompt(n, inp)));
            if let Some(v) = inp.get("image") {
                if !v.is_null() {
                    vb.insert("imageDataUrl".into(), v.clone());
                }
            }
            if let Some(v) = inp.get("video") {
                if !v.is_null() {
                    let s = value_text(v);
                    if s.to_ascii_lowercase().starts_with("http:") || s.to_ascii_lowercase().starts_with("https:") {
                        vb.insert("videoUrl".into(), v.clone());
                    } else {
                        vb.insert("videoDataUrl".into(), v.clone());
                    }
                }
            }
            Value::Object(vb)
        }
        "audio" => {
            let mut ab = Map::new();
            ab.insert("model".into(), json!(model));
            ab.insert("input".into(), json!(endpoint_prompt(n, inp)));
            if let Some(v) = inp.get("audio") {
                if !v.is_null() {
                    let s = value_text(v);
                    if s.to_ascii_lowercase().starts_with("http:") || s.to_ascii_lowercase().starts_with("https:") {
                        ab.insert("audioUrl".into(), v.clone());
                    } else {
                        ab.insert("audioDataUrl".into(), v.clone());
                    }
                }
            }
            Value::Object(ab)
        }
        _ => {
            // json mode: pass the wired values through under their port names.
            let mut jb = Map::new();
            let text_in = inp.get("text");
            if let Some(v) = text_in {
                if !v.is_null() && value_text(v) != "" {
                    jb.insert("text".into(), v.clone());
                }
            }
            if !jb.contains_key("text") {
                let tp = endpoint_prompt(n, inp);
                if !tp.is_empty() {
                    jb.insert("text".into(), json!(tp));
                }
            }
            for key in ["image", "video", "audio"] {
                if let Some(v) = inp.get(key) {
                    if !v.is_null() {
                        jb.insert(key.into(), v.clone());
                    }
                }
            }
            Value::Object(jb)
        }
    }
}

/// JS String.replace(/^data:[^,]*,/, "") on the audio value.
fn strip_data_prefix(s: &str) -> String {
    if s.starts_with("data:") {
        if let Some(comma) = s.find(',') {
            return s[comma + 1..].to_string();
        }
    }
    s.to_string()
}

// ---------------------------------------------------------- response parsing

fn shape_hint(mode: &str) -> &'static str {
    match mode {
        "chat" => "OpenAI chat JSON { choices:[{ message:{ content } }] }",
        "image" => "{ data:[{ b64_json }] } or { data:[{ url }] }",
        "video" => "{ \"url\" } or NanoGPT { output: { video: { url } } }",
        "audio" => "{ \"url\" } or a binary audio body",
        _ => "{ \"text\" } or { \"data\": ... }",
    }
}

fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn endpoint_api_message(body: &str) -> Option<String> {
    // strip a leading "NNN: " prefix, then look for error/message/detail/title
    let stripped = {
        let b = body.trim_start();
        if b.len() > 4
            && b.as_bytes()[0].is_ascii_digit()
            && b.as_bytes()[1].is_ascii_digit()
            && b.as_bytes()[2].is_ascii_digit()
            && b.as_bytes()[3] == b':'
        {
            b[4..].trim_start()
        } else {
            body
        }
    };
    let j: Value = serde_json::from_str(stripped).ok()?;
    let msg = match j.get("error") {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        Some(e) => e
            .get("message")
            .and_then(|m| m.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string()),
        None => None,
    }
    .or_else(|| {
        j.get("message")
            .and_then(|m| m.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
    })
    .or_else(|| {
        j.get("detail")
            .and_then(|m| m.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
    })
    .or_else(|| {
        j.get("title")
            .and_then(|m| m.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
    });
    let msg = collapse_ws(&msg.unwrap_or_default());
    if msg.is_empty() {
        None
    } else {
        Some(msg)
    }
}

fn endpoint_http_error(status: u16, body: &str) -> NanoodleError {
    let extracted = endpoint_api_message(body);
    let hint = match status {
        404 => "check the custom endpoint URL",
        401 | 403 => "check the Authorization field",
        405 => "this endpoint must accept POST",
        413 => "payload too large; send a smaller body",
        415 => "send JSON (Content-Type: application/json)",
        _ => "",
    };
    if let Some(m) = extracted {
        return NanoodleError(if hint.is_empty() {
            m
        } else {
            format!("{} — {}", m, hint)
        });
    }
    let low = body.to_ascii_lowercase();
    if low.contains("<html") || low.contains("<body") || low.contains("<!doctype") {
        return NanoodleError(format!(
            "{} — the URL returned a web page, not JSON; check the custom endpoint URL",
            status
        ));
    }
    if !hint.is_empty() {
        return NanoodleError(format!("{} — {}", status, hint));
    }
    if collapse_ws(body).is_empty() {
        return NanoodleError(format!(
            "{} — the endpoint returned an error with no body; check the URL and mode",
            status
        ));
    }
    let t = body.trim_start();
    if t.starts_with('{') || t.starts_with('[') {
        return NanoodleError(format!(
            "{} — the endpoint rejected the request (check URL, mode, and the posted JSON)",
            status
        ));
    }
    NanoodleError(format!(
        "{}: {}",
        status,
        collapse_ws(body).chars().take(160).collect::<String>()
    ))
}

fn endpoint_not_json_error(mode: &str, ct: &str, raw: &str) -> NanoodleError {
    let sample: String = collapse_ws(raw).chars().take(80).collect();
    let low = raw.to_ascii_lowercase();
    if low.contains("<html") || low.contains("<body") || low.contains("<!doctype")
        || ct.to_ascii_lowercase().contains("text/html")
    {
        return NanoodleError(format!(
            "the endpoint returned a web page, not JSON — check the URL (it must POST {})",
            shape_hint(mode)
        ));
    }
    NanoodleError(format!(
        "response is not JSON — return {}{}",
        shape_hint(mode),
        if sample.is_empty() {
            String::new()
        } else {
            format!(" (got: {})", sample)
        }
    ))
}

/// media.mjs b64ImageMime: base64 prefix sniff.
fn b64_image_mime(b64: &str) -> &'static str {
    if b64.starts_with("/9j/") {
        "image/jpeg"
    } else if b64.starts_with("iVBOR") {
        "image/png"
    } else if b64.starts_with("R0lG") {
        "image/gif"
    } else if b64.starts_with("UklG") {
        "image/webp"
    } else {
        "image/png"
    }
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

fn parse_chat(j: &Value) -> Result<Map<String, Value>> {
    let msg = j
        .pointer("/choices/0/message")
        .cloned()
        .unwrap_or(Value::Null);
    let txt = msg.get("content").cloned().unwrap_or(Value::Null);
    let text = match &txt {
        Value::Null => {
            return Err(NanoodleError(format!(
                "no text in response — return {}",
                shape_hint("chat")
            )))
        }
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .map(|p| p.get("text").and_then(|t| t.as_str()).unwrap_or(""))
            .collect::<Vec<_>>()
            .join(""),
        other => value_text(other),
    };
    let mut out = Map::new();
    out.insert("text".into(), json!(text));
    Ok(out)
}

fn parse_image(j: &Value) -> Result<Map<String, Value>> {
    let mut urls: Vec<String> = Vec::new();
    if let Some(data) = j.get("data").and_then(|d| d.as_array()) {
        for d in data {
            if let Some(b) = d.get("b64_json").and_then(|v| v.as_str()) {
                urls.push(format!("data:{};base64,{}", b64_image_mime(b), b));
            } else if let Some(u) = d.get("url").and_then(|v| v.as_str()) {
                urls.push(u.to_string());
            }
        }
    }
    if urls.is_empty() {
        return Err(NanoodleError(format!(
            "no image in response — return {}",
            shape_hint("image")
        )));
    }
    let mut out = Map::new();
    out.insert("image".into(), json!(urls[0]));
    out.insert("images".into(), json!(urls));
    Ok(out)
}

fn parse_video(j: &Value) -> Result<Map<String, Value>> {
    let out_obj = j
        .pointer("/data/output")
        .or_else(|| j.get("output"))
        .cloned()
        .unwrap_or(Value::Null);
    let url = j
        .get("url")
        .or_else(|| j.get("videoUrl"))
        .filter(|v| !v.is_null())
        .map(value_text)
        .filter(|s| !s.is_empty())
        .or_else(|| {
            out_obj
                .pointer("/video/url")
                .filter(|v| !v.is_null())
                .map(value_text)
                .filter(|s| !s.is_empty())
        })
        .or_else(|| {
            out_obj
                .get("url")
                .filter(|v| !v.is_null())
                .map(value_text)
                .filter(|s| !s.is_empty())
        })
        .or_else(|| {
            out_obj
                .get("video")
                .and_then(|v| v.as_array())
                .and_then(|a| a.first())
                .and_then(|f| f.get("url"))
                .filter(|v| !v.is_null())
                .map(value_text)
                .filter(|s| !s.is_empty())
        })
        .or_else(|| {
            j.pointer("/data/url")
                .filter(|v| !v.is_null())
                .map(value_text)
                .filter(|s| !s.is_empty())
        });
    match url {
        Some(u) => {
            let mut out = Map::new();
            out.insert("video".into(), json!(u));
            Ok(out)
        }
        None => Err(NanoodleError(
            "no video url in response — return { \"url\" } or NanoGPT { output: { video: { url } } }"
                .into(),
        )),
    }
}

fn parse_audio_json(j: &Value) -> Result<Map<String, Value>> {
    let url = j
        .get("url")
        .or_else(|| j.get("audioUrl"))
        .or_else(|| j.pointer("/data/url"))
        .or_else(|| j.pointer("/data/audioUrl"))
        .filter(|v| !v.is_null())
        .map(value_text)
        .filter(|s| !s.is_empty());
    match url {
        Some(u) => {
            let mut out = Map::new();
            out.insert("audio".into(), json!(u));
            Ok(out)
        }
        None => Err(NanoodleError(
            "no audio url in response — return { \"url\" } or a binary audio body".into(),
        )),
    }
}

fn looks_like_echo(j: &Value) -> bool {
    let obj = match j.as_object() {
        Some(o) => o,
        None => return false,
    };
    let h = match obj.get("headers") {
        Some(h) if h.is_object() => h,
        _ => return false,
    };
    let _ = h;
    if !obj.contains_key("data") && !obj.contains_key("json") {
        return false;
    }
    obj.get("url").map(|v| !v.is_null()).unwrap_or(false)
        || obj.get("origin").map(|v| !v.is_null()).unwrap_or(false)
        || obj.get("method").map(|v| !v.is_null()).unwrap_or(false)
}

fn json_mode_text(v: &Value) -> String {
    match v {
        Value::String(s) => {
            let t = s.trim();
            if !t.is_empty() {
                if let Ok(p) = serde_json::from_str::<Value>(t) {
                    if p.is_object() {
                        return json_mode_text(&p);
                    }
                }
            }
            s.clone()
        }
        Value::Object(_) => {
            if looks_like_echo(v) {
                let inner = if v.get("json").map(|x| !x.is_null()).unwrap_or(false) {
                    v.get("json").unwrap()
                } else {
                    v.get("data").unwrap_or(&Value::Null)
                };
                if inner != v {
                    return json_mode_text(inner);
                }
            }
            if let Some(t) = v.get("text") {
                if !t.is_null() {
                    return value_text(t);
                }
            }
            serde_json::to_string_pretty(v).unwrap_or_else(|_| value_text(v))
        }
        Value::Null => String::new(),
        other => value_text(other),
    }
}

fn parse_json_mode(j: &Value) -> Result<Map<String, Value>> {
    let text = if j.get("text").map(|t| !t.is_null()).unwrap_or(false) && !looks_like_echo(j) {
        value_text(j.get("text").unwrap())
    } else if looks_like_echo(j) {
        let inner = if j.get("json").map(|x| !x.is_null()).unwrap_or(false) {
            j.get("json").unwrap().clone()
        } else {
            j.get("data").cloned().unwrap_or(Value::Null)
        };
        json_mode_text(&inner)
    } else if j.as_object().map(|o| o.contains_key("data")).unwrap_or(false) {
        json_mode_text(j.get("data").unwrap())
    } else {
        return Err(NanoodleError(
            "json mode expected { \"text\" } or { \"data\": ... } — not a chat/completions wrapper"
                .into(),
        ));
    };
    let mut out = Map::new();
    out.insert("text".into(), json!(text));
    Ok(out)
}

fn parse_response(mode: &str, ct: &str, bytes: &[u8]) -> Result<Map<String, Value>> {
    if mode == "audio" && !ct.to_ascii_lowercase().contains("json") {
        let mut mime = ct.split(';').next().unwrap_or("").trim().to_ascii_lowercase();
        if mime.is_empty() || mime == "application/octet-stream" || mime == "binary/octet-stream" {
            mime = "audio/mpeg".to_string();
        }
        let mut out = Map::new();
        out.insert(
            "audio".into(),
            json!(format!("data:{};base64,{}", mime, b64_encode(bytes))),
        );
        return Ok(out);
    }
    let raw = String::from_utf8_lossy(bytes).to_string();
    if raw.trim().is_empty() {
        return Err(NanoodleError(
            "empty response — the endpoint returned no body; check the URL and mode".into(),
        ));
    }
    let j: Value = match serde_json::from_str(&raw) {
        Ok(j) => j,
        Err(_) => return Err(endpoint_not_json_error(mode, ct, &raw)),
    };
    match mode {
        "chat" => parse_chat(&j),
        "image" => parse_image(&j),
        "video" => parse_video(&j),
        "audio" => parse_audio_json(&j),
        _ => parse_json_mode(&j),
    }
}

// ------------------------------------------------------------------- runner

pub fn run_endpoint(n: &Node, i: &HashMap<String, Value>, _c: &Ctx) -> Result<NodeOutcome> {
    let (url, mode) = resolve_target(n, i);
    endpoint_url_ok(&url)?;
    let body = request_body(&mode, n, i);
    // Custom auth is only fields.auth; the NanoGPT key never goes to this URL.
    let mut auth: Option<String> = None;
    let a = n.field_str("auth").unwrap_or_default();
    let a = a.trim();
    if !a.is_empty() {
        auth = Some(if a.chars().any(|c| c.is_whitespace()) {
            a.to_string()
        } else {
            format!("Bearer {}", a)
        });
    }
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .build()
        .into();
    let mut req = agent
        .post(&url)
        .header("Content-Type", "application/json");
    if let Some(a) = &auth {
        req = req.header("Authorization", a);
    }
    let mut resp = match req.send_json(&body) {
        Ok(r) => r,
        Err(e) => {
            // JS fetch failures land here as opaque network errors.
            let local = parse_url(&url).map(|u| host_is_local(&u.host)).unwrap_or(false);
            return Err(NanoodleError(if local {
                "blocked — CORS (Access-Control-Allow-Origin + OPTIONS) or Chrome local-network permission"
                    .to_string()
            } else {
                format!("blocked by CORS — your server needs Access-Control-Allow-Origin ({})", e)
            }));
        }
    };
    let status = resp.status().as_u16();
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let bytes = resp
        .body_mut()
        .read_to_vec()
        .map_err(|e| NanoodleError(format!("response read failed: {}", e)))?;
    if !(200..300).contains(&status) {
        let text = String::from_utf8_lossy(&bytes).to_string();
        let truncated: String = text.chars().take(800).collect();
        return Err(endpoint_http_error(status, &truncated));
    }
    let out = parse_response(&mode, &ct, &bytes)?;
    Ok(NodeOutcome::Done(out, None))
}
