//! Local media ops: resize / vframes / combine / soundtrack / trim /
//! extractaudio / inpaint / lipsync.
//!
//! Mirrors nanoodle-js `local-media.mjs` + the `nodes.mjs` runners (JS is the
//! source of truth; Py is ffmpeg-only where JS is pure-first):
//!   1. Pure path first — PNG resize/mask composite via a native PNG codec +
//!      zlib inflate/deflate written here, PCM-WAV trim via a native WAV
//!      codec, MP4CAT lossless remux for matching mp4s (ported from
//!      mp4cat.mjs).
//!   2. ffmpeg/ffprobe on PATH when the pure path can't handle the format —
//!      same argument vectors as JS.
//! Outputs are data: URLs. Local nodes cost Some(0.0); inpaint/lipsync bill
//! through the API like their JS twins.

use crate::{input_text, mdl, multi_image_out, prompt_of, value_text, Ctx, NanoodleError, Node, NodeError, NodeOutcome, Result};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MAX_FRAMES: i64 = 12;
const MAX_IMAGE_DIM: u32 = 8192;
const MAX_WAV_SAMPLES: usize = 50_000_000;
const MEDIA_INLINE_MAX: usize = 4_610_334; // JS 4.4 * 1024 * 1024, compared on data-URL length
const INLINE_IMAGE_BUDGET: f64 = MEDIA_INLINE_MAX as f64 - 128.0 * 1024.0;
const PROC_TIMEOUT: Duration = Duration::from_millis(120_000);

fn err(msg: impl Into<String>) -> NanoodleError {
    NanoodleError(msg.into())
}

/// JS Math.round: floor(x + 0.5).
fn js_round(x: f64) -> f64 {
    (x + 0.5).floor()
}

/// JS Number → string for the values ffmpeg args carry ("0", "0.1", "30", …).
fn js_num(x: f64) -> String {
    if x == x.trunc() && x.abs() < 1e21 {
        format!("{}", x as i64)
    } else {
        format!("{}", x)
    }
}

// ---------------------------------------------------------------- base64

const B64_ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn b64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(B64_ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(B64_ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { B64_ALPHABET[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { B64_ALPHABET[n as usize & 63] as char } else { '=' });
    }
    out
}

fn b64_decode(s: &str) -> Result<Vec<u8>> {
    let mut vals: Vec<u8> = Vec::with_capacity(s.len());
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => break,
            b' ' | b'\t' | b'\n' | b'\r' => continue,
            _ => return Err(err("invalid base64 in data: URL")),
        };
        vals.push(v);
    }
    let mut out = Vec::with_capacity(vals.len() * 3 / 4);
    for chunk in vals.chunks(4) {
        if chunk.len() < 2 {
            break;
        }
        let n = chunk.iter().fold(0u32, |acc, &v| (acc << 6) | v as u32) << (6 * (4 - chunk.len()));
        out.push((n >> 16) as u8);
        if chunk.len() > 2 {
            out.push((n >> 8) as u8);
        }
        if chunk.len() > 3 {
            out.push(n as u8);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------- data URLs / mime

/// JS sniffMime (media.mjs), verbatim order.
fn sniff_mime(b: &[u8]) -> &'static str {
    let ascii = |off: usize, s: &str| -> bool {
        b.len() >= off + s.len() && &b[off..off + s.len()] == s.as_bytes()
    };
    if b.len() >= 8 && b[0] == 0x89 && ascii(1, "PNG") {
        return "image/png";
    }
    if b.len() >= 3 && b[0] == 0xff && b[1] == 0xd8 {
        return "image/jpeg";
    }
    if ascii(0, "GIF8") {
        return "image/gif";
    }
    if ascii(0, "RIFF") && b.len() >= 12 && ascii(8, "WEBP") {
        return "image/webp";
    }
    if ascii(0, "RIFF") && b.len() >= 12 && ascii(8, "WAVE") {
        return "audio/wav";
    }
    if ascii(0, "ID3") || (b.len() >= 2 && b[0] == 0xff && (b[1] & 0xe0) == 0xe0) {
        return "audio/mpeg";
    }
    if ascii(0, "OggS") {
        return "audio/ogg";
    }
    if ascii(0, "fLaC") {
        return "audio/flac";
    }
    if b.len() >= 12 && ascii(4, "ftyp") {
        return "video/mp4";
    }
    if b.len() >= 4 && b[0] == 0x1a && b[1] == 0x45 && b[2] == 0xdf && b[3] == 0xa3 {
        return "video/webm";
    }
    if b.len() >= 4 && ascii(0, "glTF") {
        return "model/gltf-binary";
    }
    "application/octet-stream"
}

fn bytes_to_data_url(bytes: &[u8], mime: Option<&str>) -> String {
    format!("data:{};base64,{}", mime.unwrap_or_else(|| sniff_mime(bytes)), b64_encode(bytes))
}

/// JS dataUrlBytes: returns (bytes, mime). Non-base64 payloads are
/// percent-decoded as UTF-8 (decodeURIComponent).
fn data_url_bytes(url: &str) -> Result<(Vec<u8>, String)> {
    let comma = url.find(',').ok_or_else(|| err("not a data: URL"))?;
    if !url.starts_with("data:") {
        return Err(err("not a data: URL"));
    }
    let head = &url[5..comma];
    let mime = head.split(';').next().unwrap_or("");
    let mime = if mime.is_empty() { "application/octet-stream" } else { mime };
    let body = &url[comma + 1..];
    let is_b64 = head.to_ascii_lowercase().ends_with(";base64")
        || format!("{};", head).to_ascii_lowercase().contains(";base64;");
    let bytes = if is_b64 {
        b64_decode(body)?
    } else {
        percent_decode(body).into_bytes()
    };
    Ok((bytes, mime.to_string()))
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            let hex = |c: u8| -> Option<u8> {
                match c {
                    b'0'..=b'9' => Some(c - b'0'),
                    b'a'..=b'f' => Some(c - b'a' + 10),
                    b'A'..=b'F' => Some(c - b'A' + 10),
                    _ => None,
                }
            };
            if let (Some(h), Some(l)) = (hex(b[i + 1]), hex(b[i + 2])) {
                out.push(h * 16 + l);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// data:/https URL → bytes (JS urlBytes). http(s) downloads via ureq.
fn url_bytes(url: &str) -> Result<Vec<u8>> {
    if url.is_empty() {
        return Err(err("no media input"));
    }
    if url.to_ascii_lowercase().starts_with("data:") {
        return Ok(data_url_bytes(url)?.0);
    }
    if url.to_ascii_lowercase().starts_with("http:") || url.to_ascii_lowercase().starts_with("https:") {
        let mut resp = ureq::get(url)
            .call()
            .map_err(|e| err(format!("can't download media: {}", e)))?;
        return resp
            .body_mut()
            .read_to_vec()
            .map_err(|e| err(format!("can't download media: {}", e)));
    }
    Err(err("media must be a data: or http(s) URL"))
}

// ---------------------------------------------------------------- processes / temp files

/// JS runProc: 120 s cap, ENOENT → the "need ffmpeg on PATH" message,
/// non-zero exit → "{bin} failed (exit {code}): {stderr tail}".
fn run_proc(bin: &str, args: &[String]) -> Result<String> {
    let mut child = Command::new(bin)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                err(format!(
                    "local media nodes need ffmpeg on PATH (not found: {}). Install ffmpeg, or run this graph in the nanoodle browser app.",
                    bin
                ))
            } else {
                err(format!("couldn't start {}: {}", bin, e))
            }
        })?;
    let mut out_pipe = child.stdout.take().unwrap();
    let mut err_pipe = child.stderr.take().unwrap();
    let t_out = std::thread::spawn(move || {
        let mut v = Vec::new();
        let _ = out_pipe.read_to_end(&mut v);
        v
    });
    let t_err = std::thread::spawn(move || {
        let mut v = Vec::new();
        let _ = err_pipe.read_to_end(&mut v);
        v
    });
    let start = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break s,
            Ok(None) => {
                if start.elapsed() > PROC_TIMEOUT {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(err(format!("{} timed out after {}ms", bin, PROC_TIMEOUT.as_millis())));
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(e) => return Err(err(format!("{} failed: {}", bin, e))),
        }
    };
    let stdout = t_out.join().unwrap_or_default();
    let stderr = t_err.join().unwrap_or_default();
    if status.success() {
        Ok(String::from_utf8_lossy(&stdout).into_owned())
    } else {
        let err_text = String::from_utf8_lossy(&stderr);
        let trimmed = err_text.trim();
        let tail: String = {
            let chars: Vec<char> = trimmed.chars().collect();
            let start = chars.len().saturating_sub(400);
            chars[start..].iter().collect()
        };
        Err(err(format!(
            "{} failed (exit {}): {}",
            bin,
            status.code().unwrap_or(-1),
            if tail.is_empty() { "no stderr".to_string() } else { tail }
        )))
    }
}

fn is_missing_ffmpeg(e: &NanoodleError) -> bool {
    e.0.contains("need ffmpeg on PATH")
}

mod sprite_avif;
mod sprite_codecs;
mod sprite_geometry;
mod sprite_jpeg;
mod sprite_orientation;
mod sprite_stack;
mod sprite_webp;
pub use sprite_stack::run_sprite;

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Result<TempDir> {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        for _ in 0..100 {
            let n = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let p = std::env::temp_dir().join(format!(
                "nanoodle-media-{}-{}-{}",
                std::process::id(),
                nanos,
                n
            ));
            match std::fs::create_dir(&p) {
                Ok(()) => return Ok(TempDir(p)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(err(format!("couldn't create temp dir: {}", e))),
            }
        }
        Err(err("couldn't create temp dir"))
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// JS writeInput: preserve a sensible extension so ffmpeg picks the demuxer —
/// sniffed mime for data: URLs, URL path extension for http(s).
fn write_input(dir: &TempDir, name: &str, url: &str) -> Result<PathBuf> {
    let bytes = url_bytes(url)?;
    let ext: String = if url.to_ascii_lowercase().starts_with("data:") {
        let mime = sniff_mime(&bytes);
        if mime.contains("png") {
            ".png".to_string()
        } else if mime.contains("jpeg") {
            ".jpg".to_string()
        } else if mime.contains("webp") {
            ".webp".to_string()
        } else if mime.contains("gif") {
            ".gif".to_string()
        } else if mime.contains("wav") {
            ".wav".to_string()
        } else if mime.contains("mpeg") || mime.contains("mp3") {
            ".mp3".to_string()
        } else if mime.contains("mp4") {
            ".mp4".to_string()
        } else if mime.contains("webm") {
            ".webm".to_string()
        } else {
            ".bin".to_string()
        }
    } else {
        url_ext(url).unwrap_or_else(|| ".bin".to_string())
    };
    let path = dir.path(&format!("{}{}", name, ext));
    std::fs::write(&path, &bytes).map_err(|e| err(format!("couldn't write temp file: {}", e)))?;
    Ok(path)
}

/// JS regex \.([a-z0-9]{2,5})(?:\?|$) against the URL string.
fn url_ext(url: &str) -> Option<String> {
    let b = url.as_bytes();
    let mut i = b.len();
    // find last '.' such that the rest matches [a-z0-9]{2,5} then end or '?'
    while i > 0 {
        i -= 1;
        if b[i] == b'.' {
            let rest = &url[i + 1..];
            let ext_len = rest.find('?').unwrap_or(rest.len());
            let ext = &rest[..ext_len];
            if (2..=5).contains(&ext.len())
                && ext.bytes().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
                && (ext_len == rest.len() || rest.as_bytes()[ext_len] == b'?')
            {
                return Some(format!(".{}", ext.to_ascii_lowercase()));
            }
            return None;
        }
    }
    None
}

fn data_url_from_file(path: &PathBuf, mime_hint: Option<&str>) -> Result<String> {
    let bytes = std::fs::read(path).map_err(|e| err(format!("couldn't read temp file: {}", e)))?;
    Ok(bytes_to_data_url(&bytes, mime_hint))
}

// ---------------------------------------------------------------- JS number parsing

/// JS parseFloat on a JSON field value (Number → itself, String → longest
/// valid numeric prefix, anything else → NaN/None).
fn js_parse_float(v: Option<&Value>) -> Option<f64> {
    match v {
        Some(Value::Number(n)) => n.as_f64(),
        Some(Value::String(s)) => parse_float_prefix(s),
        _ => None,
    }
}

fn parse_float_prefix(s: &str) -> Option<f64> {
    let t = s.trim_start();
    let b = t.as_bytes();
    let mut i = 0;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        i += 1;
    }
    if t[i..].starts_with("Infinity") {
        return Some(if t.starts_with('-') { f64::NEG_INFINITY } else { f64::INFINITY });
    }
    let mut digits = false;
    let mut dot = false;
    let start_digits = i;
    while i < b.len() {
        match b[i] {
            b'0'..=b'9' => {
                digits = true;
                i += 1;
            }
            b'.' if !dot => {
                dot = true;
                i += 1;
            }
            _ => break,
        }
    }
    if !digits {
        return None;
    }
    let mut end = i;
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        let mut j = i + 1;
        if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
            j += 1;
        }
        let exp_start = j;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if j > exp_start {
            end = j;
        }
    }
    let _ = start_digits;
    t[..end].parse::<f64>().ok()
}

/// JS parseInt(v, 10): numeric prefix, truncated toward zero.
fn js_parse_int(v: Option<&Value>) -> Option<i64> {
    match v {
        Some(Value::Number(n)) => n.as_f64().map(|f| f as i64),
        Some(Value::String(s)) => {
            let t = s.trim_start();
            let b = t.as_bytes();
            let mut i = 0;
            if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
                i += 1;
            }
            let start = i;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            if i == start {
                return None;
            }
            t[..i].parse::<i64>().ok().or_else(|| {
                // overflow: saturate like a huge double would
                Some(if t.starts_with('-') { i64::MIN } else { i64::MAX })
            })
        }
        _ => None,
    }
}

/// JS Number(v): full-string conversion ("" → 0, invalid → NaN/None).
fn js_number(v: Option<&Value>) -> Option<f64> {
    match v {
        None | Some(Value::Null) => None, // caller distinguishes absent (NaN) vs null (0)
        Some(Value::Number(n)) => n.as_f64(),
        Some(Value::Bool(b)) => Some(if *b { 1.0 } else { 0.0 }),
        Some(Value::String(s)) => {
            let t = s.trim();
            if t.is_empty() {
                return Some(0.0);
            }
            t.parse::<f64>().ok().or_else(|| parse_float_prefix(t).filter(|_| {
                // Number() is all-or-nothing: prefix parses don't count
                false
            }))
        }
        _ => None,
    }
}

fn field<'a>(node: &'a Node, key: &str) -> Option<&'a Value> {
    node.fields.get(key)
}

/// parseFloat(fields.start) || 0 — NaN → 0 (negatives survive, like JS).
fn field_start(node: &Node) -> f64 {
    match js_parse_float(field(node, "start")) {
        Some(v) if v != 0.0 => v,
        _ => 0.0,
    }
}

// ---------------------------------------------------------------- zlib (DEFLATE)

fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &x in data {
        a = (a + x as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize, // bit position
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        BitReader { data, pos: 0 }
    }
    fn bits(&mut self, n: usize) -> Result<u32> {
        let mut v: u32 = 0;
        for i in 0..n {
            let byte = self.pos / 8;
            if byte >= self.data.len() {
                return Err(err("unexpected end of deflate stream"));
            }
            let bit = (self.data[byte] >> (self.pos % 8)) & 1;
            v |= (bit as u32) << i;
            self.pos += 1;
        }
        Ok(v)
    }
    fn align_byte(&mut self) {
        self.pos = (self.pos + 7) / 8 * 8;
    }
}

/// Canonical Huffman decoder built from code lengths.
struct Huffman {
    first_code: [u32; 16],
    first_symbol: [usize; 16],
    counts: [usize; 16],
    symbols: Vec<u16>,
    max_len: usize,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Huffman {
        let mut tree = Huffman {
            first_code: [0; 16],
            first_symbol: [0; 16],
            counts: [0; 16],
            symbols: Vec::with_capacity(lengths.len()),
            max_len: 0,
        };
        for &len in lengths {
            if len > 0 {
                tree.counts[len as usize] += 1;
                tree.max_len = tree.max_len.max(len as usize);
            }
        }
        for len in 1..=tree.max_len {
            tree.first_code[len] = (tree.first_code[len - 1] + tree.counts[len - 1] as u32) << 1;
            tree.first_symbol[len] = tree.symbols.len();
            for (symbol, &n) in lengths.iter().enumerate() {
                if n as usize == len {
                    tree.symbols.push(symbol as u16);
                }
            }
        }
        tree
    }

    fn decode(&self, br: &mut BitReader) -> Result<u16> {
        let mut code: u32 = 0;
        for len in 1..=self.max_len {
            code = (code << 1) | br.bits(1)?;
            let offset = code.wrapping_sub(self.first_code[len]) as usize;
            if offset < self.counts[len] {
                return Ok(self.symbols[self.first_symbol[len] + offset]);
            }
        }
        Err(err("bad huffman code in deflate stream"))
    }
}

const LENGTH_BASE: [u32; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [usize; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u32; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [usize; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

fn inflate_block(
    br: &mut BitReader,
    out: &mut Vec<u8>,
    lit: &Huffman,
    dist: &Huffman,
    limit: usize,
) -> Result<bool> {
    loop {
        let sym = lit.decode(br)?;
        match sym {
            0..=255 => {
                if out.len() == limit {
                    return Err(err("deflate output exceeds image dimensions"));
                }
                out.push(sym as u8);
            }
            256 => return Ok(true),
            257..=285 => {
                let idx = (sym - 257) as usize;
                if idx >= 29 {
                    return Err(err("bad length symbol in deflate stream"));
                }
                let len = LENGTH_BASE[idx] + br.bits(LENGTH_EXTRA[idx])?;
                let dsym = dist.decode(br)?;
                if dsym as usize >= 30 {
                    return Err(err("bad distance symbol in deflate stream"));
                }
                let d = DIST_BASE[dsym as usize] + br.bits(DIST_EXTRA[dsym as usize])?;
                let d = d as usize;
                if d > out.len() {
                    return Err(err("deflate distance too far back"));
                }
                if len as usize > limit - out.len() {
                    return Err(err("deflate output exceeds image dimensions"));
                }
                let start = out.len() - d;
                for i in 0..len as usize {
                    let v = out[start + i];
                    out.push(v);
                }
            }
            _ => return Err(err("bad literal/length symbol in deflate stream")),
        }
    }
}

fn inflate(data: &[u8], limit: usize) -> Result<Vec<u8>> {
    let mut br = BitReader::new(data);
    let mut out: Vec<u8> = Vec::new();
    loop {
        let bfinal = br.bits(1)?;
        let btype = br.bits(2)?;
        match btype {
            0 => {
                br.align_byte();
                let byte_pos = br.pos / 8;
                if byte_pos + 4 > data.len() {
                    return Err(err("truncated stored deflate block"));
                }
                let len = u16::from_le_bytes([data[byte_pos], data[byte_pos + 1]]) as usize;
                let nlen = u16::from_le_bytes([data[byte_pos + 2], data[byte_pos + 3]]) as usize;
                if len != (!nlen & 0xffff) {
                    return Err(err("stored deflate block length mismatch"));
                }
                let from = byte_pos + 4;
                if from + len > data.len() {
                    return Err(err("truncated stored deflate block"));
                }
                if len > limit - out.len() {
                    return Err(err("deflate output exceeds image dimensions"));
                }
                out.extend_from_slice(&data[from..from + len]);
                br.pos = (from + len) * 8;
            }
            1 => {
                let mut lit_lengths = [8u8; 288];
                lit_lengths[144..256].fill(9);
                lit_lengths[256..280].fill(7);
                let lit = Huffman::new(&lit_lengths);
                let dist = Huffman::new(&[5u8; 30]);
                inflate_block(&mut br, &mut out, &lit, &dist, limit)?;
            }
            2 => {
                let hlit = br.bits(5)? as usize + 257;
                let hdist = br.bits(5)? as usize + 1;
                let hclen = br.bits(4)? as usize + 4;
                const ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
                let mut cl_lengths = [0u8; 19];
                for i in 0..hclen {
                    cl_lengths[ORDER[i]] = br.bits(3)? as u8;
                }
                let cl = Huffman::new(&cl_lengths);
                let total = hlit + hdist;
                let mut lengths = vec![0u8; total];
                let mut i = 0;
                while i < total {
                    let sym = cl.decode(&mut br)?;
                    match sym {
                        0..=15 => {
                            lengths[i] = sym as u8;
                            i += 1;
                        }
                        16 => {
                            if i == 0 {
                                return Err(err("deflate repeat with no previous length"));
                            }
                            let prev = lengths[i - 1];
                            let rep = 3 + br.bits(2)? as usize;
                            for _ in 0..rep {
                                if i >= total {
                                    return Err(err("deflate length overflow"));
                                }
                                lengths[i] = prev;
                                i += 1;
                            }
                        }
                        17 => {
                            let rep = 3 + br.bits(3)? as usize;
                            i += rep;
                            if i > total {
                                return Err(err("deflate length overflow"));
                            }
                        }
                        18 => {
                            let rep = 11 + br.bits(7)? as usize;
                            i += rep;
                            if i > total {
                                return Err(err("deflate length overflow"));
                            }
                        }
                        _ => return Err(err("bad code-length symbol in deflate stream")),
                    }
                }
                if hlit > 288 || hdist > 32 {
                    return Err(err("bad deflate tables"));
                }
                let mut lit_l = [0u8; 288];
                lit_l[..hlit].copy_from_slice(&lengths[..hlit]);
                let mut dist_l = [0u8; 32];
                dist_l[..hdist].copy_from_slice(&lengths[hlit..]);
                let lit = Huffman::new(&lit_l);
                let dist = Huffman::new(&dist_l);
                inflate_block(&mut br, &mut out, &lit, &dist, limit)?;
            }
            _ => return Err(err("bad deflate block type")),
        }
        if bfinal == 1 {
            break;
        }
    }
    Ok(out)
}

/// Inflate at most the number of bytes declared by the image header.
fn zlib_inflate(data: &[u8], limit: usize) -> Result<Vec<u8>> {
    if data.len() < 2 {
        return Err(err("bad zlib stream"));
    }
    let out = inflate(&data[2..], limit)?;
    if data.len() >= 6 {
        let stored = u32::from_be_bytes([
            data[data.len() - 4],
            data[data.len() - 3],
            data[data.len() - 2],
            data[data.len() - 1],
        ]);
        if stored != adler32(&out) {
            return Err(err("bad zlib checksum"));
        }
    }
    Ok(out)
}

struct BitWriter {
    bytes: Vec<u8>,
    cur: u32,
    nbits: usize,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter { bytes: Vec::new(), cur: 0, nbits: 0 }
    }
    fn bits(&mut self, v: u32, n: usize) {
        self.cur |= v << self.nbits;
        self.nbits += n;
        while self.nbits >= 8 {
            self.bytes.push((self.cur & 0xff) as u8);
            self.cur >>= 8;
            self.nbits -= 8;
        }
    }
    /// Huffman codes are emitted MSB-first: reverse the code's bits.
    fn code(&mut self, code: u32, n: usize) {
        let mut rev: u32 = 0;
        for i in 0..n {
            rev = (rev << 1) | ((code >> i) & 1);
        }
        self.bits(rev, n);
    }
    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.bytes.push((self.cur & 0xff) as u8);
        }
        self.bytes
    }
}

fn fixed_lit_code(sym: u16) -> (u32, usize) {
    match sym {
        0..=143 => (0x30 + sym as u32, 8),
        144..=255 => (0x190 + (sym as u32 - 144), 9),
        256..=279 => ((sym - 256) as u32, 7),
        _ => (0xC0 + (sym as u32 - 280), 8),
    }
}

/// LZ77 + fixed-Huffman deflate. The match policy mirrors what node:zlib
/// level 9 emits on the tiny flat-color scanlines this encoder feeds the
/// suite (a length-3 match only when it reaches end of input); elsewhere the
/// output is a valid — not necessarily byte-identical — deflate stream.
fn deflate_fixed(data: &[u8]) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.bits(1, 1); // BFINAL
    w.bits(1, 2); // BTYPE = fixed
    // RFC 1951 limits distances to 32 KiB. Keep only the hash chains that
    // can still be referenced, even when encoding a many-megabyte atlas.
    const WINDOW: usize = 32_768;
    const END: usize = usize::MAX;
    let buckets = data.len().clamp(64, 65_536).next_power_of_two();
    let mut heads = vec![END; buckets];
    let mut previous = vec![END; data.len().min(WINDOW)];
    let hash = |pos: usize| -> usize {
        let key = u32::from_be_bytes([0, data[pos], data[pos + 1], data[pos + 2]]);
        (key.wrapping_mul(0x9e37_79b1) >> (32 - buckets.trailing_zeros())) as usize
    };
    let mut i = 0usize;
    while i < data.len() {
        let mut best_len = 0usize;
        let mut best_dist = 0usize;
        if i + 2 < data.len() {
            let max_len = (data.len() - i).min(258);
            let mut candidate = heads[hash(i)];
            for _ in 0..32 {
                if candidate == END || i - candidate > WINDOW { break; }
                let mut length = 0;
                while length < max_len && data[candidate + length] == data[i + length] {
                    length += 1;
                }
                if length > best_len {
                    best_len = length;
                    best_dist = i - candidate;
                    if length == max_len { break; }
                }
                candidate = previous[candidate & (WINDOW - 1)];
            }
        }
        let start = i;
        if best_len >= 4 || (best_len == 3 && i + best_len == data.len()) {
            let len = best_len;
            let mut sym_idx = 28usize;
            for (idx, &base) in LENGTH_BASE.iter().enumerate() {
                let next = if idx + 1 < 29 { LENGTH_BASE[idx + 1] } else { 259 };
                if (len as u32) >= base && (len as u32) < next {
                    sym_idx = idx;
                    break;
                }
            }
            let sym = 257 + sym_idx as u16;
            let (code, n) = fixed_lit_code(sym);
            w.code(code, n);
            let extra = LENGTH_EXTRA[sym_idx];
            if extra > 0 {
                w.bits((len as u32 - LENGTH_BASE[sym_idx]) as u32, extra);
            }
            let mut didx = 29usize;
            for (idx, &base) in DIST_BASE.iter().enumerate() {
                let next = if idx + 1 < 30 { DIST_BASE[idx + 1] } else { 32769 };
                if (best_dist as u32) >= base && (best_dist as u32) < next {
                    didx = idx;
                    break;
                }
            }
            w.code(didx as u32, 5);
            let dextra = DIST_EXTRA[didx];
            if dextra > 0 {
                w.bits((best_dist as u32 - DIST_BASE[didx]) as u32, dextra);
            }
            i += len;
        } else {
            let (code, n) = fixed_lit_code(data[i] as u16);
            w.code(code, n);
            i += 1;
        }
        for pos in start..i.min(data.len().saturating_sub(2)) {
            let bucket = hash(pos);
            previous[pos & (WINDOW - 1)] = heads[bucket];
            heads[bucket] = pos;
        }
    }
    let (code, n) = fixed_lit_code(256);
    w.code(code, n);
    w.finish()
}

/// node:zlib deflateSync(level 9) equivalent: header 78 DA, deflate, adler32.
/// Picks stored blocks when the fixed-Huffman encoding wouldn't win.
fn zlib_deflate(data: &[u8]) -> Vec<u8> {
    let fixed = deflate_fixed(data);
    let stored_len = data.len() + 5 * ((data.len() + 65534) / 65535).max(1);
    let mut out = vec![0x78, 0xDA];
    if data.is_empty() || fixed.len() <= stored_len {
        out.extend_from_slice(&fixed);
    } else {
        let mut chunks: Vec<&[u8]> = data.chunks(65535).collect();
        if chunks.is_empty() {
            chunks.push(&[]);
        }
        let last = chunks.len() - 1;
        for (ci, chunk) in chunks.iter().enumerate() {
            out.push(if ci == last { 1 } else { 0 });
            let len = chunk.len() as u16;
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(&(!len).to_le_bytes());
            out.extend_from_slice(chunk);
        }
    }
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

#[cfg(test)]
#[test]
fn deflate_match_outside_32k_window_reencodes_as_literals() {
    let marker = b"sprite-atlas-row";
    let mut pixels = marker.to_vec();
    pixels.resize(65_536, 0);
    pixels.extend_from_slice(marker);
    assert_eq!(zlib_inflate(&zlib_deflate(&pixels), pixels.len()).unwrap(), pixels);
}

fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for n in 0..256u32 {
        let mut c = n;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB88320 ^ (c >> 1) } else { c >> 1 };
        }
        table[n as usize] = c;
    }
    let mut c: u32 = 0xFFFFFFFF;
    for &b in data {
        c = table[((c ^ b as u32) & 0xff) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFFFFFF
}

// ---------------------------------------------------------------- PNG codec

struct PngImage {
    width: u32,
    height: u32,
    rgba: Vec<u8>, // RGBA, width*height*4
}

/// Local media operations retain their existing per-side image limit.
fn decode_png(bytes: &[u8]) -> Result<PngImage> {
    sprite_codecs::read_png(bytes, false)
}

fn png_chunk(out: &mut Vec<u8>, ty: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(ty);
    out.extend_from_slice(data);
    let mut crc_data = Vec::with_capacity(4 + data.len());
    crc_data.extend_from_slice(ty);
    crc_data.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_data).to_be_bytes());
}

/// encodePngRgba mirror: filter-0 rows, RGBA color type 6, zlib level 9.
fn encode_png_rgba(img: &PngImage) -> Vec<u8> {
    let stride = img.width as usize * 4;
    let mut raw = Vec::with_capacity((stride + 1) * img.height as usize);
    for y in 0..img.height as usize {
        raw.push(0);
        raw.extend_from_slice(&img.rgba[y * stride..(y + 1) * stride]);
    }
    let mut out = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&img.width.to_be_bytes());
    ihdr.extend_from_slice(&img.height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
    png_chunk(&mut out, b"IHDR", &ihdr);
    png_chunk(&mut out, b"IDAT", &zlib_deflate(&raw));
    png_chunk(&mut out, b"IEND", &[]);
    out
}

/// sampleBilinear mirror (x/y clamped by callers into [0, dim-1]).
fn sample_bilinear(img: &PngImage, x: f64, y: f64) -> [u8; 4] {
    let (sw, sh) = (img.width as i64, img.height as i64);
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;
    let cx = |v: i64, m: i64| v.max(0).min(m - 1);
    let px = |xx: i64, yy: i64, ch: usize| -> f64 {
        img.rgba[((cx(yy, sh) * sw + cx(xx, sw)) * 4 + ch as i64) as usize] as f64
    };
    let mut out = [0u8; 4];
    for ch in 0..4 {
        let v = px(x0, y0, ch) * (1.0 - fx) * (1.0 - fy)
            + px(x0 + 1, y0, ch) * fx * (1.0 - fy)
            + px(x0, y0 + 1, ch) * (1.0 - fx) * fy
            + px(x0 + 1, y0 + 1, ch) * fx * fy;
        out[ch] = js_round(v) as u8;
    }
    out
}

struct ResizePlan {
    cw: u32,
    ch: u32,
    dx: f64,
    dy: f64,
    dw: f64,
    dh: f64,
}

/// resizePlan mirror.
fn resize_plan(sw: u32, sh: u32, mode: &str, tw: i64, th: i64) -> ResizePlan {
    let (swf, shf) = (sw as f64, sh as f64);
    let mut bw = tw;
    let mut bh = th;
    if mode == "fit" {
        let mut scale = f64::INFINITY;
        if tw > 0 {
            scale = scale.min(tw as f64 / swf);
        }
        if th > 0 {
            scale = scale.min(th as f64 / shf);
        }
        if !scale.is_finite() || scale <= 0.0 {
            scale = 1.0;
        }
        if scale > 1.0 {
            scale = 1.0;
        }
        let w = (js_round(swf * scale) as i64).max(1);
        let h = (js_round(shf * scale) as i64).max(1);
        return ResizePlan { cw: w as u32, ch: h as u32, dx: 0.0, dy: 0.0, dw: w as f64, dh: h as f64 };
    }
    if mode == "fill" {
        if bw <= 0 && bh <= 0 {
            bw = sw as i64;
            bh = sh as i64;
        } else if bw <= 0 {
            bw = js_round(bh as f64 * swf / shf) as i64;
        } else if bh <= 0 {
            bh = js_round(bw as f64 * shf / swf) as i64;
        }
        let scale = (bw as f64 / swf).max(bh as f64 / shf);
        let dw = swf * scale;
        let dh = shf * scale;
        return ResizePlan {
            cw: bw.max(1) as u32,
            ch: bh.max(1) as u32,
            dx: (bw as f64 - dw) / 2.0,
            dy: (bh as f64 - dh) / 2.0,
            dw,
            dh,
        };
    }
    // exact
    if bw <= 0 && bh <= 0 {
        bw = sw as i64;
        bh = sh as i64;
    } else if bw <= 0 {
        bw = js_round(bh as f64 * swf / shf) as i64;
    } else if bh <= 0 {
        bh = js_round(bw as f64 * shf / swf) as i64;
    }
    ResizePlan { cw: bw.max(1) as u32, ch: bh.max(1) as u32, dx: 0.0, dy: 0.0, dw: bw as f64, dh: bh as f64 }
}

fn resize_png_pure(src: &PngImage, plan: &ResizePlan) -> PngImage {
    let (sw, sh) = (src.width as f64, src.height as f64);
    let mut out = PngImage {
        width: plan.cw,
        height: plan.ch,
        rgba: vec![0u8; plan.cw as usize * plan.ch as usize * 4],
    };
    for y in 0..plan.ch as usize {
        for x in 0..plan.cw as usize {
            let u = (x as f64 + 0.5 - plan.dx) / plan.dw * sw - 0.5;
            let v = (y as f64 + 0.5 - plan.dy) / plan.dh * sh - 0.5;
            if u < -0.5 || v < -0.5 || u > sw - 0.5 || v > sh - 0.5 {
                continue;
            }
            let sx = u.max(0.0).min(sw - 1.0);
            let sy = v.max(0.0).min(sh - 1.0);
            let pix = sample_bilinear(src, sx, sy);
            let o = (y * plan.cw as usize + x) * 4;
            out.rgba[o..o + 4].copy_from_slice(&pix);
        }
    }
    out
}

/// probeImageSizeFfmpeg: [width, height] via ffprobe csv.
fn probe_image_size(path: &PathBuf) -> Result<(u32, u32)> {
    let out = run_proc(
        "ffprobe",
        &[
            "-v".into(),
            "error".into(),
            "-select_streams".into(),
            "v:0".into(),
            "-show_entries".into(),
            "stream=width,height".into(),
            "-of".into(),
            "csv=p=0".into(),
            path.to_string_lossy().into_owned(),
        ],
    )?;
    let first = out.trim().lines().next().unwrap_or("");
    let mut it = first.split(',');
    let w: u32 = it.next().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
    let h: u32 = it.next().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
    if w == 0 || h == 0 {
        return Err(err("couldn't read that image"));
    }
    Ok((w, h))
}

/// resizeCropImage mirror: pure PNG first for PNG sources, ffmpeg otherwise.
fn resize_crop_image(url: &str, mode: &str, tw_in: i64, th_in: i64) -> Result<String> {
    let tw = tw_in.max(0);
    let th = th_in.max(0);
    if tw == 0 && th == 0 {
        return Err(err("set a width or height to resize to"));
    }
    let bytes = url_bytes(url)?;
    if sniff_mime(&bytes) == "image/png" {
        match decode_png(&bytes) {
            Ok(src) => {
                let plan = resize_plan(src.width, src.height, mode, tw, th);
                let out = encode_png_rgba(&resize_png_pure(&src, &plan));
                let url_out = bytes_to_data_url(&out, Some("image/png"));
                if url_out.len() > MEDIA_INLINE_MAX {
                    return Err(err(format!(
                        "resized image is too large to pass inline ({} bytes > ~4 MB limit) — pick smaller dimensions",
                        url_out.len()
                    )));
                }
                return Ok(url_out);
            }
            Err(e) => {
                let msg = e.0.clone();
                if msg.contains("width or height") || msg.contains("inline limit") || msg.contains("too large") {
                    return Err(e);
                }
                if !msg.contains("couldn't read that image") {
                    return Err(e);
                }
                // fall through to ffmpeg
            }
        }
    }
    // ffmpeg path
    let dir = TempDir::new()?;
    let in_path = write_input(&dir, "in", url)?;
    let (sw, sh) = probe_image_size(&in_path)?;
    let plan = resize_plan(sw, sh, mode, tw, th);
    let want_png = url.to_ascii_lowercase().starts_with("data:image/png") || in_path.to_string_lossy().ends_with(".png");
    let vf = if mode == "fill" {
        format!(
            "scale={}:{}:force_original_aspect_ratio=increase,crop={}:{}",
            plan.cw, plan.ch, plan.cw, plan.ch
        )
    } else {
        format!("scale={}:{}", plan.cw, plan.ch)
    };
    let out_path = dir.path(if want_png { "out.png" } else { "out.jpg" });
    let mut args: Vec<String> = vec![
        "-y".into(),
        "-i".into(),
        in_path.to_string_lossy().into_owned(),
        "-vf".into(),
        vf,
        "-frames:v".into(),
        "1".into(),
    ];
    if !want_png {
        args.push("-q:v".into());
        args.push("2".into());
    }
    args.push(out_path.to_string_lossy().into_owned());
    run_proc("ffmpeg", &args)?;
    let out_url = data_url_from_file(&out_path, None)?;
    if out_url.len() > MEDIA_INLINE_MAX {
        return Err(err(format!(
            "resized image is too large to pass inline ({} bytes > ~4 MB limit) — pick smaller dimensions",
            out_url.len()
        )));
    }
    Ok(out_url)
}

/// maskToSource pure path: composite the mask over opaque black at source size.
fn mask_to_source_png_pure(src_bytes: &[u8], mask_bytes: &[u8]) -> Result<String> {
    let src = decode_png(src_bytes).map_err(|_| err("couldn't read the source image"))?;
    let mask = decode_png(mask_bytes).map_err(|_| err("couldn't read the mask"))?;
    let (sw, sh) = (src.width, src.height);
    let (mw, mh) = (mask.width as f64, mask.height as f64);
    let mut out = PngImage { width: sw, height: sh, rgba: vec![0u8; sw as usize * sh as usize * 4] };
    for i in 0..(sw as usize * sh as usize) {
        out.rgba[i * 4 + 3] = 255;
    }
    for y in 0..sh as usize {
        for x in 0..sw as usize {
            let u = (x as f64 + 0.5) / sw as f64 * mw - 0.5;
            let v = (y as f64 + 0.5) / sh as f64 * mh - 0.5;
            let sx = u.max(0.0).min(mw - 1.0);
            let sy = v.max(0.0).min(mh - 1.0);
            let pix = sample_bilinear(&mask, sx, sy);
            let a = pix[3] as f64 / 255.0;
            let o = (y * sw as usize + x) * 4;
            for ch in 0..3 {
                out.rgba[o + ch] = js_round(pix[ch] as f64 * a + out.rgba[o + ch] as f64 * (1.0 - a)) as u8;
            }
            out.rgba[o + 3] = 255;
        }
    }
    Ok(bytes_to_data_url(&encode_png_rgba(&out), Some("image/png")))
}

/// maskToSource mirror: pure PNG composite when both sides are PNG, else the
/// ffmpeg black-canvas + overlay composite.
fn mask_to_source(mask_url: &str, source_url: &str) -> Result<String> {
    if mask_url.is_empty() {
        return Err(err("couldn't read the mask"));
    }
    if source_url.is_empty() {
        return Err(err("couldn't read the source image"));
    }
    let src_bytes = url_bytes(source_url)?;
    let mask_bytes = url_bytes(mask_url)?;
    if sniff_mime(&src_bytes) == "image/png" && sniff_mime(&mask_bytes) == "image/png" {
        match mask_to_source_png_pure(&src_bytes, &mask_bytes) {
            Ok(u) => return Ok(u),
            Err(e) => {
                if e.0.contains("couldn't read the mask")
                    || e.0.contains("couldn't read the source")
                    || e.0.contains("too large")
                {
                    return Err(e);
                }
                // fall through to ffmpeg
            }
        }
    }
    let dir = TempDir::new()?;
    let src_path = write_input(&dir, "src", source_url)?;
    let (sw, sh) = probe_image_size(&src_path).map_err(|_| err("couldn't read the source image"))?;
    let mask_path = write_input(&dir, "mask", mask_url)?;
    let out_path = dir.path("mask.png");
    run_proc(
        "ffmpeg",
        &[
            "-y".into(),
            "-f".into(),
            "lavfi".into(),
            "-i".into(),
            format!("color=c=black:s={}x{}:d=1", sw, sh),
            "-i".into(),
            mask_path.to_string_lossy().into_owned(),
            "-filter_complex".into(),
            format!(
                "[1:v]scale={}:{}:flags=bilinear,format=rgba[m];[0:v][m]overlay=0:0:format=auto",
                sw, sh
            ),
            "-frames:v".into(),
            "1".into(),
            out_path.to_string_lossy().into_owned(),
        ],
    )
    .map_err(|_| err("couldn't read the mask"))?;
    data_url_from_file(&out_path, Some("image/png"))
}

/// fitImage mirror: shrink an oversized inline image through the fit ladder.
fn fit_image_inline(url: &str) -> Result<String> {
    if !url.to_ascii_lowercase().starts_with("data:") || (url.len() as f64) <= INLINE_IMAGE_BUDGET {
        return Ok(url.to_string());
    }
    for dim in [2048i64, 1448, 1024, 724, 512] {
        match resize_crop_image(url, "fit", dim, dim) {
            Ok(out) => {
                if (out.len() as f64) <= INLINE_IMAGE_BUDGET {
                    return Ok(out);
                }
            }
            Err(e) => {
                if e.0.contains("inline limit") {
                    continue;
                }
                return Err(err(format!(
                    "image is too large to send inline (~4 MB max) and couldn't be resized down ({}) — use a smaller image",
                    e.0
                )));
            }
        }
    }
    Err(err("image is too large to send inline even after resizing (~4 MB max) — use a smaller image"))
}

// ---------------------------------------------------------------- WAV codec (trim pure path)

struct PcmWav {
    sample_rate: u32,
    mono: Vec<f32>,
}

/// parsePcmWav mirror: PCM (fmt 1) / float (fmt 3) WAV → mono f32 samples.
fn parse_pcm_wav(bytes: &[u8]) -> Result<PcmWav> {
    let bad = || err("unsupported WAV format for in-process trim");
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(bad());
    }
    let u16le = |b: &[u8], o: usize| -> u16 { u16::from_le_bytes([b[o], b[o + 1]]) };
    let u32le = |b: &[u8], o: usize| -> u32 { u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) };
    let i16le = |b: &[u8], o: usize| -> i16 { i16::from_le_bytes([b[o], b[o + 1]]) };
    let i32le = |b: &[u8], o: usize| -> i32 { i32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) };
    let f32le = |b: &[u8], o: usize| -> f32 { f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) };
    let (mut format, mut channels, mut sample_rate, mut bits) = (0u16, 0u16, 0u32, 0i32);
    let mut data: Option<(usize, usize)> = None;
    let mut pos = 12usize;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32le(bytes, pos + 4) as usize;
        let body = pos + 8;
        if body + size > bytes.len() + 8 {
            break;
        }
        let avail = size.min(bytes.len().saturating_sub(body));
        if id == b"fmt " && avail >= 16 {
            format = u16le(bytes, body);
            channels = u16le(bytes, body + 2).max(1);
            sample_rate = u32le(bytes, body + 4);
            bits = u16le(bytes, body + 14) as i32;
            if format == 3 {
                bits = -32;
            }
        } else if id == b"data" {
            data = Some((body, avail));
            break;
        }
        pos = body + size + (size & 1);
    }
    let (data_off, data_len) = data.ok_or_else(|| bad())?;
    if (format != 1 && format != 3) || sample_rate == 0 || channels == 0 {
        return Err(bad());
    }
    let ch = channels as usize;
    let mut mono: Vec<f32> = Vec::new();
    let push_frames = |mono: &mut Vec<f32>, frame_count: usize, get: &dyn Fn(usize) -> f32| {
        mono.reserve(frame_count);
        for f in 0..frame_count {
            let mut sum = 0.0f64;
            for c in 0..ch {
                sum += get(f * ch + c) as f64;
            }
            mono.push((sum / ch as f64) as f32);
        }
    };
    match bits {
        16 => {
            let n = data_len / 2;
            if n > MAX_WAV_SAMPLES {
                return Err(err("audio is too long to trim in-process — use a shorter clip"));
            }
            push_frames(&mut mono, n / ch, &|i| i16le(bytes, data_off + i * 2) as f32 / 0x8000 as f32);
        }
        8 => {
            let n = data_len;
            if n > MAX_WAV_SAMPLES {
                return Err(err("audio is too long to trim in-process — use a shorter clip"));
            }
            push_frames(&mut mono, n / ch, &|i| (bytes[data_off + i] as f32 - 128.0) / 128.0);
        }
        -32 => {
            let n = data_len / 4;
            if n > MAX_WAV_SAMPLES {
                return Err(err("audio is too long to trim in-process — use a shorter clip"));
            }
            push_frames(&mut mono, n / ch, &|i| f32le(bytes, data_off + i * 4));
        }
        32 => {
            let n = data_len / 4;
            if n > MAX_WAV_SAMPLES {
                return Err(err("audio is too long to trim in-process — use a shorter clip"));
            }
            push_frames(&mut mono, n / ch, &|i| i32le(bytes, data_off + i * 4) as f32 / 2147483648.0);
        }
        _ => return Err(bad()),
    }
    Ok(PcmWav { sample_rate, mono })
}

/// encodeWavMono mirror: 16-bit PCM mono WAV.
fn encode_wav_mono(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_len = samples.len() * 2;
    let mut out = Vec::with_capacity(44 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    for &v in samples {
        let v = (v as f64).clamp(-1.0, 1.0);
        let int = if v < 0.0 { v * 0x8000 as f64 } else { v * 0x7fff as f64 };
        out.extend_from_slice(&(int as i16).to_le_bytes());
    }
    out
}

/// resampleMono mirror: linear interpolation.
fn resample_mono(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = from as f64 / to as f64;
    let n = ((samples.len() as f64 / ratio).floor() as usize).max(1);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let x = i as f64 * ratio;
        let i0 = x.floor() as usize;
        let i1 = (i0 + 1).min(samples.len() - 1);
        let f = x - i0 as f64;
        out.push((samples[i0] as f64 * (1.0 - f) + samples[i1] as f64 * f) as f32);
    }
    out
}

fn past_end_error(start: f64, dur: f64) -> NanoodleError {
    err(format!(
        "the start point ({}s) is past the end of this clip, which is only {:.1}s long — pick an earlier start",
        js_num(js_round(start * 10.0) / 10.0),
        dur
    ))
}

/// trimPcmWavPure mirror.
fn trim_pcm_wav_pure(bytes: &[u8], start: f64, length: f64, target_rate: u32, whole_if_blank: bool) -> Result<Vec<u8>> {
    let wav = parse_pcm_wav(bytes)?;
    let dur = wav.mono.len() as f64 / wav.sample_rate as f64;
    let s0 = if start == 0.0 || start.is_nan() { 0.0 } else { start };
    if s0 >= dur {
        return Err(past_end_error(s0, dur));
    }
    let s = s0.max(0.0).min((dur - 0.05).max(0.0));
    let take = if whole_if_blank && !(length > 0.0) {
        (dur - s).max(0.05)
    } else {
        let l = if length.is_finite() && length > 0.0 { length } else { 30.0 };
        (l.min(dur - s)).max(0.05)
    };
    let i0 = (s * wav.sample_rate as f64).floor() as usize;
    let n = ((take * wav.sample_rate as f64).floor() as usize).max(1);
    let end = (i0 + n).min(wav.mono.len());
    let sliced = &wav.mono[i0.min(wav.mono.len())..end];
    let resampled = resample_mono(sliced, wav.sample_rate, target_rate);
    Ok(encode_wav_mono(&resampled, target_rate))
}

/// probeDurationFfmpeg: seconds or None (ffprobe failure → None).
fn probe_duration(path: &PathBuf) -> Option<f64> {
    let out = run_proc(
        "ffprobe",
        &[
            "-v".into(),
            "error".into(),
            "-show_entries".into(),
            "format=duration".into(),
            "-of".into(),
            "csv=p=0".into(),
            path.to_string_lossy().into_owned(),
        ],
    )
    .ok()?;
    let d: f64 = out.trim().parse().ok()?;
    if d.is_finite() && d > 0.0 {
        Some(d)
    } else {
        None
    }
}

/// trimAudioToWavFfmpeg mirror.
fn trim_audio_to_wav_ffmpeg(
    url: &str,
    start: f64,
    length: f64,
    target_rate: u32,
    whole_if_blank: bool,
) -> Result<String> {
    let dir = TempDir::new()?;
    let in_path = write_input(&dir, "in", url)?;
    let dur = probe_duration(&in_path);
    let s = if start == 0.0 || start.is_nan() { 0.0 } else { start };
    if let Some(d) = dur {
        if s >= d {
            return Err(past_end_error(s, d));
        }
    }
    let take = if whole_if_blank && !(length > 0.0) {
        dur.map(|d| (d - s).max(0.05))
    } else {
        let l = if length.is_finite() && length > 0.0 { length } else { 30.0 };
        Some(match dur {
            Some(d) => l.min(d - s).max(0.05),
            None => l,
        })
    };
    let out_path = dir.path("out.wav");
    let mut args: Vec<String> = vec![
        "-y".into(),
        "-ss".into(),
        js_num(s),
        "-i".into(),
        in_path.to_string_lossy().into_owned(),
    ];
    if let Some(t) = take {
        args.push("-t".into());
        args.push(js_num(t));
    }
    args.extend([
        "-vn".into(),
        "-ac".into(),
        "1".into(),
        "-ar".into(),
        target_rate.to_string(),
        "-f".into(),
        "wav".into(),
        out_path.to_string_lossy().into_owned(),
    ]);
    match run_proc("ffmpeg", &args) {
        Ok(_) => data_url_from_file(&out_path, Some("audio/wav")),
        Err(e) => {
            let msg = e.0.to_lowercase();
            if msg.contains("does not contain any stream") || msg.contains("output file does not contain") || msg.contains("no audio") {
                return Err(err("that input has no audio to extract — for a video, wire one that has sound"));
            }
            if msg.contains("invalid data") || msg.contains("could not find codec") {
                return Err(err("couldn't decode that media — try an mp3, wav, or an mp4 with audio"));
            }
            Err(e)
        }
    }
}

/// trimAudioToWav mirror: pure PCM-WAV path first for WAV sources.
fn trim_audio_to_wav(url: &str, start: f64, length: f64, target_rate: u32, whole_if_blank: bool) -> Result<String> {
    let bytes = url_bytes(url)?;
    if sniff_mime(&bytes) == "audio/wav" {
        match trim_pcm_wav_pure(&bytes, start, length, target_rate, whole_if_blank) {
            Ok(out) => return Ok(bytes_to_data_url(&out, Some("audio/wav"))),
            Err(e) => {
                if e.0.contains("past the end") || e.0.contains("too long") {
                    return Err(e);
                }
                if !e.0.contains("unsupported format") {
                    return Err(e);
                }
                // fall through to ffmpeg
            }
        }
    }
    trim_audio_to_wav_ffmpeg(url, start, length, target_rate, whole_if_blank)
}

// ---------------------------------------------------------------- vframes

/// extractVideoFrames mirror.
fn extract_video_frames(url: &str, count_in: i64, gap_in: Option<f64>, dir_field: &str) -> Result<Vec<String>> {
    let count = if count_in == 0 { 1 } else { count_in }.clamp(1, MAX_FRAMES);
    let gap = match gap_in {
        Some(g) if g.is_finite() => g.max(0.0),
        _ => 0.5,
    };
    let dir = TempDir::new()?;
    let in_path = write_input(&dir, "in", url)?;
    let probe = run_proc(
        "ffprobe",
        &[
            "-v".into(),
            "error".into(),
            "-select_streams".into(),
            "v:0".into(),
            "-show_entries".into(),
            "stream=duration,avg_frame_rate".into(),
            "-show_entries".into(),
            "format=duration".into(),
            "-of".into(),
            "json".into(),
            in_path.to_string_lossy().into_owned(),
        ],
    )?;
    let pj: Value = serde_json::from_str(&probe).unwrap_or(json!({}));
    let stream = pj.get("streams").and_then(|s| s.get(0)).cloned().unwrap_or(json!({}));
    let parse_dur = |v: Option<&Value>| -> Option<f64> {
        match v {
            Some(Value::Number(n)) => n.as_f64(),
            Some(Value::String(s)) => s.parse().ok(),
            _ => None,
        }
        .filter(|d| d.is_finite() && *d > 0.0)
    };
    let vdur = parse_dur(stream.get("duration"));
    let fdur = parse_dur(pj.get("format").and_then(|f| f.get("duration")));
    let dur = [vdur, fdur].into_iter().flatten().fold(f64::INFINITY, f64::min);
    if !dur.is_finite() {
        return Err(err("video has no readable duration"));
    }
    let mut fps = 0.0f64;
    if let Some(fr) = stream.get("avg_frame_rate").and_then(|v| v.as_str()) {
        let mut it = fr.split('/');
        let num: f64 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
        let den: f64 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
        if den > 0.0 {
            fps = num / den;
        }
    }
    let eps = if fps > 0.0 { (1.5 / fps).max(0.04) } else { 0.1 };
    let from_end = dir_field == "end";
    let grab = |t: f64, out_path: &PathBuf| -> Result<()> {
        run_proc(
            "ffmpeg",
            &[
                "-y".into(),
                "-ss".into(),
                js_num(t),
                "-i".into(),
                in_path.to_string_lossy().into_owned(),
                "-frames:v".into(),
                "1".into(),
                "-q:v".into(),
                "2".into(),
                out_path.to_string_lossy().into_owned(),
            ],
        )?;
        Ok(())
    };
    let mut out = Vec::new();
    for i in 0..count {
        let mut t = if from_end { dur - eps - i as f64 * gap } else { i as f64 * gap };
        t = t.max(0.0).min((dur - eps).max(0.0));
        let frame_path = dir.path(&format!("f{}.jpg", i + 1));
        match grab(t, &frame_path) {
            Ok(()) => {}
            Err(e) => {
                if t <= 0.0 {
                    return Err(e);
                }
                grab((t - 0.5).max(0.0), &frame_path)?;
            }
        }
        out.push(data_url_from_file(&frame_path, Some("image/jpeg"))?);
    }
    Ok(out)
}

// ---------------------------------------------------------------- soundtrack

/// muxSoundtrack mirror: stream-copy video, AAC audio, -shortest (or -t when
/// looping to a known video duration); one re-encode retry.
fn mux_soundtrack(video_url: &str, audio_url: &str, loop_audio: bool) -> Result<String> {
    let dir = TempDir::new()?;
    let v_path = write_input(&dir, "v", video_url)?;
    let a_path = write_input(&dir, "a", audio_url)?;
    let vdur = probe_duration(&v_path);
    let out_path = dir.path("out.mp4");
    let build_args = |reencode: bool| -> Vec<String> {
        let mut args: Vec<String> = vec!["-y".into(), "-i".into(), v_path.to_string_lossy().into_owned()];
        if loop_audio {
            args.push("-stream_loop".into());
            args.push("-1".into());
        }
        args.push("-i".into());
        args.push(a_path.to_string_lossy().into_owned());
        args.push("-map".into());
        args.push("0:v:0".into());
        args.push("-map".into());
        args.push("1:a:0".into());
        if reencode {
            args.extend(["-c:v".into(), "libx264".into(), "-preset".into(), "veryfast".into(), "-crf".into(), "18".into()]);
        } else {
            args.extend(["-c:v".into(), "copy".into()]);
        }
        args.extend(["-c:a".into(), "aac".into(), "-b:a".into(), "128k".into()]);
        if loop_audio {
            if let Some(d) = vdur {
                args.push("-t".into());
                args.push(js_num(d));
            }
        } else {
            args.push("-shortest".into());
        }
        args.extend(["-movflags".into(), "+faststart".into(), out_path.to_string_lossy().into_owned()]);
        args
    };
    match run_proc("ffmpeg", &build_args(false)) {
        Ok(_) => data_url_from_file(&out_path, Some("video/mp4")),
        Err(e) => {
            if is_missing_ffmpeg(&e) {
                return Err(e);
            }
            run_proc("ffmpeg", &build_args(true))?;
            data_url_from_file(&out_path, Some("video/mp4"))
        }
    }
}

// ---------------------------------------------------------------- combine (ffmpeg fallback)

/// concatVideosFfmpeg mirror: optional first-frame dedup trims (re-encode),
/// concat demuxer, -c copy with a re-encode retry.
fn concat_videos_ffmpeg(urls: &[String], dedup: bool) -> Result<String> {
    let dir = TempDir::new()?;
    let mut parts: Vec<PathBuf> = Vec::new();
    for (i, u) in urls.iter().enumerate() {
        parts.push(write_input(&dir, &format!("c{}", i), u)?);
    }
    if dedup && parts.len() > 1 {
        for i in 1..parts.len() {
            let trimmed = dir.path(&format!("t{}.mp4", i));
            run_proc(
                "ffmpeg",
                &[
                    "-y".into(),
                    "-ss".into(),
                    "0.033".into(),
                    "-i".into(),
                    parts[i].to_string_lossy().into_owned(),
                    "-c:v".into(),
                    "libx264".into(),
                    "-preset".into(),
                    "ultrafast".into(),
                    "-crf".into(),
                    "18".into(),
                    "-c:a".into(),
                    "aac".into(),
                    "-b:a".into(),
                    "128k".into(),
                    "-movflags".into(),
                    "+faststart".into(),
                    trimmed.to_string_lossy().into_owned(),
                ],
            )?;
            parts[i] = trimmed;
        }
    }
    let list_path = dir.path("list.txt");
    let list: String = parts
        .iter()
        .map(|p| format!("file '{}'\n", p.to_string_lossy().replace('\'', "'\\''")))
        .collect();
    std::fs::write(&list_path, list).map_err(|e| err(format!("couldn't write temp file: {}", e)))?;
    let out_path = dir.path("out.mp4");
    let build_args = |reencode: bool| -> Vec<String> {
        let mut args: Vec<String> = vec![
            "-y".into(),
            "-f".into(),
            "concat".into(),
            "-safe".into(),
            "0".into(),
            "-i".into(),
            list_path.to_string_lossy().into_owned(),
        ];
        if reencode {
            args.extend([
                "-c:v".into(),
                "libx264".into(),
                "-preset".into(),
                "ultrafast".into(),
                "-crf".into(),
                "18".into(),
                "-c:a".into(),
                "aac".into(),
                "-b:a".into(),
                "128k".into(),
            ]);
        } else {
            args.extend(["-c".into(), "copy".into()]);
        }
        args.extend(["-movflags".into(), "+faststart".into(), out_path.to_string_lossy().into_owned()]);
        args
    };
    match run_proc("ffmpeg", &build_args(false)) {
        Ok(_) => data_url_from_file(&out_path, Some("video/mp4")),
        Err(e) => {
            if is_missing_ffmpeg(&e) {
                return Err(e);
            }
            run_proc("ffmpeg", &build_args(true))?;
            data_url_from_file(&out_path, Some("video/mp4"))
        }
    }
}

/// concatVideos mirror: MP4CAT lossless remux when every clip is an mp4 with
/// matching params, else the ffmpeg path. The remux NEVER applies dedup.
fn concat_videos(urls: &[String], dedup: bool) -> Result<String> {
    if urls.len() < 2 {
        return Err(err("wire at least two clips to combine"));
    }
    let mut bufs: Vec<Vec<u8>> = Vec::new();
    let mut all_mp4 = true;
    for u in urls {
        let b = url_bytes(u)?;
        if !is_mp4(&b) {
            all_mp4 = false;
        }
        bufs.push(b);
    }
    if all_mp4 && mp4_params_match(&bufs) {
        match concat_mp4(&bufs) {
            Ok(out) => return Ok(bytes_to_data_url(&out, Some("video/mp4"))),
            Err(_) => {
                // MP4CAT parse issues fall through to ffmpeg, like JS
            }
        }
    }
    concat_videos_ffmpeg(urls, dedup)
}

// ---------------------------------------------------------------- MP4CAT port (mp4cat.mjs)

fn be_u16(b: &[u8], off: usize) -> std::result::Result<u16, String> {
    if off + 2 > b.len() {
        return Err("mp4 parse out of bounds".into());
    }
    Ok(u16::from_be_bytes([b[off], b[off + 1]]))
}
fn be_u32(b: &[u8], off: usize) -> std::result::Result<u32, String> {
    if off + 4 > b.len() {
        return Err("mp4 parse out of bounds".into());
    }
    Ok(u32::from_be_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]]))
}
fn be_i32(b: &[u8], off: usize) -> std::result::Result<i32, String> {
    Ok(be_u32(b, off)? as i32)
}
fn be_u64(b: &[u8], off: usize) -> std::result::Result<u64, String> {
    if off + 8 > b.len() {
        return Err("mp4 parse out of bounds".into());
    }
    Ok(u64::from_be_bytes([
        b[off], b[off + 1], b[off + 2], b[off + 3], b[off + 4], b[off + 5], b[off + 6], b[off + 7],
    ]))
}
fn fourcc(b: &[u8], off: usize) -> String {
    if off + 4 > b.len() {
        return String::new();
    }
    String::from_utf8_lossy(&b[off..off + 4]).into_owned()
}

#[derive(Clone)]
struct FoundBox {
    ty: String,
    start: usize,
    end: usize,
    body: usize,
}

fn walk_boxes(buf: &[u8], start: usize, end: usize) -> Vec<FoundBox> {
    let mut out = Vec::new();
    let mut p = start;
    while p + 8 <= end && p + 8 <= buf.len() {
        let size32 = match be_u32(buf, p) {
            Ok(v) => v,
            Err(_) => break,
        };
        let ty = fourcc(buf, p + 4);
        let (size, hs) = if size32 == 1 {
            match be_u64(buf, p + 8) {
                Ok(v) => (v as usize, 16usize),
                Err(_) => break,
            }
        } else if size32 == 0 {
            (end - p, 8usize)
        } else {
            (size32 as usize, 8usize)
        };
        if size < 8 || p + size > end {
            break;
        }
        out.push(FoundBox { ty, start: p, end: p + size, body: p + hs });
        p += size;
    }
    out
}

fn find_box<'a>(boxes: &'a [FoundBox], ty: &str) -> Option<&'a FoundBox> {
    boxes.iter().find(|b| b.ty == ty)
}

fn scan_for_box(buf: &[u8], ty: &str) -> Option<Vec<u8>> {
    let mut p = 0usize;
    while p + 8 <= buf.len() {
        if fourcc(buf, p + 4) == ty {
            if let Ok(size) = be_u32(buf, p) {
                let size = size as usize;
                if size >= 8 && p + size <= buf.len() {
                    return Some(buf[p..p + size].to_vec());
                }
            }
        }
        p += 1;
    }
    None
}

struct Mp4Sample {
    offset: usize,
    size: usize,
    dur: u32,
    cts: i32,
    sync: bool,
}

struct Mp4Track {
    kind: String,
    timescale: u32,
    stsd_raw: Vec<u8>,
    codec_cfg: Option<Vec<u8>>,
    samples: Vec<Mp4Sample>,
    width: u32,
    height: u32,
    channels: u32,
    sample_rate: u32,
}

fn parse_mp4(buf: &[u8]) -> std::result::Result<Vec<Mp4Track>, String> {
    let top = walk_boxes(buf, 0, buf.len());
    let moov = find_box(&top, "moov").ok_or_else(|| "no moov".to_string())?;
    let moov_boxes = walk_boxes(buf, moov.body, moov.end);
    let mut tracks = Vec::new();
    for trak in moov_boxes.iter().filter(|b| b.ty == "trak") {
        let tb = walk_boxes(buf, trak.body, trak.end);
        let mdia = match find_box(&tb, "mdia") {
            Some(m) => m.clone(),
            None => continue,
        };
        let mb = walk_boxes(buf, mdia.body, mdia.end);
        let mdhd = find_box(&mb, "mdhd").ok_or_else(|| "no mdhd".to_string())?;
        let hdlr = find_box(&mb, "hdlr").ok_or_else(|| "no hdlr".to_string())?;
        let mdhd_v1 = buf.get(mdhd.body).copied().unwrap_or(0) == 1;
        let timescale = be_u32(buf, mdhd.body + if mdhd_v1 { 20 } else { 12 })?;
        let handler = fourcc(buf, hdlr.body + 8);
        let kind = match handler.as_str() {
            "vide" => "video".to_string(),
            "soun" => "audio".to_string(),
            _ => handler.clone(),
        };
        let minf = find_box(&mb, "minf").ok_or_else(|| "no minf".to_string())?;
        let minf_b = walk_boxes(buf, minf.body, minf.end);
        let stbl = find_box(&minf_b, "stbl").ok_or_else(|| "no stbl".to_string())?;
        let sb = walk_boxes(buf, stbl.body, stbl.end);
        let stsd = find_box(&sb, "stsd").ok_or_else(|| "no stsd".to_string())?;
        let stsd_raw = buf[stsd.start..stsd.end].to_vec();
        let codec_cfg = scan_for_box(&stsd_raw, if kind == "video" { "avcC" } else { "esds" })
            .or_else(|| scan_for_box(&stsd_raw, "hvcC"))
            .or_else(|| scan_for_box(&stsd_raw, "vpcC"))
            .or_else(|| scan_for_box(&stsd_raw, "av1C"));
        let (mut width, mut height) = (0u32, 0u32);
        if kind == "video" {
            if let Some(tkhd) = find_box(&tb, "tkhd") {
                width = be_u16(buf, tkhd.end - 8)? as u32;
                height = be_u16(buf, tkhd.end - 4)? as u32;
            }
        }
        let (mut channels, mut sample_rate) = (0u32, 0u32);
        if kind == "audio" {
            if let Some(mp4a) = scan_for_box(&stsd_raw, "mp4a") {
                channels = be_u16(&mp4a, 24)? as u32;
                sample_rate = be_u16(&mp4a, 32)? as u32;
            }
        }
        let stts = find_box(&sb, "stts").ok_or_else(|| "no stts".to_string())?;
        let stsz = find_box(&sb, "stsz").ok_or_else(|| "no stsz".to_string())?;
        let stsc = find_box(&sb, "stsc").ok_or_else(|| "no stsc".to_string())?;
        let stco = find_box(&sb, "stco");
        let co64 = find_box(&sb, "co64");
        let ctts = find_box(&sb, "ctts");
        let stss = find_box(&sb, "stss");

        let stsz_sample_size = be_u32(buf, stsz.body + 4)?;
        let sample_count = be_u32(buf, stsz.body + 8)? as usize;
        let mut sizes = vec![0u32; sample_count];
        if stsz_sample_size == 0 {
            for i in 0..sample_count {
                sizes[i] = be_u32(buf, stsz.body + 12 + i * 4)?;
            }
        } else {
            sizes.fill(stsz_sample_size);
        }
        let stts_n = be_u32(buf, stts.body + 4)? as usize;
        let mut durs = vec![0u32; sample_count];
        let mut si = 0usize;
        for e in 0..stts_n {
            let cnt = be_u32(buf, stts.body + 8 + e * 8)? as usize;
            let delta = be_u32(buf, stts.body + 12 + e * 8)?;
            for _ in 0..cnt {
                if si >= sample_count {
                    break;
                }
                durs[si] = delta;
                si += 1;
            }
        }
        while si < sample_count {
            // JS: durs[si] = durs[si-2] || 0 (holes read as undefined → 0)
            durs[si] = if si >= 2 { durs[si - 2] } else { 0 };
            si += 1;
        }
        let mut cts = vec![0i32; sample_count];
        if let Some(ct) = ctts {
            let n = be_u32(buf, ct.body + 4)? as usize;
            let mut ci = 0usize;
            for e in 0..n {
                let cnt = be_u32(buf, ct.body + 8 + e * 8)? as usize;
                let off = be_i32(buf, ct.body + 12 + e * 8)?;
                for _ in 0..cnt {
                    if ci >= sample_count {
                        break;
                    }
                    cts[ci] = off;
                    ci += 1;
                }
            }
        }
        let mut sync_set: Option<std::collections::HashSet<u32>> = None;
        if let Some(ss) = stss {
            let n = be_u32(buf, ss.body + 4)? as usize;
            let mut set = std::collections::HashSet::new();
            for e in 0..n {
                set.insert(be_u32(buf, ss.body + 8 + e * 4)?);
            }
            sync_set = Some(set);
        }
        let co = stco.or(co64).ok_or_else(|| "no chunk offsets".to_string())?;
        let is64 = co64.is_some();
        let co_n = be_u32(buf, co.body + 4)? as usize;
        let mut chunk_offsets: Vec<usize> = Vec::with_capacity(co_n);
        for e in 0..co_n {
            let v = if is64 {
                be_u64(buf, co.body + 8 + e * 8)? as usize
            } else {
                be_u32(buf, co.body + 8 + e * 4)? as usize
            };
            chunk_offsets.push(v);
        }
        let stsc_n = be_u32(buf, stsc.body + 4)? as usize;
        let mut stsc_entries: Vec<(u32, u32)> = Vec::with_capacity(stsc_n);
        for e in 0..stsc_n {
            stsc_entries.push((
                be_u32(buf, stsc.body + 8 + e * 12)?,
                be_u32(buf, stsc.body + 12 + e * 12)?,
            ));
        }
        let mut samples: Vec<Mp4Sample> = Vec::new();
        let mut s_idx = 0usize;
        for c in 0..co_n {
            let mut spc = 1u32;
            for e in (0..stsc_entries.len()).rev() {
                if (c as u32 + 1) >= stsc_entries[e].0 {
                    spc = stsc_entries[e].1;
                    break;
                }
            }
            let mut off = chunk_offsets[c];
            for _ in 0..spc {
                if s_idx >= sample_count {
                    break;
                }
                samples.push(Mp4Sample {
                    offset: off,
                    size: sizes[s_idx] as usize,
                    dur: durs[s_idx],
                    cts: cts[s_idx],
                    sync: sync_set.as_ref().map(|s| s.contains(&((s_idx + 1) as u32))).unwrap_or(true),
                });
                off += sizes[s_idx] as usize;
                s_idx += 1;
            }
        }
        if samples.len() != sample_count {
            return Err(format!("sample count mismatch {}/{}", samples.len(), sample_count));
        }
        tracks.push(Mp4Track {
            kind,
            timescale,
            stsd_raw,
            codec_cfg,
            samples,
            width,
            height,
            channels,
            sample_rate,
        });
    }
    Ok(tracks)
}

fn is_mp4(b: &[u8]) -> bool {
    b.len() >= 12 && fourcc(b, 4) == "ftyp"
}

fn mp4_params_match(bufs: &[Vec<u8>]) -> bool {
    if bufs.len() < 2 {
        return false;
    }
    let parsed: std::result::Result<Vec<Vec<Mp4Track>>, String> = bufs.iter().map(|b| parse_mp4(b)).collect();
    let ps = match parsed {
        Ok(p) => p,
        Err(_) => return false,
    };
    // signature: exactly one video track, at most one audio track
    let sig = |p: &Vec<Mp4Track>| -> Option<(usize, Option<usize>, usize)> {
        let vids: Vec<usize> = p.iter().enumerate().filter(|(_, t)| t.kind == "video").map(|(i, _)| i).collect();
        let auds: Vec<usize> = p.iter().enumerate().filter(|(_, t)| t.kind == "audio").map(|(i, _)| i).collect();
        if vids.len() != 1 || auds.len() > 1 {
            return None;
        }
        Some((vids[0], auds.first().copied(), auds.len()))
    };
    let (bv, ba, bna) = match sig(&ps[0]) {
        Some(s) => s,
        None => return false,
    };
    let base_v = &ps[0][bv];
    let base_a = ba.map(|i| &ps[0][i]);
    for p in &ps[1..] {
        let (sv, sa, sna) = match sig(p) {
            Some(s) => s,
            None => return false,
        };
        if sna != bna {
            return false;
        }
        let s_v = &p[sv];
        if s_v.width != base_v.width || s_v.height != base_v.height {
            return false;
        }
        match (&s_v.codec_cfg, &base_v.codec_cfg) {
            (Some(a), Some(b)) if a == b => {}
            _ => return false,
        }
        if let Some(ba_t) = base_a {
            let sa_t = &p[sa.unwrap()];
            if sa_t.sample_rate != ba_t.sample_rate || sa_t.channels != ba_t.channels {
                return false;
            }
        }
    }
    true
}

// box writers
fn bx_u32(n: u32) -> [u8; 4] {
    n.to_be_bytes()
}
fn bx_u16(n: u16) -> [u8; 2] {
    n.to_be_bytes()
}
fn bx_box(ty: &str, parts: &[&[u8]]) -> Vec<u8> {
    let body_len: usize = parts.iter().map(|p| p.len()).sum();
    let mut out = Vec::with_capacity(body_len + 8);
    out.extend_from_slice(&((body_len + 8) as u32).to_be_bytes());
    out.extend_from_slice(ty.as_bytes());
    for p in parts {
        out.extend_from_slice(p);
    }
    out
}
fn bx_fullbox(ty: &str, version: u8, flags: u32, parts: &[&[u8]]) -> Vec<u8> {
    let vf = [(version), ((flags >> 16) & 0xff) as u8, ((flags >> 8) & 0xff) as u8, (flags & 0xff) as u8];
    let mut all: Vec<&[u8]> = vec![&vf];
    all.extend_from_slice(parts);
    bx_box(ty, &all)
}
fn rle(values: &[i64]) -> Vec<(u32, i64)> {
    let mut runs = Vec::new();
    let mut i = 0;
    while i < values.len() {
        let mut j = i + 1;
        while j < values.len() && values[j] == values[i] {
            j += 1;
        }
        runs.push(((j - i) as u32, values[i]));
        i = j;
    }
    runs
}

struct MergedSample {
    buf_idx: usize,
    offset: usize,
    size: usize,
    dur: i64,
    cts: i64,
    sync: bool,
    new_offset: usize,
}

struct MergedTrack {
    kind: String,
    timescale: u32,
    stsd_raw: Vec<u8>,
    width: u32,
    height: u32,
    samples: Vec<MergedSample>,
}

/// concatMp4 mirror with dedup fixed OFF (the browser remux never drops the
/// later clips' first sample — it carries the keyframe).
fn concat_mp4(buffers: &[Vec<u8>]) -> std::result::Result<Vec<u8>, String> {
    let parsed: std::result::Result<Vec<Vec<Mp4Track>>, String> =
        buffers.iter().map(|b| parse_mp4(b)).collect();
    let parsed = parsed?;
    let base = &parsed[0];
    let mut out_tracks: Vec<MergedTrack> = Vec::new();
    for bt in base.iter() {
        let kind = &bt.kind;
        if kind != "video" && kind != "audio" {
            continue;
        }
        let out_ts = bt.timescale;
        let mut merged = MergedTrack {
            kind: kind.clone(),
            timescale: out_ts,
            stsd_raw: bt.stsd_raw.clone(),
            width: bt.width,
            height: bt.height,
            samples: Vec::new(),
        };
        for (ci, clip) in parsed.iter().enumerate() {
            let t = clip
                .iter()
                .find(|x| &x.kind == kind)
                .ok_or_else(|| format!("clip {} missing {} track", ci, kind))?;
            let scale = out_ts as f64 / t.timescale as f64;
            for s in &t.samples {
                merged.samples.push(MergedSample {
                    buf_idx: ci,
                    offset: s.offset,
                    size: s.size,
                    dur: js_round(s.dur as f64 * scale) as i64,
                    cts: js_round(s.cts as f64 * scale) as i64,
                    sync: s.sync,
                    new_offset: 0,
                });
            }
        }
        out_tracks.push(merged);
    }

    let ftyp = bx_box("ftyp", &[b"isom", &bx_u32(0x200), b"isomiso2avc1mp41"]);
    let mut mdat_data: Vec<u8> = Vec::new();
    let mut cursor = ftyp.len() + 8;
    for t in out_tracks.iter_mut() {
        for s in t.samples.iter_mut() {
            s.new_offset = cursor;
            let src = &buffers[s.buf_idx];
            let end = (s.offset + s.size).min(src.len());
            let slice = if s.offset <= end { &src[s.offset..end] } else { &[][..] };
            mdat_data.extend_from_slice(slice);
            cursor += s.size;
        }
    }
    let mdat = bx_box("mdat", &[&mdat_data]);

    let moov = movie_box(&out_tracks);
    let mut out = Vec::with_capacity(ftyp.len() + mdat.len() + moov.len());
    out.extend_from_slice(&ftyp);
    out.extend_from_slice(&mdat);
    out.extend_from_slice(&moov);
    Ok(out)
}

/// Build the sample tables and headers for both stream-copy and freshly encoded movies.
fn movie_box(out_tracks: &[MergedTrack]) -> Vec<u8> {
    let mv_timescale: i64 = 1000;
    let mut max_dur_ms: i64 = 0;
    let mut trak_boxes: Vec<Vec<u8>> = Vec::new();
    let mut track_id: u32 = 1;
    for t in out_tracks {
        let total_ticks: i64 = t.samples.iter().map(|s| s.dur).sum();
        let dur_ms = js_round(total_ticks as f64 / t.timescale as f64 * mv_timescale as f64) as i64;
        if dur_ms > max_dur_ms {
            max_dur_ms = dur_ms;
        }
        let durs: Vec<i64> = t.samples.iter().map(|s| s.dur).collect();
        let stts_runs = rle(&durs);
        let mut stts_parts: Vec<u8> = Vec::new();
        for (count, val) in &stts_runs {
            stts_parts.extend_from_slice(&bx_u32(*count));
            stts_parts.extend_from_slice(&bx_u32(*val as u32));
        }
        let stts = bx_fullbox("stts", 0, 0, &[&bx_u32(stts_runs.len() as u32), &stts_parts]);
        let mut stsz_parts: Vec<u8> = Vec::new();
        for s in &t.samples {
            stsz_parts.extend_from_slice(&bx_u32(s.size as u32));
        }
        let stsz = bx_fullbox("stsz", 0, 0, &[&bx_u32(0), &bx_u32(t.samples.len() as u32), &stsz_parts]);
        let stsc_entry = [bx_u32(1), bx_u32(1), bx_u32(1)].concat();
        let stsc = bx_fullbox("stsc", 0, 0, &[&bx_u32(1), &stsc_entry]);
        let mut stco_parts: Vec<u8> = Vec::new();
        for s in &t.samples {
            stco_parts.extend_from_slice(&bx_u32(s.new_offset as u32));
        }
        let stco = bx_fullbox("stco", 0, 0, &[&bx_u32(t.samples.len() as u32), &stco_parts]);
        let mut children: Vec<Vec<u8>> = vec![t.stsd_raw.clone(), stts];
        if t.kind == "video" {
            if t.samples.iter().any(|s| s.cts != 0) {
                let cts_vals: Vec<i64> = t.samples.iter().map(|s| s.cts).collect();
                let ctts_runs = rle(&cts_vals);
                let mut ctts_parts: Vec<u8> = Vec::new();
                for (count, val) in &ctts_runs {
                    ctts_parts.extend_from_slice(&bx_u32(*count));
                    ctts_parts.extend_from_slice(&bx_u32(*val as u32));
                }
                children.push(bx_fullbox("ctts", 0, 0, &[&bx_u32(ctts_runs.len() as u32), &ctts_parts]));
            }
            let sync_idx: Vec<u32> = t
                .samples
                .iter()
                .enumerate()
                .filter(|(_, s)| s.sync)
                .map(|(i, _)| (i + 1) as u32)
                .collect();
            if !sync_idx.is_empty() && sync_idx.len() != t.samples.len() {
                let mut stss_parts: Vec<u8> = Vec::new();
                for v in &sync_idx {
                    stss_parts.extend_from_slice(&bx_u32(*v));
                }
                children.push(bx_fullbox("stss", 0, 0, &[&bx_u32(sync_idx.len() as u32), &stss_parts]));
            }
        }
        children.push(stsc);
        children.push(stsz);
        children.push(stco);
        let child_refs: Vec<&[u8]> = children.iter().map(|c| c.as_slice()).collect();
        let stbl = bx_box("stbl", &child_refs);

        let media_header = if t.kind == "video" {
            bx_box("vmhd", &[&[0u8, 0, 0, 1], &[0u8; 8]])
        } else {
            bx_box("smhd", &[&[0u8; 8]])
        };
        let url_box = bx_fullbox("url ", 0, 1, &[]);
        let dref = bx_fullbox("dref", 0, 0, &[&bx_u32(1), &url_box]);
        let dinf = bx_box("dinf", &[&dref]);
        let minf = bx_box("minf", &[&media_header, &dinf, &stbl]);

        let hdlr_name: &[u8] = if t.kind == "video" { b"VideoHandler\0" } else { b"SoundHandler\0" };
        let hdlr = bx_fullbox(
            "hdlr",
            0,
            0,
            &[
                &bx_u32(0),
                if t.kind == "video" { &b"vide"[..] } else { &b"soun"[..] },
                &[0u8; 12],
                hdlr_name,
            ],
        );
        let mdhd = bx_fullbox(
            "mdhd",
            0,
            0,
            &[
                &bx_u32(0),
                &bx_u32(0),
                &bx_u32(t.timescale),
                &bx_u32(total_ticks as u32),
                &[0x55u8, 0xc4, 0, 0],
            ],
        );
        let mdia = bx_box("mdia", &[&mdhd, &hdlr, &minf]);

        let matrix: Vec<u8> = [
            bx_u32(0x00010000),
            bx_u32(0),
            bx_u32(0),
            bx_u32(0),
            bx_u32(0x00010000),
            bx_u32(0),
            bx_u32(0),
            bx_u32(0),
            bx_u32(0x40000000),
        ]
        .concat();
        let mut tkhd_body: Vec<u8> = Vec::new();
        tkhd_body.extend_from_slice(&bx_u32(0));
        tkhd_body.extend_from_slice(&bx_u32(0));
        tkhd_body.extend_from_slice(&bx_u32(track_id));
        tkhd_body.extend_from_slice(&bx_u32(0));
        tkhd_body.extend_from_slice(&bx_u32(dur_ms as u32));
        tkhd_body.extend_from_slice(&[0u8; 8]);
        tkhd_body.extend_from_slice(&bx_u16(0));
        tkhd_body.extend_from_slice(&bx_u16(0));
        tkhd_body.extend_from_slice(&bx_u16(if t.kind == "audio" { 0x0100 } else { 0 }));
        tkhd_body.extend_from_slice(&bx_u16(0));
        tkhd_body.extend_from_slice(&matrix);
        tkhd_body.extend_from_slice(&bx_u32(t.width << 16));
        tkhd_body.extend_from_slice(&bx_u32(t.height << 16));
        let tkhd = bx_fullbox("tkhd", 0, 7, &[&tkhd_body]);
        trak_boxes.push(bx_box("trak", &[&tkhd, &mdia]));
        track_id += 1;
    }
    let matrix: Vec<u8> = [
        bx_u32(0x00010000),
        bx_u32(0),
        bx_u32(0),
        bx_u32(0),
        bx_u32(0x00010000),
        bx_u32(0),
        bx_u32(0),
        bx_u32(0),
        bx_u32(0x40000000),
    ]
    .concat();
    let mvhd = bx_fullbox(
        "mvhd",
        0,
        0,
        &[
            &bx_u32(0),
            &bx_u32(0),
            &bx_u32(mv_timescale as u32),
            &bx_u32(max_dur_ms as u32),
            &bx_u32(0x00010000),
            &bx_u16(0x0100),
            &bx_u16(0),
            &[0u8; 8],
            &matrix,
            &[0u8; 24],
            &bx_u32(track_id),
        ],
    );
    let mut moov_parts: Vec<&[u8]> = vec![&mvhd];
    for tb in &trak_boxes {
        moov_parts.push(tb);
    }
    bx_box("moov", &moov_parts)
}

// ---------------------------------------------------------------- API legs (inpaint / lipsync)

fn b64_image_mime(b64: &str) -> &'static str {
    if b64.starts_with("/9j/") {
        "image/jpeg"
    } else if b64.starts_with("iVBOR") {
        "image/png"
    } else if b64.starts_with("R0lG") {
        "image/gif"
    } else if b64.starts_with("UklGR") {
        "image/webp"
    } else {
        "image/png"
    }
}

fn model_opts(node: &Node) -> Map<String, Value> {
    match node.fields.get("modelOpts") {
        Some(Value::Object(m)) => m.clone(),
        _ => Map::new(),
    }
}

/// The inpaint image call: same body shape as lib.rs gen_image plus the mask.
/// Also returns the per-item `layer` metadata aligned by index with the urls
/// (None when no item carries any; the top-level url fallback has none).
fn gen_image_local(
    ctx: &Ctx,
    node: &Node,
    prompt: &str,
    image_data_url: Option<&str>,
    mask_data_url: Option<&str>,
) -> Result<(Vec<String>, Option<Vec<Value>>, Option<f64>)> {
    let model = mdl(node)?;
    let size = node.field_str("size").filter(|s| !s.is_empty()).unwrap_or_else(|| "1024x1024".to_string());
    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert("size".into(), json!(size));
    body.insert("n".into(), json!(1));
    body.insert("response_format".into(), json!("b64_json"));
    if !prompt.is_empty() {
        body.insert("prompt".into(), json!(prompt));
    }
    if let Some(u) = image_data_url {
        body.insert("imageDataUrl".into(), json!(u));
    }
    if let Some(u) = mask_data_url {
        body.insert("maskDataUrl".into(), json!(u));
    }
    let mut extra = model_opts(node);
    if let Some(n) = node.fields.get("seed") {
        if n.is_number() {
            extra.insert("seed".into(), n.clone());
        }
    }
    for (k, v) in extra {
        body.insert(k, v);
    }
    let (j, cost) = ctx.post_json("/v1/images/generations", &Value::Object(body))?;
    let mut urls: Vec<String> = Vec::new();
    let mut layers: Vec<Value> = Vec::new();
    if let Some(data) = j.get("data").and_then(|d| d.as_array()) {
        for item in data {
            let mut pushed = false;
            if let Some(b64) = item.get("b64_json").and_then(|v| v.as_str()) {
                if !b64.is_empty() {
                    urls.push(format!("data:{};base64,{}", b64_image_mime(b64), b64));
                    pushed = true;
                }
            }
            if !pushed {
                if let Some(u) = item.get("url").and_then(|v| v.as_str()) {
                    if !u.is_empty() {
                        urls.push(u.to_string());
                        pushed = true;
                    }
                }
            }
            if pushed {
                layers.push(item.get("layer").cloned().unwrap_or(Value::Null));
            }
        }
    }
    if urls.is_empty() {
        for key in ["url", "image_url"] {
            if let Some(u) = j.get(key).and_then(|v| v.as_str()) {
                if !u.is_empty() {
                    urls.push(u.to_string());
                    break;
                }
            }
        }
    }
    if urls.is_empty() {
        return Err(err("no image in response"));
    }
    let layers = if layers.iter().any(|l| !l.is_null()) { Some(layers) } else { None };
    Ok((urls, layers, cost))
}

enum AudioPart {
    Url(String),
    DataUrl(String),
}

fn url_encode_component(s: &str) -> String {
    let mut out = String::new();
    for &b in s.as_bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{:02X}", b));
        }
    }
    out
}

fn http_status_prefix(msg: &str) -> Option<u32> {
    let b = msg.as_bytes();
    if b.len() >= 4 && b[0].is_ascii_digit() && b[1].is_ascii_digit() && b[2].is_ascii_digit() && b[3] == b':' {
        msg[..3].parse().ok()
    } else {
        None
    }
}

/// client.video mirror: submit + poll /api/video/status until terminal.
fn gen_video_local(
    ctx: &Ctx,
    model: &str,
    prompt: &str,
    image_data_url: Option<&str>,
    audio: &AudioPart,
    extra: &Map<String, Value>,
) -> std::result::Result<(String, Option<f64>), NodeError> {
    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert("prompt".into(), json!(prompt));
    if let Some(u) = image_data_url {
        body.insert("imageDataUrl".into(), json!(u));
    }
    match audio {
        AudioPart::Url(u) => {
            body.insert("audioUrl".into(), json!(u));
        }
        AudioPart::DataUrl(u) => {
            body.insert("audioDataUrl".into(), json!(u));
        }
    }
    for (k, v) in extra {
        body.insert(k.clone(), v.clone());
    }
    let (j, cost) = ctx.post_json("/api/generate-video", &Value::Object(body))?;
    // Everything after the submit carries its cost (JS bills at submit).
    let billed = |msg: String| NodeError { message: msg, cost };
    let run_id = match j.get("runId").or_else(|| j.get("id")) {
        Some(v) => value_text(v),
        None => return Err(billed("no runId returned".into())),
    };
    if run_id.is_empty() {
        return Err(billed("no runId returned".into()));
    }
    let path = format!("/api/video/status?requestId={}", url_encode_component(&run_id));
    let deadline = Instant::now() + Duration::from_secs(3600);
    loop {
        std::thread::sleep(Duration::from_millis(400));
        let sj = match ctx.get_json(&path) {
            Ok((v, _)) => v,
            Err(e) => {
                // transient poll errors are swallowed; client errors are not
                if let Some(s) = http_status_prefix(&e.0) {
                    if (400..500).contains(&s) && s != 408 && s != 429 {
                        return Err(billed(e.0));
                    }
                }
                if Instant::now() > deadline {
                    return Err(billed("video timed out (3600s) — the job may still finish; check NanoGPT history".into()));
                }
                continue;
            }
        };
        let data = sj.get("data").filter(|d| d.is_object()).cloned().unwrap_or_else(|| sj.clone());
        let st = data
            .get("status")
            .or_else(|| sj.get("status"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_ascii_uppercase();
        match st.as_str() {
            "COMPLETED" | "SUCCEEDED" => {
                let out = data.get("output").cloned().or_else(|| sj.get("output").cloned()).unwrap_or(Value::Null);
                let url = extract_video_url(&out);
                return match url {
                    Some(u) => Ok((u, cost)),
                    None => Err(billed("completed but no video url".into())),
                };
            }
            "FAILED" | "ERROR" | "CANCELED" | "CANCELLED" | "CONTENT_POLICY_VIOLATION" => {
                let detail = data
                    .get("error")
                    .map(|v| value_text(v))
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| st.clone());
                return Err(billed(format!("video failed: {}", detail)));
            }
            _ => {}
        }
        if Instant::now() > deadline {
            return Err(billed("video timed out (3600s) — the job may still finish; check NanoGPT history".into()));
        }
    }
}

fn extract_video_url(out: &Value) -> Option<String> {
    match out {
        Value::Object(_) => {
            if let Some(v) = out.get("video") {
                if let Some(u) = v.get("url").and_then(|x| x.as_str()) {
                    if !u.is_empty() {
                        return Some(u.to_string());
                    }
                }
            }
            if let Some(u) = out.get("url").and_then(|x| x.as_str()) {
                if !u.is_empty() {
                    return Some(u.to_string());
                }
            }
            if let Some(arr) = out.get("video").and_then(|v| v.as_array()) {
                if let Some(u) = arr.first().and_then(|v| v.get("url")).and_then(|x| x.as_str()) {
                    if !u.is_empty() {
                        return Some(u.to_string());
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// JS /up to\s+(\d+(?:\.\d+)?)\s*second/i AND /INVALID_AUDIO_DURATION|audio.{0,15}duration/i
fn parse_duration_cap(msg: &str) -> Option<f64> {
    let low = msg.to_lowercase();
    let durationish = low.contains("invalid_audio_duration") || window_match(&low, "audio", "duration", 15);
    if !durationish {
        return None;
    }
    let bytes = low.as_bytes();
    let mut search = 0usize;
    while let Some(pos) = low[search..].find("up to").map(|p| p + search) {
        let mut i = pos + 5;
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        let num_start = i;
        let mut dot = false;
        while i < bytes.len() && (bytes[i].is_ascii_digit() || (bytes[i] == b'.' && !dot)) {
            if bytes[i] == b'.' {
                dot = true;
            }
            i += 1;
        }
        if i > num_start {
            let mut j = i;
            while j < bytes.len() && (bytes[j] as char).is_whitespace() {
                j += 1;
            }
            if low[j..].starts_with("second") {
                if let Ok(v) = low[num_start..i].parse::<f64>() {
                    return Some(v);
                }
            }
        }
        search = pos + 5;
    }
    None
}

/// "a" followed within `gap` chars by "b" (the JS regex /a.{0,N}b/).
fn window_match(hay: &str, a: &str, b: &str, gap: usize) -> bool {
    let mut search = 0usize;
    while let Some(pos) = hay[search..].find(a).map(|p| p + search) {
        let after = pos + a.len();
        let window_end = (after + gap + b.len()).min(hay.len());
        if hay[after..window_end].contains(b) {
            // ensure b starts within the gap
            if let Some(bp) = hay[after..].find(b) {
                if bp <= gap {
                    return true;
                }
            }
        }
        search = pos + 1;
    }
    false
}

fn has_left_right_audio(msg: &str) -> bool {
    let low = msg.to_lowercase();
    low.contains("left and right") || window_match(&low, "left", "audio", 6) || window_match(&low, "right", "audio", 6)
}

// ---------------------------------------------------------------- node runners

fn done(map: Map<String, Value>, cost: Option<f64>) -> NodeOutcome {
    NodeOutcome::Done(map, cost)
}

fn single(key: &str, value: String, cost: Option<f64>) -> NodeOutcome {
    let mut m = Map::new();
    m.insert(key.into(), json!(value));
    done(m, cost)
}

/// JS/Py fitImageJpeg (Decide image shrink): composite onto white at
/// <= max_dim on the long edge, stepping mjpeg quality (3, 5, 8, 12) then
/// scale (x0.75, 6 rounds) until the data: URL fits `budget` chars.
/// Deterministic against the same ffmpeg the JS/Py twins drive.
pub fn fit_image_jpeg(url: &str, max_dim: i64, budget: usize) -> Result<String> {
    let dir = TempDir::new()?;
    let in_path = write_input(&dir, "in", url)?;
    let in_s = in_path.to_string_lossy().to_string();
    let out = run_proc(
        "ffprobe",
        &[
            "-v".into(), "error".into(),
            "-select_streams".into(), "v:0".into(),
            "-show_entries".into(), "stream=width,height".into(),
            "-of".into(), "csv=p=0:s=x".into(),
            in_s.clone(),
        ],
    )?;
    let dims: Vec<i64> = out
        .trim()
        .split('x')
        .map(|p| p.trim().parse::<i64>().unwrap_or(0))
        .collect();
    let (sw, sh) = if dims.len() == 2 { (dims[0], dims[1]) } else { (0, 0) };
    if !(sw > 0 && sh > 0) {
        return Err(err("couldn't read an image to judge"));
    }
    let mut s = (max_dim as f64 / sw.max(sh) as f64).min(1.0);
    for round in 0..6 {
        // Python round() is half-to-even; match it or dims can differ by 2px
        let w = (((sw as f64 * s / 2.0).round_ties_even() as i64) * 2).max(2);
        let h = (((sh as f64 * s / 2.0).round_ties_even() as i64) * 2).max(2);
        for q in [3, 5, 8, 12] {
            let out_path = dir.path(&format!("fit-{}-{}.jpg", round, q));
            run_proc(
                "ffmpeg",
                &[
                    "-y".into(),
                    "-i".into(), in_s.clone(),
                    "-filter_complex".into(),
                    format!(
                        "color=c=white:s={}x{}[bg];[0:v]scale={}:{},format=rgba[fg];[bg][fg]overlay=shortest=1,format=yuvj420p",
                        w, h, w, h
                    ),
                    "-frames:v".into(), "1".into(),
                    "-q:v".into(), q.to_string(),
                    out_path.to_string_lossy().to_string(),
                ],
            )?;
            let bytes = std::fs::read(&out_path)
                .map_err(|e| err(format!("couldn't read fitted image: {}", e)))?;
            let result = bytes_to_data_url(&bytes, Some("image/jpeg"));
            if result.len() <= budget {
                return Ok(result);
            }
        }
        s *= 0.75;
    }
    Err(err("couldn't shrink an image small enough for the decision model"))
}

pub fn run_resize(node: &Node, inputs: &HashMap<String, Value>, _ctx: &Ctx) -> Result<NodeOutcome> {
    let src = match input_text(inputs, "image") {
        Some(s) if !s.is_empty() => s,
        _ => return Err(err("no image input")),
    };
    let mode = node.field_str("mode").filter(|s| !s.is_empty()).unwrap_or_else(|| "fit".to_string());
    let w = js_parse_int(field(node, "width")).unwrap_or(0);
    let h = js_parse_int(field(node, "height")).unwrap_or(0);
    let out = resize_crop_image(&src, &mode, w, h)?;
    Ok(single("image", out, Some(0.0)))
}

pub fn run_vframes(node: &Node, inputs: &HashMap<String, Value>, _ctx: &Ctx) -> Result<NodeOutcome> {
    let src = match input_text(inputs, "video") {
        Some(s) if !s.is_empty() => s,
        _ => return Err(err("no video input")),
    };
    let frames_n = js_parse_int(field(node, "frames")).unwrap_or(1);
    let gap = js_number(field(node, "gap"));
    let dir = node.field_str("dir").filter(|s| !s.is_empty()).unwrap_or_else(|| "end".to_string());
    let frames = extract_video_frames(&src, frames_n, gap, &dir)?;
    let mut m = Map::new();
    for (i, f) in frames.into_iter().enumerate() {
        m.insert(format!("frame{}", i + 1), json!(f));
    }
    Ok(done(m, Some(0.0)))
}

/// clipN / vidN port index (JS CLIP_PORT_RE / VID_PORT_RE).
fn clip_port_idx(port: &str) -> Option<(i64, bool)> {
    for (prefix, is_clip) in [("clip", true), ("vid", false)] {
        if let Some(rest) = port.strip_prefix(prefix) {
            if !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()) {
                return rest.parse::<i64>().ok().map(|n| (n, is_clip));
            }
        }
    }
    None
}

pub fn run_combine(node: &Node, inputs: &HashMap<String, Value>, _ctx: &Ctx) -> Result<NodeOutcome> {
    let mut wired: Vec<(i64, String, String)> = Vec::new();
    for (port, v) in inputs.iter() {
        if let Some((idx, _)) = clip_port_idx(port) {
            let t = value_text(v);
            if !t.is_empty() {
                wired.push((idx, port.clone(), t));
            }
        }
    }
    wired.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let mut urls: Vec<String> = Vec::new();
    for (_, _, t) in wired {
        if !urls.contains(&t) {
            urls.push(t);
        }
    }
    if urls.len() < 2 {
        return Err(err("wire at least two clips to combine"));
    }
    let dedup = match node.fields.get("dedup") {
        None | Some(Value::Null) => true,
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => s != "false",
        Some(Value::Number(n)) => n.as_f64() != Some(0.0),
        _ => true,
    };
    let out = concat_videos(&urls, dedup)?;
    Ok(single("video", out, Some(0.0)))
}

pub fn run_soundtrack(node: &Node, inputs: &HashMap<String, Value>, _ctx: &Ctx) -> Result<NodeOutcome> {
    let video = match input_text(inputs, "video") {
        Some(s) if !s.is_empty() => s,
        _ => return Err(err("no video input")),
    };
    let audio = match input_text(inputs, "audio") {
        Some(s) if !s.is_empty() => s,
        _ => return Err(err("no audio input")),
    };
    let loop_audio = match node.fields.get("loop") {
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64() == Some(1.0),
        Some(Value::String(s)) => s == "true" || s == "1",
        _ => false,
    };
    let out = mux_soundtrack(&video, &audio, loop_audio)?;
    Ok(single("video", out, Some(0.0)))
}

pub fn run_trim(node: &Node, inputs: &HashMap<String, Value>, _ctx: &Ctx) -> Result<NodeOutcome> {
    let audio = match input_text(inputs, "audio") {
        Some(s) if !s.is_empty() => s,
        _ => return Err(err("no audio input")),
    };
    let start = field_start(node);
    let length = match js_parse_float(field(node, "length")) {
        Some(v) if v.is_finite() => v,
        _ => 30.0,
    };
    let out = trim_audio_to_wav(&audio, start, length, 16000, false)?;
    Ok(single("audio", out, Some(0.0)))
}

pub fn run_extractaudio(node: &Node, inputs: &HashMap<String, Value>, _ctx: &Ctx) -> Result<NodeOutcome> {
    let video = match input_text(inputs, "video") {
        Some(s) if !s.is_empty() => s,
        _ => return Err(err("no video input")),
    };
    let length = match js_parse_float(field(node, "length")) {
        Some(v) if v.is_finite() && v > 0.0 => v,
        _ => 0.0,
    };
    let out = trim_audio_to_wav(&video, 0.0, length, 16000, true)?;
    Ok(single("audio", out, Some(0.0)))
}

/// JS inpaint input resolution: wired value wins over the field unless null.
fn wired_or_field(node: &Node, inputs: &HashMap<String, Value>, key: &str) -> String {
    match inputs.get(key) {
        Some(v) if !v.is_null() => value_text(v),
        _ => node.field_str(key).unwrap_or_default(),
    }
}

pub fn run_inpaint(node: &Node, inputs: &HashMap<String, Value>, ctx: &Ctx) -> Result<NodeOutcome> {
    let source = wired_or_field(node, inputs, "image");
    if source.is_empty() {
        return Err(err("no image — supply the image to repaint"));
    }
    let mask = wired_or_field(node, inputs, "mask");
    if mask.is_empty() {
        return Err(err("no mask — supply a B/W mask (white = repaint)"));
    }
    let prompt = prompt_of(node, inputs)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| err("no prompt — say what to paint into the masked area"))?;
    let mask_url = mask_to_source(&mask, &source)?;
    let (urls, layers, cost) = gen_image_local(ctx, node, &prompt, Some(&source), Some(&mask_url))?;
    Ok(NodeOutcome::Done(multi_image_out(urls, layers), cost))
}

fn add_costs(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x + y),
        (Some(x), None) => Some(x),
        (None, y) => y,
    }
}

pub fn run_lipsync(node: &Node, inputs: &HashMap<String, Value>, ctx: &Ctx) -> std::result::Result<NodeOutcome, NodeError> {
    // No catalog access from Ctx: JS's avatarNeedsPortrait is true on a
    // catalog miss, so the portrait requirement applies unconditionally here.
    let image = match input_text(inputs, "image") {
        Some(s) if !s.is_empty() => s,
        _ => return Err(err("no image input").into()),
    };
    let audio = match input_text(inputs, "audio") {
        Some(s) if !s.is_empty() => s,
        _ => return Err(err("no audio input").into()),
    };
    let prompt = prompt_of(node, inputs).map(|s| s.trim().to_string()).unwrap_or_default();
    let model = mdl(node)?;
    let img = fit_image_inline(&image)?;
    let extra = model_opts(node);
    let mut trim_sec: Option<f64> = None;
    // Every submit bills, including a failed attempt's (JS onCost per submit).
    let mut spent: Option<f64> = None;
    for attempt in 0..2 {
        let audio_part = if let Some(ts) = trim_sec {
            match trim_audio_to_wav(&audio, 0.0, ts, 24000, false) {
                Ok(t) => AudioPart::DataUrl(t),
                Err(e) => {
                    let low = audio.to_ascii_lowercase();
                    if low.starts_with("http:") || low.starts_with("https:") {
                        return Err(NodeError {
                            message: "the audio is too long for this model and the trim fallback can't fetch remote audio in this runtime — host the audio as a data URL or pre-trim it to under a minute".into(),
                            cost: spent,
                        });
                    }
                    return Err(NodeError { message: e.0, cost: spent });
                }
            }
        } else if audio.to_ascii_lowercase().starts_with("http:") || audio.to_ascii_lowercase().starts_with("https:") {
            AudioPart::Url(audio.clone())
        } else {
            AudioPart::DataUrl(audio.clone())
        };
        match gen_video_local(ctx, &model, &prompt, Some(&img), &audio_part, &extra) {
            Ok((url, cost)) => return Ok(single("video", url, add_costs(spent, cost))),
            Err(e) => {
                spent = add_costs(spent, e.cost);
                let e = NodeError { cost: spent, ..e };
                if attempt > 0 {
                    return Err(e);
                }
                let msg = e.message.clone();
                if msg.starts_with("video failed:") {
                    return Err(e);
                }
                if let Some(cap) = parse_duration_cap(&msg) {
                    trim_sec = Some((cap - 0.1).clamp(1.0, 60.0));
                    continue;
                }
                let low = msg.to_lowercase();
                if low.contains("large") || low.contains("media_inline") || low.contains("~4 mb") || low.contains("inline") {
                    trim_sec = Some(15.0);
                    continue;
                }
                if has_left_right_audio(&msg) {
                    return Err(NodeError {
                        message: "this clip has different voices on the left and right channels — the model can't tell who's speaking. Use a mono or single-speaker clip.".into(),
                        cost: spent,
                    });
                }
                return Err(e);
            }
        }
    }
    Err(NodeError { message: "lipsync failed".into(), cost: spent })
}
