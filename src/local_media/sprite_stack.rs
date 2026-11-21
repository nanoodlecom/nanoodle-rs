//! Local sprite nodes. Each call owns its QuickJS realm and temporary media.
use super::{
    b64_decode, b64_encode, bytes_to_data_url, decode_png, done, encode_png_rgba, err, input_text,
    run_proc, sniff_mime, url_bytes, PngImage, TempDir,
};
use crate::{Node, NodeOutcome, Result};
use rquickjs::{function::Func, Context, Function, Runtime};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::fs;

fn ffmpeg(arguments: &[&str]) -> Result<()> {
    let args = ["-nostdin", "-y", "-v", "error"]
        .iter()
        .chain(arguments)
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    run_proc("ffmpeg", &args).map(|_| ())
}

fn read_image(encoded: &str, temp: &TempDir) -> Result<String> {
    let raw = b64_decode(encoded)?;
    let mime = sniff_mime(&raw);
    let ext = mime
        .strip_prefix("image/")
        .ok_or_else(|| err("Expected a PNG, JPEG, WebP or GIF image"))?;
    let image = if let Ok(img) = decode_png(&raw) {
        img
    } else {
        let source = temp.path(&format!("source.{ext}"));
        let target = temp.path("decoded.pam");
        fs::write(&source, raw).map_err(err_io)?;
        ffmpeg(&[
            "-protocol_whitelist",
            "file",
            "-f",
            "image2",
            "-i",
            &source.to_string_lossy(),
            "-frames:v",
            "1",
            "-c:v",
            "pam",
            "-pix_fmt",
            "rgba",
            &target.to_string_lossy(),
        ])?;
        let pam = fs::read(target).map_err(err_io)?;
        let marker = b"ENDHDR\n";
        let end = pam
            .windows(marker.len())
            .position(|x| x == marker)
            .ok_or_else(|| err("Missing PAM header"))?;
        let header = std::str::from_utf8(&pam[..end]).map_err(|_| err("Invalid PAM header"))?;
        let number = |prefix: &str| {
            header
                .lines()
                .find_map(|s| s.strip_prefix(prefix)?.parse::<u32>().ok())
                .ok_or_else(|| err("Invalid PAM dimensions"))
        };
        let width = number("WIDTH ")?;
        let height = number("HEIGHT ")?;
        let rgba = pam[end + marker.len()..].to_vec();
        if number("DEPTH ")? != 4 || rgba.len() as u64 != u64::from(width) * u64::from(height) * 4 {
            return Err(err("Invalid RGBA image"));
        }
        PngImage {
            width,
            height,
            rgba,
        }
    };
    Ok(
        json!({"width": image.width, "height": image.height, "rgba": b64_encode(&image.rgba)})
            .to_string(),
    )
}

fn err_io(e: std::io::Error) -> crate::NanoodleError {
    err(e.to_string())
}
fn err_js(e: rquickjs::Error) -> crate::NanoodleError {
    err(format!("Sprite renderer: {e}"))
}
fn json_value(s: &str) -> Result<Value> {
    serde_json::from_str(s).map_err(|e| err(format!("Invalid sprite result: {e}")))
}

fn png(frame: &Value) -> Result<Vec<u8>> {
    let dim = |name: &str| {
        frame[name]
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .filter(|&v| v > 0)
            .ok_or_else(|| err("Invalid sprite dimensions"))
    };
    let width = dim("width")?;
    let height = dim("height")?;
    let rgba = b64_decode(
        frame["rgba"]
            .as_str()
            .ok_or_else(|| err("Missing sprite pixels"))?,
    )?;
    if rgba.len() as u64 != u64::from(width) * u64::from(height) * 4 {
        return Err(err("Invalid sprite pixels"));
    }
    Ok(encode_png_rgba(&PngImage {
        width,
        height,
        rgba,
    }))
}

fn turntable(frames: Function, plan: &Value, temp: &TempDir) -> Result<String> {
    let count = plan["count"]
        .as_u64()
        .ok_or_else(|| err("Invalid frame count"))?;
    let duration = plan["duration"]
        .as_f64()
        .ok_or_else(|| err("Invalid duration"))?;
    for index in 0..count {
        let frame: String = frames.call((index,)).map_err(err_js)?;
        fs::write(
            temp.path(&format!("{index:05}.png")),
            png(&json_value(&frame)?)?,
        )
        .map_err(err_io)?;
    }
    let rate = (count as f64 / duration).to_string();
    let source = temp.path("%05d.png");
    let target = temp.path("turntable.mp4");
    let encode = |codec: &str| {
        ffmpeg(&[
            "-framerate",
            &rate,
            "-i",
            &source.to_string_lossy(),
            "-an",
            "-c:v",
            codec,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            &target.to_string_lossy(),
        ])
    };
    if let Err(e) = encode("libx264") {
        if !e.0.to_lowercase().contains("unknown encoder") || !e.0.contains("libx264") {
            return Err(e);
        }
        encode("mpeg4")?;
    }
    Ok(bytes_to_data_url(
        &fs::read(target).map_err(err_io)?,
        Some("video/mp4"),
    ))
}

pub fn run_sprite(node: &Node, inputs: &HashMap<String, Value>) -> Result<NodeOutcome> {
    let port = if node.ty == "slice-3d" {
        "model"
    } else {
        "sheet"
    };
    let url = input_text(inputs, port)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| err(format!("Wire {port} into {}.", node.ty)))?;
    let source = b64_encode(&url_bytes(&url)?);
    let palette = match input_text(inputs, "paletteImage").filter(|s| !s.is_empty()) {
        Some(url) if node.field_str("palette").as_deref() != Some("0") => {
            b64_encode(&url_bytes(&url)?)
        }
        _ => String::new(),
    };
    let fields = Value::Object(node.fields.clone()).to_string();
    let temp = std::rc::Rc::new(TempDir::new()?);
    let image_temp = temp.clone();
    let runtime = Runtime::new().map_err(err_js)?;
    let context = Context::full(&runtime).map_err(err_js)?;
    context.with(|js| {
        let globals = js.globals();
        globals
            .set(
                "nativeUtf8",
                Func::from(|s: String| {
                    b64_decode(&s)
                        .map(|b| String::from_utf8_lossy(&b).into_owned())
                        .unwrap_or_default()
                }),
            )
            .map_err(err_js)?;
        globals
            .set(
                "nativeEncodeUtf8",
                Func::from(|s: String| b64_encode(s.as_bytes())),
            )
            .map_err(err_js)?;
        globals
            .set(
                "nativeImage",
                Func::from(move |s: String| {
                    read_image(&s, &image_temp)
                        .unwrap_or_else(|e| json!({"error": e.0}).to_string())
                }),
            )
            .map_err(err_js)?;
        js.eval::<(), _>(include_str!("../sprite-core.js"))
            .map_err(err_js)?;
        globals
            .get::<_, Function>("startSprite")
            .map_err(err_js)?
            .call::<_, ()>((node.ty.as_str(), fields, source, palette))
            .map_err(err_js)?;
        while js.execute_pending_job() {}
        let result = json_value(&globals.get::<_, String>("spriteResult").map_err(err_js)?)?;
        if let Some(message) = result["error"].as_str() {
            return Err(err(message));
        }
        let out = &result["out"];
        let mut ports = Map::new();
        for name in ["sheet", "image"] {
            if let Some(frame) = out.get(name) {
                ports.insert(
                    name.to_string(),
                    json!(bytes_to_data_url(&png(frame)?, Some("image/png"))),
                );
            }
        }
        if let Some(layout) = out.get("layout") {
            ports.insert("layout".into(), layout.clone());
        }
        if !out["plan"].is_null() {
            let frames = globals.get::<_, Function>("spriteFrame").map_err(err_js)?;
            ports.insert(
                "video".into(),
                json!(turntable(frames, &out["plan"], &temp)?),
            );
        }
        Ok(done(ports, Some(0.0)))
    })
}
