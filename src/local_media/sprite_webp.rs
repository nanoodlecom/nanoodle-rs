//! Decode the first displayed WebP frame into the sprite renderer's RGBA layout.
//! The bitstream routines live in sprite_webp/ with their source and license notice.
#![forbid(unsafe_code)]
use super::{err, PngImage};
use crate::Result;
use std::io::Cursor;

mod decoder;
mod huffman;
mod loop_filter;
mod lossless;
mod lossless_transform;
mod transform;
mod vp8;
mod vp8_arithmetic_decoder;
mod yuv;

fn u24(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], 0])
}

/// RIFF chunk length excludes its 8-byte header and alignment byte.
fn chunk<'a>(input: &mut &'a [u8]) -> Result<(&'a [u8], &'a [u8])> {
    let header = input
        .get(..8)
        .ok_or_else(|| err("Truncated WebP chunk header"))?;
    let len = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
    let body = &input[8..];
    let padded = len
        .checked_add(len & 1)
        .ok_or_else(|| err("Invalid WebP chunk length"))?;
    let rest = body
        .get(padded..)
        .ok_or_else(|| err("Truncated WebP chunk"))?;
    *input = rest;
    Ok((&header[..4], &body[..len]))
}

fn check_dimensions(w: u32, h: u32) -> Result<()> {
    let max = super::MAX_IMAGE_DIM;
    let max_w = if h <= 128 { 16384 } else { max };
    if w == 0 || h == 0 || w > max_w || h > max {
        return Err(err(format!("Unsupported WebP dimensions {w}×{h}")));
    }
    Ok(())
}

fn apply_alpha(rgba: &mut [u8], w: u32, h: u32, data: &[u8]) -> Result<()> {
    let (&flags, payload) = data
        .split_first()
        .ok_or_else(|| err("Truncated WebP alpha header"))?;
    if (flags & 0x30) >> 4 > 1 {
        return Err(err("Invalid WebP alpha preprocessing mode"));
    }
    let count = w as usize * h as usize;
    let decoded;
    let alpha = match flags & 3 {
        0 if payload.len() == count => payload,
        1 => {
            let mut colors = vec![0; count * 4];
            lossless::LosslessDecoder::new(Cursor::new(payload))
                .decode_frame(w, h, true, &mut colors)
                .map_err(|e| err(format!("Invalid WebP alpha: {e}")))?;
            decoded = colors.chunks_exact(4).map(|c| c[1]).collect::<Vec<_>>();
            &decoded
        }
        _ => return Err(err("Invalid WebP alpha plane")),
    };
    let filter = (flags >> 2) & 3;
    let stride = w as usize * 4;
    for (i, &residual) in alpha.iter().enumerate() {
        let j = 4 * i + 3;
        let x = i % w as usize;
        let y = i / w as usize;
        let predictor = match (filter, x, y) {
            (0, _, _) | (_, 0, 0) => 0,
            (_, _, 0) | (1, 1.., _) => rgba[j - 4],
            (_, 0, _) | (2, _, _) => rgba[j - stride],
            _ => (i16::from(rgba[j - 4]) + i16::from(rgba[j - stride])
                - i16::from(rgba[j - stride - 4]))
            .clamp(0, 255) as u8,
        };
        rgba[j] = predictor.wrapping_add(residual);
    }
    Ok(())
}

fn picture(
    tag: &[u8],
    data: &[u8],
    alpha: Option<&[u8]>,
    expected: Option<(u32, u32)>,
) -> Result<PngImage> {
    let (w, h) = if tag == b"VP8L" {
        if data.len() < 5 || data[0] != 0x2f {
            return Err(err("Invalid WebP lossless frame"));
        }
        let bits = u32::from_le_bytes(data[1..5].try_into().unwrap());
        ((bits & 0x3fff) + 1, ((bits >> 14) & 0x3fff) + 1)
    } else {
        if data.len() < 10 || data[0] & 1 != 0 || data[3..6] != [0x9d, 0x01, 0x2a] {
            return Err(err("Invalid WebP key frame"));
        }
        (
            u32::from(u16::from_le_bytes([data[6], data[7]]) & 0x3fff),
            u32::from(u16::from_le_bytes([data[8], data[9]]) & 0x3fff),
        )
    };
    check_dimensions(w, h)?;
    if expected.is_some_and(|dims| dims != (w, h)) {
        return Err(err("WebP frame dimensions disagree with its container"));
    }
    let mut rgba = vec![0; w as usize * h as usize * 4];
    if tag == b"VP8L" {
        lossless::LosslessDecoder::new(Cursor::new(data))
            .decode_frame(w, h, false, &mut rgba)
            .map_err(|e| err(format!("Invalid WebP lossless frame: {e}")))?;
    } else {
        let frame = vp8::Vp8Decoder::decode_frame(data)
            .map_err(|e| err(format!("Invalid WebP key frame: {e}")))?;
        frame.fill_rgba(&mut rgba);
        if let Some(a) = alpha {
            apply_alpha(&mut rgba, w, h, a)?;
        } else {
            for pixel in rgba.chunks_exact_mut(4) {
                pixel[3] = 255;
            }
        }
    }
    Ok(PngImage {
        width: w,
        height: h,
        rgba,
    })
}

fn animated_picture(frame: &[u8], w: u32, h: u32) -> Result<PngImage> {
    let header = frame
        .get(..16)
        .ok_or_else(|| err("Truncated WebP animation frame header"))?;
    let (x, y) = (u24(header) * 2, u24(&header[3..]) * 2);
    let (fw, fh) = (u24(&header[6..]) + 1, u24(&header[9..]) + 1);
    if x + fw > w || y + fh > h {
        return Err(err("WebP animation frame outside canvas"));
    }
    let mut chunks = &frame[16..];
    let mut alpha = None;
    while !chunks.is_empty() {
        let (tag, payload) = chunk(&mut chunks)?;
        if tag == b"ALPH" {
            alpha = Some(payload);
        }
        if tag != b"VP8L" && tag != b"VP8 " {
            continue;
        }
        let image = picture(tag, payload, alpha, Some((fw, fh)))?;
        if w == fw && h == fh {
            return Ok(image);
        }
        // The browser starts an animation on a transparent canvas. Replacing
        // or blending its first rectangle onto that canvas yields frame RGBA.
        let mut rgba = vec![0; w as usize * h as usize * 4];
        let width = fw as usize * 4;
        for (row, src) in image.rgba.chunks_exact(width).enumerate() {
            let start = ((row + y as usize) * w as usize + x as usize) * 4;
            rgba[start..start + width].copy_from_slice(src);
        }
        return Ok(PngImage {
            width: w,
            height: h,
            rgba,
        });
    }
    Err(err("WebP animation frame is missing pixels"))
}

pub(super) fn decode_webp(raw: &[u8]) -> Result<PngImage> {
    if raw.get(..4) != Some(b"RIFF") || raw.get(8..12) != Some(b"WEBP") {
        return Err(err("Invalid WebP RIFF header"));
    }
    let declared = u32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize;
    let mut chunks = raw[8..]
        .get(4..declared)
        .ok_or_else(|| err("Invalid WebP RIFF length"))?;
    let mut canvas = None;
    let mut alpha = None;
    while !chunks.is_empty() {
        let (tag, payload) = chunk(&mut chunks)?;
        match tag {
            b"VP8X" => {
                let hdr = payload
                    .get(..10)
                    .ok_or_else(|| err("Truncated WebP canvas header"))?;
                let size = (u24(&hdr[4..]) + 1, u24(&hdr[7..]) + 1);
                check_dimensions(size.0, size.1)?;
                canvas = Some(size);
            }
            b"ALPH" => alpha = Some(payload),
            b"VP8 " | b"VP8L" => return picture(tag, payload, alpha, canvas),
            b"ANMF" => {
                let (w, h) = canvas.ok_or_else(|| err("WebP animation has no canvas header"))?;
                return animated_picture(payload, w, h);
            }
            _ => {}
        }
    }
    Err(err("WebP image is missing a frame"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local_media::crc32;

    // Independently decoded RGBA, including the transparent area outside
    // the first animation rectangle and odd-width chroma interpolation.
    #[test]
    fn webp_still_and_first_frame_match_reference() {
        let samples: &[(&[u8], u32, u32, u32)] = &[
            (
                include_bytes!("sprite_webp/fixtures/lossy.webp"),
                41,
                33,
                0x618b7d97,
            ),
            (
                include_bytes!("sprite_webp/fixtures/lossless.webp"),
                37,
                29,
                0x83e52896,
            ),
            (
                include_bytes!("sprite_webp/fixtures/alpha.webp"),
                39,
                35,
                0x56c1c64c,
            ),
            (
                include_bytes!("sprite_webp/fixtures/animated.webp"),
                39,
                27,
                0xb29026ee,
            ),
        ];
        for &(bytes, width, height, checksum) in samples {
            let image = decode_webp(bytes).unwrap();
            assert_eq!((image.width, image.height), (width, height));
            assert_eq!(crc32(&image.rgba), checksum);
        }
    }

    #[test]
    fn rejects_truncated_and_oversized_chunks_before_decoding() {
        let data = include_bytes!("sprite_webp/fixtures/alpha.webp");
        assert!(decode_webp(&data[..data.len() - 1]).is_err());
        let mut invalid = data.to_vec();
        invalid[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode_webp(&invalid).is_err());
        let mut wide = data.to_vec();
        wide[24..27].fill(255);
        assert!(decode_webp(&wide).is_err());
    }

    #[test]
    fn uncompressed_alpha_filters_respect_first_row_and_column() {
        let mut rgba = vec![0; 3 * 2 * 4];
        for (mode, input) in [
            (0, [20, 40, 60, 50, 90, 110]),
            (1, [20, 20, 20, 30, 40, 20]),
            (2, [20, 20, 20, 30, 50, 50]),
            (3, [20, 20, 20, 30, 20, 0]),
        ] {
            let mut plane = vec![mode << 2];
            plane.extend(input);
            apply_alpha(&mut rgba, 3, 2, &plane).unwrap();
            let values: Vec<_> = rgba.chunks_exact(4).map(|c| c[3]).collect();
            assert_eq!(values, [20, 40, 60, 50, 90, 110]);
        }
    }
}
