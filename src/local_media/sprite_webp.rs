//! Decode the first displayed WebP frame directly into the sprite canvas.
use super::{err, sprite_codecs, PngImage};
use crate::Result;
use libwebp_sys as webp;
use std::{mem::MaybeUninit, ptr};

pub(super) fn decode_webp(raw: &[u8]) -> Result<PngImage> {
    let data = webp::WebPData {
        bytes: raw.as_ptr(),
        size: raw.len(),
    };
    // Full demux validates RIFF chunks and frame dimensions without allocating
    // image pixels. The input stays borrowed until the demuxer is destroyed.
    unsafe {
        let demux =
            webp::WebPDemuxInternal(&data, 0, ptr::null_mut(), webp::WebPGetDemuxABIVersion());
        if demux.is_null() {
            return Err(err("Invalid WebP container"));
        }
        let mut frame = MaybeUninit::uninit();
        let result = if webp::WebPDemuxGetFrame(demux, 1, frame.as_mut_ptr()) == 0 {
            Err(err("WebP image is missing a frame"))
        } else {
            use webp::WebPFormatFeature::{WEBP_FF_CANVAS_HEIGHT, WEBP_FF_CANVAS_WIDTH};
            let width = webp::WebPDemuxGetI(demux, WEBP_FF_CANVAS_WIDTH);
            let height = webp::WebPDemuxGetI(demux, WEBP_FF_CANVAS_HEIGHT);
            // A successful GetFrame initializes the iterator. Its fragment
            // remains valid until the iterator and demuxer are released.
            let mut frame = frame.assume_init();
            let result = decode_frame(width, height, &frame);
            webp::WebPDemuxReleaseIterator(&mut frame);
            result
        };
        webp::WebPDemuxDelete(demux);
        result
    }
}

/// The frame must belong to a live libwebp demuxer and iterator.
unsafe fn decode_frame(width: u32, height: u32, frame: &webp::WebPIterator) -> Result<PngImage> {
    sprite_codecs::dimensions(width as usize, height as usize, true)?;
    let (x, y, w, h) = (frame.x_offset, frame.y_offset, frame.width, frame.height);
    if x < 0
        || y < 0
        || w <= 0
        || h <= 0
        || i64::from(x) + i64::from(w) > i64::from(width)
        || i64::from(y) + i64::from(h) > i64::from(height)
    {
        return Err(err("WebP first frame exceeds the canvas"));
    }
    let stride = width as usize * 4;
    let offset = y as usize * stride + x as usize * 4;
    let mut rgba = vec![0; stride * height as usize];
    // Frame one is drawn on a transparent canvas regardless of the animation
    // background hint. The stride places each row at its final canvas offset.
    let target = &mut rgba[offset..];
    if unsafe {
        webp::WebPDecodeRGBAInto(
            frame.fragment.bytes,
            frame.fragment.size,
            target.as_mut_ptr(),
            target.len(),
            stride as i32,
        )
    }
    .is_null()
    {
        return Err(err("Invalid WebP frame pixels"));
    }
    Ok(PngImage {
        width,
        height,
        rgba,
    })
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
        // A 3 × 2 VP8 keyframe with a separately filtered ALPH plane.
        let keyframe = b"VP8 \x28\0\0\0\x70\x01\0\x9d\x01\x2a\x03\0\x02\0\x02\x80\x42\x25\xa0\x02\x74\x01\x40\0\0\xfe\xef\xdb\xd8\xa1\xff\xa8\x33\xff\xf5\x06\x7f\xfe\xa0\xcf\xd3\xc0\0\0";
        for (mode, input) in [
            (0, [20, 40, 60, 50, 90, 110]),
            (1, [20, 20, 20, 30, 40, 20]),
            (2, [20, 20, 20, 30, 50, 50]),
            (3, [20, 20, 20, 30, 20, 0]),
        ] {
            let mut raw =
                b"RIFF\0\0\0\0WEBPVP8X\x0a\0\0\0\x10\0\0\0\x02\0\0\x01\0\0ALPH\x07\0\0\0".to_vec();
            raw.push(mode << 2);
            raw.extend(input);
            raw.push(0); // RIFF padding
            raw.extend(keyframe);
            let size = (raw.len() - 8) as u32;
            raw[4..8].copy_from_slice(&size.to_le_bytes());
            let image = decode_webp(&raw).unwrap();
            let values: Vec<_> = image.rgba.chunks_exact(4).map(|c| c[3]).collect();
            assert_eq!(values, [20, 40, 60, 50, 90, 110]);
        }
    }
}
