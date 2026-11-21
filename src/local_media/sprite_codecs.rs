//! Local codecs used for sprite textures, atlas images and turntables.
use super::{err, PngImage};
use crate::Result;

mod bmp;
mod gif;
mod png;
mod video;

pub(super) use png::read_png;
pub(super) use video::encode_turntable;

pub(super) fn decode_image(raw: &[u8]) -> Result<PngImage> {
    if raw.starts_with(b"\x89PNG\r\n\x1a\n") {
        read_png(raw, true)
    } else if raw.starts_with(&[0xff, 0xd8]) {
        super::sprite_jpeg::decode_jpeg(raw)
    } else if raw.starts_with(b"RIFF") && raw.get(8..12) == Some(b"WEBP") {
        super::sprite_webp::decode_webp(raw)
    } else if raw.starts_with(b"GIF87a") || raw.starts_with(b"GIF89a") {
        gif::decode(raw)
    } else if raw.starts_with(b"BM") {
        bmp::decode_file(raw)
    } else if raw.starts_with(&[0, 0, 1, 0]) || raw.starts_with(&[0, 0, 2, 0]) {
        bmp::decode_icon(raw)
    } else {
        Err(err(
            "Unsupported sprite image format; use PNG, JPEG, GIF, WebP, BMP or ICO.",
        ))
    }
}

fn dimensions(width: usize, height: usize, allow_strip: bool) -> Result<()> {
    let max_width = if allow_strip && height <= 128 {
        128 * 128
    } else {
        super::MAX_IMAGE_DIM as usize
    };
    if width == 0 || height == 0 || width > max_width || height > super::MAX_IMAGE_DIM as usize {
        return Err(err(format!(
            "unsupported image dimensions {width}×{height}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local_media::{encode_png_rgba, parse_mp4, png_chunk, zlib_deflate};

    #[test]
    fn widest_slice_sheet_round_trips() {
        let width = 16384;
        let height = 2;
        let rgba: Vec<u8> = (0..width * height)
            .flat_map(|i| {
                let value = ((i >> 8) ^ i) as u8;
                [value, value / 2, 0, 255]
            })
            .collect();
        let image = PngImage {
            width,
            height,
            rgba,
        };
        let mut png = encode_png_rgba(&image);
        assert_eq!(decode_image(&png).unwrap().rgba, image.rgba);
        assert!(read_png(&png, false).is_err());
        png.truncate(35);
        assert!(decode_image(&png).is_err());
    }

    #[test]
    fn transparent_packed_indices_across_adam7_passes() {
        // Coordinates of the three populated Adam7 passes in a 2×2 image:
        // (0,0), (1,0), (0,1)+(1,1). One-bit palette values are MSB first.
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png_chunk(&mut png, b"IHDR", &[0, 0, 0, 2, 0, 0, 0, 2, 1, 3, 0, 0, 1]);
        png_chunk(&mut png, b"PLTE", &[0, 0, 0, 255, 0, 42]);
        png_chunk(&mut png, b"tRNS", &[0, 200]);
        png_chunk(&mut png, b"IDAT", &zlib_deflate(&[0, 128, 0, 0, 0, 64]));
        png_chunk(&mut png, b"IEND", &[]);
        let image = decode_image(&png).unwrap();
        assert_eq!(
            image.rgba,
            [255, 0, 42, 200, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 42, 200]
        );
    }

    #[test]
    fn streamed_video_records_crops_keyframes_and_exact_duration() {
        let movie = encode_turntable(18, 20, 65, 2.5, |i| {
            let mut rgba = vec![0; 18 * 20 * 4];
            let offset = (i % 18 * 4) as usize;
            rgba[offset..offset + 4].copy_from_slice(&[255, 0, 0, 180]);
            PngImage {
                width: 18,
                height: 20,
                rgba,
            }
        })
        .unwrap();
        let tracks = parse_mp4(&movie).unwrap();
        let track = &tracks[0];
        assert_eq!(
            (track.width, track.height, track.samples.len()),
            (18, 20, 65)
        );
        assert_eq!(track.timescale, 90000);
        assert_eq!(track.samples.iter().map(|s| s.dur).sum::<u32>(), 225000);
        let keys: Vec<_> = track
            .samples
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.sync.then_some(i))
            .collect();
        assert_eq!(keys, [0, 60]);
        for sample in &track.samples {
            let bytes = &movie[sample.offset..sample.offset + sample.size];
            assert_eq!(
                u32::from_be_bytes(bytes[..4].try_into().unwrap()) as usize,
                bytes.len() - 4
            );
            assert_eq!(bytes[4], if sample.sync { 0x65 } else { 0x41 });
        }
    }
}
