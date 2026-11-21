//! Match browser bitmap orientation for JPEG APP1 and PNG eXIf metadata.
use super::PngImage;

fn tiff_orientation(tiff: &[u8]) -> Option<u8> {
    let little = match tiff.get(..2)? {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let word = |buf: &[u8]| -> Option<u16> {
        let bytes = buf.get(..2)?.try_into().ok()?;
        Some(if little {
            u16::from_le_bytes(bytes)
        } else {
            u16::from_be_bytes(bytes)
        })
    };
    let dword = |buf: &[u8]| -> Option<u32> {
        let bytes = buf.get(..4)?.try_into().ok()?;
        Some(if little {
            u32::from_le_bytes(bytes)
        } else {
            u32::from_be_bytes(bytes)
        })
    };
    if word(tiff.get(2..)?)? != 42 {
        return None;
    }
    let offset = usize::try_from(dword(tiff.get(4..)?)?).ok()?;
    if offset < 8 {
        return None;
    }
    let ifd = tiff.get(offset..)?;
    let count = usize::from(word(ifd)?);
    for entry in ifd.get(2..)?.get(..count * 12)?.chunks_exact(12) {
        // IFD0 Orientation is a single inline SHORT. Offsets to thumbnails
        // and sub-IFDs describe other images and must not affect this bitmap.
        if word(entry)? == 0x0112 && word(&entry[2..])? == 3 && dword(&entry[4..])? == 1 {
            let value = word(&entry[8..])?;
            return (1..=8).contains(&value).then_some(value as u8);
        }
    }
    None
}

fn orientation(raw: &[u8]) -> Option<u8> {
    if let Some(mut rest) = raw.strip_prefix(b"\xff\xd8") {
        while let Some((&0xff, mut marker)) = rest.split_first() {
            while marker.first() == Some(&0xff) {
                marker = &marker[1..];
            }
            let (&id, body) = marker.split_first()?;
            match id {
                0xd9 | 0xda => break,
                0x01 | 0xd0..=0xd7 => {
                    rest = body;
                    continue;
                }
                _ => (),
            }
            let len = usize::from(u16::from_be_bytes(body.get(..2)?.try_into().ok()?));
            let payload = body.get(2..len)?;
            if id == 0xe1 {
                if let Some(tag) = payload.strip_prefix(b"Exif\0\0").and_then(tiff_orientation) {
                    return Some(tag);
                }
            }
            rest = body.get(len..)?;
        }
    } else if let Some(mut chunks) = raw.strip_prefix(b"\x89PNG\r\n\x1a\n") {
        while chunks.len() >= 12 {
            let n = usize::try_from(u32::from_be_bytes(chunks[..4].try_into().ok()?)).ok()?;
            let body = &chunks[8..];
            if body.len() - 4 < n {
                return None;
            }
            match &chunks[4..8] {
                b"eXIf" => return tiff_orientation(&body[..n]),
                b"IEND" => break,
                _ => chunks = &body[n + 4..],
            }
        }
    }
    None
}

pub(super) fn apply(raw: &[u8], img: PngImage) -> PngImage {
    orient(img, orientation(raw).unwrap_or(1))
}

fn orient(mut img: PngImage, mode: u8) -> PngImage {
    if mode <= 1 {
        return img;
    }
    let (w, h) = (img.width as usize, img.height as usize);
    let (pixels, _) = img.rgba.as_chunks_mut::<4>();
    match mode {
        2 => {
            for row in pixels.chunks_exact_mut(w) {
                row.reverse();
            }
        }
        3 => pixels.reverse(),
        4 => {
            for y in 0..h / 2 {
                let (top, bottom) = pixels.split_at_mut((h - y - 1) * w);
                top[y * w..(y + 1) * w].swap_with_slice(&mut bottom[..w]);
            }
        }
        5..=8 => {
            let mut dst = vec![0; img.rgba.len()];
            let (dst_pixels, _) = dst.as_chunks_mut::<4>();
            let mirror_x = mode == 6 || mode == 7;
            let mirror_y = mode == 7 || mode == 8;
            for (y, row) in img.rgba.chunks_exact(w * 4).enumerate() {
                let dx = if mirror_x { h - 1 - y } else { y };
                for (x, pixel) in row.as_chunks::<4>().0.iter().enumerate() {
                    let dy = if mirror_y { w - 1 - x } else { x };
                    dst_pixels[dy * h + dx] = *pixel;
                }
            }
            return PngImage {
                width: img.height,
                height: img.width,
                rgba: dst,
            };
        }
        _ => (),
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiff(mode: u8, little: bool) -> Vec<u8> {
        let word = |v: u16| {
            if little {
                v.to_le_bytes()
            } else {
                v.to_be_bytes()
            }
        };
        let dword = |v: u32| {
            if little {
                v.to_le_bytes()
            } else {
                v.to_be_bytes()
            }
        };
        let mut tiff = if little {
            b"II".to_vec()
        } else {
            b"MM".to_vec()
        };
        tiff.extend(word(42));
        tiff.extend(dword(8));
        tiff.extend(word(1));
        tiff.extend(word(0x112));
        tiff.extend(word(3));
        tiff.extend(dword(1));
        tiff.extend(word(u16::from(mode)));
        tiff.extend([0; 6]);
        tiff
    }

    #[test]
    fn jpeg_and_png_orientation_in_each_byte_order() {
        let expected: [[u8; 6]; 8] = [
            [1, 2, 3, 4, 5, 6],
            [3, 2, 1, 6, 5, 4],
            [6, 5, 4, 3, 2, 1],
            [4, 5, 6, 1, 2, 3],
            [1, 4, 2, 5, 3, 6],
            [4, 1, 5, 2, 6, 3],
            [6, 3, 5, 2, 4, 1],
            [3, 6, 2, 5, 1, 4],
        ];
        for mode in 1..=8 {
            for little in [false, true] {
                let data = tiff(mode, little);
                let mut jpg = b"\xff\xd8\xff\xe1".to_vec();
                jpg.extend(((data.len() + 8) as u16).to_be_bytes());
                jpg.extend(b"Exif\0\0");
                jpg.extend(&data);
                let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
                png.extend((data.len() as u32).to_be_bytes());
                png.extend(b"eXIf");
                png.extend(&data);
                png.extend([0; 4]);
                for bytes in [jpg, png] {
                    let img = PngImage {
                        width: 3,
                        height: 2,
                        rgba: (1..=6).flat_map(|v| [v; 4]).collect(),
                    };
                    let out = apply(&bytes, img);
                    assert_eq!(
                        (out.width, out.height),
                        if mode < 5 { (3, 2) } else { (2, 3) }
                    );
                    let recovered: Vec<_> = out.rgba.chunks_exact(4).map(|p| p[0]).collect();
                    assert_eq!(recovered, expected[usize::from(mode) - 1]);
                }
            }
        }
    }

    #[test]
    fn invalid_directory_or_value_is_ignored() {
        assert_eq!(tiff_orientation(&tiff(9, true)), None);
        let mut dir = tiff(6, false);
        dir[4..8].fill(255);
        assert_eq!(tiff_orientation(&dir), None);
        for length in 0..22 {
            assert_eq!(tiff_orientation(&tiff(6, true)[..length]), None);
        }
    }
}
