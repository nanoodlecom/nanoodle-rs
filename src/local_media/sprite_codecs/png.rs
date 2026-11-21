use super::super::zlib_inflate;
use super::{dimensions, err, PngImage, Result};

const ADAM7: [(usize, usize, usize, usize); 7] = [
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
];
const SEQUENTIAL: [(usize, usize, usize, usize); 1] = [(0, 0, 1, 1)];

fn sample(row: &[u8], index: usize, depth: u8) -> u16 {
    match depth {
        8 => row[index] as u16,
        16 => u16::from_be_bytes([row[2 * index], row[2 * index + 1]]),
        _ => {
            let bit = index * depth as usize;
            ((row[bit / 8] >> (8 - depth - (bit % 8) as u8)) & ((1 << depth) - 1)) as u16
        }
    }
}

fn unfilter(row: &mut [u8], previous: &[u8], step: usize, filter: u8) -> Result<()> {
    if filter == 0 {
        return Ok(());
    }
    if filter > 4 {
        return Err(err("Unknown PNG row filter"));
    }
    for x in 0..row.len() {
        let a = if x >= step { row[x - step] } else { 0 };
        let b = previous[x];
        let c = if x >= step { previous[x - step] } else { 0 };
        let pred = match filter {
            1 => a,
            2 => b,
            3 => ((a as u16 + b as u16) / 2) as u8,
            _ => {
                let p = a as i32 + b as i32 - c as i32;
                let (da, db, dc) = (
                    (p - a as i32).abs(),
                    (p - b as i32).abs(),
                    (p - c as i32).abs(),
                );
                if da <= db && da <= dc {
                    a
                } else if db <= dc {
                    b
                } else {
                    c
                }
            }
        };
        row[x] = row[x].wrapping_add(pred);
    }
    Ok(())
}

pub(in crate::local_media) fn read_png(raw: &[u8], allow_strip: bool) -> Result<PngImage> {
    if !raw.starts_with(b"\x89PNG\r\n\x1a\n")
        || raw.get(8..16) != Some(b"\0\0\0\rIHDR")
        || raw.len() < 33
    {
        return Err(err("couldn't read that image to resize"));
    }
    let width = u32::from_be_bytes(raw[16..20].try_into().unwrap()) as usize;
    let height = u32::from_be_bytes(raw[20..24].try_into().unwrap()) as usize;
    dimensions(width, height, allow_strip)?;
    let (depth, color, interlace) = (raw[24], raw[25], raw[28]);
    let channels: usize = match (color, depth) {
        (0, 1 | 2 | 4 | 8 | 16) | (3, 1 | 2 | 4 | 8) => 1,
        (2, 8 | 16) => 3,
        (4, 8 | 16) => 2,
        (6, 8 | 16) => 4,
        _ => return Err(err("Unsupported PNG depth or color type")),
    };
    if raw[26] != 0 || raw[27] != 0 || interlace > 1 {
        return Err(err("Unsupported PNG coding method"));
    }
    let (canvas_width, canvas_height) = (width, height);
    let (mut width, mut height) = (width, height);
    let (mut left, mut top) = (0usize, 0usize);
    let mut animated = false;
    let mut got_frame = false;
    let mut separate_poster = false;
    let mut blend_over = false;
    let mut idat = Vec::new();
    let mut palette = &[][..];
    let mut alpha = &[][..];
    let mut pos = 33usize;
    while let Some(header) = raw.get(pos..pos + 8) {
        let len = u32::from_be_bytes(header[..4].try_into().unwrap()) as usize;
        let end = pos
            .checked_add(12)
            .and_then(|n| n.checked_add(len))
            .filter(|n| *n <= raw.len())
            .ok_or_else(|| err("Truncated PNG chunk"))?;
        let body = &raw[pos + 8..end - 4];
        match &header[4..8] {
            b"acTL" if allow_strip && body.len() == 8 && idat.is_empty() => {
                animated = u32::from_be_bytes(body[..4].try_into().unwrap()) != 0;
            }
            b"fcTL" if animated => {
                if got_frame {
                    break;
                }
                if body.len() != 26 || body[25] > 1 {
                    return Err(err("Invalid APNG frame control"));
                }
                let word =
                    |i: usize| u32::from_be_bytes(body[i..i + 4].try_into().unwrap()) as usize;
                (width, height, left, top) = (word(4), word(8), word(12), word(16));
                if width == 0
                    || height == 0
                    || width > canvas_width
                    || height > canvas_height
                    || left > canvas_width - width
                    || top > canvas_height - height
                {
                    return Err(err("APNG frame lies outside the canvas"));
                }
                // An fcTL after IDAT starts the first animation frame. Those earlier IDAT
                // pixels are a standalone poster and do not participate in composition.
                separate_poster = !idat.is_empty();
                if separate_poster {
                    idat.clear();
                }
                blend_over = body[25] == 1;
                got_frame = true;
            }
            b"fdAT" if got_frame && separate_poster => {
                idat.extend_from_slice(
                    body.get(4..)
                        .ok_or_else(|| err("Truncated APNG frame data"))?,
                );
            }
            b"IDAT" if !separate_poster => idat.extend_from_slice(body),
            b"PLTE" => palette = body,
            b"tRNS" => alpha = body,
            b"IEND" => break,
            _ => (),
        }
        pos = end;
    }
    let passes = if interlace == 0 {
        &SEQUENTIAL[..]
    } else {
        &ADAM7[..]
    };
    let bits_per_pixel = channels * depth as usize;
    let extents = |start: usize, end: usize, step: usize| end.saturating_sub(start).div_ceil(step);
    let row_bytes = |w: usize| (w * bits_per_pixel).div_ceil(8);
    let mut expected = 0usize;
    for &(x, y, dx, dy) in passes {
        let columns = extents(x, width, dx);
        if columns != 0 {
            expected += (1 + row_bytes(columns)) * extents(y, height, dy);
        }
    }
    let mut scan = zlib_inflate(&idat, expected)?;
    if scan.len() != expected {
        return Err(err("Invalid PNG scanline length"));
    }
    let max_sample = (1u32 << depth) - 1;
    let scale = |v: u16| ((u32::from(v) * 255 + max_sample / 2) / max_sample) as u8;
    let transparent: Vec<u16> = alpha
        .chunks_exact(2)
        .map(|b| u16::from_be_bytes([b[0], b[1]]))
        .collect();
    let mut rgba = vec![0u8; canvas_width * canvas_height * 4];
    let mut lines = scan.as_mut_slice();
    for &(x, y, dx, dy) in passes {
        let columns = extents(x, width, dx);
        if columns == 0 {
            continue;
        }
        let stride = row_bytes(columns);
        let mut prev = vec![0u8; stride];
        for row_y in (y..height).step_by(dy) {
            let (line, rest) = lines.split_at_mut(1 + stride);
            lines = rest;
            let (filter, row) = line.split_at_mut(1);
            unfilter(row, &prev, bits_per_pixel.div_ceil(8), filter[0])?;
            prev.copy_from_slice(row);
            if color == 6 && depth == 8 && dx == 1 {
                let out = 4 * ((top + row_y) * canvas_width + left + x);
                rgba[out..out + stride].copy_from_slice(row);
                continue;
            }
            for lane in 0..columns {
                let get = |c| sample(row, channels * lane + c, depth);
                let first = get(0);
                let pixel = match color {
                    0 => [
                        scale(first),
                        scale(first),
                        scale(first),
                        if transparent.first() == Some(&first) {
                            0
                        } else {
                            255
                        },
                    ],
                    2 => {
                        let rgb = [first, get(1), get(2)];
                        [
                            scale(rgb[0]),
                            scale(rgb[1]),
                            scale(rgb[2]),
                            if rgb.as_slice() == transparent {
                                0
                            } else {
                                255
                            },
                        ]
                    }
                    3 => {
                        let i = usize::from(first);
                        let c = palette
                            .get(i * 3..i * 3 + 3)
                            .ok_or_else(|| err("Invalid PNG palette index"))?;
                        [c[0], c[1], c[2], alpha.get(i).copied().unwrap_or(255)]
                    }
                    4 => [scale(first), scale(first), scale(first), scale(get(1))],
                    _ => [scale(first), scale(get(1)), scale(get(2)), scale(get(3))],
                };
                let out = 4 * ((top + row_y) * canvas_width + left + x + lane * dx);
                rgba[out..out + 4].copy_from_slice(&pixel);
            }
        }
    }
    // SOURCE and OVER both draw the first frame onto transparent black. Only
    // hidden RGB from completely transparent OVER pixels must be discarded.
    if blend_over {
        for pixel in rgba.chunks_exact_mut(4) {
            if pixel[3] == 0 {
                pixel.fill(0);
            }
        }
    }
    Ok(PngImage {
        width: canvas_width as u32,
        height: canvas_height as u32,
        rgba,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local_media::{png_chunk, zlib_deflate};

    #[test]
    fn first_animation_frame_uses_its_subrectangle_and_excludes_the_poster() {
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png_chunk(&mut png, b"IHDR", &[0, 0, 0, 3, 0, 0, 0, 1, 8, 6, 0, 0, 0]);
        png_chunk(&mut png, b"acTL", &[0, 0, 0, 1, 0, 0, 0, 0]);
        png_chunk(
            &mut png,
            b"IDAT",
            &zlib_deflate(&[0, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255]),
        );
        let mut control = Vec::new();
        for v in [0u32, 2, 1, 1, 0] {
            control.extend_from_slice(&v.to_be_bytes());
        }
        control.extend_from_slice(&[0, 1, 0, 1, 0, 1]); // duration 1s; source over
        png_chunk(&mut png, b"fcTL", &control);
        let mut frame = 1u32.to_be_bytes().to_vec();
        frame.extend_from_slice(&zlib_deflate(&[0, 0, 255, 0, 128, 255, 0, 0, 0]));
        png_chunk(&mut png, b"fdAT", &frame);
        png_chunk(&mut png, b"IEND", &[]);
        let image = read_png(&png, true).unwrap();
        assert_eq!((image.width, image.height), (3, 1));
        assert_eq!(image.rgba, [0, 0, 0, 0, 0, 255, 0, 128, 0, 0, 0, 0]);
    }
}
