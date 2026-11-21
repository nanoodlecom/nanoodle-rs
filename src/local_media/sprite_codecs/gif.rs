//! Decode the first displayed GIF frame for a static palette or texture.
use super::{dimensions, err, PngImage, Result};

fn take<'a>(src: &mut &'a [u8], size: usize) -> Result<&'a [u8]> {
    let (value, rest) = src
        .split_at_checked(size)
        .ok_or_else(|| err("Truncated GIF"))?;
    *src = rest;
    Ok(value)
}

fn table<'a>(src: &mut &'a [u8], packed: u8) -> Result<&'a [u8]> {
    if packed & 128 == 0 {
        return Ok(&[]);
    }
    take(src, 3 * (2usize << (packed & 7)))
}

fn subblocks(src: &mut &[u8]) -> Result<Vec<u8>> {
    let mut data = vec![];
    loop {
        let size = take(src, 1)?[0] as usize;
        if size == 0 {
            return Ok(data);
        }
        data.extend_from_slice(take(src, size)?);
    }
}

fn indices(data: &[u8], min_size: u8, count: usize) -> Result<Vec<u8>> {
    if !(2..=8).contains(&min_size) {
        return Err(err("Invalid GIF LZW size"));
    }
    let clear = 1usize << min_size;
    let end = clear + 1;
    let mut prefix = [0u16; 4096];
    let mut suffix = [0u8; 4096];
    let mut stack = [0u8; 4096];
    let mut bits = min_size as usize + 1;
    let mut next = end + 1;
    let mut last = None;
    let mut first = 0;
    let mut pos = 0;
    let mut out = Vec::with_capacity(count);
    while pos + bits <= data.len() * 8 {
        let byte = pos / 8;
        let word = u32::from(data[byte])
            | (u32::from(data.get(byte + 1).copied().unwrap_or(0)) << 8)
            | (u32::from(data.get(byte + 2).copied().unwrap_or(0)) << 16);
        let code = (word as usize >> (pos % 8)) & ((1usize << bits) - 1);
        pos += bits;
        if code == clear {
            bits = min_size as usize + 1;
            next = end + 1;
            last = None;
            continue;
        }
        if code == end {
            break;
        }
        let mut length = 0;
        let mut head = code;
        if head == next {
            head = last.ok_or_else(|| err("Invalid first GIF code"))?;
            stack[length] = first;
            length += 1;
        }
        if head >= next {
            return Err(err("Invalid GIF LZW reference"));
        }
        while head >= end + 1 {
            if length >= stack.len() - 1 {
                return Err(err("Invalid GIF LZW chain"));
            }
            stack[length] = suffix[head];
            length += 1;
            head = usize::from(prefix[head]);
        }
        first = head as u8;
        stack[length] = first;
        length += 1;
        if out.len() + length > count {
            return Err(err("GIF frame exceeds declared dimensions"));
        }
        out.extend(stack[..length].iter().rev().copied());
        if let Some(previous) = last {
            if next < 4096 {
                prefix[next] = previous as u16;
                suffix[next] = first;
                next += 1;
                if next == 1 << bits && bits < 12 {
                    bits += 1;
                }
            }
        }
        last = Some(code);
    }
    if out.len() != count {
        return Err(err("Truncated GIF pixels"));
    }
    Ok(out)
}

fn u16le(raw: &[u8]) -> usize {
    u16::from_le_bytes([raw[0], raw[1]]) as usize
}

pub(super) fn decode(raw: &[u8]) -> Result<PngImage> {
    let mut src = &raw[6..];
    let header = take(&mut src, 7)?;
    let (width, height) = (u16le(header), u16le(&header[2..]));
    dimensions(width, height, true)?;
    let palette = table(&mut src, header[4])?;
    let mut transparent = None;
    loop {
        match take(&mut src, 1)?[0] {
            0x21 => {
                let label = take(&mut src, 1)?[0];
                let payload = subblocks(&mut src)?;
                if label == 0xf9 && payload.len() == 4 {
                    transparent = (payload[0] & 1 != 0).then_some(payload[3]);
                }
            }
            0x2c => {
                let desc = take(&mut src, 9)?;
                let (x, y, w, h) = (
                    u16le(desc),
                    u16le(&desc[2..]),
                    u16le(&desc[4..]),
                    u16le(&desc[6..]),
                );
                if w == 0 || h == 0 || x + w > width || y + h > height {
                    return Err(err("Invalid GIF frame bounds"));
                }
                let local = table(&mut src, desc[8])?;
                let color = if local.is_empty() { palette } else { local };
                let min_size = take(&mut src, 1)?[0];
                let pixels = indices(&subblocks(&mut src)?, min_size, w * h)?;
                let mut rgba = vec![0u8; width * height * 4];
                if transparent.is_none() {
                    let off = usize::from(header[5]) * 3;
                    if let Some(bg) = palette.get(off..off + 3) {
                        for p in rgba.chunks_exact_mut(4) {
                            p.copy_from_slice(&[bg[0], bg[1], bg[2], 255]);
                        }
                    }
                }
                let mut offset = 0;
                let passes: &[(usize, usize)] = if desc[8] & 64 != 0 {
                    &[(0, 8), (4, 8), (2, 4), (1, 2)]
                } else {
                    &[(0, 1)]
                };
                for &(start, step) in passes {
                    for row in (start..h).step_by(step) {
                        for col in 0..w {
                            let index = pixels[offset];
                            offset += 1;
                            if transparent == Some(index) {
                                continue;
                            }
                            let i = usize::from(index) * 3;
                            let data = color
                                .get(i..i + 3)
                                .ok_or_else(|| err("Invalid GIF color index"))?;
                            let dest = 4 * ((y + row) * width + x + col);
                            rgba[dest..dest + 4].copy_from_slice(&[data[0], data[1], data[2], 255]);
                        }
                    }
                }
                return Ok(PngImage {
                    width: width as u32,
                    height: height as u32,
                    rgba,
                });
            }
            _ => return Err(err("GIF contains no decodable image")),
        }
    }
}
