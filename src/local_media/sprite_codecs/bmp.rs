//! Windows and OS/2 bitmaps, including the DIBs embedded in Windows icons.
use super::{dimensions, err, read_png, PngImage, Result};

fn bytes(src: &[u8], start: usize, n: usize) -> Result<&[u8]> {
    src.get(
        start
            ..start
                .checked_add(n)
                .ok_or_else(|| err("Invalid bitmap offset"))?,
    )
    .ok_or_else(|| err("Truncated bitmap data"))
}
fn u16le(src: &[u8], at: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(bytes(src, at, 2)?.try_into().unwrap()))
}
fn u32le(src: &[u8], at: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(bytes(src, at, 4)?.try_into().unwrap()))
}

pub(super) fn decode_file(src: &[u8]) -> Result<PngImage> {
    let offset = (u32le(src, 10)? as usize)
        .checked_sub(14)
        .ok_or_else(|| err("Invalid BMP pixel offset"))?;
    decode_dib(bytes(src, 14, src.len().saturating_sub(14))?, Some(offset))
}

pub(super) fn decode_icon(src: &[u8]) -> Result<PngImage> {
    let count = u16le(src, 4)? as usize;
    let cursor = src.get(2) == Some(&2);
    let directory = bytes(src, 6, 16 * count)?;
    // Prefer the largest source, breaking ties by bit depth. Decode the next
    // entry if a damaged or unknown icon encoding has a valid alternative.
    let mut entries: Vec<_> = directory.chunks_exact(16).collect();
    entries.sort_by_key(|entry| {
        let side = |n: u8| if n == 0 { 256u32 } else { n as u32 };
        let bits = if cursor {
            0
        } else {
            u16::from_le_bytes([entry[6], entry[7]])
        };
        std::cmp::Reverse((side(entry[0]) * side(entry[1]), bits))
    });
    for entry in entries {
        let offset = u32le(entry, 12)? as usize;
        let size = u32le(entry, 8)? as usize;
        if offset < 6 + directory.len() {
            continue;
        }
        if let Ok(data) = bytes(src, offset, size) {
            let image = if data.starts_with(b"\x89PNG\r\n\x1a\n") {
                read_png(data, true)
            } else {
                decode_dib(data, None)
            };
            if image.is_ok() {
                return image;
            }
        }
    }
    Err(err("Icon contains no supported, complete image"))
}

struct Dib<'a> {
    width: usize,
    height: usize,
    top_down: bool,
    bits: u16,
    compression: u32,
    header_end: usize,
    palette: &'a [u8],
    entry_bytes: usize,
    masks: [u32; 4],
}

impl<'a> Dib<'a> {
    fn parse(src: &'a [u8], icon: bool) -> Result<Self> {
        let size = u32le(src, 0)? as usize;
        if !matches!(size, 12 | 16 | 40 | 52 | 56 | 64 | 108 | 124) {
            return Err(err("Unsupported bitmap header"));
        }
        let hdr = bytes(src, 0, size)?;
        let core = size == 12;
        let (width, height, planes, bits) = if core {
            (
                i32::from(u16le(hdr, 4)?),
                i32::from(u16le(hdr, 6)?),
                u16le(hdr, 8)?,
                u16le(hdr, 10)?,
            )
        } else {
            (
                u32le(hdr, 4)? as i32,
                u32le(hdr, 8)? as i32,
                u16le(hdr, 12)?,
                u16le(hdr, 14)?,
            )
        };
        let top_down = height < 0;
        let signed_height = height;
        let height = height.unsigned_abs() as usize;
        if width < 1
            || height == 0
            || planes != 1
            || (icon && (signed_height < 0 || height % 2 != 0))
        {
            return Err(err("Invalid bitmap dimensions"));
        }
        let width = width as usize;
        let height = if icon { height / 2 } else { height };
        dimensions(width, height, true)?;
        let compression = if size >= 40 { u32le(hdr, 16)? } else { 0 };
        let os2 = size == 64 || size == 16 || core;
        let supported = match compression {
            0 => matches!(bits, 1 | 2 | 4 | 8 | 16 | 24 | 32),
            1 => bits == 8 && !top_down,
            2 => bits == 4 && !top_down,
            3 | 6 => !os2 && matches!(bits, 16 | 32),
            _ => false,
        };
        if !supported {
            return Err(err("Unsupported bitmap compression or pixel depth"));
        }
        let mut header_end = size;
        let masks = if (compression == 3 || compression == 6) && !os2 {
            let n = if size >= 56 || compression == 6 { 4 } else { 3 };
            if size < 40 + n * 4 {
                header_end = 40 + n * 4;
            }
            let mut masks = [0; 4];
            let mut used = 0;
            for (i, mask) in masks.iter_mut().take(n).enumerate() {
                *mask = u32le(src, 40 + 4 * i)?;
                if i == 3 && *mask == 0 {
                    continue;
                }
                let trimmed = *mask >> mask.trailing_zeros().min(31);
                if *mask == 0
                    || used & *mask != 0
                    || (bits < 32 && *mask >> bits != 0)
                    || trimmed & trimmed.wrapping_add(1) != 0
                {
                    return Err(err("Invalid bitmap component masks"));
                }
                used |= *mask;
            }
            masks
        } else if bits == 16 {
            [0x7c00, 0x03e0, 0x001f, 0]
        } else {
            [0xff0000, 0xff00, 0xff, 0]
        };
        let colors = if size >= 40 {
            u32le(hdr, 32)? as usize
        } else {
            0
        };
        let colors = if bits <= 8 {
            if colors > 1 << bits {
                return Err(err("Invalid bitmap palette size"));
            }
            if colors == 0 {
                1 << bits
            } else {
                colors
            }
        } else {
            colors
        };
        let entry_bytes = if core { 3 } else { 4 };
        let pal_bytes = colors
            .checked_mul(entry_bytes)
            .ok_or_else(|| err("Invalid bitmap palette size"))?;
        let palette = bytes(src, header_end, pal_bytes)?;
        Ok(Self {
            width,
            height,
            top_down,
            bits,
            compression,
            header_end: header_end + pal_bytes,
            palette,
            entry_bytes,
            masks,
        })
    }

    fn color(&self, index: usize) -> Result<[u8; 4]> {
        let p = bytes(self.palette, self.entry_bytes * index, 3)?;
        Ok([p[2], p[1], p[0], 255])
    }

    fn unpack(&self, row: &[u8], x: usize, icon: bool) -> Result<[u8; 4]> {
        let bits = self.bits as usize;
        if bits <= 8 {
            let pos = x * bits;
            let index = (usize::from(row[pos / 8]) >> (8 - bits - pos % 8)) & ((1 << bits) - 1);
            return self.color(index);
        }
        let pos = x * (bits / 8);
        if bits == 24 || (bits == 32 && self.compression == 0) {
            return Ok([
                row[pos + 2],
                row[pos + 1],
                row[pos],
                if icon && bits == 32 {
                    row[pos + 3]
                } else {
                    255
                },
            ]);
        }
        let value = if bits == 16 {
            u16le(row, pos)? as u32
        } else {
            u32le(row, pos)?
        };
        let channel = |mask: u32| {
            if mask == 0 {
                return 255;
            }
            let shift = mask.trailing_zeros();
            let max = (mask >> shift) as u64;
            ((u64::from((value & mask) >> shift) * 255 + max / 2) / max) as u8
        };
        Ok(self.masks.map(channel))
    }
}

/// RLE4/RLE8 use a bottom-up pixel cursor, with explicit row breaks, deltas
/// and word-aligned literal runs. Return the byte after the end marker.
fn rle(info: &Dib, data: &[u8], rgba: &mut [u8]) -> Result<usize> {
    let (mut x, mut y, mut pos) = (0usize, 0usize, 0usize);
    let background = info.color(0)?;
    for pixel in rgba.chunks_exact_mut(4) {
        pixel.copy_from_slice(&background);
    }
    loop {
        let pair = bytes(data, pos, 2)?;
        pos += 2;
        let n = usize::from(pair[0]);
        let value = usize::from(pair[1]);
        if n == 0 {
            match value {
                0 => {
                    x = 0;
                    y += 1;
                    continue;
                }
                1 => return Ok(pos),
                2 => {
                    let delta = bytes(data, pos, 2)?;
                    pos += 2;
                    x += usize::from(delta[0]);
                    y += usize::from(delta[1]);
                    if x > info.width || y >= info.height {
                        return Err(err("BMP RLE delta outside image"));
                    }
                    continue;
                }
                _ => (),
            }
        }
        let len = if n == 0 { value } else { n };
        if y >= info.height || x + len > info.width {
            return Err(err("BMP RLE run outside image"));
        }
        let literal = if n == 0 {
            let size = if info.bits == 4 { len.div_ceil(2) } else { len };
            let chunk = bytes(data, pos, size)?;
            pos += size.next_multiple_of(2);
            chunk
        } else {
            &[]
        };
        for i in 0..len {
            let byte = if n > 0 {
                value
            } else if info.bits == 4 {
                usize::from(literal[i / 2])
            } else {
                usize::from(literal[i])
            };
            let index = if info.bits == 4 {
                (byte >> (if i % 2 == 0 { 4 } else { 0 })) & 15
            } else {
                byte
            };
            let out = 4 * ((info.height - 1 - y) * info.width + x + i);
            rgba[out..out + 4].copy_from_slice(&info.color(index)?);
        }
        x += len;
    }
}

fn decode_dib(src: &[u8], pixel_offset: Option<usize>) -> Result<PngImage> {
    let icon = pixel_offset.is_none();
    let info = Dib::parse(src, icon)?;
    let offset = pixel_offset.unwrap_or(info.header_end);
    if offset < info.header_end || offset > src.len() {
        return Err(err("Invalid bitmap pixel offset"));
    }
    let w = info.width;
    let h = info.height;
    let data = &src[offset..];
    let mut rgba = vec![0; w * h * 4];
    let used = if info.compression == 1 || info.compression == 2 {
        rle(&info, data, &mut rgba)?
    } else {
        let stride = (w * info.bits as usize).div_ceil(32) * 4;
        let total = stride * h;
        let packed = bytes(data, 0, total)?;
        for (i, row) in packed.chunks_exact(stride).enumerate() {
            let y = if info.top_down { i } else { h - 1 - i };
            for x in 0..w {
                let p = 4 * (y * w + x);
                rgba[p..p + 4].copy_from_slice(&info.unpack(row, x, icon)?);
            }
        }
        total
    };
    if icon {
        let bgra = info.bits == 32 && (info.compression == 0 || info.masks[3] != 0);
        if !bgra || rgba.chunks_exact(4).all(|p| p[3] == 0) {
            let stride = w.div_ceil(32) * 4;
            // Modern icons can omit the legacy mask when alpha is present.
            let mask = bytes(data, used, stride * h)?;
            for y in 0..h {
                for x in 0..w {
                    let hidden = mask[(h - 1 - y) * stride + x / 8] & (0x80 >> (x % 8)) != 0;
                    rgba[(y * w + x) * 4 + 3] = if hidden { 0 } else { 255 };
                }
            }
        }
    }
    Ok(PngImage {
        width: w as u32,
        height: h as u32,
        rgba,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn header(w: i32, h: i32, bits: u16, format: u32) -> Vec<u8> {
        let mut b = vec![0; 40];
        b[0..4].copy_from_slice(&40u32.to_le_bytes());
        b[4..8].copy_from_slice(&w.to_le_bytes());
        b[8..12].copy_from_slice(&h.to_le_bytes());
        b[12..14].copy_from_slice(&1u16.to_le_bytes());
        b[14..16].copy_from_slice(&bits.to_le_bytes());
        b[16..20].copy_from_slice(&format.to_le_bytes());
        b
    }
    fn file(dib: &[u8], offset: u32) -> Vec<u8> {
        let mut b = vec![0; 14];
        b[0..2].copy_from_slice(b"BM");
        b[2..6].copy_from_slice(&(14 + dib.len() as u32).to_le_bytes());
        b[10..14].copy_from_slice(&(14 + offset).to_le_bytes());
        b.extend_from_slice(dib);
        b
    }
    fn icon(dib: &[u8], width: u8, height: u8, bits: u16) -> Vec<u8> {
        let mut b = vec![0, 0, 1, 0, 1, 0, width, height, 0, 0, 1, 0];
        b.extend_from_slice(&bits.to_le_bytes());
        b.extend_from_slice(&(dib.len() as u32).to_le_bytes());
        b.extend_from_slice(&22u32.to_le_bytes());
        b.extend_from_slice(dib);
        b
    }

    #[test]
    fn core_palette_and_top_down_565_masks() {
        let mut dib = vec![12, 0, 0, 0, 2, 0, 1, 0, 1, 0, 1, 0];
        dib.extend_from_slice(&[0, 0, 255, 0, 255, 0, 0x80, 0, 0, 0]);
        assert_eq!(
            decode_file(&file(&dib, 18)).unwrap().rgba,
            [0, 255, 0, 255, 255, 0, 0, 255]
        );
        let mut dib = header(1, -2, 16, 3);
        for mask in [0xf800u32, 0x07e0, 0x001f] {
            dib.extend_from_slice(&mask.to_le_bytes());
        }
        dib.extend_from_slice(&[0, 248, 0, 0, 31, 0, 0, 0]);
        assert_eq!(
            decode_file(&file(&dib, 52)).unwrap().rgba,
            [255, 0, 0, 255, 0, 0, 255, 255]
        );
    }

    #[test]
    fn rle4_literal_and_run() {
        let mut dib = header(7, 1, 4, 2);
        for i in 0..16 {
            dib.extend_from_slice(&[0, 0, i, 0]);
        }
        // Three alternating indices then four packed literals.
        dib.extend_from_slice(&[3, 0x12, 0, 4, 0x34, 0x56, 0, 1]);
        let image = decode_file(&file(&dib, 104)).unwrap();
        assert_eq!(
            image.rgba.chunks_exact(4).map(|p| p[0]).collect::<Vec<_>>(),
            [1, 2, 1, 3, 4, 5, 6]
        );
    }

    #[test]
    fn icons_use_alpha_or_the_legacy_and_mask() {
        let mut dib = header(2, 2, 32, 0);
        dib.extend_from_slice(&[6, 5, 4, 0, 9, 8, 7, 0, 0x80, 0, 0, 0]);
        assert_eq!(
            decode_icon(&icon(&dib, 2, 1, 32)).unwrap().rgba,
            [4, 5, 6, 0, 7, 8, 9, 255]
        );
        dib[43] = 99;
        assert_eq!(
            decode_icon(&icon(&dib, 2, 1, 32)).unwrap().rgba,
            [4, 5, 6, 99, 7, 8, 9, 0]
        );
        let png = super::super::super::encode_png_rgba(&PngImage {
            width: 1,
            height: 1,
            rgba: vec![4, 5, 6, 7],
        });
        assert_eq!(
            decode_icon(&icon(&png, 1, 1, 32)).unwrap().rgba,
            [4, 5, 6, 7]
        );
    }

    #[test]
    fn malformed_offsets_and_rle_runs_return_errors() {
        assert!(decode_file(&file(&header(10, 1, 4, 0), 0)).is_err());
        let mut b = file(&header(10, 1, 32, 0), 40);
        b[10..14].copy_from_slice(&0xffffffffu32.to_le_bytes());
        assert!(decode_file(&b).is_err());
        let mut dib = header(3, 1, 8, 1);
        dib[32..36].copy_from_slice(&1u32.to_le_bytes());
        dib.extend_from_slice(&[0, 0, 0, 0, 8, 0, 0, 1]);
        assert!(decode_file(&file(&dib, 44)).is_err());
    }
}
