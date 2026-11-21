//! AVC Baseline for pixel-art turntables. I_PCM stores changed 16×16 blocks;
//! P_Skip repeats blocks that already match their reference picture exactly.
use super::super::{bx_box, bx_fullbox, bx_u16, bx_u32, movie_box, MergedSample, MergedTrack};
use super::{err, PngImage, Result};

const GOP: u32 = 60;
const BLOCK: usize = 384; // Y (16×16), Cb and Cr (8×8 each).

#[derive(Default)]
struct Bits {
    bytes: Vec<u8>,
    partial: u8,
    used: u8,
}

impl Bits {
    fn bit(&mut self, value: bool) {
        self.partial = (self.partial << 1) | u8::from(value);
        self.used += 1;
        if self.used == 8 {
            self.align();
        }
    }
    fn align(&mut self) {
        if self.used > 0 {
            self.bytes.push(self.partial << (8 - self.used));
            self.partial = 0;
            self.used = 0;
        }
    }
    fn unsigned(&mut self, value: u32, width: u32) {
        for bit in (0..width).rev() {
            self.bit(value & (1 << bit) != 0);
        }
    }
    fn ue(&mut self, value: u32) {
        let code = value + 1;
        let width = 32 - code.leading_zeros();
        self.unsigned(0, width - 1);
        self.unsigned(code, width);
    }
    fn finish(mut self) -> Vec<u8> {
        self.bit(true);
        self.align();
        self.bytes
    }
}

fn write_nal(out: &mut Vec<u8>, header: u8, rbsp: &[u8]) {
    out.push(header);
    let mut zeros = 0;
    for &byte in rbsp {
        if zeros == 2 && byte <= 3 {
            out.push(3);
            zeros = 0;
        }
        out.push(byte);
        zeros = if byte == 0 { zeros + 1 } else { 0 };
    }
}

// Annex A limits by macroblocks/frame, macroblocks/second and kb/s.
fn level(cols: u32, rows: u32, rate: f64, kbps: f64) -> Result<u8> {
    const LEVELS: &[(u8, u32, u32, u32)] = &[
        (30, 1620, 40500, 10000),
        (31, 3600, 108000, 14000),
        (32, 5120, 216000, 20000),
        (40, 8192, 245760, 20000),
        (41, 8192, 245760, 50000),
        (42, 8704, 522240, 50000),
        (50, 22080, 589824, 135000),
        (51, 36864, 983040, 240000),
        (52, 36864, 2073600, 240000),
        (60, 139264, 4177920, 240000),
        (61, 139264, 8355840, 480000),
        (62, 139264, 16711680, 800000),
    ];
    let blocks = cols * rows;
    LEVELS
        .iter()
        .find(|&&(_, limit, mbps, bitrate)| {
            blocks <= limit
                && cols.max(rows).pow(2) <= 8 * limit
                && f64::from(blocks) * rate <= f64::from(mbps)
                && kbps <= f64::from(bitrate)
        })
        .map(|v| v.0)
        .ok_or_else(|| err("Sprite video exceeds H.264 Baseline level limits"))
}

fn avc_sample_description(width: u32, height: u32, level: u8) -> Vec<u8> {
    let cols = width.div_ceil(16);
    let rows = height.div_ceil(16);
    let mut sps = Bits::default();
    sps.bytes.extend_from_slice(&[66, 0xc0, level]);
    for x in [0, 0, 2, 1] {
        sps.ue(x);
    } // SPS id, frame numbering, POC type, one reference
    sps.bit(false);
    sps.ue(cols - 1);
    sps.ue(rows - 1);
    sps.bit(true);
    sps.bit(true);
    let crop_x = (cols * 16 - width) / 2;
    let crop_y = (rows * 16 - height) / 2;
    sps.bit(crop_x != 0 || crop_y != 0);
    if crop_x != 0 || crop_y != 0 {
        for x in [0, crop_x, 0, crop_y] {
            sps.ue(x);
        }
    }
    // VUI identifies square pixels and the matrix actually used below,
    // including on HD displays whose default would otherwise be BT.709.
    sps.bit(true);
    sps.bit(true);
    sps.unsigned(1, 8);
    sps.bit(false);
    sps.bit(true);
    sps.unsigned(5, 3);
    sps.bit(false);
    sps.bit(true);
    for value in [1, 13, 6] {
        sps.unsigned(value, 8);
    }
    sps.bit(true);
    sps.ue(1);
    sps.ue(1);
    sps.unsigned(0, 5);
    let mut pps = Bits::default();
    pps.ue(0);
    pps.ue(0);
    pps.unsigned(0, 2); // PPS/SPS ids, CAVLC, no pic-order flag
    for _ in 0..3 {
        pps.ue(0);
    }
    pps.unsigned(0, 3); // no weighted prediction
    for _ in 0..3 {
        pps.ue(0);
    } // signed zero offsets
    pps.bit(true);
    pps.unsigned(0, 2);
    let mut sequence = vec![];
    write_nal(&mut sequence, 0x67, &sps.finish());
    let mut picture = vec![];
    write_nal(&mut picture, 0x68, &pps.finish());
    let avcc = bx_box(
        "avcC",
        &[
            &[1, 66, 0xc0, level, 0xff, 0xe1],
            &bx_u16(sequence.len() as u16),
            &sequence,
            &[1],
            &bx_u16(picture.len() as u16),
            &picture,
        ],
    );
    let avc1 = bx_box(
        "avc1",
        &[
            &[0; 6],
            &bx_u16(1),
            &[0; 16],
            &bx_u16(width as u16),
            &bx_u16(height as u16),
            &bx_u32(0x00480000),
            &bx_u32(0x00480000),
            &bx_u32(0),
            &bx_u16(1),
            &[0; 32],
            &bx_u16(24),
            &bx_u16(0xffff),
            &avcc,
        ],
    );
    bx_fullbox("stsd", 0, 0, &[&bx_u32(1), &avc1])
}

fn pack_blocks(frame: &PngImage, cols: usize, blocks: &mut [u8]) {
    let width = frame.width as usize;
    let height = frame.height as usize;
    for (id, block) in blocks.chunks_exact_mut(BLOCK).enumerate() {
        let ox = id % cols * 16;
        let oy = id / cols * 16;
        for y in (0..16).step_by(2) {
            for x in (0..16).step_by(2) {
                let (mut red, mut green, mut blue) = (0, 0, 0);
                for dy in 0..2 {
                    let row = (oy + y + dy).min(height - 1) * width;
                    for dx in 0..2 {
                        let offset = 4 * (row + (ox + x + dx).min(width - 1));
                        let pix = &frame.rgba[offset..offset + 4];
                        let alpha = i32::from(pix[3]);
                        let channel = |c: u8| (i32::from(c) * alpha + 127) / 255;
                        let (r, g, b) = (channel(pix[0]), channel(pix[1]), channel(pix[2]));
                        block[(y + dy) * 16 + x + dx] =
                            (16 + ((66 * r + 129 * g + 25 * b + 128) >> 8)) as u8;
                        red += r;
                        green += g;
                        blue += b;
                    }
                }
                let c = y / 2 * 8 + x / 2;
                block[256 + c] = (128 + ((-38 * red - 74 * green + 112 * blue + 512) >> 10)) as u8;
                block[320 + c] = (128 + ((112 * red - 94 * green - 18 * blue + 512) >> 10)) as u8;
            }
        }
    }
}

fn flat_color(block: &[u8]) -> Option<[u8; 3]> {
    let colors = [block[0], block[256], block[320]];
    if block[0..256].iter().all(|&v| v == colors[0])
        && block[256..320].iter().all(|&v| v == colors[1])
        && block[320..384].iter().all(|&v| v == colors[2])
    {
        Some(colors)
    } else {
        None
    }
}

fn slice(index: u32, current: &[u8], previous: &[u8], columns: usize) -> Vec<u8> {
    let idr = index % GOP == 0;
    let mut bits = Bits::default();
    bits.ue(0);
    bits.ue(if idr { 2 } else { 0 });
    bits.ue(0);
    bits.unsigned((index % GOP) & 15, 4);
    if idr {
        bits.ue((index / GOP) & 1);
        bits.unsigned(0, 2);
    } else {
        bits.unsigned(0, 3);
    }
    bits.ue(0);
    bits.ue(1); // unadjusted QP, deblocking disabled
    let mut skip = 0;
    let mut colors = vec![None; columns];
    let mut pcm = vec![false; columns];
    for (index, (block, reference)) in current
        .chunks_exact(BLOCK)
        .zip(previous.chunks_exact(BLOCK))
        .enumerate()
    {
        let x = index % columns;
        let color = flat_color(block);
        let left = color.is_some() && x != 0 && colors[x - 1] == color;
        let top = color.is_some() && index >= columns && colors[x] == color;
        colors[x] = color;
        if !idr {
            if block == reference {
                pcm[x] = false;
                skip += 1;
                continue;
            }
            bits.ue(skip);
            skip = 0;
        }
        if left || top {
            // Directional Intra16x16 exactly predicts a matching constant neighbor.
            bits.ue(if idr { 1 } else { 6 } + u32::from(left));
            bits.ue(if left { 1 } else { 2 });
            bits.ue(0);
            // A PCM neighbor reports 16 coefficients. All other block modes here
            // have zero, so the nC table for the zero DC token is 0 or >=8.
            if (x != 0 && pcm[x - 1]) || (index >= columns && pcm[x]) {
                bits.unsigned(3, 6);
            } else {
                bits.bit(true);
            }
            pcm[x] = false;
        } else {
            bits.ue(if idr { 25 } else { 30 });
            bits.align();
            bits.bytes.extend_from_slice(block);
            pcm[x] = true;
        }
    }
    if skip > 0 {
        bits.ue(skip);
    }
    bits.finish()
}

pub(in crate::local_media) fn encode_turntable(
    width: u32,
    height: u32,
    count: u32,
    duration: f64,
    mut frame: impl FnMut(u32) -> PngImage,
) -> Result<Vec<u8>> {
    if !(2..=65534).contains(&width)
        || !(2..=65534).contains(&height)
        || width % 2 != 0
        || height % 2 != 0
    {
        return Err(err("Sprite video requires even pixel dimensions"));
    }
    let ticks = (duration * 90000.0).round();
    if count == 0 || !ticks.is_finite() || ticks < f64::from(count) || ticks > f64::from(u32::MAX) {
        return Err(err("Invalid sprite frame count or duration"));
    }
    let (cols, rows) = (width.div_ceil(16), height.div_ceil(16));
    let rate = f64::from(count) / duration;
    level(cols, rows, rate, 0.0)?;
    let block_bytes = cols as usize * rows as usize * BLOCK;
    let mut current = vec![0; block_bytes];
    let mut previous = vec![0; block_bytes];
    let ftyp = bx_box("ftyp", &[b"isom", &bx_u32(0x200), b"isomiso2avc1mp41"]);
    let mdat_start = ftyp.len();
    let mut output = ftyp;
    output.extend_from_slice(b"\0\0\0\0mdat");
    let mut samples = Vec::new();
    let ticks = ticks as u64;
    let time = |index: u32| (u64::from(index) * ticks + u64::from(count) / 2) / u64::from(count);
    for index in 0..count {
        let img = frame(index);
        if img.width != width
            || img.height != height
            || img.rgba.len() != width as usize * height as usize * 4
        {
            return Err(err("Sprite video frames have inconsistent dimensions"));
        }
        pack_blocks(&img, cols as usize, &mut current);
        let rbsp = slice(index, &current, &previous, cols as usize);
        let extra = rbsp.len() + rbsp.len() / 2 + 5;
        let overhead = 2048 + u64::from(count) * 28; // sample table and header upper bound
        if output.len() as u64 + extra as u64 + overhead > u64::from(u32::MAX) {
            return Err(err("Turntable exceeds the 32-bit MP4 size limit"));
        }
        output
            .try_reserve(extra)
            .map_err(|_| err("Insufficient memory to encode turntable"))?;
        let offset = output.len();
        output.extend_from_slice(&[0; 4]);
        let idr = index % GOP == 0;
        write_nal(&mut output, if idr { 0x65 } else { 0x41 }, &rbsp);
        let size = output.len() - offset;
        output[offset..offset + 4].copy_from_slice(&bx_u32((size - 4) as u32));
        samples.push(MergedSample {
            buf_idx: 0,
            offset: 0,
            size,
            dur: (time(index + 1) - time(index)) as i64,
            cts: 0,
            sync: idr,
            new_offset: offset,
        });
        std::mem::swap(&mut current, &mut previous);
    }
    let mdat_size = output.len() - mdat_start;
    output[mdat_start..mdat_start + 4].copy_from_slice(&bx_u32(mdat_size as u32));
    let level = level(
        cols,
        rows,
        rate,
        (mdat_size - 8) as f64 * 8.0 / duration / 1000.0,
    )?;
    let track = MergedTrack {
        kind: "video".into(),
        timescale: 90000,
        width,
        height,
        stsd_raw: avc_sample_description(width, height, level),
        samples,
    };
    let mut tracks = [track];
    // Place the index first for progressive playback. The offsets are fixed-width.
    let header_size = movie_box(&tracks).len();
    for s in &mut tracks[0].samples {
        s.new_offset += header_size;
    }
    let moov = movie_box(&tracks);
    output
        .try_reserve(moov.len())
        .map_err(|_| err("Insufficient memory for MP4 header"))?;
    output.splice(mdat_start..mdat_start, moov);
    Ok(output)
}
