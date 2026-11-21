//! Animated stacks are full horizontal slice sheets in an APNG. The private
//! nsTA chunk opts in to clip playback; ordinary animated images keep their
//! existing first-frame behavior.
use super::{crc32, err, png_chunk, sprite_codecs, zlib_deflate, PngImage};
use crate::Result;

const SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";
const RGBA8: &[u8] = &[8, 6, 0, 0, 0];
const MAX_CHUNKS: usize = 65_536;
const MAX_FILE: usize = 544 * 1024 * 1024;

fn word(data: &[u8]) -> u32 {
    u32::from_be_bytes(data[..4].try_into().unwrap())
}

fn valid_count(count: usize) -> Result<()> {
    if !(2..=64).contains(&count) {
        return Err(err("An animated stack must have 2 to 64 frames"));
    }
    Ok(())
}

fn check_sheet(width: u32, height: u32) -> Result<()> {
    if !(16..=128).contains(&height)
        || width % height != 0
        || !(8..=128).contains(&(width / height))
    {
        return Err(err("Invalid animated stack sheet dimensions"));
    }
    Ok(())
}

pub(super) struct SequenceWriter {
    count: usize,
    written: usize,
    delay: [u8; 4],
    size: Option<(u32, u32)>,
    output: Vec<u8>,
}

impl SequenceWriter {
    pub(super) fn new(count: usize, duration_seconds: f64) -> Result<Self> {
        valid_count(count)?;
        let seconds = duration_seconds / count as f64;
        if !seconds.is_finite() || seconds <= 0.0 || seconds > 65_535.0 {
            return Err(err("Invalid animated stack frame delay"));
        }
        let den = (65_535.0 / seconds).floor().clamp(1.0, 60_000.0) as u16;
        let num = (seconds * f64::from(den)).round().clamp(1.0, 65_535.0) as u16;
        let [a, b] = num.to_be_bytes();
        let [c, d] = den.to_be_bytes();
        Ok(Self {
            count,
            written: 0,
            delay: [a, b, c, d],
            size: None,
            output: SIGNATURE.to_vec(),
        })
    }

    pub(super) fn push(&mut self, img: &PngImage) -> Result<()> {
        let size = (img.width, img.height);
        check_sheet(img.width, img.height)?;
        let stride = img.width as usize * 4;
        if self.written >= self.count
            || self.size.is_some_and(|expected| expected != size)
            || img.rgba.len() != stride * img.height as usize
        {
            return Err(err("Invalid animated stack frame dimensions or count"));
        }
        if self.written == 0 {
            let mut header = Vec::with_capacity(13);
            header.extend_from_slice(&img.width.to_be_bytes());
            header.extend_from_slice(&img.height.to_be_bytes());
            header.extend_from_slice(RGBA8);
            png_chunk(&mut self.output, b"IHDR", &header);
            png_chunk(&mut self.output, b"nsTA", &[1]);
            let mut control = [0u8; 8]; // plays=0 loops indefinitely
            control[..4].copy_from_slice(&(self.count as u32).to_be_bytes());
            png_chunk(&mut self.output, b"acTL", &control);
            self.size = Some(size);
        }

        // The first frame's IDAT has no sequence number. Every later fcTL
        // and fdAT consumes one number.
        let seq = (2 * self.written).saturating_sub(1) as u32;
        let mut control = [0u8; 26]; // offsets, disposal and blend are zero
        control[..4].copy_from_slice(&seq.to_be_bytes());
        control[4..8].copy_from_slice(&img.width.to_be_bytes());
        control[8..12].copy_from_slice(&img.height.to_be_bytes());
        control[20..24].copy_from_slice(&self.delay);
        png_chunk(&mut self.output, b"fcTL", &control);
        let mut rows = Vec::with_capacity((stride + 1) * img.height as usize);
        for row in img.rgba.chunks_exact(stride) {
            rows.push(0);
            rows.extend_from_slice(row);
        }
        let compressed = zlib_deflate(&rows);
        if self.written == 0 {
            png_chunk(&mut self.output, b"IDAT", &compressed);
        } else {
            let mut body = Vec::with_capacity(4 + compressed.len());
            body.extend_from_slice(&(seq + 1).to_be_bytes());
            body.extend_from_slice(&compressed);
            png_chunk(&mut self.output, b"fdAT", &body);
        }
        self.written += 1;
        Ok(())
    }

    pub(super) fn finish(mut self) -> Result<Vec<u8>> {
        if self.written != self.count {
            return Err(err("Missing animated stack frames"));
        }
        png_chunk(&mut self.output, b"IEND", &[]);
        Ok(self.output)
    }
}

struct Chunk<'a> {
    kind: &'a [u8],
    body: &'a [u8],
    checksummed: &'a [u8],
    crc: u32,
}

fn next_chunk<'a>(raw: &mut &'a [u8]) -> Result<Chunk<'a>> {
    let length = raw
        .get(..4)
        .ok_or_else(|| err("Missing animated stack chunk"))?;
    let end = (word(length) as usize)
        .checked_add(12)
        .filter(|end| *end <= raw.len())
        .ok_or_else(|| err("Truncated animated stack chunk"))?;
    let (bytes, rest) = raw.split_at(end);
    *raw = rest;
    Ok(Chunk {
        kind: &bytes[4..8],
        body: &bytes[8..end - 4],
        checksummed: &bytes[4..end - 4],
        crc: word(&bytes[end - 4..]),
    })
}

fn is_marked(mut raw: &[u8]) -> bool {
    for _ in 0..MAX_CHUNKS {
        if raw.len() < 8 {
            break;
        }
        if &raw[4..8] == b"nsTA" {
            return true;
        }
        match next_chunk(&mut raw) {
            Ok(chunk) if chunk.kind != b"IEND" => (),
            _ => break,
        }
    }
    false
}

#[derive(Default)]
struct Frame<'a> {
    parts: Vec<&'a [u8]>,
    size: usize,
}

pub(super) struct SpriteSequence<'a> {
    header: &'a [u8],
    frames: Vec<Frame<'a>>,
    seconds_per_frame: f64,
}

impl SpriteSequence<'_> {
    pub(super) fn len(&self) -> usize {
        self.frames.len()
    }

    pub(super) fn duration_seconds(&self) -> f64 {
        self.seconds_per_frame * self.len() as f64
    }

    pub(super) fn frame(&self, index: usize) -> Result<PngImage> {
        let frame = self
            .frames
            .get(index)
            .ok_or_else(|| err("Invalid animated stack frame index"))?;
        // Each pose is an independent PNG zlib stream with the shared IHDR.
        // Reuse the bounded PNG decoder for its row filters and exact-length checks.
        let mut png = Vec::with_capacity(frame.size + 57 + 12 * frame.parts.len());
        png.extend_from_slice(SIGNATURE);
        png_chunk(&mut png, b"IHDR", self.header);
        for part in &frame.parts {
            png_chunk(&mut png, b"IDAT", part);
        }
        png_chunk(&mut png, b"IEND", &[]);
        sprite_codecs::read_png(&png, true)
    }
}

fn check_seq(data: &[u8], expected: &mut u32) -> Result<()> {
    if word(data) != *expected {
        return Err(err("Out of order animated stack chunk"));
    }
    *expected += 1;
    Ok(())
}

pub(super) fn read_sequence(raw: &[u8]) -> Result<Option<SpriteSequence<'_>>> {
    let Some(mut chunks) = raw.strip_prefix(SIGNATURE) else {
        return Ok(None);
    };
    if !is_marked(chunks) {
        return Ok(None);
    }
    if raw.len() > MAX_FILE {
        return Err(err("Animated stack exceeds the file size limit"));
    }
    let first = next_chunk(&mut chunks)?;
    let header = first.body;
    if first.kind != b"IHDR"
        || header.len() != 13
        || &header[8..] != RGBA8
        || crc32(first.checksummed) != first.crc
    {
        return Err(err("Invalid animated stack PNG header"));
    }
    let (width, height) = (word(header), word(&header[4..]));
    check_sheet(width, height)?;
    let max_compressed = (4 * width as usize + 1) * height as usize + 65_536;
    let mut marked = false;
    let mut count = 0;
    let mut seq = 0;
    let mut delay = None;
    let mut frames: Vec<Frame<'_>> = Vec::new();
    let mut previous = first.kind;
    for _ in 1..MAX_CHUNKS {
        let chunk = next_chunk(&mut chunks)?;
        if !chunk.kind.iter().all(u8::is_ascii_alphabetic) || !chunk.kind[2].is_ascii_uppercase() {
            return Err(err("Invalid animated stack PNG chunk name"));
        }
        if crc32(chunk.checksummed) != chunk.crc {
            return Err(err("Invalid animated stack PNG checksum"));
        }
        let data = chunk.body;
        match chunk.kind {
            b"nsTA" if !marked && count == 0 && data == [1] => marked = true,
            b"acTL" if marked && count == 0 && data.len() == 8 && word(&data[4..]) == 0 => {
                count = word(data) as usize;
                valid_count(count)?;
            }
            b"fcTL" => {
                if frames.len() >= count
                    || frames.last().is_some_and(|frame| frame.size == 0)
                    || data.len() != 26
                    || data[4..12] != header[..8]
                    || data[12..20] != [0; 8]
                    || data[24..26] != [0; 2]
                {
                    return Err(err("Invalid animated stack frame control"));
                }
                check_seq(data, &mut seq)?;
                let num = u16::from_be_bytes([data[20], data[21]]) as u32;
                let den = u16::from_be_bytes([data[22], data[23]]) as u32;
                if num == 0 || den == 0 || delay.is_some_and(|(n, d)| n * den != num * d) {
                    return Err(err("Invalid or nonuniform animated stack frame delay"));
                }
                delay = Some((num, den));
                frames.push(Frame::default());
            }
            b"IDAT" | b"fdAT" => {
                let part = match (chunk.kind, frames.len()) {
                    (b"IDAT", 1) if frames[0].parts.is_empty() || previous == b"IDAT" => data,
                    (b"fdAT", 2..) if data.len() >= 4 => {
                        check_seq(data, &mut seq)?;
                        &data[4..]
                    }
                    _ => return Err(err("Unexpected animated stack frame data")),
                };
                let frame = frames.last_mut().unwrap();
                if part.len() > max_compressed - frame.size {
                    return Err(err(
                        "Animated stack frame exceeds the compressed size limit",
                    ));
                }
                frame.size += part.len();
                frame.parts.push(part);
            }
            b"IEND" => {
                if !data.is_empty()
                    || !chunks.is_empty()
                    || frames.len() != count
                    || frames.last().is_none_or(|f| f.size == 0)
                {
                    return Err(err("Incomplete animated stack"));
                }
                let (num, den) = delay.unwrap();
                return Ok(Some(SpriteSequence {
                    frames,
                    header,
                    seconds_per_frame: f64::from(num) / f64::from(den),
                }));
            }
            b"nsTA" | b"acTL" => return Err(err("Invalid animated stack control chunk")),
            b"PLTE" => (), // optional suggested palette in an RGBA PNG
            kind if kind[0].is_ascii_lowercase() => (),
            _ => return Err(err("Unsupported animated stack PNG chunk")),
        }
        previous = chunk.kind;
    }
    Err(err("Animated stack exceeds the PNG chunk limit"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local_media::encode_png_rgba;

    // Independent, Pillow-verified APNG shared with the JS/Python executors.
    // Each pose's zlib stream spans two chunks.
    const SAMPLE: &[u8] = include_bytes!("../../tests/fixtures/stack-animation.apng");

    #[test]
    fn split_frames_write_and_read_in_pose_order() {
        let src = read_sequence(SAMPLE).unwrap().unwrap();
        assert_eq!(src.len(), 3);
        assert_eq!(src.duration_seconds(), 0.375);
        assert!(src.frame(3).is_err());
        let mut writer = SequenceWriter::new(3, 0.375).unwrap();
        for (i, expected) in [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]]
            .into_iter()
            .enumerate()
        {
            let img = src.frame(i).unwrap();
            assert_eq!((img.width, img.height), (128, 16));
            let pixel = 4 * (7 * 128 + 6);
            assert_eq!(img.rgba[pixel..pixel + 4], expected);
            writer.push(&img).unwrap();
        }
        let output = writer.finish().unwrap();
        let decoded = read_sequence(&output).unwrap().unwrap();
        assert_eq!(decoded.duration_seconds(), 0.375);
        assert_eq!(decoded.len(), src.len());
        for i in 0..src.len() {
            assert_eq!(decoded.frame(i).unwrap().rgba, src.frame(i).unwrap().rgba);
        }
    }

    // Rewrite a selected chunk with a new CRC so that structural tests do
    // not accidentally exercise only the checksum check.
    fn edit(kind: &[u8; 4], occurrence: usize, mutate: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
        let mut out = SIGNATURE.to_vec();
        let mut raw = &SAMPLE[SIGNATURE.len()..];
        let mut f = Some(mutate);
        let mut count = 0;
        while !raw.is_empty() {
            let chunk = next_chunk(&mut raw).unwrap();
            let mut body = chunk.body.to_vec();
            if chunk.kind == kind {
                if count == occurrence {
                    f.take().unwrap()(&mut body);
                }
                count += 1;
            }
            png_chunk(&mut out, chunk.kind.try_into().unwrap(), &body);
        }
        assert!(f.is_none());
        out
    }

    #[test]
    fn unmarked_media_keeps_original_first_frame_semantics() {
        let mut unmarked = SAMPLE.to_vec();
        unmarked.drain(33..46); // marker with its length, type and CRC
        assert!(read_sequence(&unmarked).unwrap().is_none());
        let frame = read_sequence(SAMPLE).unwrap().unwrap().frame(0).unwrap();
        assert_eq!(
            sprite_codecs::decode_image(&unmarked).unwrap().rgba,
            frame.rgba
        );
        assert!(read_sequence(&encode_png_rgba(&frame)).unwrap().is_none());
        assert!(read_sequence(b"GIF89a").unwrap().is_none());
    }

    #[test]
    fn bad_controls_and_data_fail_without_falling_back_to_static() {
        for malformed in [
            edit(b"nsTA", 0, |b| b[0] = 2),
            edit(b"acTL", 0, |b| b[3] = 65),
            edit(b"acTL", 0, |b| b[3] = 2),
            edit(b"fcTL", 1, |b| b[3] += 1),
            edit(b"fdAT", 0, |b| b[3] += 1),
            edit(b"fcTL", 0, |b| b[15] = 1),
            edit(b"fcTL", 0, |b| b[24] = 1),
            edit(b"fcTL", 0, |b| b[25] = 1),
            edit(b"fcTL", 2, |b| b[22..24].fill(0)),
            edit(b"fcTL", 2, |b| b[21] += 1),
            edit(b"IHDR", 0, |b| b[3] = 127),
            edit(b"fdAT", 0, |b| b.resize(74_000, 0)),
            [SAMPLE, &[0]].concat(),
            SAMPLE[..SAMPLE.len() - 12].to_vec(),
        ] {
            assert!(read_sequence(&malformed).is_err());
        }
        let mut bad_crc = SAMPLE.to_vec();
        bad_crc[45] ^= 1;
        assert!(read_sequence(&bad_crc).is_err());
        let corrupt = edit(b"fdAT", 0, |b| b[6] ^= 255);
        let seq = read_sequence(&corrupt).unwrap().unwrap();
        assert!(seq.frame(0).is_ok());
        assert!(seq.frame(1).is_err());
    }

    #[test]
    fn encoder_checks_inputs_before_accepting_a_frame() {
        assert!(SequenceWriter::new(1, 1.0).is_err());
        assert!(SequenceWriter::new(65, 1.0).is_err());
        for invalid in [f64::NAN, 0.0, -1.0, f64::INFINITY, 131_071.0] {
            assert!(SequenceWriter::new(2, invalid).is_err());
        }
        let frame = read_sequence(SAMPLE).unwrap().unwrap().frame(0).unwrap();
        let mut writer = SequenceWriter::new(2, 2.0 / 24.0).unwrap();
        writer.push(&frame).unwrap();
        assert!(writer
            .push(&PngImage {
                width: 128,
                height: 16,
                rgba: Vec::new()
            })
            .is_err());
        assert!(writer
            .push(&PngImage {
                width: 256,
                height: 16,
                rgba: vec![0; 16384]
            })
            .is_err());
        writer.push(&frame).unwrap();
        assert!(writer.push(&frame).is_err());
        let output = writer.finish().unwrap();
        let seq = read_sequence(&output).unwrap().unwrap();
        assert_eq!(seq.duration_seconds(), 2.0 / 24.0);
        assert!(SequenceWriter::new(2, 1.0).unwrap().finish().is_err());
    }
}
