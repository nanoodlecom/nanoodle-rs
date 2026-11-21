//! In-process 8-bit Huffman JPEG texture decoding (ITU-T T.81, Annexes F and G).
//! Sequential and progressive scans share quantization, IDCT and color sampling.
use super::{err, PngImage};
use crate::{NanoodleError, Result};
use std::sync::OnceLock;

const ZIG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];
const MAX_PIXELS: usize = 16 * 1024 * 1024;

fn invalid() -> NanoodleError {
    err("Invalid or truncated JPEG texture")
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or_else(invalid)?;
        let result = self.data.get(self.pos..end).ok_or_else(invalid)?;
        self.pos = end;
        Ok(result)
    }
    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn word(&mut self) -> Result<u16> {
        let bytes = self.take(2)?;
        Ok(u16::from_be_bytes([bytes[0], bytes[1]]))
    }
    fn marker(&mut self) -> Result<u8> {
        if self.byte()? != 0xff {
            return Err(invalid());
        }
        let mut m = self.byte()?;
        while m == 0xff {
            m = self.byte()?;
        }
        Ok(m)
    }
    fn segment(&mut self) -> Result<Reader<'a>> {
        let n = self.word()?.checked_sub(2).ok_or_else(invalid)?;
        Ok(Reader::new(self.take(n.into())?))
    }
    fn end(&self) -> bool {
        self.pos == self.data.len()
    }
}

struct Bits<'b, 'a> {
    stream: &'b mut Reader<'a>,
    byte: u8,
    left: u8,
}
impl Bits<'_, '_> {
    fn bit(&mut self) -> Result<i32> {
        if self.left == 0 {
            self.byte = self.stream.byte()?;
            if self.byte == 0xff && self.stream.byte()? != 0 {
                return Err(invalid());
            }
            self.left = 8;
        }
        self.left -= 1;
        Ok(((self.byte >> self.left) & 1).into())
    }
    fn read(&mut self, n: u8) -> Result<i32> {
        let mut value = 0;
        for _ in 0..n {
            value = value * 2 + self.bit()?;
        }
        Ok(value)
    }
    fn amplitude(&mut self, n: u8) -> Result<i32> {
        if n == 0 {
            return Ok(0);
        }
        let bits = self.read(n)?;
        Ok(if bits < 1 << (n - 1) {
            bits + 1 - (1 << n)
        } else {
            bits
        })
    }
}

#[derive(Default)]
struct Huff {
    // The first canonical code of each length and its half-open symbol range.
    codes: [i32; 16],
    offsets: [usize; 17],
    symbols: Vec<u8>,
}
impl Huff {
    fn parse(r: &mut Reader) -> Result<Self> {
        let counts = r.take(16)?;
        let mut table = Self::default();
        let mut code = 0;
        for (i, &n) in counts.iter().enumerate() {
            table.codes[i] = code;
            table.offsets[i + 1] = table.offsets[i] + n as usize;
            code += i32::from(n);
            if code >= (1 << (i + 1)) {
                return Err(invalid());
            } // all-ones is reserved
            code <<= 1;
        }
        if table.offsets[16] > 256 {
            return Err(invalid());
        }
        table.symbols = r.take(table.offsets[16])?.to_vec();
        Ok(table)
    }
    fn symbol(&self, bits: &mut Bits) -> Result<u8> {
        let mut code = 0;
        for i in 0..16 {
            code = code * 2 + bits.bit()?;
            let index = code - self.codes[i];
            if index >= 0 && (index as usize) < self.offsets[i + 1] - self.offsets[i] {
                return Ok(self.symbols[self.offsets[i] + index as usize]);
            }
        }
        Err(invalid())
    }
}

struct Component {
    id: u8,
    h: usize,
    v: usize,
    selector: usize,
    q: Option<[u16; 64]>,
    stride: usize, // padded blocks per MCU row
    sw: usize,     // unpadded sample dimensions
    sh: usize,
    blocks: Vec<[i16; 64]>,
    approximation: [i8; 64], // -1 means that spectral band has not arrived
}
struct Frame {
    w: usize,
    h: usize,
    max_h: usize,
    max_v: usize,
    cols: usize,
    rows: usize,
    progressive: bool,
    components: Vec<Component>,
}
impl Frame {
    fn parse(mut r: Reader, progressive: bool) -> Result<Self> {
        if r.byte()? != 8 {
            return Err(err("JPEG texture requires 8-bit samples"));
        }
        let h = usize::from(r.word()?);
        let w = usize::from(r.word()?);
        let count = usize::from(r.byte()?);
        if !matches!(count, 1 | 3 | 4) {
            return Err(err("Unsupported JPEG component count"));
        }
        if w == 0 || h == 0 || w * h > MAX_PIXELS {
            return Err(err("Invalid or oversized JPEG dimensions"));
        }
        let mut components = Vec::with_capacity(count);
        for _ in 0..count {
            let id = r.byte()?;
            let hv = r.byte()?;
            let (h, v) = (usize::from(hv >> 4), usize::from(hv & 15));
            let selector = usize::from(r.byte()?);
            if !(1..=4).contains(&h)
                || !(1..=4).contains(&v)
                || selector > 3
                || components.iter().any(|c: &Component| c.id == id)
            {
                return Err(invalid());
            }
            components.push(Component {
                id,
                h,
                v,
                selector,
                q: None,
                stride: 0,
                sw: 0,
                sh: 0,
                blocks: Vec::new(),
                approximation: [-1; 64],
            });
        }
        if !r.end() {
            return Err(invalid());
        }
        let max_h = components.iter().map(|c| c.h).max().ok_or_else(invalid)?;
        let max_v = components.iter().map(|c| c.v).max().ok_or_else(invalid)?;
        let cols = w.div_ceil(max_h * 8);
        let rows = h.div_ceil(max_v * 8);
        let mut total_blocks = 0;
        for c in &mut components {
            c.stride = cols * c.h;
            c.sw = (w * c.h).div_ceil(max_h);
            c.sh = (h * c.v).div_ceil(max_v);
            let n = c.stride * rows * c.v;
            total_blocks += n;
            if total_blocks > MAX_PIXELS / 16 {
                return Err(err("JPEG coefficient planes exceed memory budget"));
            }
            c.blocks
                .try_reserve_exact(n)
                .map_err(|_| err("Could not allocate JPEG coefficient plane"))?;
            c.blocks.resize(n, [0; 64]);
        }
        Ok(Self {
            w,
            h,
            max_h,
            max_v,
            cols,
            rows,
            progressive,
            components,
        })
    }
}

struct Scan {
    start: usize,
    end: usize,
    high: u8,
    low: u8,
}

impl Scan {
    fn block(
        &self,
        bits: &mut Bits,
        coeff: &mut [i16; 64],
        dc: &Huff,
        ac: &Huff,
        predictor: &mut i32,
        eob: &mut usize,
    ) -> Result<()> {
        let delta = 1i32 << self.low;
        if self.start == 0 {
            if self.high == 0 {
                let size = dc.symbol(bits)?;
                if size > 11 {
                    return Err(invalid());
                }
                *predictor += bits.amplitude(size)?;
                coeff[0] = i16::try_from(*predictor * delta).map_err(|_| invalid())?;
            } else {
                coeff[0] |= (bits.bit()? * delta) as i16;
            }
        }
        let mut k = self.start.max(1);
        if k > self.end {
            return Ok(());
        }
        if self.high == 0 {
            if *eob > 0 {
                *eob -= 1;
                return Ok(());
            }
            while k <= self.end {
                let symbol = ac.symbol(bits)?;
                let run = usize::from(symbol >> 4);
                let size = symbol & 15;
                if size == 0 && run < 15 {
                    // For a sequential scan only EOB (0x00) and ZRL are valid.
                    if self.start == 0 && run != 0 {
                        return Err(invalid());
                    }
                    *eob = (1 << run) + bits.read(run as u8)? as usize - 1;
                    break;
                }
                k += run;
                if k > self.end || size > 10 {
                    return Err(invalid());
                }
                if size != 0 {
                    coeff[ZIG[k]] =
                        i16::try_from(bits.amplitude(size)? * delta).map_err(|_| invalid())?;
                }
                k += 1;
            }
            return Ok(());
        }
        // Refinement bits for previously nonzero ACs are interleaved with the
        // new coefficient's zero run. They continue through the end-of-band run.
        while k <= self.end && *eob == 0 {
            let symbol = ac.symbol(bits)?;
            let mut zeros = usize::from(symbol >> 4);
            let new = match symbol & 15 {
                0 if zeros < 15 => {
                    *eob = (1 << zeros) + bits.read(zeros as u8)? as usize;
                    break;
                }
                0 => {
                    zeros = 16;
                    0
                }
                1 => {
                    if bits.bit()? != 0 {
                        delta
                    } else {
                        -delta
                    }
                }
                _ => return Err(invalid()),
            };
            while k <= self.end {
                let val = &mut coeff[ZIG[k]];
                if *val != 0 {
                    refine(val, bits, delta)?;
                } else if zeros > 0 {
                    zeros -= 1;
                } else {
                    break;
                }
                k += 1;
                if new == 0 && zeros == 0 {
                    break;
                }
            }
            if zeros != 0 {
                return Err(invalid());
            }
            if new != 0 {
                if k > self.end {
                    return Err(invalid());
                }
                coeff[ZIG[k]] = new as i16;
                k += 1;
            }
        }
        if *eob > 0 {
            for &i in &ZIG[k..=self.end] {
                if coeff[i] != 0 {
                    refine(&mut coeff[i], bits, delta)?;
                }
            }
            *eob -= 1;
        }
        Ok(())
    }
}

fn refine(value: &mut i16, bits: &mut Bits, delta: i32) -> Result<()> {
    if bits.bit()? != 0 && (i32::from(*value).abs() & delta) == 0 {
        let updated = i32::from(*value) + if *value > 0 { delta } else { -delta };
        *value = i16::try_from(updated).map_err(|_| invalid())?;
    }
    Ok(())
}

fn scan(
    mut header: Reader,
    data: &mut Reader,
    f: &mut Frame,
    qt: &[Option<[u16; 64]>; 4],
    huff: &[Huff; 8],
    interval: usize,
) -> Result<()> {
    let count = usize::from(header.byte()?);
    if count == 0 || count > f.components.len() {
        return Err(invalid());
    }
    let mut selectors = Vec::with_capacity(count);
    for _ in 0..count {
        let id = header.byte()?;
        let i = f
            .components
            .iter()
            .position(|c| c.id == id)
            .ok_or_else(invalid)?;
        let pair = header.byte()?;
        let (dc, ac) = (usize::from(pair >> 4), usize::from(pair & 15));
        if dc > 3 || ac > 3 || selectors.iter().any(|&(index, _, _)| index == i) {
            return Err(invalid());
        }
        selectors.push((i, dc, ac + 4));
    }
    let start = usize::from(header.byte()?);
    let end = usize::from(header.byte()?);
    let approximation = header.byte()?;
    let (high, low) = (approximation >> 4, approximation & 15);
    if !header.end()
        || start > end
        || end > 63
        || low > 13
        || (high != 0 && high != low + 1)
        || (f.progressive && ((start == 0 && end != 0) || (start != 0 && count != 1)))
        || (!f.progressive && (start != 0 || end != 63 || approximation != 0))
    {
        return Err(invalid());
    }
    for &(i, _, _) in &selectors {
        let c = &mut f.components[i];
        if c.q.is_none() {
            c.q = Some(qt[c.selector].ok_or_else(invalid)?);
        }
        if start > 0 && c.approximation[0] < 0 {
            return Err(invalid());
        }
        for prev in &mut c.approximation[start..=end] {
            if *prev != if high == 0 { -1 } else { high as i8 } {
                return Err(invalid());
            }
            *prev = low as i8;
        }
    }
    let scan = Scan {
        start,
        end,
        high,
        low,
    };
    let interleaved = count > 1;
    let first = &f.components[selectors[0].0];
    let (cols, rows) = if interleaved {
        (f.cols, f.rows)
    } else {
        (first.sw.div_ceil(8), first.sh.div_ceil(8))
    };
    let mut bits = Bits {
        stream: data,
        left: 0,
        byte: 0,
    };
    let mut eob = 0;
    let mut preds = [0; 4];
    let mut restart = 0;
    for mcu in 0..cols * rows {
        if interval > 0 && mcu != 0 && mcu % interval == 0 {
            bits.left = 0;
            if bits.stream.marker()? != 0xd0 + restart || eob != 0 {
                return Err(invalid());
            }
            restart = (restart + 1) % 8;
            preds.fill(0);
        }
        for &(i, dc, ac) in &selectors {
            let c = &mut f.components[i];
            let (bh, bv) = if interleaved { (c.h, c.v) } else { (1, 1) };
            let base = mcu / cols * bv * c.stride + mcu % cols * bh;
            for y in 0..bv {
                for x in 0..bh {
                    scan.block(
                        &mut bits,
                        &mut c.blocks[base + y * c.stride + x],
                        &huff[dc],
                        &huff[ac],
                        &mut preds[i],
                        &mut eob,
                    )?;
                }
            }
        }
    }
    if eob != 0 {
        return Err(invalid());
    }
    Ok(())
}

fn byte(x: f64) -> u8 {
    (x + 0.5).floor().clamp(0.0, 255.0) as u8
}

fn idct(block: &[i16; 64], q: &[u16; 64], out: &mut [u8], stride: usize) {
    if block[1..].iter().all(|&v| v == 0) {
        let level = byte(f64::from(block[0]) * f64::from(q[0]) / 8.0 + 128.0);
        for y in 0..8 {
            out[y * stride..y * stride + 8].fill(level);
        }
        return;
    }
    static BASIS: OnceLock<[[f64; 8]; 8]> = OnceLock::new();
    let basis = BASIS.get_or_init(|| {
        std::array::from_fn(|x| {
            std::array::from_fn(|u| {
                ((2 * x + 1) as f64 * u as f64 * std::f64::consts::PI / 16.).cos()
                    * if u == 0 {
                        std::f64::consts::FRAC_1_SQRT_2
                    } else {
                        1.
                    }
            })
        })
    });
    let mut tmp = [[0.; 8]; 8];
    for v in 0..8 {
        for x in 0..8 {
            for u in 0..8 {
                let i = v * 8 + u;
                tmp[v][x] += f64::from(block[i]) * f64::from(q[i]) * basis[x][u];
            }
        }
    }
    for y in 0..8 {
        for x in 0..8 {
            let mut value = 0.;
            for v in 0..8 {
                value += tmp[v][x] * basis[y][v];
            }
            out[y * stride + x] = byte(value * 0.25 + 128.);
        }
    }
}

struct Plane {
    w: usize,
    h: usize,
    stride: usize,
    data: Vec<u8>,
    h_factor: usize,
    v_factor: usize,
}
impl Plane {
    fn from_component(c: Component) -> Result<Self> {
        let quant = c.q.ok_or_else(invalid)?;
        let stride = c.stride * 8;
        let mut data = vec![0; c.blocks.len() * 64];
        for y in 0..c.sh.div_ceil(8) {
            for x in 0..c.sw.div_ceil(8) {
                idct(
                    &c.blocks[y * c.stride + x],
                    &quant,
                    &mut data[y * 8 * stride + x * 8..],
                    stride,
                );
            }
        }
        Ok(Self {
            w: c.sw,
            h: c.sh,
            stride,
            data,
            h_factor: c.h,
            v_factor: c.v,
        })
    }
    fn sample(&self, x: usize, y: usize, max_h: usize, max_v: usize) -> f64 {
        if self.h_factor == max_h && self.v_factor == max_v {
            return f64::from(self.data[y * self.stride + x]);
        }
        let fx = ((x as f64 + 0.5) * self.h_factor as f64 / max_h as f64 - 0.5)
            .clamp(0., (self.w - 1) as f64);
        let fy = ((y as f64 + 0.5) * self.v_factor as f64 / max_v as f64 - 0.5)
            .clamp(0., (self.h - 1) as f64);
        let x0 = fx as usize;
        let y0 = fy as usize;
        let x1 = (x0 + 1).min(self.w - 1);
        let y1 = (y0 + 1).min(self.h - 1);
        let wx = fx - x0 as f64;
        let wy = fy - y0 as f64;
        let row = |y: usize| {
            f64::from(self.data[y * self.stride + x0]) * (1. - wx)
                + f64::from(self.data[y * self.stride + x1]) * wx
        };
        (row(y0) * (1. - wy) + row(y1) * wy + 0.5).floor()
    }
}

fn render(f: Frame, adobe: Option<u8>) -> Result<PngImage> {
    let rgb = adobe == Some(0) || f.components.iter().map(|c| c.id).eq(b"RGB".iter().copied());
    let planes = f
        .components
        .into_iter()
        .map(Plane::from_component)
        .collect::<Result<Vec<_>>>()?;
    let mut rgba = Vec::with_capacity(f.w * f.h * 4);
    for y in 0..f.h {
        for x in 0..f.w {
            let sample = |i: usize| planes[i].sample(x, y, f.max_h, f.max_v);
            let luma = sample(0);
            let decode_ycc = || {
                let cb = sample(1) - 128.;
                let cr = sample(2) - 128.;
                [
                    byte(luma + 1.402 * cr),
                    byte(luma - 0.344136 * cb - 0.714136 * cr),
                    byte(luma + 1.772 * cb),
                ]
            };
            let color = match planes.len() {
                1 => [luma as u8; 3],
                3 if rgb => [luma as u8, sample(1) as u8, sample(2) as u8],
                3 => decode_ycc(),
                4 => {
                    let mut cmy = if adobe == Some(2) {
                        decode_ycc()
                    } else {
                        [luma as u8, sample(1) as u8, sample(2) as u8]
                    };
                    let mut black = sample(3);
                    if adobe == Some(2) || adobe.is_none() {
                        cmy = cmy.map(|v| 255 - v);
                    }
                    if adobe.is_none() {
                        black = 255. - black;
                    }
                    cmy.map(|v| byte(f64::from(v) * black / 255.))
                }
                _ => unreachable!(),
            };
            rgba.extend_from_slice(&[color[0], color[1], color[2], 255]);
        }
    }
    Ok(PngImage {
        width: f.w as u32,
        height: f.h as u32,
        rgba,
    })
}

pub(super) fn decode_jpeg(raw: &[u8]) -> Result<PngImage> {
    let mut data = Reader::new(raw);
    if data.word()? != 0xffd8 {
        return Err(invalid());
    }
    let mut frame = None;
    let mut quant = [None; 4];
    let mut huff = std::array::from_fn(|_| Huff::default());
    let mut interval = 0usize;
    let mut adobe = None;
    loop {
        let m = data.marker()?;
        if m == 0xd9 {
            break;
        }
        let mut r = data.segment()?;
        match m {
            0xc0 | 0xc1 | 0xc2 => {
                if frame.is_some() {
                    return Err(invalid());
                }
                frame = Some(Frame::parse(r, m == 0xc2)?);
            }
            0xdb => {
                while !r.end() {
                    let info = r.byte()?;
                    if (info >> 4) > 1 || info & 15 > 3 {
                        return Err(invalid());
                    }
                    let mut table = [0; 64];
                    for &i in &ZIG {
                        table[i] = if info >> 4 == 1 {
                            r.word()?
                        } else {
                            u16::from(r.byte()?)
                        };
                        if table[i] == 0 {
                            return Err(invalid());
                        }
                    }
                    quant[usize::from(info & 15)] = Some(table);
                }
            }
            0xc4 => {
                while !r.end() {
                    let info = r.byte()?;
                    if info >> 4 > 1 || info & 15 > 3 {
                        return Err(invalid());
                    }
                    huff[usize::from((info >> 4) * 4 + (info & 15))] = Huff::parse(&mut r)?;
                }
            }
            0xdd => {
                interval = r.word()?.into();
                if !r.end() {
                    return Err(invalid());
                }
            }
            0xda => scan(
                r,
                &mut data,
                frame.as_mut().ok_or_else(invalid)?,
                &quant,
                &huff,
                interval,
            )?,
            0xee if r.data.starts_with(b"Adobe") && r.data.len() >= 12 => adobe = Some(r.data[11]),
            0xe0..=0xef | 0xfe => (),
            _ => {
                return Err(err(
                    "JPEG texture requires sequential or progressive Huffman coding",
                ))
            }
        }
    }
    render(frame.ok_or_else(invalid)?, adobe)
}

#[cfg(test)]
mod tests {
    use super::decode_jpeg;

    // Same 4:2:0 texture, odd dimensions, multiple MCUs, restart every two blocks.
    const BASELINE: &[u8] = include_bytes!("../../tests/fixtures/sprite-texture-baseline.jpg");
    const PROGRESSIVE: &[u8] =
        include_bytes!("../../tests/fixtures/sprite-texture-progressive.jpg");

    #[test]
    fn spectral_refinement_matches_the_sequential_texture() {
        let img = decode_jpeg(BASELINE).unwrap();
        let progressive = decode_jpeg(PROGRESSIVE).unwrap();
        assert_eq!((img.width, img.height), (35, 19));
        assert_eq!(img.rgba, progressive.rgba);
        for (x, y, rgb) in [
            (0, 0, [0, 34, 27]),
            (4, 7, [56, 127, 121]),
            (20, 11, [206, 240, 163]),
            (34, 18, [30, 97, 105]),
        ] {
            let offset = 4 * (y * 35 + x);
            for (sample, reference) in img.rgba[offset..offset + 3].iter().zip(rgb) {
                assert!(sample.abs_diff(reference) <= 3);
            }
            assert_eq!(img.rgba[offset + 3], 255);
        }
    }

    #[test]
    fn incomplete_scan_is_an_error() {
        assert!(decode_jpeg(&PROGRESSIVE[..PROGRESSIVE.len() - 8]).is_err());
        assert!(decode_jpeg(&BASELINE[..BASELINE.len() / 2]).is_err());
    }
}
