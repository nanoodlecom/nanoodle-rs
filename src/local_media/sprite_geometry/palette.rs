//! Palette order follows first occurrence and stable weighted median splits,
//! keeping nearest-color ties portable across the executors.
use super::{err, PngImage};
use crate::Result;
use std::collections::HashMap;

type Rgb = [u8; 3];

struct ColorBox {
    colors: Vec<[f64; 4]>,
    count: f64,
    axis: usize,
    score: f64,
}
impl ColorBox {
    fn new(colors: Vec<[f64; 4]>) -> Self {
        let mut low = [255_f64; 3];
        let mut high = [0_f64; 3];
        let mut count = 0.;
        for rgb in &colors {
            for k in 0..3 {
                low[k] = low[k].min(rgb[k]);
                high[k] = high[k].max(rgb[k]);
            }
            count += rgb[3];
        }
        let mut axis = 0;
        for k in 1..3 {
            if high[k] - low[k] > high[axis] - low[axis] {
                axis = k;
            }
        }
        let score = if colors.len() > 1 {
            (high[axis] - low[axis]) * count
        } else {
            0.
        };
        Self {
            colors,
            count,
            axis,
            score,
        }
    }
}

fn colors(data: &[u8], limit: usize) -> Result<Vec<Rgb>> {
    let mut exact = Vec::with_capacity(limit + 1);
    let mut indices = HashMap::new();
    let mut bins = Vec::<[u64; 4]>::new();
    for px in data.chunks_exact(4).filter(|p| p[3] != 0) {
        let rgb = [px[0], px[1], px[2]];
        if exact.len() <= limit && !exact.contains(&rgb) {
            exact.push(rgb);
        }
        let key = ((px[0] as u16 >> 3) << 10) | ((px[1] as u16 >> 3) << 5) | (px[2] as u16 >> 3);
        let idx = *indices.entry(key).or_insert_with(|| {
            bins.push([0; 4]);
            bins.len() - 1
        });
        for k in 0..3 {
            bins[idx][k] += rgb[k] as u64;
        }
        bins[idx][3] += 1;
    }
    if exact.is_empty() {
        return Err(err("Palette source has no visible pixels."));
    }
    if exact.len() <= limit {
        return Ok(exact);
    }
    let bins = bins
        .into_iter()
        .map(|bin| {
            let n = bin[3] as f64;
            [bin[0] as f64 / n, bin[1] as f64 / n, bin[2] as f64 / n, n]
        })
        .collect();
    let mut boxes = vec![ColorBox::new(bins)];
    while boxes.len() < limit {
        boxes.sort_by(|a, b| b.score.total_cmp(&a.score));
        if boxes[0].score == 0. {
            break;
        }
        let mut old = boxes.remove(0);
        old.colors
            .sort_by(|a, b| a[old.axis].total_cmp(&b[old.axis]));
        let mut cut = 1;
        let mut weight = old.colors[0][3];
        while cut < old.colors.len() - 1 && weight < old.count / 2. {
            weight += old.colors[cut][3];
            cut += 1;
        }
        let right = old.colors.split_off(cut);
        boxes.push(ColorBox::new(old.colors));
        boxes.push(ColorBox::new(right));
    }
    Ok(boxes
        .into_iter()
        .map(|b| {
            std::array::from_fn(|k| {
                (b.colors.iter().map(|c| c[k] * c[3]).sum::<f64>() / b.count).round() as u8
            })
        })
        .collect())
}

pub(super) fn finish(
    img: &mut PngImage,
    size: usize,
    limit: usize,
    outline: bool,
    supplied: Option<&PngImage>,
) -> Result<()> {
    let mut ink = [24; 3];
    if limit > 0 && (supplied.is_some() || img.rgba.chunks_exact(4).any(|p| p[3] != 0)) {
        let reserve = outline && supplied.is_none() && limit > 1;
        let source = supplied.map(|p| p.rgba.as_slice()).unwrap_or(&img.rgba);
        let palette = colors(source, limit - usize::from(reserve))?;
        if !reserve {
            ink = *palette
                .iter()
                .min_by_key(|rgb| rgb.iter().map(|&c| c as u16).sum::<u16>())
                .unwrap();
        }
        let mut cache = HashMap::new();
        for px in img.rgba.chunks_exact_mut(4).filter(|p| p[3] != 0) {
            let rgb = [px[0], px[1], px[2]];
            let c = cache.entry(rgb).or_insert_with(|| {
                *palette
                    .iter()
                    .min_by_key(|c| {
                        (0..3)
                            .map(|k| (rgb[k] as i32 - c[k] as i32).pow(2))
                            .sum::<i32>()
                    })
                    .unwrap()
            });
            px[..3].copy_from_slice(c);
        }
    }
    if outline {
        let width = img.width as usize;
        let alpha: Vec<_> = img.rgba.chunks_exact(4).map(|v| v[3]).collect();
        for (i, px) in img.rgba.chunks_exact_mut(4).enumerate() {
            if alpha[i] != 0 {
                continue;
            }
            let x = i % size;
            let mut a = 0;
            if x > 0 {
                a = alpha[i - 1];
            }
            if x + 1 < size {
                a = a.max(alpha[i + 1]);
            }
            if i >= width {
                a = a.max(alpha[i - width]);
            }
            if i + width < alpha.len() {
                a = a.max(alpha[i + width]);
            }
            if a != 0 {
                px[..3].copy_from_slice(&ink);
                px[3] = a;
            }
        }
    }
    Ok(())
}
