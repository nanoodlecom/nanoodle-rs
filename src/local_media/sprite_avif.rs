//! In-memory wrapper for the AVIF decoder built into this crate.
use super::{err, PngImage};
use crate::Result;
use std::{
    ffi::CStr,
    os::raw::{c_char, c_int},
    ptr,
};

extern "C" {
    fn nanoodle_avif_decode(
        encoded: *const u8,
        size: usize,
        width: *mut c_int,
        height: *mut c_int,
        rgba: *mut *mut u8,
    ) -> c_int;
    fn nanoodle_avif_error(code: c_int) -> *const c_char;
    fn nanoodle_avif_free(rgba: *mut u8);
}

pub(super) fn is_avif(raw: &[u8]) -> bool {
    if raw.get(4..8) != Some(b"ftyp") {
        return false;
    }
    let size = u32::from_be_bytes(raw[0..4].try_into().unwrap());
    let (offset, size) = match size {
        0 => (8, raw.len()),
        1 if raw.len() >= 16 => (
            16,
            usize::try_from(u64::from_be_bytes(raw[8..16].try_into().unwrap())).unwrap_or(0),
        ),
        n => (8, n as usize),
    };
    let Some(brands) = raw.get(offset..size) else {
        return false;
    };
    brands.len() >= 8
        && brands
            .chunks_exact(4)
            .enumerate()
            .any(|(i, brand)| i != 1 && (brand == b"avif" || brand == b"avis"))
}

struct Pixels(*mut u8);
impl Drop for Pixels {
    fn drop(&mut self) {
        // The C adapter owns this allocation and accepts NULL.
        unsafe { nanoodle_avif_free(self.0) }
    }
}

pub(super) fn decode(raw: &[u8]) -> Result<PngImage> {
    let (mut width, mut height, mut rgba) = (0, 0, ptr::null_mut());
    // The C decoder borrows input only for this call. Its byte budget and
    // dimension limits are checked before creating the tightly packed pixels.
    let status = unsafe {
        nanoodle_avif_decode(raw.as_ptr(), raw.len(), &mut width, &mut height, &mut rgba)
    };
    let pixels = Pixels(rgba);
    if status != 0 {
        // Error messages are immutable strings with static lifetime.
        let message = unsafe { nanoodle_avif_error(status) };
        if message.is_null() {
            return Err(err("Could not decode AVIF image"));
        }
        return Err(err(format!(
            "AVIF decode failed: {}",
            unsafe { CStr::from_ptr(message) }.to_string_lossy()
        )));
    }
    let width = u32::try_from(width).map_err(|_| err("Invalid AVIF width"))?;
    let height = u32::try_from(height).map_err(|_| err("Invalid AVIF height"))?;
    let count = u64::from(width) * u64::from(height);
    if pixels.0.is_null() || count == 0 || count > 16_777_216 {
        return Err(err("Invalid AVIF output dimensions"));
    }
    let len = (count * 4) as usize;
    // On success the adapter returns exactly width * height * 4 initialized bytes.
    let rgba = unsafe { std::slice::from_raw_parts(pixels.0, len) }.to_vec();
    Ok(PngImage {
        width,
        height,
        rgba,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALPHA: &[u8] = include_bytes!("../../tests/fixtures/sprite-palette-alpha.avif");

    #[test]
    fn palette_decodes_with_chroma_interpolation_and_alpha() {
        let image = decode(ALPHA).unwrap();
        assert_eq!((image.width, image.height), (16, 16));
        // The transparent part of the first scanline begins at column 12.
        assert_eq!(&image.rgba[..4], &[10, 180, 45, 255]);
        assert_eq!(&image.rgba[44..48], &[36, 164, 62, 255]);
        assert_eq!(image.rgba[12 * 4 + 3], 0);
    }

    #[test]
    fn accepts_compatible_brands_and_rejects_truncated_or_oversized_images() {
        assert!(is_avif(ALPHA));
        assert!(is_avif(b"\0\0\0\x14ftypmif1\0\0\0\0avif"));
        assert!(!is_avif(b"\0\0\0\x14ftypmif1avifheic"));
        assert!(!is_avif(b"\0\0\0\x01ftyp"));
        assert!(decode(&ALPHA[..ALPHA.len() / 2]).is_err());

        // Increase ispe's displayed width before requesting frame decoding.
        let mut wide = ALPHA.to_vec();
        let ispe = wide.windows(4).position(|s| s == b"ispe").unwrap();
        wide[ispe + 8..ispe + 12].copy_from_slice(&16385u32.to_be_bytes());
        assert!(decode(&wide).is_err());
    }
}
