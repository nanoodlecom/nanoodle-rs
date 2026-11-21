//! Errors and little-endian reads used by the VP8/VP8L bitstream decoder.
use std::fmt;
use std::io::{self, Read};

#[derive(Debug)]
pub(super) enum DecodingError {
    Io(io::Error),
    BitStreamError,
    ChromaPredictionModeInvalid(i8),
    ColorSpaceInvalid(u8),
    HuffmanError,
    InconsistentImageSizes,
    IntraPredictionModeInvalid(i8),
    InvalidColorCacheBits(u8),
    LosslessSignatureInvalid(u8),
    LumaPredictionModeInvalid(i8),
    NotEnoughInitData,
    TransformError,
    UnsupportedFeature(String),
    VersionNumberInvalid(u8),
    Vp8MagicInvalid([u8; 3]),
}

impl From<io::Error> for DecodingError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl fmt::Display for DecodingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::BitStreamError => f.write_str("corrupt bitstream"),
            Self::ChromaPredictionModeInvalid(m) => write!(f, "invalid chroma prediction mode {m}"),
            Self::ColorSpaceInvalid(s) => write!(f, "unsupported color space {s}"),
            Self::HuffmanError => f.write_str("invalid Huffman code"),
            Self::InconsistentImageSizes => f.write_str("inconsistent dimensions"),
            Self::IntraPredictionModeInvalid(m) => write!(f, "invalid intra prediction mode {m}"),
            Self::InvalidColorCacheBits(b) => write!(f, "invalid color cache bits {b}"),
            Self::LosslessSignatureInvalid(s) => write!(f, "invalid lossless signature {s:#x}"),
            Self::LumaPredictionModeInvalid(m) => write!(f, "invalid luma prediction mode {m}"),
            Self::NotEnoughInitData => f.write_str("truncated partition"),
            Self::TransformError => f.write_str("invalid transform"),
            Self::UnsupportedFeature(s) => f.write_str(s),
            Self::VersionNumberInvalid(v) => write!(f, "unsupported version {v}"),
            Self::Vp8MagicInvalid(m) => write!(f, "invalid frame signature {m:x?}"),
        }
    }
}

pub(super) trait LittleEndianRead: Read {
    fn read_u16_le(&mut self) -> io::Result<u16> {
        let mut b = [0; 2];
        self.read_exact(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }
    fn read_u24_le(&mut self) -> io::Result<u32> {
        let mut b = [0; 4];
        self.read_exact(&mut b[..3])?;
        Ok(u32::from_le_bytes(b))
    }
}

impl<T: Read> LittleEndianRead for T {}
