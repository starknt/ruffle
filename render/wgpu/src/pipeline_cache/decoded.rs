//! Immutable, decoder-native symbol pixels. Only bounded images in the measured size range qualify.
use super::{CacheFile, Hash};
use ruffle_render::bitmap::{Bitmap, BitmapFormat, BitmapSize, EncodedBitmap};
use sha2::{Digest, Sha256};
use std::path::Path;

const MAX_BYTES: u64 = 4 << 20;
const MIN_BYTES: u64 = 128 << 10;
const BUDGET: u64 = if cfg!(target_os = "android") {
    32 << 20
} else {
    64 << 20
};

pub(crate) struct Store(super::derived::Files);
impl Store {
    pub fn from_env(dir: &Path) -> Option<Self> {
        super::derived::Files::open(dir, "decoded-v1", "RUFFLE_BITMAP_DECODE_CACHE", BUDGET)
            .map(Self)
    }
    pub fn stats(&self) -> (u64, u64) {
        self.0.stats()
    }
    pub fn decode(
        &self,
        source: EncodedBitmap<'_>,
        size: BitmapSize,
    ) -> Result<Bitmap<'static>, ruffle_render::error::Error> {
        let max = u64::from(size.width)
            .saturating_mul(size.height.into())
            .saturating_mul(4);
        if !(MIN_BYTES..=MAX_BYTES).contains(&max) {
            return source.decode();
        }
        let file = entry(&self.0.dir, &source, size);
        if let Some(bitmap) = read(&file, size) {
            self.0.note(true);
            return Ok(bitmap);
        }
        self.0.note(false);
        let bitmap = source.decode()?;
        if bitmap.width() == size.width
            && bitmap.height() == size.height
            && let Some(payload) = encode(&bitmap)
        {
            self.0.store(file, payload);
        }
        Ok(bitmap)
    }
}
fn entry(dir: &Path, source: &EncodedBitmap<'_>, size: BitmapSize) -> CacheFile {
    let mut hash = Sha256::new();
    // Decoder ABI includes channel order, JPEG alpha multiplication and lossless palette rules.
    hash.update(b"ruffle-symbol-pixels-v1");
    hash.update(std::env::consts::ARCH.as_bytes());
    hash.update(size.width.to_le_bytes());
    hash.update(size.height.to_le_bytes());
    match source {
        EncodedBitmap::Jpeg { data, alpha } => {
            hash.update([0, u8::from(alpha.is_some())]);
            hash.update((data.len() as u64).to_le_bytes());
            hash.update(data);
            if let Some(a) = alpha {
                hash.update(a)
            }
        }
        EncodedBitmap::Lossless(tag) => {
            let format = match tag.format {
                swf::BitmapFormat::Rgb15 => [1, 0],
                swf::BitmapFormat::Rgb32 => [2, 0],
                swf::BitmapFormat::ColorMap8 { num_colors } => [3, num_colors],
            };
            hash.update([1, tag.version]);
            hash.update(format);
            hash.update(tag.data.as_ref());
        }
    }
    let identity: Hash = hash.finalize();
    CacheFile {
        path: dir.join(format!("{identity:x}.pixels")),
        identity,
    }
}
fn encode(bitmap: &Bitmap<'_>) -> Option<Vec<u8>> {
    let id = match bitmap.format() {
        BitmapFormat::Rgb => 0,
        BitmapFormat::Rgba => 1,
        BitmapFormat::Bgra => 2,
        _ => return None,
    };
    let mut payload = Vec::with_capacity(1 + bitmap.data().len());
    payload.extend(bitmap.data());
    payload.push(id);
    Some(payload)
}
fn read(file: &CacheFile, size: BitmapSize) -> Option<Bitmap<'static>> {
    let mut bytes = file.load_bounded(MAX_BYTES + 1).ok()?;
    let format = match bytes.pop()? {
        0 => BitmapFormat::Rgb,
        1 => BitmapFormat::Rgba,
        2 => BitmapFormat::Bgra,
        _ => return None,
    };
    let length = u64::from(size.width)
        .checked_mul(size.height.into())?
        .checked_mul(format.bytes_per_pixel()?.get() as u64)?;
    (bytes.len() as u64 == length).then(|| Bitmap::new(size.width, size.height, format, bytes))
}
#[cfg(test)]
mod tests {
    use super::super::derived::write;
    use super::*;
    use std::fs;
    #[test]
    fn decoded_symbol_checks_input_alpha_format_dimensions_and_budget() {
        let temp = tempfile::tempdir_in(std::env::temp_dir().canonicalize().unwrap()).unwrap();
        let dir = super::super::private_dir(&temp.path().join("pixels")).unwrap();
        let size = BitmapSize {
            width: 256,
            height: 256,
        };
        let first = EncodedBitmap::Jpeg {
            data: b"abc",
            alpha: Some(b"d"),
        };
        let second = EncodedBitmap::Jpeg {
            data: b"abcd",
            alpha: None,
        };
        let file = entry(&dir, &first, size);
        let other = entry(&dir, &second, size);
        assert_ne!(file.identity, other.identity);
        let pixels = Bitmap::new(256, 256, BitmapFormat::Rgba, vec![0x41; 256 * 256 * 4]);
        let payload = encode(&pixels).unwrap();
        let budget = (payload.len() + super::super::HEADER_LEN) as u64;
        write(&file, &payload, budget).unwrap();
        assert_eq!(read(&file, size).unwrap().data(), pixels.data());
        assert!(
            read(
                &file,
                BitmapSize {
                    width: 255,
                    height: 256
                }
            )
            .is_none()
        );
        write(&other, &payload, budget).unwrap();
        assert!(!file.path.exists());
        let mut bad = fs::read(&other.path).unwrap();
        *bad.last_mut().unwrap() ^= 1;
        fs::write(&other.path, bad).unwrap();
        assert!(read(&other, size).is_none());
    }
}
