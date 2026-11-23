use freetype::bitmap::PixelMode;
use freetype::face::KerningMode;
use freetype::face::LoadFlag;
use ruffle_core::font::FontAtlases;
use std::ffi::OsStr;
use thiserror::Error;

use ruffle_core::font::FontMetrics;
use ruffle_core::font::FontRenderer;
use ruffle_core::font::Glyph;
use ruffle_core::swf::Twips;
use ruffle_render::bitmap::Bitmap;
use ruffle_render::bitmap::BitmapFormat;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Missing glyph: {0}")]
    MissingGlyph(char),

    #[error("FreeType error: {0}")]
    FreetypeError(#[from] freetype::Error),

    #[error("Unsupported FreeType glyph pixel mode: {0:?}")]
    UnsupportedPixelMode(PixelMode),
}

pub struct FreetypeFontRenderer {
    face: freetype::Face,
    atlases: FontAtlases,
}

impl std::fmt::Debug for FreetypeFontRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FreetypeFontRenderer")
            .field("family_name", &self.face.family_name())
            .field("style_name", &self.face.style_name())
            .finish()
    }
}

impl FreetypeFontRenderer {
    /// Render fonts with size 64px. It affects the bitmap size.
    const SIZE_PX: f64 = 64.0;

    /// Divide each pixel into 20 (use twips precision). It affects metrics.
    const SCALE: f64 = 20.0;

    pub fn new<P>(path: P, face_index: u32, atlases: &FontAtlases) -> Result<Self, Error>
    where
        P: AsRef<OsStr>,
    {
        let ft = freetype::Library::init()?;
        let face = ft.new_face(path, face_index as isize)?;
        face.set_char_size((Self::SIZE_PX * 64.0) as isize, 0, 0, 0)?;
        Ok(Self {
            face,
            atlases: atlases.clone(),
        })
    }

    fn size_metrics(&self) -> freetype::ffi::FT_Size_Metrics {
        self.face
            .size_metrics()
            .expect("face should have a char size set")
    }

    fn ascent(&self) -> Twips {
        convert_26_6_to_twips(self.size_metrics().ascender)
    }

    fn descent(&self) -> Twips {
        // FreeType's `descender` is a signed distance from the baseline
        // (negative since it points below it); this returns the positive
        // magnitude expected by `FontMetrics::descent`.
        convert_26_6_to_twips(-self.size_metrics().descender)
    }

    fn render_glyph_internal(&self, character: char) -> Result<Glyph, Error> {
        let index = self
            .face
            .get_char_index(character as usize)
            .ok_or(Error::MissingGlyph(character))?;

        // These atlas glyphs are scaled again at display time. Auto-hinting
        // to the atlas pixel grid can shift an unhinted outline by a row.
        self.face.load_glyph(
            index,
            LoadFlag::NO_BITMAP | LoadFlag::NO_AUTOHINT | LoadFlag::RENDER,
        )?;
        let glyph = self.face.glyph();
        let bitmap = glyph.bitmap();
        let advance = convert_26_6_to_twips(glyph.advance().x);

        // Glyphs with no ink (e.g. space) have an empty bitmap. Skip the
        // atlas entirely for these.
        if bitmap.width() == 0 || bitmap.rows() == 0 {
            return Ok(Glyph::whitespace(character, advance));
        }

        let tx = Twips::from_pixels(glyph.bitmap_left() as f64);

        // `bitmap_top` is the distance from the baseline up to the top of
        // the bitmap. Glyph bitmaps are positioned relative to the top of
        // the line, not the baseline, so convert it into a downward offset
        // from the top of the line.
        let bitmap_top = Twips::from_pixels(glyph.bitmap_top() as f64);
        let ty = self.ascent() - bitmap_top;

        let bitmap = Bitmap::new(
            bitmap.width() as u32,
            bitmap.rows() as u32,
            BitmapFormat::Rgba,
            convert_bitmap(&bitmap)?,
        );
        let atlas_glyph = self.atlases.rgba().new_glyph(bitmap, tx, ty);

        Ok(Glyph::from_atlas(character, atlas_glyph, advance))
    }

    fn calculate_kerning_internal(&self, left: char, right: char) -> Result<Twips, Error> {
        if !self.face.has_kerning() {
            return Ok(Twips::ZERO);
        }

        let left_index = self
            .face
            .get_char_index(left as usize)
            .ok_or(Error::MissingGlyph(left))?;
        let right_index = self
            .face
            .get_char_index(right as usize)
            .ok_or(Error::MissingGlyph(right))?;

        let kerning =
            self.face
                .get_kerning(left_index, right_index, KerningMode::KerningDefault)?;

        Ok(convert_26_6_to_twips(kerning.x))
    }
}

impl FontRenderer for FreetypeFontRenderer {
    fn scale(&self) -> f32 {
        (Self::SIZE_PX * Self::SCALE) as f32
    }

    fn get_font_metrics(&self) -> FontMetrics {
        FontMetrics {
            scale: self.scale(),
            ascent: self.ascent().get(),
            descent: self.descent().get(),
            leading: 0,
        }
    }

    fn has_kerning_info(&self) -> bool {
        self.face.has_kerning()
    }

    fn render_glyph(&self, character: char) -> Option<Glyph> {
        self.render_glyph_internal(character)
            .map_err(|err| tracing::error!("Failed to render a glyph: {err:?}"))
            .ok()
    }

    fn calculate_kerning(&self, left: char, right: char) -> Twips {
        self.calculate_kerning_internal(left, right)
            .map_err(|err| tracing::error!("Failed to calculate kerning: {err:?}"))
            .unwrap_or(Twips::ZERO)
    }

    fn atlases(&self) -> Option<&FontAtlases> {
        Some(&self.atlases)
    }
}

/// Converts a FreeType 26.6 fixed-point value (1/64th of a pixel) to Twips
/// (1/20th of a pixel), rounding to the nearest twip.
///
/// Note: the FreeType type differs between 64-bit and 32-bit architectures,
/// hence the `Into<i64>`.
fn convert_26_6_to_twips(value_26_6: impl Into<i64>) -> Twips {
    let value_26_6 = value_26_6.into();
    let whole_pixels = (value_26_6 / 64) as i32;

    let fractional_twips = (value_26_6 % 64) as f64 / 64.0 * 20.0;
    let fractional_twips = fractional_twips.round_ties_even() as i32;

    Twips::new(fractional_twips + whole_pixels * 20)
}

fn convert_bitmap(bitmap: &freetype::Bitmap) -> Result<Vec<u8>, Error> {
    let mode = bitmap.pixel_mode()?;
    let bits = match mode {
        PixelMode::Mono => 1,
        PixelMode::Gray2 => 2,
        PixelMode::Gray4 => 4,
        PixelMode::Gray => 8,
        _ => return Err(Error::UnsupportedPixelMode(mode)),
    };
    let mask = (1u16 << bits) - 1;
    let max_coverage = match (mode, bitmap.raw().num_grays) {
        (PixelMode::Gray, levels @ 2..=256) => (levels - 1) as u16,
        _ => mask,
    };
    let width = bitmap.width() as usize;
    let height = bitmap.rows() as usize;
    let stride = bitmap.pitch().unsigned_abs() as usize;
    let buffer = bitmap.buffer();
    let mut rgba = Vec::with_capacity(width * height * 4);
    for y in 0..height {
        let row_index = if bitmap.pitch() < 0 {
            height - 1 - y
        } else {
            y
        };
        let row = &buffer[row_index * stride..(row_index + 1) * stride];
        if bits == 8 && max_coverage == 255 {
            for &alpha in &row[..width] {
                rgba.extend_from_slice(&[alpha; 4]);
            }
        } else {
            for x in 0..width {
                let bit = x * bits;
                let sample = (u16::from(row[bit / 8]) >> (8 - bits - bit % 8)) & mask;
                let alpha = (sample * 255 / max_coverage).min(255) as u8;
                rgba.extend_from_slice(&[alpha; 4]);
            }
        }
    }
    Ok(rgba)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glyph_rows_respect_coverage_depth_pitch_and_flow() {
        use freetype::ffi::{
            FT_PIXEL_MODE_GRAY, FT_PIXEL_MODE_GRAY2, FT_PIXEL_MODE_GRAY4, FT_PIXEL_MODE_MONO,
        };

        fn check(mode: u32, pitch: i32, levels: i16, bytes: &mut [u8], expected: &[u8]) {
            let raw = freetype::ffi::FT_Bitmap {
                rows: 2,
                width: 3,
                pitch,
                buffer: bytes.as_mut_ptr(),
                num_grays: levels,
                pixel_mode: mode as _,
                palette_mode: 0,
                palette: std::ptr::null_mut(),
            };
            assert_eq!(bytes.len(), 2 * pitch.unsigned_abs() as usize);
            // Both the descriptor and its backing rows live through conversion.
            let bitmap = unsafe { freetype::Bitmap::from_raw(&raw) };
            let rgba = convert_bitmap(&bitmap).unwrap();
            assert_eq!(
                rgba,
                expected.iter().flat_map(|&a| [a; 4]).collect::<Vec<_>>()
            );
        }

        check(
            FT_PIXEL_MODE_GRAY,
            4,
            256,
            &mut [1, 2, 3, 0, 4, 5, 6, 0],
            &[1, 2, 3, 4, 5, 6],
        );
        check(
            FT_PIXEL_MODE_GRAY,
            3,
            4,
            &mut [0, 1, 2, 3, 2, 1],
            &[0, 85, 170, 255, 170, 85],
        );
        check(
            FT_PIXEL_MODE_MONO,
            -2,
            2,
            &mut [0x40, 0, 0xa0, 0],
            &[255, 0, 255, 0, 255, 0],
        );
        check(
            FT_PIXEL_MODE_GRAY2,
            1,
            4,
            &mut [0x18, 0xe4],
            &[0, 85, 170, 255, 170, 85],
        );
        check(
            FT_PIXEL_MODE_GRAY4,
            2,
            16,
            &mut [0x05, 0xa0, 0xfa, 0x50],
            &[0, 85, 170, 255, 170, 85],
        );
    }
}
