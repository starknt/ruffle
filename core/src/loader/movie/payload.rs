//! Reserve loader payloads before decoding. Decoder workspace and the navigator's
//! backing allocations belong to their respective implementations.
use super::{ContentType, PreparedContent};
use bytes::Bytes;
use swf::error::Error;

pub(super) struct MoviePayload {
    pub source: Bytes,
    pub transport_len: usize,
    pub kind: ContentType,
    pub decode_peak: usize,
    pub image_error: Option<String>,
    swf_capacity: usize,
}

impl MoviePayload {
    // These probes can allocate (DER extraction, JPEG repair and JPEG-XR metadata).
    // Run them under exclusive decode admission before settling to the known size.
    pub fn needs_exclusive_probe(source: &[u8]) -> bool {
        source.first() == Some(&0x30)
            || matches!(
                ContentType::sniff(source),
                ContentType::Jpeg | ContentType::JpegXr
            )
    }

    pub fn inspect(transport: Bytes) -> Result<Self, Error> {
        let transport_len = transport.len();
        let inner = (transport.first() == Some(&0x30))
            .then(|| swf::read::extract_swz(&transport).ok())
            .flatten();
        let extracted = inner.as_ref().map_or(0, Vec::len);
        let source = inner.map_or(transport, Bytes::from);
        let kind = ContentType::sniff(&source);
        let mut image_error = None;
        let mut swf_capacity = 0;
        let overflow = || Error::invalid_data("Movie payload size overflow");
        let decode_peak = if kind == ContentType::Swf {
            let size = source
                .get(4..8)
                .ok_or_else(|| Error::invalid_data("Truncated SWF header"))?;
            let declared = u32::from_le_bytes(size.try_into().unwrap()) as usize;
            let body = declared
                .checked_sub(8)
                .ok_or_else(|| Error::invalid_data("Malformed SWF length"))?;
            // decompress_swf caps its initial Vec at 128 MiB. Larger bodies
            // can retain the next doubled capacity, including on mmap failure.
            let capacity = if body > 128 * 1024 * 1024 {
                body.checked_next_power_of_two().ok_or_else(overflow)?
            } else {
                body
            };
            let fws = declared.checked_add(32).ok_or_else(overflow)?;
            swf_capacity = capacity.max(fws);
            swf_capacity
                .checked_add(fws)
                .and_then(|n| n.checked_add(extracted))
                .ok_or_else(overflow)?
        } else if kind.is_image() {
            // JPEG marker repair can grow a full-size source copy before the
            // pixel plane is returned. Vec growth can double its capacity.
            let source_bytes = if kind == ContentType::Jpeg {
                source
                    .len()
                    .checked_add(2)
                    .and_then(|n| n.checked_mul(2))
                    .and_then(|n| n.checked_add(source.len()))
                    .ok_or_else(overflow)?
            } else {
                source.len()
            };
            match image_geometry(kind, &source) {
                Ok((w, h, bytes_per_pixel)) => (w as usize)
                    .checked_mul(h as usize)
                    .and_then(|n| n.checked_mul(bytes_per_pixel))
                    .and_then(|n| n.checked_add(source_bytes))
                    .ok_or_else(overflow)?,
                Err(error) => {
                    image_error = Some(error);
                    source_bytes
                }
            }
        } else {
            source.len()
        };
        Ok(Self {
            source,
            transport_len,
            kind,
            decode_peak: decode_peak.max(extracted),
            image_error,
            swf_capacity,
        })
    }

    pub fn handoff_bytes(&self, content: &PreparedContent) -> usize {
        match content {
            // MovieBytes keeps the fallback Vec capacity private. Charge its
            // upper bound even when the final mapping is smaller.
            PreparedContent::Movie(_) => self.swf_capacity,
            PreparedContent::Image { source, pixels } => {
                source.len()
                    + pixels.as_ref().map_or(0, |p| {
                        p.colors.capacity() * size_of::<crate::bitmap::bitmap_data::Color>()
                    })
            }
            PreparedContent::Other(body) => body.len(),
        }
    }
}

fn image_geometry(kind: ContentType, source: &[u8]) -> Result<(u32, u32, usize), String> {
    // IHDR precedes PNG metadata. Reading these fixed bytes avoids allocating
    // a PNG reader before admission. The real decoder validates the chunks.
    if kind == ContentType::Png {
        let header = source
            .get(12..24)
            .and_then(|bytes| bytes.strip_prefix(b"IHDR"))
            .ok_or("Invalid PNG header")?;
        let (w, h) = header.split_at(4);
        return Ok((
            u32::from_be_bytes(w.try_into().unwrap()),
            u32::from_be_bytes(h.try_into().unwrap()),
            8,
        ));
    }
    // GIF decodes the first image descriptor, which can differ from the logical screen.
    if kind == ContentType::Gif {
        return gif_frame(source)
            .map(|(w, h)| (w.into(), h.into(), 8))
            .ok_or_else(|| "Invalid GIF image descriptor".into());
    }
    #[cfg(feature = "jpegxr")]
    if kind == ContentType::JpegXr {
        let decoder = jpegxr::ImageDecode::with_reader(std::io::Cursor::new(source))
            .map_err(|e| e.to_string())?;
        let (w, h) = decoder.get_size().map_err(|e| e.to_string())?;
        let format = decoder.get_pixel_format().map_err(|e| e.to_string())?;
        let stride = jpegxr::PixelInfo::from_format(format)
            .bits_per_pixel()
            .div_ceil(8);
        return Ok((
            w.try_into().map_err(|_| "Invalid JPEG-XR width")?,
            h.try_into().map_err(|_| "Invalid JPEG-XR height")?,
            stride.saturating_mul(2).saturating_add(8),
        ));
    }
    let (w, h) = ruffle_render::utils::decode_define_bits_jpeg_dimensions(source)
        .map_err(|e| e.to_string())?;
    Ok((w, h, 8)) // decoded plane and VM colors can coexist
}

fn gif_frame(source: &[u8]) -> Option<(u16, u16)> {
    let flags = *source.get(10)?;
    let palette = if flags & 0x80 != 0 {
        3 << ((flags & 7) + 1)
    } else {
        0
    };
    let mut blocks = source.get(13 + palette..)?;
    while blocks.first() == Some(&0x21) {
        blocks = blocks.get(2..)?;
        loop {
            let (&len, rest) = blocks.split_first()?;
            blocks = rest.get(len as usize..)?;
            if len == 0 {
                break;
            }
        }
    }
    let [0x2c, _, _, _, _, w0, w1, h0, h1, ..] = blocks else {
        return None;
    };
    Some((
        u16::from_le_bytes([*w0, *w1]),
        u16::from_le_bytes([*h0, *h1]),
    ))
}
