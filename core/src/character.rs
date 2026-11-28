use std::cell::OnceCell;

use crate::backend::audio::SoundHandle;
use crate::binary_data::BinaryData;
use crate::display_object::{
    Avm1Button, Avm2Button, BitmapClass, EditText, Graphic, MorphShape, MovieClip, Text, Video,
};
use crate::font::Font;
use gc_arena::barrier::unlock;
use gc_arena::lock::Lock;
use gc_arena::{Collect, Gc, Mutation};
use ruffle_render::backend::RenderBackend;
use ruffle_render::bitmap::{Bitmap as RenderBitmap, BitmapHandle, BitmapSize};
use ruffle_render::error::Error as RenderError;
use swf::DefineBitsLossless;

#[derive(Copy, Clone, Collect, Debug)]
#[collect(no_drop)]
pub enum Character<'gc> {
    EditText(EditText<'gc>),
    Graphic(Graphic<'gc>),
    MovieClip(MovieClip<'gc>),
    Bitmap(Gc<'gc, BitmapCharacter<'gc>>),
    Avm1Button(Avm1Button<'gc>),
    Avm2Button(Avm2Button<'gc>),
    Font(Font<'gc>),
    MorphShape(MorphShape<'gc>),
    Text(Text<'gc>),
    Sound(Gc<'gc, SoundHandle>),
    Video(Video<'gc>),
    BinaryData(Gc<'gc, BinaryData>),
}

#[derive(Collect, Debug)]
#[collect(no_drop)]
pub struct BitmapCharacter<'gc> {
    #[collect(require_static)]
    compressed: std::rc::Rc<crate::bitmap::symbol_pixels::SymbolImage>,
    /// A lazily constructed GPU handle, used when performing fills with this bitmap
    #[collect(require_static)]
    handle: OnceCell<BitmapHandle>,
    /// The bitmap class set by `SymbolClass` - this is used when we instantaite
    /// a `Bitmap` displayobject.
    avm2_class: Lock<BitmapClass<'gc>>,
}

impl<'gc> BitmapCharacter<'gc> {
    pub fn new(symbol: ruffle_memory::SymbolId, compressed: CompressedBitmap) -> Self {
        Self {
            compressed: std::rc::Rc::new(crate::bitmap::symbol_pixels::SymbolImage::new(
                Some(symbol),
                compressed,
            )),
            handle: OnceCell::default(),
            avm2_class: Lock::new(BitmapClass::NoSubclass),
        }
    }

    pub fn compressed(&self) -> &CompressedBitmap {
        &self.compressed.data
    }

    pub(crate) fn symbol_image(&self) -> std::rc::Rc<crate::bitmap::symbol_pixels::SymbolImage> {
        self.compressed.clone()
    }

    pub fn avm2_class(&self) -> BitmapClass<'gc> {
        self.avm2_class.get()
    }

    pub fn set_avm2_class(this: Gc<'gc, Self>, bitmap_class: BitmapClass<'gc>, mc: &Mutation<'gc>) {
        unlock!(Gc::write(mc, this), Self, avm2_class).set(bitmap_class);
    }

    pub fn bitmap_handle(
        &self,
        backend: &mut dyn RenderBackend,
    ) -> Result<BitmapHandle, RenderError> {
        // FIXME - use `OnceCell::get_or_try_init` when stabilized.
        if let Some(handle) = self.handle.get() {
            return Ok(handle.clone());
        }
        let new_handle = self.compressed.view(backend)?;
        if !backend.cache_immutable_bitmap(&new_handle) {
            let _ = self.handle.set(new_handle.clone());
        }
        Ok(new_handle)
    }
}

/// Holds a bitmap from an SWF tag, plus the decoded width/height.
/// We avoid decompressing the image until it's actually needed - some pathological SWFS
/// like 'House' have thousands of highly-compressed (mostly empty) bitmaps, which can
/// take over 10GB of ram if we decompress them all during preloading.
#[derive(Clone, Debug)]
pub enum CompressedBitmap {
    Jpeg {
        data: Vec<u8>,
        alpha: Option<Vec<u8>>,
        width: u32,
        height: u32,
    },
    Lossless(DefineBitsLossless<'static>),
}

impl CompressedBitmap {
    pub(crate) fn shared_key(&self) -> ruffle_memory::ContentId {
        use ruffle_memory::ContentId;
        match self {
            Self::Jpeg {
                data,
                alpha,
                width,
                height,
            } => ContentId::from_parts(&[
                b"jpeg",
                &width.to_le_bytes(),
                &height.to_le_bytes(),
                &[u8::from(alpha.is_some())],
                data,
                alpha.as_deref().unwrap_or_default(),
            ]),
            Self::Lossless(tag) => {
                let format = match tag.format {
                    swf::BitmapFormat::ColorMap8 { num_colors } => [3, num_colors],
                    swf::BitmapFormat::Rgb15 => [4, 0],
                    swf::BitmapFormat::Rgb32 => [5, 0],
                };
                ContentId::from_parts(&[
                    b"lossless",
                    &[tag.version],
                    &format,
                    &tag.width.to_le_bytes(),
                    &tag.height.to_le_bytes(),
                    &tag.data,
                ])
            }
        }
    }

    pub(crate) fn capacity(&self) -> usize {
        match self {
            Self::Jpeg { data, alpha, .. } => {
                data.capacity() + alpha.as_ref().map_or(0, Vec::capacity)
            }
            Self::Lossless(tag) => match &tag.data {
                std::borrow::Cow::Owned(bytes) => bytes.capacity(),
                std::borrow::Cow::Borrowed(_) => 0,
            },
        }
    }

    pub fn size(&self) -> BitmapSize {
        match self {
            CompressedBitmap::Jpeg { width, height, .. } => BitmapSize {
                width: *width,
                height: *height,
            },
            CompressedBitmap::Lossless(define_bits_lossless) => BitmapSize {
                width: define_bits_lossless.width.into(),
                height: define_bits_lossless.height.into(),
            },
        }
    }
    pub(crate) fn encoded(&self) -> ruffle_render::bitmap::EncodedBitmap<'_> {
        use ruffle_render::bitmap::EncodedBitmap;
        match self {
            Self::Jpeg { data, alpha, .. } => EncodedBitmap::Jpeg {
                data,
                alpha: alpha.as_deref(),
            },
            Self::Lossless(tag) => EncodedBitmap::Lossless(tag),
        }
    }
    pub fn decode(&self) -> Result<RenderBitmap<'static>, RenderError> {
        self.encoded().decode()
    }
}
