//! Embedded bitmap frames keep their original lossless/JPEG source. An AS3
//! BitmapData can be displayed without also materializing a private CPU copy.
//! Only the display-list read path shares a texture; all writable GPU handles
//! and all modified CPU buffers are private to the BitmapData.
use super::bitmap_data::Color;
use crate::character::CompressedBitmap;
use ruffle_render::{
    backend::RenderBackend,
    bitmap::{BitmapHandle, BitmapHandleImpl},
    profile::BitmapAllocation,
};
use std::{
    cell::{OnceCell, RefCell},
    ops::{Deref, DerefMut},
    rc::Rc,
    sync::Weak,
};

#[cfg(test)]
pub(crate) mod tests;

#[derive(Debug)]
pub struct EncodedSymbol {
    data: CompressedBitmap,
    _charge: ruffle_memory::MemoryLease,
}
impl Deref for EncodedSymbol {
    type Target = CompressedBitmap;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

#[derive(Debug)]
pub struct SymbolImage {
    pub data: std::sync::Arc<EncodedSymbol>,
    symbol: Option<ruffle_memory::SymbolId>,
    decoded: RefCell<std::rc::Weak<PixelBacking>>,
    view: RefCell<Option<Weak<dyn BitmapHandleImpl>>>,
}
impl SymbolImage {
    pub fn new(symbol: Option<ruffle_memory::SymbolId>, data: CompressedBitmap) -> Self {
        let scope = ruffle_memory::MemoryScope::current();
        let store = scope.manager().resources::<EncodedSymbol>();
        let data = store.intern(data.shared_key(), || {
            let charge = scope.manager().shared().track(
                ruffle_memory::MemoryKind::EncodedBitmap,
                data.capacity() as u64,
            );
            std::sync::Arc::new(EncodedSymbol {
                data,
                _charge: charge,
            })
        });
        Self {
            data,
            symbol,
            decoded: Default::default(),
            view: Default::default(),
        }
    }
    pub(crate) fn view(
        &self,
        renderer: &mut dyn RenderBackend,
    ) -> Result<BitmapHandle, ruffle_render::error::Error> {
        if let Some(h) = self.view.borrow().as_ref().and_then(Weak::upgrade) {
            return Ok(BitmapHandle(h));
        }
        let decoded = renderer.decode_bitmap_symbol(self.data.encoded(), self.data.size())?;
        let h = ruffle_memory::SymbolId::enter(self.symbol, || {
            renderer
                .register_bitmap_with_usage(decoded, ruffle_render::backend::BitmapUsage::Symbol)
        })?;
        *self.view.borrow_mut() = Some(std::sync::Arc::downgrade(&h.0));
        Ok(h)
    }
}
struct PixelBacking {
    colors: ruffle_memory::MemoryBuffer<Color>,
    _profile: BitmapAllocation,
}

impl PixelBacking {
    fn new(pixels: Vec<Color>) -> Self {
        let profile = BitmapAllocation::new(pixels.capacity() * size_of::<Color>());
        Self {
            colors: ruffle_memory::MemoryBuffer::new(
                &ruffle_memory::MemoryScope::current(),
                ruffle_memory::MemoryKind::BitmapPixels,
                pixels,
            ),
            _profile: profile,
        }
    }
}

impl Clone for PixelBacking {
    fn clone(&self) -> Self {
        Self::new(self.colors.to_vec())
    }
}

#[derive(Clone)]
pub struct Pixels {
    symbol: Option<Rc<SymbolImage>>,
    view: Option<BitmapHandle>,
    data: OnceCell<Rc<PixelBacking>>,
}
impl Pixels {
    pub fn owned(p: Vec<Color>) -> Self {
        Self {
            symbol: None,
            view: None,
            data: OnceCell::from(Rc::new(PixelBacking::new(p))),
        }
    }
    pub fn symbol(s: Rc<SymbolImage>) -> Self {
        Self {
            symbol: Some(s),
            view: None,
            data: OnceCell::new(),
        }
    }
    pub fn view(
        &mut self,
        renderer: &mut dyn RenderBackend,
    ) -> Result<Option<BitmapHandle>, ruffle_render::error::Error> {
        renderer.health().check()?;
        let Some(s) = self.symbol.as_ref() else {
            return Ok(None);
        };
        if let Some(view) = &self.view {
            return Ok(Some(view.clone()));
        }
        let view = s.view(renderer)?;
        // Only unmodified symbols reach this path. Their compressed source can
        // recreate an evicted upload without readback or changing AS3 lifetime.
        if !renderer.cache_immutable_bitmap(&view) {
            self.view = Some(view.clone());
        }
        Ok(Some(view))
    }
    fn materialize(&self) -> &Rc<PixelBacking> {
        self.data.get_or_init(|| {
            let symbol = self
                .symbol
                .as_ref()
                .expect("unmaterialized bitmap has a symbol");
            if let Some(pixels) = symbol.decoded.borrow().upgrade() {
                return pixels;
            }
            let decoded = symbol.data.decode().expect("Failed to decode BitmapData");
            let pixels = Rc::new(PixelBacking::new(
                decoded.as_colors().map(Color::from).collect(),
            ));
            *symbol.decoded.borrow_mut() = Rc::downgrade(&pixels);
            pixels
        })
    }
}
impl Deref for Pixels {
    type Target = [Color];
    fn deref(&self) -> &Self::Target {
        &self.materialize().colors
    }
}
impl DerefMut for Pixels {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.materialize();
        self.symbol = None;
        self.view = None;
        &mut Rc::make_mut(self.data.get_mut().unwrap()).colors
    }
}
