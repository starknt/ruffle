//! Embedded bitmap frames keep their original lossless/JPEG source. An AS3
//! BitmapData can be displayed without also materializing a private CPU copy.
//! Only the display-list read path shares a texture; all writable GPU handles
//! and all modified CPU buffers are private to the BitmapData.
use super::bitmap_data::Color;
use crate::character::CompressedBitmap;
use ruffle_memory::{MemoryBuffer, MemoryKind, MemoryScope};
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
    colors: MemoryBuffer<Color>,
    _profile: BitmapAllocation,
}

impl PixelBacking {
    fn new(scope: &MemoryScope, pixels: Vec<Color>) -> Self {
        Self::from_buffer(MemoryBuffer::new(scope, MemoryKind::BitmapPixels, pixels))
    }

    fn from_buffer(colors: MemoryBuffer<Color>) -> Self {
        let profile = BitmapAllocation::new(colors.capacity() * size_of::<Color>());
        Self {
            colors,
            _profile: profile,
        }
    }

    fn cached(cache: &RefCell<std::rc::Weak<Self>>, create: impl FnOnce() -> Self) -> Rc<Self> {
        if let Some(pixels) = cache.borrow().upgrade() {
            return pixels;
        }
        let pixels = Rc::new(create());
        *cache.borrow_mut() = Rc::downgrade(&pixels);
        pixels
    }
}

impl Clone for PixelBacking {
    fn clone(&self) -> Self {
        Self::from_buffer(self.colors.clone())
    }
}

struct SolidFill {
    len: usize,
    color: Color,
    scope: MemoryScope,
    expanded: RefCell<std::rc::Weak<PixelBacking>>,
}

#[derive(Clone)]
enum PixelSource {
    Owned,
    Solid(Rc<SolidFill>),
    Symbol(Rc<SymbolImage>),
}

#[derive(Clone)]
pub struct Pixels {
    source: PixelSource,
    view: Option<BitmapHandle>,
    data: OnceCell<Rc<PixelBacking>>,
}
impl Pixels {
    pub fn owned(p: Vec<Color>) -> Self {
        Self {
            source: PixelSource::Owned,
            view: None,
            data: OnceCell::from(Rc::new(PixelBacking::new(&MemoryScope::current(), p))),
        }
    }
    pub fn solid(len: usize, color: Color, scope: MemoryScope) -> Self {
        Self {
            source: PixelSource::Solid(Rc::new(SolidFill {
                len,
                color,
                scope,
                expanded: RefCell::default(),
            })),
            view: None,
            data: OnceCell::new(),
        }
    }
    pub fn symbol(s: Rc<SymbolImage>) -> Self {
        Self {
            source: PixelSource::Symbol(s),
            view: None,
            data: OnceCell::new(),
        }
    }
    #[inline]
    pub fn pixel(&self, index: usize) -> Color {
        if let PixelSource::Solid(fill) = &self.source {
            assert!(index < fill.len);
            fill.color
        } else {
            self[index]
        }
    }

    pub fn fill(&mut self, color: Color) {
        match &self.source {
            PixelSource::Solid(fill) if fill.color == color => {}
            PixelSource::Solid(fill)
                if !self
                    .data
                    .get()
                    .is_some_and(|data| Rc::strong_count(data) == 1) =>
            {
                *self = Self::solid(fill.len, color, fill.scope.clone());
            }
            // Displayed bitmaps typically have a private upload backing. Refill it in place.
            _ => self.deref_mut().fill(color),
        }
    }
    pub fn view(
        &mut self,
        renderer: &mut dyn RenderBackend,
    ) -> Result<Option<BitmapHandle>, ruffle_render::error::Error> {
        renderer.health().check()?;
        let PixelSource::Symbol(s) = &self.source else {
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
        self.data.get_or_init(|| match &self.source {
            PixelSource::Solid(fill) => PixelBacking::cached(&fill.expanded, || {
                PixelBacking::new(&fill.scope, vec![fill.color; fill.len])
            }),
            PixelSource::Symbol(symbol) => PixelBacking::cached(&symbol.decoded, || {
                let decoded = symbol.data.decode().expect("Failed to decode BitmapData");
                PixelBacking::new(
                    &MemoryScope::current(),
                    decoded.as_colors().map(Color::from).collect(),
                )
            }),
            PixelSource::Owned => unreachable!("owned pixels have a backing"),
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
        self.source = PixelSource::Owned;
        self.view = None;
        &mut Rc::make_mut(self.data.get_mut().unwrap()).colors
    }
}
