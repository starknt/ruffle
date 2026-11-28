//! Embedded bitmap frames keep their original lossless/JPEG source. An AS3
//! BitmapData can be displayed without also materializing a private CPU copy.
//! Only the display-list read path shares a texture; all writable GPU handles
//! and all modified CPU buffers are private to the BitmapData.
use super::bitmap_data::Color;
use crate::character::CompressedBitmap;
use ruffle_memory::{MemoryBuffer, MemoryKind, MemoryScope};
use ruffle_render::{
    backend::RenderBackend,
    bitmap::{Bitmap, BitmapFormat, BitmapHandle, BitmapHandleImpl, PixelRegion},
    matrix::Matrix,
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

/// A rectangle painted on a uniform image. Each texel of the compact view
/// stands for a power-of-two block of equal pixels in the logical bitmap.
struct RectFill {
    background: Rc<SolidFill>,
    color: Color,
    region: PixelRegion,
    width: u32,
    block: [u32; 2],
    expanded: RefCell<std::rc::Weak<PixelBacking>>,
}

impl RectFill {
    fn pixel(&self, x: u32, y: u32) -> Color {
        if (self.region.x_min..self.region.x_max).contains(&x)
            && (self.region.y_min..self.region.y_max).contains(&y)
        {
            self.color
        } else {
            self.background.color
        }
    }

    fn expand(&self) -> PixelBacking {
        let stride = self.width as usize;
        let mut pixels = vec![self.background.color; self.background.len];
        for y in self.region.y_min..self.region.y_max {
            let row = y as usize * stride;
            pixels[row + self.region.x_min as usize..row + self.region.x_max as usize]
                .fill(self.color);
        }
        PixelBacking::new(&self.background.scope, pixels)
    }
}

#[derive(Clone)]
enum PixelSource {
    Owned,
    Solid(Rc<SolidFill>),
    Rect(Rc<RectFill>),
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
    pub fn solid_color(&self) -> Option<Color> {
        match &self.source {
            PixelSource::Solid(fill) => Some(fill.color),
            _ => None,
        }
    }

    #[inline]
    pub fn pixel(&self, index: usize) -> Color {
        match &self.source {
            PixelSource::Solid(fill) => {
                assert!(index < fill.len);
                fill.color
            }
            PixelSource::Rect(fill) => {
                assert!(index < fill.background.len);
                fill.pixel(
                    (index % fill.width as usize) as u32,
                    (index / fill.width as usize) as u32,
                )
            }
            _ => self[index],
        }
    }

    pub fn fill_rect(&mut self, width: u32, height: u32, rect: PixelRegion, color: Color) {
        if let PixelSource::Solid(fill) = &self.source
            && self.data.get().is_none()
        {
            if fill.color == color {
                return;
            }
            let block_x = 1 << (width | rect.x_min | rect.x_max).trailing_zeros();
            let block_y = 1 << (height | rect.y_min | rect.y_max).trailing_zeros();
            // Keep a rectangular fill only when at most 256 equal-color blocks describe the image.
            if (block_x > 1 || block_y > 1)
                && u64::from(width / block_x) * u64::from(height / block_y) <= 256
            {
                self.source = PixelSource::Rect(Rc::new(RectFill {
                    background: fill.clone(),
                    color,
                    region: rect,
                    width,
                    block: [block_x, block_y],
                    expanded: Default::default(),
                }));
                return;
            }
        }
        let pixels = self.deref_mut();
        for y in rect.y_min..rect.y_max {
            let row = y as usize * width as usize;
            pixels[row + rect.x_min as usize..row + rect.x_max as usize].fill(color);
        }
    }

    pub fn nearest_view(
        &mut self,
        renderer: &mut dyn RenderBackend,
        matrix: &Matrix,
    ) -> Result<Option<(BitmapHandle, [u32; 2], PixelRegion)>, ruffle_render::error::Error> {
        let PixelSource::Rect(fill) = &self.source else {
            return Ok(None);
        };
        // Point filtering can round a fractional texel boundary differently on
        // different devices. Align blocks to attachment pixels and keep the
        // sample center well inside even the least precise sampler's texel.
        let [bx, by] = fill.block.map(|n| n.min(8));
        let aligned = |scale: f32, block: u32| {
            let size = scale * block as f32;
            (1.0..=4.0).contains(&size) && size.fract() == 0.0
        };
        if matrix.b != 0.0
            || matrix.c != 0.0
            || matrix.tx.get() % swf::Twips::TWIPS_PER_PIXEL != 0
            || matrix.ty.get() % swf::Twips::TWIPS_PER_PIXEL != 0
            || !aligned(matrix.a, bx)
            || !aligned(matrix.d, by)
        {
            return Ok(None);
        }
        let w = fill.width / bx;
        let h = (fill.background.len / fill.width as usize) as u32 / by;
        let handle = if let Some(handle) = &self.view {
            handle.clone()
        } else {
            let bytes: Vec<u8> = (0..h)
                .flat_map(|y| {
                    (0..w).flat_map(move |x| {
                        let c = fill.pixel(x * bx, y * by);
                        [c.red(), c.green(), c.blue(), c.alpha()]
                    })
                })
                .collect();
            let handle = renderer.register_bitmap_with_usage(
                Bitmap::new(w, h, BitmapFormat::Rgba, bytes),
                ruffle_render::backend::BitmapUsage::Excluded,
            )?;
            self.view = Some(handle.clone());
            handle
        };
        Ok(Some((handle, [bx, by], PixelRegion::for_whole_size(w, h))))
    }

    pub fn has_view(&self) -> bool {
        self.view.is_some()
    }

    pub fn release_view(&mut self) {
        self.view = None;
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
            PixelSource::Rect(fill) => PixelBacking::cached(&fill.expanded, || fill.expand()),
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
