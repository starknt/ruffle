use crate::avm2::object::BitmapDataObject;
use crate::context::RenderContext;
use crate::display_object::{BoundsMode, DisplayObject, DisplayObjectWeak, TDisplayObject};
use bitflags::bitflags;
use gc_arena::lock::GcRefLock;
use gc_arena::{Collect, Gc, Mutation};
use ruffle_render::backend::RenderBackend;
use ruffle_render::bitmap::coverage::Coverage;
use ruffle_render::bitmap::{
    Bitmap, BitmapFormat, BitmapHandle, LoaderRequest, PixelRegion, PixelSnapping, SyncHandle,
};
use ruffle_render::error::{RenderFailure, RenderFailureKind};
use ruffle_wstr::WStr;
use std::cell::Ref;
use std::fmt::Debug;
use std::ops::Range;
use swf::{Rectangle, Twips};
use tracing::instrument;

/// An implementation of the Lehmer/Park-Miller random number generator
/// Uses the fixed parameters m = 2,147,483,647 and a = 16,807
pub struct LehmerRng {
    x: u32,
}

impl LehmerRng {
    pub fn with_seed(seed: u32) -> Self {
        Self { x: seed }
    }

    /// Generate the next value in the sequence via the following formula
    /// X_(k+1) = a * X_k mod m
    pub fn random(&mut self) -> u32 {
        self.x = ((self.x as u64).overflowing_mul(16_807).0 % 2_147_483_647) as u32;
        self.x
    }

    pub fn random_range(&mut self, rng: Range<u8>) -> u8 {
        rng.start + (self.random() % ((rng.end - rng.start) as u32 + 1)) as u8
    }
}

/// This can represent both a premultiplied and an unmultiplied ARGB color value.
///
/// Note that most operations only make sense on one of these representations:
/// For example, blending on premultiplied values, and applying a `ColorTransform` on
/// unmultiplied values. Make sure to convert the color to the correct form beforehand.
// TODO: Maybe split the type into `PremultipliedColor(u32)` and
//   `UnmultipliedColor(u32)`?
#[derive(
    Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, bytemuck::Pod, bytemuck::Zeroable,
)]
#[repr(C, align(4))]
pub struct Color {
    // Byte order of a little-endian ActionScript ARGB word.
    b: u8,
    g: u8,
    r: u8,
    a: u8,
}

#[derive(Debug, Clone)]
pub enum BitmapDataDrawError {
    Unimplemented,
    Render(RenderFailure),
}

impl From<RenderFailure> for BitmapDataDrawError {
    fn from(error: RenderFailure) -> Self {
        Self::Render(error)
    }
}

pub enum BitmapDataWriteError {
    ByteArray(crate::avm2::bytearray::ByteArrayError),
    Render(RenderFailure),
}
impl From<crate::avm2::bytearray::ByteArrayError> for BitmapDataWriteError {
    fn from(error: crate::avm2::bytearray::ByteArrayError) -> Self {
        Self::ByteArray(error)
    }
}
impl From<RenderFailure> for BitmapDataWriteError {
    fn from(error: RenderFailure) -> Self {
        Self::Render(error)
    }
}
impl BitmapDataWriteError {
    pub fn to_avm<'gc>(
        self,
        activation: &mut crate::avm2::Activation<'_, 'gc>,
    ) -> crate::avm2::Error<'gc> {
        match self {
            Self::ByteArray(error) => error.to_avm(activation),
            Self::Render(error) => error.into(),
        }
    }
}

impl Color {
    #[must_use]
    pub fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    #[must_use]
    pub fn bgra_u32(bgra: u32) -> Self {
        let [b, g, r, a] = bgra.to_le_bytes();
        Self { r, g, b, a }
    }

    pub fn to_bgra_u32(&self) -> u32 {
        u32::from_le_bytes([self.b, self.g, self.r, self.a])
    }

    pub fn blue(&self) -> u8 {
        self.b
    }

    pub fn green(&self) -> u8 {
        self.g
    }

    pub fn red(&self) -> u8 {
        self.r
    }

    pub fn alpha(&self) -> u8 {
        self.a
    }

    #[must_use]
    pub fn to_premultiplied_alpha(self, transparency: bool) -> Self {
        // This has some accuracy issues with some alpha values

        let old_alpha = if transparency { self.alpha() } else { 255 };

        let a = old_alpha as u32;
        let r = ((self.red() as u32 * a + 127) / 255) as u8;
        let g = ((self.green() as u32 * a + 127) / 255) as u8;
        let b = ((self.blue() as u32 * a + 127) / 255) as u8;

        Self::rgba(r, g, b, old_alpha)
    }

    #[must_use]
    pub fn to_un_multiplied_alpha(self) -> Self {
        let alpha_factor = ruffle_render::bitmap::FLASH_PREMUL_FACTOR[self.alpha() as usize];
        let unmultiply = |c| ((c as u32 * alpha_factor + 0x8000) >> 16) as u8;

        let r = unmultiply(self.red());
        let g = unmultiply(self.green());
        let b = unmultiply(self.blue());

        Self::rgba(r, g, b, self.alpha())
    }

    #[must_use]
    pub fn with_alpha(&self, alpha: u8) -> Self {
        Self::rgba(self.red(), self.green(), self.blue(), alpha)
    }

    /// # Arguments
    ///
    /// * `self` - Must be in premultiplied form.
    /// * `source` - Must be in premultiplied form.
    #[must_use]
    pub fn blend_over(&self, source: &Self) -> Self {
        // The following formula was verified against Flash Player output across
        // the full 0..255 alpha range.
        let inv_sa = 256 - source.alpha() as u16;
        let r = source.red() + ((self.red() as u16 * inv_sa) >> 8) as u8;
        let g = source.green() + ((self.green() as u16 * inv_sa) >> 8) as u8;
        let b = source.blue() + ((self.blue() as u16 * inv_sa) >> 8) as u8;
        let a = source.alpha() + ((self.alpha() as u16 * inv_sa) >> 8) as u8;
        Self::rgba(r, g, b, a)
    }

    fn slice_as_bgra(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
}

impl std::fmt::Display for Color {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("{:#x}", self.to_bgra_u32()))
    }
}

impl From<Color> for u32 {
    fn from(c: Color) -> Self {
        c.to_bgra_u32()
    }
}

impl From<u32> for Color {
    fn from(i: u32) -> Self {
        Color::bgra_u32(i)
    }
}

impl From<swf::Color> for Color {
    fn from(c: swf::Color) -> Self {
        Self::rgba(c.r, c.g, c.b, c.a)
    }
}

impl From<Color> for swf::Color {
    fn from(c: Color) -> Self {
        let r = c.red();
        let g = c.green();
        let b = c.blue();
        let a = c.alpha();
        Self { r, g, b, a }
    }
}

bitflags! {
    pub struct ChannelOptions: u8 {
        const RED = 1 << 0;
        const GREEN = 1 << 1;
        const BLUE = 1 << 2;
        const ALPHA = 1 << 3;
        const RGB = Self::RED.bits() | Self::GREEN.bits() | Self::BLUE.bits();
    }
}

#[derive(Copy, Clone, Collect, Debug)]
#[collect(no_drop)]
pub struct BitmapData<'gc>(BitmapRawDataWrapper<'gc>);

impl<'gc> BitmapData<'gc> {
    pub fn new(
        mc: &Mutation<'gc>,
        width: u32,
        height: u32,
        transparency: bool,
        fill_color: u32,
    ) -> Self {
        let data = BitmapRawData::new(width, height, transparency, fill_color);
        let data = BitmapRawDataWrapper::new(Gc::new(mc, data.into()));

        Self(data)
    }

    pub fn new_with_pixels(
        mc: &Mutation<'gc>,
        width: u32,
        height: u32,
        transparency: bool,
        pixels: Vec<Color>,
    ) -> Self {
        Self::from_loaded_image(mc, width, height, transparency, pixels, None)
    }

    pub(crate) fn from_loaded_image(
        mc: &Mutation<'gc>,
        width: u32,
        height: u32,
        transparency: bool,
        pixels: Vec<Color>,
        loader_request: Option<LoaderRequest>,
    ) -> Self {
        let mut data = BitmapRawData::new_with_pixels(width, height, transparency, pixels);
        data.loader_request = loader_request;
        Self(BitmapRawDataWrapper::new(Gc::new(mc, data.into())))
    }

    pub(crate) fn from_symbol(
        mc: &Mutation<'gc>,
        symbol: std::rc::Rc<crate::bitmap::symbol_pixels::SymbolImage>,
    ) -> Self {
        let size = symbol.data.size();
        let mut raw = BitmapRawData::new_with_pixels(size.width, size.height, true, Vec::new());
        raw.pixels = crate::bitmap::symbol_pixels::Pixels::symbol(symbol);
        Self(BitmapRawDataWrapper::new(Gc::new(mc, raw.into())))
    }

    pub fn dummy(mc: &Mutation<'gc>) -> Self {
        Self(BitmapRawDataWrapper::dummy(mc))
    }

    pub fn clone_data(
        &self,
        mc: &Mutation<'gc>,
        renderer: &mut dyn RenderBackend,
    ) -> Result<BitmapData<'gc>, RenderFailure> {
        let data = self.0.clone_data(renderer)?;
        let data = BitmapRawDataWrapper::new(Gc::new(mc, data.into()));
        Ok(Self(data))
    }

    pub fn sync(
        &self,
        renderer: &mut dyn RenderBackend,
    ) -> Result<GcRefLock<'gc, BitmapRawData<'gc>>, RenderFailure> {
        self.0.sync(renderer)
    }

    pub fn bitmap_handle(
        &self,
        gc_context: &Mutation<'gc>,
        renderer: &mut dyn RenderBackend,
    ) -> Result<BitmapHandle, RenderFailure> {
        self.0.bitmap_handle(gc_context, renderer)
    }

    pub fn mark_gpu_dirty(
        &self,
        gc_context: &Mutation<'gc>,
        sync_handle: Box<dyn SyncHandle>,
        region: PixelRegion,
    ) {
        self.0.mark_gpu_dirty(gc_context, sync_handle, region);
    }

    pub fn overwrite_cpu_pixels_from_gpu(
        &self,
        mc: &Mutation<'gc>,
    ) -> (GcRefLock<'gc, BitmapRawData<'gc>>, Option<PixelRegion>) {
        self.0.overwrite_cpu_pixels_from_gpu(mc)
    }

    pub fn read_area(
        &self,
        read_area: PixelRegion,
        renderer: &mut dyn RenderBackend,
    ) -> Result<Ref<'_, BitmapRawData<'gc>>, RenderFailure> {
        self.0.read_area(read_area, renderer)
    }

    pub(crate) fn read_pixel(
        &self,
        x: u32,
        y: u32,
        renderer: &mut dyn RenderBackend,
    ) -> Result<Ref<'_, BitmapRawData<'gc>>, RenderFailure> {
        self.0.read_pixel(x, y, renderer)
    }

    pub fn write_area(
        &self,
        area: PixelRegion,
        read: bool,
        renderer: &mut dyn RenderBackend,
    ) -> Result<GcRefLock<'gc, BitmapRawData<'gc>>, RenderFailure> {
        self.0.write_area(area, read, renderer)
    }

    pub fn height(&self) -> u32 {
        self.0.height()
    }

    pub fn width(&self) -> u32 {
        self.0.width()
    }

    pub fn object2(&self) -> Option<BitmapDataObject<'gc>> {
        self.0.object2()
    }

    pub fn disposed(&self) -> bool {
        self.0.disposed()
    }

    pub fn transparency(&self) -> bool {
        self.0.transparency()
    }

    pub fn check_valid(
        &self,
        activation: &mut crate::avm2::Activation<'_, 'gc>,
    ) -> Result<(), crate::avm2::Error<'gc>> {
        self.0.check_valid(activation)
    }

    pub fn dispose(&self, mc: &Mutation<'gc>) {
        self.0.dispose(mc);
    }

    pub fn init_object2(&self, mc: &Mutation<'gc>, object: BitmapDataObject<'gc>) {
        self.0.init_object2(mc, object);
    }

    pub fn remove_display_object(&self, mc: &Mutation<'gc>, callback: DisplayObjectWeak<'gc>) {
        self.0.remove_display_object(mc, callback);
    }

    pub fn add_display_object(&self, mc: &Mutation<'gc>, callback: DisplayObjectWeak<'gc>) {
        self.0.add_display_object(mc, callback);
    }

    pub fn render(
        &self,
        smoothing: bool,
        context: &mut RenderContext<'_, 'gc>,
        pixel_snapping: PixelSnapping,
    ) {
        self.0.render(smoothing, context, pixel_snapping);
    }

    pub fn can_read(&self, read_area: PixelRegion) -> bool {
        self.0.can_read(read_area)
    }

    #[cfg(feature = "egui")]
    pub fn debug_sync_status(&self) -> std::borrow::Cow<'static, str> {
        self.0.debug_sync_status()
    }

    pub fn is_point_in_bounds(&self, x: i32, y: i32) -> bool {
        self.0.is_point_in_bounds(x, y)
    }

    pub fn ptr_eq(&self, other: BitmapData<'gc>) -> bool {
        self.0.ptr_eq(other.0)
    }
}

#[derive(Collect)]
#[collect(no_drop)]
pub struct BitmapRawData<'gc> {
    /// The pixels in the bitmap, stored as a array of pre-multiplied ARGB colour values
    #[collect(require_static)]
    pixels: crate::bitmap::symbol_pixels::Pixels,

    width: u32,
    height: u32,
    transparency: bool,

    // Note that it's technically possible to have a BitmapData with zero width and height,
    // (by embedding it in the SWF instead of using the BitmapData constructor),
    // so we need a separate 'disposed' flag.
    disposed: bool,

    /// The bitmap handle for this data.
    ///
    /// This is lazily initialized; a value of `None` indicates that
    /// initialization has not yet happened.
    #[collect(require_static)]
    bitmap_handle: Option<BitmapHandle>,

    #[collect(require_static)]
    loader_request: Option<LoaderRequest>,

    /// The AVM2 side of this `BitmapData`.
    ///
    /// AVM1 cannot retrieve `BitmapData` back from the display object tree, so
    /// this does not need to hold an AVM1 object.
    avm2_object: Option<BitmapDataObject<'gc>>,

    /// A list of display objects that are backed by this BitmapData
    display_objects: Vec<DisplayObjectWeak<'gc>>,

    #[collect(require_static)]
    validity: BitmapValidity,

    /// Holds an egui texture handle, used for rendering this Bitmap in the debug ui.
    /// This is automatically set to `None` when the texture is updated (either from
    /// marking the CPU side dirty, or from performing a GPU -> CPU sync).
    #[cfg(feature = "egui")]
    pub egui_texture: std::cell::RefCell<Option<egui::TextureHandle>>,
}

/// Exact, disjoint authority regions. Unmarked pixels are valid on both sides.
/// Sync tokens belong to the last GPU operation; their readback origin is kept
/// separately from the union of dirty pixels. Native reads use the live texture.
#[derive(Debug)]
pub struct BitmapValidity {
    cpu: Coverage,
    gpu: Coverage,
    pending: Option<(Box<dyn SyncHandle>, PixelRegion)>,
    failure: Option<RenderFailure>,
    /// Locality hint only: authoritative coverage still decides what is readable.
    last_pixel_read_tile: Option<(u32, u32)>,
    coherent_pixel_reads: bool,
}
impl BitmapValidity {
    fn new(width: u32, height: u32) -> Self {
        Self {
            cpu: Coverage::empty(width, height),
            gpu: Coverage::empty(width, height),
            pending: None,
            failure: None,
            last_pixel_read_tile: None,
            coherent_pixel_reads: false,
        }
    }
    fn check(&self) -> Result<(), RenderFailure> {
        self.failure.clone().map_or(Ok(()), Err)
    }
    fn is_clean(&self) -> bool {
        self.cpu.is_empty() && self.gpu.is_empty() && self.failure.is_none()
    }
}

/// A diagnostic view, never an owner of a one-shot GPU synchronization token.
#[derive(Clone, Debug)]
pub enum DirtyState {
    Clean,
    CpuModified(PixelRegion),
    GpuModified(PixelRegion),
    Mixed { cpu: PixelRegion, gpu: PixelRegion },
    Failed(RenderFailure),
}

#[cfg(test)]
mod failure_tests {
    use super::*;
    use ruffle_render::backend::{ViewportDimensions, null::NullRenderer};

    #[derive(Debug)]
    struct FailedSync;
    impl SyncHandle for FailedSync {}

    #[test]
    fn failed_bitmap_readback_never_exposes_stale_pixels() {
        let arena = gc_arena::Arena::<gc_arena::Rootable![BitmapData<'_>]>::new(|mc| {
            BitmapData::new(mc, 1, 1, true, 0xff123456)
        });
        arena.mutate(|mc, bitmap| {
            let mut renderer = NullRenderer::new(ViewportDimensions {
                width: 1,
                height: 1,
                scale_factor: 1.0,
            });
            bitmap.mark_gpu_dirty(mc, Box::new(FailedSync), PixelRegion::for_whole_size(1, 1));
            assert!(bitmap.sync(&mut renderer).is_err());
            assert!(bitmap.sync(&mut renderer).is_err());
            assert!(
                bitmap
                    .read_area(PixelRegion::for_pixel(0, 0), &mut renderer)
                    .is_err()
            );
            assert!(!bitmap.can_read(PixelRegion::for_pixel(0, 0)));
            assert!(renderer.health().check().is_err());
        });
    }
}

mod wrapper {
    use crate::avm2::object::BitmapDataObject;
    use crate::context::RenderContext;
    use crate::display_object::DisplayObjectWeak;
    use gc_arena::barrier::Write;
    use gc_arena::lock::GcRefLock;
    use gc_arena::{Collect, Gc, Mutation};
    use ruffle_render::backend::RenderBackend;
    use ruffle_render::bitmap::{BitmapHandle, PixelRegion, PixelSnapping, SyncHandle};
    use ruffle_render::commands::CommandHandler;
    use ruffle_render::error::RenderFailure;
    use std::cell::Ref;
    use std::time::Instant;

    #[cfg(feature = "egui")]
    use super::DirtyState;
    use super::{BitmapRawData, BitmapValidity, copy_pixels_to_bitmapdata};
    use ruffle_render::bitmap::coverage::Coverage;
    use ruffle_render::error::RenderFailureKind;

    /// Guards CPU pixel access with exact GPU validity. Metadata and GPU handle
    /// access do not wait; nearby scalar reads can prefetch bounded dirty coverage.
    #[derive(Copy, Clone, Collect, Debug)]
    #[collect(no_drop)]
    pub struct BitmapRawDataWrapper<'gc>(GcRefLock<'gc, BitmapRawData<'gc>>);

    impl<'gc> BitmapRawDataWrapper<'gc> {
        pub fn new(data: GcRefLock<'gc, BitmapRawData<'gc>>) -> Self {
            BitmapRawDataWrapper(data)
        }

        // Creates a dummy BitmapData with no pixels or handle, marked as disposed.
        // This is used for AS3 `Bitmap` instances without a corresponding AS3 `BitmapData` instance.
        // Marking it as disposed skips rendering, and the unset `avm2_object` will cause this to
        // be inaccessible to AS3 code.
        pub fn dummy(mc: &Mutation<'gc>) -> Self {
            BitmapRawDataWrapper(Gc::new(
                mc,
                BitmapRawData {
                    pixels: crate::bitmap::symbol_pixels::Pixels::owned(Vec::new()),
                    width: 0,
                    height: 0,
                    transparency: false,
                    disposed: true,
                    loader_request: None,
                    bitmap_handle: None,
                    avm2_object: None,
                    display_objects: vec![],
                    validity: BitmapValidity::new(0, 0),
                    #[cfg(feature = "egui")]
                    egui_texture: Default::default(),
                }
                .into(),
            ))
        }

        /// Clones the underlying data, producing a new `BitmapData`
        /// that has no GPU texture or associated display objects
        pub fn clone_data(
            &self,
            renderer: &mut dyn RenderBackend,
        ) -> Result<BitmapRawData<'gc>, RenderFailure> {
            // Sync from the GPU to CPU, since our new BitmapData starts out
            // with no GPU texture
            let data = self.sync(renderer)?.borrow();
            Ok(BitmapRawData {
                pixels: data.pixels.clone(),
                width: data.width,
                height: data.height,
                transparency: data.transparency,
                disposed: data.disposed,
                loader_request: None,
                bitmap_handle: None,
                avm2_object: None,
                display_objects: vec![],
                // We have no GPU texture, so there's no need to mark as dirty
                validity: BitmapValidity::new(data.width, data.height),
                #[cfg(feature = "egui")]
                egui_texture: Default::default(),
            })
        }

        // Provides access to the underlying `BitmapData`. If a GPU -> CPU sync
        // is in progress, waits for it to complete
        pub fn sync(
            &self,
            renderer: &mut dyn RenderBackend,
        ) -> Result<GcRefLock<'gc, BitmapRawData<'gc>>, RenderFailure> {
            self.sync_area(
                PixelRegion::for_whole_size(self.width(), self.height()),
                false,
                renderer,
            )?;
            Ok(self.0)
        }

        fn sync_area(
            &self,
            mut area: PixelRegion,
            prefetch: bool,
            renderer: &mut dyn RenderBackend,
        ) -> Result<(), RenderFailure> {
            renderer.health().check()?;
            let mut pixel_read_tile = None;
            let mut coherent_pixel_reads = false;
            {
                let read = self.0.borrow();
                read.validity.check()?;
                // Aliased input operands may already hold a shared pixel view.
                // A clean region needs no exclusive access to those pixels.
                if !read.validity.gpu.intersects(area) {
                    return Ok(());
                }
                if prefetch
                    && area.width() == 1
                    && area.height() == 1
                    && renderer.supports_bitmap_region_readback()
                {
                    // Keep isolated reads exact. Repeated nearby reads amortize the
                    // queue/map wait over at most 4 KiB, not the whole bitmap.
                    let tile = (area.x_min / 32 * 32, area.y_min / 32 * 32);
                    pixel_read_tile = Some(tile);
                    coherent_pixel_reads = read.validity.coherent_pixel_reads
                        || read.validity.last_pixel_read_tile == Some(tile);
                    if coherent_pixel_reads {
                        area = PixelRegion::for_region(
                            tile.0,
                            tile.1,
                            read.width.saturating_sub(tile.0).min(32),
                            read.height.saturating_sub(tile.1).min(32),
                        );
                    }
                }
            }
            // SAFETY: Only owned pixel/validity data changes; no GC references are written.
            let mut write = unsafe { Write::assume(Gc::as_ref(self.0)) }
                .unlock()
                .borrow_mut();
            let empty = Coverage::empty(write.width, write.height);
            let mut gpu = std::mem::replace(&mut write.validity.gpu, empty);
            let mut fallback_done = false;
            let result =
                gpu.for_each_readback_in(area, |region, parts| -> Result<(), RenderFailure> {
                    if fallback_done {
                        return Ok(());
                    }
                    let started = Instant::now();
                    let result = if let Some(handle) = write.bitmap_handle.clone() {
                        renderer
                            .read_bitmap_regions_native(
                                &handle,
                                region,
                                parts,
                                &mut |part, buffer, stride, format| {
                                    copy_pixels_to_bitmapdata(
                                        &mut write, buffer, stride, format, part, part,
                                    )
                                },
                            )
                            .map_err(|error| error.into_render_failure())?
                    } else {
                        None
                    };
                    let bytes = if result.is_some() {
                        parts
                            .iter()
                            .map(|r| u64::from(r.width()) * u64::from(r.height()) * 4)
                            .sum()
                    } else {
                        let mut copied = Err(RenderFailure::new(
                            RenderFailureKind::MapFailed,
                            "bitmap readback returned no pixels",
                        ));
                        let (pending, origin) = write.validity.pending.take().ok_or_else(|| {
                            RenderFailure::new(
                                RenderFailureKind::MapFailed,
                                "bitmap has no pending readback",
                            )
                        })?;
                        renderer
                            .resolve_sync_handle_native(
                                pending,
                                Box::new(|buffer, stride, format| {
                                    copied = gpu.for_each(|part| {
                                        copy_pixels_to_bitmapdata(
                                            &mut write, buffer, stride, format, origin, part,
                                        )
                                    });
                                }),
                            )
                            .map_err(|error| error.into_render_failure())?;
                        copied?;
                        fallback_done = true;
                        u64::from(origin.width()) * u64::from(origin.height()) * 4
                    };
                    renderer.health().check()?;
                    ruffle_render::profile::bitmap_readback(
                        bytes,
                        started.elapsed().as_nanos() as u64,
                    );
                    Ok(())
                });
            if let Err(error) = result {
                write.validity.gpu = gpu;
                let error = renderer.health().record(error);
                write.validity.failure = Some(error.clone());
                return Err(error);
            }
            if !fallback_done {
                gpu.remove(area);
            }
            if fallback_done || gpu.is_empty() {
                gpu = Coverage::empty(write.width, write.height);
                write.validity.pending = None;
            }
            write.validity.gpu = gpu;
            write.validity.last_pixel_read_tile = pixel_read_tile;
            write.validity.coherent_pixel_reads = coherent_pixel_reads;
            #[cfg(feature = "egui")]
            write.egui_texture.borrow_mut().take();
            Ok(())
        }

        /// Access for a CPU write that preserves other regions. `read` is true
        /// only when this operation needs the previous pixels (e.g. setPixel alpha).
        pub fn write_area(
            &self,
            area: PixelRegion,
            read: bool,
            renderer: &mut dyn RenderBackend,
        ) -> Result<GcRefLock<'gc, BitmapRawData<'gc>>, RenderFailure> {
            renderer.health().check()?;
            self.0.borrow().validity.check()?;
            if read {
                self.sync_area(area, false, renderer)?;
            }
            Ok(self.0)
        }

        /// Provides access to the underlying `BitmapHandle`.
        /// If the CPU pixels are dirty, syncs them to the GPU.
        /// If the GPU pixels are dirty, then handle is returned immediately
        /// without waiting for the sync to complete, as a BitmapHandle can
        /// only be used to access the GPU data. Unlike `overwrite_cpu_pixels_from_gpu`,
        /// this does not cancel the GPU -> CPU sync.
        pub fn bitmap_handle(
            &self,
            gc_context: &Mutation<'gc>,
            renderer: &mut dyn RenderBackend,
        ) -> Result<BitmapHandle, RenderFailure> {
            let mut bitmap_data = self.0.borrow_mut(gc_context);
            bitmap_data.update_dirty_texture(renderer)?;
            bitmap_data.bitmap_handle(renderer)
        }

        pub fn mark_gpu_dirty(
            &self,
            gc_context: &Mutation<'gc>,
            sync_handle: Box<dyn SyncHandle>,
            region: PixelRegion,
        ) {
            self.0
                .borrow_mut(gc_context)
                .set_gpu_dirty(gc_context, sync_handle, region);
        }

        /// Provides access to the underlying `BitmapData`.
        /// This should only be used when you will be overwriting the entire
        /// `pixels` vec without reading from it. Cancels any in-progress GPU -> CPU sync.
        /// This does not sync from cpu to gpu.
        pub fn overwrite_cpu_pixels_from_gpu(
            &self,
            mc: &Mutation<'gc>,
        ) -> (GcRefLock<'gc, BitmapRawData<'gc>>, Option<PixelRegion>) {
            let mut write = self.0.borrow_mut(mc);
            let dirty_rect = (!write.validity.gpu.is_empty()).then(|| write.validity.gpu.bounds());
            write.validity.gpu = Coverage::empty(write.width, write.height);
            write.validity.pending = None;
            #[cfg(feature = "egui")]
            write.egui_texture.borrow_mut().take();
            (self.0, dirty_rect)
        }

        /// Provides read access to the BitmapData pixels.
        /// Only the provided region is guaranteed to be up-to-date.
        /// It is an error to access any other pixels outside of that region.
        pub fn read_area(
            &self,
            read_area: PixelRegion,
            renderer: &mut dyn RenderBackend,
        ) -> Result<Ref<'_, BitmapRawData<'gc>>, RenderFailure> {
            self.sync_area(read_area, false, renderer)?;
            Ok(self.0.borrow())
        }

        pub fn read_pixel(
            &self,
            x: u32,
            y: u32,
            renderer: &mut dyn RenderBackend,
        ) -> Result<Ref<'_, BitmapRawData<'gc>>, RenderFailure> {
            self.sync_area(PixelRegion::for_pixel(x, y), true, renderer)?;
            Ok(self.0.borrow())
        }

        // These methods do not require a sync to complete, as they do not depend on the
        // CPU-side pixels. They are implemented directly on `BitmapRawDataWrapper`, allowing
        // callers to avoid calling sync()

        pub fn height(&self) -> u32 {
            self.0.borrow().height
        }

        pub fn width(&self) -> u32 {
            self.0.borrow().width
        }

        pub fn object2(&self) -> Option<BitmapDataObject<'gc>> {
            self.0.borrow().object2()
        }

        pub fn disposed(&self) -> bool {
            self.0.borrow().disposed
        }

        pub fn transparency(&self) -> bool {
            self.0.borrow().transparency
        }

        pub fn check_valid(
            &self,
            activation: &mut crate::avm2::Activation<'_, 'gc>,
        ) -> Result<(), crate::avm2::Error<'gc>> {
            if self.disposed() {
                Err(crate::avm2::error::make_error_2015(activation))
            } else {
                Ok(())
            }
        }

        pub fn dispose(&self, mc: &Mutation<'gc>) {
            let mut data = self.0.borrow_mut(mc);
            data.dispose();
            data.inform_display_objects(mc);
        }

        pub fn init_object2(&self, mc: &Mutation<'gc>, object: BitmapDataObject<'gc>) {
            self.0.borrow_mut(mc).avm2_object = Some(object);
        }

        pub fn remove_display_object(&self, mc: &Mutation<'gc>, callback: DisplayObjectWeak<'gc>) {
            // [NA] Removing is a rare operation, whereas insert is often, and iteration is extremely frequent.
            // The list will typically be 0-1 entries long too, so I think retain is fine for quick iteration.
            self.0
                .borrow_mut(mc)
                .display_objects
                .retain(|c| !std::ptr::eq(c.as_ptr(), callback.as_ptr()))
        }

        pub fn add_display_object(&self, mc: &Mutation<'gc>, callback: DisplayObjectWeak<'gc>) {
            self.0.borrow_mut(mc).display_objects.push(callback);
        }

        pub fn render(
            &self,
            smoothing: bool,
            context: &mut RenderContext<'_, 'gc>,
            pixel_snapping: PixelSnapping,
        ) {
            let mut inner_bitmap_data = self.0.borrow_mut(context.gc());
            if inner_bitmap_data.disposed() {
                return;
            }

            // Note - we do a CPU -> GPU sync, but we do *not* do a GPU -> CPU sync
            // (rendering is done on the GPU, so the CPU pixels don't need to be up-to-date).
            let view = if inner_bitmap_data.bitmap_handle.is_none()
                && inner_bitmap_data.validity.is_clean()
            {
                match inner_bitmap_data.pixels.view(context.renderer) {
                    Ok(view) => view,
                    Err(error) => {
                        context
                            .renderer
                            .health()
                            .record(error.into_render_failure());
                        return;
                    }
                }
            } else {
                None
            };
            let handle = match view {
                Some(handle) => handle,
                None => {
                    if inner_bitmap_data
                        .update_dirty_texture(context.renderer)
                        .is_err()
                    {
                        return;
                    }
                    match inner_bitmap_data.bitmap_handle(context.renderer) {
                        Ok(handle) => handle,
                        Err(_) => return,
                    }
                }
            };

            context.commands.render_bitmap(
                handle,
                context.transform_stack.transform(),
                smoothing,
                pixel_snapping,
                inner_bitmap_data.full_region(),
            );
        }

        pub fn can_read(&self, read_area: PixelRegion) -> bool {
            let data = self.0.borrow();
            data.validity.failure.is_none() && !data.validity.gpu.intersects(read_area)
        }

        #[cfg(feature = "egui")]
        pub fn debug_sync_status(&self) -> std::borrow::Cow<'static, str> {
            match self.0.borrow().dirty_state() {
                DirtyState::Failed(ref error) => std::borrow::Cow::Owned(error.to_string()),
                DirtyState::Clean => std::borrow::Cow::Borrowed("Clean"),
                DirtyState::Mixed { cpu, gpu } => {
                    std::borrow::Cow::Owned(format!("CPU modified {cpu:?}; GPU modified {gpu:?}"))
                }
                DirtyState::CpuModified(area) => std::borrow::Cow::Owned(format!(
                    "CPU modified from {}, {} to {}, {}",
                    area.x_min, area.y_min, area.x_max, area.y_max
                )),
                DirtyState::GpuModified(area) => std::borrow::Cow::Owned(format!(
                    "GPU modified from {}, {} to {}, {}",
                    area.x_min, area.y_min, area.x_max, area.y_max
                )),
            }
        }

        pub fn is_point_in_bounds(&self, x: i32, y: i32) -> bool {
            x >= 0 && x < self.width() as i32 && y >= 0 && y < self.height() as i32
        }

        pub fn ptr_eq(&self, other: BitmapRawDataWrapper<'gc>) -> bool {
            Gc::ptr_eq(self.0, other.0)
        }
    }
}

pub use wrapper::BitmapRawDataWrapper;

impl std::fmt::Debug for BitmapRawData<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BitmapRawData")
            .field("validity", &self.validity)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("transparency", &self.transparency)
            .field("disposed", &self.disposed)
            .field("bitmap_handle", &self.bitmap_handle)
            .finish()
    }
}

impl<'gc> BitmapRawData<'gc> {
    pub fn new(width: u32, height: u32, transparency: bool, fill_color: u32) -> Self {
        Self {
            pixels: crate::bitmap::symbol_pixels::Pixels::owned(vec![
                Color::bgra_u32(fill_color)
                    .to_premultiplied_alpha(
                        transparency
                    );
                width as usize
                    * height as usize
            ]),
            width,
            height,
            transparency,
            disposed: false,
            loader_request: None,
            bitmap_handle: None,
            avm2_object: None,
            display_objects: vec![],
            validity: BitmapValidity::new(width, height),
            #[cfg(feature = "egui")]
            egui_texture: Default::default(),
        }
    }

    pub fn new_with_pixels(
        width: u32,
        height: u32,
        transparency: bool,
        pixels: Vec<Color>,
    ) -> Self {
        Self {
            pixels: crate::bitmap::symbol_pixels::Pixels::owned(pixels),
            width,
            height,
            transparency,
            loader_request: None,
            bitmap_handle: None,
            avm2_object: None,
            disposed: false,
            validity: BitmapValidity::new(width, height),
            display_objects: vec![],
            #[cfg(feature = "egui")]
            egui_texture: Default::default(),
        }
    }

    pub fn disposed(&self) -> bool {
        self.disposed
    }

    pub fn dispose(&mut self) {
        self.width = 0;
        self.height = 0;
        self.pixels = crate::bitmap::symbol_pixels::Pixels::owned(Vec::new()); // free the CPU pixel buffer
        self.bitmap_handle = None;
        // There's no longer a handle to update
        self.validity = BitmapValidity::new(self.width, self.height);
        self.disposed = true;
    }

    pub fn try_bitmap_handle(
        &mut self,
        renderer: &mut dyn RenderBackend,
    ) -> Result<BitmapHandle, ruffle_render::error::Error> {
        renderer.health().check()?;
        if let Some(ref handle) = self.bitmap_handle {
            return Ok(handle.clone());
        }

        let bitmap = Bitmap::new(
            self.width(),
            self.height(),
            BitmapFormat::Bgra,
            Color::slice_as_bgra(&self.pixels),
        )
        .with_loader_request(self.loader_request);
        let usage = if self.transparency() {
            ruffle_render::backend::BitmapUsage::Mutable
        } else {
            ruffle_render::backend::BitmapUsage::OpaqueMutable
        };
        let bitmap_handle = renderer.register_bitmap_with_usage(bitmap, usage);
        if let Ok(ref handle) = bitmap_handle {
            self.validity = BitmapValidity::new(self.width, self.height);
            self.bitmap_handle = Some(handle.clone());
        }
        bitmap_handle
    }

    pub fn bitmap_handle(
        &mut self,
        renderer: &mut dyn RenderBackend,
    ) -> Result<BitmapHandle, RenderFailure> {
        self.try_bitmap_handle(renderer)
            .map_err(|error| renderer.health().record(error.into_render_failure()))
    }

    pub fn transparency(&self) -> bool {
        self.transparency
    }

    pub fn dirty_state(&self) -> DirtyState {
        if let Some(error) = &self.validity.failure {
            return DirtyState::Failed(error.clone());
        }
        match (self.validity.cpu.is_empty(), self.validity.gpu.is_empty()) {
            (true, true) => DirtyState::Clean,
            (false, true) => DirtyState::CpuModified(self.validity.cpu.bounds()),
            (true, false) => DirtyState::GpuModified(self.validity.gpu.bounds()),
            (false, false) => DirtyState::Mixed {
                cpu: self.validity.cpu.bounds(),
                gpu: self.validity.gpu.bounds(),
            },
        }
    }

    pub fn set_gpu_dirty(
        &mut self,
        gc_context: &Mutation<'gc>,
        sync_handle: Box<dyn SyncHandle>,
        region: PixelRegion,
    ) {
        if self.validity.failure.is_some() {
            return;
        }
        self.validity.cpu.remove(region);
        self.validity.gpu.add(region);
        let live_read = self
            .bitmap_handle
            .as_ref()
            .is_some_and(|handle| sync_handle.can_read_current_without_token(handle));
        self.validity.pending = (!live_read).then_some((sync_handle, region));
        self.validity.last_pixel_read_tile = None;
        self.validity.coherent_pixel_reads = false;
        self.inform_display_objects(gc_context);
    }

    pub fn set_cpu_dirty(&mut self, gc_context: &Mutation<'gc>, region: PixelRegion) {
        debug_assert!(region.x_max <= self.width && region.y_max <= self.height);
        if self.validity.failure.is_some() || region.is_empty() {
            return;
        }
        #[cfg(feature = "egui")]
        self.egui_texture.borrow_mut().take();
        self.validity.gpu.remove(region);
        if self.validity.gpu.is_empty() {
            self.validity.pending = None;
        }
        self.validity.cpu.add(region);
        self.inform_display_objects(gc_context);
    }

    pub fn pixels(&self) -> &[Color] {
        &self.pixels
    }

    pub fn pixels_bgra(&self) -> &[u8] {
        Color::slice_as_bgra(&self.pixels)
    }

    pub fn pixels_rgba(&self) -> Vec<u8> {
        self.pixels
            .iter()
            .flat_map(|c| [c.r, c.g, c.b, c.a])
            .collect()
    }

    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn full_region(&self) -> PixelRegion {
        PixelRegion::for_whole_size(self.width(), self.height())
    }

    pub fn is_point_in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && x < self.width() as i32 && y >= 0 && y < self.height() as i32
    }

    #[inline]
    pub fn set_pixel32_raw(&mut self, x: u32, y: u32, color: Color) {
        self.pixels[(x + y * self.width) as usize] = color;
    }

    #[inline]
    pub fn set_pixel32_row_raw(&mut self, x1: u32, x2: u32, y: u32, color: Color) {
        let p1 = (x1 + y * self.width) as usize;
        let p2 = (x2 + y * self.width) as usize;
        let slice = &mut self.pixels[p1..p2];
        slice.fill(color);
    }

    #[inline]
    pub fn fill(&mut self, color: Color) {
        self.pixels.fill(color);
    }

    #[inline]
    pub fn get_pixel32_raw(&self, x: u32, y: u32) -> Color {
        self.pixels[(x + y * self.width()) as usize]
    }

    pub fn raw_pixels_mut(&mut self) -> &mut [Color] {
        &mut self.pixels
    }

    pub fn raw_pixels(&self) -> &[Color] {
        &self.pixels
    }

    // Updates the data stored with our `BitmapHandle` if this `BitmapRawData`
    // is dirty
    pub fn update_dirty_texture(
        &mut self,
        renderer: &mut dyn RenderBackend,
    ) -> Result<(), RenderFailure> {
        renderer.health().check()?;
        let handle = self.bitmap_handle(renderer)?;
        self.validity.check()?;
        self.validity.cpu.for_each(|region| {
            renderer
                .update_texture(
                    &handle,
                    Bitmap::new(
                        self.width(),
                        self.height(),
                        BitmapFormat::Bgra,
                        Color::slice_as_bgra(&self.pixels),
                    ),
                    region,
                )
                .map_err(|error| renderer.health().record(error.into_render_failure()))
        })?;
        self.validity.cpu = Coverage::empty(self.width, self.height);
        Ok(())
    }

    fn inform_display_objects(&self, gc_context: &Mutation<'gc>) {
        for object in &self.display_objects {
            if let Some(object) = object.upgrade(gc_context) {
                object.invalidate_cached_bitmap();
            }
        }
    }

    pub fn object2(&self) -> Option<BitmapDataObject<'gc>> {
        self.avm2_object
    }

    pub fn init_object2(&mut self, object: BitmapDataObject<'gc>) {
        self.avm2_object = Some(object);
    }
}

pub enum IBitmapDrawable<'gc> {
    BitmapData(BitmapData<'gc>),
    DisplayObject(DisplayObject<'gc>),
}

impl IBitmapDrawable<'_> {
    pub fn bounds(&self) -> Rectangle<Twips> {
        match self {
            IBitmapDrawable::BitmapData(bmd) => Rectangle {
                x_min: Twips::ZERO,
                x_max: Twips::from_pixels(bmd.width() as f64),
                y_min: Twips::ZERO,
                y_max: Twips::from_pixels(bmd.height() as f64),
            },
            IBitmapDrawable::DisplayObject(o) => o.bounds(BoundsMode::Engine),
        }
    }
}

#[instrument(level = "debug", skip_all)]
fn copy_pixels_to_bitmapdata(
    write: &mut BitmapRawData,
    buffer: &[u8],
    stride: u32,
    format: BitmapFormat,
    origin: PixelRegion,
    area: PixelRegion,
) -> Result<(), RenderFailure> {
    let row_bytes = u64::from(origin.width()) * 4;
    let required = u64::from(stride) * u64::from(origin.height().saturating_sub(1)) + row_bytes;
    if stride % 4 != 0
        || u64::from(stride) < row_bytes
        || required > buffer.len() as u64
        || area.x_min < origin.x_min
        || area.y_min < origin.y_min
        || area.x_max > origin.x_max
        || area.y_max > origin.y_max
        || area.x_max > write.width
        || area.y_max > write.height
    {
        return Err(RenderFailure::new(
            RenderFailureKind::MapFailed,
            "bitmap readback does not cover authoritative pixels",
        ));
    }
    let width = write.width as usize;
    let transparent = write.transparency();
    for y in area.y_min..area.y_max {
        let src = (y - origin.y_min) as usize * stride as usize
            + (area.x_min - origin.x_min) as usize * 4;
        let count = area.width() as usize;
        let dst = y as usize * width + area.x_min as usize;
        let pixels = &mut write.pixels[dst..dst + count];
        let source = &buffer[src..src + count * 4];
        if format == BitmapFormat::Bgra && transparent {
            bytemuck::cast_slice_mut(pixels).copy_from_slice(source);
        } else {
            for (pixel, bytes) in pixels.iter_mut().zip(source.as_chunks::<4>().0) {
                let [r, g, b, a] = *bytes;
                let a = if transparent { a } else { 255 };
                *pixel = if format == BitmapFormat::Bgra {
                    Color::rgba(b, g, r, a)
                } else {
                    Color::rgba(r, g, b, a)
                };
            }
        }
    }
    Ok(())
}

#[derive(Copy, Clone, Debug)]
pub enum ThresholdOperation {
    Equals,
    NotEquals,
    LessThan,
    LessThanOrEquals,
    GreaterThan,
    GreaterThanOrEquals,
}

impl ThresholdOperation {
    pub fn from_wstr(str: &WStr) -> Option<Self> {
        if str == b"==" {
            Some(Self::Equals)
        } else if str == b"!=" {
            Some(Self::NotEquals)
        } else if str == b"<" {
            Some(Self::LessThan)
        } else if str == b"<=" {
            Some(Self::LessThanOrEquals)
        } else if str == b">" {
            Some(Self::GreaterThan)
        } else if str == b">=" {
            Some(Self::GreaterThanOrEquals)
        } else {
            None
        }
    }

    pub fn matches(&self, value: u32, masked_threshold: u32) -> bool {
        match self {
            ThresholdOperation::Equals => value == masked_threshold,
            ThresholdOperation::NotEquals => value != masked_threshold,
            ThresholdOperation::LessThan => value < masked_threshold,
            ThresholdOperation::LessThanOrEquals => value <= masked_threshold,
            ThresholdOperation::GreaterThan => value > masked_threshold,
            ThresholdOperation::GreaterThanOrEquals => value >= masked_threshold,
        }
    }
}
