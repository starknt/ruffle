use std::borrow::Cow;
use std::num::NonZeroU32;
use std::sync::Arc;

#[test]
fn nearest_bitmap_blocks_require_pixel_aligned_boundaries_even_after_caching() {
    use ruffle_render::{bitmap::PixelRegion, matrix::Matrix};
    let mut gpu = renderer(false);
    let mut pixels = super::Pixels::solid(
        1024 * 1024,
        super::Color::from(0xff337755),
        ruffle_memory::MemoryScope::current(),
    );
    pixels.fill_rect(
        1024,
        1024,
        PixelRegion::for_region(0, 0, 256, 256),
        super::Color::from(0xff6584ab),
    );
    let aligned = Matrix::scale(0.25, 0.25);
    let (image, blocks, region) = pixels.nearest_view(&mut gpu, &aligned).unwrap().unwrap();
    assert_eq!(blocks, [8, 8]);
    assert_eq!(region, PixelRegion::for_whole_size(128, 128));
    let texels = test_handle(&image).pixels.borrow();
    assert_eq!(
        &texels[4 * 31..4 * 33],
        &[101, 132, 171, 255, 51, 119, 85, 255]
    );
    for matrix in [
        Matrix::scale(0.35, 0.35),
        Matrix {
            tx: swf::Twips::HALF_PX,
            ..aligned
        },
        Matrix { b: 0.1, ..aligned },
    ] {
        assert!(pixels.nearest_view(&mut gpu, &matrix).unwrap().is_none());
    }
}

#[test]
fn aligned_rect_fill_can_be_read_and_cloned_before_expanding_pixels() {
    use crate::bitmap::{bitmap_data::BitmapData, operations};
    let manager = ruffle_memory::MemoryManager::new(32 << 20);
    manager.player().enter(|| {
        gc_arena::arena::rootless_mutate(|mc| {
            let mut gpu = renderer(false);
            let source = BitmapData::new(mc, 512, 256, true, 0xff123456);
            let patched = source.clone_data(mc, &mut gpu).unwrap();
            operations::fill_rect(mc, &mut gpu, patched, 128, 64, 256, 128, 0xffabcdef).unwrap();
            let other = patched.clone_data(mc, &mut gpu).unwrap();
            let sample = |data, gpu: &mut TestRenderer, x, y| {
                operations::get_pixel32(data, gpu, x, y).unwrap()
            };
            assert_eq!(sample(patched, &mut gpu, 383, 191), 0xffabcdef);
            assert_eq!(sample(other, &mut gpu, 384, 191), 0xff123456);
            assert_eq!(sample(source, &mut gpu, 128, 64), 0xff123456);
            assert_eq!(
                manager.snapshot().active_bytes,
                0,
                "aligned fill expanded all CPU pixels"
            );
            operations::set_pixel32(mc, &mut gpu, patched, 128, 64, 0xff012345).unwrap();
            assert_eq!(sample(other, &mut gpu, 128, 64), 0xffabcdef);
            let handle = patched.bitmap_handle(mc, &mut gpu).unwrap();
            assert_eq!(test_handle(&handle).pixels.borrow().len(), 512 * 256 * 4);
            for (x, y, color) in [
                (128, 64, [1, 35, 69, 255]),
                (383, 191, [171, 205, 239, 255]),
                (384, 191, [18, 52, 86, 255]),
            ] {
                let i = 4 * (y * 512 + x);
                assert_eq!(&test_handle(&handle).pixels.borrow()[i..i + 4], &color);
            }
        })
    });
}

#[test]
fn owned_bitmap_clone_shares_storage_until_first_pixel_write() {
    let mut first = super::Pixels::owned(vec![super::Color::from(0xff123456); 16]);
    let other = first.clone();
    assert_eq!(first.as_ptr(), other.as_ptr());
    first[0] = super::Color::from(0xffabcdef);
    assert_ne!(first.as_ptr(), other.as_ptr());
    assert_eq!(u32::from(other[0]), 0xff123456);
}

#[test]
fn uniform_bitmap_only_allocates_pixels_for_the_modified_clone() {
    use crate::bitmap::{bitmap_data::BitmapData, operations};
    use ruffle_memory::{MemoryKind, MemoryManager};

    let manager = MemoryManager::new(32 << 20);
    let scope = manager.player();
    let pixel_bytes = || {
        manager.snapshot().players[0]
            .active
            .get(&MemoryKind::BitmapPixels)
            .copied()
            .unwrap_or(0)
    };
    gc_arena::arena::rootless_mutate(|mc| {
        let mut renderer = renderer(false);
        let source = scope.enter(|| BitmapData::new(mc, 128, 128, false, 0x00123456));
        let copy = source.clone_data(mc, &mut renderer).unwrap();
        let read = |image, renderer: &mut TestRenderer| {
            operations::get_pixel32(image, renderer, 1, 1).unwrap()
        };
        assert_eq!(read(source, &mut renderer), 0xff123456);
        assert_eq!(pixel_bytes(), 0, "single-pixel reads of a solid fill");
        operations::fill_rect(mc, &mut renderer, copy, 0, 0, 4, 4, 0xff337755).unwrap();
        assert_eq!(
            pixel_bytes(),
            128 * 128 * 4,
            "charge the allocating bitmap's scope"
        );
        assert_eq!(read(source, &mut renderer), 0xff123456);
        assert_eq!(read(copy, &mut renderer), 0xff337755);
        assert_eq!(
            operations::get_pixel32(copy, &mut renderer, 4, 4).unwrap(),
            0xff123456
        );
        operations::fill_rect(mc, &mut renderer, source, 0, 0, 128, 128, 0xff224466).unwrap();
        assert_eq!(read(source, &mut renderer), 0xff224466);
        copy.dispose(mc);
        assert_eq!(pixel_bytes(), 0);
    });
}

#[test]
fn uploaded_uniform_bitmap_reuses_its_pixels_on_full_refills() {
    use crate::bitmap::{bitmap_data::BitmapData, operations};
    use ruffle_memory::MemoryManager;

    let manager = MemoryManager::new(32 << 20);
    let scope = manager.player();
    scope.enter(|| {
        gc_arena::arena::rootless_mutate(|mc| {
            let mut renderer = renderer(false);
            let bitmap = BitmapData::new(mc, 128, 128, false, 0xff123456);
            let clone = bitmap.clone_data(mc, &mut renderer).unwrap();
            let handle = bitmap.bitmap_handle(mc, &mut renderer).unwrap();
            let data = bitmap.sync(&mut renderer).unwrap();
            let ptr = data.borrow().pixels().as_ptr();
            for color in [0xff446688, 0xff224466] {
                operations::fill_rect(mc, &mut renderer, bitmap, 0, 0, 128, 128, color).unwrap();
                assert_eq!(manager.snapshot().active_bytes, 128 * 128 * 4);
                bitmap.bitmap_handle(mc, &mut renderer).unwrap();
                assert_eq!(data.borrow().pixels().as_ptr(), ptr);
                let [a, r, g, b] = color.to_be_bytes();
                assert_eq!(&test_handle(&handle).pixels.borrow()[..4], &[r, g, b, a]);
            }
            let cold = clone.sync(&mut renderer).unwrap();
            assert!(
                cold.borrow()
                    .pixels()
                    .iter()
                    .all(|p| u32::from(*p) == 0xff123456)
            );
        })
    });
}

#[test]
fn uniform_bitmap_clones_share_bulk_reads_and_detach_on_write() {
    use crate::bitmap::{bitmap_data::BitmapData, operations};
    let manager = ruffle_memory::MemoryManager::new(32 << 20);
    let scope = manager.player();
    scope.enter(|| {
        gc_arena::arena::rootless_mutate(|mc| {
            let mut renderer = renderer(false);
            let source = BitmapData::new(mc, 128, 128, true, 0x80800000);
            let copy = source.clone_data(mc, &mut renderer).unwrap();
            assert_eq!(manager.snapshot().active_bytes, 0);
            let source_data = source.sync(&mut renderer).unwrap();
            let copy_data = copy.sync(&mut renderer).unwrap();
            assert_eq!(
                source_data.borrow().pixels().as_ptr(),
                copy_data.borrow().pixels().as_ptr()
            );
            assert_eq!(manager.snapshot().active_bytes, 128 * 128 * 4);
            operations::set_pixel32(mc, &mut renderer, copy, 0, 0, 0xff447755).unwrap();
            assert_ne!(
                source_data.borrow().pixels()[0],
                copy_data.borrow().pixels()[0]
            );
            assert_eq!(
                source_data.borrow().pixels()[0],
                copy_data.borrow().pixels()[1]
            );
            assert_eq!(manager.snapshot().active_bytes, 2 * 128 * 128 * 4);
        })
    });
}

use ruffle_render::backend::{
    BitmapCacheEntry, RenderBackend, ShapeHandle, ShapeHandleImpl, ViewportDimensions,
};
use ruffle_render::bitmap::{
    Bitmap, BitmapHandle, BitmapHandleImpl, BitmapSource, PixelRegion, RgbaBufRead, SyncHandle,
};
use ruffle_render::commands::CommandList;
use ruffle_render::error::Error;
use ruffle_render::pixel_bender::{PixelBenderShader, PixelBenderShaderHandle};
use ruffle_render::pixel_bender_support::PixelBenderShaderArgument;
use ruffle_render::quality::StageQuality;
use ruffle_render::shape_utils::DistilledShape;
use swf::Color;

use ruffle_render::backend::{Context3D, Context3DProfile, PixelBenderOutput, PixelBenderTarget};

pub struct TestRenderer {
    health: ruffle_render::error::RendererHealth,
    dimensions: ViewportDimensions,
    managed: bool,
    retained: Option<BitmapHandle>,
    registrations: usize,
    pub(crate) loader_uploads: Rc<std::cell::RefCell<Vec<ruffle_render::bitmap::LoaderRequest>>>,
    fail_upload: bool,
    regional: bool,
    reads: Vec<PixelRegion>,
    uploads: Vec<PixelRegion>,
    sync_reads: usize,
    copies: usize,
    fail_read: bool,
    pub(crate) allocation_failure: Option<ruffle_render::error::RenderFailure>,
    pub(crate) allocation_calls: std::rc::Rc<std::cell::Cell<usize>>,
}

#[derive(Clone, Debug)]
struct TestBitmapHandle {
    pixels: std::cell::RefCell<Vec<u8>>,
    width: u32,
}
impl BitmapHandleImpl for TestBitmapHandle {}

#[derive(Clone, Debug)]
struct NullShapeHandle;
impl ShapeHandleImpl for NullShapeHandle {}

impl RenderBackend for TestRenderer {
    fn health(&self) -> &ruffle_render::error::RendererHealth {
        &self.health
    }
    fn viewport_dimensions(&self) -> ViewportDimensions {
        self.dimensions
    }
    fn set_viewport_dimensions(
        &mut self,
        dimensions: ViewportDimensions,
    ) -> Result<(), ruffle_render::error::RenderFailure> {
        self.health.check()?;
        let output = (|| {
            self.dimensions = dimensions;
        })();
        self.health.check()?;
        Ok(output)
    }
    fn register_shape(
        &mut self,
        _shape: DistilledShape,
        _bitmap_source: &dyn BitmapSource,
    ) -> Result<ShapeHandle, ruffle_render::error::RenderFailure> {
        self.health.check()?;
        let output = (|| ShapeHandle(Arc::new(NullShapeHandle)))();
        self.health.check()?;
        Ok(output)
    }

    fn render_offscreen(
        &mut self,
        _handle: BitmapHandle,
        _commands: CommandList,
        _quality: StageQuality,
        _bounds: PixelRegion,
    ) -> Result<Option<Box<dyn SyncHandle>>, ruffle_render::error::Error> {
        self.health.check()?;
        let output = (|| None)();
        self.health.check()?;
        Ok(output)
    }

    fn submit_prepared_frame(
        &mut self,
        _clear: Color,
        _commands: CommandList,
        _cache_entries: Vec<BitmapCacheEntry>,
    ) -> Result<ruffle_render::backend::FrameSubmission, ruffle_render::error::RenderFailure> {
        self.health.check()?;
        Ok(ruffle_render::backend::FrameSubmission::NoOutput)
    }
    fn register_bitmap(&mut self, bitmap: Bitmap<'_>) -> Result<BitmapHandle, Error> {
        self.registrations += 1;
        if let Some(request) = bitmap.loader_request() {
            self.loader_uploads.borrow_mut().push(request);
        }
        Ok(BitmapHandle(Arc::new(TestBitmapHandle {
            width: bitmap.width(),
            pixels: std::cell::RefCell::new(bitmap.to_rgba().data().to_vec()),
        })))
    }
    fn cache_immutable_bitmap(&mut self, bitmap: &BitmapHandle) -> bool {
        if self.managed {
            self.retained = Some(bitmap.clone());
        }
        self.managed
    }

    fn update_texture(
        &mut self,
        handle: &BitmapHandle,
        bitmap: Bitmap<'_>,
        region: PixelRegion,
    ) -> Result<(), Error> {
        if self.fail_upload {
            return Err(ruffle_render::error::RenderFailure::new(
                ruffle_render::error::RenderFailureKind::OutOfMemory,
                "injected upload failure",
            )
            .into());
        }
        self.uploads.push(region);
        let bitmap = bitmap.to_rgba();
        let target = test_handle(handle);
        let mut pixels = target.pixels.borrow_mut();
        for y in region.y_min..region.y_max {
            let start = ((y * target.width + region.x_min) * 4) as usize;
            let end = start + region.width() as usize * 4;
            pixels[start..end].copy_from_slice(&bitmap.data()[start..end]);
        }
        Ok(())
    }

    fn copy_bitmap_region(
        &mut self,
        source: BitmapHandle,
        source_region: PixelRegion,
        destination: BitmapHandle,
        destination_region: PixelRegion,
    ) -> Result<Option<Box<dyn SyncHandle>>, Error> {
        self.copies += 1;
        let source = snapshot(&source, source_region);
        let dest = test_handle(&destination);
        {
            let mut pixels = dest.pixels.borrow_mut();
            for y in 0..destination_region.height() {
                let start = (((destination_region.y_min + y) * dest.width
                    + destination_region.x_min)
                    * 4) as usize;
                let row = (y * source.stride) as usize;
                pixels[start..start + source.stride as usize]
                    .copy_from_slice(&source.pixels[row..row + source.stride as usize]);
            }
        }
        Ok(Some(Box::new(snapshot(&destination, destination_region))))
    }

    fn create_context3d(
        &mut self,
        _profile: Context3DProfile,
    ) -> Result<Box<dyn Context3D>, Error> {
        self.allocation_calls.set(self.allocation_calls.get() + 1);
        if let Some(error) = &self.allocation_failure {
            return Err(error.clone().into());
        }
        Err(Error::Unimplemented("createContext3D".into()))
    }

    fn debug_info(&self) -> Cow<'static, str> {
        Cow::Borrowed("Renderer: Null")
    }

    fn name(&self) -> &'static str {
        ""
    }

    fn set_quality(
        &mut self,
        _quality: StageQuality,
    ) -> Result<(), ruffle_render::error::RenderFailure> {
        self.health.check()?;
        let output = (|| {})();
        self.health.check()?;
        Ok(output)
    }

    fn run_pixelbender_shader(
        &mut self,
        _shader: PixelBenderShaderHandle,
        _arguments: &[PixelBenderShaderArgument],
        _target: &PixelBenderTarget,
    ) -> Result<PixelBenderOutput, Error> {
        Err(Error::Unimplemented("Pixel bender shader".into()))
    }

    fn resolve_sync_handle(
        &mut self,
        handle: Box<dyn SyncHandle>,
        with_rgba: RgbaBufRead,
    ) -> Result<(), Error> {
        self.sync_reads += 1;
        if self.fail_read {
            return Err(ruffle_render::error::RenderFailure::new(
                ruffle_render::error::RenderFailureKind::MapFailed,
                "injected map failure",
            )
            .into());
        }
        let snapshot = (handle.as_ref() as &dyn std::any::Any)
            .downcast_ref::<SnapshotSync>()
            .unwrap();
        with_rgba(&snapshot.pixels, snapshot.stride);
        Ok(())
    }

    fn supports_bitmap_region_readback(&self) -> bool {
        self.regional
    }

    fn read_bitmap_region(
        &mut self,
        handle: &BitmapHandle,
        region: PixelRegion,
        with_rgba: RgbaBufRead,
    ) -> Result<Option<()>, Error> {
        if !self.regional {
            return Ok(None);
        }
        self.reads.push(region);
        if self.fail_read {
            return Err(ruffle_render::error::RenderFailure::new(
                ruffle_render::error::RenderFailureKind::MapFailed,
                "injected map failure",
            )
            .into());
        }
        let snapshot = snapshot(handle, region);
        with_rgba(&snapshot.pixels, snapshot.stride);
        Ok(Some(()))
    }

    fn compile_pixelbender_shader(
        &mut self,
        _shader: PixelBenderShader,
    ) -> Result<PixelBenderShaderHandle, Error> {
        self.allocation_calls.set(self.allocation_calls.get() + 1);
        if let Some(error) = &self.allocation_failure {
            return Err(error.clone().into());
        }
        Err(Error::Unimplemented(
            "Pixel bender shader compilation".into(),
        ))
    }

    fn is_offscreen_supported(&self) -> bool {
        true
    }

    fn create_empty_texture(
        &mut self,
        _width: NonZeroU32,
        _height: NonZeroU32,
    ) -> Result<BitmapHandle, Error> {
        Ok(BitmapHandle(Arc::new(TestBitmapHandle {
            pixels: Default::default(),
            width: _width.get(),
        })))
    }
}

use super::{Pixels, SymbolImage};
use crate::character::CompressedBitmap;
use std::{borrow::Cow as Borrowed, io::Write, rc::Rc};

fn source() -> Rc<SymbolImage> {
    source_with_id(1)
}

fn source_with_id(id: u16) -> Rc<SymbolImage> {
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(&[255, 12, 34, 56]).unwrap();
    Rc::new(SymbolImage::new(
        None,
        CompressedBitmap::Lossless(swf::DefineBitsLossless {
            version: 2,
            id,
            format: swf::BitmapFormat::Rgb32,
            width: 1,
            height: 1,
            data: Borrowed::Owned(encoder.finish().unwrap()),
        }),
    ))
}
pub(crate) fn renderer(managed: bool) -> TestRenderer {
    TestRenderer {
        health: Default::default(),
        dimensions: ViewportDimensions {
            width: 1,
            height: 1,
            scale_factor: 1.0,
        },
        managed,
        retained: None,
        registrations: 0,
        loader_uploads: Default::default(),
        fail_upload: false,
        regional: false,
        reads: vec![],
        uploads: vec![],
        sync_reads: 0,
        copies: 0,
        fail_read: false,
        allocation_failure: None,
        allocation_calls: Default::default(),
    }
}
fn test_handle(handle: &BitmapHandle) -> &TestBitmapHandle {
    (handle.0.as_ref() as &dyn std::any::Any)
        .downcast_ref::<TestBitmapHandle>()
        .unwrap()
}
fn rgba(handle: &BitmapHandle) -> Vec<u8> {
    test_handle(handle).pixels.borrow().clone()
}

#[derive(Debug)]
struct SnapshotSync {
    pixels: Vec<u8>,
    stride: u32,
}
impl SyncHandle for SnapshotSync {}
fn snapshot(handle: &BitmapHandle, region: PixelRegion) -> SnapshotSync {
    let handle = test_handle(handle);
    let pixels = handle.pixels.borrow();
    let mut data = Vec::new();
    for y in region.y_min..region.y_max {
        let start = ((y * handle.width + region.x_min) * 4) as usize;
        data.extend_from_slice(&pixels[start..start + region.width() as usize * 4]);
    }
    SnapshotSync {
        pixels: data,
        stride: region.width() * 4,
    }
}
fn gpu_fill(handle: &BitmapHandle, region: PixelRegion, color: [u8; 4]) -> Box<dyn SyncHandle> {
    let target = test_handle(handle);
    {
        let mut pixels = target.pixels.borrow_mut();
        for y in region.y_min..region.y_max {
            for x in region.x_min..region.x_max {
                let offset = ((y * target.width + x) * 4) as usize;
                pixels[offset..offset + 4].copy_from_slice(&color);
            }
        }
    }
    Box::new(snapshot(handle, region))
}

#[test]
fn tracking_current_pixels_releases_a_redundant_version_reader() {
    use crate::bitmap::{bitmap_data::BitmapData, operations::get_pixel32};
    #[derive(Debug)]
    struct Reader(BitmapHandle, Arc<()>);
    impl SyncHandle for Reader {
        fn can_read_current_without_token(&self, bitmap: &BitmapHandle) -> bool {
            self.0 == *bitmap
        }
    }
    gc_arena::arena::rootless_mutate(|mc| {
        let bitmap = BitmapData::new(mc, 4, 1, true, 0xff000000);
        let mut renderer = renderer(false);
        renderer.regional = true;
        let handle = bitmap.bitmap_handle(mc, &mut renderer).unwrap();
        let region = PixelRegion::for_whole_size(4, 1);
        drop(gpu_fill(&handle, region, [10, 20, 30, 255]));
        let reader = Reader(handle, Arc::new(()));
        let live = Arc::downgrade(&reader.1);
        bitmap.mark_gpu_dirty(mc, Box::new(reader), region);
        assert!(
            live.upgrade().is_none(),
            "tracking current GPU pixels retained a version reader"
        );
        assert_eq!(
            get_pixel32(bitmap, &mut renderer, 1, 0).unwrap(),
            0xff0a141e
        );
    });
}

#[test]
fn evicted_symbol_can_release_its_gpu_view_while_as3_pixels_remain_alive() {
    let mut renderer = renderer(true);
    let mut pixels = Pixels::symbol(source());
    let first = pixels.view(&mut renderer).unwrap().unwrap();
    assert_eq!(rgba(&first), [12, 34, 56, 255]);
    let weak = Arc::downgrade(&first.0);
    drop(first);
    renderer.retained = None; // Backend evicts its reference; no GC/AS3 lifetime change.
    assert!(
        weak.upgrade().is_none(),
        "immutable Pixels pinned an evicted GPU allocation"
    );
    let rebuilt = pixels.view(&mut renderer).unwrap().unwrap();
    assert_eq!(rgba(&rebuilt), [12, 34, 56, 255]);
    assert_eq!(renderer.registrations, 2);
}

#[test]
fn unmanaged_backend_preserves_strong_view_retention() {
    let mut renderer = renderer(false);
    let mut pixels = Pixels::symbol(source());
    let first = pixels.view(&mut renderer).unwrap().unwrap();
    let weak = Arc::downgrade(&first.0);
    drop(first);
    assert!(weak.upgrade().is_some());
    pixels.view(&mut renderer).unwrap().unwrap();
    assert_eq!(renderer.registrations, 1);
}

#[test]
fn cached_symbol_clones_share_uploads_but_mutations_detach() {
    let mut renderer = renderer(true);
    let mut first = Pixels::symbol(source());
    let mut second = first.clone();
    let original = first.view(&mut renderer).unwrap().unwrap();
    let shared = second.view(&mut renderer).unwrap().unwrap();
    assert!(Arc::ptr_eq(&original.0, &shared.0));
    assert_eq!(renderer.registrations, 1);
    first[0] = crate::bitmap::bitmap_data::Color::from(0xffabcdef);
    assert!(first.view(&mut renderer).unwrap().is_none());
    assert_eq!(
        rgba(&second.view(&mut renderer).unwrap().unwrap()),
        [12, 34, 56, 255]
    );
    assert_ne!(first[0], second[0]);
}

#[test]
fn failed_bitmap_upload_preserves_dirty_cpu_pixels() {
    use crate::bitmap::bitmap_data::{BitmapRawData, DirtyState};
    let mut renderer = renderer(false);
    gc_arena::arena::rootless_mutate(|mc| {
        let mut raw = BitmapRawData::new(1, 1, true, 0xff123456);
        raw.bitmap_handle(&mut renderer).unwrap();
        raw.set_cpu_dirty(mc, PixelRegion::for_whole_size(1, 1));
        renderer.fail_upload = true;
        let error = raw.update_dirty_texture(&mut renderer).unwrap_err();
        assert_eq!(
            error.kind(),
            ruffle_render::error::RenderFailureKind::OutOfMemory
        );
        assert!(matches!(raw.dirty_state(), DirtyState::CpuModified(_)));
        assert_eq!(raw.pixels_rgba(), [0x12, 0x34, 0x56, 0xff]);
        assert!(renderer.health().check().is_err());
    });
}

#[test]
fn sequential_pixel_reads_amortize_native_synchronization_without_changing_pixels() {
    use crate::bitmap::{bitmap_data::BitmapData, operations::get_pixel32};
    gc_arena::arena::rootless_mutate(|mc| {
        // The manual game trace scanned 16,380 pixels using 16,380 synchronous reads.
        let bitmap = BitmapData::new(mc, 130, 126, true, 0xff000000);
        let mut renderer = renderer(false);
        renderer.regional = true;
        let handle = bitmap.bitmap_handle(mc, &mut renderer).unwrap();
        {
            let mut pixels = test_handle(&handle).pixels.borrow_mut();
            for y in 0..126usize {
                for x in 0..130usize {
                    let offset = (y * 130 + x) * 4;
                    pixels[offset..offset + 4].copy_from_slice(&[
                        x as u8,
                        y as u8,
                        (x ^ y) as u8,
                        255,
                    ]);
                }
            }
        }
        let full = PixelRegion::for_whole_size(130, 126);
        bitmap.mark_gpu_dirty(mc, Box::new(snapshot(&handle, full)), full);
        for y in 0..126 {
            for x in 0..130 {
                assert_eq!(
                    get_pixel32(bitmap, &mut renderer, x, y).unwrap(),
                    0xff000000 | (x << 16) | (y << 8) | (x ^ y),
                    "pixel ({x}, {y})"
                );
            }
        }
        assert!(
            renderer.reads.len() <= 32,
            "a coherent pixel scan issued {} GPU synchronizations",
            renderer.reads.len()
        );
        let bytes: u64 = renderer
            .reads
            .iter()
            .map(|r| u64::from(r.width()) * u64::from(r.height()) * 4)
            .sum();
        assert_eq!(
            bytes, 65_520,
            "each authoritative pixel should be copied once"
        );
        assert!(bitmap.can_read(full));
    });
}

#[test]
fn nearby_pixel_reads_keep_prefetch_bounded_and_preserve_cpu_writes() {
    use crate::bitmap::{
        bitmap_data::BitmapData,
        operations::{get_pixel32, set_pixel32},
    };
    gc_arena::arena::rootless_mutate(|mc| {
        let bitmap = BitmapData::new(mc, 256, 128, true, 0xff000000);
        let mut renderer = renderer(false);
        renderer.regional = true;
        let handle = bitmap.bitmap_handle(mc, &mut renderer).unwrap();
        let full = PixelRegion::for_whole_size(256, 128);
        bitmap.mark_gpu_dirty(mc, gpu_fill(&handle, full, [10, 20, 30, 255]), full);
        set_pixel32(mc, &mut renderer, bitmap, 6, 3, 0xffabcdef).unwrap();
        assert_eq!(
            get_pixel32(bitmap, &mut renderer, 2, 1).unwrap(),
            0xff0a141e
        );
        assert_eq!(
            renderer.reads,
            [PixelRegion::for_pixel(2, 1)],
            "one isolated read stays exact"
        );
        assert_eq!(
            get_pixel32(bitmap, &mut renderer, 3, 1).unwrap(),
            0xff0a141e
        );
        assert!(
            bitmap.can_read(PixelRegion::for_pixel(4, 1)),
            "nearby reads should warm adjacent pixels"
        );
        assert_eq!(
            get_pixel32(bitmap, &mut renderer, 6, 3).unwrap(),
            0xffabcdef
        );
        assert!(
            !bitmap.can_read(PixelRegion::for_pixel(255, 127)),
            "do not download the entire bitmap"
        );
        let bytes: u64 = renderer
            .reads
            .iter()
            .map(|r| u64::from(r.width()) * u64::from(r.height()) * 4)
            .sum();
        assert!(bytes <= 4_096, "bounded locality hint read {bytes} bytes");
        bitmap.bitmap_handle(mc, &mut renderer).unwrap();
        assert_eq!(renderer.uploads, [PixelRegion::for_pixel(6, 3)]);
        let changed = PixelRegion::for_pixel(4, 1);
        bitmap.mark_gpu_dirty(mc, gpu_fill(&handle, changed, [70, 80, 90, 255]), changed);
        assert_eq!(
            get_pixel32(bitmap, &mut renderer, 4, 1).unwrap(),
            0xff46505a
        );
    });
}

#[test]
fn bitmap_validity_reads_only_requested_gpu_pixels_and_preserves_cpu_holes() {
    use crate::bitmap::{
        bitmap_data::BitmapData,
        operations::{get_pixel32, set_pixel32},
    };
    gc_arena::arena::rootless_mutate(|mc| {
        let bitmap = BitmapData::new(mc, 8, 2, true, 0xff000000);
        let mut renderer = renderer(false);
        renderer.regional = true;
        let handle = bitmap.bitmap_handle(mc, &mut renderer).unwrap();
        let full = PixelRegion::for_whole_size(8, 2);
        bitmap.mark_gpu_dirty(mc, gpu_fill(&handle, full, [10, 20, 30, 255]), full);
        assert_eq!(
            (bitmap.width(), bitmap.height(), bitmap.transparency()),
            (8, 2, true)
        );
        assert!(renderer.reads.is_empty());
        assert_eq!(
            get_pixel32(bitmap, &mut renderer, 2, 1).unwrap(),
            0xff0a141e
        );
        assert_eq!(renderer.reads, [PixelRegion::for_pixel(2, 1)]);
        assert!(!bitmap.can_read(PixelRegion::for_pixel(3, 1)));
        set_pixel32(mc, &mut renderer, bitmap, 1, 0, 0xffabcdef).unwrap();
        set_pixel32(mc, &mut renderer, bitmap, 6, 0, 0xff123456).unwrap();
        assert_eq!(
            renderer.reads.len(),
            1,
            "overwrites must not read GPU pixels"
        );
        bitmap.bitmap_handle(mc, &mut renderer).unwrap();
        assert_eq!(
            renderer.uploads,
            [PixelRegion::for_pixel(1, 0), PixelRegion::for_pixel(6, 0)]
        );
        let second = PixelRegion::for_region(4, 1, 2, 1);
        bitmap.mark_gpu_dirty(mc, gpu_fill(&handle, second, [70, 80, 90, 255]), second);
        let read = bitmap.sync(&mut renderer).unwrap();
        let read = read.borrow();
        assert_eq!(u32::from(read.get_pixel32_raw(0, 0)), 0xff0a141e);
        assert_eq!(u32::from(read.get_pixel32_raw(1, 0)), 0xffabcdef);
        assert_eq!(u32::from(read.get_pixel32_raw(6, 0)), 0xff123456);
        assert_eq!(u32::from(read.get_pixel32_raw(4, 1)), 0xff46505a);
        assert_eq!(renderer.sync_reads, 0);
    });
}

#[test]
fn bitmap_validity_fallback_sync_preserves_cpu_overwrites_and_buffer_origin() {
    use crate::bitmap::{
        bitmap_data::BitmapData,
        operations::{get_pixel32, set_pixel32},
    };
    gc_arena::arena::rootless_mutate(|mc| {
        let bitmap = BitmapData::new(mc, 8, 2, true, 0xff000000);
        let mut renderer = renderer(false);
        let handle = bitmap.bitmap_handle(mc, &mut renderer).unwrap();
        let area = PixelRegion::for_region(2, 1, 5, 1);
        bitmap.mark_gpu_dirty(mc, gpu_fill(&handle, area, [40, 50, 60, 255]), area);
        set_pixel32(mc, &mut renderer, bitmap, 3, 1, 0xff112233).unwrap();
        assert_eq!(renderer.sync_reads, 0);
        assert_eq!(
            get_pixel32(bitmap, &mut renderer, 6, 1).unwrap(),
            0xff28323c
        );
        assert_eq!(
            get_pixel32(bitmap, &mut renderer, 3, 1).unwrap(),
            0xff112233
        );
        assert_eq!(
            get_pixel32(bitmap, &mut renderer, 0, 0).unwrap(),
            0xff000000
        );
        assert_eq!(renderer.sync_reads, 1);
    });
}

#[test]
fn bitmap_validity_failed_partial_read_stays_terminal() {
    use crate::bitmap::{bitmap_data::BitmapData, operations::get_pixel32};
    gc_arena::arena::rootless_mutate(|mc| {
        let bitmap = BitmapData::new(mc, 4, 1, true, 0xff000000);
        let mut renderer = renderer(false);
        renderer.regional = true;
        let handle = bitmap.bitmap_handle(mc, &mut renderer).unwrap();
        let full = PixelRegion::for_whole_size(4, 1);
        bitmap.mark_gpu_dirty(mc, gpu_fill(&handle, full, [1, 2, 3, 255]), full);
        assert_eq!(
            get_pixel32(bitmap, &mut renderer, 0, 0).unwrap(),
            0xff010203
        );
        renderer.fail_read = true;
        assert!(get_pixel32(bitmap, &mut renderer, 3, 0).is_err());
        assert!(get_pixel32(bitmap, &mut renderer, 0, 0).is_err());
        assert!(!bitmap.can_read(full));
    });
}

#[test]
fn bitmap_validity_full_overwrite_discards_pending_readback() {
    use crate::bitmap::{
        bitmap_data::BitmapData,
        operations::{fill_rect, get_pixel32},
    };
    gc_arena::arena::rootless_mutate(|mc| {
        let bitmap = BitmapData::new(mc, 4, 1, true, 0xff000000);
        let mut renderer = renderer(false);
        renderer.regional = true;
        let handle = bitmap.bitmap_handle(mc, &mut renderer).unwrap();
        let full = PixelRegion::for_whole_size(4, 1);
        bitmap.mark_gpu_dirty(mc, gpu_fill(&handle, full, [1, 2, 3, 255]), full);
        renderer.fail_read = true;
        fill_rect(mc, &mut renderer, bitmap, 0, 0, 4, 1, 0xff345678).unwrap();
        assert_eq!(
            get_pixel32(bitmap, &mut renderer, 3, 0).unwrap(),
            0xff345678
        );
        assert!(renderer.reads.is_empty());
        assert_eq!(renderer.sync_reads, 0);
    });
}

#[test]
fn bitmap_validity_gpu_copies_chain_and_self_overlap_in_as_order() {
    use crate::bitmap::{
        bitmap_data::BitmapData,
        operations::{copy_pixels_inner, get_pixel32, set_pixel32},
    };
    gc_arena::arena::rootless_mutate(|mc| {
        let source = BitmapData::new(mc, 6, 1, true, 0xff102030);
        let dest = BitmapData::new(mc, 6, 1, true, 0xff000000);
        let mut renderer = renderer(false);
        renderer.regional = true;
        copy_pixels_inner(mc, &mut renderer, dest, source, (0, 0, 2, 1), (0, 0), false).unwrap();
        set_pixel32(mc, &mut renderer, source, 0, 0, 0xffaabbcc).unwrap();
        copy_pixels_inner(mc, &mut renderer, dest, source, (0, 0, 2, 1), (4, 0), false).unwrap();
        copy_pixels_inner(mc, &mut renderer, dest, dest, (0, 0, 5, 1), (1, 0), false).unwrap();
        assert_eq!(renderer.copies, 3);
        assert!(renderer.reads.is_empty());
        let expected = [
            0xff102030, 0xff102030, 0xff102030, 0xff000000, 0xff000000, 0xffaabbcc,
        ];
        for (x, color) in expected.into_iter().enumerate() {
            assert_eq!(
                get_pixel32(dest, &mut renderer, x as u32, 0).unwrap(),
                color
            );
        }
    });
}

#[test]
fn bitmap_validity_self_overlap_preserves_flash_down_left_row_smear() {
    use crate::bitmap::{
        bitmap_data::{BitmapData, Color},
        operations::{copy_pixels_inner, get_pixel32},
    };
    gc_arena::arena::rootless_mutate(|mc| {
        let pixels = (1..=9).map(|n| Color::from(0xff000000 | n)).collect();
        let bitmap = BitmapData::new_with_pixels(mc, 3, 3, true, pixels);
        let mut renderer = renderer(false);
        renderer.regional = true;
        copy_pixels_inner(
            mc,
            &mut renderer,
            bitmap,
            bitmap,
            (1, 0, 2, 2),
            (0, 1),
            false,
        )
        .unwrap();
        let expected = [1, 2, 3, 2, 3, 6, 3, 6, 9];
        for (i, blue) in expected.into_iter().enumerate() {
            assert_eq!(
                get_pixel32(bitmap, &mut renderer, i as u32 % 3, i as u32 / 3).unwrap(),
                0xff000000 | blue
            );
        }
        assert_eq!(renderer.copies, 0);
    });
}

#[test]
fn bitmap_validity_clean_alias_reads_do_not_require_an_exclusive_borrow() {
    use crate::bitmap::{bitmap_data::BitmapData, operations::compare};
    gc_arena::arena::rootless_mutate(|mc| {
        let bitmap = BitmapData::new(mc, 2, 1, true, 0xff123456);
        let mut renderer = renderer(false);
        assert!(
            compare(mc, &mut renderer, bitmap, bitmap)
                .unwrap()
                .is_none()
        );
    });
}

#[test]
fn bitmap_validity_self_hit_test_reads_both_disjoint_gpu_regions() {
    use crate::bitmap::{bitmap_data::BitmapData, operations::hit_test_bitmapdata};
    gc_arena::arena::rootless_mutate(|mc| {
        let bitmap = BitmapData::new(mc, 8, 1, true, 0);
        let mut renderer = renderer(false);
        renderer.regional = true;
        let handle = bitmap.bitmap_handle(mc, &mut renderer).unwrap();
        let full = PixelRegion::for_whole_size(8, 1);
        bitmap.mark_gpu_dirty(mc, gpu_fill(&handle, full, [10, 20, 30, 255]), full);
        assert!(
            hit_test_bitmapdata(&mut renderer, bitmap, (0, 0), 255, bitmap, (7, 0), 255).unwrap()
        );
        assert_eq!(renderer.reads.len(), 2);
        assert!(renderer.reads.contains(&PixelRegion::for_pixel(0, 0)));
        assert!(renderer.reads.contains(&PixelRegion::for_pixel(7, 0)));
        assert!(!bitmap.can_read(PixelRegion::for_pixel(3, 0)));
    });
}

#[test]
fn bitmap_validity_alpha_copy_reads_aliased_source_regions_before_borrowing() {
    use crate::bitmap::{
        bitmap_data::BitmapData,
        operations::{copy_pixels_with_alpha_source, get_pixel32},
    };
    let mut renderer = renderer(false);
    renderer.regional = true;
    let player = crate::PlayerBuilder::new().with_renderer(renderer).build();
    player
        .lock()
        .unwrap()
        .mutate_with_update_context(|context| {
            let source = BitmapData::new(context.gc(), 8, 1, true, 0);
            let target = BitmapData::new(context.gc(), 1, 1, true, 0);
            let handle = source
                .bitmap_handle(context.gc(), context.renderer)
                .unwrap();
            let full = PixelRegion::for_whole_size(8, 1);
            source.mark_gpu_dirty(
                context.gc(),
                gpu_fill(&handle, full, [10, 20, 30, 255]),
                full,
            );
            copy_pixels_with_alpha_source(
                context,
                target,
                source,
                (0, 0, 1, 1),
                (0, 0),
                source,
                (7, 0),
                false,
            )
            .unwrap();
            assert_eq!(
                get_pixel32(target, context.renderer, 0, 0).unwrap(),
                0xff0a141e
            );
            assert!(!source.can_read(PixelRegion::for_pixel(3, 0)));
        });
}

#[test]
fn actionscript_pixel_words_have_native_bgra_byte_order() {
    let color = crate::bitmap::bitmap_data::Color::bgra_u32(0x80563412);
    assert_eq!(bytemuck::bytes_of(&color), &[0x12, 0x34, 0x56, 0x80]);
    assert_eq!(color.to_bgra_u32(), 0x80563412);
}

#[test]
fn independent_symbols_share_compressed_bytes_and_clones_share_decoded_pixels() {
    let left = source();
    let right = source_with_id(2);
    let address = |s: &SymbolImage| match s.data.encoded() {
        ruffle_render::bitmap::EncodedBitmap::Lossless(tag) => tag.data.as_ptr(),
        _ => unreachable!(),
    };
    assert_eq!(address(&left), address(&right));
    let a = Pixels::symbol(left.clone());
    let mut b = Pixels::symbol(left);
    assert_eq!(a.as_ptr(), b.as_ptr());
    b[0] = 0xff003355.into();
    assert_ne!(a.as_ptr(), b.as_ptr());
}
