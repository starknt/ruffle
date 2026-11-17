use std::borrow::Cow;
use std::num::NonZeroU32;
use std::sync::Arc;

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
    fail_upload: bool,
}

#[derive(Clone, Debug)]
struct TestBitmapHandle {
    pixels: Vec<u8>,
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

    fn submit_frame(
        &mut self,
        _clear: Color,
        _commands: CommandList,
        _cache_entries: Vec<BitmapCacheEntry>,
    ) -> Result<(), ruffle_render::error::RenderFailure> {
        self.health.check()?;
        let output = (|| {})();
        self.health.check()?;
        Ok(output)
    }
    fn register_bitmap(&mut self, bitmap: Bitmap<'_>) -> Result<BitmapHandle, Error> {
        self.registrations += 1;
        Ok(BitmapHandle(Arc::new(TestBitmapHandle {
            pixels: bitmap.to_rgba().data().to_vec(),
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
        _handle: &BitmapHandle,
        _bitmap: Bitmap<'_>,
        _region: PixelRegion,
    ) -> Result<(), Error> {
        if self.fail_upload {
            return Err(ruffle_render::error::RenderFailure::new(
                ruffle_render::error::RenderFailureKind::OutOfMemory,
                "injected upload failure",
            )
            .into());
        }
        Ok(())
    }

    fn create_context3d(
        &mut self,
        _profile: Context3DProfile,
    ) -> Result<Box<dyn Context3D>, Error> {
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
        _handle: Box<dyn SyncHandle>,
        _with_rgba: RgbaBufRead,
    ) -> Result<(), Error> {
        Err(Error::Unimplemented("Sync handle resolution".into()))
    }

    fn compile_pixelbender_shader(
        &mut self,
        _shader: PixelBenderShader,
    ) -> Result<PixelBenderShaderHandle, Error> {
        Err(Error::Unimplemented(
            "Pixel bender shader compilation".into(),
        ))
    }

    fn create_empty_texture(
        &mut self,
        _width: NonZeroU32,
        _height: NonZeroU32,
    ) -> Result<BitmapHandle, Error> {
        Ok(BitmapHandle(Arc::new(TestBitmapHandle {
            pixels: Vec::new(),
        })))
    }
}

use super::{Pixels, SymbolImage};
use crate::character::CompressedBitmap;
use std::{borrow::Cow as Borrowed, io::Write, rc::Rc};

fn source() -> Rc<SymbolImage> {
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(&[255, 12, 34, 56]).unwrap();
    Rc::new(SymbolImage::new(CompressedBitmap::Lossless(
        swf::DefineBitsLossless {
            version: 2,
            id: 1,
            format: swf::BitmapFormat::Rgb32,
            width: 1,
            height: 1,
            data: Borrowed::Owned(encoder.finish().unwrap()),
        },
    )))
}
fn renderer(managed: bool) -> TestRenderer {
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
        fail_upload: false,
    }
}
fn rgba(handle: &BitmapHandle) -> &[u8] {
    (handle.0.as_ref() as &dyn std::any::Any)
        .downcast_ref::<TestBitmapHandle>()
        .unwrap()
        .pixels
        .as_slice()
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
