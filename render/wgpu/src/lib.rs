#![recursion_limit = "256"]
// Renderer ownership stays on the player thread; preparation workers use owned CPU data.
#![allow(clippy::arc_with_non_send_sync)]

#[cfg(target_family = "wasm")]
compile_error!("ruffle_render_wgpu supports native targets only");

use crate::backend::ActiveFrame;
use crate::bitmaps::BitmapSamplers;
use crate::buffer_pool::{BufferPool, PoolEntry};
use crate::mesh::BitmapBinds;
use crate::pipelines::Pipelines;
use crate::target::{RenderTarget, SwapChainTarget};
use crate::utils::{BufferDimensions, capture_image, format_list, get_backend_names};
use bytemuck::{Pod, Zeroable};
use descriptors::Descriptors;
pub use gpu_profiler as wgpu_profiler;
use ruffle_render::backend::RawTexture;
use ruffle_render::bitmap::{BitmapHandle, BitmapHandleImpl, PixelRegion, SyncHandle};
use ruffle_render::shape_utils::GradientType;
use ruffle_render::tessellator::{Gradient as TessGradient, Vertex as TessVertex};
use std::any::Any;
use std::sync::OnceLock;
use std::sync::{
    Arc,
    mpsc::{Receiver, TryRecvError},
};
use swf::GradientSpread;
pub use wgpu;

type Error = Box<dyn std::error::Error>;

#[macro_use]
pub mod utils;

mod bitmap_batch;
mod bitmap_cache;
mod bitmap_recipe;
mod bitmap_residency;
mod bitmaps;
mod context3d;
mod filtered_tiles;
mod frame_upload;
mod globals;
mod gpu_metrics;
pub mod gpu_profiler;
pub mod gpu_resources;
pub mod gpu_runtime;
mod graph_encoder;
mod pass_graph;
mod pipelines;
mod pixel_bender;
mod prepare;
mod render_snapshot;
pub use render_snapshot::FramePacket;
mod resource_scheduler;
mod shape_recipe;
pub mod target;
mod transfers;
pub use ruffle_render::profile;
mod idle_pool;

pub mod backend;
mod blend;
mod blend_inputs;
mod buffer_builder;
mod buffer_pool;
#[cfg(feature = "clap")]
pub mod clap;
pub mod descriptors;
pub mod display_bitmap;
mod filters;
mod layouts;
mod material_cache;
mod mesh;
mod shaders;
mod surface;

impl BitmapHandleImpl for Texture {}

pub fn as_texture<'a>(
    handle: &'a BitmapHandle,
    device: &crate::gpu_resources::Device,
) -> Result<&'a Texture, crate::gpu_runtime::RenderFailure> {
    let value = &*handle.0 as &dyn Any;
    if let Some(recipe) = value.downcast_ref::<crate::bitmap_recipe::BitmapRecipe>() {
        return recipe.resolve(device);
    }
    if let Some(cache) = value.downcast_ref::<crate::bitmap_cache::CacheImage>() {
        return cache.resolve(device);
    }
    value.downcast_ref::<Texture>().ok_or_else(|| {
        crate::gpu_runtime::RenderFailure::new(
            crate::gpu_runtime::RenderFailureKind::Validation,
            "bitmap handle belongs to another renderer",
        )
    })
}

fn record_bitmap_write(handle: &BitmapHandle) -> Result<(), crate::gpu_runtime::RenderFailure> {
    if let Some(recipe) =
        (&*handle.0 as &dyn Any).downcast_ref::<crate::bitmap_recipe::BitmapRecipe>()
    {
        recipe.record_gpu_write()?;
    } else if let Some(cache) =
        (&*handle.0 as &dyn Any).downcast_ref::<crate::bitmap_cache::CacheImage>()
    {
        cache.written()?;
    } else if let Some(texture) = (&*handle.0 as &dyn Any).downcast_ref::<Texture>() {
        texture
            .version
            .try_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |v| v.checked_add(1),
            )
            .map_err(|_| {
                crate::gpu_runtime::RenderFailure::new(
                    crate::gpu_runtime::RenderFailureKind::Internal,
                    "texture content version exhausted",
                )
            })?;
    }
    Ok(())
}

pub fn raw_texture_as_texture(handle: &dyn RawTexture) -> &crate::gpu_resources::Texture {
    <dyn Any>::downcast_ref(handle).unwrap()
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Transforms {
    world_matrix: [[f32; 4]; 4],
    mult_color: [f32; 4],
    add_color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct PosVertex {
    position: [f32; 2],
}

impl From<TessVertex> for PosVertex {
    fn from(vertex: TessVertex) -> Self {
        Self {
            position: [vertex.x, vertex.y],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(crate) struct PosUvVertex {
    position: [f32; 2],
    uv: [f32; 3],
}

impl PosUvVertex {
    pub fn new(x: f32, y: f32, u: f32, v: f32, t: f32) -> Self {
        let position = [x, y];
        let uv = [u, v, t];
        Self { position, uv }
    }

    pub fn from_tessellator(vertex: TessVertex, texture_matrix: &[[f32; 3]; 3]) -> Self {
        let position = [vertex.x, vertex.y];
        let uv = Self::transform_uv(texture_matrix, vertex.x, vertex.y);
        Self { position, uv }
    }

    fn transform_uv(matrix: &[[f32; 3]; 3], x: f32, y: f32) -> [f32; 3] {
        [
            matrix[0][0] * x + matrix[1][0] * y + matrix[2][0],
            matrix[0][1] * x + matrix[1][1] * y + matrix[2][1],
            1.0,
        ]
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(crate) struct PosColorVertex {
    position: [f32; 2],
    color: [f32; 4],
}

impl From<TessVertex> for PosColorVertex {
    fn from(vertex: TessVertex) -> Self {
        Self {
            position: [vertex.x, vertex.y],
            color: [
                f32::from(vertex.color.r) / 255.0,
                f32::from(vertex.color.g) / 255.0,
                f32::from(vertex.color.b) / 255.0,
                f32::from(vertex.color.a) / 255.0,
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GradientUniforms {
    focal_point: f32,
    interpolation: i32,
    shape: i32,
    repeat: i32,
}

impl From<TessGradient> for GradientUniforms {
    fn from(gradient: TessGradient) -> Self {
        Self {
            focal_point: gradient.focal_point.to_f32().clamp(-0.98, 0.98),
            interpolation: (gradient.interpolation == swf::GradientInterpolation::LinearRgb) as i32,
            shape: match gradient.gradient_type {
                GradientType::Linear => 1,
                GradientType::Radial => 2,
                GradientType::Focal => 3,
            },
            repeat: match gradient.repeat_mode {
                GradientSpread::Pad => 1,
                GradientSpread::Reflect => 2,
                GradientSpread::Repeat => 3,
            },
        }
    }
}

pub struct AsyncReadback {
    buffer: PoolEntry<crate::gpu_resources::Buffer, BufferDimensions>,
    dimensions: BufferDimensions,
    descriptors: Arc<Descriptors>,
    receiver: Receiver<Result<(), wgpu::BufferAsyncError>>,
    started: std::time::Instant,
    mapping: bool,
}

impl AsyncReadback {
    fn new(
        buffer: PoolEntry<crate::gpu_resources::Buffer, BufferDimensions>,
        dimensions: BufferDimensions,
        descriptors: Arc<Descriptors>,
        index: Option<wgpu::SubmissionIndex>,
    ) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer
            .slice(..dimensions.size())
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        let _ = index;
        let _ = descriptors.device.poll(wgpu::PollType::Poll);
        Self {
            buffer,
            dimensions,
            descriptors,
            receiver,
            started: std::time::Instant::now(),
            mapping: true,
        }
    }

    pub fn poll(&mut self) -> Result<Option<(Vec<u8>, u32)>, crate::gpu_runtime::RenderFailure> {
        use crate::gpu_runtime::{RenderFailure, RenderFailureKind};
        self.descriptors.runtime.check_health()?;
        if !self.mapping {
            return Ok(None);
        }
        self.descriptors
            .device
            .poll(wgpu::PollType::Poll)
            .map_err(|error| {
                self.descriptors
                    .runtime
                    .health()
                    .record(crate::utils::poll_failure(error))
            })?;
        match self.receiver.try_recv() {
            Ok(Ok(())) => {
                let map = self
                    .buffer
                    .slice(..self.dimensions.size())
                    .get_mapped_range()
                    .map_err(|error| {
                        self.descriptors.runtime.health().record(RenderFailure::new(
                            RenderFailureKind::MapFailed,
                            error.to_string(),
                        ))
                    })?;
                let bytes = map.to_vec();
                drop(map);
                self.buffer.unmap();
                self.mapping = false;
                Ok(Some((bytes, self.dimensions.padded_bytes_per_row)))
            }
            Ok(Err(error)) => {
                self.mapping = false;
                Err(self.descriptors.runtime.health().record(RenderFailure::new(
                    RenderFailureKind::MapFailed,
                    error.to_string(),
                )))
            }
            Err(TryRecvError::Empty)
                if self.started.elapsed() < std::time::Duration::from_secs(2) =>
            {
                Ok(None)
            }
            Err(TryRecvError::Empty) => {
                Err(self.descriptors.runtime.health().record(RenderFailure::new(
                    RenderFailureKind::Timeout,
                    "asynchronous readback timed out",
                )))
            }
            Err(TryRecvError::Disconnected) => {
                self.mapping = false;
                Err(self.descriptors.runtime.health().record(RenderFailure::new(
                    RenderFailureKind::MapFailed,
                    "buffer map callback disconnected",
                )))
            }
        }
    }
}

impl Drop for AsyncReadback {
    fn drop(&mut self) {
        // Completed reads already unmapped the buffer. An unobserved failed map
        // is also unmapped; only cancel a pending map or release a successful one.
        if self.mapping
            && !matches!(
                self.receiver.try_recv(),
                Ok(Err(_)) | Err(TryRecvError::Disconnected)
            )
        {
            self.buffer.unmap();
        }
    }
}

#[derive(Debug)]
struct CopiedReadback {
    index: Option<wgpu::SubmissionIndex>,
    buffer: PoolEntry<crate::gpu_resources::Buffer, BufferDimensions>,
    dimensions: BufferDimensions,
}

#[derive(Debug)]
struct VersionedRead {
    dependency: render_snapshot::OperationDependency,
    source: render_snapshot::ResourceVersion,
    area: PixelRegion,
    saved: std::sync::Mutex<Option<Result<CopiedReadback, gpu_runtime::RenderFailure>>>,
}

impl VersionedRead {
    fn record(
        &self,
        handle: &BitmapHandle,
        descriptors: &Descriptors,
        pool: &BufferPool<gpu_resources::Buffer, BufferDimensions>,
        frame: &mut ActiveFrame,
    ) -> Result<CopiedReadback, gpu_runtime::RenderFailure> {
        self.dependency.check(&frame.operations)?;
        if render_snapshot::bitmap_version(handle)? != self.source {
            return Err(gpu_runtime::RenderFailure::new(
                gpu_runtime::RenderFailureKind::Validation,
                "bitmap readback version changed before its copy",
            ));
        }
        CopiedReadback::record_regions(
            handle,
            self.area,
            std::slice::from_ref(&self.area),
            descriptors,
            pool,
            frame,
        )
    }
}

impl CopiedReadback {
    fn record_regions(
        handle: &BitmapHandle,
        bounds: PixelRegion,
        parts: &[PixelRegion],
        descriptors: &Descriptors,
        pool: &BufferPool<gpu_resources::Buffer, BufferDimensions>,
        frame: &mut ActiveFrame,
    ) -> Result<Self, gpu_runtime::RenderFailure> {
        let texture = as_texture(handle, &descriptors.device)?;
        if bounds.is_empty()
            || bounds.x_max > texture.texture.width()
            || bounds.y_max > texture.texture.height()
            || parts.is_empty()
            || parts.iter().any(|r| {
                r.is_empty()
                    || r.x_min < bounds.x_min
                    || r.y_min < bounds.y_min
                    || r.x_max > bounds.x_max
                    || r.y_max > bounds.y_max
            })
        {
            return Err(gpu_runtime::RenderFailure::new(
                gpu_runtime::RenderFailureKind::Validation,
                "invalid bitmap readback regions",
            ));
        }
        let dimensions = BufferDimensions::new(
            bounds.width() as usize,
            bounds.height() as usize,
            texture.texture.format(),
        );
        let buffer = pool.take(descriptors, dimensions.clone())?;
        transfers::readback(
            descriptors,
            frame.encoder(),
            handle,
            bounds,
            parts,
            &buffer,
            &dimensions,
        )?;
        let index = Some(frame.submit_direct(descriptors)?);
        Ok(Self {
            index,
            buffer,
            dimensions,
        })
    }
}

#[derive(Debug)]
enum ReadbackSource {
    Copied(CopiedReadback),
    Deferred {
        handle: BitmapHandle,
        read: Arc<VersionedRead>,
        pool: Arc<BufferPool<gpu_resources::Buffer, BufferDimensions>>,
    },
}

#[derive(Debug)]
pub struct QueueSyncHandle {
    dependency: render_snapshot::OperationDependency,
    descriptors: Arc<Descriptors>,
    source: ReadbackSource,
}

impl SyncHandle for QueueSyncHandle {
    fn can_read_current_without_token(&self, bitmap: &BitmapHandle) -> bool {
        matches!(&self.source, ReadbackSource::Deferred { handle, .. } if handle == bitmap)
    }
}

impl QueueSyncHandle {
    fn deferred(
        handle: BitmapHandle,
        area: PixelRegion,
        descriptors: Arc<Descriptors>,
        pool: Arc<BufferPool<gpu_resources::Buffer, BufferDimensions>>,
        dependency: render_snapshot::OperationDependency,
        track: bool,
    ) -> Result<Self, gpu_runtime::RenderFailure> {
        let read = Arc::new(VersionedRead {
            dependency: dependency.clone(),
            source: render_snapshot::bitmap_version(&handle)?,
            area,
            saved: Default::default(),
        });
        // One sync token is issued per write. The next writer orders a copy
        // first only while somebody still holds this version's token.
        if track {
            *as_texture(&handle, &descriptors.device)?
                .pending_read
                .lock()
                .unwrap() = Arc::downgrade(&read);
        }
        Ok(Self {
            dependency,
            descriptors,
            source: ReadbackSource::Deferred { handle, read, pool },
        })
    }

    fn into_copy(
        self,
        frame: &mut ActiveFrame,
    ) -> Result<(Arc<Descriptors>, CopiedReadback), gpu_runtime::RenderFailure> {
        self.dependency.check(&frame.operations)?;
        let copy = match self.source {
            ReadbackSource::Copied(copy) => copy,
            ReadbackSource::Deferred { handle, read, pool } => {
                let saved = read.saved.lock().unwrap().take();
                match saved {
                    Some(result) => result?,
                    None => read.record(&handle, &self.descriptors, &pool, frame)?,
                }
            }
        };
        frame.operations.record()?;
        Ok((self.descriptors, copy))
    }

    pub fn prepare_async(
        self,
        frame: &mut ActiveFrame,
    ) -> Result<AsyncReadback, gpu_runtime::RenderFailure> {
        let (descriptors, copy) = self.into_copy(frame)?;
        Ok(AsyncReadback::new(
            copy.buffer,
            copy.dimensions,
            descriptors,
            copy.index,
        ))
    }

    pub fn capture_native<R, F: FnOnce(&[u8], u32, ruffle_render::bitmap::BitmapFormat) -> R>(
        self,
        read: F,
        frame: &mut ActiveFrame,
    ) -> Result<R, gpu_runtime::RenderFailure> {
        let _readback = ruffle_render::profile::time(ruffle_render::profile::Phase::Readback);
        let (descriptors, copy) = self.into_copy(frame)?;
        let format = if copy.dimensions.format == wgpu::TextureFormat::Bgra8Unorm {
            ruffle_render::bitmap::BitmapFormat::Bgra
        } else {
            ruffle_render::bitmap::BitmapFormat::Rgba
        };
        capture_image(
            &descriptors.device,
            &copy.buffer,
            &copy.dimensions,
            copy.index,
            |bytes, stride| read(bytes, stride, format),
        )
    }
}

#[derive(Debug)]
pub struct Texture {
    batch_source: crate::bitmap_batch::Source,
    pub(crate) _allocation: ruffle_render::profile::GpuBitmap,
    pub(crate) texture: crate::gpu_resources::Texture,
    view: crate::gpu_resources::TextureView,
    repeating_linear: OnceLock<BitmapBinds>,
    repeating_nearest: OnceLock<BitmapBinds>,
    clamped_linear: OnceLock<BitmapBinds>,
    clamped_nearest: OnceLock<BitmapBinds>,
    pending_read: std::sync::Mutex<std::sync::Weak<VersionedRead>>,
    version: std::sync::atomic::AtomicU64,
    _lifetime: Option<render_snapshot::ResourceLifetime>,
}

impl Texture {
    pub fn bind_group(
        &self,
        repeating: bool,
        smoothed: bool,
        device: &crate::gpu_resources::Device,
        layout: &crate::gpu_resources::BindGroupLayout,
        handle: BitmapHandle,
        samplers: &BitmapSamplers,
    ) -> Result<&BitmapBinds, crate::gpu_runtime::RenderFailure> {
        let bind = match (repeating, smoothed) {
            (true, true) => &self.repeating_linear,
            (true, false) => &self.repeating_nearest,
            (false, true) => &self.clamped_linear,
            (false, false) => &self.clamped_nearest,
        };
        crate::gpu_resources::try_sync_cell(bind, || {
            BitmapBinds::new(
                device,
                layout,
                samplers.get_sampler(repeating, smoothed),
                self.view.clone(),
                create_debug_label!("Bitmap {:?} bind group (smoothed: {})", handle.0, smoothed),
            )
        })
    }
}
