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

mod bitmap_recipe;
mod bitmap_residency;
mod bitmaps;
mod context3d;
mod globals;
pub mod gpu_profiler;
pub mod gpu_resources;
pub mod gpu_runtime;
mod graph_encoder;
mod pass_graph;
mod pipelines;
mod pixel_bender;
mod render_snapshot;
mod resource_scheduler;
mod shape_recipe;
pub mod target;
pub use ruffle_render::profile;
mod idle_pool;

pub mod backend;
mod blend;
mod buffer_builder;
mod buffer_pool;
#[cfg(feature = "clap")]
pub mod clap;
pub mod descriptors;
mod filters;
mod layouts;
mod mesh;
mod material_cache;
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
struct PosUvVertex {
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
struct PosColorVertex {
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
            .slice(..)
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
        }
    }

    pub fn poll(&mut self) -> Result<Option<(Vec<u8>, u32)>, crate::gpu_runtime::RenderFailure> {
        use crate::gpu_runtime::{RenderFailure, RenderFailureKind};
        self.descriptors.runtime.check_health()?;
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
                let map = self.buffer.slice(..).get_mapped_range().map_err(|error| {
                    self.descriptors.runtime.health().record(RenderFailure::new(
                        RenderFailureKind::MapFailed,
                        error.to_string(),
                    ))
                })?;
                let bytes = map.to_vec();
                drop(map);
                self.buffer.unmap();
                Ok(Some((bytes, self.dimensions.padded_bytes_per_row)))
            }
            Ok(Err(error)) => Err(self.descriptors.runtime.health().record(RenderFailure::new(
                RenderFailureKind::MapFailed,
                error.to_string(),
            ))),
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
        self.buffer.unmap();
    }
}

#[derive(Debug)]
pub enum QueueSyncHandle {
    AlreadyCopied {
        index: Option<wgpu::SubmissionIndex>,
        buffer: PoolEntry<crate::gpu_resources::Buffer, BufferDimensions>,
        copy_dimensions: BufferDimensions,
        descriptors: Arc<Descriptors>,
    },
    NotCopied {
        handle: BitmapHandle,
        copy_area: PixelRegion,
        descriptors: Arc<Descriptors>,
        pool: Arc<BufferPool<crate::gpu_resources::Buffer, BufferDimensions>>,
    },
}

impl SyncHandle for QueueSyncHandle {}

impl QueueSyncHandle {
    pub fn prepare_async(
        self,
        frame: &mut ActiveFrame,
    ) -> Result<AsyncReadback, crate::gpu_runtime::RenderFailure> {
        Ok(match self {
            QueueSyncHandle::AlreadyCopied {
                index,
                buffer,
                copy_dimensions,
                descriptors,
            } => AsyncReadback::new(buffer, copy_dimensions, descriptors, index),
            QueueSyncHandle::NotCopied {
                handle,
                copy_area,
                descriptors,
                pool,
            } => {
                let texture = as_texture(&handle, &descriptors.device)?;
                let dimensions = BufferDimensions::new(
                    copy_area.width() as usize,
                    copy_area.height() as usize,
                    texture.texture.format(),
                );
                let buffer = pool.take(&descriptors, dimensions.clone())?;
                frame.encoder().copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: copy_area.x_min,
                            y: copy_area.y_min,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: &buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(dimensions.padded_bytes_per_row),
                            rows_per_image: None,
                        },
                    },
                    wgpu::Extent3d {
                        width: copy_area.width(),
                        height: copy_area.height(),
                        depth_or_array_layers: 1,
                    },
                );
                let index = frame.submit_direct(&descriptors)?;
                AsyncReadback::new(buffer, dimensions, descriptors, Some(index))
            }
        })
    }

    pub fn capture<R, F: FnOnce(&[u8], u32) -> R>(
        self,
        with_rgba: F,
        frame: &mut ActiveFrame,
    ) -> Result<R, crate::gpu_runtime::RenderFailure> {
        let _readback = ruffle_render::profile::time(ruffle_render::profile::Phase::Readback);
        match self {
            QueueSyncHandle::AlreadyCopied {
                index,
                buffer,
                copy_dimensions,
                descriptors,
            } => capture_image(
                &descriptors.device,
                &buffer,
                &copy_dimensions,
                index,
                with_rgba,
            ),
            QueueSyncHandle::NotCopied {
                handle,
                copy_area,
                descriptors,
                pool,
            } => {
                let texture = as_texture(&handle, &descriptors.device)?;

                let buffer_dimensions = BufferDimensions::new(
                    copy_area.width() as usize,
                    copy_area.height() as usize,
                    texture.texture.format(),
                );

                let buffer = pool.take(&descriptors, buffer_dimensions.clone())?;
                frame.encoder().copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: copy_area.x_min,
                            y: copy_area.y_min,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: &buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(buffer_dimensions.padded_bytes_per_row),
                            rows_per_image: None,
                        },
                    },
                    wgpu::Extent3d {
                        width: copy_area.width(),
                        height: copy_area.height(),
                        depth_or_array_layers: 1,
                    },
                );
                let index = frame.submit_direct(&descriptors)?;

                let image = capture_image(
                    &descriptors.device,
                    &buffer,
                    &buffer_dimensions,
                    Some(index),
                    with_rgba,
                );

                // After we've read pixels from a texture enough times, we'll store this buffer so that
                // future reads will be faster (it'll copy as part of the draw process instead)
                let _ = texture.copy_count.try_update(
                    std::sync::atomic::Ordering::Relaxed,
                    std::sync::atomic::Ordering::Relaxed,
                    |count| Some(count.saturating_add(1)),
                );

                image
            }
        }
    }
}

#[derive(Debug)]
pub struct Texture {
    pub(crate) _allocation: ruffle_render::profile::GpuBitmap,
    pub(crate) texture: crate::gpu_resources::Texture,
    view: crate::gpu_resources::TextureView,
    repeating_linear: OnceLock<BitmapBinds>,
    repeating_nearest: OnceLock<BitmapBinds>,
    clamped_linear: OnceLock<BitmapBinds>,
    clamped_nearest: OnceLock<BitmapBinds>,
    copy_count: std::sync::atomic::AtomicU8,
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
