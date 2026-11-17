use crate::buffer_builder::BufferBuilder;
use crate::buffer_pool::{BufferPool, TexturePool};
use crate::context3d::WgpuContext3D;
use crate::filters::FilterSource;
use crate::mesh::{CommonGradient, Mesh, PendingDraw};
use crate::pixel_bender::{ShaderMode, run_pixelbender_shader_impl};
use crate::surface::buffers::ChunkBuffers;
use crate::surface::{LayerRef, Surface};
use crate::target::{MaybeOwnedBuffer, TextureTarget};
use crate::target::{RenderTargetFrame, TextureBufferInfo};
use crate::utils::{BufferDimensions, run_copy_pipeline};
use crate::{
    AsyncReadback, Descriptors, Error, QueueSyncHandle, RenderTarget, SwapChainTarget, Texture,
    as_texture, format_list, get_backend_names,
};
use ruffle_render::backend::{
    BitmapCacheEntry, Context3D, Context3DProfile, PixelBenderOutput, PixelBenderPending,
    PixelBenderTarget,
};
use ruffle_render::backend::{RenderBackend, ShapeHandle, ViewportDimensions};
use ruffle_render::bitmap::{
    Bitmap, BitmapHandle, BitmapSource, PixelRegion, RgbaBufRead, SyncHandle,
};
use ruffle_render::commands::CommandList;
use ruffle_render::error::Error as BitmapError;
use ruffle_render::filters::Filter;
use ruffle_render::pixel_bender::{PixelBenderShader, PixelBenderShaderHandle};
use ruffle_render::pixel_bender_support::PixelBenderShaderArgument;
use ruffle_render::quality::StageQuality;
use ruffle_render::shape_utils::DistilledShape;
use ruffle_render::tessellator::ShapeTessellator;
use std::any::Any;
use std::borrow::Cow;
use std::cell::Cell;
use std::num::NonZeroU32;
use std::sync::Arc;
use swf::Color;
use tracing::instrument;
use wgpu::SubmissionIndex;
use wgpu_profiler::{GpuProfiler, GpuProfilerSettings};

pub const DEFAULT_IMMUTABLE_BITMAP_CACHE_LIMIT: u64 = 256 * 1024 * 1024;

/// Creates a wgpu instance with Ruffle's required configuration.
///
/// This disables indirect call validation because wgpu's validation runs a compute
/// shader that uses `array<u32>`, which requires the `DYNAMIC_ARRAY_SIZE` feature.
/// However, wgpu runs this shader without first checking if the device supports
/// that feature, causing device creation to fail on GPUs that lack it.
/// Since Ruffle doesn't use indirect draws, disabling this validation has no
/// functional impact.
///
/// See <https://github.com/gfx-rs/wgpu/issues/8799>
pub fn create_wgpu_instance(
    backends: wgpu::Backends,
    backend_options: wgpu::BackendOptions,
    display: Option<Box<dyn wgpu::wgt::WgpuHasDisplayHandle>>,
) -> wgpu::Instance {
    let descriptor = match display {
        Some(display) => wgpu::InstanceDescriptor::new_with_display_handle(display),
        None => wgpu::InstanceDescriptor::new_without_display_handle(),
    };
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        flags: wgpu::InstanceFlags::default()
            .difference(wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL)
            .with_env(),
        backend_options,
        ..descriptor
    })
}

pub struct WgpuRenderBackend<T: RenderTarget> {
    pub(crate) descriptors: Arc<Descriptors>,
    budget: crate::idle_pool::Budget,
    target: T,
    surface: Surface,
    meshes: Vec<Mesh>,
    shape_tessellator: ShapeTessellator,
    // This is currently unused - we just store it to report in
    // `get_viewport_dimensions`
    viewport_scale_factor: f64,
    texture_pool: TexturePool,
    offscreen_texture_pool: TexturePool,
    pub(crate) offscreen_buffer_pool: Arc<BufferPool<wgpu::Buffer, BufferDimensions>>,
    chunk_buffers: ChunkBuffers,
    active_frame: ActiveFrame,
    profiler: GpuProfiler,
    last_submission: Option<SubmissionIndex>,
    bitmap_residency: Option<crate::bitmap_residency::BitmapResidency>,
}

impl WgpuRenderBackend<SwapChainTarget> {
    /// # Safety
    ///  See [`wgpu::SurfaceTargetUnsafe`] variants for safety requirements.
    ///
    /// Since wgpu 29, a display handle is needed at instance creation time:
    /// pass one via `display`, or make sure the `window` target carries a raw
    /// display handle (note that `SurfaceTargetUnsafe::from_window` does not
    /// provide one). Prefer passing `display` - some backends (e.g. GL via
    /// EGL) select their platform when the instance is created, before the
    /// target's display handle is seen.
    pub unsafe fn for_window_unsafe(
        window: wgpu::SurfaceTargetUnsafe,
        size: (u32, u32),
        backend: wgpu::Backends,
        power_preference: wgpu::PowerPreference,
        display: Option<Box<dyn wgpu::wgt::WgpuHasDisplayHandle>>,
    ) -> Result<Self, Error> {
        if wgpu::Backends::SECONDARY.contains(backend) {
            tracing::warn!(
                "{} graphics backend support may not be fully supported.",
                format_list(&get_backend_names(backend), "and")
            );
        }
        let instance = create_wgpu_instance(backend, wgpu::BackendOptions::default(), display);
        let surface = unsafe { instance.create_surface_unsafe(window)? };
        let (adapter, device, queue) = futures::executor::block_on(request_adapter_and_device(
            backend,
            &instance,
            Some(&surface),
            power_preference,
        ))?;
        let descriptors = Descriptors::new(instance, adapter, device, queue)?;
        let target =
            SwapChainTarget::new(surface, &descriptors.adapter, size, &descriptors.device)?;
        Self::new(Arc::new(descriptors), target)
    }

    /// # Safety
    ///  See [`wgpu::SurfaceTargetUnsafe`] variants for safety requirements.
    pub unsafe fn recreate_surface_unsafe(
        &mut self,
        window: wgpu::SurfaceTargetUnsafe,
        size: (u32, u32),
    ) -> Result<(), Error> {
        let descriptors = &self.descriptors;
        let surface = unsafe { descriptors.wgpu_instance.create_surface_unsafe(window)? };
        self.target =
            SwapChainTarget::new(surface, &descriptors.adapter, size, &descriptors.device)?;
        Ok(())
    }
}

impl WgpuRenderBackend<crate::target::TextureTarget> {
    pub fn for_offscreen(
        size: (u32, u32),
        backend: wgpu::Backends,
        power_preference: wgpu::PowerPreference,
    ) -> Result<Self, Error> {
        if wgpu::Backends::SECONDARY.contains(backend) {
            tracing::warn!(
                "{} graphics backend support may not be fully supported.",
                format_list(&get_backend_names(backend), "and")
            );
        }
        let instance = create_wgpu_instance(backend, wgpu::BackendOptions::default(), None);
        let (adapter, device, queue) = futures::executor::block_on(request_adapter_and_device(
            backend,
            &instance,
            None,
            power_preference,
        ))?;
        let descriptors = Descriptors::new(instance, adapter, device, queue)?;
        let target = crate::target::TextureTarget::new(&descriptors.device, size)?;
        Self::new(Arc::new(descriptors), target)
    }

    pub fn capture_frame(
        &self,
    ) -> Result<Option<image::RgbaImage>, crate::gpu_runtime::RenderFailure> {
        self.descriptors.runtime.check_health()?;
        use crate::utils::buffer_to_image;
        if let Some(buffer) = &self.target.buffer {
            let (buffer, dimensions) = buffer.buffer.inner();
            Ok(Some(
                buffer_to_image(
                    &self.descriptors.device,
                    buffer,
                    dimensions,
                    None,
                    self.target.size,
                )
                .map_err(|error| self.descriptors.runtime.health().record(error))?,
            ))
        } else {
            Ok(None)
        }
    }
}

impl<T: RenderTarget> WgpuRenderBackend<T> {
    pub fn new(descriptors: Arc<Descriptors>, target: T) -> Result<Self, Error> {
        Self::with_options(descriptors, target, false)
    }

    pub fn with_options(
        descriptors: Arc<Descriptors>,
        target: T,
        measure_gpu: bool,
    ) -> Result<Self, Error> {
        descriptors.runtime.check_health()?;
        if target.width() > descriptors.limits.max_texture_dimension_2d
            || target.height() > descriptors.limits.max_texture_dimension_2d
        {
            return Err(format!(
                "Render target texture cannot be larger than {}px on either dimension (requested {} x {})",
                descriptors.limits.max_texture_dimension_2d,
                target.width(),
                target.height()
            )
                .into());
        }

        let surface = Surface::new(
            &descriptors,
            StageQuality::Low,
            target.width(),
            target.height(),
            target.format(),
        );

        let budget = crate::idle_pool::Budget::new(Some(crate::idle_pool::IDLE_LIMIT));
        let offscreen_buffer_pool = BufferPool::new(
            budget.clone(),
            1,
            Box::new(BufferDimensions::size),
            Box::new(|descriptors: &Descriptors, dimensions: &BufferDimensions| {
                descriptors.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: dimensions.size(),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                })
            }),
        );

        let chunk_buffers = ChunkBuffers::new(&descriptors);
        let active_frame = ActiveFrame::new(&descriptors, budget.clone());

        let profiler_settings = GpuProfilerSettings {
            enable_timer_queries: cfg!(feature = "profile-with-tracy")
                || measure_gpu
                    && descriptors
                        .device
                        .features()
                        .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS),
            enable_debug_groups: cfg!(feature = "render_debug_labels"),
            ..Default::default()
        };
        #[cfg(feature = "profile-with-tracy")]
        let profiler = GpuProfiler::new_with_tracy_client(
            profiler_settings,
            descriptors.backend,
            &descriptors.device,
            &descriptors.queue,
        )?;
        #[cfg(not(feature = "profile-with-tracy"))]
        let profiler = GpuProfiler::new(&descriptors.device, profiler_settings)?;

        descriptors.runtime.check_health()?;
        Ok(Self {
            budget: budget.clone(),
            descriptors,
            target,
            surface,
            meshes: Vec::new(),
            shape_tessellator: ShapeTessellator::new(),
            viewport_scale_factor: 1.0,
            texture_pool: TexturePool::new(budget.clone()),
            offscreen_texture_pool: TexturePool::new(budget),
            offscreen_buffer_pool: Arc::new(offscreen_buffer_pool),
            chunk_buffers,
            active_frame,
            profiler,
            last_submission: None,
            bitmap_residency: Some(crate::bitmap_residency::BitmapResidency::new(
                DEFAULT_IMMUTABLE_BITMAP_CACHE_LIMIT,
            )),
        })
    }

    /// Limit retained, reconstructible symbol uploads. In-flight commands and
    /// writable bitmaps are separate owners and are not constrained by this cache.
    /// A zero or absent limit disables the cache for focused renderer tests.
    pub fn set_immutable_bitmap_cache_limit(&mut self, limit: Option<u64>) {
        self.bitmap_residency = limit
            .filter(|limit| *limit > 0)
            .map(crate::bitmap_residency::BitmapResidency::new);
    }

    fn register_shape_internal(
        &mut self,
        shape: DistilledShape,
        bitmap_source: &dyn BitmapSource,
        scale: f32,
    ) -> Mesh {
        let shape_id = shape.id;
        let lyon_mesh =
            self.shape_tessellator
                .tessellate_shape_with_scale(shape, bitmap_source, scale);

        let retainable = !lyon_mesh
            .draws
            .iter()
            .any(|d| matches!(d.draw_type, ruffle_render::tessellator::DrawType::Bitmap(_)));
        let gradient_count = lyon_mesh.gradients.len();
        let mut bounds: Option<[f64; 4]> = None;
        for vertex in lyon_mesh.draws.iter().flat_map(|draw| &draw.vertices) {
            let (x, y) = (f64::from(vertex.x), f64::from(vertex.y));
            let b = bounds.get_or_insert([x, y, x, y]);
            b[0] = b[0].min(x);
            b[1] = b[1].min(y);
            b[2] = b[2].max(x);
            b[3] = b[3].max(y);
        }
        let mut draws = Vec::with_capacity(lyon_mesh.draws.len());
        let mut uniform_buffer = BufferBuilder::new(
            &self.descriptors.limits,
            self.descriptors.limits.min_uniform_buffer_offset_alignment,
        );
        let mut vertex_buffer = BufferBuilder::new(&self.descriptors.limits, 0);
        let mut index_buffer = BufferBuilder::new(&self.descriptors.limits, 0);
        let mut indices_so_far = 0;
        let mut gradients = Vec::with_capacity(lyon_mesh.gradients.len());

        for gradient in lyon_mesh.gradients {
            gradients.push(CommonGradient::new(
                &self.descriptors,
                gradient,
                &mut uniform_buffer,
            ));
        }

        for draw in lyon_mesh.draws {
            let draw_id = draws.len();
            if let Some(draw) = PendingDraw::new(
                self,
                bitmap_source,
                draw,
                shape_id,
                draw_id,
                &mut vertex_buffer,
                &mut index_buffer,
                &mut indices_so_far,
            ) {
                draws.push(draw);
            }
        }

        let uniform_buffer = uniform_buffer.finish(
            &self.descriptors.device,
            create_debug_label!("Shape {} uniforms", shape_id),
            wgpu::BufferUsages::UNIFORM,
        );
        let vertex_buffer = vertex_buffer.finish(
            &self.descriptors.device,
            create_debug_label!("Shape {} vertices", shape_id),
            wgpu::BufferUsages::VERTEX,
        );
        let index_buffer = index_buffer.finish(
            &self.descriptors.device,
            create_debug_label!("Shape {} indices", shape_id),
            wgpu::BufferUsages::INDEX,
        );

        let draws: Vec<_> = draws
            .into_iter()
            .map(|d| d.finish(&self.descriptors, &uniform_buffer, &gradients))
            .collect();

        let allocation = ruffle_render::profile::GpuMesh::new(
            uniform_buffer.size() + vertex_buffer.size() + index_buffer.size(),
        );

        let retained_bytes = retainable.then_some(()).and_then(|()| {
            let buffers = uniform_buffer
                .size()
                .checked_add(vertex_buffer.size())?
                .checked_add(index_buffer.size())?;
            crate::mesh::retained_mesh_bytes(buffers, gradient_count, draws.capacity())
        });
        Mesh {
            retained_bytes,
            bounds,
            draws,
            vertex_buffer,
            index_buffer,
            _allocation: allocation,
        }
    }

    fn validate_bitmap_size(
        &self,
        bitmap: &Bitmap<'_>,
    ) -> Result<(), crate::gpu_runtime::RenderFailure> {
        let max_size = self.descriptors.limits.max_texture_dimension_2d;
        if bitmap.width() == 0
            || bitmap.height() == 0
            || bitmap.width() > max_size
            || bitmap.height() > max_size
        {
            return Err(self.descriptors.runtime.health().record(
                crate::gpu_runtime::RenderFailure::new(
                    crate::gpu_runtime::RenderFailureKind::Validation,
                    "bitmap dimensions exceed device limits",
                ),
            ));
        }
        Ok(())
    }

    /// Collect completed timestamp queries without blocking or inventing missing samples.
    pub fn collect_gpu_samples(&mut self) {
        let timestamp_period = self.descriptors.queue.get_timestamp_period();
        while let Some(queries) = self.profiler.process_finished_frame(timestamp_period) {
            let mut ranges = queries.iter().filter_map(|q| q.time.as_ref());
            if let Some(first) = ranges.next() {
                let (lo, hi) = ranges.fold((first.start, first.end), |(a, b), t| {
                    (a.min(t.start), b.max(t.end))
                });
                let frame = queries
                    .iter()
                    .find_map(|q| q.label.strip_prefix("Stage ")?.parse().ok());
                ruffle_render::profile::record_gpu(((hi - lo) * 1e9) as u64, frame);
            }
        }
    }
    pub fn descriptors(&self) -> &Arc<Descriptors> {
        &self.descriptors
    }

    pub fn last_submission(&self) -> Option<SubmissionIndex> {
        self.last_submission.clone()
    }

    pub fn target(&self) -> &T {
        &self.target
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.descriptors.device
    }

    pub fn make_queue_sync_handle(
        &self,
        target: TextureTarget,
        index: Option<SubmissionIndex>,
        destination: BitmapHandle,
        copy_area: PixelRegion,
    ) -> Box<QueueSyncHandle> {
        match target.take_buffer() {
            None => Box::new(QueueSyncHandle::NotCopied {
                handle: destination,
                copy_area,
                descriptors: self.descriptors.clone(),
                pool: self.offscreen_buffer_pool.clone(),
            }),
            Some(TextureBufferInfo {
                buffer: MaybeOwnedBuffer::Borrowed(buffer, copy_dimensions),
                ..
            }) => Box::new(QueueSyncHandle::AlreadyCopied {
                index,
                buffer,
                copy_dimensions,
                descriptors: self.descriptors.clone(),
            }),
            Some(TextureBufferInfo {
                buffer: MaybeOwnedBuffer::Owned(..),
                ..
            }) => unreachable!("Buffer must be Borrowed as it was set to be Borrowed earlier"),
        }
    }
}

enum WgpuPixelBenderPending {
    Bitmap(Option<Box<dyn SyncHandle>>),
    Bytes {
        sync_handle: Option<QueueSyncHandle>,
        readback: Option<AsyncReadback>,
        width: usize,
        output_channels: usize,
        has_padding: bool,
    },
}

impl<T: RenderTarget + 'static> RenderBackend for WgpuRenderBackend<T> {
    fn health(&self) -> &ruffle_render::error::RendererHealth {
        self.descriptors.runtime.health()
    }

    fn set_viewport_dimensions(
        &mut self,
        dimensions: ViewportDimensions,
    ) -> Result<(), crate::gpu_runtime::RenderFailure> {
        self.descriptors.runtime.check_health()?;
        if dimensions.width > self.descriptors.limits.max_texture_dimension_2d
            || dimensions.height > self.descriptors.limits.max_texture_dimension_2d
        {
            return Err(self.descriptors.runtime.health().record(
                crate::gpu_runtime::RenderFailure::new(
                    crate::gpu_runtime::RenderFailureKind::Validation,
                    "viewport exceeds device texture limits",
                ),
            ));
        }
        let width = dimensions.width.max(1);
        let height = dimensions.height.max(1);
        self.target
            .resize(&self.descriptors.device, width, height)?;

        self.surface = Surface::new(
            &self.descriptors,
            self.surface.quality(),
            width,
            height,
            self.target.format(),
        );

        self.viewport_scale_factor = dimensions.scale_factor;
        self.texture_pool = TexturePool::new(self.budget.clone());

        self.descriptors.runtime.check_health()?;
        Ok(())
    }

    fn create_context3d(
        &mut self,
        profile: Context3DProfile,
    ) -> Result<Box<dyn Context3D>, BitmapError> {
        self.descriptors.runtime.check_health()?;
        self.descriptors.runtime.check_health()?;
        Ok(Box::new(WgpuContext3D::new(
            self.descriptors.clone(),
            profile,
        )))
    }

    fn debug_info(&self) -> Cow<'static, str> {
        let mut result = vec![];
        result.push("Renderer: wgpu".to_string());

        let info = self.descriptors.adapter.get_info();
        result.push(format!("Adapter Backend: {:?}", info.backend));
        result.push(format!("Adapter Name: {:?}", info.name));
        result.push(format!("Adapter Device Type: {:?}", info.device_type));
        result.push(format!("Adapter Driver Name: {:?}", info.driver));
        result.push(format!("Adapter Driver Info: {:?}", info.driver_info));

        let enabled_features = self.descriptors.device.features();
        let available_features = self.descriptors.adapter.features() - enabled_features;
        let current_limits = &self.descriptors.limits;

        result.push(format!("Enabled features: {enabled_features:?}"));
        result.push(format!("Available features: {available_features:?}"));
        result.push(format!("Current limits: {current_limits:?}"));
        result.push(format!("Surface quality: {}", self.surface.quality()));
        result.push(format!("Surface samples: {}", self.surface.sample_count()));
        result.push(format!("Surface size: {:?}", self.surface.size()));

        Cow::Owned(result.join("\n"))
    }

    fn name(&self) -> &'static str {
        "wgpu"
    }

    fn set_quality(
        &mut self,
        quality: StageQuality,
    ) -> Result<(), crate::gpu_runtime::RenderFailure> {
        self.descriptors.runtime.check_health()?;
        self.surface = Surface::new(
            &self.descriptors,
            quality,
            self.surface.size().width,
            self.surface.size().height,
            self.target.format(),
        );

        self.descriptors.runtime.check_health()?;
        Ok(())
    }

    fn viewport_dimensions(&self) -> ViewportDimensions {
        ViewportDimensions {
            width: self.target.width(),
            height: self.target.height(),
            scale_factor: self.viewport_scale_factor,
        }
    }

    #[instrument(level = "debug", skip_all)]
    fn register_shape(
        &mut self,
        shape: DistilledShape,
        bitmap_source: &dyn BitmapSource,
    ) -> Result<ShapeHandle, crate::gpu_runtime::RenderFailure> {
        self.descriptors.runtime.check_health()?;
        let mesh = self.register_shape_internal(shape, bitmap_source, 1.0);
        self.descriptors.runtime.check_health()?;
        Ok(ShapeHandle(Arc::new(mesh)))
    }

    #[instrument(level = "debug", skip_all)]
    fn register_shape_with_scale(
        &mut self,
        shape: DistilledShape,
        bitmap_source: &dyn BitmapSource,
        scale: f32,
    ) -> Result<ShapeHandle, crate::gpu_runtime::RenderFailure> {
        self.descriptors.runtime.check_health()?;
        let mesh = self.register_shape_internal(shape, bitmap_source, scale);
        self.descriptors.runtime.check_health()?;
        Ok(ShapeHandle(Arc::new(mesh)))
    }

    #[instrument(level = "debug", skip_all)]
    fn submit_frame(
        &mut self,
        clear: Color,
        commands: CommandList,
        cache_entries: Vec<BitmapCacheEntry>,
    ) -> Result<(), crate::gpu_runtime::RenderFailure> {
        self.descriptors.runtime.check_health()?;
        let _render_backend =
            ruffle_render::profile::time(ruffle_render::profile::Phase::RenderBackend);
        let frame_output = {
            let _timer = ruffle_render::profile::native_time(
                ruffle_render::profile::NativePhase::SurfaceAcquire,
            );
            self.target.get_next_texture()?
        };
        let Some(frame_output) = frame_output else {
            // Attempt to recreate the swap chain in this case.
            self.target.resize(
                &self.descriptors.device,
                self.target.width(),
                self.target.height(),
            )?;
            return Ok(());
        };

        for entry in cache_entries {
            let texture = as_texture(&entry.handle);
            ruffle_render::profile::work(|w| {
                w.offscreen_targets += 1;
                w.offscreen_pixels +=
                    u64::from(texture.texture.width()) * u64::from(texture.texture.height());
            });
            let surface = Surface::new(
                &self.descriptors,
                self.surface.quality(),
                texture.texture.width(),
                texture.texture.height(),
                wgpu::TextureFormat::Rgba8Unorm,
            );
            if entry.filters.is_empty() {
                surface.draw_commands(
                    RenderTargetMode::ExistingWithColor(
                        texture.texture.clone(),
                        wgpu::Color {
                            r: f64::from(entry.clear.r) / 255.0,
                            g: f64::from(entry.clear.g) / 255.0,
                            b: f64::from(entry.clear.b) / 255.0,
                            a: f64::from(entry.clear.a) / 255.0,
                        },
                    ),
                    &self.descriptors,
                    &self.meshes,
                    entry.commands,
                    &mut self.active_frame.staging_belt,
                    &mut self.chunk_buffers,
                    &mut self
                        .profiler
                        .scope("Draw to CAB", self.active_frame.command_encoder.record()),
                    LayerRef::None,
                    &mut self.offscreen_texture_pool,
                );
            } else {
                let mut scope = self
                    .profiler
                    .scope("Filters", self.active_frame.command_encoder.record());
                // We're relying on there being no impotent filters here,
                // so that we can safely start by using the actual CAB texture.
                // It's guaranteed that at least one filter would have used it and moved the target to something else,
                // letting us safely copy back to it later.
                let mut target = surface.draw_commands(
                    RenderTargetMode::ExistingWithColor(
                        texture.texture.clone(),
                        wgpu::Color {
                            r: f64::from(entry.clear.r) / 255.0,
                            g: f64::from(entry.clear.g) / 255.0,
                            b: f64::from(entry.clear.b) / 255.0,
                            a: f64::from(entry.clear.a) / 255.0,
                        },
                    ),
                    &self.descriptors,
                    &self.meshes,
                    entry.commands,
                    &mut self.active_frame.staging_belt,
                    &mut self.chunk_buffers,
                    &mut scope.scope("Draw to CAB"),
                    LayerRef::None,
                    &mut self.offscreen_texture_pool,
                );
                for filter in entry.filters {
                    target = self.descriptors.filters.apply(
                        &self.descriptors,
                        &mut scope.scope(filter.name()),
                        &mut self.offscreen_texture_pool,
                        &mut self.active_frame.staging_belt,
                        FilterSource::for_entire_texture(target.color_texture()),
                        filter,
                    );
                }
                run_copy_pipeline(
                    &self.descriptors,
                    texture.texture.format(),
                    &texture.texture.create_view(&Default::default()),
                    target.color_view(),
                    target.color_texture().sample_count(),
                    &mut scope.scope("Copy filtered to CAB"),
                );
            }
            // Periodically flush GPU work to prevent OOM when many cache entries
            // accumulate (e.g. when a large container's cacheAsBitmap is skipped
            // but its hundreds of children each have their own bitmap caches).
            self.active_frame.maybe_flush(&self.descriptors)?;
        }

        self.surface.draw_commands_and_copy_to(
            frame_output.view(),
            RenderTargetMode::FreshWithColor(wgpu::Color {
                r: f64::from(clear.r) / 255.0,
                g: f64::from(clear.g) / 255.0,
                b: f64::from(clear.b) / 255.0,
                a: f64::from(clear.a) / 255.0,
            }),
            &self.descriptors,
            &mut self.active_frame.staging_belt,
            &mut self.chunk_buffers,
            &mut self.profiler.scope(
                ruffle_render::profile::next_frame()
                    .map(|n| format!("Stage {n}"))
                    .unwrap_or_else(|| "Frame commands".into()),
                self.active_frame.command_encoder.record(),
            ),
            &self.meshes,
            commands,
            LayerRef::None,
            &mut self.texture_pool,
        );
        self.profiler
            .resolve_queries(self.active_frame.command_encoder.record());
        self.active_frame.staging_belt.finish();

        self.last_submission = Some(self.active_frame.submit_for_target(
            &self.descriptors,
            &self.target,
            frame_output,
        )?);
        let (idle, peak, combined) = self.budget.stats();
        ruffle_render::profile::pool(idle, peak, combined);
        self.texture_pool.end_frame();
        self.offscreen_texture_pool.end_frame();
        self.profiler.end_frame().map_err(|error| {
            self.descriptors
                .runtime
                .health()
                .record(crate::gpu_runtime::RenderFailure::new(
                    crate::gpu_runtime::RenderFailureKind::Internal,
                    format!("GPU profiler: {error}"),
                ))
        })?;
        ruffle_render::profile::submitted();
        self.collect_gpu_samples();

        self.descriptors.runtime.check_health()?;
        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    fn register_bitmap(&mut self, bitmap: Bitmap<'_>) -> Result<BitmapHandle, BitmapError> {
        self.descriptors.runtime.check_health()?;
        let _upload = ruffle_render::profile::time(ruffle_render::profile::Phase::Upload);
        let bitmap = bitmap.to_rgba();

        self.validate_bitmap_size(&bitmap)?;

        let extent = wgpu::Extent3d {
            width: bitmap.width(),
            height: bitmap.height(),
            depth_or_array_layers: 1,
        };

        let texture_label = create_debug_label!("Bitmap");
        let texture = self
            .descriptors
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: texture_label.as_deref(),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC,
            });

        ruffle_render::profile::uploaded(u64::from(extent.width) * u64::from(extent.height) * 4);
        self.descriptors.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: Default::default(),
                aspect: wgpu::TextureAspect::All,
            },
            bitmap.data(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * extent.width),
                rows_per_image: None,
            },
            extent,
        );

        let handle = BitmapHandle(Arc::new(Texture {
            _allocation: ruffle_render::profile::GpuBitmap::new(
                u64::from(texture.width())
                    * u64::from(texture.height())
                    * u64::from(texture.format().block_copy_size(None).unwrap_or(4)),
            ),
            view: texture.create_view(&Default::default()),
            texture,
            repeating_linear: Default::default(),
            repeating_nearest: Default::default(),
            clamped_linear: Default::default(),
            clamped_nearest: Default::default(),
            copy_count: Cell::new(0),
        }));

        self.descriptors.runtime.check_health()?;
        Ok(handle)
    }

    fn cache_immutable_bitmap(&mut self, bitmap: &BitmapHandle) -> bool {
        let Some(cache) = &mut self.bitmap_residency else {
            return false;
        };
        let texture = as_texture(bitmap);
        let bytes = u64::from(texture.texture.width())
            * u64::from(texture.texture.height())
            * u64::from(texture.texture.format().block_copy_size(None).unwrap_or(4));
        cache.retain(bitmap, bytes);
        ruffle_render::profile::record_bitmap_residency(cache.stats());
        true
    }

    #[instrument(level = "debug", skip_all)]
    fn update_texture(
        &mut self,
        handle: &BitmapHandle,
        bitmap: Bitmap<'_>,
        region: PixelRegion,
    ) -> Result<(), BitmapError> {
        self.descriptors.runtime.check_health()?;
        if region.width() == 0 || region.height() == 0 {
            // Nothing to do. It's important to bail out now, as the
            // write_texture call panics when the source buffer is of zero size.
            return Ok(());
        }

        let texture = as_texture(handle);

        let _upload = ruffle_render::profile::time(ruffle_render::profile::Phase::Upload);
        let bitmap = bitmap.to_rgba();
        self.validate_bitmap_size(&bitmap)?;

        let extent = wgpu::Extent3d {
            width: region.width(),
            height: region.height(),
            depth_or_array_layers: 1,
        };

        if self.active_frame.command_encoder.dirty {
            self.active_frame.submit_direct(&self.descriptors)?;
        }
        ruffle_render::profile::uploaded(u64::from(extent.width) * u64::from(extent.height) * 4);
        self.descriptors.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: region.x_min,
                    y: region.y_min,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &bitmap.data()[(region.y_min * texture.texture.width() * 4) as usize
                ..(region.y_max * texture.texture.width() * 4) as usize],
            wgpu::TexelCopyBufferLayout {
                offset: (region.x_min * 4) as wgpu::BufferAddress,
                bytes_per_row: Some(4 * texture.texture.width()),
                rows_per_image: None,
            },
            extent,
        );

        self.descriptors.runtime.check_health()?;
        Ok(())
    }

    fn copy_bitmap_region(
        &mut self,
        source: BitmapHandle,
        source_region: PixelRegion,
        destination: BitmapHandle,
        destination_region: PixelRegion,
    ) -> Result<Option<Box<dyn SyncHandle>>, BitmapError> {
        self.descriptors.runtime.check_health()?;
        if source == destination
            || source_region.is_empty()
            || source_region.width() != destination_region.width()
            || source_region.height() != destination_region.height()
        {
            return Ok(None);
        }

        let source_texture = as_texture(&source);
        let destination_texture = as_texture(&destination);
        if source_region.x_max > source_texture.texture.width()
            || source_region.y_max > source_texture.texture.height()
            || destination_region.x_max > destination_texture.texture.width()
            || destination_region.y_max > destination_texture.texture.height()
        {
            return Ok(None);
        }

        let target = TextureTarget {
            size: wgpu::Extent3d {
                width: destination_texture.texture.width(),
                height: destination_texture.texture.height(),
                depth_or_array_layers: 1,
            },
            texture: destination_texture.texture.clone(),
            format: wgpu::TextureFormat::Rgba8Unorm,
            buffer: None,
        };

        self.active_frame
            .command_encoder
            .record()
            .copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &source_texture.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: source_region.x_min,
                        y: source_region.y_min,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &destination_texture.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: destination_region.x_min,
                        y: destination_region.y_min,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: source_region.width(),
                    height: source_region.height(),
                    depth_or_array_layers: 1,
                },
            );

        self.active_frame.maybe_flush(&self.descriptors)?;
        self.descriptors.runtime.check_health()?;
        Ok(Some(self.make_queue_sync_handle(
            target,
            None,
            destination,
            destination_region,
        )))
    }

    #[instrument(level = "debug", skip_all)]
    fn render_offscreen(
        &mut self,
        handle: BitmapHandle,
        commands: CommandList,
        quality: StageQuality,
        bounds: PixelRegion,
    ) -> Result<Option<Box<dyn SyncHandle>>, BitmapError> {
        self.descriptors.runtime.check_health()?;
        let texture = as_texture(&handle);

        let extent = wgpu::Extent3d {
            width: texture.texture.width(),
            height: texture.texture.height(),
            depth_or_array_layers: 1,
        };

        let mut target = TextureTarget {
            size: extent,
            texture: texture.texture.clone(),
            format: wgpu::TextureFormat::Rgba8Unorm,
            buffer: None,
        };

        let frame_output = target.get_next_texture()?.ok_or_else(|| {
            self.descriptors
                .runtime
                .health()
                .record(crate::gpu_runtime::RenderFailure::new(
                    crate::gpu_runtime::RenderFailureKind::Internal,
                    "texture target returned no attachment",
                ))
        })?;

        let surface = Surface::new(
            &self.descriptors,
            quality,
            texture.texture.width(),
            texture.texture.height(),
            wgpu::TextureFormat::Rgba8Unorm,
        );
        surface.draw_commands_and_copy_to(
            frame_output.view(),
            RenderTargetMode::FreshWithTexture(target.get_texture()),
            &self.descriptors,
            &mut self.active_frame.staging_belt,
            &mut self.chunk_buffers,
            &mut self.profiler.scope(
                "Offscreen commands",
                self.active_frame.command_encoder.record(),
            ),
            &self.meshes,
            commands,
            LayerRef::Current,
            &mut self.offscreen_texture_pool,
        );

        self.active_frame.maybe_flush(&self.descriptors)?;
        self.descriptors.runtime.check_health()?;
        Ok(Some(
            self.make_queue_sync_handle(target, None, handle, bounds),
        ))
    }

    fn is_filter_supported(&self, filter: &Filter) -> bool {
        matches!(
            filter,
            Filter::BlurFilter(_)
                | Filter::GlowFilter(_)
                | Filter::DropShadowFilter(_)
                | Filter::ColorMatrixFilter(_)
                | Filter::ShaderFilter(_)
                | Filter::BevelFilter(_)
                | Filter::DisplacementMapFilter(_)
        )
    }

    fn is_offscreen_supported(&self) -> bool {
        true
    }

    fn apply_filter(
        &mut self,
        source: BitmapHandle,
        source_point: (u32, u32),
        source_size: (u32, u32),
        destination: BitmapHandle,
        dest_point: (i32, i32),
        filter: Filter,
    ) -> Result<Option<Box<dyn SyncHandle>>, BitmapError> {
        self.descriptors.runtime.check_health()?;
        let source_texture = as_texture(&source);
        let dest_texture = as_texture(&destination);

        let copy_area = PixelRegion::for_whole_size(
            dest_texture.texture.width(),
            dest_texture.texture.height(),
        );

        let target = TextureTarget {
            size: wgpu::Extent3d {
                width: dest_texture.texture.width(),
                height: dest_texture.texture.height(),
                depth_or_array_layers: 1,
            },
            texture: dest_texture.texture.clone(),
            format: wgpu::TextureFormat::Rgba8Unorm,
            buffer: None,
        };

        let applied_filter = self.descriptors.filters.apply(
            &self.descriptors,
            self.active_frame.command_encoder.record(),
            &mut self.offscreen_texture_pool,
            &mut self.active_frame.staging_belt,
            FilterSource {
                texture: &source_texture.texture,
                point: source_point,
                size: source_size,
            },
            filter,
        );

        let (dest_x, dest_y) = dest_point;

        let src_offset_x = dest_x.min(0).unsigned_abs();
        let src_offset_y = dest_y.min(0).unsigned_abs();

        let final_dest_x = dest_x.max(0) as u32;
        let final_dest_y = dest_y.max(0) as u32;

        let available_width = applied_filter.width().saturating_sub(src_offset_x);
        let available_height = applied_filter.height().saturating_sub(src_offset_y);
        let dest_available_width = dest_texture.texture.width().saturating_sub(final_dest_x);
        let dest_available_height = dest_texture.texture.height().saturating_sub(final_dest_y);

        let copy_width = available_width.min(dest_available_width);
        let copy_height = available_height.min(dest_available_height);

        if copy_width == 0 || copy_height == 0 {
            return Ok(None);
        }

        self.active_frame
            .command_encoder
            .record()
            .copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: applied_filter.color_texture(),
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: src_offset_x,
                        y: src_offset_y,
                        z: 0,
                    },
                    aspect: Default::default(),
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &dest_texture.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: final_dest_x,
                        y: final_dest_y,
                        z: 0,
                    },
                    aspect: Default::default(),
                },
                wgpu::Extent3d {
                    width: copy_width,
                    height: copy_height,
                    depth_or_array_layers: 1,
                },
            );

        self.active_frame.maybe_flush(&self.descriptors)?;
        self.descriptors.runtime.check_health()?;
        Ok(Some(self.make_queue_sync_handle(
            target,
            None,
            destination,
            copy_area,
        )))
    }

    fn compile_pixelbender_shader(
        &mut self,
        shader: PixelBenderShader,
    ) -> Result<PixelBenderShaderHandle, BitmapError> {
        self.descriptors.runtime.check_health()?;
        self.compile_pixelbender_shader_impl(shader)
    }

    fn run_pixelbender_shader(
        &mut self,
        shader: PixelBenderShaderHandle,
        arguments: &[PixelBenderShaderArgument],
        target: &PixelBenderTarget,
    ) -> Result<PixelBenderOutput, BitmapError> {
        self.descriptors.runtime.check_health()?;
        let mut pending = self.submit_pixelbender_shader(shader, arguments, target)?;
        loop {
            if let Some(output) = self.poll_pixelbender_shader(&mut pending)? {
                return Ok(output);
            }
            self.descriptors
                .device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(2)),
                })
                .map_err(|error| {
                    self.descriptors
                        .runtime
                        .health()
                        .record(crate::utils::poll_failure(error))
                })?;
        }
    }

    fn poll_pixelbender_shader(
        &mut self,
        pending: &mut PixelBenderPending,
    ) -> Result<Option<PixelBenderOutput>, BitmapError> {
        self.descriptors.runtime.check_health()?;
        let state =
            pending
                .backend_mut::<WgpuPixelBenderPending>()
                .ok_or(BitmapError::Unimplemented(Cow::Borrowed(
                    "Invalid Pixel Bender pending state",
                )))?;
        match state {
            WgpuPixelBenderPending::Bitmap(handle) => {
                Ok(handle.take().map(PixelBenderOutput::Bitmap))
            }
            WgpuPixelBenderPending::Bytes {
                sync_handle,
                readback,
                width,
                output_channels,
                has_padding,
            } => {
                if readback.is_none() {
                    let handle = sync_handle
                        .take()
                        .expect("Pixel Bender readback handle missing");
                    *readback = Some(handle.prepare_async(&mut self.active_frame)?);
                    return Ok(None);
                }
                let Some((raw_pixels, buffer_width)) = readback.as_mut().unwrap().poll()? else {
                    return Ok(None);
                };
                let pixels = if buffer_width as usize
                    != *width * *output_channels as usize * std::mem::size_of::<f32>()
                {
                    let mut new_pixels = Vec::new();
                    for row in raw_pixels.chunks(buffer_width as usize) {
                        let actual_row = &row[0..(*width * std::mem::size_of::<[f32; 4]>())];
                        for pixel in actual_row
                            .as_chunks::<{ std::mem::size_of::<[f32; 4]>() }>()
                            .0
                        {
                            if *has_padding {
                                new_pixels
                                    .extend_from_slice(&pixel[0..(3 * std::mem::size_of::<f32>())]);
                            } else {
                                new_pixels.extend_from_slice(pixel);
                            }
                        }
                    }
                    new_pixels
                } else {
                    raw_pixels
                };
                Ok(Some(PixelBenderOutput::Bytes(pixels)))
            }
        }
    }

    fn submit_pixelbender_shader(
        &mut self,
        shader: PixelBenderShaderHandle,
        arguments: &[PixelBenderShaderArgument],
        target: &PixelBenderTarget,
    ) -> Result<PixelBenderPending, BitmapError> {
        self.descriptors.runtime.check_health()?;
        let output_channels = shader
            .0
            .parsed_shader()
            .output_channels()
            .expect("No output parameter");
        let has_padding = output_channels == 3;

        let texture_format =
            crate::pixel_bender::temporary_texture_format_for_channels(output_channels as u32);

        let target_handle = match target {
            PixelBenderTarget::Bitmap(handle) => handle.clone(),
            PixelBenderTarget::Bytes { width, height } => {
                let extent = wgpu::Extent3d {
                    width: *width,
                    height: *height,
                    depth_or_array_layers: 1,
                };
                // FIXME - cache this texture somehow. We might also want to consider using
                // a compute shader
                let texture_label = create_debug_label!("Temporary pixelbender output texture");
                let texture = self
                    .descriptors
                    .device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: texture_label.as_deref(),
                        size: extent,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: texture_format,
                        view_formats: &[texture_format],
                        usage: wgpu::TextureUsages::TEXTURE_BINDING
                            | wgpu::TextureUsages::COPY_DST
                            | wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::COPY_SRC,
                    });
                BitmapHandle(Arc::new(Texture {
                    _allocation: ruffle_render::profile::GpuBitmap::new(
                        u64::from(texture.width())
                            * u64::from(texture.height())
                            * u64::from(texture.format().block_copy_size(None).unwrap_or(4)),
                    ),
                    view: texture.create_view(&Default::default()),
                    texture,
                    repeating_linear: Default::default(),
                    repeating_nearest: Default::default(),
                    clamped_linear: Default::default(),
                    clamped_nearest: Default::default(),
                    copy_count: Cell::new(0),
                }))
            }
        };

        let target_texture = as_texture(&target_handle);

        let extent = wgpu::Extent3d {
            width: target_texture.texture.width(),
            height: target_texture.texture.height(),
            depth_or_array_layers: 1,
        };

        let copy_dimensions = BufferDimensions::new(
            target_texture.texture.width() as usize,
            target_texture.texture.height() as usize,
            target_texture.texture.format(),
        );
        let buffer_info = Some(TextureBufferInfo {
            buffer: MaybeOwnedBuffer::Borrowed(
                self.offscreen_buffer_pool
                    .take(&self.descriptors, copy_dimensions.clone()),
                copy_dimensions,
            ),
            copy_area: PixelRegion::for_whole_size(
                target_texture.texture.width(),
                target_texture.texture.height(),
            ),
        });

        let mut texture_target = TextureTarget {
            size: extent,
            texture: target_texture.texture.clone(),
            format: target_texture.texture.format(),
            buffer: buffer_info,
        };

        let frame_output = texture_target.get_next_texture()?.ok_or_else(|| {
            self.descriptors
                .runtime
                .health()
                .record(crate::gpu_runtime::RenderFailure::new(
                    crate::gpu_runtime::RenderFailureKind::Internal,
                    "texture target returned no attachment",
                ))
        })?;

        run_pixelbender_shader_impl(
            &self.descriptors,
            shader,
            ShaderMode::ShaderJob,
            arguments,
            &target_texture.texture,
            self.active_frame.command_encoder.record(),
            Some(wgpu::RenderPassColorAttachment {
                view: frame_output.view(),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            }),
            1,
            // When running a standalone shader, we always process the entire image
            &FilterSource::for_entire_texture(&target_texture.texture),
        )?;

        let index = Some(self.active_frame.submit_for_target(
            &self.descriptors,
            &texture_target,
            frame_output,
        )?);

        let sync_handle = self.make_queue_sync_handle(
            texture_target,
            index,
            target_handle,
            PixelRegion::for_whole_size(extent.width, extent.height),
        );

        match target {
            PixelBenderTarget::Bitmap(_) => Ok(PixelBenderPending::from_backend(
                WgpuPixelBenderPending::Bitmap(Some(sync_handle)),
            )),
            PixelBenderTarget::Bytes { width, .. } => Ok(PixelBenderPending::from_backend(
                WgpuPixelBenderPending::Bytes {
                    sync_handle: Some(*sync_handle),
                    readback: None,
                    width: *width as usize,
                    output_channels,
                    has_padding,
                },
            )),
        }
    }

    fn create_empty_texture(
        &mut self,
        width: NonZeroU32,
        height: NonZeroU32,
    ) -> Result<BitmapHandle, BitmapError> {
        self.descriptors.runtime.check_health()?;
        let width = width.get();
        let height = height.get();

        if width > self.descriptors.limits.max_texture_dimension_2d
            || height > self.descriptors.limits.max_texture_dimension_2d
        {
            return Err(self
                .descriptors
                .runtime
                .health()
                .record(crate::gpu_runtime::RenderFailure::new(
                    crate::gpu_runtime::RenderFailureKind::Validation,
                    "bitmap exceeds device texture limits",
                ))
                .into());
        }

        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture_label = create_debug_label!("Bitmap");
        let texture = self
            .descriptors
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: texture_label.as_deref(),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC,
            });
        self.descriptors.runtime.check_health()?;
        Ok(BitmapHandle(Arc::new(Texture {
            _allocation: ruffle_render::profile::GpuBitmap::new(
                u64::from(texture.width())
                    * u64::from(texture.height())
                    * u64::from(texture.format().block_copy_size(None).unwrap_or(4)),
            ),
            view: texture.create_view(&Default::default()),
            texture,
            repeating_linear: Default::default(),
            repeating_nearest: Default::default(),
            clamped_linear: Default::default(),
            clamped_nearest: Default::default(),
            copy_count: Cell::new(0),
        })))
    }

    fn resolve_sync_handle(
        &mut self,
        handle: Box<dyn SyncHandle>,
        with_rgba: RgbaBufRead,
    ) -> Result<(), ruffle_render::error::Error> {
        self.descriptors.runtime.check_health()?;
        let handle = Box::<dyn Any>::downcast::<QueueSyncHandle>(handle).map_err(|_| {
            self.descriptors
                .runtime
                .health()
                .record(crate::gpu_runtime::RenderFailure::new(
                    crate::gpu_runtime::RenderFailureKind::Internal,
                    "foreign readback handle",
                ))
        })?;
        handle
            .capture(with_rgba, &mut self.active_frame)
            .map_err(|error| self.descriptors.runtime.health().record(error))?;
        self.descriptors.runtime.check_health()?;
        Ok(())
    }
}

pub async fn request_adapter_and_device(
    backend: wgpu::Backends,
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface<'static>>,
    power_preference: wgpu::PowerPreference,
) -> Result<(wgpu::Adapter, wgpu::Device, wgpu::Queue), Error> {
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference,
        compatible_surface: surface,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }).await
        .map_err(|_e| {
            let names = get_backend_names(backend);
            if names.is_empty() {
                "Ruffle requires hardware acceleration, but no compatible graphics device was found (no backend provided?)".to_string()
            } else if cfg!(target_vendor = "apple") {
                "Ruffle does not support OpenGL on macOS/iOS.".to_string()
            } else {
                format!("Ruffle requires hardware acceleration, but no compatible graphics device was found supporting {}", format_list(&names, "or"))
            }
        })?;

    let (device, queue) = request_device(&adapter).await?;
    Ok((adapter, device, queue))
}

// We try to request the highest limits we can get away with
async fn request_device(
    adapter: &wgpu::Adapter,
) -> Result<(wgpu::Device, wgpu::Queue), wgpu::RequestDeviceError> {
    // We start off with the lowest limits we actually need - basically GL-ES 3.0
    let mut limits = wgpu::Limits::downlevel_webgl2_defaults();
    // Then we increase parts of it to the maximum supported by the adapter, to take advantage of
    // more powerful hardware or capabilities
    limits = limits.using_resolution(adapter.limits());
    limits = limits.using_alignment(adapter.limits());
    limits.max_uniform_buffer_binding_size = adapter.limits().max_uniform_buffer_binding_size;
    limits.max_inter_stage_shader_variables = adapter.limits().max_inter_stage_shader_variables;
    // This will be a default limit in a future wgpu version (down from 8).
    // Keep the GLES-compatible baseline for native GL adapters.
    limits.max_color_attachments = 4;

    let mut features = Default::default();

    let optional_features = wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
        | wgpu::Features::TEXTURE_COMPRESSION_BC
        | wgpu::Features::FLOAT32_FILTERABLE
        | GpuProfiler::ALL_WGPU_TIMER_FEATURES;

    features |= optional_features & adapter.features();

    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: features,
            required_limits: limits,
            memory_hints: Default::default(),
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        })
        .await
}

/// Determines how we choose our frame buffer
#[derive(Clone)]
pub enum RenderTargetMode {
    // Construct a new frame buffer, clearng it with the provided color.
    // This is used when rendering to the actual display,
    // or when applying a filter. In both cases, we have a fixed background color,
    // and don't need to blend with anything else
    FreshWithColor(wgpu::Color),
    // Construct a new frame buffer, cleared with an existing texture.
    // we will blend with the previous contents of the texture.
    // This is used in `render_offscreen`, as we need to blend with the previous
    // contents of our `BitmapData` texture
    FreshWithTexture(wgpu::Texture),
    // Use the provided texture as our frame buffer, and clear it with the given color.
    ExistingWithColor(wgpu::Texture, wgpu::Color),
}

impl RenderTargetMode {
    pub fn color(&self) -> Option<wgpu::Color> {
        match self {
            RenderTargetMode::FreshWithColor(color) => Some(*color),
            RenderTargetMode::FreshWithTexture(_) => None,
            RenderTargetMode::ExistingWithColor(_, color) => Some(*color),
        }
    }
}

struct TrackedEncoder {
    raw: wgpu::CommandEncoder,
    dirty: bool,
}
impl TrackedEncoder {
    fn record(&mut self) -> &mut wgpu::CommandEncoder {
        self.dirty = true;
        &mut self.raw
    }
}
pub struct ActiveFrame {
    pub staging_belt: wgpu::util::StagingBelt,
    command_encoder: TrackedEncoder,
    budget: crate::idle_pool::Budget,

    draws_since_flush: u32,
}

impl ActiveFrame {
    const MAX_DRAWS_PER_FLUSH: u32 = 100;

    pub fn new(descriptors: &Descriptors, budget: crate::idle_pool::Budget) -> Self {
        Self {
            budget,
            command_encoder: TrackedEncoder {
                dirty: false,
                raw: descriptors
                    .device
                    .create_command_encoder(&Default::default()),
            },
            staging_belt: wgpu::util::StagingBelt::new(descriptors.device.clone(), 65536),
            draws_since_flush: 0,
        }
    }

    pub(crate) fn encoder(&mut self) -> &mut wgpu::CommandEncoder {
        self.command_encoder.record()
    }
    pub fn submit_for_target<T: RenderTarget>(
        &mut self,
        descriptors: &Descriptors,
        target: &T,
        frame: T::Frame,
    ) -> Result<SubmissionIndex, crate::gpu_runtime::RenderFailure> {
        descriptors.runtime.check_health()?;
        self.draws_since_flush = 0;
        self.staging_belt.finish();
        let draw_encoder = std::mem::replace(
            &mut self.command_encoder.raw,
            descriptors
                .device
                .create_command_encoder(&Default::default()),
        );
        let index = {
            let _submit = ruffle_render::profile::time(ruffle_render::profile::Phase::RenderSubmit);
            let command_buffer = {
                let _timer = ruffle_render::profile::native_time(
                    ruffle_render::profile::NativePhase::EncoderFinish,
                );
                draw_encoder.finish()
            };
            let _submission = {
                let _timer = ruffle_render::profile::native_time(
                    ruffle_render::profile::NativePhase::SubmitLock,
                );
                descriptors.lock_submission()
            };
            target.submit(
                &descriptors.device,
                &descriptors.queue,
                descriptors.runtime.health(),
                Some(command_buffer),
                frame,
            )?
        };
        self.staging_belt.recall();
        self.command_encoder.dirty = false;
        self.budget.submitted(&descriptors.queue);
        descriptors.runtime.check_health()?;
        Ok(index)
    }

    pub fn submit_direct(
        &mut self,
        descriptors: &Descriptors,
    ) -> Result<SubmissionIndex, crate::gpu_runtime::RenderFailure> {
        descriptors.runtime.check_health()?;
        self.draws_since_flush = 0;
        self.staging_belt.finish();
        let draw_encoder = std::mem::replace(
            &mut self.command_encoder.raw,
            descriptors
                .device
                .create_command_encoder(&Default::default()),
        );
        let index = {
            let _submit = ruffle_render::profile::time(ruffle_render::profile::Phase::RenderSubmit);
            let command_buffer = {
                let _timer = ruffle_render::profile::native_time(
                    ruffle_render::profile::NativePhase::EncoderFinish,
                );
                draw_encoder.finish()
            };
            let _submission = {
                let _timer = ruffle_render::profile::native_time(
                    ruffle_render::profile::NativePhase::SubmitLock,
                );
                descriptors.lock_submission()
            };
            let _timer = ruffle_render::profile::native_time(
                ruffle_render::profile::NativePhase::QueueSubmit,
            );
            descriptors.queue.submit(Some(command_buffer))
        };
        self.staging_belt.recall();
        self.command_encoder.dirty = false;
        self.budget.submitted(&descriptors.queue);
        descriptors.runtime.check_health()?;
        Ok(index)
    }

    pub fn maybe_flush(
        &mut self,
        descriptors: &Descriptors,
    ) -> Result<(), crate::gpu_runtime::RenderFailure> {
        descriptors.runtime.check_health()?;
        // [NA] This is kind of a hack.
        // If we do "too much" during one frame, the submission ends up being way too large and goes OutOfMemory.
        // What it is that we're OOMing on is likely buffers and temporary textures and such from render_offscreen
        // Hard to track that though... so let's just flush it out if we do more than X draws per frame
        self.draws_since_flush += 1;

        if self.draws_since_flush > Self::MAX_DRAWS_PER_FLUSH {
            self.submit_direct(descriptors)?;
        }

        descriptors.runtime.check_health()?;
        Ok(())
    }
}
