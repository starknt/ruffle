use crate::buffer_builder::BufferBuilder;
use crate::buffer_pool::{BufferPool, TexturePool};
use crate::context3d::WgpuContext3D;
use crate::filters::FilterSource;
use crate::gpu_profiler::{GpuProfiler, GpuProfilerSettings};
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
    Bitmap, BitmapBufRead, BitmapFormat, BitmapHandle, BitmapSource, PixelRegion, RgbaBufRead,
    SyncHandle,
};
use ruffle_render::commands::CommandList;
use ruffle_render::error::Error as BitmapError;
use ruffle_render::filters::Filter;
use ruffle_render::pixel_bender::{PixelBenderShader, PixelBenderShaderHandle};
use ruffle_render::pixel_bender_support::PixelBenderShaderArgument;
use ruffle_render::quality::StageQuality;
use ruffle_render::shape_utils::DistilledShape;
use std::any::Any;
use std::borrow::Cow;
use std::num::NonZeroU32;
use std::sync::Arc;
use swf::Color;
use tracing::instrument;
use wgpu::SubmissionIndex;

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
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds {
            for_resource_creation: Some(90),
            for_device_loss: None,
        },
        ..descriptor
    })
}

pub struct WgpuRenderBackend<T: RenderTarget> {
    pub(crate) descriptors: Arc<Descriptors>,
    budget: crate::idle_pool::Budget,
    target: T,
    surface: Surface,
    meshes: Vec<Mesh>,
    small_meshes: crate::mesh::SmallMeshCache,
    pub(crate) preparation_generation: crate::prepare::PreparationGeneration,
    // This is currently unused - we just store it to report in
    // `get_viewport_dimensions`
    viewport_scale_factor: f64,
    texture_pool: TexturePool,
    offscreen_texture_pool: TexturePool,
    pub(crate) offscreen_buffer_pool:
        Arc<BufferPool<crate::gpu_resources::Buffer, BufferDimensions>>,
    chunk_buffers: ChunkBuffers,
    active_frame: ActiveFrame,
    profiler: GpuProfiler,
    last_submission: Option<SubmissionIndex>,
    bitmap_residency:
        Option<Arc<crate::bitmap_residency::BitmapResidency<crate::bitmap_recipe::BitmapRecipe>>>,
}

impl<T: RenderTarget> Drop for WgpuRenderBackend<T> {
    fn drop(&mut self) {
        self.preparation_generation.cancel();
        // Descriptors may outlive a movie/renderer in a shared-device host.
        // Cached bindings must not keep that renderer's targets alive there.
        self.descriptors.materials.clear();
    }
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
        )?;

        let budget = crate::idle_pool::Budget::with_runtime(&descriptors.runtime);
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

        let chunk_buffers = ChunkBuffers::new(&descriptors)?;
        let active_frame = ActiveFrame::new(&descriptors, budget.clone())?;

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

        let bitmap_residency = crate::bitmap_residency::BitmapResidency::with_runtime(
            &descriptors.runtime,
            descriptors.runtime.config().cache_limit,
        );
        descriptors.runtime.check_health()?;
        Ok(Self {
            budget: budget.clone(),
            descriptors,
            target,
            surface,
            meshes: Vec::new(),
            small_meshes: Default::default(),
            preparation_generation: crate::prepare::PreparationGeneration::default(),
            viewport_scale_factor: 1.0,
            texture_pool: TexturePool::new(budget.clone()),
            offscreen_texture_pool: TexturePool::new(budget),
            offscreen_buffer_pool: Arc::new(offscreen_buffer_pool),
            chunk_buffers,
            active_frame,
            profiler,
            last_submission: None,
            bitmap_residency: Some(Arc::new(bitmap_residency)),
        })
    }

    /// Limit retained, reconstructible symbol uploads. In-flight commands and
    /// writable bitmaps are separate owners and are not constrained by this cache.
    /// A zero or absent limit disables the cache for focused renderer tests.
    pub fn set_immutable_bitmap_cache_limit(&mut self, limit: Option<u64>) {
        self.bitmap_residency = limit.filter(|limit| *limit > 0).map(|limit| {
            Arc::new(crate::bitmap_residency::BitmapResidency::with_runtime(
                &self.descriptors.runtime,
                limit,
            ))
        });
    }

    pub(crate) fn register_prepared_shape(
        &mut self,
        prepared: crate::prepare::mesh::PreparedMesh,
        shape_id: swf::CharacterId,
        bitmap_source: &dyn BitmapSource,
        owner: crate::gpu_runtime::ResourceOwner,
    ) -> Result<Arc<Mesh>, crate::gpu_runtime::RenderFailure> {
        let key = crate::mesh::SmallMeshCache::key(&prepared);
        if let Some(key) = &key
            && let Some(mesh) = self.small_meshes.get(key)
        {
            self.descriptors.runtime.check_health()?;
            return Ok(mesh);
        }
        let device = self.descriptors.device.with_owner(owner);
        let retainable = !prepared
            .draws
            .iter()
            .any(|d| matches!(d.draw_type, ruffle_render::tessellator::DrawType::Bitmap(_)));
        let gradient_count = prepared.gradients.len();
        let bounds = prepared.bounds;
        let mut draws = Vec::with_capacity(prepared.draws.len());
        let mut uniform_buffer = BufferBuilder::new(
            &self.descriptors.limits,
            self.descriptors.limits.min_uniform_buffer_offset_alignment,
        );
        let mut vertex_buffer = BufferBuilder::new(&self.descriptors.limits, 0);
        let mut index_buffer = BufferBuilder::new(&self.descriptors.limits, 0);
        let mut indices_so_far = 0;
        let mut gradients = Vec::with_capacity(prepared.gradients.len());

        for gradient in prepared.gradients {
            gradients.push(CommonGradient::new(
                &self.descriptors,
                &device,
                gradient,
                &mut uniform_buffer,
            )?);
        }

        for draw in prepared.draws {
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
            )? {
                draws.push(draw);
            }
        }

        // Immutable sections share one allocation and lease. Uniform offsets
        // keep their original alignment; draw ranges select the vertex/index
        // sections. Avoid two extra scoped creations for every small shape.
        let packed_indices = uniform_buffer.len().checked_add(vertex_buffer.len());
        let packed_size = packed_indices.and_then(|start| start.checked_add(index_buffer.len()));
        let packed = packed_indices
            .zip(packed_size)
            .filter(|(index_start, bytes)| {
                uniform_buffer.len().is_multiple_of(4)
                    && index_start.is_multiple_of(4)
                    && *bytes as u64 <= self.descriptors.limits.max_buffer_size
                    && u32::try_from(*index_start / 4)
                        .ok()
                        .and_then(|base| base.checked_add(indices_so_far))
                        .is_some()
            });
        let (uniform_buffer, vertex_buffer, index_buffer, storage_bytes) =
            if let Some((index_start, _)) = packed {
                let vertex_start = uniform_buffer.len() as u64;
                let index_start = (index_start / 4) as u32;
                let mut packed = uniform_buffer;
                let invalid = |_| {
                    crate::gpu_runtime::RenderFailure::new(
                        crate::gpu_runtime::RenderFailureKind::Internal,
                        "checked immutable mesh sections no longer fit",
                    )
                };
                packed.append_section(&vertex_buffer).map_err(invalid)?;
                packed.append_section(&index_buffer).map_err(invalid)?;
                for draw in &mut draws {
                    draw.vertices.start += vertex_start;
                    draw.vertices.end += vertex_start;
                    draw.index_offset += index_start;
                }
                let buffer = packed.finish(
                    &device,
                    create_debug_label!("Shape {} storage", shape_id),
                    wgpu::BufferUsages::UNIFORM
                        | wgpu::BufferUsages::VERTEX
                        | wgpu::BufferUsages::INDEX,
                )?;
                let bytes = buffer.size();
                (buffer.clone(), buffer.clone(), buffer, bytes)
            } else {
                // A legal mesh can exceed the device's single-buffer limit in
                // aggregate. Choose separate storage before any allocation or
                // writes; allocation failures themselves are never retried here.
                let uniforms = uniform_buffer.finish(
                    &device,
                    create_debug_label!("Shape {} uniforms", shape_id),
                    wgpu::BufferUsages::UNIFORM,
                )?;
                let vertices = vertex_buffer.finish(
                    &device,
                    create_debug_label!("Shape {} vertices", shape_id),
                    wgpu::BufferUsages::VERTEX,
                )?;
                let indices = index_buffer.finish(
                    &device,
                    create_debug_label!("Shape {} indices", shape_id),
                    wgpu::BufferUsages::INDEX,
                )?;
                let bytes = uniforms.size() + vertices.size() + indices.size();
                (uniforms, vertices, indices, bytes)
            };

        let draws: Vec<_> = draws
            .into_iter()
            .map(|d| d.finish(&self.descriptors, &uniform_buffer, &gradients))
            .collect::<Result<_, _>>()?;

        let allocation = ruffle_render::profile::GpuMesh::new(storage_bytes);

        let retained_bytes = retainable.then_some(()).and_then(|()| {
            crate::mesh::retained_mesh_bytes(storage_bytes, gradient_count, draws.capacity())
        });
        let mesh = Arc::new(Mesh {
            retained_bytes,
            bounds,
            draws,
            vertex_buffer,
            index_buffer,
            _allocation: allocation,
        });
        if let Some(key) = key {
            self.small_meshes.insert(key, &mesh);
        }
        Ok(mesh)
    }

    fn prepare_commands(
        &mut self,
        commands: &CommandList,
        width: u32,
        height: u32,
        quality: StageQuality,
        format: wgpu::TextureFormat,
        load: crate::pass_graph::Load,
    ) -> Result<crate::graph_encoder::GraphEncoder, crate::gpu_runtime::RenderFailure> {
        let samples = crate::utils::supported_sample_count(
            &self.descriptors.adapter,
            quality.sample_count(),
            format,
        );
        crate::graph_encoder::GraphEncoder::prepare(
            self,
            commands,
            ruffle_render::region::RenderRect {
                x: 0,
                y: 0,
                width,
                height,
            },
            samples,
            load,
        )
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
        if ruffle_render::profile::enabled() {
            ruffle_render::profile::record_native_resources(
                self.descriptors.runtime.diagnostic_snapshot(),
            );
        }
        let timestamp_period = self.descriptors.queue.get_timestamp_period();
        while let Some(queries) = self.profiler.process_finished_frame(timestamp_period) {
            if let Some(sample) = crate::gpu_metrics::sample_queries(&queries) {
                ruffle_render::profile::record_gpu_sample(sample);
            }
        }
    }
    pub fn descriptors(&self) -> &Arc<Descriptors> {
        &self.descriptors
    }

    pub fn sample_count(&self) -> u32 {
        self.surface.sample_count()
    }
    pub fn quality(&self) -> StageQuality {
        self.surface.quality()
    }

    pub fn last_submission(&self) -> Option<SubmissionIndex> {
        self.last_submission.clone()
    }

    pub fn target(&self) -> &T {
        &self.target
    }

    pub fn device(&self) -> &crate::gpu_resources::Device {
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
        self.preparation_generation.invalidate();
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
        )?;

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
        )?))
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

    fn stage_sample_count(&self) -> u32 {
        self.surface.sample_count()
    }

    fn set_quality(
        &mut self,
        quality: StageQuality,
    ) -> Result<(), crate::gpu_runtime::RenderFailure> {
        self.preparation_generation.invalidate();
        self.descriptors.runtime.check_health()?;
        self.surface = Surface::new(
            &self.descriptors,
            quality,
            self.surface.size().width,
            self.surface.size().height,
            self.target.format(),
        )?;

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
        self.register_shape_with_scale(shape, bitmap_source, 1.0)
    }

    #[instrument(level = "debug", skip_all)]
    fn register_shape_with_scale(
        &mut self,
        shape: DistilledShape,
        bitmap_source: &dyn BitmapSource,
        scale: f32,
    ) -> Result<ShapeHandle, crate::gpu_runtime::RenderFailure> {
        self.descriptors.runtime.check_health()?;
        let geometry = Arc::new(ruffle_render::shape_utils::OwnedDistilledShape::from(shape));
        let mut bitmaps = crate::shape_recipe::FrozenBitmapSource::default();
        for id in geometry.bitmap_ids() {
            if let std::collections::hash_map::Entry::Vacant(entry) = bitmaps.0.entry(id) {
                if let Some(size) = bitmap_source.bitmap_size(id) {
                    entry.insert((size, bitmap_source.bitmap_handle(id, self)));
                }
            }
        }
        self.descriptors.runtime.check_health()?;
        Ok(ShapeHandle(Arc::new(crate::shape_recipe::ShapeRecipe {
            owner: self
                .descriptors
                .runtime
                .issue_owner(crate::gpu_runtime::OwnerKind::Graphic)?,
            device_generation: self.descriptors.runtime.generation(),
            mesh: Default::default(),
            geometry: std::cell::RefCell::new(Some(geometry)),
            bitmaps,
            scale,
        })))
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
            let prepared = crate::render_snapshot::PreparedSnapshot::capture(&entry.commands)?;
            let cache =
                (&*entry.handle.0 as &dyn Any).downcast_ref::<crate::bitmap_cache::CacheImage>();
            let texture = as_texture(&entry.handle, &self.descriptors.device)?;
            let valid = ruffle_render::region::RenderRect {
                x: 0,
                y: 0,
                width: texture.texture.width(),
                height: texture.texture.height(),
            };
            let (dirty, history) = if let Some(cache) = cache {
                cache.prepare_redraw(
                    &prepared.snapshot,
                    entry.clear,
                    self.surface.quality(),
                    &entry.filters,
                    self.descriptors.runtime.scheduler(),
                )
            } else {
                (ruffle_render::region::RenderRegion::Full, None)
            };
            let Some(dirty) = dirty.clip(valid) else {
                if let Some(cache) = cache {
                    cache.finish_redraw(history);
                }
                continue;
            };
            let (input, output) = crate::bitmap_cache::filter_regions(dirty, valid, &entry.filters);
            let partial = input != valid;
            let surface = Surface::new(
                &self.descriptors,
                self.surface.quality(),
                input.width,
                input.height,
                texture.texture.format(),
            )?
            .with_origin([input.x, input.y]);
            let graph = crate::graph_encoder::GraphEncoder::from_prepared(
                self,
                prepared,
                input,
                surface.sample_count(),
                crate::pass_graph::Load::Clear(entry.clear),
            )?;
            let commands = graph.into_commands()?;
            ruffle_render::profile::work(|w| {
                w.offscreen_targets += 1;
                w.offscreen_pixels += u64::from(input.width) * u64::from(input.height);
            });
            let color = wgpu::Color {
                r: f64::from(entry.clear.r) / 255.0,
                g: f64::from(entry.clear.g) / 255.0,
                b: f64::from(entry.clear.b) / 255.0,
                a: f64::from(entry.clear.a) / 255.0,
            };
            let mode = if partial {
                RenderTargetMode::FreshWithColor(color)
            } else {
                RenderTargetMode::ExistingWithColor(texture.texture.clone(), color)
            };
            let mut scope = self
                .profiler
                .scope("Draw to CAB", self.active_frame.command_encoder.record());
            let mut target = surface.draw_commands(
                mode,
                &self.descriptors,
                &self.meshes,
                commands,
                &mut self.active_frame.staging_belt,
                &mut self.chunk_buffers,
                &mut scope,
                LayerRef::None,
                &mut self.offscreen_texture_pool,
            )?;
            target.ensure_cleared(&mut scope);
            let mut filtered = false;
            if !entry.filters.is_empty()
                && let Some(result) = self.descriptors.filters.apply_chain(
                    &self.descriptors,
                    &mut scope.scope("Filter chain"),
                    &mut self.offscreen_texture_pool,
                    &mut self.active_frame.staging_belt,
                    FilterSource::for_entire_texture(target.color_texture()),
                    entry.filters,
                )?
            {
                target = result;
                filtered = true;
            }
            if partial {
                scope.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: target.color_texture(),
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: output.x - input.x,
                            y: output.y - input.y,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: output.x,
                            y: output.y,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: output.width,
                        height: output.height,
                        depth_or_array_layers: 1,
                    },
                );
            } else if filtered {
                run_copy_pipeline(
                    &self.descriptors,
                    texture.texture.format(),
                    &texture.texture.create_view(&Default::default())?,
                    target.color_view(),
                    target.color_texture().sample_count(),
                    &mut scope.scope("Copy filtered to CAB"),
                )?;
            }
            drop(scope);
            crate::record_bitmap_write(&entry.handle)?;
            if let Some(cache) = cache {
                cache.finish_redraw(history);
            }
            self.active_frame.maybe_flush(&self.descriptors)?;
        }

        let graph = self.prepare_commands(
            &commands,
            self.target.width(),
            self.target.height(),
            self.surface.quality(),
            self.target.format(),
            crate::pass_graph::Load::Clear(clear),
        )?;
        self.surface.draw_graph_and_copy_to(
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
            graph,
            LayerRef::None,
            &mut self.texture_pool,
        )?;
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
        self.profiler
            .end_frame()
            .map_err(|error| self.descriptors.runtime.health().record(error.into()))?;
        ruffle_render::profile::submitted();
        self.collect_gpu_samples();

        self.descriptors.runtime.check_health()?;
        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    fn register_bitmap(&mut self, bitmap: Bitmap<'_>) -> Result<BitmapHandle, BitmapError> {
        self.descriptors.runtime.check_health()?;
        let bitmap = if bitmap.format() == BitmapFormat::Bgra
            && self
                .descriptors
                .adapter
                .get_texture_format_features(wgpu::TextureFormat::Bgra8Unorm)
                .allowed_usages
                .contains(
                    wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::COPY_DST,
                ) {
            bitmap
        } else {
            bitmap.to_rgba()
        };
        self.validate_bitmap_size(&bitmap)?;
        Ok(BitmapHandle(crate::bitmap_recipe::BitmapRecipe::new(
            bitmap,
            &self.descriptors.device,
        )?))
    }

    fn cache_immutable_bitmap(&mut self, bitmap: &BitmapHandle) -> bool {
        let Some(cache) = &self.bitmap_residency else {
            return false;
        };
        let Some(recipe) =
            (&*bitmap.0 as &dyn Any).downcast_ref::<crate::bitmap_recipe::BitmapRecipe>()
        else {
            return false;
        };
        recipe.retain_in(cache);
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

        if let Some(recipe) =
            (&*handle.0 as &dyn Any).downcast_ref::<crate::bitmap_recipe::BitmapRecipe>()
        {
            if recipe.is_resident() && self.active_frame.command_encoder.dirty {
                self.active_frame.submit_direct(&self.descriptors)?;
            }
            return Ok(recipe.update(bitmap, region)?);
        }
        let texture = as_texture(handle, &self.descriptors.device)?;

        let _upload = ruffle_render::profile::time(ruffle_render::profile::Phase::Upload);
        let bitmap = if texture.texture.format() == wgpu::TextureFormat::Bgra8Unorm {
            bitmap.to_bgra()
        } else {
            bitmap.to_rgba()
        };
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
        )?;

        crate::record_bitmap_write(handle)?;
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
        if source_region.is_empty()
            || source_region.width() != destination_region.width()
            || source_region.height() != destination_region.height()
        {
            return Ok(None);
        }

        let source_texture = as_texture(&source, &self.descriptors.device)?;
        let destination_texture = as_texture(&destination, &self.descriptors.device)?;
        if source_region.x_max > source_texture.texture.width()
            || source_region.y_max > source_texture.texture.height()
            || destination_region.x_max > destination_texture.texture.width()
            || destination_region.y_max > destination_texture.texture.height()
        {
            return Ok(None);
        }

        if source_texture.texture.format() != destination_texture.texture.format()
            || !matches!(
                source_texture.texture.format(),
                wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm
            )
        {
            return Ok(None);
        }

        let target = TextureTarget {
            cached_view: None,
            size: wgpu::Extent3d {
                width: destination_texture.texture.width(),
                height: destination_texture.texture.height(),
                depth_or_array_layers: 1,
            },
            texture: destination_texture.texture.clone(),
            format: destination_texture.texture.format(),
            buffer: None,
        };

        crate::transfers::copy(
            &self.descriptors,
            &mut self.offscreen_texture_pool,
            self.active_frame.command_encoder.record(),
            &source,
            source_region,
            &destination,
            destination_region,
        )?;

        crate::record_bitmap_write(&destination)?;
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
        let texture = as_texture(&handle, &self.descriptors.device)?;

        let extent = wgpu::Extent3d {
            width: texture.texture.width(),
            height: texture.texture.height(),
            depth_or_array_layers: 1,
        };

        let mut target = TextureTarget {
            cached_view: None,
            size: extent,
            texture: texture.texture.clone(),
            format: texture.texture.format(),
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
            texture.texture.format(),
        )?;
        let graph = self.prepare_commands(
            &commands,
            extent.width,
            extent.height,
            quality,
            texture.texture.format(),
            crate::pass_graph::Load::Load,
        )?;
        surface.draw_graph_and_copy_to(
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
            graph,
            LayerRef::Current,
            &mut self.offscreen_texture_pool,
        )?;

        crate::record_bitmap_write(&handle)?;
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
        let can_skip_identity = filter.impotent()
            && match (
                (source.0.as_ref() as &dyn Any)
                    .downcast_ref::<crate::bitmap_recipe::BitmapRecipe>(),
                (destination.0.as_ref() as &dyn Any)
                    .downcast_ref::<crate::bitmap_recipe::BitmapRecipe>(),
            ) {
                (Some(input), Some(output)) => {
                    input.can_blit_identity(output, source_point, source_size)
                }
                _ => false,
            };
        let source_texture = as_texture(&source, &self.descriptors.device)?;
        let dest_texture = as_texture(&destination, &self.descriptors.device)?;

        let copy_area = PixelRegion::for_whole_size(
            dest_texture.texture.width(),
            dest_texture.texture.height(),
        );

        let target = TextureTarget {
            cached_view: None,
            size: wgpu::Extent3d {
                width: dest_texture.texture.width(),
                height: dest_texture.texture.height(),
                depth_or_array_layers: 1,
            },
            texture: dest_texture.texture.clone(),
            format: dest_texture.texture.format(),
            buffer: None,
        };

        let input = FilterSource {
            texture: &source_texture.texture,
            point: source_point,
            size: source_size,
        };
        let applied_filter = self.descriptors.filters.apply(
            &self.descriptors,
            self.active_frame.command_encoder.record(),
            &mut self.offscreen_texture_pool,
            &mut self.active_frame.staging_belt,
            input,
            filter,
            can_skip_identity,
        )?;
        let (dest_x, dest_y) = dest_point;

        let src_offset_x = dest_x.min(0).unsigned_abs();
        let src_offset_y = dest_y.min(0).unsigned_abs();

        let final_dest_x = dest_x.max(0) as u32;
        let final_dest_y = dest_y.max(0) as u32;

        let available_width = source_size.0.saturating_sub(src_offset_x);
        let available_height = source_size.1.saturating_sub(src_offset_y);
        let dest_available_width = dest_texture.texture.width().saturating_sub(final_dest_x);
        let dest_available_height = dest_texture.texture.height().saturating_sub(final_dest_y);

        let copy_width = available_width.min(dest_available_width);
        let copy_height = available_height.min(dest_available_height);

        if copy_width == 0 || copy_height == 0 {
            return Ok(None);
        }

        if let Some(result) = &applied_filter {
            let input = result.color_texture();
            let (input_x, input_y) = (src_offset_x, src_offset_y);
            let converted;
            let input = if input.format() != dest_texture.texture.format() {
                converted = self.offscreen_texture_pool.get_texture(
                    &self.descriptors,
                    input.size(),
                    wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    dest_texture.texture.format(),
                    1,
                )?;
                run_copy_pipeline(
                    &self.descriptors,
                    dest_texture.texture.format(),
                    &converted.1,
                    &input.create_view(&Default::default())?,
                    1,
                    self.active_frame.command_encoder.record(),
                )?;
                &converted.0
            } else {
                input
            };
            self.active_frame
                .command_encoder
                .record()
                .copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: input,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: input_x,
                            y: input_y,
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
        } else {
            crate::transfers::copy(
                &self.descriptors,
                &mut self.offscreen_texture_pool,
                self.active_frame.command_encoder.record(),
                &source,
                PixelRegion::for_region(
                    source_point.0 + src_offset_x,
                    source_point.1 + src_offset_y,
                    copy_width,
                    copy_height,
                ),
                &destination,
                PixelRegion::for_region(final_dest_x, final_dest_y, copy_width, copy_height),
            )?;
        }
        crate::record_bitmap_write(&destination)?;
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
                let texture = crate::pixel_bender::as_cache_holder(&shader)
                    .device()
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
                    })?;
                BitmapHandle(Arc::new(Texture {
                    _allocation: ruffle_render::profile::GpuBitmap::new(
                        u64::from(texture.width())
                            * u64::from(texture.height())
                            * u64::from(texture.format().block_copy_size(None).unwrap_or(4)),
                    ),
                    view: texture.create_view(&Default::default())?,
                    texture,
                    batch_source: Default::default(),
                    repeating_linear: Default::default(),
                    repeating_nearest: Default::default(),
                    clamped_linear: Default::default(),
                    clamped_nearest: Default::default(),
                    copy_count: Default::default(),
                    version: Default::default(),
                }))
            }
        };

        let target_texture = as_texture(&target_handle, &self.descriptors.device)?;

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
                    .take(&self.descriptors, copy_dimensions.clone())?,
                copy_dimensions,
            ),
            copy_area: PixelRegion::for_whole_size(
                target_texture.texture.width(),
                target_texture.texture.height(),
            ),
        });

        let mut texture_target = TextureTarget {
            cached_view: None,
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
            None,
            1,
            // When running a standalone shader, we always process the entire image
            &FilterSource::for_entire_texture(&target_texture.texture),
        )?;

        crate::record_bitmap_write(&target_handle)?;
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

    fn create_bitmap_cache(
        &mut self,
        width: NonZeroU32,
        height: NonZeroU32,
    ) -> Result<BitmapHandle, BitmapError> {
        Ok(BitmapHandle(crate::bitmap_cache::CacheImage::new(
            width.get(),
            height.get(),
            &self.descriptors.device,
        )?))
    }
    fn retain_bitmap_cache(
        &mut self,
        handle: &BitmapHandle,
    ) -> Option<Arc<dyn ruffle_render::backend::BitmapCacheLease>> {
        let cache = (&*handle.0 as &dyn Any).downcast_ref::<crate::bitmap_cache::CacheImage>()?;
        Some(cache.retain(self.descriptors.runtime.scheduler()))
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
            })?;
        self.descriptors.runtime.check_health()?;
        Ok(BitmapHandle(Arc::new(Texture {
            _allocation: ruffle_render::profile::GpuBitmap::new(
                u64::from(texture.width())
                    * u64::from(texture.height())
                    * u64::from(texture.format().block_copy_size(None).unwrap_or(4)),
            ),
            view: texture.create_view(&Default::default())?,
            texture,
            batch_source: Default::default(),
            repeating_linear: Default::default(),
            repeating_nearest: Default::default(),
            clamped_linear: Default::default(),
            clamped_nearest: Default::default(),
            copy_count: Default::default(),
            version: Default::default(),
        })))
    }

    fn supports_bitmap_region_readback(&self) -> bool {
        true
    }

    fn read_bitmap_region(
        &mut self,
        handle: &BitmapHandle,
        region: PixelRegion,
        read: RgbaBufRead,
    ) -> Result<Option<()>, ruffle_render::error::Error> {
        self.read_bitmap_region_native(handle, region, rgba_readback(read))
    }

    fn read_bitmap_region_native(
        &mut self,
        handle: &BitmapHandle,
        region: PixelRegion,
        read: BitmapBufRead,
    ) -> Result<Option<()>, BitmapError> {
        self.descriptors.runtime.check_health()?;
        let texture = as_texture(handle, &self.descriptors.device)?;
        if region.is_empty()
            || region.x_max > texture.texture.width()
            || region.y_max > texture.texture.height()
        {
            return Err(crate::gpu_runtime::RenderFailure::new(
                crate::gpu_runtime::RenderFailureKind::Validation,
                "invalid bitmap readback region",
            )
            .into());
        }
        QueueSyncHandle::NotCopied {
            handle: handle.clone(),
            copy_area: region,
            descriptors: self.descriptors.clone(),
            pool: self.offscreen_buffer_pool.clone(),
        }
        .capture_native(read, &mut self.active_frame)
        .map_err(|error| self.descriptors.runtime.health().record(error))?;
        self.descriptors.runtime.check_health()?;
        Ok(Some(()))
    }

    fn resolve_sync_handle(
        &mut self,
        handle: Box<dyn SyncHandle>,
        read: RgbaBufRead,
    ) -> Result<(), ruffle_render::error::Error> {
        self.resolve_sync_handle_native(handle, rgba_readback(read))
    }

    fn resolve_sync_handle_native(
        &mut self,
        handle: Box<dyn SyncHandle>,
        read: BitmapBufRead,
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
            .capture_native(read, &mut self.active_frame)
            .map_err(|error| self.descriptors.runtime.health().record(error))?;
        self.descriptors.runtime.check_health()?;
        Ok(())
    }
}

fn rgba_readback(read: RgbaBufRead<'_>) -> BitmapBufRead<'_> {
    Box::new(move |bytes, stride, format| {
        if format == BitmapFormat::Bgra {
            let mut rgba = bytes.to_vec();
            for pixel in rgba.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
            read(&rgba, stride);
        } else {
            read(bytes, stride);
        }
    })
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
    /// Final attachment selected by the pass graph (including explicit view format).
    ExistingWithView(crate::gpu_resources::TextureView, wgpu::Color),
    // Construct a new frame buffer, cleared with an existing texture.
    // we will blend with the previous contents of the texture.
    // This is used in `render_offscreen`, as we need to blend with the previous
    // contents of our `BitmapData` texture
    FreshWithTexture(crate::gpu_resources::Texture),
    // Use the provided texture as our frame buffer, and clear it with the given color.
    ExistingWithColor(crate::gpu_resources::Texture, wgpu::Color),
}

impl RenderTargetMode {
    pub fn color(&self) -> Option<wgpu::Color> {
        match self {
            RenderTargetMode::FreshWithColor(color)
            | RenderTargetMode::ExistingWithView(_, color) => Some(*color),
            RenderTargetMode::FreshWithTexture(_) => None,
            RenderTargetMode::ExistingWithColor(_, color) => Some(*color),
        }
    }
}

struct TrackedEncoder {
    raw: crate::gpu_resources::CommandEncoder,
    dirty: bool,
}
impl TrackedEncoder {
    fn record(&mut self) -> &mut crate::gpu_resources::CommandEncoder {
        self.dirty = true;
        &mut self.raw
    }
}
pub struct ActiveFrame {
    pub staging_belt: crate::gpu_resources::StagingBelt,
    command_encoder: TrackedEncoder,
    budget: crate::idle_pool::Budget,
}

impl ActiveFrame {
    pub fn new(
        descriptors: &Descriptors,
        budget: crate::idle_pool::Budget,
    ) -> Result<Self, crate::gpu_runtime::RenderFailure> {
        Ok(Self {
            budget,
            command_encoder: TrackedEncoder {
                dirty: false,
                raw: descriptors
                    .device
                    .create_command_encoder(&Default::default())?,
            },
            staging_belt: crate::gpu_resources::StagingBelt::new(descriptors.device.clone(), 65536),
        })
    }

    pub(crate) fn encoder(&mut self) -> &mut crate::gpu_resources::CommandEncoder {
        self.command_encoder.record()
    }
    pub fn submit_for_target<T: RenderTarget>(
        &mut self,
        descriptors: &Descriptors,
        target: &T,
        frame: T::Frame,
    ) -> Result<SubmissionIndex, crate::gpu_runtime::RenderFailure> {
        descriptors.runtime.check_health()?;
        self.staging_belt.finish();
        let draw_encoder = std::mem::replace(
            &mut self.command_encoder.raw,
            descriptors
                .device
                .create_command_encoder(&Default::default())?,
        );
        let index = {
            let _submit = ruffle_render::profile::time(ruffle_render::profile::Phase::RenderSubmit);
            let command_buffer = {
                let _timer = ruffle_render::profile::native_time(
                    ruffle_render::profile::NativePhase::EncoderFinish,
                );
                draw_encoder.finish()?
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
        self.staging_belt.finish();
        let draw_encoder = std::mem::replace(
            &mut self.command_encoder.raw,
            descriptors
                .device
                .create_command_encoder(&Default::default())?,
        );
        let index = {
            let _submit = ruffle_render::profile::time(ruffle_render::profile::Phase::RenderSubmit);
            let command_buffer = {
                let _timer = ruffle_render::profile::native_time(
                    ruffle_render::profile::NativePhase::EncoderFinish,
                );
                draw_encoder.finish()?
            };
            let _timer = ruffle_render::profile::native_time(
                ruffle_render::profile::NativePhase::QueueSubmit,
            );
            descriptors.queue.submit(Some(command_buffer))?
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
        let pending = self
            .command_encoder
            .raw
            .transient_bytes()
            .saturating_add(descriptors.queue.pending_upload_bytes());
        if self.command_encoder.dirty && pending >= descriptors.runtime.config().transient_watermark
        {
            self.submit_direct(descriptors)?;
        }

        descriptors.runtime.check_health()?;
        Ok(())
    }
}
