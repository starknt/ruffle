use crate::buffer_builder::BufferBuilder;
use crate::buffer_pool::{BufferPool, TexturePool};
use crate::context3d::WgpuContext3D;
use crate::filters::FilterSource;
use crate::gpu_profiler::{GpuProfiler, GpuProfilerSettings};
use crate::gpu_runtime::{RenderFailure, RenderFailureKind};
use crate::mesh::{CommonGradient, Mesh, PendingDraw};
use crate::pixel_bender::{ShaderMode, run_pixelbender_shader_impl};
use crate::render_snapshot::{PacketKey, StageDrawRoute};
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

mod stage;
use stage::PendingStage;

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
    stage_frame: Option<T::Frame>,
    packet_key: PacketKey,
    pending_stage: Option<PendingStage>,
    owned_frames: bool,
    display_enhancer: Option<Arc<dyn crate::display_bitmap::DisplayBitmapEnhancer>>,
    surface: Surface,
    meshes: Vec<Mesh>,
    small_meshes: crate::mesh::SmallMeshCache,
    definition_meshes: crate::mesh::DefinitionMeshes,
    filtered_tiles: Option<crate::filtered_tiles::FilteredTiles>,
    retained_bitmaps: Option<Arc<crate::bitmap_cache::RetainedBitmaps>>,
    #[cfg(target_os = "linux")]
    pool_trim_counts: Option<[u64; 4]>,
    pub(crate) preparation_generation: crate::prepare::PreparationGeneration,
    // This is currently unused - we just store it to report in
    // `get_viewport_dimensions`
    viewport_scale_factor: f64,
    texture_pool: TexturePool,
    offscreen_texture_pool: TexturePool,
    pub(crate) offscreen_buffer_pool:
        Arc<BufferPool<crate::gpu_resources::Buffer, BufferDimensions>>,
    chunk_buffers: ChunkBuffers,
    pub(super) active_frame: ActiveFrame,
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
        self.stage_frame = None;
        self.invalidate_stage_packets()?;
        let descriptors = &self.descriptors;
        let surface = unsafe { descriptors.wgpu_instance.create_surface_unsafe(window)? };
        let target =
            SwapChainTarget::new(surface, &descriptors.adapter, size, &descriptors.device)?;
        let stage = Surface::new(
            descriptors,
            self.surface.quality(),
            size.0,
            size.1,
            target.format(),
        )?;
        self.target = target;
        self.surface = stage;
        self.texture_pool = TexturePool::new(self.budget.clone());
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
        let filtered_tiles = (std::env::var_os("RUFFLE_FILTERED_TILES").as_deref()
            == Some(std::ffi::OsStr::new("1")))
        .then(|| crate::filtered_tiles::FilteredTiles::new(descriptors.runtime.scheduler()));
        let retained_bitmaps = (std::env::var_os("RUFFLE_INDEXED_BITMAP_LEASES").as_deref()
            == Some(std::ffi::OsStr::new("1")))
        .then(|| crate::bitmap_cache::RetainedBitmaps::new(descriptors.runtime.scheduler()));
        descriptors.runtime.check_health()?;
        Ok(Self {
            budget: budget.clone(),
            packet_key: PacketKey {
                device: descriptors.runtime.generation(),
                session: descriptors
                    .runtime
                    .issue_owner(crate::gpu_runtime::OwnerKind::Scene)?,
                target_generation: 1,
                reservation: None,
                packet: 0,
                operation: 0,
            },
            pending_stage: None,
            owned_frames: std::env::var_os("RUFFLE_OWNED_FRAMES").as_deref() == Some("1".as_ref()),
            descriptors,
            target,
            stage_frame: None,
            surface,
            display_enhancer: None,
            meshes: Vec::new(),
            small_meshes: Default::default(),
            definition_meshes: Default::default(),
            filtered_tiles,
            retained_bitmaps,
            #[cfg(target_os = "linux")]
            pool_trim_counts: std::env::var_os("RUFFLE_VK_POOL_TRIM").map(|_| [0; 4]),
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
    pub fn set_display_bitmap_enhancer(
        &mut self,
        enhancer: Arc<dyn crate::display_bitmap::DisplayBitmapEnhancer>,
    ) {
        self.display_enhancer = Some(enhancer);
    }

    pub fn set_immutable_bitmap_cache_limit(&mut self, limit: Option<u64>) {
        self.bitmap_residency = limit.filter(|limit| *limit > 0).map(|limit| {
            Arc::new(crate::bitmap_residency::BitmapResidency::with_runtime(
                &self.descriptors.runtime,
                limit,
            ))
        });
    }

    #[cfg(test)]
    pub(crate) fn enable_indexed_bitmap_leases(&mut self) {
        self.retained_bitmaps.get_or_insert_with(|| {
            crate::bitmap_cache::RetainedBitmaps::new(self.descriptors.runtime.scheduler())
        });
    }

    #[cfg(test)]
    pub(crate) fn set_filtered_tiles(&mut self, enabled: bool) {
        self.filtered_tiles = enabled.then(|| {
            crate::filtered_tiles::FilteredTiles::new(self.descriptors.runtime.scheduler())
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
        commands: CommandList,
        width: u32,
        height: u32,
        quality: StageQuality,
        format: wgpu::TextureFormat,
        load: crate::pass_graph::Load,
    ) -> Result<crate::graph_encoder::SurfaceCommands, crate::gpu_runtime::RenderFailure> {
        let _preparation = ruffle_render::profile::native_time(
            ruffle_render::profile::NativePhase::CommandPreparation,
        );
        use crate::graph_encoder::SurfaceCommands;
        // Stage cache redraws have finished. Keep these handles only through this
        // submission. Offscreen targets may alias their inputs and use the graph.
        if matches!(load, crate::pass_graph::Load::Clear(_)) {
            match self.stage_draw_route(&commands, format) {
                StageDrawRoute::AlignedBitmaps => {
                    return Ok(SurfaceCommands::CachedBitmaps(commands));
                }
                StageDrawRoute::Resident => return Ok(SurfaceCommands::ResidentDraws(commands)),
                StageDrawRoute::Graph => {}
            }
        }
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
        .map(SurfaceCommands::Graph)
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
        if self.pending_stage.as_ref().is_some_and(|p| p.cancelled()) {
            self.pending_stage = None;
        }
        if ruffle_render::profile::enabled() {
            #[cfg(target_os = "linux")]
            if let Some(previous) = &mut self.pool_trim_counts {
                // Read only atomic counters while the device guard is live.
                let stats = unsafe { self.descriptors.device.as_hal::<wgpu::hal::api::Vulkan>() }
                    .and_then(|device| device.command_pool_trim_counts());
                if let Some(current) = stats {
                    ruffle_render::profile::work(|w| {
                        w.vulkan_pool_release_resets += current[0].saturating_sub(previous[0]);
                        w.vulkan_pool_freed_buffers += current[1].saturating_sub(previous[1]);
                        w.vulkan_pool_replacements += current[2].saturating_sub(previous[2]);
                        w.vulkan_pool_reserved_buffers += current[3].saturating_sub(previous[3]);
                    });
                    *previous = current;
                }
            }
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

    fn finish_bitmap_write(&mut self, handle: &BitmapHandle) -> Result<(), RenderFailure> {
        crate::record_bitmap_write(handle)?;
        self.active_frame.operations.record()
    }

    fn preserve_read_before_write(
        &mut self,
        destination: &BitmapHandle,
    ) -> Result<(), crate::gpu_runtime::RenderFailure> {
        self.preserve_stage_before_write(destination)?;
        if let Some(recipe) =
            (&*destination.0 as &dyn Any).downcast_ref::<crate::bitmap_recipe::BitmapRecipe>()
            && !recipe.is_resident()
        {
            return Ok(());
        }
        let texture = as_texture(destination, &self.descriptors.device)?;
        let pending = {
            let mut weak = texture.pending_read.lock().unwrap();
            let pending = weak.upgrade();
            if let Some(read) = &pending {
                read.dependency.check(&self.active_frame.operations)?;
            }
            *weak = Default::default();
            pending
        };
        if let Some(read) = pending {
            let result = read.record(
                destination,
                &self.descriptors,
                &self.offscreen_buffer_pool,
                &mut self.active_frame,
            );
            let status = result.as_ref().map(|_| ()).map_err(Clone::clone);
            *read.saved.lock().unwrap() = Some(result);
            status?;
        }
        Ok(())
    }

    fn make_queue_sync_handle(
        &self,
        target: TextureTarget,
        index: Option<SubmissionIndex>,
        destination: BitmapHandle,
        copy_area: PixelRegion,
    ) -> Result<Box<QueueSyncHandle>, crate::gpu_runtime::RenderFailure> {
        let handle = match target.take_buffer() {
            None => QueueSyncHandle::deferred(
                destination,
                copy_area,
                self.descriptors.clone(),
                self.offscreen_buffer_pool.clone(),
                self.active_frame.operations.dependency(),
                true,
            )?,
            Some(TextureBufferInfo {
                buffer: MaybeOwnedBuffer::Borrowed(buffer, dimensions),
                ..
            }) => QueueSyncHandle {
                dependency: self.active_frame.operations.dependency(),
                descriptors: self.descriptors.clone(),
                source: crate::ReadbackSource::Copied(crate::CopiedReadback {
                    index,
                    buffer,
                    dimensions,
                }),
            },
            Some(TextureBufferInfo {
                buffer: MaybeOwnedBuffer::Owned(..),
                ..
            }) => unreachable!("offscreen sync uses a pooled buffer"),
        };
        Ok(Box::new(handle))
    }
}

enum WgpuPixelBenderPending {
    Bitmap(Option<Box<dyn SyncHandle>>),
    Bytes {
        sync_handle: Option<QueueSyncHandle>,
        readback: Option<AsyncReadback>,
        width: usize,
        output_channels: usize,
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
        self.stage_frame = None;
        self.invalidate_stage_packets()?;
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
        let context = WgpuContext3D::for_session(
            self.descriptors.clone(),
            profile,
            self.active_frame.operations.clone(),
        )?;
        Ok(Box::new(context))
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
        self.invalidate_stage_packets()?;
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
            _lifetime: self.active_frame.operations.create_resource()?,
        })))
    }

    fn register_swf_shape(
        &mut self,
        definition: &std::rc::Rc<swf::Shape>,
        bitmap_source: &dyn BitmapSource,
        scale: f32,
    ) -> Result<ShapeHandle, crate::gpu_runtime::RenderFailure> {
        self.descriptors.runtime.check_health()?;
        if let Some(handle) = self.definition_meshes.get(definition, scale) {
            return Ok(handle);
        }
        let shape = DistilledShape::from(definition.as_ref());
        // A bitmap ID is scoped to its movie, even when that movie has no
        // bitmap registered under the ID yet. Inspect uses before resolving it.
        use ruffle_render::shape_utils::DrawPath;
        let movie_independent = shape.paths.iter().all(|path| {
            let fill = match path {
                DrawPath::Fill { style, .. } => *style,
                DrawPath::Stroke { style, .. } => style.fill_style(),
            };
            !matches!(fill, swf::FillStyle::Bitmap { .. })
        });
        let handle = self.register_shape_with_scale(shape, bitmap_source, scale)?;
        if movie_independent {
            self.definition_meshes.insert(definition, scale, &handle);
        }
        Ok(handle)
    }

    fn prepare_frame(&mut self) -> Result<bool, crate::gpu_runtime::RenderFailure> {
        self.descriptors.runtime.check_health()?;
        if self.stage_frame.is_none() {
            let _timer = ruffle_render::profile::native_time(
                ruffle_render::profile::NativePhase::SurfaceAcquire,
            );
            self.stage_frame = self.target.get_next_texture()?;
        }
        Ok(self.stage_frame.is_some())
    }

    fn discard_frame(&mut self) {
        self.stage_frame = None;
        self.pending_stage = None;
    }

    #[instrument(level = "debug", skip_all)]
    fn submit_prepared_frame(
        &mut self,
        clear: Color,
        commands: CommandList,
        cache_entries: Vec<BitmapCacheEntry>,
    ) -> Result<ruffle_render::backend::FrameSubmission, crate::gpu_runtime::RenderFailure> {
        self.submit_inline_frame(clear, commands, cache_entries)
    }

    #[instrument(level = "debug", skip_all)]
    fn display_sampling_revision(&self) -> u64 {
        self.display_enhancer.as_ref().map_or(0, |e| e.revision())
    }
    fn register_bitmap(&mut self, bitmap: Bitmap<'_>) -> Result<BitmapHandle, BitmapError> {
        self.register_bitmap_with_usage(bitmap, ruffle_render::backend::BitmapUsage::Mutable)
    }
    fn register_bitmap_with_usage(
        &mut self,
        bitmap: Bitmap<'_>,
        usage: ruffle_render::backend::BitmapUsage,
    ) -> Result<BitmapHandle, BitmapError> {
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
        let detail = self
            .display_enhancer
            .as_ref()
            .filter(|e| e.enabled() && usage != ruffle_render::backend::BitmapUsage::Excluded)
            .map(|_| {
                let pixels = bitmap.data().chunks_exact(4);
                let mut count = 0u32;
                let mut transparent = 0;
                let mut edges = 0;
                let mut last = None;
                for px in pixels
                    .clone()
                    .step_by((pixels.len() / 128).max(1))
                    .take(128)
                {
                    count += 1;
                    transparent += u32::from(px[3] < 255);
                    let value = u16::from(px[1]) + u16::from(px[0]) / 2 + u16::from(px[2]) / 2;
                    edges += u32::from(last.is_some_and(|v: u16| v.abs_diff(value) > 40));
                    last = Some(value);
                }
                crate::display_bitmap::BitmapDetail {
                    transparent: (transparent * 255 / count.max(1)) as u8,
                    edges: (edges * 255 / count.max(1)) as u8,
                }
            });
        Ok(BitmapHandle(
            crate::bitmap_recipe::BitmapRecipe::with_usage(
                bitmap,
                &self.descriptors.device,
                usage,
                detail,
                &self.active_frame.operations,
            )?,
        ))
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

        self.preserve_read_before_write(handle)?;
        if let Some(recipe) =
            (&*handle.0 as &dyn Any).downcast_ref::<crate::bitmap_recipe::BitmapRecipe>()
        {
            if recipe.is_resident() && self.active_frame.command_encoder.dirty {
                self.active_frame.submit_direct(&self.descriptors)?;
            }
            recipe.update(bitmap, region)?;
            self.active_frame.operations.record()?;
            return Ok(());
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

        self.finish_bitmap_write(handle)?;
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

        self.preserve_read_before_write(&destination)?;
        crate::transfers::copy(
            &self.descriptors,
            &mut self.offscreen_texture_pool,
            self.active_frame.command_encoder.record(),
            &source,
            source_region,
            &destination,
            destination_region,
        )?;

        self.finish_bitmap_write(&destination)?;
        self.active_frame.maybe_flush(&self.descriptors)?;
        self.descriptors.runtime.check_health()?;
        Ok(Some(self.make_queue_sync_handle(
            target,
            None,
            destination,
            destination_region,
        )?))
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
        self.preserve_read_before_write(&handle)?;
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
            commands,
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

        self.finish_bitmap_write(&handle)?;
        self.active_frame.maybe_flush(&self.descriptors)?;
        self.descriptors.runtime.check_health()?;
        Ok(Some(
            self.make_queue_sync_handle(target, None, handle, bounds)?,
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
        self.preserve_read_before_write(&destination)?;
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
        self.finish_bitmap_write(&destination)?;
        self.active_frame.maybe_flush(&self.descriptors)?;
        self.descriptors.runtime.check_health()?;
        Ok(Some(self.make_queue_sync_handle(
            target,
            None,
            destination,
            copy_area,
        )?))
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
                let stride = buffer_width as usize;
                let row_bytes = *width * *output_channels * size_of::<f32>();
                let pixels = if row_bytes == stride {
                    raw_pixels
                } else {
                    let mut packed = Vec::with_capacity(raw_pixels.len() / stride * row_bytes);
                    for row in raw_pixels.chunks_exact(stride) {
                        if *output_channels == 3 {
                            // RGB uses an RGBA32Float texture. Omit each pixel's unused alpha.
                            for pixel in row.chunks_exact(size_of::<[f32; 4]>()).take(*width) {
                                packed.extend_from_slice(&pixel[..size_of::<[f32; 3]>()]);
                            }
                        } else {
                            packed.extend_from_slice(&row[..row_bytes]);
                        }
                    }
                    packed
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
                    pending_read: Default::default(),
                    _lifetime: Some(self.active_frame.operations.create_resource()?),
                    version: Default::default(),
                }))
            }
        };

        self.preserve_read_before_write(&target_handle)?;
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

        self.finish_bitmap_write(&target_handle)?;
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
        )?;

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
            &self.active_frame.operations,
        )?))
    }
    fn retain_bitmap_cache(
        &mut self,
        handle: &BitmapHandle,
    ) -> Option<Arc<dyn ruffle_render::backend::BitmapCacheLease>> {
        let cache = (&*handle.0 as &dyn Any).downcast_ref::<crate::bitmap_cache::CacheImage>()?;
        Some(cache.retain(
            self.descriptors.runtime.scheduler(),
            self.retained_bitmaps.as_ref(),
        ))
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
            pending_read: Default::default(),
            _lifetime: Some(self.active_frame.operations.create_resource()?),
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
        QueueSyncHandle::deferred(
            handle.clone(),
            region,
            self.descriptors.clone(),
            self.offscreen_buffer_pool.clone(),
            self.active_frame.operations.dependency(),
            false,
        )?
        .capture_native(read, &mut self.active_frame)
        .map_err(|error| self.descriptors.runtime.health().record(error))?;
        self.descriptors.runtime.check_health()?;
        Ok(Some(()))
    }

    fn read_bitmap_regions_native(
        &mut self,
        handle: &BitmapHandle,
        bounds: PixelRegion,
        parts: &[PixelRegion],
        visit: ruffle_render::bitmap::BitmapRegionRead<'_>,
    ) -> Result<Option<()>, BitmapError> {
        self.descriptors.runtime.check_health()?;
        let copy = crate::CopiedReadback::record_regions(
            handle,
            bounds,
            parts,
            &self.descriptors,
            &self.offscreen_buffer_pool,
            &mut self.active_frame,
        )?;
        let pixel_bytes = copy.dimensions.format.block_copy_size(None).unwrap() as usize;
        let token = QueueSyncHandle {
            dependency: self.active_frame.operations.dependency(),
            descriptors: self.descriptors.clone(),
            source: crate::ReadbackSource::Copied(copy),
        };
        token.capture_native(
            |bytes, stride, format| {
                for &part in parts {
                    let start = (part.y_min - bounds.y_min) as usize * stride as usize
                        + (part.x_min - bounds.x_min) as usize * pixel_bytes;
                    let length = (part.height() - 1) as usize * stride as usize
                        + part.width() as usize * pixel_bytes;
                    visit(part, &bytes[start..start + length], stride, format)?;
                }
                Ok::<_, RenderFailure>(())
            },
            &mut self.active_frame,
        )??;
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
        handle.dependency.check(&self.active_frame.operations)?;
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
    // This wgpu preset also matches native GLES 3.0, which lacks compute and storage buffers.
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
    pub(crate) operations: crate::render_snapshot::OperationSequence,
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
            operations: crate::render_snapshot::OperationSequence::new(
                descriptors.runtime.health(),
            ),
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
