use super::prepared_draws::{Group, Key as DrawKey, PreparedDraws};
use crate::backend::RenderTargetMode;
use crate::blend::BlendType;
use crate::blend::TrivialBlend;
use crate::buffer_pool::TexturePool;
use crate::gpu_profiler::Scope;
use crate::graph_encoder::{PlannedList, PlannedOp};
use crate::mesh::{DrawType, Mesh, as_mesh};
use crate::surface::Surface;
use crate::surface::buffers::{ChunkBufferBuilder, ChunkBuffers, DrawOffsets, DrawVertices};
use crate::surface::chunk::{Chunk, ChunkBlendMode, DrawCommand};
use crate::surface::mask::{MaskMode, MaskState};
use crate::surface::target::{CommandTarget, PoolOrArcTexture};
use crate::{Descriptors, PosUvVertex, as_texture};
use ruffle_render::backend::ShapeHandle;
use ruffle_render::bitmap::{BitmapHandle, PixelRegion, PixelSnapping};
use ruffle_render::commands::{Command, CommandHandler, CommandList, RenderBlendMode};
use ruffle_render::lines::{emulate_line, emulate_line_rect};
use ruffle_render::matrix::Matrix;
use ruffle_render::quality::StageQuality;
use ruffle_render::region::RenderRect;
use ruffle_render::transform::Transform;
use std::{mem, rc::Rc, sync::Arc};
use swf::{BlendMode, Color, ColorTransform, Twips};
use wgpu::Backend;

#[derive(Copy, Clone)]
pub enum LayerRef<'a> {
    None,
    Current,
    Parent(&'a CommandTarget),
}

/// Replaces every blend with a RenderBitmap, with the subcommands rendered out to a temporary texture
/// Every complex blend will be its own item, but every other draw will be chunked together
#[expect(clippy::too_many_arguments)]
pub fn chunk_blends<'encoder, 'global: 'encoder>(
    display: Option<&'encoder Arc<dyn crate::display_bitmap::DisplayBitmapEnhancer>>,
    commands: PlannedList,
    descriptors: &'encoder Descriptors,
    staging_belt: &'encoder mut crate::gpu_resources::StagingBelt,
    chunk_buffers: &'encoder mut ChunkBuffers,
    draw_encoder: &'encoder mut Scope<'global, crate::gpu_resources::CommandEncoder>,
    meshes: &'encoder Vec<Mesh>,
    quality: StageQuality,
    width: u32,
    height: u32,
    origin: [u32; 2],
    destination: &'encoder crate::gpu_resources::Texture,
    nearest_layer: LayerRef,
    texture_pool: &'encoder mut TexturePool,
    mask_state: MaskState,
) -> Result<Vec<Chunk>, crate::gpu_runtime::RenderFailure> {
    WgpuCommandHandler::new(
        display,
        descriptors,
        staging_belt,
        chunk_buffers,
        draw_encoder,
        meshes,
        quality,
        width,
        height,
        origin,
        destination,
        nearest_layer,
        texture_pool,
        mask_state,
    )
    .chunk_blends(commands)
}

// Exact, immutable solid geometry in the same target-local coordinate system.
// This cache lives only while preparing one bounded sibling set; it never spans
// frames, changes draw order, or retains an image after the last encoded use.
#[derive(PartialEq)]
struct SolidInputKey {
    vertices: crate::gpu_runtime::ResourceId,
    indices: crate::gpu_runtime::ResourceId,
    vertex_range: std::ops::Range<u64>,
    index_range: std::ops::Range<u32>,
    transform: Transform,
    size: [u32; 2],
    clear: wgpu::Color,
}
impl SolidInputKey {
    fn new(
        commands: &PlannedList,
        region: RenderRect,
        clear: wgpu::Color,
    ) -> Result<Option<Self>, crate::gpu_runtime::RenderFailure> {
        let Some(Command::RenderShape { shape, transform }) = commands.single_leaf() else {
            return Ok(None);
        };
        if transform.tz != 0.0 || transform.perspective_projection.is_some() {
            return Ok(None);
        }
        let mesh = as_mesh(shape)?;
        let [draw] = mesh.draws.as_slice() else {
            return Ok(None);
        };
        if !matches!(draw.draw_type, DrawType::Color) {
            return Ok(None);
        }
        let mut transform = transform.clone();
        transform.matrix.tx -= Twips::from_pixels(region.x as f64);
        transform.matrix.ty -= Twips::from_pixels(region.y as f64);
        Ok(Some(Self {
            vertices: mesh.vertex_buffer.lease().id(),
            indices: mesh.index_buffer.lease().id(),
            vertex_range: draw.vertices.clone(),
            index_range: draw.indices.clone(),
            transform,
            size: [region.width, region.height],
            clear,
        }))
    }
}

struct WgpuCommandHandler<'encoder, 'global: 'encoder> {
    display: Option<&'encoder Arc<dyn crate::display_bitmap::DisplayBitmapEnhancer>>,
    descriptors: &'encoder Descriptors,
    quality: StageQuality,
    width: u32,
    height: u32,
    origin: [u32; 2],
    destination: &'encoder crate::gpu_resources::Texture,
    nearest_layer: LayerRef<'encoder>,
    meshes: &'encoder Vec<Mesh>,
    staging_belt: &'encoder mut crate::gpu_resources::StagingBelt,
    chunk_buffers: &'encoder mut ChunkBuffers,
    draw_encoder: &'encoder mut Scope<'global, crate::gpu_resources::CommandEncoder>,
    texture_pool: &'encoder mut TexturePool,
    emulate_lines: bool,
    recording_block: bool,

    solid_inputs: Vec<(SolidInputKey, Arc<PoolOrArcTexture>)>,
    result: Vec<Chunk>,
    current: Vec<DrawCommand>,
    chunk_buffer_builder: ChunkBufferBuilder,
    needs_stencil: bool,
    mask_state: MaskState,
}

impl<'encoder, 'global: 'encoder> WgpuCommandHandler<'encoder, 'global> {
    #[expect(clippy::too_many_arguments)]
    fn new(
        display: Option<&'encoder Arc<dyn crate::display_bitmap::DisplayBitmapEnhancer>>,
        descriptors: &'encoder Descriptors,
        staging_belt: &'encoder mut crate::gpu_resources::StagingBelt,
        chunk_buffers: &'encoder mut ChunkBuffers,
        draw_encoder: &'encoder mut Scope<'global, crate::gpu_resources::CommandEncoder>,
        meshes: &'encoder Vec<Mesh>,
        quality: StageQuality,
        width: u32,
        height: u32,
        origin: [u32; 2],
        destination: &'encoder crate::gpu_resources::Texture,
        nearest_layer: LayerRef<'encoder>,
        texture_pool: &'encoder mut TexturePool,
        mask_state: MaskState,
    ) -> Self {
        let chunk_buffer_builder = ChunkBufferBuilder::new(chunk_buffers, &descriptors.limits);

        // DirectX does support drawing lines, but it's very inconsistent.
        // With MSAA, lines have 1.4px thickness, which makes them too thick.
        // Without MSAA, lines have 1px thickness, but their placement is sometimes off.
        let emulate_lines = descriptors.backend == Backend::Dx12;

        Self {
            display,
            descriptors,
            quality,
            width,
            height,
            origin,
            destination,
            nearest_layer,
            meshes,
            staging_belt,
            chunk_buffers,
            draw_encoder,
            texture_pool,
            emulate_lines,
            recording_block: false,

            solid_inputs: vec![],
            result: vec![],
            current: vec![],
            chunk_buffer_builder,
            needs_stencil: mask_state.count > 0,
            mask_state,
        }
    }

    /// Replaces every blend with a RenderBitmap, with the subcommands rendered out to a temporary texture
    /// Every complex blend will be its own item, but every other draw will be chunked together
    fn chunk_blends(
        &mut self,
        commands: PlannedList,
    ) -> Result<Vec<Chunk>, crate::gpu_runtime::RenderFailure> {
        let commands = commands.into_commands();
        self.current.reserve(commands.len().min(64));
        for command in commands {
            match command.op {
                PlannedOp::Leaf(command) => command.execute(self),
                PlannedOp::Blend(commands, mode, region) => {
                    self.blend_planned(commands, mode, region)
                }
                PlannedOp::AlphaMask(content, mask, region) => {
                    self.alpha_mask_planned(content, mask, region)
                }
            }
        }
        self.descriptors.runtime.check_health()?;

        let current = mem::take(&mut self.current);
        let mut result = mem::take(&mut self.result);
        let needs_stencil = mem::take(&mut self.needs_stencil);
        let chunk_buffer_builder = mem::replace(
            &mut self.chunk_buffer_builder,
            ChunkBufferBuilder::new(self.chunk_buffers, &self.descriptors.limits),
        );

        if !current.is_empty() {
            result.push(Chunk::Draw {
                chunk: current,
                needs_stencil,
                chunk_buffer_builder,
            });
        }

        Ok(result)
    }

    fn finish_draw_group(&mut self) {
        let next = ChunkBufferBuilder::new(self.chunk_buffers, &self.descriptors.limits);
        if self.current.is_empty() {
            self.chunk_buffer_builder = next;
            return;
        }
        self.result.push(Chunk::Draw {
            chunk: mem::take(&mut self.current),
            needs_stencil: self.needs_stencil,
            chunk_buffer_builder: mem::replace(&mut self.chunk_buffer_builder, next),
        });
    }

    fn replay_window(&mut self, group: &Group, window: std::ops::Range<u32>) {
        let stride = size_of::<crate::Transforms>();
        let bytes = &group.instances[window.start as usize * stride..window.end as usize * stride];
        let first = self
            .chunk_buffer_builder
            .append_instance_bytes(bytes)
            .expect("window fits the current transform buffer");
        for draw in &group.draws {
            if let Some(command) = draw.in_window(&window, first) {
                self.push_command(command);
            }
        }
    }

    fn replay_groups(&mut self, prepared: &PreparedDraws) {
        for group in &prepared.groups {
            let count = (group.instances.len() / size_of::<crate::Transforms>()) as u32;
            let remaining = self
                .chunk_buffer_builder
                .remaining_instances(&self.descriptors.limits);
            if count > remaining {
                if remaining != 0 {
                    self.replay_window(group, 0..remaining);
                }
                self.finish_draw_group();
                self.replay_window(group, remaining..count);
            } else {
                self.replay_window(group, 0..count);
            }
        }
        ruffle_render::profile::work(|w| w.prepared_draw_replays += 1);
    }

    fn push_command(&mut self, command: DrawCommand) {
        if let Some(existing) = self.current.last_mut()
            && existing.try_merge(&command)
        {
            return;
        }
        self.current.push(command);
    }

    fn add_to_current(
        &mut self,
        transform: &Transform,
        instance: &mut Option<u32>,
        vertices: DrawVertices,
        command_builder: impl FnOnce(DrawOffsets) -> DrawCommand,
    ) {
        let mut local_transform;
        let transform = if self.origin != [0, 0] {
            local_transform = transform.clone();
            local_transform.matrix.tx -= Twips::from_pixels(self.origin[0] as f64);
            local_transform.matrix.ty -= Twips::from_pixels(self.origin[1] as f64);
            &local_transform
        } else {
            transform
        };
        let result = match *instance {
            Some(existing) => Ok(existing),
            None => self.chunk_buffer_builder.add_transform(transform),
        };
        let result = match result {
            Ok(new_instance) => {
                *instance = Some(new_instance);
                self.chunk_buffer_builder
                    .add_draw(&self.descriptors.quad, new_instance, vertices)
            }
            Err(e) => Err((e, vertices)),
        };

        match result {
            Ok(offsets) => {
                self.push_command(command_builder(offsets));
            }
            Err((_, vertices)) => {
                self.result.push(Chunk::Draw {
                    chunk: mem::take(&mut self.current),
                    needs_stencil: self.needs_stencil,
                    chunk_buffer_builder: mem::replace(
                        &mut self.chunk_buffer_builder,
                        ChunkBufferBuilder::new(self.chunk_buffers, &self.descriptors.limits),
                    ),
                });
                // The transform may already have been written to the old builder, but it's not valid
                // in the new one - so we must write it again.
                let new_instance = self
                    .chunk_buffer_builder
                    .add_transform(transform)
                    .expect("Buffer must be able to fit a new transform, it was just emptied");
                *instance = Some(new_instance);
                let offsets = self
                    .chunk_buffer_builder
                    .add_draw(&self.descriptors.quad, new_instance, vertices)
                    .expect("Buffer must be able to fit a new thing, it was just emptied");
                self.push_command(command_builder(offsets));
            }
        }
    }
}

impl CommandHandler for WgpuCommandHandler<'_, '_> {
    fn render_shared(&mut self, commands: Rc<CommandList>) {
        // Only certified immutable vector blocks can store prepared materials.
        // Bitmap draws and textured meshes still resolve their views on replay.
        let key = DrawKey {
            device: self.descriptors.runtime.generation(),
            origin: self.origin,
            quality: self.quality,
            mask_geometry: matches!(
                self.mask_state.mode,
                MaskMode::DrawMaskStencil | MaskMode::ClearMaskStencil
            ),
        };
        if !self.recording_block
            && let Some(prepared) = commands.prepared_draws::<PreparedDraws>()
            && prepared.key == key
        {
            ruffle_render::profile::work(|w| w.prepared_chunk_hits += 1);
            self.replay_groups(&prepared);
            return;
        }
        if self.recording_block
            || commands.iter().len() < 2
            || !commands.may_prepare_draws(key.device)
        {
            commands.execute_ref(self);
            return;
        }
        // Only certified immutable resident draws reach this builder. Capture
        // their CPU-side instance bytes and merged material runs in isolation.
        let outer_current = mem::take(&mut self.current);
        let outer_result = mem::take(&mut self.result);
        let fresh = ChunkBufferBuilder::new(self.chunk_buffers, &self.descriptors.limits);
        let outer_buffers = mem::replace(&mut self.chunk_buffer_builder, fresh);
        self.recording_block = true;
        commands.execute_ref(self);
        self.recording_block = false;
        self.finish_draw_group();
        let chunks = mem::replace(&mut self.result, outer_result);
        self.current = outer_current;
        self.chunk_buffer_builder = outer_buffers;
        if let Some(prepared) = PreparedDraws::from_chunks(key, chunks) {
            let bytes = prepared.bytes();
            let prepared = Rc::new(prepared);
            self.replay_groups(&prepared);
            commands.store_draws(prepared, bytes);
        } else {
            commands.decline_draw_packet();
            commands.execute_ref(self);
        }
    }

    fn blend(&mut self, _: CommandList, _: RenderBlendMode) {
        self.descriptors
            .runtime
            .fail(crate::gpu_runtime::RenderFailure::new(
                crate::gpu_runtime::RenderFailureKind::Internal,
                "unplanned blend reached native encoding",
            ));
    }

    fn render_bitmap(
        &mut self,
        bitmap: BitmapHandle,
        mut transform: Transform,
        smoothing: bool,
        pixel_snapping: PixelSnapping,
        region: PixelRegion,
    ) {
        if self.descriptors.runtime.check_health().is_err() {
            return;
        }
        let result = (|| -> Result<(), crate::gpu_runtime::RenderFailure> {
            let original = as_texture(&bitmap, &self.descriptors.device)?;
            let derived = if smoothing
                && let Some(enhancer) = self.display
                && let Some(source) = crate::display_bitmap::Source::for_handle(&bitmap)
            {
                let m = &transform.matrix;
                enhancer.sample(
                    source,
                    m.a.hypot(m.b).max(m.c.hypot(m.d)),
                    crate::display_bitmap::Sampling::Direct,
                    self.draw_encoder,
                )?
            } else {
                None
            };
            let texture = derived.as_ref().map_or(original, |d| d.texture.as_ref());
            let uv = derived.as_ref().map_or([0., 0., 1., 1.], |d| d.uv);

            pixel_snapping.apply(&mut transform.matrix);
            transform.matrix *= Matrix::scale(region.width() as f32, region.height() as f32);

            let binds = texture.bind_group(
                false,
                smoothing,
                &self.descriptors.device,
                &self.descriptors.bind_layouts.bitmap,
                bitmap.clone(),
                &self.descriptors.bitmap_samplers,
            )?;

            let command_builder = |draw_offsets| DrawCommand::RenderTexture {
                _texture: Some(PoolOrArcTexture::Manual((
                    texture.texture.clone(),
                    texture.view.clone(),
                ))),
                binds: binds.bind_group.clone(),
                batch_sampler: (texture.texture.mip_level_count() == 1).then(|| {
                    (
                        self.descriptors
                            .bitmap_samplers
                            .get_sampler(false, smoothing)
                            .clone(),
                        texture.batch_source.clone(),
                    )
                }),
                draw_offsets,
                blend_mode: TrivialBlend::Normal,
                render_stage3d: false,
            };

            let texture_width = original.texture.width();
            let texture_height = original.texture.height();

            if uv == [0., 0., 1., 1.]
                && region.x_min == 0
                && region.y_min == 0
                && region.width() == texture_width
                && region.height() == texture_height
            {
                // Special case "full texture" so that we can merge subsequent draw calls
                self.add_to_current(
                    &transform,
                    &mut None,
                    DrawVertices::PosUvQuad,
                    command_builder,
                );
            } else {
                let (u0, u1, v0, v1) = (
                    uv[0] + region.x_min as f32 / texture_width as f32 * (uv[2] - uv[0]),
                    uv[0] + region.x_max as f32 / texture_width as f32 * (uv[2] - uv[0]),
                    uv[1] + region.y_min as f32 / texture_height as f32 * (uv[3] - uv[1]),
                    uv[1] + region.y_max as f32 / texture_height as f32 * (uv[3] - uv[1]),
                );
                let vertices = &[
                    PosUvVertex::new(0.0, 0.0, u0, v0, 1.0),
                    PosUvVertex::new(1.0, 0.0, u1, v0, 1.0),
                    PosUvVertex::new(1.0, 1.0, u1, v1, 1.0),
                    PosUvVertex::new(0.0, 1.0, u0, v1, 1.0),
                ];
                self.add_to_current(
                    &transform,
                    &mut None,
                    DrawVertices::CustomPosUvQuad(bytemuck::cast_slice(vertices)),
                    command_builder,
                );
            };

            Ok(())
        })();
        if let Err(error) = result {
            self.descriptors.runtime.fail(error);
        }
    }

    fn render_stage3d(&mut self, bitmap: BitmapHandle, mut transform: Transform) {
        if self.descriptors.runtime.check_health().is_err() {
            return;
        }
        let result = (|| -> Result<(), crate::gpu_runtime::RenderFailure> {
            {
                let texture = as_texture(&bitmap, &self.descriptors.device)?;
                transform.matrix *= Matrix::scale(
                    texture.texture.width() as f32,
                    texture.texture.height() as f32,
                );
            }
            let texture = as_texture(&bitmap, &self.descriptors.device)?;

            let binds = texture.bind_group(
                false,
                false,
                &self.descriptors.device,
                &self.descriptors.bind_layouts.bitmap,
                bitmap.clone(),
                &self.descriptors.bitmap_samplers,
            )?;
            self.add_to_current(
                &transform,
                &mut None,
                DrawVertices::PosUvQuad,
                |draw_offsets| DrawCommand::RenderTexture {
                    _texture: Some(PoolOrArcTexture::Manual((
                        texture.texture.clone(),
                        texture.view.clone(),
                    ))),
                    draw_offsets,
                    blend_mode: TrivialBlend::Normal,
                    render_stage3d: true,
                    batch_sampler: None,
                    binds: binds.bind_group.clone(),
                },
            );

            Ok(())
        })();
        if let Err(error) = result {
            self.descriptors.runtime.fail(error);
        }
    }

    fn render_shape(&mut self, shape: ShapeHandle, transform: Transform) {
        let mesh = match as_mesh(&shape) {
            Ok(mesh) => mesh,
            Err(error) => {
                self.descriptors.runtime.fail(error);
                return;
            }
        };
        let mut instance = None;
        for draw in &mesh.draws {
            let indices = if self.mask_state.mode != MaskMode::DrawMaskStencil
                && self.mask_state.mode != MaskMode::ClearMaskStencil
            {
                draw.indices.clone()
            } else {
                // Omit strokes when drawing a mask stencil.
                draw.mask_indices.clone()
            };
            if indices.is_empty() {
                continue;
            }

            let vertices = DrawVertices::CustomBuffers {
                vertex_buffer: mesh.vertex_buffer.clone(),
                vertex_range: draw.vertices.clone(),
                index_buffer: mesh.index_buffer.clone(),
                indices,
            };
            match &draw.draw_type {
                DrawType::Color => {
                    self.add_to_current(&transform, &mut instance, vertices, |draw_offsets| {
                        DrawCommand::DrawColor { draw_offsets }
                    });
                }
                DrawType::Gradient { bind_group } => {
                    self.add_to_current(&transform, &mut instance, vertices, |draw_offsets| {
                        DrawCommand::RenderGradient {
                            bind_group: bind_group.clone(),
                            draw_offsets,
                        }
                    });
                }
                DrawType::Bitmap { binds, display } => {
                    let mut binds = binds.bind_group.clone();
                    if let (Some(enhancer), Some(fill)) = (self.display, display)
                        && let Some(source) = fill.source.refresh()
                    {
                        let sampling = if fill.repeating {
                            crate::display_bitmap::Sampling::RepeatingFill
                        } else {
                            crate::display_bitmap::Sampling::ClampedFill
                        };
                        let result = enhancer
                            .sample(
                                source,
                                fill.magnification(&transform),
                                sampling,
                                self.draw_encoder,
                            )
                            .and_then(|sample| {
                                sample
                                    .map(|s| {
                                        s.texture
                                            .bind_group(
                                                fill.repeating,
                                                true,
                                                &self.descriptors.device,
                                                &self.descriptors.bind_layouts.bitmap,
                                                BitmapHandle(s.texture.clone()),
                                                &self.descriptors.bitmap_samplers,
                                            )
                                            .map(|b| b.bind_group.clone())
                                    })
                                    .transpose()
                            });
                        match result {
                            Ok(Some(enhanced)) => binds = enhanced,
                            Ok(None) => {}
                            Err(error) => {
                                self.descriptors.runtime.fail(error);
                                return;
                            }
                        }
                    }
                    self.add_to_current(&transform, &mut instance, vertices, |draw_offsets| {
                        DrawCommand::RenderTexture {
                            _texture: None,
                            batch_sampler: None,
                            binds: binds.clone(),
                            blend_mode: TrivialBlend::Normal,
                            render_stage3d: false,
                            draw_offsets,
                        }
                    });
                }
            }
        }
    }

    fn draw_rect(&mut self, color: Color, matrix: Matrix) {
        self.add_to_current(
            &Transform {
                matrix,
                tz: 0.0,
                color_transform: ColorTransform::multiply_from(color),
                perspective_projection: None,
            },
            &mut None,
            DrawVertices::PosColorQuad,
            |draw_offsets| DrawCommand::DrawColor { draw_offsets },
        );
    }

    fn draw_line(&mut self, color: Color, mut matrix: Matrix) {
        if self.emulate_lines {
            let mut cl = CommandList::new();
            emulate_line(&mut cl, color, matrix);
            cl.execute(self);
        } else {
            matrix.tx += Twips::HALF_PX;
            matrix.ty += Twips::HALF_PX;
            self.add_to_current(
                &Transform {
                    matrix,
                    tz: 0.0,
                    color_transform: ColorTransform::multiply_from(color),
                    perspective_projection: None,
                },
                &mut None,
                DrawVertices::Line,
                |draw_offsets| DrawCommand::DrawLine { draw_offsets },
            );
        }
    }

    fn draw_line_rect(&mut self, color: Color, mut matrix: Matrix) {
        if self.emulate_lines {
            let mut cl = CommandList::new();
            emulate_line_rect(&mut cl, color, matrix);
            cl.execute(self);
        } else {
            matrix.tx += Twips::HALF_PX;
            matrix.ty += Twips::HALF_PX;
            self.add_to_current(
                &Transform {
                    matrix,
                    tz: 0.0,
                    color_transform: ColorTransform::multiply_from(color),
                    perspective_projection: None,
                },
                &mut None,
                DrawVertices::RectLine,
                |draw_offsets| DrawCommand::DrawLine { draw_offsets },
            );
        }
    }

    fn push_mask(&mut self) {
        self.needs_stencil = true;
        self.mask_state.push_mask();
        self.push_command(DrawCommand::SetMask(self.mask_state));
    }

    fn activate_mask(&mut self) {
        self.needs_stencil = true;
        self.mask_state.activate_mask();
        self.push_command(DrawCommand::SetMask(self.mask_state));
    }

    fn deactivate_mask(&mut self) {
        self.needs_stencil = true;
        self.mask_state.deactivate_mask();
        self.push_command(DrawCommand::SetMask(self.mask_state));
    }

    fn pop_mask(&mut self) {
        self.needs_stencil = true;
        self.mask_state.pop_mask();
        self.push_command(DrawCommand::SetMask(self.mask_state));
    }

    fn render_alpha_mask(&mut self, _: CommandList, _: CommandList) {
        self.descriptors
            .runtime
            .fail(crate::gpu_runtime::RenderFailure::new(
                crate::gpu_runtime::RenderFailureKind::Internal,
                "unplanned alpha mask reached native encoding",
            ));
    }
}

impl WgpuCommandHandler<'_, '_> {
    fn blend_planned(
        &mut self,
        commands: PlannedList,
        blend_mode: RenderBlendMode,
        region: RenderRect,
    ) {
        if self.descriptors.runtime.check_health().is_err() {
            return;
        }
        let result = (|| -> Result<(), crate::gpu_runtime::RenderFailure> {
            let target_layer = if let RenderBlendMode::Builtin(BlendMode::Layer) = &blend_mode {
                LayerRef::Current
            } else {
                self.nearest_layer
            };
            let blend_type = BlendType::from(blend_mode);
            let clear_color = blend_type.default_color();
            let commands = if self.display.is_none()
                && self.descriptors.direct_bitmap_blends
                && matches!(blend_type, BlendType::Trivial(_))
            {
                commands.remove_redundant_add_layers(region)
            } else {
                commands
            };
            let direct = if self.display.is_none()
                && self.descriptors.direct_bitmap_blends
                && matches!(blend_type, BlendType::Trivial(_))
                && matches!(
                    self.mask_state.mode,
                    MaskMode::NoMask | MaskMode::DrawMaskedContent
                )
                && let Some((texture, matrix)) = self.aligned_bitmap(&commands)?
            {
                // Transparent AA padding has no effect for these four blends.
                // Keep clipped inputs on the general path so their UVs stay exact.
                let x = matrix.tx.to_pixels();
                let y = matrix.ty.to_pixels();
                let width = texture.texture.width();
                let height = texture.texture.height();
                (x >= f64::from(region.x)
                    && y >= f64::from(region.y)
                    && x + f64::from(width) <= f64::from(region.x + region.width)
                    && y + f64::from(height) <= f64::from(region.y + region.height))
                .then(|| {
                    (
                        PoolOrArcTexture::Manual((texture.texture.clone(), texture.view.clone())),
                        RenderRect {
                            x: x as u32,
                            y: y as u32,
                            width,
                            height,
                        },
                    )
                })
            } else {
                None
            };
            let key = if matches!(blend_type, BlendType::Shader(_)) {
                None
            } else {
                SolidInputKey::new(&commands, region, clear_color)?
            };
            let reused = key.as_ref().and_then(|key| {
                self.solid_inputs
                    .iter()
                    .find(|(candidate, _)| candidate == key)
                    .map(|(_, texture)| texture.clone())
            });
            let raster_key = self.descriptors.blend_inputs.as_ref().and_then(|cache| {
                if self.display.is_some()
                    || direct.is_some()
                    || reused.is_some()
                    || matches!(blend_type, BlendType::Shader(_))
                {
                    return None;
                }
                Some((
                    cache,
                    crate::blend_inputs::Key::capture(&commands, region, self.quality)?,
                ))
            });
            let mut region = region;
            let texture = if let Some((texture, bounds)) = direct {
                region = bounds;
                ruffle_render::profile::work(|w| w.direct_bitmap_blends += 1);
                texture
            } else if let Some(texture) = reused {
                PoolOrArcTexture::Shared(texture)
            } else if let Some(texture) = raster_key
                .as_ref()
                .and_then(|(cache, key)| cache.lookup(key))
            {
                ruffle_render::profile::work(|w| w.blend_raster_hits += 1);
                PoolOrArcTexture::Manual(texture)
            } else {
                ruffle_render::profile::work(|w| {
                    w.offscreen_targets += 1;
                    w.offscreen_pixels += u64::from(region.width) * u64::from(region.height);
                });
                let surface = Surface::new(
                    self.descriptors,
                    self.quality,
                    region.width,
                    region.height,
                    wgpu::TextureFormat::Rgba8Unorm,
                )?
                .with_origin([region.x, region.y])
                .with_display(self.display.cloned());
                let target = surface.draw_commands(
                    RenderTargetMode::FreshWithColor(clear_color),
                    self.descriptors,
                    self.meshes,
                    commands,
                    self.staging_belt,
                    self.chunk_buffers,
                    self.draw_encoder,
                    target_layer,
                    self.texture_pool,
                )?;
                target.ensure_cleared(self.draw_encoder);
                let admitted =
                    raster_key.and_then(|(cache, key)| Some((cache, cache.admit(&key)?, key)));
                let texture = match (target.take_color_texture(), admitted) {
                    (PoolOrArcTexture::Pool(entry), Some((cache, ticket, key))) => {
                        let raster = entry.into_owned();
                        cache.store(key, raster.clone(), ticket);
                        PoolOrArcTexture::Manual(raster)
                    }
                    (texture, _) => texture,
                };
                if let Some(key) = key
                    && self.solid_inputs.len() < 64
                {
                    let texture = Arc::new(texture);
                    self.solid_inputs.push((key, texture.clone()));
                    PoolOrArcTexture::Shared(texture)
                } else {
                    texture
                }
            };

            match blend_type {
                BlendType::Trivial(blend_mode) => {
                    let transform = Transform {
                        matrix: Matrix {
                            tx: Twips::from_pixels(region.x as f64),
                            ty: Twips::from_pixels(region.y as f64),
                            ..Matrix::scale(region.width as f32, region.height as f32)
                        },
                        tz: 0.0,
                        color_transform: Default::default(),
                        perspective_projection: None,
                    };
                    let bind_group =
                        self.descriptors
                            .cached_bind_group(&wgpu::BindGroupDescriptor {
                                layout: &self.descriptors.bind_layouts.bitmap,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: wgpu::BindingResource::TextureView(
                                            texture.view(),
                                        ),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: wgpu::BindingResource::Sampler(
                                            self.descriptors
                                                .bitmap_samplers
                                                .get_sampler(false, false),
                                        ),
                                    },
                                ],
                                label: None,
                            })?;
                    self.add_to_current(
                        &transform,
                        &mut None,
                        DrawVertices::PosUvQuad,
                        |draw_offsets| DrawCommand::RenderTexture {
                            _texture: Some(texture),
                            batch_sampler: None,
                            binds: bind_group,
                            draw_offsets,
                            blend_mode,
                            render_stage3d: false,
                        },
                    );
                }
                blend_type => {
                    if !self.current.is_empty() {
                        self.result.push(Chunk::Draw {
                            chunk: mem::take(&mut self.current),
                            needs_stencil: self.needs_stencil,
                            chunk_buffer_builder: mem::replace(
                                &mut self.chunk_buffer_builder,
                                ChunkBufferBuilder::new(
                                    self.chunk_buffers,
                                    &self.descriptors.limits,
                                ),
                            ),
                        });
                    }
                    let chunk_blend_mode = match blend_type {
                        BlendType::Complex(complex) => ChunkBlendMode::Complex(complex),
                        BlendType::Shader(shader) => ChunkBlendMode::Shader(shader),
                        _ => unreachable!(),
                    };
                    self.result.push(Chunk::Blend {
                        region: RenderRect {
                            x: region.x - self.origin[0],
                            y: region.y - self.origin[1],
                            ..region
                        },
                        texture,
                        blend_mode: chunk_blend_mode,
                        needs_stencil: self.mask_state.count > 0,
                    });
                    self.needs_stencil = self.mask_state.count > 0;
                }
            }

            Ok(())
        })();
        if let Err(error) = result {
            self.descriptors.runtime.fail(error);
        }
    }

    fn aligned_bitmap<'a>(
        &self,
        commands: &'a PlannedList,
    ) -> Result<Option<(&'a crate::Texture, Matrix)>, crate::gpu_runtime::RenderFailure> {
        let Some(Command::RenderBitmap {
            bitmap,
            transform,
            pixel_snapping,
            region,
            ..
        }) = commands.single_leaf()
        else {
            return Ok(None);
        };
        let mut matrix = transform.matrix;
        pixel_snapping.apply(&mut matrix);
        if matrix.a != 1.0
            || matrix.d != 1.0
            || matrix.b != 0.0
            || matrix.c != 0.0
            || matrix.tx.to_pixels().fract() != 0.0
            || matrix.ty.to_pixels().fract() != 0.0
            || transform.tz != 0.0
            || transform.perspective_projection.is_some()
            || transform.color_transform != ColorTransform::default()
        {
            return Ok(None);
        }
        let texture = as_texture(bitmap, &self.descriptors.device)?;
        if texture.texture.raw() == self.destination.raw()
            || texture.texture.format() != wgpu::TextureFormat::Rgba8Unorm
            || texture.texture.sample_count() != 1
            || *region
                != PixelRegion::for_whole_size(texture.texture.width(), texture.texture.height())
        {
            return Ok(None);
        }
        Ok(Some((texture, matrix)))
    }

    fn alpha_mask_bitmap_input(
        &self,
        commands: &PlannedList,
        output: RenderRect,
    ) -> Result<Option<(PoolOrArcTexture, [PosUvVertex; 4])>, crate::gpu_runtime::RenderFailure>
    {
        let Some((texture, matrix)) = self.aligned_bitmap(commands)? else {
            return Ok(None);
        };
        // Integer, one-to-one RGBA8 rasterization would reproduce these exact texels.
        // Keep the parent target's bounded quad and map its UVs into the original image;
        // the shader supplies transparent pixels beyond the image's coverage.
        let vertices =
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]].map(|position| PosUvVertex {
                position,
                uv: [
                    ((f64::from(output.x) + f64::from(position[0]) * f64::from(output.width)
                        - matrix.tx.to_pixels())
                        / f64::from(texture.texture.width())) as f32,
                    ((f64::from(output.y) + f64::from(position[1]) * f64::from(output.height)
                        - matrix.ty.to_pixels())
                        / f64::from(texture.texture.height())) as f32,
                    1.0,
                ],
            });
        Ok(Some((
            PoolOrArcTexture::Manual((texture.texture.clone(), texture.view.clone())),
            vertices,
        )))
    }

    fn alpha_mask_planned(
        &mut self,
        maskee_commands: PlannedList,
        mask_commands: PlannedList,
        region: RenderRect,
    ) {
        if self.descriptors.runtime.check_health().is_err() {
            return;
        }
        let result = (|| -> Result<(), crate::gpu_runtime::RenderFailure> {
            let direct = self.alpha_mask_bitmap_input(&maskee_commands, region)?;
            let scheduler = self.descriptors.runtime.scheduler();
            let cache_input = match mask_commands.single_leaf() {
                Some(Command::RenderBitmap {
                    bitmap,
                    transform,
                    smoothing,
                    pixel_snapping,
                    region: source,
                }) => (&*bitmap.0 as &dyn std::any::Any)
                    .downcast_ref::<crate::bitmap_cache::CacheImage>()
                    .map(|image| {
                        (
                            image.mask_raster(scheduler),
                            crate::bitmap_cache::MaskKey {
                                version: image.version(),
                                output: region,
                                quality: self.quality,
                                transform: transform.clone(),
                                source: *source,
                                snapping: *pixel_snapping,
                                smoothing: *smoothing,
                            },
                        )
                    }),
                _ => None,
            };
            let (cached, ticket) = cache_input
                .as_ref()
                .map_or((None, None), |(raster, key)| raster.lookup(key, scheduler));
            let inputs = u64::from(direct.is_none()) + u64::from(cached.is_none());
            ruffle_render::profile::work(|w| {
                w.offscreen_targets += inputs;
                w.offscreen_pixels += inputs * u64::from(region.width) * u64::from(region.height);
            });
            let surface = (inputs != 0)
                .then(|| {
                    Surface::new(
                        self.descriptors,
                        self.quality,
                        region.width,
                        region.height,
                        wgpu::TextureFormat::Rgba8Unorm,
                    )
                    .map(|surface| {
                        surface
                            .with_origin([region.x, region.y])
                            .with_display(self.display.cloned())
                    })
                })
                .transpose()?;

            let (maskee, vertices) = if let Some((texture, vertices)) = direct {
                (texture, Some(vertices))
            } else {
                let maskee = surface.as_ref().expect("uncached content").draw_commands(
                    RenderTargetMode::FreshWithColor(wgpu::Color::TRANSPARENT),
                    self.descriptors,
                    self.meshes,
                    maskee_commands,
                    self.staging_belt,
                    self.chunk_buffers,
                    self.draw_encoder,
                    LayerRef::None,
                    self.texture_pool,
                )?;
                maskee.ensure_cleared(self.draw_encoder);
                (maskee.take_color_texture(), None)
            };
            let matrix = Matrix {
                tx: Twips::from_pixels(region.x as f64),
                ty: Twips::from_pixels(region.y as f64),
                ..Matrix::scale(region.width as f32, region.height as f32)
            };

            let mask = if let Some(texture) = cached {
                PoolOrArcTexture::Manual(texture)
            } else {
                let mask = surface.as_ref().expect("uncached mask").draw_commands(
                    RenderTargetMode::FreshWithColor(wgpu::Color::TRANSPARENT),
                    self.descriptors,
                    self.meshes,
                    mask_commands,
                    self.staging_belt,
                    self.chunk_buffers,
                    self.draw_encoder,
                    LayerRef::None,
                    self.texture_pool,
                )?;
                mask.ensure_cleared(self.draw_encoder);
                match (mask.take_color_texture(), cache_input, ticket) {
                    (PoolOrArcTexture::Pool(entry), Some((cache, key)), Some(ticket)) => {
                        let texture = entry.into_owned();
                        cache.store(&key, texture.clone(), ticket, scheduler);
                        PoolOrArcTexture::Manual(texture)
                    }
                    (texture, _, _) => texture,
                }
            };
            let binds = self
                .descriptors
                .cached_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &self.descriptors.bind_layouts.alpha_mask,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(maskee.view()),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(mask.view()),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(
                                self.descriptors.bitmap_samplers.get_sampler(false, false),
                            ),
                        },
                    ],
                    label: None,
                })?;

            self.add_to_current(
                &Transform {
                    matrix,
                    tz: 0.0,
                    color_transform: Default::default(),
                    perspective_projection: None,
                },
                &mut None,
                match &vertices {
                    Some(vertices) => DrawVertices::CustomPosUvQuad(bytemuck::cast_slice(vertices)),
                    None => DrawVertices::PosUvQuad,
                },
                |draw_offsets| DrawCommand::RenderAlphaMask {
                    maskee,
                    mask,
                    binds,
                    draw_offsets,
                },
            );

            Ok(())
        })();
        if let Err(error) = result {
            self.descriptors.runtime.fail(error);
        }
    }
}
