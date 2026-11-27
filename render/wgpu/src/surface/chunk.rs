use crate::blend::{ComplexBlend, TrivialBlend};
use crate::gpu_runtime::RenderFailure;
use crate::pipelines::Pipelines;
use crate::surface::buffers::{ChunkBufferBuilder, ChunkBuffers, DrawOffsets};
use crate::surface::mask::MaskState;
use crate::surface::target::PoolOrArcTexture;
use ruffle_render::pixel_bender::PixelBenderShaderHandle;

pub enum Chunk {
    Draw {
        chunk: Vec<DrawCommand>,
        needs_stencil: bool,
        chunk_buffer_builder: ChunkBufferBuilder,
    },
    Blend {
        region: ruffle_render::region::RenderRect,
        texture: PoolOrArcTexture,
        blend_mode: ChunkBlendMode,
        needs_stencil: bool,
    },
}

#[derive(Debug)]
pub enum ChunkBlendMode {
    Complex(ComplexBlend),
    Shader(PixelBenderShaderHandle),
}

#[derive(Debug)]
pub enum DrawCommand {
    RenderTexture {
        // If this is from a texture pool, we need to keep a copy to make sure it isn't freed before we use it.
        // We don't actually _need_ that texture though, thus the _
        _texture: Option<PoolOrArcTexture>,
        binds: crate::gpu_resources::BindGroup,
        batch_sampler: Option<(crate::gpu_resources::Sampler, crate::bitmap_batch::Source)>,
        blend_mode: TrivialBlend,
        render_stage3d: bool,
        draw_offsets: DrawOffsets,
    },
    RenderTextureBatch {
        _textures: Vec<PoolOrArcTexture>,
        indirect: Option<super::indirect::Draw>,
        binds: crate::gpu_resources::BindGroup,
        draw_offsets: DrawOffsets,
    },
    RenderAlphaMask {
        maskee: PoolOrArcTexture,
        mask: PoolOrArcTexture,
        binds: crate::gpu_resources::BindGroup,
        draw_offsets: DrawOffsets,
    },
    DrawColor {
        draw_offsets: DrawOffsets,
    },
    RenderGradient {
        bind_group: crate::gpu_resources::BindGroup,
        draw_offsets: DrawOffsets,
    },
    DrawLine {
        draw_offsets: DrawOffsets,
    },
    SetMask(MaskState),
}

impl DrawCommand {
    fn bitmap_input(
        &self,
    ) -> Option<(
        &PoolOrArcTexture,
        &crate::gpu_resources::Sampler,
        &DrawOffsets,
    )> {
        match self {
            Self::RenderTexture {
                _texture: Some(texture),
                batch_sampler: Some((sampler, _)),
                blend_mode: TrivialBlend::Normal,
                render_stage3d: false,
                draw_offsets,
                ..
            } if draw_offsets.instances.end - draw_offsets.instances.start == 1 => {
                Some((texture, sampler, draw_offsets))
            }
            _ => None,
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            DrawCommand::RenderTexture { .. } => "render texture",
            DrawCommand::RenderTextureBatch { .. } => "render texture batch",
            DrawCommand::DrawColor { .. } => "draw color",
            DrawCommand::DrawLine { .. } => "draw line",
            DrawCommand::SetMask(_) => "set mask state",
            DrawCommand::RenderAlphaMask { .. } => "render alpha mask",
            DrawCommand::RenderGradient { .. } => "draw gradient",
        }
    }

    fn batch_key(&self) -> Option<BatchKey> {
        Some(match self {
            Self::DrawColor { .. } => BatchKey::Color,
            Self::DrawLine { .. } => BatchKey::Line,
            Self::RenderGradient { bind_group, .. } => BatchKey::Gradient(bind_group.lease().id()),
            Self::RenderTexture {
                binds,
                blend_mode,
                render_stage3d,
                ..
            } => BatchKey::Texture(binds.lease().id(), *blend_mode, *render_stage3d),
            Self::RenderAlphaMask { binds, .. } => BatchKey::AlphaMask(binds.lease().id()),
            Self::SetMask(_) | Self::RenderTextureBatch { .. } => return None,
        })
    }
    fn offsets_mut(&mut self) -> Option<&mut DrawOffsets> {
        match self {
            Self::DrawColor { draw_offsets }
            | Self::DrawLine { draw_offsets }
            | Self::RenderGradient { draw_offsets, .. }
            | Self::RenderTexture { draw_offsets, .. }
            | Self::RenderTextureBatch { draw_offsets, .. }
            | Self::RenderAlphaMask { draw_offsets, .. } => Some(draw_offsets),
            Self::SetMask(_) => None,
        }
    }
    pub fn try_merge(&mut self, other: &DrawCommand) -> bool {
        if let (Self::SetMask(current), Self::SetMask(next)) = (&mut *self, other) {
            *current = *next;
            return true;
        }
        let Some(key) = self.batch_key() else {
            return false;
        };
        if other.batch_key() != Some(key) {
            return false;
        }
        let primitive_size = if key == BatchKey::Line { 2 } else { 3 };
        let offsets = match other {
            Self::DrawColor { draw_offsets }
            | Self::DrawLine { draw_offsets }
            | Self::RenderGradient { draw_offsets, .. }
            | Self::RenderTexture { draw_offsets, .. }
            | Self::RenderTextureBatch { draw_offsets, .. }
            | Self::RenderAlphaMask { draw_offsets, .. } => draw_offsets,
            Self::SetMask(_) => return false,
        };
        self.offsets_mut()
            .is_some_and(|current| current.try_merge(offsets, primitive_size))
    }
}

/// Copy small compatible sources into consecutive array layers in draw order.
fn bitmap_layers(
    run: &[DrawCommand],
    first_instance: u32,
    sampler: &crate::gpu_resources::Sampler,
    descriptors: &crate::Descriptors,
    encoder: &mut crate::gpu_resources::CommandEncoder,
    pool: &mut crate::buffer_pool::TexturePool,
) -> Result<
    Option<(PoolOrArcTexture, crate::gpu_resources::BindGroup)>,
    crate::gpu_runtime::RenderFailure,
> {
    let source = |i: usize| run[i].bitmap_input().unwrap().0.texture();
    let size = source(0).size();
    let format = source(0).format();
    let count = crate::layouts::BITMAP_BATCH_TEXTURES as u32;
    // These experiments pack small conventional bitmaps. Render targets or differing
    // extents keep their original ordered single-image draws.
    if size.width > 256
        || size.height > 256
        || size.depth_or_array_layers != 1
        || !matches!(
            format,
            wgpu::TextureFormat::Rgba8Unorm
                | wgpu::TextureFormat::Rgba8UnormSrgb
                | wgpu::TextureFormat::Bgra8Unorm
                | wgpu::TextureFormat::Bgra8UnormSrgb
        )
        || (0..run.len()).any(|i| {
            let texture = source(i);
            texture.size() != size
                || texture.format() != format
                || texture.sample_count() != 1
                || !texture.usage().contains(wgpu::TextureUsages::COPY_SRC)
        })
    {
        return Ok(None);
    }
    let storage = pool.get_texture(
        descriptors,
        wgpu::Extent3d {
            depth_or_array_layers: count,
            ..size
        },
        wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        format,
        1,
    )?;
    for i in 0..run.len() {
        let mut destination = storage.0.as_image_copy();
        destination.origin.z = (first_instance + i as u32) % count;
        encoder.copy_texture_to_texture(source(i).as_image_copy(), destination, size);
    }
    let group = descriptors
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Adjacent bitmap layers"),
            layout: &descriptors.bind_layouts.bitmap_batch,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&storage.1),
                },
                wgpu::BindGroupEntry {
                    binding: count,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })?;
    Ok(Some((PoolOrArcTexture::Pool(storage), group)))
}

/// Fixed-size, adjacent bitmap runs. Only one-instance quads with identical
/// geometry and sampler are combined; all other commands remain barriers.
pub(super) fn batch_bitmap_textures(
    commands: &mut Vec<DrawCommand>,
    descriptors: &crate::Descriptors,
    encoder: &mut crate::gpu_resources::CommandEncoder,
    texture_pool: &mut crate::buffer_pool::TexturePool,
) -> Result<(), crate::gpu_runtime::RenderFailure> {
    use crate::capabilities::BitmapBindings;
    use crate::layouts::BITMAP_BATCH_TEXTURES as N;
    let mut input = std::mem::take(commands).into_iter().peekable();
    let mut counts =
        ruffle_render::profile::enabled().then(ruffle_render::profile::RenderWork::default);
    while let Some(first) = input.next() {
        let Some((_, sampler, offsets)) = first.bitmap_input() else {
            commands.push(first);
            continue;
        };
        let sampler = sampler.clone();
        let mut offsets = offsets.clone();
        let mut run = smallvec::SmallVec::<[DrawCommand; N]>::new();
        run.push(first);
        loop {
            let Some(next) = input.peek() else {
                if let Some(work) = &mut counts {
                    work.bitmap_run_chunk_end += 1;
                }
                break;
            };
            let Some((_, next_sampler, next_offsets)) = next.bitmap_input() else {
                if let Some(work) = &mut counts {
                    match next {
                        DrawCommand::SetMask(_) | DrawCommand::RenderAlphaMask { .. } => {
                            work.bitmap_split_mask += 1
                        }
                        DrawCommand::RenderTexture { blend_mode, .. }
                            if *blend_mode != TrivialBlend::Normal =>
                        {
                            work.bitmap_split_blend += 1
                        }
                        _ => work.bitmap_split_command += 1,
                    }
                }
                break;
            };
            if *next_sampler != sampler {
                if let Some(work) = &mut counts {
                    work.bitmap_split_sampler += 1;
                }
                break;
            }
            if next_offsets.instances.start != offsets.instances.end
                || next_offsets.vertex_buffer != offsets.vertex_buffer
                || next_offsets.index_buffer != offsets.index_buffer
                || next_offsets.vertices != offsets.vertices
                || next_offsets.indices != offsets.indices
            {
                if let Some(work) = &mut counts {
                    work.bitmap_split_geometry += 1;
                }
                break;
            }
            if run.len() == N {
                if let Some(work) = &mut counts {
                    work.bitmap_split_capacity += 1;
                }
                break;
            }
            offsets.instances.end = next_offsets.instances.end;
            run.extend(input.next());
        }
        if run.len() == 1 {
            commands.extend(run);
            continue;
        }
        let layers = if descriptors.bitmap_batches.mode == BitmapBindings::Layers {
            let Some(layers) = bitmap_layers(
                &run,
                offsets.instances.start,
                &sampler,
                descriptors,
                encoder,
                texture_pool,
            )?
            else {
                commands.extend(run);
                continue;
            };
            Some(layers)
        } else {
            None
        };
        if let Some(work) = &mut counts {
            work.bitmap_batch_runs += 1;
            work.bitmap_batched_instances += run.len() as u64;
            work.bitmap_binding_array_batches +=
                u64::from(descriptors.bitmap_batches.mode == BitmapBindings::Array);
            work.bitmap_layer_batches += u64::from(layers.is_some());
        }
        // The first instance may start anywhere in a chunk's uniform window.
        // Populate each binding once, using the first image for unused slots.
        let views = layers.is_none().then(|| {
            std::array::from_fn(|slot| {
                let index = (slot + N - offsets.instances.start as usize % N) % N;
                run.get(index)
                    .unwrap_or(&run[0])
                    .bitmap_input()
                    .unwrap()
                    .0
                    .view()
                    .clone()
            })
        });
        let mut sources = smallvec::SmallVec::<[_; N]>::new();
        let mut textures = Vec::with_capacity(run.len() + usize::from(layers.is_some()));
        for command in run {
            let DrawCommand::RenderTexture {
                _texture: Some(texture),
                batch_sampler: Some((_, source)),
                ..
            } = command
            else {
                unreachable!()
            };
            textures.push(texture);
            sources.push(source);
        }
        let binds = if let Some((storage, group)) = layers {
            textures.push(storage);
            group
        } else {
            descriptors.bitmap_batches.get_or_create(
                &descriptors.device,
                &descriptors.bind_layouts.bitmap_batch,
                views.as_ref().expect("explicit batch texture views"),
                &sampler,
                &sources,
            )?
        };
        commands.push(DrawCommand::RenderTextureBatch {
            _textures: textures,
            indirect: None,
            binds,
            draw_offsets: offsets,
        });
    }
    if let Some(counts) = counts {
        ruffle_render::profile::work(|w| w.add(counts));
    }
    Ok(())
}

/// Complete material/pipeline identity for adjacent draws. Vertex/index ranges
/// and instances are checked by DrawOffsets. Mask commands are barriers; each
/// chunk fixes its attachments/scissor and cannot cross an upload or blend pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BatchKey {
    Color,
    Line,
    Gradient(crate::gpu_runtime::ResourceId),
    Texture(crate::gpu_runtime::ResourceId, TrivialBlend, bool),
    AlphaMask(crate::gpu_runtime::ResourceId),
}

pub struct CommandRenderer<'encoder> {
    pipelines: &'encoder Pipelines,
    mask_state: &'encoder mut MaskState,
    needs_stencil: bool,
    chunk_buffers: &'encoder ChunkBuffers,
    current_pipeline: Option<&'encoder crate::gpu_resources::RenderPipeline>,
    current_index_buffer: Option<&'encoder crate::gpu_resources::Buffer>,
    current_material: Option<&'encoder crate::gpu_resources::BindGroup>,
    work: Option<ruffle_render::profile::RenderWork>,
}

impl<'encoder> CommandRenderer<'encoder> {
    pub fn new(
        pipelines: &'encoder Pipelines,
        chunk_buffers: &'encoder ChunkBuffers,
        mask_state: &'encoder mut MaskState,
        needs_stencil: bool,
    ) -> Self {
        Self {
            pipelines,
            mask_state,
            needs_stencil,
            chunk_buffers,
            current_pipeline: None,
            current_index_buffer: None,
            current_material: None,
            work: ruffle_render::profile::enabled().then(|| ruffle_render::profile::RenderWork {
                mask_passes: u64::from(needs_stencil),
                ..Default::default()
            }),
        }
    }

    pub fn execute(
        &mut self,
        render_pass: &mut crate::gpu_resources::RenderPass<'encoder>,
        command: &'encoder DrawCommand,
    ) -> Result<(), RenderFailure> {
        if self.needs_stencil {
            self.mask_state.apply_stencil_reference(render_pass);
        }

        match command {
            DrawCommand::RenderTexture {
                _texture,
                binds,
                blend_mode,
                draw_offsets,
                render_stage3d,
                ..
            } => self.render_texture(
                render_pass,
                binds,
                *blend_mode,
                *render_stage3d,
                draw_offsets,
            )?,
            DrawCommand::RenderTextureBatch {
                binds,
                indirect,
                draw_offsets,
                ..
            } => {
                self.set_pipeline(
                    render_pass,
                    if self.needs_stencil {
                        self.pipelines
                            .bitmap_batch
                            .pipeline_for(self.mask_state.mode)?
                    } else {
                        self.pipelines.bitmap_batch.stencilless_pipeline()?
                    },
                );
                self.set_material(render_pass, binds);
                if let Some(indirect) = indirect {
                    self.bind_geometry(render_pass, draw_offsets);
                    indirect.encode(render_pass);
                    if let Some(work) = &mut self.work {
                        work.bitmap_indirect_draws += 1;
                        work.bitmap_cull_batches += u64::from(indirect.mode.culls());
                        work.bitmap_count_batches +=
                            u64::from(indirect.mode == crate::capabilities::BitmapDraw::Count);
                    }
                } else {
                    self.draw(render_pass, draw_offsets);
                }
            }
            DrawCommand::DrawColor { draw_offsets } => {
                self.draw_color(render_pass, draw_offsets)?
            }
            DrawCommand::DrawLine { draw_offsets } => self.draw_lines(render_pass, draw_offsets)?,
            DrawCommand::SetMask(mask_state) => self.set_mask_state(*mask_state),
            DrawCommand::RenderAlphaMask {
                maskee,
                mask,
                binds,
                draw_offsets,
            } => self.render_alpha_mask(render_pass, maskee, mask, binds, draw_offsets)?,
            DrawCommand::RenderGradient {
                bind_group,
                draw_offsets,
            } => self.draw_gradient(render_pass, bind_group, draw_offsets)?,
        }
        Ok(())
    }

    pub fn set_pipeline(
        &mut self,
        render_pass: &mut crate::gpu_resources::RenderPass<'encoder>,
        pipeline: &'encoder crate::gpu_resources::RenderPipeline,
    ) {
        if self.current_pipeline != Some(pipeline) {
            if let Some(work) = &mut self.work {
                work.pipeline_changes += 1;
            }
            render_pass.set_pipeline(pipeline);
            self.current_pipeline = Some(pipeline);
            self.current_material = None;
        }
    }

    fn set_material(
        &mut self,
        render_pass: &mut crate::gpu_resources::RenderPass<'encoder>,
        bind_group: &'encoder crate::gpu_resources::BindGroup,
    ) {
        if self.current_material != Some(bind_group) {
            if let Some(work) = &mut self.work {
                work.material_binds += 1;
            }
            render_pass.set_bind_group(2, bind_group.raw(), &[]);
            self.current_material = Some(bind_group);
        }
    }

    fn bind_geometry(
        &mut self,
        render_pass: &mut crate::gpu_resources::RenderPass<'encoder>,
        draw_offsets: &'encoder DrawOffsets,
    ) {
        if let Some(work) = &mut self.work {
            work.draw_calls += 1;
        }
        let vertex_offset = if draw_offsets.vertex_buffer.is_none() {
            self.chunk_buffers.vertex_offset
        } else {
            0
        };
        render_pass.set_vertex_buffer(
            0,
            draw_offsets
                .vertex_buffer
                .as_ref()
                .unwrap_or(&self.chunk_buffers.vertex_buffer)
                .slice(
                    draw_offsets.vertices.start + vertex_offset
                        ..draw_offsets.vertices.end + vertex_offset,
                ),
        );

        if self.current_index_buffer != Some(&draw_offsets.index_buffer) {
            render_pass.set_index_buffer(
                draw_offsets.index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            self.current_index_buffer = Some(&draw_offsets.index_buffer);
        }
    }

    fn draw(
        &mut self,
        render_pass: &mut crate::gpu_resources::RenderPass<'encoder>,
        draw_offsets: &'encoder DrawOffsets,
    ) {
        self.bind_geometry(render_pass, draw_offsets);
        render_pass.draw_indexed(
            draw_offsets.indices.clone(),
            0,
            draw_offsets.instances.clone(),
        );
    }

    pub fn render_texture(
        &mut self,
        render_pass: &mut crate::gpu_resources::RenderPass<'encoder>,
        bind_group: &'encoder crate::gpu_resources::BindGroup,
        blend_mode: TrivialBlend,
        render_stage3d: bool,
        draw_offsets: &'encoder DrawOffsets,
    ) -> Result<(), RenderFailure> {
        match (self.needs_stencil, render_stage3d) {
            (true, true) => {
                self.set_pipeline(
                    render_pass,
                    self.pipelines
                        .bitmap_opaque
                        .pipeline_for(crate::surface::mask::MaskMode::NoMask)?,
                );
            }
            (true, false) => {
                self.set_pipeline(
                    render_pass,
                    self.pipelines.bitmap[blend_mode].pipeline_for(self.mask_state.mode)?,
                );
            }
            (false, true) => {
                self.set_pipeline(
                    render_pass,
                    self.pipelines.bitmap_opaque.stencilless_pipeline()?,
                );
            }
            (false, false) => {
                self.set_pipeline(
                    render_pass,
                    self.pipelines.bitmap[blend_mode].stencilless_pipeline()?,
                );
            }
        }

        self.set_material(render_pass, bind_group);

        self.draw(render_pass, draw_offsets);
        Ok(())
    }

    pub fn draw_gradient(
        &mut self,
        render_pass: &mut crate::gpu_resources::RenderPass<'encoder>,
        bind_group: &'encoder crate::gpu_resources::BindGroup,
        draw_offsets: &'encoder DrawOffsets,
    ) -> Result<(), RenderFailure> {
        self.set_pipeline(
            render_pass,
            if self.needs_stencil {
                self.pipelines
                    .gradients
                    .pipeline_for(self.mask_state.mode)?
            } else {
                self.pipelines.gradients.stencilless_pipeline()?
            },
        );

        self.set_material(render_pass, bind_group);

        self.draw(render_pass, draw_offsets);
        Ok(())
    }

    pub fn render_alpha_mask(
        &mut self,
        render_pass: &mut crate::gpu_resources::RenderPass<'encoder>,
        _maskee: &PoolOrArcTexture,
        _mask: &PoolOrArcTexture,
        bind_group: &'encoder crate::gpu_resources::BindGroup,
        draw_offsets: &'encoder DrawOffsets,
    ) -> Result<(), RenderFailure> {
        if let Some(work) = &mut self.work {
            // Count the pass once, even if it contains several alpha masks or
            // also uses stencil masking.
            work.mask_passes = 1;
        }
        self.set_pipeline(
            render_pass,
            if self.needs_stencil {
                self.pipelines
                    .alpha_mask
                    .pipeline_for(self.mask_state.mode)?
            } else {
                self.pipelines.alpha_mask.stencilless_pipeline()?
            },
        );

        self.set_material(render_pass, bind_group);

        self.draw(render_pass, draw_offsets);
        Ok(())
    }

    pub fn draw_color(
        &mut self,
        render_pass: &mut crate::gpu_resources::RenderPass<'encoder>,
        draw_offsets: &'encoder DrawOffsets,
    ) -> Result<(), RenderFailure> {
        self.set_pipeline(
            render_pass,
            if self.needs_stencil {
                self.pipelines.color.pipeline_for(self.mask_state.mode)?
            } else {
                self.pipelines.color.stencilless_pipeline()?
            },
        );

        self.draw(render_pass, draw_offsets);
        Ok(())
    }

    pub fn draw_lines(
        &mut self,
        render_pass: &mut crate::gpu_resources::RenderPass<'encoder>,
        draw_offsets: &'encoder DrawOffsets,
    ) -> Result<(), RenderFailure> {
        self.set_pipeline(
            render_pass,
            if self.needs_stencil {
                self.pipelines.lines.pipeline_for(self.mask_state.mode)?
            } else {
                self.pipelines.lines.stencilless_pipeline()?
            },
        );

        self.draw(render_pass, draw_offsets);
        Ok(())
    }

    pub fn set_mask_state(&mut self, mask_state: MaskState) {
        *self.mask_state = mask_state;
    }
}

impl Drop for CommandRenderer<'_> {
    fn drop(&mut self) {
        if let Some(local) = self.work.take() {
            ruffle_render::profile::work(|work| work.add(local));
        }
    }
}
