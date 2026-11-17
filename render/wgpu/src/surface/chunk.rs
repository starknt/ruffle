use crate::blend::{ComplexBlend, TrivialBlend};
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
        blend_mode: TrivialBlend,
        render_stage3d: bool,
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
    pub fn name(&self) -> &'static str {
        match self {
            DrawCommand::RenderTexture { .. } => "render texture",
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
            Self::SetMask(_) => return None,
        })
    }
    fn offsets_mut(&mut self) -> Option<&mut DrawOffsets> {
        match self {
            Self::DrawColor { draw_offsets }
            | Self::DrawLine { draw_offsets }
            | Self::RenderGradient { draw_offsets, .. }
            | Self::RenderTexture { draw_offsets, .. }
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
            | Self::RenderAlphaMask { draw_offsets, .. } => draw_offsets,
            Self::SetMask(_) => return false,
        };
        self.offsets_mut()
            .is_some_and(|current| current.try_merge(offsets, primitive_size))
    }
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
    ) {
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
            } => self.render_texture(
                render_pass,
                binds,
                *blend_mode,
                *render_stage3d,
                draw_offsets,
            ),
            DrawCommand::DrawColor { draw_offsets } => self.draw_color(render_pass, draw_offsets),
            DrawCommand::DrawLine { draw_offsets } => self.draw_lines(render_pass, draw_offsets),
            DrawCommand::SetMask(mask_state) => self.set_mask_state(*mask_state),
            DrawCommand::RenderAlphaMask {
                maskee,
                mask,
                binds,
                draw_offsets,
            } => self.render_alpha_mask(render_pass, maskee, mask, binds, draw_offsets),
            DrawCommand::RenderGradient {
                bind_group,
                draw_offsets,
            } => self.draw_gradient(render_pass, bind_group, draw_offsets),
        }
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

    pub fn draw(
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
    ) {
        match (self.needs_stencil, render_stage3d) {
            (true, true) => {
                self.set_pipeline(render_pass, &self.pipelines.bitmap_opaque_dummy_stencil);
            }
            (true, false) => {
                self.set_pipeline(
                    render_pass,
                    self.pipelines.bitmap[blend_mode].pipeline_for(self.mask_state.mode),
                );
            }
            (false, true) => {
                self.set_pipeline(render_pass, &self.pipelines.bitmap_opaque);
            }
            (false, false) => {
                self.set_pipeline(
                    render_pass,
                    self.pipelines.bitmap[blend_mode].stencilless_pipeline(),
                );
            }
        }

        self.set_material(render_pass, bind_group);

        self.draw(render_pass, draw_offsets);
    }

    pub fn draw_gradient(
        &mut self,
        render_pass: &mut crate::gpu_resources::RenderPass<'encoder>,
        bind_group: &'encoder crate::gpu_resources::BindGroup,
        draw_offsets: &'encoder DrawOffsets,
    ) {
        self.set_pipeline(
            render_pass,
            if self.needs_stencil {
                self.pipelines.gradients.pipeline_for(self.mask_state.mode)
            } else {
                self.pipelines.gradients.stencilless_pipeline()
            },
        );

        self.set_material(render_pass, bind_group);

        self.draw(render_pass, draw_offsets);
    }

    pub fn render_alpha_mask(
        &mut self,
        render_pass: &mut crate::gpu_resources::RenderPass<'encoder>,
        _maskee: &PoolOrArcTexture,
        _mask: &PoolOrArcTexture,
        bind_group: &'encoder crate::gpu_resources::BindGroup,
        draw_offsets: &'encoder DrawOffsets,
    ) {
        if let Some(work) = &mut self.work {
            // Count the pass once, even if it contains several alpha masks or
            // also uses stencil masking.
            work.mask_passes = 1;
        }
        self.set_pipeline(
            render_pass,
            if self.needs_stencil {
                self.pipelines.alpha_mask.pipeline_for(self.mask_state.mode)
            } else {
                self.pipelines.alpha_mask.stencilless_pipeline()
            },
        );

        self.set_material(render_pass, bind_group);

        self.draw(render_pass, draw_offsets);
    }

    pub fn draw_color(
        &mut self,
        render_pass: &mut crate::gpu_resources::RenderPass<'encoder>,
        draw_offsets: &'encoder DrawOffsets,
    ) {
        self.set_pipeline(
            render_pass,
            if self.needs_stencil {
                self.pipelines.color.pipeline_for(self.mask_state.mode)
            } else {
                self.pipelines.color.stencilless_pipeline()
            },
        );

        self.draw(render_pass, draw_offsets);
    }

    pub fn draw_lines(
        &mut self,
        render_pass: &mut crate::gpu_resources::RenderPass<'encoder>,
        draw_offsets: &'encoder DrawOffsets,
    ) {
        self.set_pipeline(
            render_pass,
            if self.needs_stencil {
                self.pipelines.lines.pipeline_for(self.mask_state.mode)
            } else {
                self.pipelines.lines.stencilless_pipeline()
            },
        );

        self.draw(render_pass, draw_offsets);
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
