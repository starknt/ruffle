mod batches;
pub mod buffers;
mod chunk;
mod commands;
pub mod mask;
pub mod target;
#[cfg(test)]
mod tests;

use crate::backend::RenderTargetMode;
use crate::blend::ComplexBlend;
use crate::buffer_pool::TexturePool;
use crate::filters::FilterSource;
use crate::gpu_profiler::Scope;
use crate::graph_encoder::PlannedList;
use crate::mesh::Mesh;
use crate::pixel_bender::{ShaderMode, run_pixelbender_shader_impl};
use crate::surface::buffers::ChunkBuffers;
pub use crate::surface::commands::LayerRef;
use crate::surface::commands::chunk_blends;
use crate::utils::run_copy_pipeline;
use crate::utils::supported_sample_count;
use crate::{Descriptors, Pipelines};
use batches::CommandBatches;
use chunk::{Chunk, CommandRenderer};
use ruffle_render::pixel_bender_support::{ImageInputTexture, PixelBenderShaderArgument};
use ruffle_render::quality::StageQuality;
use std::sync::Arc;
use target::CommandTarget;
use tracing::instrument;

use crate::surface::mask::MaskState;
use chunk::ChunkBlendMode;

#[derive(Debug)]
pub struct Surface {
    size: wgpu::Extent3d,
    origin: [u32; 2],
    quality: StageQuality,
    sample_count: u32,
    pipelines: Arc<Pipelines>,
    format: wgpu::TextureFormat,
}

impl Surface {
    pub fn with_origin(mut self, origin: [u32; 2]) -> Self {
        self.origin = origin;
        self
    }
    pub fn new(
        descriptors: &Descriptors,
        quality: StageQuality,
        width: u32,
        height: u32,
        frame_buffer_format: wgpu::TextureFormat,
    ) -> Result<Self, crate::gpu_runtime::RenderFailure> {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let sample_count = supported_sample_count(
            &descriptors.adapter,
            quality.sample_count(),
            frame_buffer_format,
        );
        let pipelines = descriptors.pipelines(sample_count, frame_buffer_format)?;
        Ok(Self {
            origin: [0, 0],
            size,
            quality,
            sample_count,
            pipelines,
            format: frame_buffer_format,
        })
    }

    #[expect(clippy::too_many_arguments)]
    #[instrument(level = "debug", skip_all)]
    pub fn draw_graph_and_copy_to<'encoder, 'global: 'encoder>(
        &self,
        frame_view: &crate::gpu_resources::TextureView,
        render_target_mode: RenderTargetMode,
        descriptors: &'global Descriptors,
        staging_belt: &'global mut crate::gpu_resources::StagingBelt,
        chunk_buffers: &'global mut ChunkBuffers,
        draw_encoder: &'encoder mut Scope<'global, crate::gpu_resources::CommandEncoder>,
        meshes: &'global Vec<Mesh>,
        graph: crate::graph_encoder::GraphEncoder,
        layer: LayerRef<'encoder>,
        texture_pool: &'global mut TexturePool,
    ) -> Result<(), crate::gpu_runtime::RenderFailure> {
        let direct = matches!(render_target_mode, RenderTargetMode::FreshWithColor(_))
            && (graph.graph.direct_attachment() || {
                // Complex compositing copies the destination before sampling it.
                // Host texture targets can supply that copy directly; presentation
                // surfaces with attachment-only usage still need the intermediate.
                let texture = frame_view.texture_handle()?;
                texture.usage().contains(wgpu::TextureUsages::COPY_SRC)
                    && texture.format() == self.format
            });
        let render_target_mode = match render_target_mode {
            RenderTargetMode::FreshWithColor(color) if direct => {
                RenderTargetMode::ExistingWithView(frame_view.clone(), color)
            }
            mode => mode,
        };
        let commands = graph.into_commands()?;
        let target = self.draw_commands(
            render_target_mode,
            descriptors,
            meshes,
            commands,
            staging_belt,
            chunk_buffers,
            draw_encoder,
            layer,
            texture_pool,
        )?;

        if direct {
            return Ok(());
        }
        run_copy_pipeline(
            descriptors,
            self.format,
            frame_view,
            target.color_view(),
            1,
            draw_encoder,
        )?;

        Ok(())
    }

    #[expect(clippy::too_many_arguments)]
    #[instrument(level = "debug", skip_all)]
    pub fn draw_commands<'encoder, 'global: 'encoder>(
        &self,
        render_target_mode: RenderTargetMode,
        descriptors: &'encoder Descriptors,
        meshes: &'encoder Vec<Mesh>,
        commands: PlannedList,
        staging_belt: &'encoder mut crate::gpu_resources::StagingBelt,
        chunk_buffers: &'encoder mut ChunkBuffers,
        draw_encoder: &'encoder mut Scope<'global, crate::gpu_resources::CommandEncoder>,
        nearest_layer: LayerRef<'encoder>,
        texture_pool: &'encoder mut TexturePool,
    ) -> Result<CommandTarget, crate::gpu_runtime::RenderFailure> {
        let target = CommandTarget::new(
            descriptors,
            texture_pool,
            self.size,
            self.format,
            self.sample_count,
            render_target_mode,
            draw_encoder,
        )?;

        let mut mask_state = MaskState::empty();
        for commands in
            CommandBatches::new(commands, descriptors.runtime.config().transient_watermark)
        {
            let chunks = chunk_blends(
                commands,
                descriptors,
                staging_belt,
                chunk_buffers,
                draw_encoder,
                meshes,
                self.quality,
                target.width(),
                target.height(),
                self.origin,
                target.color_texture(),
                match nearest_layer {
                    LayerRef::Current => LayerRef::Parent(&target),
                    layer => layer,
                },
                texture_pool,
                mask_state,
            )?;

            for chunk in chunks {
                draw_encoder.flush_uploads_if_needed(staging_belt.page_size())?;
                match chunk {
                    Chunk::Draw {
                        mut chunk,
                        needs_stencil,
                        chunk_buffer_builder,
                    } => {
                        chunk::batch_bitmap_textures(&mut chunk, descriptors)?;
                        target.restore_multisample_regions(descriptors, draw_encoder)?;
                        chunk_buffer_builder.copy_to(
                            staging_belt,
                            descriptors,
                            draw_encoder,
                            chunk_buffers,
                        )?;
                        ruffle_render::profile::work(|w| {
                            w.render_passes += 1;
                            w.pass_pixels += u64::from(target.width()) * u64::from(target.height());
                        });
                        let mut render_pass = draw_encoder.scoped_render_pass(
                            format!(
                                "Chunked draw calls {}",
                                if needs_stencil {
                                    "(with stencil)"
                                } else {
                                    "(Stencilless)"
                                }
                            ),
                            wgpu::RenderPassDescriptor {
                                color_attachments: &[target.color_attachments()],
                                depth_stencil_attachment: if needs_stencil {
                                    target.stencil_attachment(descriptors, texture_pool)?
                                } else {
                                    None
                                },
                                ..Default::default()
                            },
                        );
                        render_pass.set_bind_group(0, (target.globals().bind_group()).raw(), &[]);
                        render_pass.set_bind_group(
                            1,
                            chunk_buffers.bind_group.raw(),
                            &[chunk_buffers.transform_offset],
                        );
                        let mut renderer = CommandRenderer::new(
                            &self.pipelines,
                            chunk_buffers,
                            &mut mask_state,
                            needs_stencil,
                        );

                        for command in &chunk {
                            renderer.execute(&mut render_pass.scope(command.name()), command);
                        }
                    }
                    Chunk::Blend {
                        texture,
                        blend_mode: ChunkBlendMode::Shader(shader),
                        needs_stencil,
                        region: _,
                    } => {
                        let parent_blend_buffer =
                            target.update_blend_buffer(descriptors, texture_pool, draw_encoder)?;
                        target.restore_multisample_regions(descriptors, draw_encoder)?;
                        run_pixelbender_shader_impl(
                            descriptors,
                            shader,
                            ShaderMode::Filter,
                            &[
                                PixelBenderShaderArgument::ImageInput {
                                    index: 0,
                                    channels: 0xFF,
                                    name: "background".to_string(),
                                    texture: Some(ImageInputTexture::TextureRef(
                                        parent_blend_buffer.texture(),
                                    )),
                                },
                                PixelBenderShaderArgument::ImageInput {
                                    index: 1,
                                    channels: 0xff,
                                    name: "foreground".to_string(),
                                    texture: Some(ImageInputTexture::TextureRef(texture.texture())),
                                },
                            ],
                            parent_blend_buffer.texture(),
                            draw_encoder,
                            target.color_attachments(),
                            if needs_stencil {
                                target
                                    .stencil_attachment(descriptors, texture_pool)?
                                    .map(|attachment| (mask_state, attachment))
                            } else {
                                None
                            },
                            target.sample_count(),
                            &FilterSource::for_entire_texture(texture.texture()),
                        )
                        .map_err(ruffle_render::error::Error::into_render_failure)?;
                    }
                    Chunk::Blend {
                        texture,
                        blend_mode: ChunkBlendMode::Complex(blend_mode),
                        needs_stencil,
                        region,
                    } => {
                        let parent = match blend_mode {
                            ComplexBlend::Alpha | ComplexBlend::Erase => {
                                match nearest_layer {
                                    LayerRef::None => {
                                        // An Alpha or Erase with no Layer above it should be ignored
                                        continue;
                                    }
                                    LayerRef::Current => &target,
                                    LayerRef::Parent(layer) => layer,
                                }
                            }
                            _ => &target,
                        };

                        let bounded_buffer;
                        let parent_blend_buffer = if region.x == 0
                            && region.y == 0
                            && region.width == parent.width()
                            && region.height == parent.height()
                        {
                            parent.update_blend_buffer(descriptors, texture_pool, draw_encoder)?
                        } else {
                            bounded_buffer = parent.copy_blend_region(
                                descriptors,
                                texture_pool,
                                draw_encoder,
                                region,
                            )?;
                            &bounded_buffer
                        };

                        let blend_bind_group =
                            descriptors.cached_bind_group(&wgpu::BindGroupDescriptor {
                                label: create_debug_label!(
                                    "Complex blend binds {:?} {}",
                                    blend_mode,
                                    if needs_stencil {
                                        "(with stencil)"
                                    } else {
                                        "(Stencilless)"
                                    }
                                )
                                .as_deref(),
                                layout: &descriptors.bind_layouts.blend,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: wgpu::BindingResource::TextureView(
                                            parent_blend_buffer.view(),
                                        ),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: wgpu::BindingResource::TextureView(
                                            texture.view(),
                                        ),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: wgpu::BindingResource::Sampler(
                                            descriptors.bitmap_samplers.get_sampler(false, false),
                                        ),
                                    },
                                ],
                            })?;

                        let resolved = !needs_stencil && target.can_composite_resolved();
                        let pipelines = if resolved {
                            target.ensure_cleared(draw_encoder);
                            descriptors.pipelines(1, self.format)?
                        } else {
                            target.restore_multisample_regions(descriptors, draw_encoder)?;
                            self.pipelines.clone()
                        };
                        ruffle_render::profile::work(|w| {
                            w.render_passes += 1;
                            w.blend_passes += 1;
                            w.mask_passes += u64::from(needs_stencil);
                            w.draw_calls += 1;
                            w.pipeline_changes += 1;
                            w.material_binds += 2;
                            w.pass_pixels += u64::from(target.width()) * u64::from(target.height());
                        });
                        let mut render_pass =
                            draw_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: create_debug_label!(
                                    "Complex blend {:?} {}",
                                    blend_mode,
                                    if needs_stencil {
                                        "(with stencil)"
                                    } else {
                                        "(Stencilless)"
                                    }
                                )
                                .as_deref(),
                                color_attachments: &[if resolved {
                                    target.resolved_attachment(region)
                                } else {
                                    target.color_attachments()
                                }],
                                depth_stencil_attachment: if needs_stencil {
                                    target.stencil_attachment(descriptors, texture_pool)?
                                } else {
                                    None
                                },
                                ..Default::default()
                            });
                        render_pass.set_bind_group(0, (target.globals().bind_group()).raw(), &[]);

                        if needs_stencil {
                            mask_state.apply_stencil_reference(&mut render_pass);
                            render_pass.set_pipeline(
                                pipelines.complex_blends[blend_mode].pipeline_for(mask_state.mode),
                            );
                        } else {
                            render_pass.set_pipeline(
                                pipelines.complex_blends[blend_mode].stencilless_pipeline(),
                            );
                        }

                        render_pass.set_viewport(
                            region.x as f32,
                            region.y as f32,
                            region.width as f32,
                            region.height as f32,
                            0.0,
                            1.0,
                        );
                        render_pass.set_scissor_rect(
                            region.x,
                            region.y,
                            region.width,
                            region.height,
                        );
                        render_pass.set_bind_group(2, (&blend_bind_group).raw(), &[]);

                        render_pass.set_vertex_buffer(
                            0,
                            descriptors
                                .quad
                                .vertex_buffer
                                .slice(descriptors.quad.vertices_pos.clone()),
                        );
                        render_pass.set_index_buffer(
                            descriptors.quad.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );

                        render_pass.draw_indexed(descriptors.quad.indices_quad.clone(), 0, 0..1);
                    }
                }
            }
        }
        // If nothing happened, ensure it's cleared so we don't operate on garbage data
        target.ensure_cleared(draw_encoder);

        Ok(target)
    }

    pub fn quality(&self) -> StageQuality {
        self.quality
    }

    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    pub fn size(&self) -> wgpu::Extent3d {
        self.size
    }
}
