mod bevel;
mod blur;
mod color_matrix;
mod convolution;
mod displacement_map;
mod drop_shadow;
mod glow;
mod gradient_glow;
pub(crate) mod graph;
mod shader;

use crate::buffer_pool::TexturePool;
use crate::descriptors::Descriptors;
use crate::filters::bevel::BevelFilter;
use crate::filters::blur::BlurFilter;
use crate::filters::color_matrix::ColorMatrixFilter;
use crate::filters::convolution::ConvolutionFilter;
use crate::filters::displacement_map::DisplacementMapFilter;
use crate::filters::drop_shadow::DropShadowFilter;
use crate::filters::glow::GlowFilter;
use crate::filters::gradient_glow::GradientGlowFilter;
use crate::filters::shader::ShaderFilter;
use crate::gpu_resources::StagingBelt;
use crate::surface::target::CommandTarget;
use bytemuck::{Pod, Zeroable};
use ruffle_render::filters::Filter;
use wgpu::vertex_attr_array;

#[derive(Default)]
struct FilterPipelines {
    rgba: crate::utils::SampleCountMap<std::sync::OnceLock<crate::gpu_resources::RenderPipeline>>,
    bgra: crate::utils::SampleCountMap<std::sync::OnceLock<crate::gpu_resources::RenderPipeline>>,
}

impl FilterPipelines {
    fn get_or_init(
        &self,
        format: wgpu::TextureFormat,
        samples: u32,
        build: impl FnOnce() -> Result<
            crate::gpu_resources::RenderPipeline,
            crate::gpu_runtime::RenderFailure,
        >,
    ) -> Result<&crate::gpu_resources::RenderPipeline, crate::gpu_runtime::RenderFailure> {
        let cache = match format {
            wgpu::TextureFormat::Rgba8Unorm => &self.rgba,
            wgpu::TextureFormat::Bgra8Unorm => &self.bgra,
            _ => {
                return Err(crate::gpu_runtime::RenderFailure::new(
                    crate::gpu_runtime::RenderFailureKind::Validation,
                    "unsupported filter attachment format",
                ));
            }
        };
        cache.get_or_init(samples, build)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FilterSource<'a> {
    pub texture: &'a crate::gpu_resources::Texture,
    pub view: &'a crate::gpu_resources::TextureView,
    pub point: (u32, u32),
    pub size: (u32, u32),
}

impl<'a> FilterSource<'a> {
    pub fn for_entire_texture(
        texture: &'a crate::gpu_resources::Texture,
        view: &'a crate::gpu_resources::TextureView,
    ) -> Self {
        Self {
            texture,
            view,
            point: (0, 0),
            size: (texture.width(), texture.height()),
        }
    }

    pub fn vertices(&self) -> [FilterVertex; 4] {
        let source_width = self.texture.width() as f32;
        let source_height = self.texture.height() as f32;
        let left = self.point.0;
        let top = self.point.1;
        let right = left + self.size.0;
        let bottom = top + self.size.1;
        [
            FilterVertex {
                position: [0.0, 0.0],
                uv: [left as f32 / source_width, top as f32 / source_height],
            },
            FilterVertex {
                position: [1.0, 0.0],
                uv: [right as f32 / source_width, top as f32 / source_height],
            },
            FilterVertex {
                position: [1.0, 1.0],
                uv: [right as f32 / source_width, bottom as f32 / source_height],
            },
            FilterVertex {
                position: [0.0, 1.0],
                uv: [left as f32 / source_width, bottom as f32 / source_height],
            },
        ]
    }

    pub fn vertices_with_blur_offset(&self, blur_offset: (f32, f32)) -> [FilterVertexWithBlur; 4] {
        let source_width = self.texture.width() as f32;
        let source_height = self.texture.height() as f32;
        let source_left = self.point.0;
        let source_top = self.point.1;
        let source_right = source_left + self.size.0;
        let source_bottom = source_top + self.size.1;
        [
            FilterVertexWithBlur {
                position: [0.0, 0.0],
                source_uv: [
                    source_left as f32 / source_width,
                    source_top as f32 / source_height,
                ],
                blur_uv: [
                    (source_left as f32 + blur_offset.0) / source_width,
                    (source_top as f32 + blur_offset.1) / source_height,
                ],
            },
            FilterVertexWithBlur {
                position: [1.0, 0.0],
                source_uv: [
                    source_right as f32 / source_width,
                    source_top as f32 / source_height,
                ],
                blur_uv: [
                    (source_right as f32 + blur_offset.0) / source_width,
                    (source_top as f32 + blur_offset.1) / source_height,
                ],
            },
            FilterVertexWithBlur {
                position: [1.0, 1.0],
                source_uv: [
                    source_right as f32 / source_width,
                    source_bottom as f32 / source_height,
                ],
                blur_uv: [
                    (source_right as f32 + blur_offset.0) / source_width,
                    (source_bottom as f32 + blur_offset.1) / source_height,
                ],
            },
            FilterVertexWithBlur {
                position: [0.0, 1.0],
                source_uv: [
                    source_left as f32 / source_width,
                    source_bottom as f32 / source_height,
                ],
                blur_uv: [
                    (source_left as f32 + blur_offset.0) / source_width,
                    (source_bottom as f32 + blur_offset.1) / source_height,
                ],
            },
        ]
    }

    pub fn vertices_with_highlight_and_shadow(
        &self,
        blur_offset: (f32, f32),
    ) -> [FilterVertexWithDoubleBlur; 4] {
        let source_width = self.texture.width() as f32;
        let source_height = self.texture.height() as f32;
        let source_left = self.point.0 as f32;
        let source_top = self.point.1 as f32;
        let source_right = (self.point.0 + self.size.0) as f32;
        let source_bottom = (self.point.1 + self.size.1) as f32;
        [
            FilterVertexWithDoubleBlur {
                position: [0.0, 0.0],
                source_uv: [source_left / source_width, source_top / source_height],
                blur_uv_left: [
                    (source_left + blur_offset.0) / source_width,
                    (source_top + blur_offset.1) / source_height,
                ],
                blur_uv_right: [
                    (source_left - blur_offset.0) / source_width,
                    (source_top - blur_offset.1) / source_height,
                ],
            },
            FilterVertexWithDoubleBlur {
                position: [1.0, 0.0],
                source_uv: [source_right / source_width, source_top / source_height],
                blur_uv_left: [
                    (source_right + blur_offset.0) / source_width,
                    (source_top + blur_offset.1) / source_height,
                ],
                blur_uv_right: [
                    (source_right - blur_offset.0) / source_width,
                    (source_top - blur_offset.1) / source_height,
                ],
            },
            FilterVertexWithDoubleBlur {
                position: [1.0, 1.0],
                source_uv: [source_right / source_width, source_bottom / source_height],
                blur_uv_left: [
                    (source_right + blur_offset.0) / source_width,
                    (source_bottom + blur_offset.1) / source_height,
                ],
                blur_uv_right: [
                    (source_right - blur_offset.0) / source_width,
                    (source_bottom - blur_offset.1) / source_height,
                ],
            },
            FilterVertexWithDoubleBlur {
                position: [0.0, 1.0],
                source_uv: [source_left / source_width, source_bottom / source_height],
                blur_uv_left: [
                    (source_left + blur_offset.0) / source_width,
                    (source_bottom + blur_offset.1) / source_height,
                ],
                blur_uv_right: [
                    (source_left - blur_offset.0) / source_width,
                    (source_bottom - blur_offset.1) / source_height,
                ],
            },
        ]
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum GradientKind {
    Glow = 0,
    Bevel = 1,
}

impl GradientKind {
    pub(crate) fn is_identity(self, filter: &swf::GradientFilter) -> bool {
        match self {
            Self::Glow => filter.colors.is_empty() || filter.strength <= swf::Fixed8::ZERO,
            Self::Bevel => {
                filter.colors.is_empty()
                    && !filter.is_knockout()
                    && filter
                        .flags
                        .contains(swf::GradientFilterFlags::COMPOSITE_SOURCE)
            }
        }
    }
}

pub struct Filters {
    pub blur: BlurFilter,
    pub color_matrix: ColorMatrixFilter,
    convolution: ConvolutionFilter,
    pub shader: ShaderFilter,
    pub glow: GlowFilter,
    pub gradient_glow: GradientGlowFilter,
    pub bevel: BevelFilter,
    pub displacement_map: DisplacementMapFilter,
}

impl Filters {
    pub fn new(
        device: &crate::gpu_resources::Device,
    ) -> Result<Self, crate::gpu_runtime::RenderFailure> {
        Ok(Self {
            blur: BlurFilter::new(device)?,
            color_matrix: ColorMatrixFilter::new(device)?,
            convolution: ConvolutionFilter::new(device)?,
            shader: ShaderFilter::new(),
            glow: GlowFilter::new(device)?,
            gradient_glow: GradientGlowFilter::new(device)?,
            bevel: BevelFilter::new(device)?,
            displacement_map: DisplacementMapFilter::new(device)?,
        })
    }

    pub fn apply_chain(
        &self,
        descriptors: &Descriptors,
        scope: &mut crate::gpu_profiler::Scope<'_, crate::gpu_resources::CommandEncoder>,
        texture_pool: &mut TexturePool,
        staging_belt: &mut StagingBelt,
        source: FilterSource,
        filters: &[Filter],
        plan: crate::pass_graph::FilterPasses,
        destination: Option<FilterSource<'_>>,
    ) -> Result<Option<CommandTarget>, crate::gpu_runtime::RenderFailure> {
        use crate::gpu_runtime::{RenderFailure, RenderFailureKind};
        use crate::pass_graph::PassKind;
        let invalid = || {
            RenderFailure::new(
                RenderFailureKind::Validation,
                "invalid filter pass dependency",
            )
        };
        let mut version = plan.source;
        let mut current: Option<CommandTarget> = None;
        let mut premultiplied = false;
        let pass_count = plan.passes.len();
        for (index, pass) in plan.passes.into_iter().enumerate() {
            let (PassKind::Filter { step, compute }, [output]) =
                (pass.kind, pass.writes.as_slice())
            else {
                return Err(invalid());
            };
            if pass.reads.first() != Some(&version)
                || pass.dependencies.iter().any(|&d| d >= plan.first + index)
            {
                return Err(invalid());
            }
            // A no-op aliases its input at this logical output version.
            version = *output;
            scope.flush_uploads_if_needed(staging_belt.page_size())?;
            let label = step.profile_label(filters);
            let _pass = descriptors
                .device
                .pass_context(crate::gpu_resources::diagnostics::Context::Filter(label));
            let draw_encoder = &mut scope.scope(label);
            let input = current.as_ref().map(|target| {
                FilterSource::for_entire_texture(target.color_texture(), target.color_view())
            });
            let destination = destination.filter(|_| !compute && index + 1 == pass_count);
            let source = input.as_ref().unwrap_or(&source);
            let matrix_output = match &step {
                graph::FilterStep::Pointwise(_) => true,
                graph::FilterStep::Single(i) => matches!(filters[*i], Filter::ColorMatrixFilter(_)),
            };
            let next = match step {
                graph::FilterStep::Pointwise(range) => {
                    let matrices: smallvec::SmallVec<[swf::ColorMatrixFilter; 16]> = filters[range]
                        .iter()
                        .filter_map(|filter| {
                            if let Filter::ColorMatrixFilter(matrix) = filter {
                                Some(matrix.clone())
                            } else {
                                None
                            }
                        })
                        .collect();
                    if premultiplied
                        && matrices
                            .iter()
                            .all(|matrix| *matrix == swf::ColorMatrixFilter::default())
                    {
                        continue;
                    }
                    Some(
                        self.color_matrix.apply_chain(
                            descriptors,
                            texture_pool,
                            draw_encoder,
                            staging_belt,
                            source,
                            &matrices,
                            compute,
                            destination
                                .filter(|output| output.texture != source.texture)
                                .map(|output| output.view),
                        )?,
                    )
                }
                graph::FilterStep::Single(index) => self.apply(
                    descriptors,
                    draw_encoder,
                    texture_pool,
                    staging_belt,
                    *source,
                    filters[index].clone(),
                    premultiplied,
                    false,
                    destination,
                )?,
            };
            // Encoding consumed the old input; returning its exact-key pool
            // entry here makes the following operation a legal ping-pong reuse.
            if next.is_some() {
                current = next;
                premultiplied = matrix_output;
            }
        }
        Ok(current)
    }

    pub fn apply(
        &self,
        descriptors: &Descriptors,
        draw_encoder: &mut crate::gpu_resources::CommandEncoder,
        texture_pool: &mut TexturePool,
        staging_belt: &mut StagingBelt,
        source: FilterSource,
        filter: Filter,
        can_skip_identity: bool,
        opaque_target: bool,
        destination: Option<FilterSource<'_>>,
    ) -> Result<Option<CommandTarget>, crate::gpu_runtime::RenderFailure> {
        let _pass =
            descriptors
                .device
                .pass_context(crate::gpu_resources::diagnostics::Context::Filter(
                    graph::filter_label(&filter),
                ));
        let target = match filter {
            Filter::ColorMatrixFilter(filter)
                if can_skip_identity && filter == Default::default() =>
            {
                None
            }
            Filter::ColorMatrixFilter(filter) => Some(descriptors.filters.color_matrix.apply(
                descriptors,
                texture_pool,
                draw_encoder,
                staging_belt,
                &source,
                &filter,
            )?),
            Filter::ConvolutionFilter(filter) => {
                if filter.num_matrix_cols == 0
                    || filter.num_matrix_rows == 0
                    || source.size.0 == 0
                    || source.size.1 == 0
                {
                    return Ok(None);
                }
                Some(self.convolution.apply(
                    descriptors,
                    texture_pool,
                    draw_encoder,
                    staging_belt,
                    &source,
                    &filter,
                    opaque_target,
                    destination,
                )?)
            }
            Filter::BlurFilter(filter) => descriptors.filters.blur.apply(
                descriptors,
                texture_pool,
                draw_encoder,
                staging_belt,
                &source,
                &filter,
                destination,
            )?,
            Filter::ShaderFilter(shader) => Some(descriptors.filters.shader.apply(
                descriptors,
                texture_pool,
                draw_encoder,
                &source,
                shader,
            )?),
            Filter::GlowFilter(filter) => Some(descriptors.filters.glow.apply(
                descriptors,
                texture_pool,
                draw_encoder,
                staging_belt,
                &source,
                &filter,
                &self.blur,
                (0.0, 0.0),
                destination,
            )?),
            Filter::GradientGlowFilter(ref params) | Filter::GradientBevelFilter(ref params) => {
                let kind = if matches!(filter, Filter::GradientBevelFilter(_)) {
                    GradientKind::Bevel
                } else {
                    GradientKind::Glow
                };
                if !opaque_target && kind.is_identity(params) {
                    None
                } else {
                    Some(self.gradient_glow.apply(
                        descriptors,
                        texture_pool,
                        draw_encoder,
                        staging_belt,
                        &source,
                        [source.point.0 as i32, source.point.1 as i32],
                        source.size,
                        params,
                        kind,
                        opaque_target,
                        destination,
                    )?)
                }
            }
            Filter::DropShadowFilter(filter) => Some(DropShadowFilter::apply(
                descriptors,
                texture_pool,
                draw_encoder,
                staging_belt,
                &source,
                &filter,
                &self.blur,
                &self.glow,
                destination,
            )?),
            Filter::BevelFilter(filter) => Some(descriptors.filters.bevel.apply(
                descriptors,
                texture_pool,
                draw_encoder,
                staging_belt,
                &source,
                &filter,
                &self.blur,
            )?),
            Filter::DisplacementMapFilter(filter) => descriptors.filters.displacement_map.apply(
                descriptors,
                texture_pool,
                draw_encoder,
                staging_belt,
                &source,
                &filter,
            )?,
        };

        let target = match target {
            None if !can_skip_identity => Some(self.color_matrix.apply(
                descriptors,
                texture_pool,
                draw_encoder,
                staging_belt,
                &source,
                &Default::default(),
            )?),
            target => target,
        };
        if let Some(target) = &target {
            target.ensure_cleared(draw_encoder);
        }
        Ok(target)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FilterVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
}

pub const VERTEX_BUFFERS_DESCRIPTION_FILTERS: [Option<wgpu::VertexBufferLayout>; 1] =
    [Some(wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<FilterVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &vertex_attr_array![
            0 => Float32x2,
            1 => Float32x2,
        ],
    })];

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FilterVertexWithBlur {
    pub position: [f32; 2],
    pub source_uv: [f32; 2],
    pub blur_uv: [f32; 2],
}

pub const VERTEX_BUFFERS_DESCRIPTION_FILTERS_WITH_BLUR: [Option<wgpu::VertexBufferLayout>; 1] =
    [Some(wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<FilterVertexWithBlur>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &vertex_attr_array![
            0 => Float32x2,
            1 => Float32x2,
            2 => Float32x2,
        ],
    })];

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FilterVertexWithDoubleBlur {
    pub position: [f32; 2],
    pub source_uv: [f32; 2],
    pub blur_uv_left: [f32; 2],
    pub blur_uv_right: [f32; 2],
}

pub const VERTEX_BUFFERS_DESCRIPTION_FILTERS_WITH_DOUBLE_BLUR: [Option<wgpu::VertexBufferLayout>;
    1] = [Some(wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<FilterVertexWithDoubleBlur>() as u64,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &vertex_attr_array![
        0 => Float32x2,
        1 => Float32x2,
        2 => Float32x2,
        3 => Float32x2,
    ],
})];
