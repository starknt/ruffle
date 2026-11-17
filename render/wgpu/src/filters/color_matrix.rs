use crate::backend::RenderTargetMode;
use crate::buffer_pool::TexturePool;
use crate::descriptors::Descriptors;
use crate::filters::{FilterSource, FilterVertex, VERTEX_BUFFERS_DESCRIPTION_FILTERS};
use crate::gpu_resources::StagingBelt;
use crate::surface::target::CommandTarget;
use crate::utils::SampleCountMap;
use std::sync::OnceLock;
use swf::ColorMatrixFilter as ColorMatrixFilterArgs;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatrixChain {
    channels: [u32; 4],
    matrix: [f32; 20],
}

pub struct ColorMatrixFilter {
    bind_group_layout: crate::gpu_resources::BindGroupLayout,
    pipeline_layout: crate::gpu_resources::PipelineLayout,
    vertex_buffer: crate::gpu_resources::Buffer,
    uniform_buffer: crate::gpu_resources::Buffer,
    vertices_size: wgpu::BufferSize,
    uniform_size: wgpu::BufferSize,
    pipelines: SampleCountMap<OnceLock<crate::gpu_resources::RenderPipeline>>,
}

impl ColorMatrixFilter {
    pub fn new(
        device: &crate::gpu_resources::Device,
    ) -> Result<Self, crate::gpu_runtime::RenderFailure> {
        let uniform_size = std::mem::size_of::<MatrixChain>() as u64;

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(uniform_size),
                        },
                        count: None,
                    },
                ],
                label: create_debug_label!("Color matrix filter binds").as_deref(),
            })?;

        let vertices_size = std::mem::size_of::<[FilterVertex; 4]>() as u64;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: vertices_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })?;

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        })?;

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: uniform_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })?;

        Ok(Self {
            pipelines: Default::default(),
            pipeline_layout,
            vertex_buffer,
            uniform_buffer,
            bind_group_layout,
            vertices_size: wgpu::BufferSize::new(vertices_size).expect("Definitely not zero."),
            uniform_size: wgpu::BufferSize::new(uniform_size).expect("Definitely not zero."),
        })
    }

    fn pipeline(
        &self,
        descriptors: &Descriptors,
        msaa_sample_count: u32,
    ) -> Result<&crate::gpu_resources::RenderPipeline, crate::gpu_runtime::RenderFailure> {
        self.pipelines.get_or_init(msaa_sample_count, || {
            let label = create_debug_label!("Color Matrix Filter ({} msaa)", msaa_sample_count);
            descriptors
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: label.as_deref(),
                    layout: Some(&self.pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &descriptors.shaders.color_matrix_filter,
                        entry_point: Some("main_vertex"),
                        buffers: &VERTEX_BUFFERS_DESCRIPTION_FILTERS,
                        compilation_options: Default::default(),
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: None,
                        polygon_mode: wgpu::PolygonMode::default(),
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState {
                        count: msaa_sample_count,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &descriptors.shaders.color_matrix_filter,
                        entry_point: Some("main_fragment"),
                        targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
                        compilation_options: Default::default(),
                    }),
                    multiview_mask: None,
                    cache: None,
                })
        })
    }

    pub fn apply(
        &self,
        descriptors: &Descriptors,
        texture_pool: &mut TexturePool,
        draw_encoder: &mut crate::gpu_resources::CommandEncoder,
        staging_belt: &mut StagingBelt,
        source: &FilterSource,
        filter: &ColorMatrixFilterArgs,
    ) -> Result<CommandTarget, crate::gpu_runtime::RenderFailure> {
        self.apply_chain(
            descriptors,
            texture_pool,
            draw_encoder,
            staging_belt,
            source,
            std::slice::from_ref(filter),
        )
    }

    pub fn apply_chain(
        &self,
        descriptors: &Descriptors,
        texture_pool: &mut TexturePool,
        draw_encoder: &mut crate::gpu_resources::CommandEncoder,
        staging_belt: &mut StagingBelt,
        source: &FilterSource,
        filters: &[ColorMatrixFilterArgs],
    ) -> Result<CommandTarget, crate::gpu_runtime::RenderFailure> {
        if filters.is_empty() || filters.len() > super::graph::MAX_POINTWISE_FILTERS {
            return Err(crate::gpu_runtime::RenderFailure::new(
                crate::gpu_runtime::RenderFailureKind::Validation,
                "invalid pointwise filter chain length",
            ));
        }
        let mut uniforms = MatrixChain {
            channels: [0, 1, 2, 3],
            matrix: filters[filters.len() - 1].matrix,
        };
        for filter in &filters[..filters.len() - 1] {
            let channels = super::graph::channel_permutation(filter).ok_or_else(|| {
                crate::gpu_runtime::RenderFailure::new(
                    crate::gpu_runtime::RenderFailureKind::Validation,
                    "inexact intermediate color matrix fusion",
                )
            })?;
            let previous = uniforms.channels;
            for (output, input) in channels.into_iter().enumerate() {
                uniforms.channels[output] = previous[input as usize];
            }
        }
        let sample_count = source.texture.sample_count();
        let format = source.texture.format();
        let pipeline = self.pipeline(descriptors, sample_count)?;

        let target = CommandTarget::new(
            descriptors,
            texture_pool,
            wgpu::Extent3d {
                width: source.size.0,
                height: source.size.1,
                depth_or_array_layers: 1,
            },
            format,
            sample_count,
            RenderTargetMode::FreshWithColor(wgpu::Color::TRANSPARENT),
            draw_encoder,
        )?;
        let source_view = source.texture.create_view(&Default::default())?;
        staging_belt
            .write_buffer(draw_encoder, &self.uniform_buffer, 0, self.uniform_size)?
            .copy_from_slice(bytemuck::bytes_of(&uniforms));
        staging_belt
            .write_buffer(draw_encoder, &self.vertex_buffer, 0, self.vertices_size)?
            .copy_from_slice(bytemuck::cast_slice(&[source.vertices()]));
        let filter_group = descriptors
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: create_debug_label!("Filter group").as_deref(),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&source_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(
                            descriptors.bitmap_samplers.get_sampler(false, false),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                ],
            })?;
        ruffle_render::profile::work(|w| {
            w.render_passes += 1;
            w.filter_passes += 1;
            w.draw_calls += 1;
            w.pipeline_changes += 1;
            w.material_binds += 1;
            w.pass_pixels += u64::from(target.width()) * u64::from(target.height());
        });
        let mut render_pass = draw_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: create_debug_label!("Color matrix filter").as_deref(),
            color_attachments: &[target.color_attachments()],
            ..Default::default()
        });
        render_pass.set_pipeline(pipeline);

        render_pass.set_bind_group(0, (&filter_group).raw(), &[]);

        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(
            descriptors.quad.index_buffer.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.draw_indexed(descriptors.quad.indices_quad.clone(), 0, 0..1);
        Ok(target)
    }
}
