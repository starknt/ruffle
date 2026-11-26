use super::FilterPipelines;
use crate::backend::RenderTargetMode;
use crate::buffer_pool::TexturePool;
use crate::descriptors::Descriptors;
use crate::filters::{FilterSource, FilterVertex, VERTEX_BUFFERS_DESCRIPTION_FILTERS};
use crate::gpu_resources::StagingBelt;
use crate::surface::target::CommandTarget;
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
    pipelines: FilterPipelines,
    compute_pipeline: std::sync::OnceLock<(
        crate::gpu_resources::ComputePipeline,
        crate::gpu_resources::BindGroupLayout,
    )>,
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
            compute_pipeline: Default::default(),
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
        format: wgpu::TextureFormat,
    ) -> Result<&crate::gpu_resources::RenderPipeline, crate::gpu_runtime::RenderFailure> {
        self.pipelines.get_or_init(format, msaa_sample_count, || {
            let label = create_debug_label!("Color Matrix Filter ({} msaa)", msaa_sample_count);
            descriptors
                .persistent_device
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
                        targets: &[Some(format.into())],
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
            false,
            None,
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
        compute: bool,
        destination: Option<&crate::gpu_resources::TextureView>,
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
        let permutations = if compute {
            filters
        } else {
            &filters[..filters.len() - 1]
        };
        for filter in permutations {
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
            if compute {
                RenderTargetMode::FreshStorage
            } else if let Some(view) = destination {
                RenderTargetMode::ExistingWithView(view.clone(), wgpu::Color::TRANSPARENT)
            } else {
                RenderTargetMode::FreshWithColor(wgpu::Color::TRANSPARENT)
            },
            draw_encoder,
        )?;
        staging_belt
            .write_buffer(draw_encoder, &self.uniform_buffer, 0, self.uniform_size)?
            .copy_from_slice(bytemuck::bytes_of(&uniforms));
        if compute {
            let device = &descriptors.persistent_device;
            let (pipeline, layout) =
                crate::gpu_resources::try_sync_cell(&self.compute_pipeline, || {
                    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("Color matrix compute"),
                        source: wgpu::ShaderSource::Wgsl(
                            include_str!("../../shaders/filter/color_matrix_compute.wgsl").into(),
                        ),
                    })?;
                    let pipeline =
                        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some("Color matrix compute"),
                            layout: None,
                            module: &shader,
                            entry_point: Some("main_compute"),
                            compilation_options: Default::default(),
                            cache: None,
                        })?;
                    let layout = pipeline.get_bind_group_layout(0)?;
                    Ok::<_, crate::gpu_runtime::RenderFailure>((pipeline, layout))
                })?;
            let group = descriptors.cached_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Color matrix compute"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(source.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(target.color_view()),
                    },
                ],
            })?;
            draw_encoder
                .begin_filter_compute_pass(
                    source.size,
                    &wgpu::ComputePassDescriptor {
                        label: Some("Color matrix compute"),
                        timestamp_writes: None,
                    },
                )
                .dispatch(
                    pipeline,
                    &group,
                    [source.size.0.div_ceil(8), source.size.1.div_ceil(8)],
                );
            ruffle_render::profile::work(|w| {
                w.filter_passes += 1;
                w.compute_filter_passes += 1;
                w.filter_kind_color_matrix += 1;
                w.pipeline_changes += 1;
                w.material_binds += 1;
                w.pass_pixels += u64::from(target.width()) * u64::from(target.height());
            });
            return Ok(target);
        }
        let pipeline = self.pipeline(descriptors, sample_count, format)?;
        staging_belt
            .write_buffer(draw_encoder, &self.vertex_buffer, 0, self.vertices_size)?
            .copy_from_slice(bytemuck::cast_slice(&[source.vertices()]));
        let filter_group = descriptors.cached_bind_group(&wgpu::BindGroupDescriptor {
            label: create_debug_label!("Filter group").as_deref(),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source.view),
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
            w.filter_kind_color_matrix += 1;
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

        render_pass.set_bind_group(0, filter_group.raw(), &[]);

        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(
            descriptors.quad.index_buffer.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.draw_indexed(descriptors.quad.indices_quad.clone(), 0, 0..1);
        Ok(target)
    }
}
