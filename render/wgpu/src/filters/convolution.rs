use super::{FilterPipelines, FilterSource, VERTEX_BUFFERS_DESCRIPTION_FILTERS};
use crate::backend::RenderTargetMode;
use crate::buffer_pool::TexturePool;
use crate::descriptors::Descriptors;
use crate::gpu_resources::{
    BindGroupLayout, Buffer, CommandEncoder, Device, PipelineLayout, StagingBelt,
};
use crate::gpu_runtime::{RenderFailure, RenderFailureKind};
use crate::surface::target::CommandTarget;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ConvolutionUniform {
    source_point: [i32; 2],
    source_size: [i32; 2],
    default_color: [f32; 4],
    divisor: f32,
    bias: f32,
    clamp_edges: u32,
    alpha_mode: u32,
}

const PARAMETER_SIZE: wgpu::BufferSize =
    wgpu::BufferSize::new(size_of::<ConvolutionUniform>() as u64).unwrap();

pub struct ConvolutionFilter {
    bind_group_layout: BindGroupLayout,
    pipeline_layout: PipelineLayout,
    uniform_buffer: Buffer,
    pipelines: FilterPipelines,
}

impl ConvolutionFilter {
    pub fn new(device: &Device) -> Result<Self, RenderFailure> {
        let texture_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Convolution filter binds"),
                entries: &[
                    texture_entry(0),
                    texture_entry(1),
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                PARAMETER_SIZE.get()
                                    + size_of_val(&ruffle_render::bitmap::FLASH_PREMUL_FACTOR)
                                        as u64,
                            ),
                        },
                        count: None,
                    },
                ],
            })?;
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Convolution filter"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        })?;
        // Only the parameters at the start change per draw. The Flash channel factors follow them.
        let mut uniform_data = vec![0; size_of::<ConvolutionUniform>()];
        uniform_data.extend_from_slice(bytemuck::cast_slice(
            &ruffle_render::bitmap::FLASH_PREMUL_FACTOR,
        ));
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Convolution parameters"),
            contents: &uniform_data,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })?;
        Ok(Self {
            bind_group_layout,
            pipeline_layout,
            uniform_buffer,
            pipelines: Default::default(),
        })
    }

    pub fn apply(
        &self,
        descriptors: &Descriptors,
        texture_pool: &mut TexturePool,
        encoder: &mut CommandEncoder,
        staging_belt: &mut StagingBelt,
        source: &FilterSource,
        filter: &swf::ConvolutionFilter,
        opaque_target: bool,
        destination: Option<FilterSource<'_>>,
    ) -> Result<CommandTarget, RenderFailure> {
        let cols = filter.num_matrix_cols;
        let rows = filter.num_matrix_rows;
        // AVM1 can retain trailing entries when a matrix shrinks.
        let weights = filter
            .matrix
            .get(..usize::from(cols) * usize::from(rows))
            .ok_or_else(|| {
                RenderFailure::new(
                    RenderFailureKind::Validation,
                    "incomplete convolution matrix",
                )
            })?;
        let coefficients = descriptors.device.create_texture_with_data(
            &descriptors.queue,
            &wgpu::TextureDescriptor {
                label: Some("Convolution coefficients"),
                size: wgpu::Extent3d {
                    width: u32::from(cols),
                    height: u32::from(rows),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(weights),
        )?;
        let coefficient_view = coefficients.create_view(&Default::default())?;
        let group = descriptors
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Convolution filter"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(source.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&coefficient_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                ],
            })?;
        let format = source.texture.format();
        let sample_count = source.texture.sample_count();
        let pipeline = self.pipelines.get_or_init(format, sample_count, || {
            let shader = &descriptors.shaders.convolution_filter;
            descriptors
                .persistent_device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Convolution filter"),
                    layout: Some(&self.pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: shader,
                        entry_point: Some("filter__vertex_entry_point"),
                        buffers: &VERTEX_BUFFERS_DESCRIPTION_FILTERS,
                        compilation_options: Default::default(),
                    },
                    primitive: Default::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState {
                        count: sample_count,
                        ..Default::default()
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: shader,
                        entry_point: Some("main_fragment"),
                        targets: &[Some(format.into())],
                        compilation_options: Default::default(),
                    }),
                    multiview_mask: None,
                    cache: None,
                })
        })?;
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
            match destination.filter(|dst| dst.texture != source.texture) {
                Some(dst) => {
                    RenderTargetMode::ExistingWithView(dst.view.clone(), wgpu::Color::TRANSPARENT)
                }
                None => RenderTargetMode::FreshWithColor(wgpu::Color::TRANSPARENT),
            },
            encoder,
        )?;
        let color = filter.default_color;
        let uniform = ConvolutionUniform {
            source_point: [source.point.0 as i32, source.point.1 as i32],
            source_size: [source.size.0 as i32, source.size.1 as i32],
            default_color: [color.r, color.g, color.b, color.a].map(f32::from),
            divisor: if filter.divisor == 0.0 {
                1.0
            } else {
                filter.divisor
            },
            bias: filter.bias,
            clamp_edges: u32::from(filter.is_clamped()),
            alpha_mode: if opaque_target {
                2
            } else {
                u32::from(filter.is_preserve_alpha())
            },
        };
        staging_belt
            .write_buffer(encoder, &self.uniform_buffer, 0, PARAMETER_SIZE)?
            .copy_from_slice(bytemuck::bytes_of(&uniform));
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Convolution filter"),
            color_attachments: &[target.color_attachments()],
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group.raw(), &[]);
        let quad = &descriptors.quad;
        pass.set_vertex_buffer(0, quad.vertex_buffer.slice(quad.vertices_filter.clone()));
        pass.set_index_buffer(quad.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(quad.indices_quad.clone(), 0, 0..1);
        ruffle_render::profile::work(|w| {
            w.render_passes += 1;
            w.filter_passes += 1;
            w.filter_kind_convolution += 1;
            w.draw_calls += 1;
            w.pipeline_changes += 1;
            w.material_binds += 1;
            w.pass_pixels += u64::from(target.width()) * u64::from(target.height());
        });
        Ok(target)
    }
}
