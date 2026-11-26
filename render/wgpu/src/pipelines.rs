use crate::blend::{ComplexBlend, TrivialBlend};
use crate::gpu_resources::{Device, PipelineLayout, RenderPipeline, ShaderModule};
use crate::gpu_runtime::RenderFailure;
use crate::layouts::BindLayouts;
use crate::shaders::Shaders;
use crate::surface::mask::MaskMode;
use crate::{PosColorVertex, PosUvVertex, PosVertex};
use enum_map::{EnumMap, enum_map};
use once_cell::sync::OnceCell;
use wgpu::{BlendState, PrimitiveTopology, vertex_attr_array};

pub const VERTEX_BUFFERS_DESCRIPTION_POS: [Option<wgpu::VertexBufferLayout>; 1] =
    [Some(wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<PosVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &vertex_attr_array![
            0 => Float32x2,
        ],
    })];

pub const VERTEX_BUFFERS_DESCRIPTION_POS_UV: [Option<wgpu::VertexBufferLayout>; 1] =
    [Some(wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<PosUvVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &vertex_attr_array![
            0 => Float32x2,
            1 => Float32x3,
        ],
    })];

pub const VERTEX_BUFFERS_DESCRIPTION_COLOR: [Option<wgpu::VertexBufferLayout>; 1] =
    [Some(wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<PosColorVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &vertex_attr_array![
            0 => Float32x2,
            1 => Float32x4,
        ],
    })];

#[derive(Debug)]
pub struct ShapePipeline {
    device: Device,
    name: String,
    format: wgpu::TextureFormat,
    samples: u32,
    shader: ShaderModule,
    layout: PipelineLayout,
    vertices: &'static [Option<wgpu::VertexBufferLayout<'static>>],
    topology: PrimitiveTopology,
    blend: BlendState,
    opaque: bool,
    masked: EnumMap<MaskMode, OnceCell<RenderPipeline>>,
    stencilless: OnceCell<RenderPipeline>,
}

#[derive(Debug)]
pub struct Pipelines {
    pub color: ShapePipeline,
    pub lines: ShapePipeline,
    /// Renders a bitmap without any blending, and does
    /// not write to the alpha channel. This is used for
    /// drawing a finished Stage3D buffer onto the background.
    pub bitmap_opaque: ShapePipeline,
    pub bitmap: EnumMap<TrivialBlend, ShapePipeline>,
    pub bitmap_batch: ShapePipeline,
    pub gradients: ShapePipeline,
    pub complex_blends: EnumMap<ComplexBlend, ShapePipeline>,
    pub alpha_mask: ShapePipeline,
}

impl ShapePipeline {
    pub fn pipeline_for(&self, mask: MaskMode) -> Result<&RenderPipeline, RenderFailure> {
        self.variant(Some(mask))
    }
    pub fn stencilless_pipeline(&self) -> Result<&RenderPipeline, RenderFailure> {
        self.variant(None)
    }

    fn variant(&self, mask: Option<MaskMode>) -> Result<&RenderPipeline, RenderFailure> {
        let cell = mask.map_or(&self.stencilless, |mode| &self.masked[mode]);
        cell.get_or_try_init(|| {
            let depth = mask.map(|mode| {
                let mut state = mode.pipeline_stencil();
                if self.opaque {
                    state.stencil.read_mask = 0;
                    state.stencil.write_mask = 0;
                }
                state
            });
            let write_mask = if self.opaque {
                wgpu::ColorWrites::COLOR
            } else {
                mask.map_or(wgpu::ColorWrites::ALL, MaskMode::color_writes)
            };
            let constants = if self.opaque && mask.is_none() {
                &[("late_saturate", 1.0)][..]
            } else {
                &[]
            };
            self.device
                .create_render_pipeline(&create_pipeline_descriptor(
                    create_debug_label!("{} {:?}", self.name, mask).as_deref(),
                    &self.shader,
                    &self.shader,
                    &self.layout,
                    depth,
                    &[Some(wgpu::ColorTargetState {
                        format: self.format,
                        blend: Some(self.blend),
                        write_mask,
                    })],
                    self.vertices,
                    self.samples,
                    constants,
                    self.topology,
                ))
        })
    }
}

impl Pipelines {
    pub fn new(
        device: &crate::gpu_resources::Device,
        shaders: &Shaders,
        format: wgpu::TextureFormat,
        msaa_sample_count: u32,
        bind_layouts: &BindLayouts,
    ) -> Result<Self, crate::gpu_runtime::RenderFailure> {
        let colort_bindings = vec![
            Some(bind_layouts.globals.raw()),
            Some(bind_layouts.transforms.raw()),
        ];

        let color_pipelines = create_shape_pipeline(
            "Color",
            device,
            format,
            &shaders.color_shader,
            msaa_sample_count,
            &VERTEX_BUFFERS_DESCRIPTION_COLOR,
            &colort_bindings,
            BlendState::PREMULTIPLIED_ALPHA_BLENDING,
            false,
            PrimitiveTopology::TriangleList,
        )?;

        let lines_pipelines = create_shape_pipeline(
            "Lines",
            device,
            format,
            &shaders.color_shader,
            msaa_sample_count,
            &VERTEX_BUFFERS_DESCRIPTION_COLOR,
            &colort_bindings,
            BlendState::PREMULTIPLIED_ALPHA_BLENDING,
            false,
            PrimitiveTopology::LineStrip,
        )?;

        let gradient_bindings = vec![
            Some(bind_layouts.globals.raw()),
            Some(bind_layouts.transforms.raw()),
            Some(bind_layouts.gradient.raw()),
        ];

        let gradient_pipeline = create_shape_pipeline(
            "Gradient",
            device,
            format,
            &shaders.gradient_shader,
            msaa_sample_count,
            &VERTEX_BUFFERS_DESCRIPTION_POS_UV,
            &gradient_bindings,
            BlendState::PREMULTIPLIED_ALPHA_BLENDING,
            false,
            PrimitiveTopology::TriangleList,
        )?;

        let complex_blend_bindings = vec![
            Some(bind_layouts.globals.raw()),
            None,
            Some(bind_layouts.blend.raw()),
        ];

        let complex_blend_pipelines = enum_map! {
            blend => create_shape_pipeline(
                &format!("Complex Blend: {blend:?}"),
                device,
                format,
                &shaders.blend_shaders[blend],
                msaa_sample_count,
                &VERTEX_BUFFERS_DESCRIPTION_POS,
                &complex_blend_bindings,
                BlendState::REPLACE,
                false,
                PrimitiveTopology::TriangleList,
            )?
        };

        let bitmap_blend_bindings = vec![
            Some(bind_layouts.globals.raw()),
            Some(bind_layouts.transforms.raw()),
            Some(bind_layouts.bitmap.raw()),
        ];

        let bitmap_pipelines: EnumMap<TrivialBlend, ShapePipeline> = enum_map! { blend => {
            create_shape_pipeline(
                &format!("Bitmap ({blend:?})"),
                device,
                format,
                &shaders.bitmap_shader,
                msaa_sample_count,
                &VERTEX_BUFFERS_DESCRIPTION_POS_UV,
                &bitmap_blend_bindings,
                blend.blend_state(),
                false,
                PrimitiveTopology::TriangleList,
            )?
        }};

        let bitmap_opaque = create_shape_pipeline(
            "Bitmap opaque copy",
            device,
            format,
            &shaders.bitmap_shader,
            msaa_sample_count,
            &VERTEX_BUFFERS_DESCRIPTION_POS_UV,
            &bitmap_blend_bindings,
            BlendState::REPLACE,
            true,
            PrimitiveTopology::TriangleList,
        )?;

        let bitmap_batch = create_shape_pipeline(
            "Adjacent bitmap textures",
            device,
            format,
            &shaders.bitmap_batch_shader,
            msaa_sample_count,
            &VERTEX_BUFFERS_DESCRIPTION_POS_UV,
            &[
                Some(bind_layouts.globals.raw()),
                Some(bind_layouts.transforms.raw()),
                Some(bind_layouts.bitmap_batch.raw()),
            ],
            BlendState::PREMULTIPLIED_ALPHA_BLENDING,
            false,
            PrimitiveTopology::TriangleList,
        )?;

        let alpha_mask_bindings = vec![
            Some(bind_layouts.globals.raw()),
            Some(bind_layouts.transforms.raw()),
            Some(bind_layouts.alpha_mask.raw()),
        ];

        let alpha_mask_pipeline = create_shape_pipeline(
            "Alpha Mask",
            device,
            format,
            &shaders.alpha_mask_shader,
            msaa_sample_count,
            &VERTEX_BUFFERS_DESCRIPTION_POS_UV,
            &alpha_mask_bindings,
            BlendState::PREMULTIPLIED_ALPHA_BLENDING,
            false,
            PrimitiveTopology::TriangleList,
        )?;

        Ok(Self {
            bitmap_batch,
            color: color_pipelines,
            lines: lines_pipelines,
            bitmap: bitmap_pipelines,
            bitmap_opaque,
            gradients: gradient_pipeline,
            complex_blends: complex_blend_pipelines,
            alpha_mask: alpha_mask_pipeline,
        })
    }
}

#[expect(clippy::too_many_arguments)]
fn create_pipeline_descriptor<'a>(
    label: Option<&'a str>,
    vertex_shader: &'a crate::gpu_resources::ShaderModule,
    fragment_shader: &'a crate::gpu_resources::ShaderModule,
    pipeline_layout: &'a crate::gpu_resources::PipelineLayout,
    depth_stencil_state: Option<wgpu::DepthStencilState>,
    color_target_state: &'a [Option<wgpu::ColorTargetState>],
    vertex_buffer_layout: &'a [Option<wgpu::VertexBufferLayout<'a>>],
    msaa_sample_count: u32,
    fragment_constants: &'a [(&str, f64)],
    primitive_topology: PrimitiveTopology,
) -> wgpu::RenderPipelineDescriptor<'a> {
    wgpu::RenderPipelineDescriptor {
        label,
        layout: Some(pipeline_layout),
        vertex: wgpu::VertexState {
            module: vertex_shader,
            entry_point: Some("main_vertex"),
            buffers: vertex_buffer_layout,
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: fragment_shader,
            entry_point: Some("main_fragment"),
            targets: color_target_state,
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: fragment_constants,
                ..Default::default()
            },
        }),
        primitive: wgpu::PrimitiveState {
            topology: primitive_topology,
            // All indexed draws in this backend use `Uint32` indices, and wgpu
            // requires this to be set for indexed drawing with strip topologies.
            strip_index_format: matches!(
                primitive_topology,
                PrimitiveTopology::LineStrip | PrimitiveTopology::TriangleStrip
            )
            .then_some(wgpu::IndexFormat::Uint32),
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::default(),
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: depth_stencil_state,
        multisample: wgpu::MultisampleState {
            count: msaa_sample_count,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview_mask: None,
        cache: None,
    }
}

#[expect(clippy::too_many_arguments)]
fn create_shape_pipeline(
    name: &str,
    device: &crate::gpu_resources::Device,
    format: wgpu::TextureFormat,
    shader: &crate::gpu_resources::ShaderModule,
    msaa_sample_count: u32,
    vertex_buffers_layout: &'static [Option<wgpu::VertexBufferLayout<'static>>],
    bind_group_layouts: &[Option<&wgpu::BindGroupLayout>],
    blend: BlendState,
    opaque: bool,
    primitive_topology: PrimitiveTopology,
) -> Result<ShapePipeline, crate::gpu_runtime::RenderFailure> {
    let pipeline_layout_label = create_debug_label!("{} shape pipeline layout", name);
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: pipeline_layout_label.as_deref(),
        bind_group_layouts,
        immediate_size: 0,
    })?;

    let variant = ShapePipeline {
        device: device.clone(),
        name: name.to_owned(),
        format,
        samples: msaa_sample_count,
        shader: shader.clone(),
        layout: pipeline_layout,
        vertices: vertex_buffers_layout,
        topology: primitive_topology,
        blend,
        opaque,
        masked: Default::default(),
        stencilless: OnceCell::new(),
    };
    if std::env::var_os("RUFFLE_2D_PIPELINE_LAZY").as_deref() == Some("0".as_ref()) {
        variant.variant(None)?;
        for (mode, _) in &variant.masked {
            if !opaque || mode == MaskMode::NoMask {
                variant.variant(Some(mode))?;
            }
        }
    }
    Ok(variant)
}
