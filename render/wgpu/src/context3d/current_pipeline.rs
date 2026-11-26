use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga_agal::{Filter, SamplerConfig, Wrapping};
use ruffle_render::backend::{
    Context3DCompareMode, Context3DStencilAction, Context3DTextureFilter, Context3DTriangleFace,
    Context3DWrapMode, Texture,
};

use crate::gpu_resources::Buffer;
use crate::gpu_resources::TextureView;
use wgpu::DepthStencilState;
use wgpu::{BindGroupEntry, BindingResource, BufferDescriptor, BufferUsages, FrontFace};
use wgpu::{ColorTargetState, RenderPipelineDescriptor, TextureFormat, VertexState};

pub(super) trait IntoWgpu {
    type WgpuType;

    fn into_wgpu(self) -> Self::WgpuType;
}

impl IntoWgpu for Context3DCompareMode {
    type WgpuType = wgpu::CompareFunction;

    fn into_wgpu(self) -> wgpu::CompareFunction {
        match self {
            Context3DCompareMode::Always => wgpu::CompareFunction::Always,
            Context3DCompareMode::Equal => wgpu::CompareFunction::Equal,
            Context3DCompareMode::Greater => wgpu::CompareFunction::Greater,
            Context3DCompareMode::GreaterEqual => wgpu::CompareFunction::GreaterEqual,
            Context3DCompareMode::Less => wgpu::CompareFunction::Less,
            Context3DCompareMode::LessEqual => wgpu::CompareFunction::LessEqual,
            Context3DCompareMode::Never => wgpu::CompareFunction::Never,
            Context3DCompareMode::NotEqual => wgpu::CompareFunction::NotEqual,
        }
    }
}

impl IntoWgpu for Context3DStencilAction {
    type WgpuType = wgpu::StencilOperation;

    fn into_wgpu(self) -> wgpu::StencilOperation {
        match self {
            Context3DStencilAction::DecrementSaturate => wgpu::StencilOperation::DecrementClamp,
            Context3DStencilAction::DecrementWrap => wgpu::StencilOperation::DecrementWrap,
            Context3DStencilAction::IncrementSaturate => wgpu::StencilOperation::IncrementClamp,
            Context3DStencilAction::IncrementWrap => wgpu::StencilOperation::IncrementWrap,
            Context3DStencilAction::Invert => wgpu::StencilOperation::Invert,
            Context3DStencilAction::Keep => wgpu::StencilOperation::Keep,
            Context3DStencilAction::Set => wgpu::StencilOperation::Replace,
            Context3DStencilAction::Zero => wgpu::StencilOperation::Zero,
        }
    }
}

use std::cell::Cell;
use std::num::NonZeroU64;
use std::rc::Rc;

use super::pipeline_cache::{PipelineCache, PipelineKey, Prepared, VertexInputs};
use crate::bitmaps::WgpuSamplerConfig;
use crate::context3d::shader_pair::{ShaderCompileData, ShaderTextureInfo};
use crate::descriptors::Descriptors;

use super::ShaderPairAgal;

const AGAL_NUM_VERTEX_CONSTANTS: u64 = 128;
const AGAL_NUM_FRAGMENT_CONSTANTS: u64 = 28;
pub(super) const AGAL_FLOATS_PER_REGISTER: u64 = 4;

const VERTEX_SHADER_UNIFORMS_BUFFER_SIZE: u64 =
    AGAL_NUM_VERTEX_CONSTANTS * AGAL_FLOATS_PER_REGISTER * std::mem::size_of::<f32>() as u64;
const FRAGMENT_SHADER_UNIFORMS_BUFFER_SIZE: u64 =
    AGAL_NUM_FRAGMENT_CONSTANTS * AGAL_FLOATS_PER_REGISTER * std::mem::size_of::<f32>() as u64;

// The flash Context3D API is similar to OpenGL - it has many methods
// which modify the current state (`setVertexBufferAt`, `setCulling`, etc.)
// These methods can be called at any time.
//
// A draw resolves structural changes to a device-local pipeline. Resource-only
// changes update the bindings independently of the AGAL specialization.
pub struct CurrentPipeline {
    device: crate::gpu_resources::Device,
    shaders: Option<Rc<ShaderPairAgal>>,

    culling: Context3DTriangleFace,

    bound_textures: [Option<BoundTextureData>; 8],

    pub vertex_shader_uniforms: Buffer,
    pub fragment_shader_uniforms: Buffer,

    has_depth_texture: bool,

    color_mask: wgpu::ColorWrites,

    depth_mask: bool,
    pass_compare_mode: wgpu::CompareFunction,

    /// Stencil test configuration (face operations, compare functions, masks), set via `setStencilActions`.
    stencil: wgpu::StencilState,
    /// Reference value used for stencil comparison and operations, set via `setStencilReferenceValue`.
    stencil_ref_value: u32,

    color_component: wgpu::BlendComponent,
    alpha_component: wgpu::BlendComponent,

    sample_count: u32,

    target_format: TextureFormat,

    dirty: Cell<bool>,
    bindings_dirty: bool,
    reuse: bool,
    cache: PipelineCache,
    materials: crate::material_cache::MaterialCache,
    active: Option<(Prepared, crate::gpu_resources::BindGroup)>,

    // Sampler configuration information for each texture slot.
    // This is updated by `Context3D.setSamplerStateAt`, as well
    // as in `Context3D.setProgram` (based on the sampling opcodes
    // in the program). All texture slots have a sampler set by default
    // (which allows rendering with an 'ignoresampler' tex opcode,
    // and no calls to Context3D.setSamplerStateAt)
    sampler_configs: [SamplerConfig; 8],
}

#[derive(Clone)]
pub struct BoundTextureData {
    /// This is used to allow us to remove a bound texture when
    /// it's used with `setRenderToTexture`. The actual shader binding
    /// uses `view`
    pub id: Rc<dyn Texture>,
    pub view: TextureView,
    pub cube: bool,
}

impl CurrentPipeline {
    pub fn new(
        device: &crate::gpu_resources::Device,
    ) -> Result<Self, crate::gpu_runtime::RenderFailure> {
        let vertex_shader_uniforms = device.create_buffer(&BufferDescriptor {
            label: Some("Vertex shader uniforms"),
            size: VERTEX_SHADER_UNIFORMS_BUFFER_SIZE,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })?;

        let fragment_shader_uniforms = device.create_buffer(&BufferDescriptor {
            label: Some("Fragment shader uniforms"),
            size: FRAGMENT_SHADER_UNIFORMS_BUFFER_SIZE,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })?;

        Ok(CurrentPipeline {
            device: device.clone(),
            shaders: None,
            bound_textures: std::array::from_fn(|_| None),
            vertex_shader_uniforms,
            fragment_shader_uniforms,
            dirty: Cell::new(true),
            bindings_dirty: true,
            reuse: std::env::var("RUFFLE_STAGE3D_PIPELINE_REUSE").as_deref() != Ok("0"),
            cache: PipelineCache::new(device.runtime().scheduler()),
            materials: crate::material_cache::MaterialCache::new(device.runtime(), 64),
            active: None,
            culling: Context3DTriangleFace::None,

            has_depth_texture: false,

            color_mask: wgpu::ColorWrites::ALL,

            depth_mask: true,
            pass_compare_mode: wgpu::CompareFunction::LessEqual,

            stencil: wgpu::StencilState {
                front: wgpu::StencilFaceState::IGNORE,
                back: wgpu::StencilFaceState::IGNORE,
                read_mask: u32::MAX,
                write_mask: u32::MAX,
            },
            stencil_ref_value: 0,

            color_component: wgpu::BlendComponent::REPLACE,
            alpha_component: wgpu::BlendComponent::REPLACE,
            sample_count: 1,

            target_format: TextureFormat::Rgba8Unorm,

            sampler_configs: [SamplerConfig::default(); 8],
        })
    }

    pub fn stencil_ref_value(&self) -> u32 {
        self.stencil_ref_value
    }

    pub fn set_shaders(&mut self, shaders: Option<Rc<ShaderPairAgal>>) {
        if !self.reuse
            || self.shaders.as_ref().map(|p| p.upload_id) != shaders.as_ref().map(|p| p.upload_id)
        {
            self.dirty.set(true);
        }
        self.shaders = shaders;
        if let Some(shaders) = &self.shaders {
            for (i, sampler_config) in shaders.fragment_sampler_configs().iter().enumerate() {
                // When we call `Context3D.setProgram`, sampler configs from the fragment shader override
                // any previously set sampler configs (if 'ignoresampler' was set in the program, then the corresponding
                // array entry will be `None`).
                if let Some(config) = sampler_config
                    && *config != self.sampler_configs[i]
                {
                    self.bindings_dirty = true;
                    self.sampler_configs[i] = *config;
                }
            }
        }
    }

    pub fn has_unbound_required_textures(&self) -> bool {
        if let Some(shaders) = &self.shaders {
            for (i, sampler_config) in shaders.fragment_sampler_configs().iter().enumerate() {
                if sampler_config.is_some() && self.bound_textures[i].is_none() {
                    return true;
                }
            }
        }
        false
    }

    pub fn same_texture(
        &self,
        index: usize,
        texture: Option<&Rc<dyn Texture>>,
        cube: bool,
    ) -> bool {
        self.reuse
            && match (&self.bound_textures[index], texture) {
                (Some(old), Some(new)) => {
                    old.cube == cube && std::ptr::addr_eq(Rc::as_ptr(&old.id), Rc::as_ptr(new))
                }
                (None, None) => true,
                _ => false,
            }
    }

    pub fn update_texture_at(&mut self, index: usize, texture: Option<BoundTextureData>) {
        if !self.reuse
            || self.bound_textures[index].as_ref().map(|t| t.cube)
                != texture.as_ref().map(|t| t.cube)
        {
            self.dirty.set(true);
        }
        self.bindings_dirty = true;
        self.bound_textures[index] = texture;
    }

    pub fn remove_texture(&mut self, texture: &Rc<dyn Texture>) {
        for i in 0..self.bound_textures.len() {
            if let Some(bound_texture) = &self.bound_textures[i] {
                // Ignore the vtable pointer
                if std::ptr::eq(
                    Rc::as_ptr(&bound_texture.id) as *const (),
                    Rc::as_ptr(texture) as *const (),
                ) {
                    self.update_texture_at(i, None);
                }
            }
        }
    }

    pub fn update_vertex_buffer_at(&self, layout_changed: bool) {
        if layout_changed || !self.reuse {
            self.dirty.set(true);
        }
    }

    pub fn update_color_mask(&mut self, color_mask: wgpu::ColorWrites) {
        if self.color_mask != color_mask {
            self.dirty.set(true);
        }
        self.color_mask = color_mask;
    }

    pub fn update_depth(&mut self, depth_mask: bool, pass_compare_mode: wgpu::CompareFunction) {
        if self.depth_mask != depth_mask || self.pass_compare_mode != pass_compare_mode {
            self.dirty.set(true);
        }
        self.depth_mask = depth_mask;
        self.pass_compare_mode = pass_compare_mode;
    }

    pub fn update_has_depth_texture(&mut self, has_depth_texture: bool) {
        if self.has_depth_texture != has_depth_texture {
            self.dirty.set(true);
            self.has_depth_texture = has_depth_texture;
        }
    }

    pub fn update_sample_count(&mut self, sample_count: u32) {
        if self.sample_count != sample_count {
            self.dirty.set(true);
            self.sample_count = sample_count;
        }
    }

    pub fn update_target_format(&mut self, format: TextureFormat) {
        if self.target_format != format {
            self.dirty.set(true);
            self.target_format = format;
        }
    }

    pub fn active(
        &self,
    ) -> Option<(
        &crate::gpu_resources::RenderPipeline,
        &crate::gpu_resources::BindGroup,
    )> {
        self.active.as_ref().map(|(p, b)| (&p.pipeline, b))
    }

    pub fn rebuild_pipeline(
        &mut self,
        descriptors: &Descriptors,
        inputs: &VertexInputs,
    ) -> Result<(), super::PipelineError> {
        if !self.dirty.get() && !self.bindings_dirty {
            return Ok(());
        }
        let prepared = if self.dirty.get() {
            let shaders = self.shaders.as_ref().expect("Missing shaders!");
            let cull_mode = match self.culling {
                Context3DTriangleFace::Back => Some(wgpu::Face::Back),
                Context3DTriangleFace::Front => Some(wgpu::Face::Front),
                Context3DTriangleFace::FrontAndBack => {
                    tracing::error!("FrontAndBack culling not supported!");
                    None
                }
                Context3DTriangleFace::None => None,
            };

            let texture_infos = self.bound_textures.each_ref().map(|t| {
                t.as_ref().map(|t| {
                    if t.cube {
                        ShaderTextureInfo::Cube
                    } else {
                        ShaderTextureInfo::D2
                    }
                })
            });
            let key = PipelineKey {
                program: shaders.upload_id,
                shader: ShaderCompileData {
                    vertex_attributes: inputs.attributes,
                    texture_infos,
                },
                streams: inputs.streams.clone(),
                cull: cull_mode,
                depth: self.has_depth_texture.then(|| {
                    (
                        self.depth_mask,
                        self.pass_compare_mode,
                        self.stencil.clone(),
                    )
                }),
                target: ColorTargetState {
                    format: self.target_format,
                    blend: Some(wgpu::BlendState {
                        color: self.color_component,
                        alpha: self.alpha_component,
                    }),
                    write_mask: self.color_mask,
                },
                samples: self.sample_count,
            };
            if let Some(prepared) = self.reuse.then(|| self.cache.get(&key)).flatten() {
                prepared
            } else {
                let prepared = self.compile_pipeline(shaders, &key)?;
                if self.reuse {
                    self.cache.insert(key, &prepared);
                }
                prepared
            }
        } else {
            self.active
                .as_ref()
                .expect("binding update has an active pipeline")
                .0
                .clone()
        };
        let bindings = match &self.active {
            Some((old, group))
                if self.reuse && !self.bindings_dirty && old.bindings == prepared.bindings =>
            {
                group.clone()
            }
            _ => self.bindings(descriptors, &prepared.bindings)?,
        };
        self.active = Some((prepared, bindings));
        self.dirty.set(false);
        self.bindings_dirty = false;
        Ok(())
    }

    fn bindings(
        &self,
        descriptors: &Descriptors,
        layout: &crate::gpu_resources::BindGroupLayout,
    ) -> Result<crate::gpu_resources::BindGroup, crate::gpu_runtime::RenderFailure> {
        let bind_group_label = create_debug_label!("Bind group");

        let wgpu_samplers =
            self.sampler_configs
                .map(|sampler| match (sampler.wrapping, sampler.filter) {
                    (Wrapping::Clamp, Filter::Linear) => &descriptors.bitmap_samplers.clamp_linear,
                    (Wrapping::Clamp, Filter::Nearest) => {
                        &descriptors.bitmap_samplers.clamp_nearest
                    }
                    (Wrapping::Repeat, Filter::Linear) => {
                        &descriptors.bitmap_samplers.repeat_linear
                    }
                    (Wrapping::Repeat, Filter::Nearest) => {
                        &descriptors.bitmap_samplers.repeat_nearest
                    }
                    (Wrapping::ClampURepeatV, Filter::Linear) => {
                        &descriptors.bitmap_samplers.clamp_u_repeat_v_linear
                    }
                    (Wrapping::ClampURepeatV, Filter::Nearest) => {
                        &descriptors.bitmap_samplers.clamp_u_repeat_v_nearest
                    }
                    (Wrapping::RepeatUClampV, Filter::Linear) => {
                        &descriptors.bitmap_samplers.repeat_u_clamp_v_linear
                    }
                    (Wrapping::RepeatUClampV, Filter::Nearest) => {
                        &descriptors.bitmap_samplers.repeat_u_clamp_v_nearest
                    }
                    (
                        wrapping,
                        Filter::Anisotropic2x
                        | Filter::Anisotropic4x
                        | Filter::Anisotropic8x
                        | Filter::Anisotropic16x,
                    ) => {
                        &descriptors.bitmap_samplers.anisotropic[&WgpuSamplerConfig {
                            anisotropy_clamp: match sampler.filter {
                                Filter::Anisotropic2x => 2,
                                Filter::Anisotropic4x => 4,
                                Filter::Anisotropic8x => 8,
                                Filter::Anisotropic16x => 16,
                                _ => unreachable!(),
                            },
                            address_mode_u: match wrapping {
                                Wrapping::Clamp => wgpu::AddressMode::ClampToEdge,
                                Wrapping::Repeat => wgpu::AddressMode::Repeat,
                                Wrapping::ClampURepeatV => wgpu::AddressMode::ClampToEdge,
                                Wrapping::RepeatUClampV => wgpu::AddressMode::Repeat,
                            },
                            address_mode_v: match wrapping {
                                Wrapping::Clamp => wgpu::AddressMode::ClampToEdge,
                                Wrapping::Repeat => wgpu::AddressMode::Repeat,
                                Wrapping::ClampURepeatV => wgpu::AddressMode::Repeat,
                                Wrapping::RepeatUClampV => wgpu::AddressMode::ClampToEdge,
                            },
                        }]
                    }
                });

        let mut bind_group_entries = vec![
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &self.vertex_shader_uniforms,
                    offset: 0,
                    size: Some(NonZeroU64::new(VERTEX_SHADER_UNIFORMS_BUFFER_SIZE).unwrap()),
                }),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &self.fragment_shader_uniforms,
                    offset: 0,
                    size: Some(NonZeroU64::new(FRAGMENT_SHADER_UNIFORMS_BUFFER_SIZE).unwrap()),
                }),
            },
        ];

        for (i, bound_texture) in self.bound_textures.iter().enumerate() {
            if let Some(bound_texture) = bound_texture {
                bind_group_entries.push(BindGroupEntry {
                    binding: naga_agal::TEXTURE_START_BIND_INDEX + i as u32,
                    resource: BindingResource::TextureView(&bound_texture.view),
                });
                bind_group_entries.push(BindGroupEntry {
                    binding: naga_agal::TEXTURE_SAMPLER_START_BIND_INDEX + i as u32,
                    resource: BindingResource::Sampler(wgpu_samplers[i]),
                });
            }
        }

        let desc = wgpu::BindGroupDescriptor {
            label: bind_group_label.as_deref(),
            layout,
            entries: &bind_group_entries,
        };
        if self.reuse {
            self.materials.get_or_create(&self.device, &desc)
        } else {
            self.device.create_bind_group(&desc)
        }
    }

    fn compile_pipeline(
        &self,
        shaders: &ShaderPairAgal,
        key: &PipelineKey,
    ) -> Result<Prepared, super::PipelineError> {
        let compiled = shaders.program.compile(&self.device, key.shader.clone())?;
        let depth_stencil = key
            .depth
            .as_ref()
            .map(|(write, compare, stencil)| DepthStencilState {
                format: TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: Some(*write),
                depth_compare: Some(*compare),
                stencil: stencil.clone(),
                bias: Default::default(),
            });
        let buffers: smallvec::SmallVec<[_; 4]> = key
            .streams
            .iter()
            .map(|s| {
                assert!(
                    s.attributes
                        .iter()
                        .all(|a| a.offset + a.format.size() <= s.stride),
                    "vertex attribute exceeds Stage3D buffer stride"
                );
                Some(wgpu::VertexBufferLayout {
                    array_stride: s.stride,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &s.attributes,
                })
            })
            .collect();
        let pipeline = self.device.create_agal_pipeline(
            shaders.program.content,
            &key.shader,
            &RenderPipelineDescriptor {
                label: Some("Stage3D render pipeline"),
                layout: Some(&compiled.pipeline_layout),
                vertex: VertexState {
                    module: &compiled.vertex_module,
                    entry_point: Some(naga_agal::SHADER_ENTRY_POINT),
                    buffers: &buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &compiled.fragment_module,
                    entry_point: Some(naga_agal::SHADER_ENTRY_POINT),
                    targets: &[Some(key.target.clone())],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    front_face: FrontFace::Cw,
                    cull_mode: key.cull,
                    ..Default::default()
                },
                depth_stencil,
                multisample: wgpu::MultisampleState {
                    count: key.samples,
                    ..Default::default()
                },
                multiview_mask: None,
                cache: None,
            },
        )?;
        Ok(Prepared {
            pipeline,
            bindings: compiled.bind_group_layout.clone(),
            _layout: compiled.pipeline_layout.clone(),
        })
    }

    pub fn set_culling(&mut self, face: Context3DTriangleFace) {
        self.culling = face;
        self.dirty.set(true);
    }

    pub fn update_blend_factors(
        &mut self,
        color_component: wgpu::BlendComponent,
        alpha_component: wgpu::BlendComponent,
    ) {
        if color_component != self.color_component || alpha_component != self.alpha_component {
            self.color_component = color_component;
            self.alpha_component = alpha_component;
            self.dirty.set(true);
        }
    }

    pub(crate) fn update_sampler_state_at(
        &mut self,
        sampler: usize,
        wrap: ruffle_render::backend::Context3DWrapMode,
        filter: ruffle_render::backend::Context3DTextureFilter,
    ) {
        let sampler_config = SamplerConfig {
            wrapping: match wrap {
                Context3DWrapMode::Clamp => Wrapping::Clamp,
                Context3DWrapMode::Repeat => Wrapping::Repeat,
                Context3DWrapMode::ClampURepeatV => Wrapping::ClampURepeatV,
                Context3DWrapMode::RepeatUClampV => Wrapping::RepeatUClampV,
            },
            filter: match filter {
                Context3DTextureFilter::Linear => Filter::Linear,
                Context3DTextureFilter::Nearest => Filter::Nearest,
                Context3DTextureFilter::Anisotropic2X => Filter::Anisotropic2x,
                Context3DTextureFilter::Anisotropic4X => Filter::Anisotropic4x,
                Context3DTextureFilter::Anisotropic8X => Filter::Anisotropic8x,
                Context3DTextureFilter::Anisotropic16X => Filter::Anisotropic16x,
            },
            // FIXME - implement this
            mipmap: naga_agal::Mipmap::Disable,
        };
        if !self.reuse {
            self.dirty.set(true);
        }
        if self.sampler_configs[sampler] != sampler_config {
            self.bindings_dirty = true;
        }
        self.sampler_configs[sampler] = sampler_config;
    }

    pub fn update_stencil_actions(
        &mut self,
        triangle_face: Context3DTriangleFace,
        compare_mode: Context3DCompareMode,
        on_both_pass: Context3DStencilAction,
        on_depth_fail: Context3DStencilAction,
        on_depth_pass_stencil_fail: Context3DStencilAction,
    ) {
        let stencil_state = wgpu::StencilFaceState {
            compare: compare_mode.into_wgpu(),
            fail_op: on_depth_pass_stencil_fail.into_wgpu(),
            depth_fail_op: on_depth_fail.into_wgpu(),
            pass_op: on_both_pass.into_wgpu(),
        };

        let (front, back) = match triangle_face {
            Context3DTriangleFace::None => (
                wgpu::StencilFaceState::IGNORE,
                wgpu::StencilFaceState::IGNORE,
            ),
            Context3DTriangleFace::Front => (stencil_state, wgpu::StencilFaceState::IGNORE),
            Context3DTriangleFace::Back => (wgpu::StencilFaceState::IGNORE, stencil_state),
            Context3DTriangleFace::FrontAndBack => (stencil_state, stencil_state),
        };

        self.stencil.front = front;
        self.stencil.back = back;

        self.dirty.set(true);
    }

    pub fn update_stencil_reference_value(
        &mut self,
        reference_value: u32,
        read_mask: u32,
        write_mask: u32,
    ) {
        if self.stencil.read_mask != read_mask || self.stencil.write_mask != write_mask {
            self.dirty.set(true);
            self.stencil.read_mask = read_mask;
            self.stencil.write_mask = write_mask;
        }

        self.stencil_ref_value = reference_value;
    }
}

// This is useful for debugging shader issues
#[expect(dead_code)]
fn to_wgsl(module: &naga::Module) -> String {
    let mut out = String::new();

    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    let module_info = validator
        .validate(module)
        .unwrap_or_else(|e| panic!("Validation failed: {e:#?}"));

    let mut writer =
        naga::back::wgsl::Writer::new(&mut out, naga::back::wgsl::WriterFlags::EXPLICIT_TYPES);

    writer.write(module, &module_info).expect("Writing failed");
    out
}
