use lru::LruCache;
use naga_agal::{AgalError, ParsedBytecode, SamplerConfig, VertexAttributeFormat};
use ruffle_render::backend::ShaderModule;
use std::{
    borrow::Cow,
    cell::{RefCell, RefMut},
    num::NonZeroUsize,
};
use wgpu::SamplerBindingType;

use super::MAX_VERTEX_ATTRIBUTES;

pub struct ShaderPairAgal {
    pub(super) upload_id: u64,
    vertex_shader: ParsedBytecode,

    fragment_shader: ParsedBytecode,
    fragment_sampler_configs: [Option<SamplerConfig>; 8],
    // Caches compiled wgpu shader modules. The cache key represents all of the data
    // that we need to pass to `naga_agal::agal_to_naga` to compile a shader.
    compiled: RefCell<LruCache<ShaderCompileData, CompiledShaderProgram>>,
}

impl ShaderModule for ShaderPairAgal {}

pub struct CompiledShaderProgram {
    pub vertex_module: crate::gpu_resources::ShaderModule,
    pub fragment_module: crate::gpu_resources::ShaderModule,
    pub bind_group_layout: crate::gpu_resources::BindGroupLayout,
    pub pipeline_layout: crate::gpu_resources::PipelineLayout,
}

impl ShaderPairAgal {
    pub fn new(vertex_bytecode: Vec<u8>, fragment_bytecode: Vec<u8>) -> Result<Self, AgalError> {
        let vertex_shader = naga_agal::parse_bytecode(&vertex_bytecode)?;
        let fragment_shader = naga_agal::parse_bytecode(&fragment_bytecode)?;
        let fragment_sampler_configs = naga_agal::extract_sampler_configs(&fragment_shader)?;

        static NEXT_UPLOAD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Ok(Self {
            upload_id: NEXT_UPLOAD.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            vertex_shader,
            fragment_shader,
            fragment_sampler_configs,
            // TODO - figure out a good size for this cache.
            compiled: RefCell::new(LruCache::new(NonZeroUsize::new(2).unwrap())),
        })
    }

    pub fn fragment_sampler_configs(&self) -> &[Option<SamplerConfig>; 8] {
        &self.fragment_sampler_configs
    }

    pub fn compile(
        &self,
        device: &crate::gpu_resources::Device,
        data: ShaderCompileData,
    ) -> Result<RefMut<'_, CompiledShaderProgram>, super::PipelineError> {
        let compiled = self.compiled.borrow_mut();
        let mut error = None;
        RefMut::filter_map(compiled, |compiled| {
            let result = compiled.try_get_or_insert_mut_with_key(data, |data| {
                let vertex_naga_module =
                    naga_agal::agal_to_naga(&self.vertex_shader, &data.vertex_attributes)?;
                let vertex_module = device.create_shader_module_for(
                    crate::pipeline_profile::Family::Stage3d,
                    wgpu::ShaderModuleDescriptor {
                        label: Some("AGAL vertex shader"),
                        source: wgpu::ShaderSource::Naga(Cow::Owned(vertex_naga_module)),
                    },
                )?;

                let fragment_naga_module =
                    naga_agal::agal_to_naga(&self.fragment_shader, &data.vertex_attributes)?;
                let fragment_module = device.create_shader_module_for(
                    crate::pipeline_profile::Family::Stage3d,
                    wgpu::ShaderModuleDescriptor {
                        label: Some("AGAL fragment shader"),
                        source: wgpu::ShaderSource::Naga(Cow::Owned(fragment_naga_module)),
                    },
                )?;

                let mut layout_entries = vec![
                    // Vertex shader program constants
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Fragment shader program constants
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ];

                for (i, texture_info) in data.texture_infos.iter().enumerate() {
                    if let Some(texture_info) = texture_info {
                        let dimension = match texture_info {
                            ShaderTextureInfo::D2 => wgpu::TextureViewDimension::D2,
                            ShaderTextureInfo::Cube => wgpu::TextureViewDimension::Cube,
                        };
                        layout_entries.push(wgpu::BindGroupLayoutEntry {
                            binding: naga_agal::TEXTURE_START_BIND_INDEX + i as u32,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: dimension,
                                multisampled: false,
                            },
                            count: None,
                        });
                        layout_entries.push(wgpu::BindGroupLayoutEntry {
                            binding: naga_agal::TEXTURE_SAMPLER_START_BIND_INDEX + i as u32,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(SamplerBindingType::Filtering),
                            count: None,
                        });
                    }
                }

                let globals_layout_label = create_debug_label!("Globals bind group layout");
                let bind_group_layout =
                    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: globals_layout_label.as_deref(),
                        entries: &layout_entries,
                    })?;

                let pipeline_layout =
                    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Stage3D program layout"),
                        bind_group_layouts: &[Some(&bind_group_layout)],
                        immediate_size: 0,
                    })?;
                Ok(CompiledShaderProgram {
                    pipeline_layout,
                    vertex_module,
                    fragment_module,
                    bind_group_layout,
                })
            });
            result.map_err(|e| error = Some(e)).ok()
        })
        .map_err(|_| error.expect("Compilation should have failed"))
    }
}

#[derive(Hash, PartialEq, Eq, Clone, Copy)]
pub enum ShaderTextureInfo {
    D2,
    Cube,
}

#[derive(Hash, Eq, PartialEq, Clone)]
pub struct ShaderCompileData {
    pub vertex_attributes: [Option<VertexAttributeFormat>; MAX_VERTEX_ATTRIBUTES],
    pub texture_infos: [Option<ShaderTextureInfo>; 8],
}
