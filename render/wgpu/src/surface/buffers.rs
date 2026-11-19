use crate::Transforms;
use crate::buffer_builder::{BufferBuilder, BufferFull};
use crate::descriptors::{Descriptors, Quad};
use crate::gpu_resources::StagingBelt;
use ruffle_render::transform::Transform;
use std::ops::Range;

// [NA] The higher this value is, the more memory we'll reserve - ideally we can dynamically allocate this
// but for now this feels reasonable
const MAX_TRANSFORMS_PER_CHUNK: u64 = 512;

pub struct ChunkBuffers {
    pub transforms: crate::gpu_resources::Buffer,
    pub vertex_buffer: crate::gpu_resources::Buffer,
    pub bind_group: crate::gpu_resources::BindGroup,
    pub transform_offset: u32,
    pub vertex_offset: u64,
    arena: crate::frame_upload::FrameUploadArena,
    scratch: crate::buffer_builder::ScratchPool,
}

pub fn transforms_per_draw(limits: &wgpu::Limits) -> u64 {
    let maximum = limits
        .max_uniform_buffer_binding_size
        .min(limits.max_buffer_size)
        / size_of::<Transforms>() as u64;
    maximum.min(MAX_TRANSFORMS_PER_CHUNK)
}

impl ChunkBuffers {
    pub fn new(descriptors: &Descriptors) -> Result<Self, crate::gpu_runtime::RenderFailure> {
        let transforms = descriptors.device.create_buffer(&wgpu::BufferDescriptor {
            label: create_debug_label!("Chunk transform buffer").as_deref(),
            size: transforms_per_draw(&descriptors.limits) * size_of::<Transforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })?;
        let vertex_buffer = descriptors.device.create_buffer(&wgpu::BufferDescriptor {
            label: create_debug_label!("Chunk vertex buffer").as_deref(),
            size: 1024,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })?;
        let bind_group = descriptors
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: create_debug_label!("Chunk transforms bind group").as_deref(),
                layout: &descriptors.bind_layouts.transforms,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: transforms.as_entire_binding(),
                }],
            })?;
        Ok(Self {
            transform_offset: 0,
            vertex_offset: 0,
            arena: crate::frame_upload::FrameUploadArena::new(descriptors.device.clone(), 65536),
            scratch: Default::default(),
            transforms,
            bind_group,
            vertex_buffer,
        })
    }
}

#[derive(Clone, Debug)]
pub enum DrawVertices<'a> {
    CustomPosUvQuad(&'a [u8]),
    PosUvQuad,
    PosColorQuad,
    PosQuad,
    Line,
    RectLine,
    CustomBuffers {
        vertex_buffer: crate::gpu_resources::Buffer,
        vertex_range: Range<wgpu::BufferAddress>,
        index_buffer: crate::gpu_resources::Buffer,
        indices: Range<u32>,
    },
}

#[derive(Clone, Debug)]
pub struct DrawOffsets {
    pub vertex_buffer: Option<crate::gpu_resources::Buffer>,
    pub index_buffer: crate::gpu_resources::Buffer,
    pub vertices: Range<wgpu::BufferAddress>,
    pub indices: Range<u32>,
    pub instances: Range<u32>,
}

impl DrawOffsets {
    pub fn try_merge(&mut self, other: &DrawOffsets, primitive_size: u32) -> bool {
        if self.vertex_buffer == other.vertex_buffer
            && self.index_buffer == other.index_buffer
            && self.vertices == other.vertices
            && self.indices == other.indices
            && self.instances.end == other.instances.start
        {
            // It's the same exact draw and contiguous instances, let's just engulf the instances
            self.instances.end = other.instances.end;
            return true;
        }

        // With one identical instance, adjacent index ranges are the same
        // primitive sequence. Multi-instance ranges would reorder primitives.
        if self.vertex_buffer == other.vertex_buffer
            && self.index_buffer == other.index_buffer
            && self.vertices == other.vertices
            && self.instances == other.instances
            && self.instances.end - self.instances.start == 1
            && self.indices.end == other.indices.start
            && (self.indices.end - self.indices.start).is_multiple_of(primitive_size)
            && (other.indices.end - other.indices.start).is_multiple_of(primitive_size)
        {
            self.indices.end = other.indices.end;
            return true;
        }
        false
    }
}

pub struct ChunkBufferBuilder {
    transforms: BufferBuilder,
    vertices: BufferBuilder,
}

impl ChunkBufferBuilder {
    pub fn new(chunk_buffers: &ChunkBuffers, limits: &wgpu::Limits) -> Self {
        let mut transforms = BufferBuilder::with_scratch(limits, 0, chunk_buffers.scratch.clone());
        transforms.set_buffer_limit(transforms_per_draw(limits) * size_of::<Transforms>() as u64);
        let vertices = BufferBuilder::with_scratch(limits, 0, chunk_buffers.scratch.clone());
        Self {
            transforms,
            vertices,
        }
    }

    pub(super) fn into_instance_bytes(self) -> Option<Box<[u8]>> {
        (self.vertices.len() == 0).then(|| self.transforms.into_storage())
    }
    pub(super) fn append_instance_bytes(&mut self, bytes: &[u8]) -> Result<u32, BufferFull> {
        let range = self.transforms.add(bytes)?;
        Ok((range.start / size_of::<Transforms>() as u64) as u32)
    }
    pub fn add_transform(&mut self, transform: &Transform) -> Result<u32, BufferFull> {
        let instance = Transforms {
            world_matrix: [
                [transform.matrix.a, transform.matrix.b, 0.0, 0.0],
                [transform.matrix.c, transform.matrix.d, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [
                    transform.matrix.tx.to_pixels() as f32,
                    transform.matrix.ty.to_pixels() as f32,
                    transform.tz as f32,
                    1.0,
                ],
            ],
            mult_color: transform.color_transform.mult_rgba_normalized(),
            add_color: transform.color_transform.add_rgba_normalized(),
        };
        let address = self.transforms.add(&[instance])?;
        Ok((address.start as usize / size_of::<Transforms>()) as u32)
    }

    pub fn add_draw<'a>(
        &mut self,
        quad: &Quad,
        instance: u32,
        vertices: DrawVertices<'a>,
    ) -> Result<DrawOffsets, (BufferFull, DrawVertices<'a>)> {
        let instances = instance..instance + 1;
        Ok(match vertices {
            DrawVertices::CustomPosUvQuad(vertex_bytes) => DrawOffsets {
                vertex_buffer: None,
                index_buffer: quad.index_buffer.clone(),
                vertices: self
                    .vertices
                    .add(vertex_bytes)
                    .map_err(|_| (BufferFull, vertices))?,
                indices: quad.indices_quad.clone(),
                instances,
            },
            DrawVertices::PosQuad => DrawOffsets {
                vertex_buffer: Some(quad.vertex_buffer.clone()),
                index_buffer: quad.index_buffer.clone(),
                vertices: quad.vertices_pos.clone(),
                indices: quad.indices_quad.clone(),
                instances,
            },
            DrawVertices::PosColorQuad => DrawOffsets {
                vertex_buffer: Some(quad.vertex_buffer.clone()),
                index_buffer: quad.index_buffer.clone(),
                vertices: quad.vertices_pos_color.clone(),
                indices: quad.indices_quad.clone(),
                instances,
            },
            DrawVertices::PosUvQuad => DrawOffsets {
                vertex_buffer: Some(quad.vertex_buffer.clone()),
                index_buffer: quad.index_buffer.clone(),
                vertices: quad.vertices_pos_uv.clone(),
                indices: quad.indices_quad.clone(),
                instances,
            },
            DrawVertices::Line => DrawOffsets {
                vertex_buffer: Some(quad.vertex_buffer.clone()),
                index_buffer: quad.index_buffer.clone(),
                vertices: quad.vertices_pos_color.clone(),
                indices: quad.indices_line.clone(),
                instances,
            },
            DrawVertices::RectLine => DrawOffsets {
                vertex_buffer: Some(quad.vertex_buffer.clone()),
                index_buffer: quad.index_buffer.clone(),
                vertices: quad.vertices_pos_color.clone(),
                indices: quad.indices_line_rect.clone(),
                instances,
            },
            DrawVertices::CustomBuffers {
                vertex_buffer,
                vertex_range,
                index_buffer,
                indices,
            } => DrawOffsets {
                vertex_buffer: Some(vertex_buffer),
                index_buffer,
                vertices: vertex_range,
                indices,
                instances,
            },
        })
    }

    pub fn copy_to(
        self,
        staging_belt: &mut StagingBelt,
        descriptors: &Descriptors,
        draw_encoder: &mut crate::gpu_resources::CommandEncoder,
        chunk_buffers: &mut ChunkBuffers,
    ) -> Result<(), crate::gpu_runtime::RenderFailure> {
        let uniform_size =
            transforms_per_draw(&descriptors.limits) * size_of::<Transforms>() as u64;
        let uniforms = chunk_buffers.arena.allocate_uniform(
            self.transforms.len() as u64,
            uniform_size,
            u64::from(descriptors.limits.min_uniform_buffer_offset_alignment),
        )?;
        let vertices = if self.vertices.len() == 0 {
            None
        } else {
            Some(
                chunk_buffers
                    .arena
                    .allocate_dynamic(self.vertices.len() as u64, 4)?,
            )
        };
        let transform_offset = u32::try_from(uniforms.offset).map_err(|_| {
            crate::gpu_runtime::RenderFailure::new(
                crate::gpu_runtime::RenderFailureKind::Validation,
                "transform slice exceeds dynamic offset range",
            )
        })?;
        // Reserve both destinations before encoding either write.
        let bind_group = descriptors.cached_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Paged chunk transforms"),
            layout: &descriptors.bind_layouts.transforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniforms.buffer,
                    // All aligned windows in this page share the same binding.
                    // The pass selects the reserved window through a dynamic offset.
                    offset: 0,
                    size: wgpu::BufferSize::new(uniform_size),
                }),
            }],
        })?;
        self.transforms.copy_to_at(
            staging_belt,
            draw_encoder,
            &uniforms.buffer,
            uniforms.offset,
        )?;
        if let Some(vertices) = vertices {
            self.vertices.copy_to_at(
                staging_belt,
                draw_encoder,
                &vertices.buffer,
                vertices.offset,
            )?;
            chunk_buffers.vertex_buffer = vertices.buffer;
            chunk_buffers.vertex_offset = vertices.offset;
        }
        chunk_buffers.transforms = uniforms.buffer;
        chunk_buffers.transform_offset = transform_offset;
        chunk_buffers.bind_group = bind_group;

        Ok(())
    }
}
