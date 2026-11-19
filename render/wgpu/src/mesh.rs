use crate::backend::WgpuRenderBackend;
use crate::target::RenderTarget;
use crate::{Descriptors, GradientUniforms, as_texture};
use std::any::Any;
use std::ops::Range;

use crate::buffer_builder::BufferBuilder;
use ruffle_render::backend::{ShapeHandle, ShapeHandleImpl};
use ruffle_render::bitmap::BitmapSource;
use ruffle_render::tessellator::{Bitmap, DrawType as TessDrawType, Gradient};
use swf::{CharacterId, GradientInterpolation};

/// How big to make gradient textures. Larger will keep more detail, but be slower and use more memory.
const GRADIENT_SIZE: usize = 256;

#[derive(Debug)]
pub struct Mesh {
    pub bounds: Option<[f64; 4]>,
    pub retained_bytes: Option<usize>,
    pub draws: Vec<Draw>,
    pub vertex_buffer: crate::gpu_resources::Buffer,
    pub index_buffer: crate::gpu_resources::Buffer,
    pub _allocation: ruffle_render::profile::GpuMesh,
}

/// Identical small solid meshes can use the existing adjacent instance batching.
/// Only weak references live here: a cache lookup cannot extend an unloaded
/// graphic's lifetime. Keys are exact bytes, including all index/mask boundaries.
#[derive(Default)]
pub(crate) struct SmallMeshCache {
    entries: fnv::FnvHashMap<Vec<u8>, std::sync::Weak<Mesh>>,
    lookups: u32,
}

impl SmallMeshCache {
    pub fn key(mesh: &crate::prepare::mesh::PreparedMesh) -> Option<Vec<u8>> {
        if !mesh.gradients.is_empty() {
            return None;
        }
        let mut bytes = 4usize;
        for draw in &mesh.draws {
            if !matches!(draw.draw_type, TessDrawType::Color) {
                return None;
            }
            bytes = bytes
                .checked_add(12)?
                .checked_add(draw.vertices.bytes().len())?
                .checked_add(draw.indices.len().checked_mul(4)?)?;
            // Large/curved meshes avoid hashing and retaining a second CPU copy.
            if bytes > 512 {
                return None;
            }
        }
        let mut key = Vec::with_capacity(bytes);
        key.extend_from_slice(&(mesh.draws.len() as u32).to_le_bytes());
        for draw in &mesh.draws {
            key.extend_from_slice(&(draw.vertices.bytes().len() as u32).to_le_bytes());
            key.extend_from_slice(&(draw.indices.len() as u32).to_le_bytes());
            key.extend_from_slice(&draw.mask_index_count.to_le_bytes());
            key.extend_from_slice(draw.vertices.bytes());
            key.extend_from_slice(bytemuck::cast_slice(&draw.indices));
        }
        Some(key)
    }

    pub fn get(&mut self, key: &[u8]) -> Option<std::sync::Arc<Mesh>> {
        self.lookups = self.lookups.wrapping_add(1);
        if self.lookups.is_multiple_of(64) {
            self.entries.retain(|_, mesh| mesh.strong_count() > 0);
        }
        let mesh = self.entries.get(key).and_then(std::sync::Weak::upgrade);
        if mesh.is_none() {
            self.entries.remove(key);
        }
        mesh
    }

    pub fn insert(&mut self, key: Vec<u8>, mesh: &std::sync::Arc<Mesh>) {
        // At most 512 KiB of byte keys plus bounded hash/weak metadata. This is
        // an identity index, with no GPU residency or strong owner retention.
        if self.entries.len() < 1024 {
            self.entries.insert(key, std::sync::Arc::downgrade(mesh));
        }
    }
}

/// A reloaded movie may still have a live immutable shape on this renderer.
/// The character ID selects a slot. Only the full definition and exact scale
/// establish a hit. Neither the definition nor its renderer handle is rooted.
#[derive(Default)]
pub(crate) struct DefinitionMeshes(Vec<Option<DefinitionMesh>>);

struct DefinitionMesh {
    shape: std::rc::Weak<swf::Shape>,
    scale: u32,
    handle: std::sync::Weak<dyn ShapeHandleImpl>,
}

impl DefinitionMeshes {
    const SLOTS: usize = 64;

    pub fn get(&mut self, shape: &std::rc::Rc<swf::Shape>, scale: f32) -> Option<ShapeHandle> {
        let slot = self
            .0
            .get_mut(usize::from(shape.id) % Self::SLOTS)?
            .as_mut()?;
        if slot.scale != scale.to_bits() || *slot.shape.upgrade()? != **shape {
            return None;
        }
        let handle = slot.handle.upgrade()?;
        // The next lookup can use this definition after the previous movie is collected.
        slot.shape = std::rc::Rc::downgrade(shape);
        Some(ShapeHandle(handle))
    }

    pub fn insert(&mut self, shape: &std::rc::Rc<swf::Shape>, scale: f32, handle: &ShapeHandle) {
        if self.0.is_empty() {
            self.0.resize_with(Self::SLOTS, || None);
        }
        self.0[usize::from(shape.id) % Self::SLOTS] = Some(DefinitionMesh {
            shape: std::rc::Rc::downgrade(shape),
            scale: scale.to_bits(),
            handle: std::sync::Arc::downgrade(&handle.0),
        });
    }
}

impl ShapeHandleImpl for Mesh {
    fn retained_bytes(&self) -> Option<usize> {
        self.retained_bytes
    }
    fn bounds(&self) -> Option<[f64; 4]> {
        self.bounds
    }
}

pub fn retained_mesh_bytes(
    buffer_bytes: u64,
    gradient_count: usize,
    draw_capacity: usize,
) -> Option<usize> {
    // Charge variable CPU storage explicitly. The per-draw allowance covers
    // backend bindings/handles; solid draws are intentionally overcounted.
    usize::try_from(buffer_bytes)
        .ok()?
        .checked_add(gradient_count.checked_mul(2048)?)?
        .checked_add(draw_capacity.checked_mul(std::mem::size_of::<Draw>() + 1024)?)?
        .checked_add(4096)
}

#[cfg(test)]
mod accounting_tests {
    use super::*;

    #[test]
    fn retained_mesh_accounts_for_large_solid_draw_storage() {
        let mut draws = Vec::with_capacity(8192);
        for i in 0..8192 {
            draws.push(Draw {
                draw_type: DrawType::Color,
                vertices: (i * 16)..((i + 1) * 16),
                indices: 0..6,
                mask_indices: 0..6,
            });
        }
        let bytes = retained_mesh_bytes(1024, 0, draws.capacity()).unwrap();
        assert!(bytes >= 1024 + draws.capacity() * std::mem::size_of::<Draw>());
    }
}

pub fn as_mesh(handle: &ShapeHandle) -> Result<&Mesh, crate::gpu_runtime::RenderFailure> {
    let value = &*handle.0 as &dyn Any;
    if let Some(mesh) = value.downcast_ref::<Mesh>() {
        return Ok(mesh);
    }
    value
        .downcast_ref::<crate::shape_recipe::ShapeRecipe>()
        .and_then(|recipe| recipe.mesh.get())
        .map(std::sync::Arc::as_ref)
        .ok_or_else(|| {
            crate::gpu_runtime::RenderFailure::new(
                crate::gpu_runtime::RenderFailureKind::Internal,
                "shape was not prepared before encoding",
            )
        })
}

#[derive(Debug)]
pub struct PendingDraw {
    pub draw_type: PendingDrawType,
    pub vertices: Range<wgpu::BufferAddress>,
    pub num_indices: u32,
    pub num_mask_indices: u32,
    pub index_offset: u32,
}

impl PendingDraw {
    pub fn finish(
        self,
        descriptors: &Descriptors,
        uniform_buffer: &crate::gpu_resources::Buffer,
        gradients: &[CommonGradient],
    ) -> Result<Draw, crate::gpu_runtime::RenderFailure> {
        Ok(Draw {
            draw_type: self
                .draw_type
                .finish(descriptors, uniform_buffer, gradients)?,
            vertices: self.vertices,
            indices: self.index_offset..(self.index_offset + self.num_indices),
            mask_indices: self.index_offset..(self.index_offset + self.num_mask_indices),
        })
    }
}

#[derive(Debug)]
pub struct Draw {
    pub draw_type: DrawType,
    pub vertices: Range<wgpu::BufferAddress>,
    pub indices: Range<u32>,
    pub mask_indices: Range<u32>,
}

impl PendingDraw {
    #[expect(clippy::too_many_arguments)]
    pub fn new<T: RenderTarget>(
        backend: &mut WgpuRenderBackend<T>,
        source: &dyn BitmapSource,
        draw: crate::prepare::mesh::PackedDraw,
        shape_id: CharacterId,
        draw_id: usize,
        vertex_buffer: &mut BufferBuilder,
        index_buffer: &mut BufferBuilder,
        indices_so_far: &mut u32,
    ) -> Result<Option<Self>, crate::gpu_runtime::RenderFailure> {
        let vertices = vertex_buffer.add(draw.vertices.bytes()).map_err(|_| {
            crate::gpu_runtime::RenderFailure::new(
                crate::gpu_runtime::RenderFailureKind::BudgetExceeded,
                "shape buffer exceeds the native device limit",
            )
        })?;

        index_buffer.add(&draw.indices).map_err(|_| {
            crate::gpu_runtime::RenderFailure::new(
                crate::gpu_runtime::RenderFailureKind::BudgetExceeded,
                "shape buffer exceeds the native device limit",
            )
        })?;

        let index_count = u32::try_from(draw.indices.len()).map_err(|_| {
            crate::gpu_runtime::RenderFailure::new(
                crate::gpu_runtime::RenderFailureKind::BudgetExceeded,
                "shape index count exceeds the native device limit",
            )
        })?;
        let index_offset = *indices_so_far;
        *indices_so_far = indices_so_far.checked_add(index_count).ok_or_else(|| {
            crate::gpu_runtime::RenderFailure::new(
                crate::gpu_runtime::RenderFailureKind::BudgetExceeded,
                "shape index offset exceeds the native device limit",
            )
        })?;
        let draw_type = match draw.draw_type {
            TessDrawType::Color => PendingDrawType::color(),
            TessDrawType::Gradient {
                matrix: _,
                gradient,
            } => PendingDrawType::gradient(gradient, shape_id, draw_id),
            TessDrawType::Bitmap(bitmap) => match PendingDrawType::bitmap(bitmap, source, backend)?
            {
                Some(draw) => draw,
                None => return Ok(None),
            },
        };
        Ok(Some(PendingDraw {
            draw_type,
            vertices,
            index_offset,
            num_indices: index_count,
            num_mask_indices: draw.mask_index_count,
        }))
    }
}

#[derive(Debug)]
pub enum PendingDrawType {
    Color,
    Gradient {
        gradient_index: usize,
        bind_group_label: Option<String>,
    },
    Bitmap {
        binds: BitmapBinds,
    },
}

/// Converts an RGBA color from sRGB space to linear color space.
fn srgb_to_linear(color: f32) -> f32 {
    if color <= 0.04045 {
        color / 12.92
    } else {
        f32::powf((color + 0.055) / 1.055, 2.4)
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

impl PendingDrawType {
    pub fn color() -> Self {
        PendingDrawType::Color
    }

    pub fn gradient(gradient_index: usize, shape_id: CharacterId, draw_id: usize) -> Self {
        let bind_group_label =
            create_debug_label!("Shape {} (gradient) draw {} bindgroup", shape_id, draw_id);
        PendingDrawType::Gradient {
            gradient_index,
            bind_group_label,
        }
    }

    pub fn bitmap<T: RenderTarget>(
        bitmap: Bitmap,
        source: &dyn BitmapSource,
        backend: &mut WgpuRenderBackend<T>,
    ) -> Result<Option<Self>, crate::gpu_runtime::RenderFailure> {
        let Some(handle) = source.bitmap_handle(bitmap.bitmap_id, backend) else {
            return Ok(None);
        };
        let texture = as_texture(&handle, &backend.descriptors.device)?;
        let binds = texture.bind_group(
            bitmap.is_repeating,
            bitmap.is_smoothed,
            &backend.descriptors.device,
            &backend.descriptors.bind_layouts.bitmap,
            handle.clone(),
            &backend.descriptors.bitmap_samplers,
        )?;

        Ok(Some(PendingDrawType::Bitmap {
            binds: binds.clone(),
        }))
    }

    pub fn finish(
        self,
        descriptors: &Descriptors,
        uniform_buffer: &crate::gpu_resources::Buffer,
        gradients: &[CommonGradient],
    ) -> Result<DrawType, crate::gpu_runtime::RenderFailure> {
        Ok(match self {
            PendingDrawType::Color => DrawType::Color,
            PendingDrawType::Gradient {
                gradient_index,
                bind_group_label,
            } => {
                let common = &gradients[gradient_index];
                let bind_group =
                    descriptors
                        .device
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            layout: &descriptors.bind_layouts.gradient,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                        buffer: uniform_buffer,
                                        offset: common.buffer_offset,
                                        size: wgpu::BufferSize::new(std::mem::size_of::<
                                            GradientUniforms,
                                        >(
                                        )
                                            as u64),
                                    }),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(
                                        &common.texture_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::Sampler(
                                        descriptors.bitmap_samplers.get_sampler(false, true),
                                    ),
                                },
                            ],
                            label: bind_group_label.as_deref(),
                        })?;
                DrawType::Gradient { bind_group }
            }
            PendingDrawType::Bitmap { binds } => DrawType::Bitmap { binds },
        })
    }
}

#[derive(Debug)]
pub enum DrawType {
    Color,
    Gradient {
        bind_group: crate::gpu_resources::BindGroup,
    },
    Bitmap {
        binds: BitmapBinds,
    },
}

#[derive(Debug)]
pub struct CommonGradient {
    texture_view: crate::gpu_resources::TextureView,
    buffer_offset: wgpu::BufferAddress,
}

impl CommonGradient {
    pub fn new(
        descriptors: &Descriptors,
        device: &crate::gpu_resources::Device,
        gradient: Gradient,
        uniform_buffers: &mut BufferBuilder,
    ) -> Result<Self, crate::gpu_runtime::RenderFailure> {
        let colors = if gradient.records.is_empty() {
            [0; GRADIENT_SIZE * 4]
        } else {
            let mut colors = [0; GRADIENT_SIZE * 4];
            let mut last = 0;
            let mut next;

            let convert = if gradient.interpolation == GradientInterpolation::LinearRgb {
                |c| srgb_to_linear(c / 255.0) * 255.0
            } else {
                |c| c
            };

            for t in 0..GRADIENT_SIZE {
                if last + 1 < gradient.records.len()
                    && t > gradient.records[last + 1].ratio as usize
                {
                    last += 1;
                }
                next = (last + 1).min(gradient.records.len() - 1);

                assert!(last == next || last + 1 == next);

                let last_record = &gradient.records[last];
                let next_record = &gradient.records[next];

                let a = if t <= last_record.ratio as usize || last_record.ratio == next_record.ratio
                {
                    // We are before the first gradient record,
                    // or this record's ratio is equal to the next one,
                    // meaning we need to do a full stop of this color for 1 pixel.
                    0.0
                } else if t > next_record.ratio as usize {
                    // We are after the last record
                    1.0
                } else {
                    (t as f32 - last_record.ratio as f32)
                        / (next_record.ratio as f32 - last_record.ratio as f32)
                };

                colors[t * 4] = lerp(
                    convert(last_record.color.r as f32),
                    convert(next_record.color.r as f32),
                    a,
                ) as u8;
                colors[(t * 4) + 1] = lerp(
                    convert(last_record.color.g as f32),
                    convert(next_record.color.g as f32),
                    a,
                ) as u8;
                colors[(t * 4) + 2] = lerp(
                    convert(last_record.color.b as f32),
                    convert(next_record.color.b as f32),
                    a,
                ) as u8;
                colors[(t * 4) + 3] =
                    lerp(last_record.color.a as f32, next_record.color.a as f32, a) as u8;
            }

            colors
        };
        let texture = device.create_texture_with_data(
            &descriptors.queue,
            &wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: GRADIENT_SIZE as u32,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &colors[..],
        )?;
        let view = texture.create_view(&Default::default())?;

        let buffer_offset = uniform_buffers
            .add(&[GradientUniforms::from(gradient)])
            .map_err(|_| {
                crate::gpu_runtime::RenderFailure::new(
                    crate::gpu_runtime::RenderFailureKind::BudgetExceeded,
                    "shape buffer exceeds the native device limit",
                )
            })?
            .start;

        Ok(Self {
            texture_view: view,
            buffer_offset,
        })
    }
}

#[derive(Debug, Clone)]
pub struct BitmapBinds {
    pub bind_group: crate::gpu_resources::BindGroup,
}

impl BitmapBinds {
    pub fn new(
        device: &crate::gpu_resources::Device,
        layout: &crate::gpu_resources::BindGroupLayout,
        sampler: &crate::gpu_resources::Sampler,
        texture_view: crate::gpu_resources::TextureView,
        label: Option<String>,
    ) -> Result<Self, crate::gpu_runtime::RenderFailure> {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
            label: label.as_deref(),
        })?;
        Ok(Self { bind_group })
    }
}
