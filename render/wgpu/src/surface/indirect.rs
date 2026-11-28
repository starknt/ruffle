//! GPU-generated, ordered draws within an already-compatible bitmap material run.
use super::{
    buffers::{ChunkBuffers, transforms_per_draw},
    chunk::DrawCommand,
};
use crate::capabilities::BitmapDraw;
use crate::gpu_resources::{
    BindGroupLayout, Buffer, CommandEncoder, ComputePipeline, RenderPass, try_sync_cell,
};
use crate::gpu_runtime::RenderFailure;
use crate::{Descriptors, Transforms, globals::Globals, layouts::BITMAP_BATCH_TEXTURES};
use std::sync::OnceLock;

pub(crate) type Pipeline = OnceLock<(ComputePipeline, BindGroupLayout)>;
const ARG_BYTES: u64 = size_of::<wgpu::util::DrawIndexedIndirectArgs>() as u64;

#[derive(Debug)]
pub struct Draw {
    buffer: Buffer,
    offset: u64,
    max_draws: u32,
    pub(super) mode: BitmapDraw,
}

impl Draw {
    pub(super) fn encode(&self, pass: &mut RenderPass<'_>) {
        match self.mode {
            BitmapDraw::Indirect => pass.draw_indexed_indirect(&self.buffer, self.offset),
            BitmapDraw::Cull => {
                pass.multi_draw_indexed_indirect(&self.buffer, self.offset, self.max_draws)
            }
            BitmapDraw::Count => pass.multi_draw_indexed_indirect_count(
                &self.buffer,
                self.offset,
                self.offset + u64::from(self.max_draws) * ARG_BYTES,
                self.max_draws,
            ),
            _ => unreachable!("a direct draw has no indirect arguments"),
        }
    }
}

pub(super) fn prepare(
    commands: &mut [DrawCommand],
    descriptors: &Descriptors,
    encoder: &mut CommandEncoder,
    buffers: &ChunkBuffers,
    globals: &Globals,
) -> Result<(), RenderFailure> {
    let mode = descriptors.bitmap_draw;
    let max_draws = if mode.culls() {
        BITMAP_BATCH_TEXTURES.div_ceil(2) as u32
    } else {
        1
    };
    let stride = u64::from(max_draws) * ARG_BYTES + 4;
    let mut inputs = Vec::new();
    let mut draws = Vec::new();
    for command in commands {
        if let DrawCommand::RenderTextureBatch {
            draw_offsets,
            indirect,
            ..
        } = command
        {
            // The culling shader and the vertex shader must use the same quad.
            if mode.culls()
                && (draw_offsets.vertex_buffer.as_ref() != Some(&descriptors.quad.vertex_buffer)
                    || draw_offsets.vertices != descriptors.quad.vertices_pos_uv)
                || draw_offsets.instances.start != 0
                    && !descriptors.capabilities.indirect_first_instance
            {
                continue;
            }
            inputs.push([
                draw_offsets.indices.start,
                draw_offsets.indices.len() as u32,
                draw_offsets.instances.start,
                draw_offsets.instances.len() as u32,
            ]);
            draws.push(indirect);
        }
    }
    let groups = inputs.len() as u32;
    if groups == 0 || groups > descriptors.capabilities.max_compute_groups {
        return Ok(());
    }

    let transforms = transforms_per_draw(&descriptors.limits);
    let (pipeline, layout) = try_sync_cell(&descriptors.bitmap_indirect, || {
        let device = &descriptors.persistent_device;
        let source = format!(
            "const MAX_DRAWS: u32 = {max_draws}u;\nconst MAX_TRANSFORMS: u32 = {transforms}u;\n{}",
            include_str!("../../shaders/bitmap_indirect.wgsl")
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Ordered bitmap arguments"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        })?;
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Ordered bitmap arguments"),
            layout: None,
            module: &shader,
            entry_point: Some("prepare"),
            compilation_options: Default::default(),
            cache: None,
        })?;
        let layout = pipeline.get_bind_group_layout(0)?;
        Ok::<_, RenderFailure>((pipeline, layout))
    })?;
    let device = &descriptors.device;
    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Ordered bitmap ranges"),
        contents: bytemuck::cast_slice(&inputs),
        usage: wgpu::BufferUsages::STORAGE,
    })?;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Ordered bitmap indirect arguments"),
        size: u64::from(groups) * stride,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::INDIRECT
            | if cfg!(test) {
                wgpu::BufferUsages::COPY_SRC
            } else {
                wgpu::BufferUsages::empty()
            },
        mapped_at_creation: false,
    })?;
    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Ordered bitmap arguments"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffers.transforms,
                    offset: u64::from(buffers.transform_offset),
                    size: wgpu::BufferSize::new(transforms * size_of::<Transforms>() as u64),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: globals.buffer.as_entire_binding(),
            },
        ],
    })?;
    encoder
        .begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Ordered bitmap arguments"),
            timestamp_writes: None,
        })
        .dispatch(pipeline, &bindings, [groups, 1]);
    for (index, draw) in draws.into_iter().enumerate() {
        *draw = Some(Draw {
            buffer: output.clone(),
            offset: index as u64 * stride,
            max_draws,
            mode,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::buffers::{ChunkBufferBuilder, DrawOffsets};
    use crate::{globals::Globals, gpu_resources::StagingBelt};
    use ruffle_render::transform::Transform;
    use std::time::Duration;

    #[test]
    #[ignore = "reads generated arguments on a native Vulkan device configured for bitmap culling"]
    fn gpu_visibility_preserves_spans_and_omits_an_entire_invisible_batch() {
        let descriptors = crate::surface::tests::gpu();
        if !descriptors.bitmap_draw.culls() {
            return;
        }
        let device = &descriptors.device;
        let globals = Globals::new(device, &descriptors.bind_layouts.globals, 20, 20).unwrap();
        let mut buffers = ChunkBuffers::new(&descriptors).unwrap();
        let mut chunk = ChunkBufferBuilder::new(&buffers, &descriptors.limits);
        // Leave instance zero unused. Separated runs must retain their original
        // indices because those indices are also bitmap material slots.
        for i in 0..33 {
            let x = if matches!(i, 2 | 3 | 5 | 7..=9) {
                3
            } else {
                30
            };
            let mut transform = Transform::default();
            transform.matrix.tx = swf::Twips::from_pixels_i32(x);
            chunk
                .add_transform(
                    &transform,
                    &crate::surface::buffers::DrawVertices::PosUvQuad,
                )
                .unwrap();
        }
        let mut encoder = device.create_command_encoder(&Default::default()).unwrap();
        let mut staging = StagingBelt::new(device.clone(), 65536);
        chunk
            .copy_to(&mut staging, &descriptors, &mut encoder, &mut buffers)
            .unwrap();
        let quad = &descriptors.quad;
        let mut commands = [1..17, 17..33].map(|instances| DrawCommand::RenderTextureBatch {
            _textures: vec![],
            indirect: None,
            binds: globals.bind_group().clone(),
            draw_offsets: DrawOffsets {
                vertex_buffer: Some(quad.vertex_buffer.clone()),
                index_buffer: quad.index_buffer.clone(),
                vertices: quad.vertices_pos_uv.clone(),
                indices: quad.indices_quad.clone(),
                instances,
            },
        });
        prepare(
            &mut commands,
            &descriptors,
            &mut encoder,
            &buffers,
            &globals,
        )
        .unwrap();
        let DrawCommand::RenderTextureBatch {
            indirect: Some(draw),
            ..
        } = &commands[0]
        else {
            panic!("missing GPU arguments")
        };
        let readback = device
            .create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: draw.buffer.size(),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
            .unwrap();
        encoder.copy_buffer_to_buffer(&draw.buffer, 0, &readback, 0, readback.size());
        staging.finish();
        let serial = descriptors
            .queue
            .submit([encoder.finish().unwrap()])
            .unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read, .., move |r| {
            tx.send(r).unwrap();
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(serial),
                timeout: Some(Duration::from_secs(5)),
            })
            .unwrap();
        rx.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
        let data = readback.get_mapped_range(..).unwrap();
        let words: &[u32] = bytemuck::cast_slice(&data);
        let stride = words.len() / 2;
        assert_eq!(words[stride - 1], 3);
        for (arg, (first, count)) in words[..15].chunks_exact(5).zip([(2, 2), (5, 1), (7, 3)]) {
            assert_eq!(
                arg,
                &[
                    quad.indices_quad.len() as u32,
                    count,
                    quad.indices_quad.start,
                    0,
                    first
                ]
            );
        }
        assert!(words[15..stride - 1].iter().all(|n| *n == 0));
        assert!(words[stride..].iter().all(|n| *n == 0));
        drop(data);
        readback.unmap();
    }
}
