//! Resident material/geometry draws. Instance ranges are local to a group;
//! replay reserves a fresh uniform window for every actual submission.
use super::{
    buffers::DrawOffsets,
    chunk::{Chunk, DrawCommand},
};
use crate::gpu_resources::BindGroup;
use ruffle_render::quality::StageQuality;
use std::ops::Range;

#[derive(Clone, Copy, PartialEq)]
pub(super) struct Key {
    pub device: u64,
    pub origin: [u32; 2],
    pub mask_geometry: bool,
    pub quality: StageQuality,
}

enum Material {
    Color,
    Line,
    Gradient(BindGroup),
}
pub(super) struct Draw {
    material: Material,
    offsets: DrawOffsets,
}
impl Draw {
    fn from_command(command: DrawCommand) -> Option<Self> {
        let (material, offsets) = match command {
            DrawCommand::DrawColor { draw_offsets } => (Material::Color, draw_offsets),
            DrawCommand::DrawLine { draw_offsets } => (Material::Line, draw_offsets),
            DrawCommand::RenderGradient {
                bind_group,
                draw_offsets,
            } => (Material::Gradient(bind_group), draw_offsets),
            _ => return None,
        };
        Some(Self { material, offsets })
    }
    pub fn in_window(&self, window: &Range<u32>, first: u32) -> Option<DrawCommand> {
        let range = &self.offsets.instances;
        let start = range.start.max(window.start);
        let end = range.end.min(window.end);
        if start >= end {
            return None;
        }
        let mut offsets = self.offsets.clone();
        offsets.instances = (first + (start - window.start))..(first + (end - window.start));
        Some(match &self.material {
            Material::Color => DrawCommand::DrawColor {
                draw_offsets: offsets,
            },
            Material::Line => DrawCommand::DrawLine {
                draw_offsets: offsets,
            },
            Material::Gradient(bind_group) => DrawCommand::RenderGradient {
                bind_group: bind_group.clone(),
                draw_offsets: offsets,
            },
        })
    }
}

pub(super) struct Group {
    pub instances: ruffle_memory::MemoryBuffer<u8>,
    pub draws: Box<[Draw]>,
}
pub(super) struct PreparedDraws {
    pub key: Key,
    pub groups: Vec<Group>,
}
impl PreparedDraws {
    pub fn from_chunks(key: Key, chunks: Vec<Chunk>) -> Option<Self> {
        let mut groups = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let Chunk::Draw {
                chunk,
                chunk_buffer_builder,
                ..
            } = chunk
            else {
                return None;
            };
            let instances = chunk_buffer_builder.into_instance_bytes()?;
            let draws = chunk
                .into_iter()
                .map(Draw::from_command)
                .collect::<Option<Box<[_]>>>()?;
            // Replaying two adjacent uniform windows must preserve painter order.
            // Consecutive mesh draws can share their boundary instance.
            let mut last = 0;
            let count = instances.len() / size_of::<crate::Transforms>();
            for draw in &draws {
                let range = &draw.offsets.instances;
                if range.is_empty() || range.start < last || range.end as usize > count {
                    return None;
                }
                last = range.end - 1;
            }
            groups.push(Group { instances, draws });
        }
        Some(Self { key, groups })
    }
    pub fn bytes(&self) -> usize {
        size_of::<Self>()
            + self.groups.capacity() * size_of::<Group>()
            + self
                .groups
                .iter()
                .map(|g| g.instances.len() + size_of_val(&*g.draws) + 64)
                .sum::<usize>()
    }
}
