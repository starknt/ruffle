//! Resident material/geometry draws. Instance ranges are local to a group;
//! replay reserves a fresh uniform window for every actual submission.
use super::{
    buffers::DrawOffsets,
    chunk::{Chunk, DrawCommand},
};
use crate::gpu_resources::BindGroup;
use ruffle_render::quality::StageQuality;

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
    pub fn at(&self, first: u32) -> DrawCommand {
        let mut offsets = self.offsets.clone();
        offsets.instances = (first + offsets.instances.start)..(first + offsets.instances.end);
        match &self.material {
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
        }
    }
}

pub(super) struct Group {
    pub instances: Box<[u8]>,
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
