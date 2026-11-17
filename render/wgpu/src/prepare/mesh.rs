//! Owned, GPU-free tessellation and vertex packing.

use crate::gpu_runtime::{RenderFailure, RenderFailureKind};
use crate::{PosColorVertex, PosUvVertex};
use ruffle_render::tessellator::{DrawType, Gradient, Mesh};

#[derive(Clone)]
pub struct ShapeInput {
    pub geometry: std::sync::Arc<ruffle_render::shape_utils::OwnedDistilledShape>,
    pub scale: f32,
    pub bitmap_sizes: Vec<(u16, ruffle_render::bitmap::BitmapSize)>,
}
struct Sizes(Vec<(u16, ruffle_render::bitmap::BitmapSize)>);
impl ruffle_render::bitmap::BitmapSource for Sizes {
    fn bitmap_size(&self, id: u16) -> Option<ruffle_render::bitmap::BitmapSize> {
        self.0
            .iter()
            .find_map(|(key, value)| (*key == id).then_some(*value))
    }
    fn bitmap_handle(
        &self,
        _: u16,
        _: &mut dyn ruffle_render::backend::RenderBackend,
    ) -> Option<ruffle_render::bitmap::BitmapHandle> {
        None
    }
}
pub fn input_vertices(shape: &ruffle_render::shape_utils::OwnedDistilledShape) -> usize {
    use ruffle_render::shape_utils::{DrawCommand, OwnedDrawPath};
    shape
        .paths
        .iter()
        .map(|path| {
            let commands = match path {
                OwnedDrawPath::Fill { commands, .. } | OwnedDrawPath::Stroke { commands, .. } => {
                    commands
                }
            };
            commands
                .iter()
                .map(|command| match command {
                    DrawCommand::MoveTo(_) | DrawCommand::LineTo(_) => 1usize,
                    DrawCommand::QuadraticCurveTo { .. } => 2,
                    DrawCommand::CubicCurveTo { .. } => 3,
                })
                .fold(0usize, usize::saturating_add)
        })
        .fold(0usize, usize::saturating_add)
}
impl ShapeInput {
    pub fn prepare_bounded(self, bytes: u64) -> Result<PreparedMesh, RenderFailure> {
        self.prepare_with(
            &mut ruffle_render::tessellator::ShapeTessellator::new(),
            bytes,
        )
    }

    pub fn prepare_with(
        self,
        tessellator: &mut ruffle_render::tessellator::ShapeTessellator,
        bytes: u64,
    ) -> Result<PreparedMesh, RenderFailure> {
        let mesh = tessellator.tessellate_shape_with_scale_bounded(
            self.geometry.as_distilled(),
            &Sizes(self.bitmap_sizes),
            self.scale,
            bytes,
        )?;
        PreparedMesh::pack(mesh)
    }
}

pub struct PreparedMesh {
    pub draws: Vec<PackedDraw>,
    pub gradients: Vec<Gradient>,
    pub bounds: Option<[f64; 4]>,
}
pub struct PackedDraw {
    pub draw_type: DrawType,
    pub vertices: PackedVertices,
    pub indices: Vec<u32>,
    pub mask_index_count: u32,
}
pub enum PackedVertices {
    Color(Vec<PosColorVertex>),
    Texture(Vec<PosUvVertex>),
}
impl PackedVertices {
    fn retained_bytes(&self) -> u64 {
        match self {
            Self::Color(v) => (v.capacity() * std::mem::size_of::<PosColorVertex>()) as u64,
            Self::Texture(v) => (v.capacity() * std::mem::size_of::<PosUvVertex>()) as u64,
        }
    }
    pub fn bytes(&self) -> &[u8] {
        match self {
            Self::Color(v) => bytemuck::cast_slice(v),
            Self::Texture(v) => bytemuck::cast_slice(v),
        }
    }
}
fn allocated<T>(size: usize) -> Result<Vec<T>, RenderFailure> {
    let mut result = Vec::new();
    result.try_reserve_exact(size).map_err(|_| {
        RenderFailure::new(
            RenderFailureKind::OutOfMemory,
            "native shape preparation allocation failed",
        )
    })?;
    Ok(result)
}
impl PreparedMesh {
    pub fn retained_bytes(&self) -> u64 {
        let draws = (self.draws.capacity() * std::mem::size_of::<PackedDraw>()) as u64;
        let gradients = (self.gradients.capacity() * std::mem::size_of::<Gradient>()) as u64;
        self.draws
            .iter()
            .fold(draws.saturating_add(gradients), |bytes, draw| {
                bytes
                    .saturating_add(draw.vertices.retained_bytes())
                    .saturating_add((draw.indices.capacity() * std::mem::size_of::<u32>()) as u64)
            })
            .saturating_add(
                self.gradients
                    .iter()
                    .map(|g| {
                        (g.records.capacity() * std::mem::size_of::<swf::GradientRecord>()) as u64
                    })
                    .sum::<u64>(),
            )
    }
    pub fn pack(mesh: Mesh) -> Result<Self, RenderFailure> {
        let mut draws = allocated(mesh.draws.len())?;
        let mut bounds: Option<[f64; 4]> = None;
        for draw in mesh.draws {
            for vertex in &draw.vertices {
                let (x, y) = (f64::from(vertex.x), f64::from(vertex.y));
                let b = bounds.get_or_insert([x, y, x, y]);
                b[0] = b[0].min(x);
                b[1] = b[1].min(y);
                b[2] = b[2].max(x);
                b[3] = b[3].max(y);
            }
            let vertices = match &draw.draw_type {
                DrawType::Color => {
                    let mut vertices = allocated(draw.vertices.len())?;
                    vertices.extend(draw.vertices.into_iter().map(PosColorVertex::from));
                    PackedVertices::Color(vertices)
                }
                DrawType::Gradient { matrix, .. } => {
                    let mut vertices = allocated(draw.vertices.len())?;
                    vertices.extend(
                        draw.vertices
                            .into_iter()
                            .map(|v| PosUvVertex::from_tessellator(v, matrix)),
                    );
                    PackedVertices::Texture(vertices)
                }
                DrawType::Bitmap(bitmap) => {
                    let mut vertices = allocated(draw.vertices.len())?;
                    vertices.extend(
                        draw.vertices
                            .into_iter()
                            .map(|v| PosUvVertex::from_tessellator(v, &bitmap.matrix)),
                    );
                    PackedVertices::Texture(vertices)
                }
            };
            draws.push(PackedDraw {
                draw_type: draw.draw_type,
                vertices,
                indices: draw.indices,
                mask_index_count: draw.mask_index_count,
            });
        }
        Ok(Self {
            draws,
            gradients: mesh.gradients,
            bounds,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruffle_render::tessellator::{Draw, DrawType, Mesh, Vertex};
    use swf::Color;

    #[test]
    fn native_preparation_mesh_packing_preserves_uv_arithmetic_indices_and_bounds() {
        let prepared = PreparedMesh::pack(Mesh {
            gradients: vec![],
            draws: vec![
                Draw {
                    draw_type: DrawType::Color,
                    vertices: vec![Vertex {
                        x: -2.5,
                        y: 4.25,
                        color: Color {
                            r: 255,
                            g: 0,
                            b: 0,
                            a: 255,
                        },
                    }],
                    indices: vec![0, 0, 0],
                    mask_index_count: 3,
                },
                Draw {
                    draw_type: DrawType::Gradient {
                        gradient: 0,
                        matrix: [[2.0, 3.0, 0.0], [5.0, 7.0, 0.0], [11.0, 13.0, 1.0]],
                    },
                    vertices: vec![Vertex {
                        x: 1.0,
                        y: 2.0,
                        color: Color::WHITE,
                    }],
                    indices: vec![0, 0, 0],
                    mask_index_count: 0,
                },
            ],
        })
        .unwrap();
        assert_eq!(prepared.bounds, Some([-2.5, 2.0, 1.0, 4.25]));
        assert_eq!(
            prepared.draws[0].vertices.bytes(),
            bytemuck::bytes_of(&crate::PosColorVertex {
                position: [-2.5, 4.25],
                color: [1.0, 0.0, 0.0, 1.0]
            })
        );
        assert_eq!(
            prepared.draws[1].vertices.bytes(),
            bytemuck::bytes_of(&crate::PosUvVertex {
                position: [1.0, 2.0],
                uv: [23.0, 30.0, 1.0]
            })
        );
        assert_eq!(prepared.draws[1].indices, [0, 0, 0]);
        assert_eq!(prepared.draws[1].mask_index_count, 0);
    }
}

#[cfg(test)]
#[test]
fn native_preparation_rejects_tessellation_output_over_budget_without_partial_mesh() {
    use ruffle_render::shape_utils::{DrawCommand, FillRule, OwnedDistilledShape, OwnedDrawPath};
    let point = |x, y| swf::Point::new(swf::Twips::from_pixels(x), swf::Twips::from_pixels(y));
    let input = ShapeInput {
        geometry: std::sync::Arc::new(OwnedDistilledShape {
            id: 1,
            shape_bounds: Default::default(),
            edge_bounds: Default::default(),
            paths: vec![OwnedDrawPath::Fill {
                style: swf::FillStyle::Color(swf::Color::RED),
                winding_rule: FillRule::EvenOdd,
                commands: vec![
                    DrawCommand::MoveTo(point(0.0, 0.0)),
                    DrawCommand::LineTo(point(32.0, 0.0)),
                    DrawCommand::LineTo(point(0.0, 32.0)),
                    DrawCommand::LineTo(point(0.0, 0.0)),
                ],
            }],
        }),
        scale: 1.0,
        bitmap_sizes: vec![],
    };
    assert!(
        matches!(input.prepare_bounded(32), Err(error) if error.kind() == RenderFailureKind::BudgetExceeded)
    );
}
