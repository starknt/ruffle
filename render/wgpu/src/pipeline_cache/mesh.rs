//! Static SWF geometry only. Stored draws contain CPU data. Bitmap IDs resolve against the current movie.
use super::{Hash, derived::Files};
use crate::gpu_runtime::RenderFailure;
use crate::prepare::mesh::{PackedDraw, PackedVertices, PreparedMesh, ShapeInput, input_vertices};
use ruffle_render::{
    shape_utils::GradientType,
    tessellator::{Bitmap as TessBitmap, DrawType, Gradient},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    hash::{Hash as StdHash, Hasher},
    path::Path,
};

const MAX_FILE: u64 = 4 << 20;
const MIN_VERTICES: usize = 256;
const BUDGET: u64 = if cfg!(target_os = "android") {
    16 << 20
} else {
    32 << 20
};
pub(crate) struct MeshCache(Files);
impl MeshCache {
    pub fn from_env(dir: &Path) -> Option<Self> {
        Files::open(dir, "meshes-v1", "RUFFLE_STATIC_MESH_CACHE", BUDGET).map(Self)
    }
    pub fn stats(&self) -> (u64, u64) {
        self.0.stats()
    }
    pub fn prepare(
        &self,
        input: &ShapeInput,
        budget: u64,
        build: impl FnOnce() -> Result<PreparedMesh, RenderFailure>,
    ) -> Result<PreparedMesh, RenderFailure> {
        if input_vertices(&input.geometry) < MIN_VERTICES {
            return build();
        }
        let file = self.0.file(identity(input), "mesh");
        // Bound both the archive and the restored allocations inside the existing partition permit.
        if let Ok(data) = file.load_bounded(MAX_FILE.min(budget / 2))
            && let Some(mesh) = unpack(&data, budget.saturating_sub(data.len() as u64))
        {
            self.0.note(true);
            return Ok(mesh);
        }
        self.0.note(false);
        let mesh = build()?;
        let size = mesh.retained_bytes();
        if size <= MAX_FILE
            && size.saturating_mul(2) <= budget
            && let Some(bytes) = pack(&mesh)
            && bytes.len() as u64 <= MAX_FILE
        {
            self.0.store(file, bytes);
        }
        Ok(mesh)
    }
}
fn identity(input: &ShapeInput) -> Hash {
    // Rust's structural Hash encodes enum variants, lengths and integer fields. Scope that
    // encoding, the packed vertex layout, scale tolerance and Lyon version to this ABI.
    struct Sink(Sha256);
    impl Hasher for Sink {
        fn write(&mut self, bytes: &[u8]) {
            self.0.update(bytes)
        }
        fn finish(&self) -> u64 {
            u64::from_le_bytes(self.0.clone().finalize()[..8].try_into().unwrap())
        }
    }
    let mut h = Sink(Sha256::new());
    h.write(b"ruffle-lyon-1-packed-shapes-v1");
    h.write(std::env::consts::ARCH.as_bytes());
    input.geometry.hash(&mut h);
    input.scale.to_bits().hash(&mut h);
    let mut sizes: Vec<_> = input
        .bitmap_sizes
        .iter()
        .map(|(id, size)| (*id, size.width, size.height))
        .collect();
    sizes.sort_unstable();
    sizes.hash(&mut h);
    h.0.finalize()
}
type Matrix = [[f32; 3]; 3];
#[derive(Serialize, Deserialize)]
enum Paint {
    Color,
    Gradient(Matrix, usize),
    Bitmap(Matrix, u16, bool, bool),
}
#[derive(Serialize, Deserialize)]
struct Draw {
    paint: Paint,
    vertices: u32,
    indices: u32,
    mask: u32,
}
#[derive(Serialize, Deserialize)]
struct Ramp {
    kind: u8,
    spread: u8,
    interpolation: u8,
    focal: i16,
    stops: Vec<(u8, [u8; 4])>,
}
#[derive(Serialize, Deserialize)]
struct Header {
    draws: Vec<Draw>,
    gradients: Vec<Ramp>,
    bounds: Option<[f64; 4]>,
}
fn pack(mesh: &PreparedMesh) -> Option<Vec<u8>> {
    let gradients = mesh
        .gradients
        .iter()
        .map(|g| Ramp {
            kind: match g.gradient_type {
                GradientType::Linear => 0,
                GradientType::Radial => 1,
                GradientType::Focal => 2,
            },
            spread: g.repeat_mode as u8,
            interpolation: g.interpolation as u8,
            focal: g.focal_point.get(),
            stops: g
                .records
                .iter()
                .map(|r| (r.ratio, [r.color.r, r.color.g, r.color.b, r.color.a]))
                .collect(),
        })
        .collect();
    let mut draws = Vec::with_capacity(mesh.draws.len());
    for d in &mesh.draws {
        let paint = match &d.draw_type {
            DrawType::Color => Paint::Color,
            DrawType::Gradient { matrix, gradient } => Paint::Gradient(*matrix, *gradient),
            DrawType::Bitmap(b) => {
                Paint::Bitmap(b.matrix, b.bitmap_id, b.is_smoothed, b.is_repeating)
            }
        };
        let n = match &d.vertices {
            PackedVertices::Color(v) => v.len(),
            PackedVertices::Texture(v) => v.len(),
        };
        draws.push(Draw {
            paint,
            vertices: n.try_into().ok()?,
            indices: d.indices.len().try_into().ok()?,
            mask: d.mask_index_count,
        });
    }
    let mut bytes = vec![0; 4];
    serde_json::to_writer(
        &mut bytes,
        &Header {
            draws,
            gradients,
            bounds: mesh.bounds,
        },
    )
    .ok()?;
    let header = u32::try_from(bytes.len() - 4).ok()?.to_le_bytes();
    bytes[..4].copy_from_slice(&header);
    for d in &mesh.draws {
        bytes.extend(d.vertices.bytes());
        bytes.extend(bytemuck::cast_slice(&d.indices));
    }
    Some(bytes)
}
fn take_vec<T: bytemuck::Pod>(data: &mut &[u8], count: u32) -> Option<Vec<T>> {
    let (values, rest) = data.split_at_checked((count as usize).checked_mul(size_of::<T>())?)?;
    *data = rest;
    Some(
        values
            .chunks_exact(size_of::<T>())
            .map(bytemuck::pod_read_unaligned)
            .collect(),
    )
}
fn unpack(data: &[u8], budget: u64) -> Option<PreparedMesh> {
    if data.len() as u64 > budget || data.len() as u64 > MAX_FILE {
        return None;
    }
    let (prefix, rest) = data.split_at_checked(4)?;
    let (header, mut data) =
        rest.split_at_checked(u32::from_le_bytes(prefix.try_into().ok()?) as usize)?;
    if header.len() > 256 << 10 {
        return None;
    }
    let h: Header = serde_json::from_slice(header).ok()?;
    let mut gradients = vec![];
    for r in h.gradients {
        gradients.push(Gradient {
            gradient_type: match r.kind {
                0 => GradientType::Linear,
                1 => GradientType::Radial,
                2 => GradientType::Focal,
                _ => return None,
            },
            repeat_mode: swf::GradientSpread::from_u8(r.spread)?,
            interpolation: swf::GradientInterpolation::from_u8(r.interpolation)?,
            focal_point: swf::Fixed8::from_bits(r.focal),
            records: r
                .stops
                .into_iter()
                .map(|(ratio, [r, g, b, a])| swf::GradientRecord {
                    ratio,
                    color: swf::Color { r, g, b, a },
                })
                .collect(),
        });
    }
    let mut draws = vec![];
    for d in h.draws {
        let (draw_type, vertices) = match d.paint {
            Paint::Color => (
                DrawType::Color,
                PackedVertices::Color(take_vec(&mut data, d.vertices)?),
            ),
            Paint::Gradient(matrix, gradient) if gradient < gradients.len() => (
                DrawType::Gradient { matrix, gradient },
                PackedVertices::Texture(take_vec(&mut data, d.vertices)?),
            ),
            Paint::Bitmap(matrix, bitmap_id, is_smoothed, is_repeating) => (
                DrawType::Bitmap(TessBitmap {
                    matrix,
                    bitmap_id,
                    is_smoothed,
                    is_repeating,
                }),
                PackedVertices::Texture(take_vec(&mut data, d.vertices)?),
            ),
            _ => return None,
        };
        let indices: Vec<u32> = take_vec(&mut data, d.indices)?;
        if d.mask > d.indices || indices.iter().any(|&i| i >= d.vertices) {
            return None;
        }
        draws.push(PackedDraw {
            draw_type,
            vertices,
            indices,
            mask_index_count: d.mask,
        });
    }
    let mesh = PreparedMesh {
        draws,
        gradients,
        bounds: h.bounds,
    };
    (data.is_empty() && mesh.retained_bytes() <= budget).then_some(mesh)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complete_mesh_archive_preserves_mask_uv_bounds_and_rejects_partial_or_oversize_data() {
        let mesh = PreparedMesh {
            bounds: Some([0., 0., 5., 6.]),
            gradients: vec![],
            draws: vec![PackedDraw {
                draw_type: DrawType::Bitmap(TessBitmap {
                    matrix: [[1.; 3]; 3],
                    bitmap_id: 27,
                    is_smoothed: true,
                    is_repeating: false,
                }),
                vertices: PackedVertices::Texture(vec![crate::PosUvVertex::new(
                    5., 6., 0.5, 0.25, 1.,
                )]),
                indices: vec![0, 0, 0],
                mask_index_count: 3,
            }],
        };
        let data = pack(&mesh).unwrap();
        let copy = unpack(&data, 1 << 20).unwrap();
        assert_eq!(copy.bounds, mesh.bounds);
        assert_eq!(
            copy.draws[0].vertices.bytes(),
            mesh.draws[0].vertices.bytes()
        );
        assert_eq!(copy.draws[0].indices, [0, 0, 0]);
        assert_eq!(copy.draws[0].mask_index_count, 3);
        assert!(
            matches!(&copy.draws[0].draw_type,DrawType::Bitmap(b) if b.bitmap_id==27 && b.is_smoothed && !b.is_repeating)
        );
        assert!(unpack(&data[..data.len() - 1], 1 << 20).is_none());
        assert!(unpack(&data, 4).is_none());
    }
}
