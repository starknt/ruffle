//! Bounds generated storage, including vector growth and native packed output.
use lyon::tessellation::geometry_builder::{
    FillGeometryBuilder, GeometryBuilder, GeometryBuilderError, StrokeGeometryBuilder,
};
use lyon::tessellation::{FillVertex, StrokeVertex, VertexId};

pub(super) struct BoundedBuilder<'a, B> {
    pub inner: B,
    pub remaining: &'a mut Option<u64>,
    pub exhausted: &'a mut bool,
}
impl<B> BoundedBuilder<'_, B> {
    fn charge(&mut self, bytes: u64) -> bool {
        if *self.exhausted {
            return false;
        }
        if let Some(remaining) = self.remaining {
            let Some(next) = remaining.checked_sub(bytes) else {
                *self.exhausted = true;
                return false;
            };
            *remaining = next;
        }
        true
    }
}
impl<B: GeometryBuilder> GeometryBuilder for BoundedBuilder<'_, B> {
    fn begin_geometry(&mut self) {
        self.inner.begin_geometry();
    }
    fn end_geometry(&mut self) {
        self.inner.end_geometry();
    }
    fn abort_geometry(&mut self) {
        self.inner.abort_geometry();
    }
    fn add_triangle(&mut self, a: VertexId, b: VertexId, c: VertexId) {
        if self.charge(6 * std::mem::size_of::<u32>() as u64) {
            self.inner.add_triangle(a, b, c);
        }
    }
}
impl<B: FillGeometryBuilder> FillGeometryBuilder for BoundedBuilder<'_, B> {
    fn add_fill_vertex(&mut self, vertex: FillVertex) -> Result<VertexId, GeometryBuilderError> {
        // Lyon vertices and spare capacity coexist with packed native vertices.
        if !self.charge(4 * std::mem::size_of::<super::Vertex>() as u64) {
            return Err(GeometryBuilderError::TooManyVertices);
        }
        self.inner.add_fill_vertex(vertex)
    }
}
impl<B: StrokeGeometryBuilder> StrokeGeometryBuilder for BoundedBuilder<'_, B> {
    fn add_stroke_vertex(
        &mut self,
        vertex: StrokeVertex,
    ) -> Result<VertexId, GeometryBuilderError> {
        if !self.charge(4 * std::mem::size_of::<super::Vertex>() as u64) {
            return Err(GeometryBuilderError::TooManyVertices);
        }
        self.inner.add_stroke_vertex(vertex)
    }
}
