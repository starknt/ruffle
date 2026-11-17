//! Exact command-use ownership. Raw descriptors are resolved at the actual
//! encoding call, so prebuilt slices and interleaved encoders remain correct.
//! No DerefMut escape to a raw encoder/pass is exposed.
use super::*;
use std::ops::Range;

#[derive(Clone)]
struct Uses {
    device: Device,
    recording: Arc<Mutex<Recording>>,
}
impl Uses {
    fn retain<T: Any + Eq + Hash>(&self, resource: &T) {
        match self.device.lookup(resource) {
            Ok(lease) => self
                .recording
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .retain(lease),
            Err(error) => {
                self.device.runtime().fail(error);
            }
        }
    }
}
impl CommandEncoder {
    fn recorder(&self) -> Uses {
        Uses {
            device: self.device.clone(),
            recording: self.uses.clone(),
        }
    }
    pub fn begin_render_pass(&mut self, desc: &wgpu::RenderPassDescriptor<'_>) -> RenderPass<'_> {
        let uses = self.recorder();
        for attachment in desc.color_attachments.iter().flatten() {
            uses.retain(attachment.view);
            if let Some(resolve) = attachment.resolve_target {
                uses.retain(resolve);
            }
        }
        if let Some(attachment) = &desc.depth_stencil_attachment {
            uses.retain(attachment.view);
        }
        if let Some(writes) = &desc.timestamp_writes {
            uses.retain(writes.query_set);
        }
        if let Some(queries) = desc.occlusion_query_set {
            uses.retain(queries);
        }
        RenderPass {
            raw: self.raw.begin_render_pass(desc),
            uses,
        }
    }
    pub fn begin_compute_pass(
        &mut self,
        desc: &wgpu::ComputePassDescriptor<'_>,
    ) -> ComputePass<'_> {
        let uses = self.recorder();
        if let Some(writes) = &desc.timestamp_writes {
            uses.retain(writes.query_set);
        }
        ComputePass {
            raw: self.raw.begin_compute_pass(desc),
            uses,
        }
    }
    pub fn copy_buffer_to_buffer(
        &mut self,
        source: &wgpu::Buffer,
        source_offset: u64,
        destination: &wgpu::Buffer,
        destination_offset: u64,
        size: u64,
    ) {
        let uses = self.recorder();
        uses.retain(source);
        uses.retain(destination);
        self.raw.copy_buffer_to_buffer(
            source,
            source_offset,
            destination,
            destination_offset,
            size,
        );
    }
    pub fn copy_texture_to_texture(
        &mut self,
        source: wgpu::TexelCopyTextureInfo<'_>,
        destination: wgpu::TexelCopyTextureInfo<'_>,
        size: wgpu::Extent3d,
    ) {
        let uses = self.recorder();
        uses.retain(source.texture);
        uses.retain(destination.texture);
        self.raw.copy_texture_to_texture(source, destination, size);
    }
    pub fn copy_buffer_to_texture(
        &mut self,
        source: wgpu::TexelCopyBufferInfo<'_>,
        destination: wgpu::TexelCopyTextureInfo<'_>,
        size: wgpu::Extent3d,
    ) {
        let uses = self.recorder();
        uses.retain(source.buffer);
        uses.retain(destination.texture);
        self.raw.copy_buffer_to_texture(source, destination, size);
    }
    pub fn copy_texture_to_buffer(
        &mut self,
        source: wgpu::TexelCopyTextureInfo<'_>,
        destination: wgpu::TexelCopyBufferInfo<'_>,
        size: wgpu::Extent3d,
    ) {
        let uses = self.recorder();
        uses.retain(source.texture);
        uses.retain(destination.buffer);
        self.raw.copy_texture_to_buffer(source, destination, size);
    }
    pub fn clear_buffer(&mut self, buffer: &wgpu::Buffer, offset: u64, size: Option<u64>) {
        self.recorder().retain(buffer);
        self.raw.clear_buffer(buffer, offset, size);
    }
    pub fn resolve_query_set(
        &mut self,
        queries: &wgpu::QuerySet,
        range: Range<u32>,
        buffer: &wgpu::Buffer,
        offset: u64,
    ) {
        let uses = self.recorder();
        uses.retain(queries);
        uses.retain(buffer);
        self.raw.resolve_query_set(queries, range, buffer, offset);
    }
    pub fn write_timestamp(&mut self, queries: &wgpu::QuerySet, index: u32) {
        self.recorder().retain(queries);
        self.raw.write_timestamp(queries, index);
    }
    pub fn push_debug_group(&mut self, label: &str) {
        self.raw.push_debug_group(label);
    }
    pub fn pop_debug_group(&mut self) {
        self.raw.pop_debug_group();
    }
}

pub struct RenderPass<'a> {
    raw: wgpu::RenderPass<'a>,
    uses: Uses,
}
impl RenderPass<'_> {
    pub fn set_pipeline(&mut self, pipeline: &wgpu::RenderPipeline) {
        self.uses.retain(pipeline);
        self.raw.set_pipeline(pipeline);
    }
    pub fn set_bind_group<'a>(
        &mut self,
        index: u32,
        group: impl Into<Option<&'a wgpu::BindGroup>>,
        offsets: &[u32],
    ) {
        let group = group.into();
        if let Some(group) = group {
            self.uses.retain(group);
        }
        self.raw.set_bind_group(index, group, offsets);
    }
    pub fn set_vertex_buffer<'a>(
        &mut self,
        slot: u32,
        slice: impl Into<Option<wgpu::BufferSlice<'a>>>,
    ) {
        let slice = slice.into();
        if let Some(slice) = slice {
            self.uses.retain(slice.buffer());
        }
        self.raw.set_vertex_buffer(slot, slice);
    }
    pub fn set_index_buffer(&mut self, slice: wgpu::BufferSlice<'_>, format: wgpu::IndexFormat) {
        self.uses.retain(slice.buffer());
        self.raw.set_index_buffer(slice, format);
    }
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        self.raw.draw(vertices, instances);
    }
    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        self.raw.draw_indexed(indices, base_vertex, instances);
    }
    pub fn set_stencil_reference(&mut self, value: u32) {
        self.raw.set_stencil_reference(value);
    }
    pub fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
        self.raw.set_scissor_rect(x, y, width, height);
    }
    pub fn set_viewport(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) {
        self.raw
            .set_viewport(x, y, width, height, min_depth, max_depth);
    }
    pub fn write_timestamp(&mut self, queries: &wgpu::QuerySet, index: u32) {
        self.uses.retain(queries);
        self.raw.write_timestamp(queries, index);
    }
    pub fn push_debug_group(&mut self, label: &str) {
        self.raw.push_debug_group(label);
    }
    pub fn pop_debug_group(&mut self) {
        self.raw.pop_debug_group();
    }
}

pub struct ComputePass<'a> {
    raw: wgpu::ComputePass<'a>,
    uses: Uses,
}
impl ComputePass<'_> {
    pub fn write_timestamp(&mut self, queries: &wgpu::QuerySet, index: u32) {
        self.uses.retain(queries);
        self.raw.write_timestamp(queries, index);
    }
    pub fn push_debug_group(&mut self, label: &str) {
        self.raw.push_debug_group(label);
    }
    pub fn pop_debug_group(&mut self) {
        self.raw.pop_debug_group();
    }
}
