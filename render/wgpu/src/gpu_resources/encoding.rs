//! Exact command-use ownership. Raw descriptors are resolved at the actual
//! encoding call, so prebuilt slices and interleaved encoders remain correct.
//! No DerefMut escape to a raw encoder/pass is exposed.
use super::*;
use std::ops::Range;

struct Uses<'a> {
    device: Device,
    recording: std::sync::MutexGuard<'a, Recording>,
}
impl<'a> Uses<'a> {
    fn new(device: Device, recording: &'a Mutex<Recording>) -> Self {
        Self {
            device,
            recording: recording
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        }
    }
    fn retain<T: Any + Eq + Hash>(&mut self, resource: &T) {
        let key = object_key(resource);
        if self
            .recording
            .objects
            .get(&key)
            .is_some_and(|objects| objects.iter().any(|object| object.matches(resource)))
        {
            return;
        }
        match self.device.lookup_object(resource) {
            Ok(object) => {
                self.recording.retain(object.lease());
                // Keep exact native identities resolvable until encoder finish.
                self.recording.objects.entry(key).or_default().push(object);
            }
            Err(error) => {
                self.device.runtime().fail(error);
            }
        }
    }
}
impl CommandEncoder {
    pub fn begin_render_pass(&mut self, desc: &wgpu::RenderPassDescriptor<'_>) -> RenderPass<'_> {
        let mut uses = Uses::new(self.device.clone(), &self.uses);
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
            state: PassState::default(),
            changes: StateChanges::default(),
        }
    }
    pub fn begin_compute_pass(
        &mut self,
        desc: &wgpu::ComputePassDescriptor<'_>,
    ) -> ComputePass<'_> {
        let mut uses = Uses::new(self.device.clone(), &self.uses);
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
        let mut uses = Uses::new(self.device.clone(), &self.uses);
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
        let mut uses = Uses::new(self.device.clone(), &self.uses);
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
        let mut uses = Uses::new(self.device.clone(), &self.uses);
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
        let mut uses = Uses::new(self.device.clone(), &self.uses);
        uses.retain(source.texture);
        uses.retain(destination.buffer);
        self.raw.copy_texture_to_buffer(source, destination, size);
    }
    pub fn clear_buffer(&mut self, buffer: &wgpu::Buffer, offset: u64, size: Option<u64>) {
        Uses::new(self.device.clone(), &self.uses).retain(buffer);
        self.raw.clear_buffer(buffer, offset, size);
    }
    /// Schedule a read only if this encoder's commands reach the queue.
    pub fn map_read_on_submit(
        &self,
        buffer: &Buffer,
        size: u64,
        callback: impl FnOnce(Result<(), wgpu::BufferAsyncError>) + Send + 'static,
    ) {
        self.raw
            .map_buffer_on_submit(buffer, wgpu::MapMode::Read, ..size, callback);
    }
    pub fn resolve_query_set(
        &mut self,
        queries: &wgpu::QuerySet,
        range: Range<u32>,
        buffer: &wgpu::Buffer,
        offset: u64,
    ) {
        let mut uses = Uses::new(self.device.clone(), &self.uses);
        uses.retain(queries);
        uses.retain(buffer);
        self.raw.resolve_query_set(queries, range, buffer, offset);
    }
    pub fn write_timestamp(&mut self, queries: &wgpu::QuerySet, index: u32) {
        Uses::new(self.device.clone(), &self.uses).retain(queries);
        self.raw.write_timestamp(queries, index);
    }
    pub fn push_debug_group(&mut self, label: &str) {
        self.raw.push_debug_group(label);
        self.debug_groups.push(label.to_owned());
    }
    pub fn pop_debug_group(&mut self) {
        self.raw.pop_debug_group();
        self.debug_groups.pop();
    }
}

/// Actual calls forwarded to the native pass after state suppression.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StateChanges {
    pub pipelines: u64,
    pub bind_groups: u64,
    pub vertex_buffers: u64,
    pub index_buffers: u64,
    pub scissors: u64,
    pub stencil_references: u64,
    pub viewports: u64,
}
#[derive(Clone, PartialEq, Eq)]
struct BufferRange {
    buffer: wgpu::Buffer,
    offset: u64,
    size: u64,
}
impl From<wgpu::BufferSlice<'_>> for BufferRange {
    fn from(slice: wgpu::BufferSlice<'_>) -> Self {
        Self {
            buffer: slice.buffer().clone(),
            offset: slice.offset(),
            size: slice.size(),
        }
    }
}
#[derive(Default)]
struct PassState {
    pipeline: Option<wgpu::RenderPipeline>,
    bindings: FnvHashMap<u32, (Option<wgpu::BindGroup>, smallvec::SmallVec<[u32; 4]>)>,
    vertices: FnvHashMap<u32, Option<BufferRange>>,
    index: Option<(BufferRange, wgpu::IndexFormat)>,
    scissor: Option<[u32; 4]>,
    stencil: Option<u32>,
    viewport: Option<[u32; 6]>,
}

pub struct RenderPass<'a> {
    raw: wgpu::RenderPass<'a>,
    uses: Uses<'a>,
    state: PassState,
    changes: StateChanges,
}
impl Drop for RenderPass<'_> {
    fn drop(&mut self) {
        let c = self.state_changes();
        ruffle_render::profile::work(|w| {
            w.native_pipeline_changes += c.pipelines;
            w.native_bind_groups += c.bind_groups;
            w.native_vertex_buffers += c.vertex_buffers;
            w.native_index_buffers += c.index_buffers;
            w.native_scissors += c.scissors;
            w.native_stencil_references += c.stencil_references;
            w.native_viewports += c.viewports;
        });
    }
}
impl RenderPass<'_> {
    pub fn state_changes(&self) -> StateChanges {
        self.changes
    }
    pub fn set_pipeline(&mut self, pipeline: &wgpu::RenderPipeline) {
        if self.state.pipeline.as_ref() == Some(pipeline) {
            return;
        }
        self.uses.retain(pipeline);
        self.raw.set_pipeline(pipeline);
        self.state.pipeline = Some(pipeline.clone());
        // Conservatively rebind after any pipeline-layout change.
        self.state.bindings.clear();
        self.changes.pipelines += 1;
    }
    pub fn set_bind_group<'a>(
        &mut self,
        index: u32,
        group: impl Into<Option<&'a wgpu::BindGroup>>,
        offsets: &[u32],
    ) {
        let group = group.into();
        if self
            .state
            .bindings
            .get(&index)
            .is_some_and(|(old, dynamic)| old.as_ref() == group && dynamic.as_slice() == offsets)
        {
            return;
        }
        if let Some(group) = group {
            self.uses.retain(group);
        }
        self.raw.set_bind_group(index, group, offsets);
        self.state.bindings.insert(
            index,
            (group.cloned(), smallvec::SmallVec::from_slice(offsets)),
        );
        self.changes.bind_groups += 1;
    }
    pub fn set_vertex_buffer<'a>(
        &mut self,
        slot: u32,
        slice: impl Into<Option<wgpu::BufferSlice<'a>>>,
    ) {
        let slice = slice.into();
        let key = slice.map(BufferRange::from);
        if self.state.vertices.get(&slot) == Some(&key) {
            return;
        }
        if let Some(slice) = slice {
            self.uses.retain(slice.buffer());
        }
        self.raw.set_vertex_buffer(slot, slice);
        self.state.vertices.insert(slot, key);
        self.changes.vertex_buffers += 1;
    }
    pub fn set_index_buffer(&mut self, slice: wgpu::BufferSlice<'_>, format: wgpu::IndexFormat) {
        let key = (BufferRange::from(slice), format);
        if self.state.index.as_ref() == Some(&key) {
            return;
        }
        self.uses.retain(slice.buffer());
        self.raw.set_index_buffer(slice, format);
        self.state.index = Some(key);
        self.changes.index_buffers += 1;
    }
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        self.raw.draw(vertices, instances);
    }
    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        self.raw.draw_indexed(indices, base_vertex, instances);
    }
    pub fn set_stencil_reference(&mut self, value: u32) {
        if self.state.stencil == Some(value) {
            return;
        }
        self.raw.set_stencil_reference(value);
        self.state.stencil = Some(value);
        self.changes.stencil_references += 1;
    }
    pub fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
        let key = [x, y, width, height];
        if self.state.scissor == Some(key) {
            return;
        }
        self.raw.set_scissor_rect(x, y, width, height);
        self.state.scissor = Some(key);
        self.changes.scissors += 1;
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
        let key = [x, y, width, height, min_depth, max_depth].map(f32::to_bits);
        if self.state.viewport == Some(key) {
            return;
        }
        self.raw
            .set_viewport(x, y, width, height, min_depth, max_depth);
        self.state.viewport = Some(key);
        self.changes.viewports += 1;
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
    uses: Uses<'a>,
}
impl ComputePass<'_> {
    /// Record a single-group image compute pass and its exact resource dependencies.
    pub fn dispatch(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        bindings: &wgpu::BindGroup,
        groups: [u32; 2],
    ) {
        self.uses.retain(pipeline);
        self.uses.retain(bindings);
        self.raw.set_pipeline(pipeline);
        self.raw.set_bind_group(0, bindings, &[]);
        self.raw.dispatch_workgroups(groups[0], groups[1], 1);
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
