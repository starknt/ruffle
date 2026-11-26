//! Compiled state is keyed by the vertex stream layout, never by buffer identity.
use super::shader_pair::ShaderCompileData;
use super::{MAX_VERTEX_ATTRIBUTES, VertexAttributeInfo, VertexBufferWrapper};
use crate::gpu_resources::{BindGroupLayout, PipelineLayout, RenderPipeline};
use crate::resource_scheduler::{CacheTicket, Reclaim, ReclaimClass, ResourceScheduler};
use lru::LruCache;
use naga_agal::VertexAttributeFormat;
use parking_lot::Mutex;
use ruffle_render::backend::Context3DVertexBufferFormat as FlashFormat;
use smallvec::SmallVec;
use std::{num::NonZeroUsize, rc::Rc, sync::Arc};

#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) struct Stream {
    pub stride: u64,
    pub attributes: SmallVec<[wgpu::VertexAttribute; 2]>,
}

fn formats(format: FlashFormat) -> (VertexAttributeFormat, wgpu::VertexFormat) {
    use VertexAttributeFormat as A;
    use wgpu::VertexFormat as V;
    match format {
        FlashFormat::Float4 => (A::Float4, V::Float32x4),
        FlashFormat::Float3 => (A::Float3, V::Float32x3),
        FlashFormat::Float2 => (A::Float2, V::Float32x2),
        FlashFormat::Float1 => (A::Float1, V::Float32),
        FlashFormat::Bytes4 => (A::Bytes4, V::Unorm8x4),
    }
}

#[derive(Default)]
pub(super) struct VertexInputs {
    slots: [Option<VertexAttributeInfo>; MAX_VERTEX_ATTRIBUTES],
    pub buffers: SmallVec<[Rc<VertexBufferWrapper>; 4]>,
    pub streams: SmallVec<[Stream; 4]>,
    pub attributes: [Option<VertexAttributeFormat>; MAX_VERTEX_ATTRIBUTES],
}

impl VertexInputs {
    pub fn set(&mut self, index: usize, value: Option<VertexAttributeInfo>) -> bool {
        if match (&self.slots[index], &value) {
            (Some(a), Some(b)) => {
                Rc::ptr_eq(&a.buffer, &b.buffer)
                    && a.offset_in_32bit_units == b.offset_in_32bit_units
                    && formats(a.format) == formats(b.format)
            }
            (None, None) => true,
            _ => false,
        } {
            return false;
        }
        self.slots[index] = value;
        let old_streams = std::mem::take(&mut self.streams);
        self.buffers.clear();
        self.attributes.fill(None);
        for (slot, attr) in self.slots.iter().enumerate() {
            let Some(attr) = attr else { continue };
            let stream = self
                .buffers
                .iter()
                .position(|buffer| Rc::ptr_eq(buffer, &attr.buffer))
                .unwrap_or_else(|| {
                    self.buffers.push(attr.buffer.clone());
                    self.streams.push(Stream {
                        stride: u64::from(attr.buffer.data_32_per_vertex) * 4,
                        attributes: SmallVec::new(),
                    });
                    self.buffers.len() - 1
                });
            let (agal, format) = formats(attr.format);
            self.attributes[slot] = Some(agal);
            self.streams[stream].attributes.push(wgpu::VertexAttribute {
                format,
                offset: attr.offset_in_32bit_units * 4,
                shader_location: slot as u32,
            });
        }
        self.streams != old_streams
    }
}

#[derive(Clone, Eq, PartialEq, Hash)]
pub(super) struct PipelineKey {
    pub program: u64,
    pub shader: ShaderCompileData,
    pub streams: SmallVec<[Stream; 4]>,
    pub cull: Option<wgpu::Face>,
    pub depth: Option<(bool, wgpu::CompareFunction, wgpu::StencilState)>,
    pub target: wgpu::ColorTargetState,
    pub samples: u32,
}

#[derive(Clone)]
pub(super) struct Prepared {
    pub pipeline: RenderPipeline,
    pub bindings: BindGroupLayout,
    // Keep the exact tracked layout alive alongside the executable and its binding layout.
    pub _layout: PipelineLayout,
}

struct State(LruCache<PipelineKey, (Prepared, u64, CacheTicket)>);
impl Reclaim for Mutex<State> {
    fn oldest(&self) -> Option<u64> {
        self.lock().0.peek_lru().map(|(_, entry)| entry.1)
    }
    fn evict_oldest(&self) -> bool {
        let removed = self.lock().0.pop_lru();
        removed.is_some()
    }
}

pub(super) struct PipelineCache {
    entries: Arc<Mutex<State>>,
    scheduler: ResourceScheduler,
}
impl PipelineCache {
    pub fn new(scheduler: &ResourceScheduler) -> Self {
        let entries = Arc::new(Mutex::new(State(LruCache::new(
            NonZeroUsize::new(64).unwrap(),
        ))));
        scheduler.register(ReclaimClass::Cache, entries.clone());
        Self {
            entries,
            scheduler: scheduler.clone(),
        }
    }
    pub fn get(&self, key: &PipelineKey) -> Option<Prepared> {
        let mut entries = self.entries.lock();
        let (value, age, _) = entries.0.get_mut(key)?;
        *age = self.scheduler.tick();
        value.pipeline.profile_hit();
        Some(value.clone())
    }
    pub fn insert(&self, key: PipelineKey, prepared: &Prepared) {
        if let Some(ticket) = self.scheduler.admit(
            ReclaimClass::Cache,
            prepared.pipeline.lease().retained_storage_bytes(),
        ) {
            let evicted = self
                .entries
                .lock()
                .0
                .push(key, (prepared.clone(), self.scheduler.tick(), ticket));
            drop(evicted);
        }
    }
}
