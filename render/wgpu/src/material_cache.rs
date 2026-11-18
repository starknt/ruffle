//! Small, reclaimable native binding cache. Keys include complete resource IDs
//! (and therefore device generations), layouts, ranges, and array order.
use crate::gpu_resources::{BindGroup, Device};
use crate::gpu_runtime::{GpuRuntime, RenderFailure, RenderFailureKind, ResourceId};
use crate::resource_scheduler::{CacheTicket, Reclaim, ReclaimClass, ResourceScheduler};
use lru::LruCache;
use std::{
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum Binding {
    Buffer(ResourceId, u64, Option<u64>),
    Sampler(ResourceId),
    View(ResourceId),
    Array(Vec<Binding>),
}
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct MaterialKey {
    layout: ResourceId,
    entries: Vec<(u32, Binding)>,
}
type Entry = (BindGroup, u64, CacheTicket);
struct State {
    entries: LruCache<MaterialKey, Entry>,
}
pub struct MaterialCache {
    state: Arc<Mutex<State>>,
    scheduler: ResourceScheduler,
}
impl MaterialCache {
    pub fn new(runtime: &GpuRuntime, capacity: usize) -> Self {
        let state = Arc::new(Mutex::new(State {
            entries: LruCache::new(NonZeroUsize::new(capacity.max(1)).unwrap()),
        }));
        let scheduler = runtime.scheduler().clone();
        scheduler.register(ReclaimClass::Cache, state.clone());
        Self { state, scheduler }
    }
    pub fn clear(&self) {
        let entries = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let capacity = state.entries.cap();
            std::mem::replace(&mut state.entries, LruCache::new(capacity))
        };
        drop(entries);
    }

    pub fn get_or_create(
        &self,
        device: &Device,
        desc: &wgpu::BindGroupDescriptor<'_>,
    ) -> Result<BindGroup, RenderFailure> {
        device.runtime().check_health()?;
        let buffer = |b: &wgpu::BufferBinding<'_>| {
            Ok(Binding::Buffer(
                device.lookup(b.buffer)?.id(),
                b.offset,
                b.size.map(|size| size.get()),
            ))
        };
        let sampler = |s: &wgpu::Sampler| Ok(Binding::Sampler(device.lookup(s)?.id()));
        let view = |v: &wgpu::TextureView| Ok(Binding::View(device.lookup(v)?.id()));
        let mut entries = Vec::with_capacity(desc.entries.len());
        for entry in desc.entries {
            use wgpu::BindingResource as R;
            let binding = match &entry.resource {
                R::Buffer(b) => buffer(b)?,
                R::Sampler(s) => sampler(s)?,
                R::TextureView(v) => view(v)?,
                R::BufferArray(values) => Binding::Array(
                    values
                        .iter()
                        .map(buffer)
                        .collect::<Result<_, RenderFailure>>()?,
                ),
                R::SamplerArray(values) => Binding::Array(
                    values
                        .iter()
                        .map(|s| sampler(s))
                        .collect::<Result<_, RenderFailure>>()?,
                ),
                R::TextureViewArray(values) => Binding::Array(
                    values
                        .iter()
                        .map(|v| view(v))
                        .collect::<Result<_, RenderFailure>>()?,
                ),
                _ => {
                    return Err(RenderFailure::new(
                        RenderFailureKind::Validation,
                        "unsupported cached binding resource",
                    ));
                }
            };
            entries.push((entry.binding, binding));
        }
        entries.sort_unstable_by_key(|(binding, _)| *binding);
        let key = MaterialKey {
            layout: device.lookup(desc.layout)?.id(),
            entries,
        };
        {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some((group, age, _)) = state.entries.get_mut(&key) {
                *age = self.scheduler.tick();
                return Ok(group.clone());
            }
        }
        // Allocation/reclamation may call back into this cache. Hold no cache
        // mutex across either operation. A miss never blocks mandatory rendering.
        let group = device.create_bind_group(desc)?;
        if let Some(ticket) = self
            .scheduler
            .admit(ReclaimClass::Cache, group.lease().retained_storage_bytes())
        {
            self.state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .entries
                .put(key, (group.clone(), self.scheduler.tick(), ticket));
        }
        Ok(group)
    }
}
impl Reclaim for Mutex<State> {
    fn oldest(&self) -> Option<u64> {
        self.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entries
            .peek_lru()
            .map(|(_, (_, age, _))| *age)
    }
    fn evict_oldest(&self) -> bool {
        let entry = self
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entries
            .pop_lru();
        entry.is_some()
    }
}
