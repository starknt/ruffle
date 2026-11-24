//! Bounded binding reuse that cannot keep an evicted source image resident.
use crate::gpu_resources::{BindGroup, BindGroupLayout, Device, Sampler, TextureView};
use crate::gpu_runtime::{GpuRuntime, RenderFailure, ResourceId};
use crate::layouts::BITMAP_BATCH_TEXTURES;
use crate::resource_scheduler::{CacheTicket, Reclaim, ReclaimClass, ResourceScheduler};
use lru::LruCache;
use std::{
    num::NonZeroUsize,
    sync::{Arc, Mutex, Weak},
};

type Binding = Mutex<Option<(BindGroup, CacheTicket)>>;

#[derive(Debug, Default)]
struct SourceBindings(Mutex<Vec<Weak<Binding>>>);
impl Drop for SourceBindings {
    fn drop(&mut self) {
        for binding in self
            .0
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            if let Some(binding) = binding.upgrade() {
                let retired = binding
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .take();
                drop(retired);
            }
        }
    }
}
/// A source texture owns this token. Commands may keep it until encoding, but
/// cached bindings hold only weak registrations, never the source or its token.
#[derive(Clone, Debug, Default)]
pub(crate) struct Source(Arc<SourceBindings>);
impl Source {
    fn register(&self, binding: &Arc<Binding>) {
        let mut bindings = self
            .0
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        bindings.retain(|b| b.strong_count() != 0);
        let weak = Arc::downgrade(binding);
        if !bindings.iter().any(|b| b.ptr_eq(&weak)) {
            bindings.push(weak);
        }
    }
}

#[derive(Hash, PartialEq, Eq)]
struct Key {
    layout: ResourceId,
    sampler: ResourceId,
    views: [ResourceId; BITMAP_BATCH_TEXTURES],
}
struct State {
    entries: LruCache<Key, (Arc<Binding>, u64)>,
}
pub(crate) struct BitmapBatchCache {
    pub(crate) mode: crate::capabilities::BitmapBindings,
    state: Arc<Mutex<State>>,
    scheduler: ResourceScheduler,
}
impl BitmapBatchCache {
    pub fn new(runtime: &GpuRuntime, mode: crate::capabilities::BitmapBindings) -> Self {
        let state = Arc::new(Mutex::new(State {
            entries: LruCache::new(NonZeroUsize::new(64).unwrap()),
        }));
        let scheduler = runtime.scheduler().clone();
        scheduler.register(ReclaimClass::Cache, state.clone());
        Self {
            state,
            scheduler,
            mode,
        }
    }
    pub fn get_or_create(
        &self,
        device: &Device,
        layout: &BindGroupLayout,
        views: &[TextureView; BITMAP_BATCH_TEXTURES],
        sampler: &Sampler,
        sources: &[Source],
    ) -> Result<BindGroup, RenderFailure> {
        device.runtime().check_health()?;
        let key = Key {
            layout: layout.lease().id(),
            sampler: sampler.lease().id(),
            views: std::array::from_fn(|i| views[i].lease().id()),
        };
        {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some((binding, age)) = state.entries.get_mut(&key) {
                if let Some((group, _)) = &*binding
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                {
                    *age = self.scheduler.tick();
                    return Ok(group.clone());
                }
            }
        }
        let array_views;
        let mut entries: Vec<_> = if self.mode == crate::capabilities::BitmapBindings::Array {
            array_views = views.each_ref().map(|view| view.raw());
            vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureViewArray(&array_views),
            }]
        } else {
            views
                .iter()
                .enumerate()
                .map(|(i, view)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: wgpu::BindingResource::TextureView(view),
                })
                .collect()
        };
        entries.push(wgpu::BindGroupEntry {
            binding: BITMAP_BATCH_TEXTURES as u32,
            resource: wgpu::BindingResource::Sampler(sampler),
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Adjacent bitmap batch"),
            layout,
            entries: &entries,
        })?;
        if let Some(ticket) = self
            .scheduler
            .admit(ReclaimClass::Cache, group.lease().retained_storage_bytes())
        {
            let binding = Arc::new(Mutex::new(Some((group.clone(), ticket))));
            for source in sources {
                source.register(&binding);
            }
            let replaced = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .entries
                .push(key, (binding, self.scheduler.tick()));
            drop(replaced);
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
            .map(|(_, (_, age))| *age)
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
