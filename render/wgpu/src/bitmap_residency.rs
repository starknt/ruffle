//! Reconstructible native uploads. Only concrete native storage crosses a
//! reclamation boundary; public bitmap/GC handle traits remain thread confined.
use crate::resource_scheduler::{CacheTicket, Reclaim, ReclaimClass, ResourceScheduler};
use lru::LruCache;
use ruffle_render::profile::BitmapResidencyStats;
use std::{
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};

type Entry<T> = (Arc<T>, u64, u64, CacheTicket);
struct State<T> {
    entries: LruCache<usize, Entry<T>>,
    stats: BitmapResidencyStats,
}
pub struct BitmapResidency<T: Send + Sync + 'static> {
    state: Arc<Mutex<State<T>>>,
    scheduler: ResourceScheduler,
    limit: u64,
}
impl<T: Send + Sync + 'static> BitmapResidency<T> {
    #[cfg(test)]
    pub fn new(limit: u64) -> Self {
        Self::with_scheduler(limit, ResourceScheduler::new(0, limit))
    }
    pub fn with_runtime(runtime: &crate::gpu_runtime::GpuRuntime, limit: u64) -> Self {
        Self::with_scheduler(limit, runtime.scheduler().clone())
    }
    fn with_scheduler(limit: u64, scheduler: ResourceScheduler) -> Self {
        let state = Arc::new(Mutex::new(State {
            entries: LruCache::new(NonZeroUsize::new(4096).unwrap()),
            stats: Default::default(),
        }));
        scheduler.register(ReclaimClass::Cache, state.clone());
        Self {
            state,
            scheduler,
            limit,
        }
    }
    pub fn retain(&mut self, handle: &Arc<T>, bytes: u64) {
        let key = Arc::as_ptr(handle) as usize;
        {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(entry) = state.entries.get_mut(&key) {
                entry.2 = self.scheduler.tick();
                state.stats.hits += 1;
                return;
            }
        }
        if bytes > self.limit {
            return;
        }
        while self.stats().retained_bytes > self.limit - bytes {
            self.state.evict_oldest();
        }
        let Some(ticket) = self.scheduler.admit(ReclaimClass::Cache, bytes) else {
            return;
        };
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((_, (_, old, _, _))) = state
            .entries
            .push(key, (handle.clone(), bytes, self.scheduler.tick(), ticket))
        {
            state.stats.retained_bytes -= old;
            state.stats.evictions += 1;
        }
        state.stats.retained_bytes += bytes;
        state.stats.insertions += 1;
        state.stats.retained_entries = state.entries.len();
    }
    pub fn stats(&self) -> BitmapResidencyStats {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .stats
    }
}
impl<T: Send + Sync> Reclaim for Mutex<State<T>> {
    fn oldest(&self) -> Option<u64> {
        self.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entries
            .peek_lru()
            .map(|(_, entry)| entry.2)
    }
    fn evict_oldest(&self) -> bool {
        let entry = {
            let mut state = self
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let entry = state.entries.pop_lru();
            if let Some((_, (_, bytes, _, _))) = &entry {
                state.stats.retained_bytes -= bytes;
                state.stats.evictions += 1;
                state.stats.retained_entries = state.entries.len();
            }
            entry
        };
        entry.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Debug)]
    struct Image;
    fn image() -> Arc<Image> {
        Arc::new(Image)
    }

    #[test]
    fn budget_evicts_the_oldest_upload_without_invalidating_a_recorded_draw() {
        let mut cache = BitmapResidency::new(8);
        let first = image();
        let second = image();
        let third = image();
        let second_weak = Arc::downgrade(&second);
        cache.retain(&first, 4);
        cache.retain(&second, 4);
        cache.retain(&first, 4); // First is now most recently drawn.
        cache.retain(&third, 4);
        assert_eq!(cache.stats().retained_bytes, 8);
        assert_eq!(cache.stats().retained_entries, 2);
        assert_eq!(cache.stats().evictions, 1);
        assert!(
            second_weak.upgrade().is_some(),
            "a recorded draw still owns second"
        );
        drop(second);
        assert!(second_weak.upgrade().is_none());
        let first_weak = Arc::downgrade(&first);
        drop(first);
        assert!(first_weak.upgrade().is_some());
        cache.retain(&image(), 9);
        assert!(
            first_weak.upgrade().is_some(),
            "oversized entry must not evict useful uploads"
        );
        assert_eq!(cache.stats().retained_bytes, 8);
        drop(cache);
        assert!(first_weak.upgrade().is_none());
    }

    #[test]
    fn tiny_images_also_have_a_bounded_entry_count() {
        let mut cache = BitmapResidency::new(u64::MAX);
        for _ in 0..5000 {
            cache.retain(&image(), 4);
        }
        assert_eq!(cache.stats().retained_entries, 4096);
        assert_eq!(cache.stats().retained_bytes, 4096 * 4);
        assert_eq!(cache.stats().evictions, 904);
    }
}
