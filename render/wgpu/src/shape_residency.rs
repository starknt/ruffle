//! Shape handles contain player-thread bitmap bindings. Cold eviction runs on the renderer thread.
use lru::LruCache;
use ruffle_render::backend::{CachedShape, ShapeHandle, ShapeResidency};
use std::{
    cell::RefCell,
    num::NonZeroUsize,
    rc::{Rc, Weak},
};

type Entries = RefCell<LruCache<u64, ShapeHandle>>;
thread_local! {
    // Bitmap bindings are player-thread objects; only upgrade them on their owning thread.
    static DEVICE_CACHES: RefCell<Vec<(u64, Weak<Entries>)>> = const { RefCell::new(Vec::new()) };
}

pub(crate) fn evict_for_device(device: u64) -> bool {
    DEVICE_CACHES.with_borrow(|caches| {
        caches.iter().any(|(id, weak)| {
            *id == device
                && weak.upgrade().is_some_and(|cache| {
                    let evicted = cache
                        .try_borrow_mut()
                        .ok()
                        .and_then(|mut cache| cache.pop_lru());
                    evicted.is_some()
                })
        })
    })
}
#[derive(Debug)]
pub(crate) struct ShapeCache {
    entries: Rc<Entries>,
    next: u64,
}

impl Default for ShapeCache {
    fn default() -> Self {
        Self {
            entries: Rc::new(RefCell::new(LruCache::new(
                NonZeroUsize::new(4096).unwrap(),
            ))),
            next: 0,
        }
    }
}

impl ShapeCache {
    pub fn for_device(device: u64) -> Self {
        let cache = Self::default();
        DEVICE_CACHES
            .with_borrow_mut(|caches| caches.push((device, Rc::downgrade(&cache.entries))));
        cache
    }

    pub fn retain(&mut self, handle: ShapeHandle) -> CachedShape {
        let id = self.next;
        self.next = id.checked_add(1).expect("shape cache identity exhausted");
        self.entries.borrow_mut().push(id, handle);
        CachedShape::Managed(Rc::new(Lease {
            entries: Rc::downgrade(&self.entries),
            id,
        }))
    }

    pub fn evict_oldest(&self) -> bool {
        self.entries.borrow_mut().pop_lru().is_some()
    }
    pub fn clear(&self) {
        self.entries.borrow_mut().clear();
    }
}

impl Drop for ShapeCache {
    fn drop(&mut self) {
        let ours = Rc::downgrade(&self.entries);
        DEVICE_CACHES.with_borrow_mut(|caches| caches.retain(|(_, cache)| !cache.ptr_eq(&ours)));
    }
}

#[derive(Debug)]
struct Lease {
    entries: Weak<Entries>,
    id: u64,
}
impl ShapeResidency for Lease {
    fn get(&self) -> Option<ShapeHandle> {
        self.entries.upgrade()?.borrow_mut().get(&self.id).cloned()
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        if let Some(entries) = self.entries.upgrade() {
            entries.borrow_mut().pop(&self.id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruffle_render::backend::ShapeHandleImpl;
    use std::sync::Arc;
    #[derive(Debug)]
    struct Shape;
    impl ShapeHandleImpl for Shape {}
    #[test]
    fn eviction_preserves_recorded_draws_and_unload_releases_residency() {
        let mut cache = ShapeCache::default();
        let live = cache.retain(ShapeHandle(Arc::new(Shape)));
        let recorded = live.get().unwrap();
        let weak = Arc::downgrade(&recorded.0);
        assert!(cache.evict_oldest());
        assert!(live.get().is_none());
        assert!(weak.upgrade().is_some());
        let rebuilt = cache.retain(ShapeHandle(Arc::new(Shape)));
        drop(recorded);
        assert!(weak.upgrade().is_none());
        drop(rebuilt);
        assert!(!cache.evict_oldest());
    }
}
