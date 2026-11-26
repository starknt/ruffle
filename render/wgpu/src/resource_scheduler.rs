//! Shared category admission and weak reclamation hooks. GPU ownership stays in
//! caches; the device governor retains only weak callbacks and byte counters.
use std::sync::{
    Arc, Mutex, Weak,
    atomic::{AtomicU64, Ordering},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReclaimClass {
    Idle = 0,
    Cache = 1,
}
pub trait Reclaim: Send + Sync {
    fn oldest(&self) -> Option<u64>;
    fn evict_oldest(&self) -> bool;
}
struct State {
    bytes: [u64; 2],
    hooks: Vec<(ReclaimClass, Weak<dyn Reclaim>)>,
}
struct Inner {
    limits: [u64; 2],
    clock: AtomicU64,
    state: Mutex<State>,
}
#[derive(Clone)]
pub struct ResourceScheduler(Arc<Inner>);
impl std::fmt::Debug for ResourceScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResourceScheduler").finish_non_exhaustive()
    }
}
#[derive(Debug)]
pub struct CacheTicket {
    scheduler: ResourceScheduler,
    class: ReclaimClass,
    bytes: u64,
}
impl Drop for CacheTicket {
    fn drop(&mut self) {
        self.scheduler
            .0
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .bytes[self.class as usize] -= self.bytes;
    }
}
impl ResourceScheduler {
    pub fn new(idle: u64, cache: u64) -> Self {
        Self(Arc::new(Inner {
            limits: [idle, cache],
            clock: AtomicU64::new(0),
            state: Mutex::new(State {
                bytes: [0, 0],
                hooks: vec![],
            }),
        }))
    }
    pub fn tick(&self) -> u64 {
        self.0.clock.fetch_add(1, Ordering::Relaxed)
    }
    pub fn bytes(&self, class: ReclaimClass) -> u64 {
        self.0
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .bytes[class as usize]
    }
    #[cfg(test)]
    pub(crate) fn live_reclaimers(&self, class: ReclaimClass) -> usize {
        self.0
            .state
            .lock()
            .unwrap()
            .hooks
            .iter()
            .filter(|(c, hook)| *c == class && hook.strong_count() != 0)
            .count()
    }
    pub fn register(&self, class: ReclaimClass, hook: Arc<dyn Reclaim>) {
        let mut state = self
            .0
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.hooks.retain(|(_, h)| h.strong_count() > 0);
        state.hooks.push((class, Arc::downgrade(&hook)));
    }
    /// A speculative allocation must yield to the foreground without evicting other owners.
    pub fn admit_optional(&self, class: ReclaimClass, bytes: u64) -> Option<CacheTicket> {
        let index = class as usize;
        let mut state = self
            .0
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if bytes > self.0.limits[index].saturating_sub(state.bytes[index]) {
            return None;
        }
        state.bytes[index] += bytes;
        Some(CacheTicket {
            scheduler: self.clone(),
            class,
            bytes,
        })
    }
    pub fn admit(&self, class: ReclaimClass, bytes: u64) -> Option<CacheTicket> {
        let index = class as usize;
        let limit = self.0.limits[index];
        if bytes > limit {
            return None;
        }
        let mut exhausted = false;
        loop {
            let mut state = self
                .0
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.bytes[index] <= limit - bytes {
                state.bytes[index] += bytes;
                return Some(CacheTicket {
                    scheduler: self.clone(),
                    class,
                    bytes,
                });
            }
            if exhausted {
                return None;
            }
            drop(state);
            // Even an unsuccessful scan may race a ticket being released.
            exhausted = !self.evict_one(class);
        }
    }
    pub fn evict_one(&self, class: ReclaimClass) -> bool {
        let state = self
            .0
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let initial_bytes = state.bytes[class as usize];
        let mut hooks: Vec<_> = state
            .hooks
            .iter()
            .filter(|(c, _)| *c == class)
            .filter_map(|(_, hook)| hook.upgrade())
            .collect();
        drop(state);

        while let Some((index, _)) = hooks
            .iter()
            .enumerate()
            .filter_map(|(index, hook)| hook.oldest().map(|age| (index, age)))
            .min_by_key(|&(_, age)| age)
        {
            // A checkout can win between oldest() and evict_oldest(). Let
            // the caller recheck its budget if space was freed, otherwise
            // try the remaining owners.
            if hooks.swap_remove(index).evict_oldest() || self.bytes(class) < initial_bytes {
                return true;
            }
        }
        false
    }
    pub fn reclaim(&self) {
        while self.evict_one(ReclaimClass::Idle) {}
        while self.evict_one(ReclaimClass::Cache) {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    struct Cache(Mutex<Vec<(u64, CacheTicket)>>);
    impl Reclaim for Cache {
        fn oldest(&self) -> Option<u64> {
            self.0.lock().unwrap().first().map(|e| e.0)
        }
        fn evict_oldest(&self) -> bool {
            let item = {
                let mut v = self.0.lock().unwrap();
                if v.is_empty() {
                    None
                } else {
                    Some(v.remove(0))
                }
            };
            item.is_some()
        }
    }
    #[test]
    fn cache_admission_is_shared_and_evicts_the_oldest_owner() {
        let scheduler = ResourceScheduler::new(8, 8);
        let a = Arc::new(Cache(Mutex::new(vec![])));
        let b = Arc::new(Cache(Mutex::new(vec![])));
        scheduler.register(ReclaimClass::Cache, a.clone());
        scheduler.register(ReclaimClass::Cache, b.clone());
        let first = scheduler.admit(ReclaimClass::Cache, 6).unwrap();
        a.0.lock().unwrap().push((scheduler.tick(), first));
        let second = scheduler.admit(ReclaimClass::Cache, 4).unwrap();
        b.0.lock().unwrap().push((scheduler.tick(), second));
        assert!(a.0.lock().unwrap().is_empty());
        assert_eq!(scheduler.bytes(ReclaimClass::Cache), 4);
        assert!(scheduler.admit(ReclaimClass::Cache, 9).is_none());
        assert_eq!(
            scheduler.bytes(ReclaimClass::Cache),
            4,
            "oversize entry preserves useful cache"
        );
        drop(b);
        assert_eq!(scheduler.bytes(ReclaimClass::Cache), 0);
    }
    #[test]
    fn admission_rechecks_the_budget_and_remaining_owners_after_a_missed_eviction() {
        // Another user checks the entry out after the scheduler observes its age.
        struct CheckedOut(Mutex<Option<CacheTicket>>);
        impl Reclaim for CheckedOut {
            fn oldest(&self) -> Option<u64> {
                self.0.lock().unwrap().take().map(|_| 0)
            }
            fn evict_oldest(&self) -> bool {
                false
            }
        }

        for requested in [4, 8] {
            let scheduler = ResourceScheduler::new(0, 8);
            let first = Arc::new(CheckedOut(Mutex::new(
                scheduler.admit(ReclaimClass::Cache, 4),
            )));
            let cached = scheduler.admit(ReclaimClass::Cache, 4).unwrap();
            let second = Arc::new(Cache(Mutex::new(vec![(1, cached)])));
            scheduler.register(ReclaimClass::Cache, first.clone());
            scheduler.register(ReclaimClass::Cache, second.clone());
            let ticket = scheduler.admit(ReclaimClass::Cache, requested).unwrap();
            assert_eq!(ticket.bytes, requested);
            assert_eq!(second.0.lock().unwrap().len(), usize::from(requested == 4));
            assert_eq!(scheduler.bytes(ReclaimClass::Cache), 8);
        }
    }

    #[test]
    fn reclaim_tries_the_next_owner_when_the_oldest_entry_is_unavailable() {
        struct Unavailable;
        impl Reclaim for Unavailable {
            fn oldest(&self) -> Option<u64> {
                Some(0)
            }
            fn evict_oldest(&self) -> bool {
                false
            }
        }

        let scheduler = ResourceScheduler::new(0, 4);
        let first = Arc::new(Unavailable);
        let cached = scheduler.admit(ReclaimClass::Cache, 4).unwrap();
        let second = Arc::new(Cache(Mutex::new(vec![(1, cached)])));
        scheduler.register(ReclaimClass::Cache, first.clone());
        scheduler.register(ReclaimClass::Cache, second.clone());
        scheduler.reclaim();
        assert_eq!(scheduler.bytes(ReclaimClass::Cache), 0);
    }

    #[test]
    fn reclaim_calls_run_outside_the_governor_lock() {
        let scheduler = ResourceScheduler::new(8, 8);
        struct Reentrant(ResourceScheduler);
        impl Reclaim for Reentrant {
            fn oldest(&self) -> Option<u64> {
                Some(self.0.tick())
            }
            fn evict_oldest(&self) -> bool {
                assert_eq!(self.0.bytes(ReclaimClass::Idle), 0);
                false
            }
        }
        let cache = Arc::new(Reentrant(scheduler.clone()));
        scheduler.register(ReclaimClass::Idle, cache.clone());
        assert!(!scheduler.evict_one(ReclaimClass::Idle));
    }
}
