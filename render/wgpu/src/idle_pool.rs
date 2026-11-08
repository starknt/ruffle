//! One byte budget across main/offscreen textures and mapped readback buffers.
//! Entries within an encoder are used in recorded order. Across submissions,
//! wait for queue completion before handing ownership to another operation.
use std::{
    any::Any,
    cell::RefCell,
    collections::BTreeMap,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
pub const IDLE_LIMIT: u64 = 64 * 1024 * 1024;
#[derive(Clone)]
pub struct Budget(Rc<RefCell<State>>);
struct Item {
    pool: u64,
    kind: usize,
    bytes: u64,
    epoch: u64,
    value: Box<dyn Any>,
}
struct State {
    limit: Option<u64>,
    next: u64,
    epoch: u64,
    completed: Arc<AtomicU64>,
    items: BTreeMap<u64, Item>,
    by_pool: fnv::FnvHashMap<u64, Vec<u64>>,
    idle: [u64; 2],
    peak: [u64; 2],
    peak_total: u64,
}
impl std::fmt::Debug for Budget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Budget")
            .field("idle", &self.0.borrow().idle)
            .finish()
    }
}
impl Budget {
    pub fn new(limit: Option<u64>) -> Self {
        Self(Rc::new(RefCell::new(State {
            limit,
            next: 0,
            epoch: 1,
            completed: Arc::default(),
            items: Default::default(),
            by_pool: Default::default(),
            idle: [0; 2],
            peak: [0; 2],
            peak_total: 0,
        })))
    }
    pub fn contains(&self, key: u64) -> bool {
        self.0.borrow().by_pool.contains_key(&key)
    }
    pub fn limited(&self) -> bool {
        self.0.borrow().limit.is_some()
    }
    pub fn stock_delta(&self, kind: usize, add: u64, remove: u64) {
        let mut s = self.0.borrow_mut();
        s.idle[kind] = s.idle[kind] - remove + add;
        s.peak[kind] = s.peak[kind].max(s.idle[kind]);
        s.peak_total = s.peak_total.max(s.idle.iter().sum());
    }
    pub fn key(&self) -> u64 {
        let mut s = self.0.borrow_mut();
        s.next += 1;
        s.next
    }
    pub fn submitted(&self, queue: &wgpu::Queue) {
        let mut s = self.0.borrow_mut();
        if s.limit.is_none() {
            return;
        }
        let epoch = s.epoch;
        s.epoch += 1;
        let complete = s.completed.clone();
        queue.on_submitted_work_done(move || {
            complete.fetch_max(epoch, Ordering::Release);
        });
    }
    pub fn put<T: 'static>(&self, pool: u64, kind: usize, bytes: u64, item: T) {
        let mut s = self.0.borrow_mut();
        if s.limit.is_some_and(|l| bytes > l) {
            return;
        }
        while s
            .limit
            .is_some_and(|l| s.idle.iter().sum::<u64>() + bytes > l)
        {
            let id = *s.items.first_key_value().unwrap().0;
            s.remove(id);
        }
        s.next += 1;
        let id = s.next;
        let epoch = s.epoch;
        s.items.insert(
            id,
            Item {
                pool,
                kind,
                bytes,
                epoch,
                value: Box::new(item),
            },
        );
        s.by_pool.entry(pool).or_default().push(id);
        s.idle[kind] += bytes;
        s.peak[kind] = s.peak[kind].max(s.idle[kind]);
        s.peak_total = s.peak_total.max(s.idle.iter().sum());
    }
    pub fn take<T: 'static, C: Ord>(&self, pool: u64, cost: impl Fn(&T) -> Option<C>) -> Option<T> {
        let mut s = self.0.borrow_mut();
        let completed = s.completed.load(Ordering::Acquire);
        let id = s
            .by_pool
            .get(&pool)?
            .iter()
            .copied()
            .filter_map(|id| {
                let i = &s.items[&id];
                if s.limit.is_some() && i.epoch != s.epoch && i.epoch > completed {
                    return None;
                }
                cost(i.value.downcast_ref::<T>()?).map(|c| (c, id))
            })
            .min_by(|a, b| a.0.cmp(&b.0))?
            .1;
        s.remove(id).value.downcast().ok().map(|v| *v)
    }
    pub fn clear_pool(&self, pool: u64) {
        let mut s = self.0.borrow_mut();
        if let Some(ids) = s.by_pool.remove(&pool) {
            for id in ids {
                let i = s.items.remove(&id).unwrap();
                s.idle[i.kind] -= i.bytes;
            }
        }
    }
    pub fn stats(&self) -> ([u64; 2], [u64; 2], u64) {
        let s = self.0.borrow();
        (s.idle, s.peak, s.peak_total)
    }
}
impl State {
    fn remove(&mut self, id: u64) -> Item {
        let i = self.items.remove(&id).unwrap();
        let keys = self.by_pool.get_mut(&i.pool).unwrap();
        keys.swap_remove(keys.iter().position(|k| *k == id).unwrap());
        if keys.is_empty() {
            self.by_pool.remove(&i.pool);
        }
        self.idle[i.kind] -= i.bytes;
        i
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn joint_lru_and_submission_order() {
        let b = Budget::new(Some(8));
        let a = b.key();
        let r = b.key();
        b.put(a, 0, 4, 10u8);
        b.put(r, 1, 3, 20u8);
        assert_eq!(b.take(a, |_: &u8| Some(())), Some(10));
        b.put(a, 0, 4, 11u8);
        b.put(a, 0, 3, 12u8);
        assert_eq!(b.take(r, |_: &u8| Some(())), None);
        assert_eq!(b.stats().0, [7, 0]);
        b.0.borrow_mut().epoch += 1;
        assert_eq!(b.take(a, |_: &u8| Some(())), None);
        b.0.borrow().completed.store(1, Ordering::Release);
        assert!(b.take(a, |_: &u8| Some(())).is_some());
        b.clear_pool(a);
        assert_eq!(b.stats().0, [0, 0]);
        assert!(b.stats().2 <= 8);
    }
}
