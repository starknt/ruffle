//! Strong references to reconstructible uploads, bounded independently of AS3 GC.
//! Dropping our reference never destroys resources still owned by recorded draws.
use lru::LruCache;
use ruffle_render::{bitmap::BitmapHandle, profile::BitmapResidencyStats};
use std::{num::NonZeroUsize, sync::Arc};

pub struct BitmapResidency {
    entries: LruCache<usize, (BitmapHandle, u64)>,
    limit: u64,
    stats: BitmapResidencyStats,
}
impl BitmapResidency {
    pub fn new(limit: u64) -> Self {
        Self {
            entries: LruCache::new(NonZeroUsize::new(4096).unwrap()),
            limit,
            stats: Default::default(),
        }
    }
    pub fn retain(&mut self, handle: &BitmapHandle, bytes: u64) {
        // Each cached identity is owned by its entry, so allocator reuse cannot
        // alias another live entry. No GC pointer or source identifier is stored.
        let key = Arc::as_ptr(&handle.0) as *const () as usize;
        if self.entries.get(&key).is_some() {
            self.stats.hits += 1;
            return;
        }
        if bytes > self.limit {
            return; // One oversized image must not flush the useful working set.
        }
        while self.stats.retained_bytes > self.limit - bytes {
            let (_, (_, bytes)) = self.entries.pop_lru().unwrap();
            self.stats.retained_bytes -= bytes;
            self.stats.evictions += 1;
        }
        if let Some((_, (_, bytes))) = self.entries.push(key, (handle.clone(), bytes)) {
            self.stats.retained_bytes -= bytes;
            self.stats.evictions += 1;
        }
        self.stats.retained_bytes += bytes;
        self.stats.insertions += 1;
        self.stats.retained_entries = self.entries.len();
    }
    pub fn stats(&self) -> BitmapResidencyStats {
        self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruffle_render::bitmap::BitmapHandleImpl;
    #[derive(Debug)]
    struct Image;
    impl BitmapHandleImpl for Image {}
    fn image() -> BitmapHandle {
        BitmapHandle(Arc::new(Image))
    }

    #[test]
    fn budget_evicts_the_oldest_upload_without_invalidating_a_recorded_draw() {
        let mut cache = BitmapResidency::new(8);
        let first = image();
        let second = image();
        let third = image();
        let second_weak = Arc::downgrade(&second.0);
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
        let first_weak = Arc::downgrade(&first.0);
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
