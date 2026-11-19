use ruffle_render::backend::ShapeHandle;

/// The scale ratio threshold beyond which a graphic or a drawing will be retessellated.
const RETESSELLATION_SCALE_THRESHOLD: f32 = 2.0;

/// The inverse of the retessellation scale threshold
/// Used to avoid computing the inverse repeatedly when comparing scale ratios against the threshold.
const RETESSELLATION_SCALE_THRESHOLD_INVERSE: f32 = 1.0 / RETESSELLATION_SCALE_THRESHOLD;

/// The maximum number of retessellated shapes to cache per original shape.
const RETESSELLATION_CACHE_SIZE: usize = 4;

#[derive(Clone, Debug)]
pub(crate) struct TessellationCache {
    entries: [Option<(f32, ShapeHandle)>; RETESSELLATION_CACHE_SIZE],
    len: usize,
}

/// A cache for tessellated shapes at different scales, using a simple LRU eviction policy.
/// (LRU index: 0, MRU index: len - 1)
///
/// The cache stores a fixed number of entries, and when it is full, the least recently
/// used entry is evicted to make room for new entries.
///
/// This is used to avoid retessellating shapes at similar scales multiple times, which can be expensive.
impl TessellationCache {
    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Creates a new, empty tessellation cache.
    pub(crate) fn new() -> Self {
        Self {
            entries: std::array::from_fn(|_| None),
            len: 0,
        }
    }

    /// Finds the cached shape handle with the closest scale to the target scale.
    ///
    /// If the closest scale is NOT within the retessellation threshold,
    /// then `None` is returned, indicating that the shape should be retessellated at the target scale.
    pub(crate) fn find_near_and_touch(&mut self, target_scale: f32) -> Option<ShapeHandle> {
        let mut best_index = None;
        let mut best_deviation = f32::INFINITY;

        for (index, entry) in self.entries[..self.len].iter().enumerate() {
            if let Some((cached_scale, _)) = entry {
                let ratio = f32::abs(target_scale / cached_scale);

                // Check if the cached scale is within the retessellation threshold of the target scale.
                if ratio <= RETESSELLATION_SCALE_THRESHOLD
                    && ratio >= RETESSELLATION_SCALE_THRESHOLD_INVERSE
                {
                    // Choose the cached entry with the smallest deviation from the target scale.
                    let deviation = f32::abs(ratio - 1.0);

                    if deviation < best_deviation {
                        best_deviation = deviation;
                        best_index = Some(index);
                    }
                }
            }
        }

        // If we found a suitable cached entry, move it to the most recently used position.
        best_index.map(|index| self.touch_entry(index))
    }

    /// Inserts a new cached shape handle with the given scale.
    ///
    /// If the cache is full, the least recently used entry will be evicted to make room for the new entry.
    /// We assume that the caller has already checked that the new entry is not too similar to existing entries.
    pub(crate) fn insert(&mut self, scale: f32, handle: ShapeHandle) {
        if self.len < RETESSELLATION_CACHE_SIZE {
            // If the cache is not full, simply add the new entry at the end.
            self.entries[self.len] = Some((scale, handle));
            self.len += 1;
            return;
        }

        // If the cache is full, evict the least recently used entry.
        for i in 1..self.len {
            self.entries[i - 1] = self.entries[i].take();
        }

        self.entries[self.len - 1] = Some((scale, handle));
    }

    /// Returns the number of cached entries currently stored in the cache.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Moves the entry at the given index to the most recently used position and returns its shape handle.
    fn touch_entry(&mut self, index: usize) -> ShapeHandle {
        if index == self.len - 1 {
            // The entry is already the most recently used, so we can return its handle directly.
            return self.entries[index]
                .as_ref()
                .expect("tessellation cache entry exists")
                .1
                .clone();
        }

        let entry = self.entries[index].take().expect("entry exists");

        for i in index + 1..self.len {
            self.entries[i - 1] = self.entries[i].take();
        }

        // Move touched entry to MRU position (end)
        self.entries[self.len - 1] = Some(entry.clone());
        entry.1
    }
}

#[cfg(test)]
mod tests {
    use super::TessellationCache;
    use ruffle_render::backend::{ShapeHandle, ShapeHandleImpl};
    use std::sync::Arc;

    #[derive(Debug)]
    struct TestShape;

    impl ShapeHandleImpl for TestShape {}

    fn shape(id: usize) -> ShapeHandle {
        let _ = id;
        ShapeHandle(Arc::new(TestShape))
    }

    #[test]
    fn reuses_nearby_scale_and_promotes_entry() {
        let first = shape(1);
        let second = shape(2);
        let mut cache = TessellationCache::new();
        cache.insert(1.0, first.clone());
        cache.insert(4.0, second);

        let reused = cache.find_near_and_touch(1.5).unwrap();
        assert!(Arc::ptr_eq(&reused.0, &first.0));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn evicts_least_recently_used_scale() {
        let mut cache = TessellationCache::new();
        let first = shape(0);
        let second = shape(1);
        cache.insert(1.0, first);
        cache.insert(2.0, second.clone());
        cache.insert(3.0, shape(2));
        cache.insert(4.0, shape(3));
        let _ = cache.find_near_and_touch(1.0);
        cache.insert(8.0, shape(8));

        let nearby = cache.find_near_and_touch(2.0).unwrap();
        assert!(!Arc::ptr_eq(&nearby.0, &second.0));
        assert!(cache.find_near_and_touch(8.0).is_some());
    }
}
