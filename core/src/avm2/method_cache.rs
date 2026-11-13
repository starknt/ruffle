use gc_arena::Collect;

/// Per-object cache of methods bound to a dispatch ID.
///
/// A dense table is still used by stock Ruffle because it is the original
/// representation. Optimized objects use a sorted sparse table: most objects
/// bind only a handful of methods, while dispatch IDs can be far apart. This
/// avoids allocating and tracing thousands of empty GC slots for one method.
#[derive(Clone, Collect)]
#[collect(no_drop)]
pub(super) enum MethodCache<T> {
    Dense(Vec<Option<T>>),
    SparseEmpty,
    SparseOne(usize, T),
    SparseMany(Vec<(usize, T)>),
}

impl<T> MethodCache<T> {
    pub(super) fn new(sparse: bool) -> Self {
        if sparse {
            Self::SparseEmpty
        } else {
            Self::Dense(Vec::new())
        }
    }

    pub(super) fn get(&self, id: usize) -> Option<&T> {
        match self {
            Self::Dense(entries) => entries.get(id).and_then(Option::as_ref),
            Self::SparseEmpty => None,
            Self::SparseOne(entry_id, value) => (*entry_id == id).then_some(value),
            Self::SparseMany(entries) => entries
                .binary_search_by_key(&id, |entry| entry.0)
                .ok()
                .map(|index| &entries[index].1),
        }
    }

    pub(super) fn get_copied(&self, id: usize) -> Option<T>
    where
        T: Copy,
    {
        self.get(id).copied()
    }

    pub(super) fn insert(&mut self, id: usize, value: T) {
        let current = std::mem::replace(self, Self::SparseEmpty);
        *self = match current {
            Self::Dense(mut entries) => {
                if entries.len() <= id {
                    entries.resize_with(id + 1, || None);
                }
                entries[id] = Some(value);
                Self::Dense(entries)
            }
            Self::SparseEmpty => Self::SparseOne(id, value),
            Self::SparseOne(entry_id, _) if entry_id == id => Self::SparseOne(id, value),
            Self::SparseOne(entry_id, old_value) => {
                let mut entries = Vec::with_capacity(2);
                if entry_id < id {
                    entries.push((entry_id, old_value));
                    entries.push((id, value));
                } else {
                    entries.push((id, value));
                    entries.push((entry_id, old_value));
                }
                Self::SparseMany(entries)
            }
            Self::SparseMany(mut entries) => {
                match entries.binary_search_by_key(&id, |entry| entry.0) {
                    Ok(index) => entries[index].1 = value,
                    Err(index) => entries.insert(index, (id, value)),
                }
                Self::SparseMany(entries)
            }
        };
    }

    #[cfg(test)]
    fn allocated_bytes(&self) -> usize {
        match self {
            Self::Dense(entries) => entries.capacity() * std::mem::size_of::<Option<T>>(),
            Self::SparseEmpty | Self::SparseOne(_, _) => 0,
            Self::SparseMany(entries) => entries.capacity() * std::mem::size_of::<(usize, T)>(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::MethodCache;
    use std::mem::size_of;

    #[test]
    fn sparse_cache_has_bounded_object_footprint() {
        assert!(
            size_of::<MethodCache<u8>>() <= size_of::<Vec<Option<u8>>>() + size_of::<usize>(),
            "sparse cache grew beyond one pointer word over the stock Vec footprint"
        );
    }

    #[test]
    fn sparse_dispatch_ids_do_not_allocate_empty_slots() {
        let mut cache = MethodCache::new(true);
        cache.insert(4096, 1_u8);
        assert_eq!(cache.get(4096), Some(&1));
        assert_eq!(cache.get(4095), None);
        assert!(cache.allocated_bytes() < 4096);
    }

    #[test]
    fn sparse_small_cache_entries_stay_inline() {
        let mut cache = MethodCache::new(true);
        cache.insert(3, 1_u8);

        assert_eq!(cache.get(3), Some(&1));
        assert_eq!(cache.allocated_bytes(), 0);
    }

    #[test]
    fn sparse_second_entry_spills_without_losing_lookup() {
        let mut cache = MethodCache::new(true);
        cache.insert(3, 1_u8);
        cache.insert(17, 2_u8);

        assert_eq!(cache.get(3), Some(&1));
        assert_eq!(cache.get(17), Some(&2));
        assert!(cache.allocated_bytes() > 0);
    }

    #[test]
    fn insertion_order_and_replacement_preserve_identity() {
        for sparse in [false, true] {
            let mut cache = MethodCache::new(sparse);
            for (id, value) in [(300, "first"), (0, "zero"), (41, "middle")] {
                cache.insert(id, value);
            }
            assert_eq!(cache.get(0), Some(&"zero"));
            assert_eq!(cache.get(41), Some(&"middle"));
            assert_eq!(cache.get(300), Some(&"first"));
            assert_eq!(cache.get(1), None);
            cache.insert(41, "replacement");
            assert_eq!(cache.get(41), Some(&"replacement"));
        }
    }
}
