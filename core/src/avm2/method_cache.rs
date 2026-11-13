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
    Sparse(Vec<(usize, T)>),
}

impl<T> MethodCache<T> {
    pub(super) fn new(sparse: bool) -> Self {
        if sparse {
            Self::Sparse(Vec::new())
        } else {
            Self::Dense(Vec::new())
        }
    }

    pub(super) fn get(&self, id: usize) -> Option<&T> {
        match self {
            Self::Dense(entries) => entries.get(id).and_then(Option::as_ref),
            Self::Sparse(entries) => entries
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
        match self {
            Self::Dense(entries) => {
                if entries.len() <= id {
                    entries.resize_with(id + 1, || None);
                }
                entries[id] = Some(value);
            }
            Self::Sparse(entries) => match entries.binary_search_by_key(&id, |entry| entry.0) {
                Ok(index) => entries[index].1 = value,
                Err(index) => entries.insert(index, (id, value)),
            },
        }
    }

    #[cfg(test)]
    fn allocated_bytes(&self) -> usize {
        match self {
            Self::Dense(entries) => entries.capacity() * std::mem::size_of::<Option<T>>(),
            Self::Sparse(entries) => entries.capacity() * std::mem::size_of::<(usize, T)>(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::MethodCache;

    #[test]
    fn sparse_dispatch_ids_do_not_allocate_empty_slots() {
        let mut cache = MethodCache::new(true);
        cache.insert(4096, 1_u8);
        assert_eq!(cache.get(4096), Some(&1));
        assert_eq!(cache.get(4095), None);
        assert!(cache.allocated_bytes() < 4096);
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
