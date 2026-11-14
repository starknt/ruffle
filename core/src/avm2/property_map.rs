//! Property map

use crate::avm2::AvmString;
use crate::avm2::Multiname;
use crate::avm2::Namespace;
use crate::avm2::QName;
use gc_arena::Collect;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::hash::{BuildHasher, Hash, Hasher};

#[inline]
fn hash_cached_fnv<H: Hasher>(cached: u64, state: &mut H) {
    state.write_u64(cached);
}

#[derive(Clone, Copy, Default)]
struct PropertyMapBuildHasher;

struct PropertyMapHasher {
    hash: u64,
}

impl Hasher for PropertyMapHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut hasher = fnv::FnvHasher::default();
        hasher.write(bytes);
        self.hash = hasher.finish();
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.hash = value;
    }
}

impl BuildHasher for PropertyMapBuildHasher {
    type Hasher = PropertyMapHasher;

    #[inline]
    fn build_hasher(&self) -> Self::Hasher {
        PropertyMapHasher { hash: 0 }
    }
}

#[derive(Clone, Copy, Debug, Collect)]
#[collect(no_drop)]
struct PropertyMapKey<'gc>(AvmString<'gc>);

impl PartialEq for PropertyMapKey<'_> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for PropertyMapKey<'_> {}

impl Hash for PropertyMapKey<'_> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_cached_fnv(self.0.fnv_hash(), state);
    }
}

#[cfg(test)]
mod tests {
    use super::{PropertyMapBuildHasher, hash_cached_fnv};
    use fnv::FnvHasher;
    use std::hash::{BuildHasher, Hasher};

    #[test]
    fn property_map_key_hashes_from_cached_fnv_value() {
        let cached = 0x0102_0304_0506_0708;
        let mut actual = FnvHasher::default();
        hash_cached_fnv(cached, &mut actual);

        let mut expected = FnvHasher::default();
        expected.write_u64(cached);
        assert_eq!(actual.finish(), expected.finish());
    }

    #[test]
    fn property_map_build_hasher_returns_cached_fnv_value() {
        let cached = 0x0102_0304_0506_0708;
        let actual = PropertyMapBuildHasher.hash_one(&CachedHash(cached));
        assert_eq!(actual, cached);
    }

    struct CachedHash(u64);

    impl std::hash::Hash for CachedHash {
        fn hash<H: Hasher>(&self, state: &mut H) {
            hash_cached_fnv(self.0, state);
        }
    }
}

/// Type which represents named properties on an object.
///
/// This type exposes interfaces akin to `HashMap<QName<'gc>, V>`, and is
/// intended to serve as a drop-in replacement optimized for objects where few
/// properties have overlapping local names. However, we have made slight
/// changes to the API in the following cases:
///
///  * Iterators return tuples of namespace, local-name, and value; rather than
///    a qualified name and value pair.
///  * Only `HashMap` methods and traits that we need are implemented.
///
/// The internal structure of the `PropertyMap` technically allows storage of
/// multiple values per `QName`. It's implementation enforces the invariant
/// that each `QName` only have one associated `V`.
#[derive(Clone, Debug, Collect)]
#[collect(no_drop)]
pub struct PropertyMap<'gc, V>(
    HashMap<PropertyMapKey<'gc>, SmallVec<[(Namespace<'gc>, V); 2]>, PropertyMapBuildHasher>,
);

impl<V> Default for PropertyMap<'_, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'gc, V> PropertyMap<'gc, V> {
    pub fn new() -> Self {
        Self(Default::default())
    }

    pub fn get(&self, name: QName<'gc>) -> Option<&V> {
        self.0
            .get(&PropertyMapKey(name.local_name()))
            .iter()
            .find_map(|v| {
                v.iter()
                    .filter(|(n, _)| n.matches_ns(name.namespace()))
                    .map(|(_, v)| v)
                    .next()
            })
    }

    pub fn get_for_multiname(&self, name: &Multiname<'gc>) -> Option<&V> {
        self.get_key_value_for_multiname(name)
            .map(|(_, value)| value)
    }

    /// Return the key held by this map, not the caller's possibly temporary
    /// string/namespace. VTable's identity cache relies on that ownership.
    pub fn get_key_value_for_multiname(&self, name: &Multiname<'gc>) -> Option<(QName<'gc>, &V)> {
        if name.has_lazy_component() {
            unreachable!("Lookup on lazy Multiname should never happen ({:?})", name);
        }
        if let Some(local_name) = name.local_name() {
            self.0
                .get_key_value(&PropertyMapKey(local_name))
                .and_then(|(key, values)| {
                    values
                        .iter()
                        .filter(|(n, _)| name.namespace_set().iter().any(|ns| n.matches_ns(*ns)))
                        .map(|(ns, v)| (QName::new(*ns, key.0), v))
                        .next()
                })
        } else {
            None
        }
    }

    pub fn get_with_ns_for_multiname(&self, name: &Multiname<'gc>) -> Option<(Namespace<'gc>, &V)> {
        self.get_key_value_for_multiname(name)
            .map(|(key, value)| (key.namespace(), value))
    }

    pub fn get_mut(&mut self, name: QName<'gc>) -> Option<&mut V> {
        if let Some(bucket) = self.0.get_mut(&PropertyMapKey(name.local_name()))
            && let Some((_, old_value)) = bucket
                .iter_mut()
                .find(|(n, _)| n.matches_ns(name.namespace()))
        {
            return Some(old_value);
        }

        None
    }

    pub fn contains_key(&self, name: QName<'gc>) -> bool {
        self.0
            .get(&PropertyMapKey(name.local_name()))
            .iter()
            .any(|v| v.iter().any(|(n, _)| n.matches_ns(name.namespace())))
    }

    pub fn iter(&self) -> impl Iterator<Item = (AvmString<'gc>, Namespace<'gc>, &V)> {
        self.0
            .iter()
            .flat_map(|(k, vs)| vs.iter().map(|(ns, v)| (k.0, *ns, v)))
    }

    pub fn insert(&mut self, name: QName<'gc>, value: V) -> Option<V> {
        let bucket = self.0.entry(PropertyMapKey(name.local_name())).or_default();

        if let Some(position) = bucket
            .iter()
            .position(|(n, _)| n.matches_ns(name.namespace()))
        {
            let (_, old_value) = bucket.remove(position);
            bucket.insert(0, (name.namespace(), value));

            Some(old_value)
        } else {
            bucket.insert(0, (name.namespace(), value));

            None
        }
    }

    /// Preserves Flash Player ordering in scope caches and application domains.
    pub fn insert_at_end(&mut self, name: QName<'gc>, value: V) -> Option<V> {
        let bucket = self.0.entry(PropertyMapKey(name.local_name())).or_default();

        if let Some((_, old_value)) = bucket
            .iter_mut()
            .find(|(n, _)| n.matches_ns(name.namespace()))
        {
            Some(std::mem::replace(old_value, value))
        } else {
            bucket.push((name.namespace(), value));
            None
        }
    }

    pub fn remove(&mut self, name: QName<'gc>) -> Option<V> {
        let bucket = self.0.get_mut(&PropertyMapKey(name.local_name()));

        if let Some(bucket) = bucket {
            let position = bucket
                .iter_mut()
                .enumerate()
                .find(|(_, (n, _))| n.matches_ns(name.namespace()));
            if let Some((position, _)) = position {
                return Some(bucket.remove(position).1);
            }
        }

        None
    }
}
