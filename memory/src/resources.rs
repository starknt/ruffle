//! Weak content-addressed interning. The index never keeps a movie, decoder or GPU resource alive.
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ContentId(pub [u8; 32]);

impl ContentId {
    pub fn from_parts(parts: &[&[u8]]) -> Self {
        let mut hash = Sha256::new();
        for part in parts {
            hash.update((part.len() as u64).to_le_bytes());
            hash.update(part);
        }
        Self(hash.finalize().into())
    }
}

type Slot<T> = Arc<Mutex<Weak<T>>>;

#[derive(Debug)]
pub struct ResourceManager<T> {
    entries: Mutex<HashMap<ContentId, Slot<T>>>,
}
impl<T> Default for ResourceManager<T> {
    fn default() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }
}
impl<T> ResourceManager<T> {
    /// Only construction of the same content serializes. Holding a slot also protects
    /// an in-progress producer from being removed by another thread's pruning pass.
    pub fn intern(&self, key: ContentId, make: impl FnOnce() -> Arc<T>) -> Arc<T> {
        let slot = {
            let mut entries = self
                .entries
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if !entries.contains_key(&key) && entries.len() >= entries.capacity() {
                entries.retain(|_, slot| {
                    Arc::strong_count(slot) > 1
                        || slot
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .strong_count()
                            != 0
                });
            }
            entries.entry(key).or_default().clone()
        };
        let mut weak = slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(live) = weak.upgrade() {
            return live;
        }
        let value = make();
        *weak = Arc::downgrade(&value);
        value
    }
    pub fn prune(&self) {
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        entries.retain(|_, slot| {
            Arc::strong_count(slot) > 1
                || slot
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .strong_count()
                    != 0
        });
        entries.shrink_to_fit();
    }
}
