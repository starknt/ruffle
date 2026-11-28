use super::{MemoryKind, MemoryLease, MemoryScope};
use std::ops::{Deref, DerefMut};

/// The lease travels with the backing allocation, including through a shared owner such as Bytes.
#[derive(Debug)]
pub struct MemoryBuffer<T> {
    data: Vec<T>,
    charge: MemoryLease,
}
impl<T> MemoryBuffer<T> {
    /// Reserve the budget before allocating an optional buffer and account for its actual capacity.
    pub fn try_with_capacity(
        scope: &MemoryScope,
        kind: MemoryKind,
        capacity: usize,
    ) -> Option<Self> {
        let bytes = capacity.checked_mul(size_of::<T>())? as u64;
        let reservation = scope.try_reserve(kind, bytes)?;
        let mut data = Vec::<T>::new();
        data.try_reserve_exact(capacity).ok()?;
        let charge = reservation.commit(data.capacity().saturating_mul(size_of::<T>()) as u64);
        Some(Self { data, charge })
    }

    pub fn new(scope: &MemoryScope, kind: MemoryKind, data: Vec<T>) -> Self {
        let charge = scope.track(kind, data.capacity().saturating_mul(size_of::<T>()) as u64);
        Self { data, charge }
    }
    pub fn take(&mut self) -> Self {
        let empty = Self::new(self.charge.scope(), self.charge.kind, Vec::new());
        std::mem::replace(self, empty)
    }
    pub fn clear(&mut self) {
        self.data.clear();
    }
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional);
        self.update_capacity();
    }
    pub fn reserve_exact(&mut self, additional: usize) {
        self.data.reserve_exact(additional);
        self.update_capacity();
    }
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
    pub fn shrink_to_fit(&mut self) {
        self.data.shrink_to_fit();
        self.update_capacity();
    }
    fn update_capacity(&mut self) {
        self.charge
            .resize(self.data.capacity().saturating_mul(size_of::<T>()) as u64);
    }
}
impl<T: Clone> MemoryBuffer<T> {
    pub fn extend_from_slice(&mut self, values: &[T]) {
        self.data.extend_from_slice(values);
        self.update_capacity();
    }
    pub fn resize(&mut self, len: usize, value: T) {
        self.data.resize(len, value);
        self.update_capacity();
    }
}
impl<T: Clone> Clone for MemoryBuffer<T> {
    fn clone(&self) -> Self {
        Self::new(self.charge.scope(), self.charge.kind, self.data.clone())
    }
}
impl<T> AsRef<[T]> for MemoryBuffer<T> {
    fn as_ref(&self) -> &[T] {
        &self.data
    }
}
impl<T> Deref for MemoryBuffer<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.data
    }
}
impl<T> DerefMut for MemoryBuffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}
