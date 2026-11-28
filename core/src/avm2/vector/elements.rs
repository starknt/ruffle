use crate::avm2::{class::Class, value::Value};
use gc_arena::Collect;
use std::ops::RangeBounds;

/// AVM2 coerces at the public method boundary, before borrowing vector storage.
#[derive(Collect, Clone, Debug)]
#[collect(no_drop)]
pub(super) enum Elements<'gc> {
    Int(Vec<i32>),
    Uint(Vec<u32>),
    Number(Vec<f64>),
    Values(Vec<Value<'gc>>),
}

trait Element<'gc>: Copy + Into<Value<'gc>> {
    fn unbox(value: Value<'gc>) -> Self;
}
impl<'gc> Element<'gc> for i32 {
    fn unbox(v: Value<'gc>) -> Self {
        v.as_i32()
    }
}
impl<'gc> Element<'gc> for u32 {
    fn unbox(v: Value<'gc>) -> Self {
        v.as_u32()
    }
}
impl<'gc> Element<'gc> for f64 {
    fn unbox(v: Value<'gc>) -> Self {
        v.as_f64()
    }
}
impl<'gc> Element<'gc> for Value<'gc> {
    fn unbox(v: Self) -> Self {
        v
    }
}

macro_rules! with_vec {
    ($this:expr, |$v:ident| $operation:expr) => {
        match $this {
            Self::Int($v) => $operation,
            Self::Uint($v) => $operation,
            Self::Number($v) => $operation,
            Self::Values($v) => $operation,
        }
    };
}

impl<'gc> Elements<'gc> {
    pub(super) fn new(class: Option<Class<'gc>>) -> Self {
        match class {
            Some(c) if c.is_builtin_int() => Self::Int(Vec::new()),
            Some(c) if c.is_builtin_uint() => Self::Uint(Vec::new()),
            Some(c) if c.is_builtin_number() => Self::Number(Vec::new()),
            _ => Self::Values(Vec::new()),
        }
    }
    pub(super) fn len(&self) -> usize {
        with_vec!(self, |v| v.len())
    }
    pub(super) fn reserve_exact(&mut self, n: usize) {
        with_vec!(self, |v| v.reserve_exact(n))
    }
    pub(super) fn get(&self, i: usize) -> Option<Value<'gc>> {
        with_vec!(self, |v| v.get(i).copied().map(Into::into))
    }
    pub(super) fn set(&mut self, i: usize, value: Value<'gc>) {
        with_vec!(self, |v| v[i] = Element::unbox(value))
    }
    pub(super) fn resize(&mut self, n: usize, value: Value<'gc>) {
        with_vec!(self, |v| v.resize(n, Element::unbox(value)))
    }
    pub(super) fn push(&mut self, value: Value<'gc>) {
        with_vec!(self, |v| v.push(Element::unbox(value)))
    }
    pub(super) fn insert(&mut self, i: usize, value: Value<'gc>) {
        with_vec!(self, |v| v.insert(i, Element::unbox(value)))
    }
    pub(super) fn remove(&mut self, i: usize) -> Value<'gc> {
        with_vec!(self, |v| v.remove(i).into())
    }
    pub(super) fn pop(&mut self) -> Option<Value<'gc>> {
        with_vec!(self, |v| v.pop().map(Into::into))
    }
    pub(super) fn reverse(&mut self) {
        with_vec!(self, |v| v.reverse())
    }
    pub(super) fn replace(&mut self, values: Vec<Value<'gc>>) {
        if let Self::Values(current) = self {
            *current = values;
        } else {
            self.replace_iter(values);
        }
    }
    pub(super) fn replace_iter(&mut self, values: impl IntoIterator<Item = Value<'gc>>) {
        fn fill<'gc, T: Element<'gc>>(
            v: &mut Vec<T>,
            values: impl IntoIterator<Item = Value<'gc>>,
        ) {
            v.clear();
            v.extend(values.into_iter().map(T::unbox));
        }
        with_vec!(self, |v| fill(v, values));
    }
    pub(super) fn splice(
        &mut self,
        range: impl RangeBounds<usize>,
        values: Vec<Value<'gc>>,
    ) -> Vec<Value<'gc>> {
        with_vec!(self, |v| v
            .splice(range, values.into_iter().map(Element::unbox))
            .map(Into::into)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn typed_splice_and_iteration_preserve_uint_and_ieee_values() {
        let mut u = Elements::Uint(vec![u32::MAX]);
        let removed = u.splice(..1, vec![0x8000_0000u32.into()]);
        assert_eq!(removed[0].as_u32(), u32::MAX);
        assert_eq!(u.get(0).unwrap().as_u32(), 0x8000_0000);
        let nan = f64::from_bits(0x7ff8_0000_0000_0042);
        let mut n = Elements::Number(vec![]);
        n.replace_iter([(-0.0).into(), nan.into()]);
        assert_eq!(n.get(0).unwrap().as_f64().to_bits(), (-0.0f64).to_bits());
        assert_eq!(n.pop().unwrap().as_f64().to_bits(), nan.to_bits());
    }
}
