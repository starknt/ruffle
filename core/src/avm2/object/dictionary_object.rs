//! Object representation for `flash.utils.Dictionary`

use crate::avm2::Error;
use crate::avm2::activation::Activation;
use crate::avm2::dynamic_map::DynamicKey;
use crate::avm2::object::script_object::ScriptObjectData;
use crate::avm2::object::{ClassObject, Object, TObject};
use crate::avm2::value::Value;
use crate::string::AvmString;
use core::fmt;
use gc_arena::collect::Trace;
use gc_arena::{Collect, Gc, GcWeak, Mutation};
use ruffle_common::utils::HasPrefixField;
use std::cell::Cell;

/// A class instance allocator that allocates Dictionary objects.
pub fn dictionary_allocator<'gc>(
    class: ClassObject<'gc>,
    activation: &mut Activation<'_, 'gc>,
) -> Result<Object<'gc>, Error<'gc>> {
    let base = ScriptObjectData::new(class);

    Ok(DictionaryObject(Gc::new(
        activation.gc(),
        DictionaryObjectData {
            base,
            weak_keys: Cell::new(false),
        },
    ))
    .into())
}

/// An object that allows associations between objects and values.
///
/// This is implemented by way of "object space", parallel to the property
/// space that ordinary properties live in. This space has no namespaces, and
/// keys are objects instead of strings.
#[derive(Clone, Collect, Copy)]
#[collect(no_drop)]
pub struct DictionaryObject<'gc>(pub Gc<'gc, DictionaryObjectData<'gc>>);

#[derive(Clone, Collect, Copy, Debug)]
#[collect(no_drop)]
pub struct DictionaryObjectWeak<'gc>(pub GcWeak<'gc, DictionaryObjectData<'gc>>);

impl fmt::Debug for DictionaryObject<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DictionaryObject")
            .field("ptr", &Gc::as_ptr(self.0))
            .finish()
    }
}

#[derive(Clone, HasPrefixField)]
#[repr(C, align(8))]
pub struct DictionaryObjectData<'gc> {
    /// Base script object
    base: ScriptObjectData<'gc>,
    weak_keys: Cell<bool>,
}

// SAFETY: The base owns every GC pointer. Pruning only removes references.
unsafe impl<'gc> Collect<'gc> for DictionaryObjectData<'gc> {
    fn trace<C: Trace<'gc>>(&self, cc: &mut C) {
        if self.weak_keys.get() {
            // Release a dead key's value even if the dictionary is never read again.
            self.base.prune_dropped_weak_keys();
        }
        cc.trace(&self.base);
    }
}

impl<'gc> DictionaryObject<'gc> {
    pub fn use_weak_keys(self) {
        self.0.weak_keys.set(true);
    }

    pub fn has_weak_keys(self) -> bool {
        self.0.weak_keys.get()
    }

    fn object_key(self, name: Object<'gc>) -> DynamicKey<'gc> {
        if self.has_weak_keys() {
            DynamicKey::WeakObject(name.downgrade())
        } else {
            DynamicKey::Object(name)
        }
    }

    /// Retrieve a value in the dictionary's object space.
    pub fn get_property_by_object(self, name: Object<'gc>) -> Value<'gc> {
        self.base()
            .values()
            .get(&self.object_key(name))
            .map(|v| v.value)
            .unwrap_or(Value::Undefined)
    }

    /// Set a value in the dictionary's object space.
    pub fn set_property_by_object(self, name: Object<'gc>, value: Value<'gc>, mc: &Mutation<'gc>) {
        self.base()
            .values_mut(mc)
            .insert(self.object_key(name), value);
    }

    /// Delete a value from the dictionary's object space.
    pub fn delete_property_by_object(self, name: Object<'gc>, mc: &Mutation<'gc>) {
        self.base().values_mut(mc).remove(&self.object_key(name));
    }

    pub fn has_property_by_object(self, name: Object<'gc>) -> bool {
        self.base().values().contains_key(&self.object_key(name))
    }
}

impl<'gc> TObject<'gc> for DictionaryObject<'gc> {
    fn gc_base(&self) -> Gc<'gc, ScriptObjectData<'gc>> {
        HasPrefixField::as_prefix_gc(self.0)
    }

    // Calling `setPropertyIsEnumerable` on a `Dictionary` has no effect -
    // stringified properties are always enumerable.
    fn set_local_property_is_enumerable(
        &self,
        _mc: &Mutation<'gc>,
        _name: AvmString<'gc>,
        _is_enumerable: bool,
    ) {
    }

    fn get_enumerant_value(
        self,
        index: u32,
        activation: &mut Activation<'_, 'gc>,
    ) -> Result<Value<'gc>, Error<'gc>> {
        let base = self.base();
        let values = base.values();
        if !self.has_weak_keys() {
            return Ok(values
                .value_at(index as usize)
                .copied()
                .unwrap_or(Value::Undefined));
        }
        Ok(values
            .pair_at(index as usize)
            .and_then(|(key, property)| key.as_value(activation.gc()).map(|_| property.value))
            .unwrap_or(Value::Undefined))
    }

    fn get_next_enumerant(
        self,
        last_index: u32,
        activation: &mut Activation<'_, 'gc>,
    ) -> Result<u32, Error<'gc>> {
        let base = self.base();
        let mut index = base.get_next_enumerant(last_index);
        // A sweep may have collected keys since the last trace of this dictionary.
        if self.has_weak_keys() {
            while index != 0 && base.get_enumerant_name(index, activation.gc()).is_none() {
                index = base.get_next_enumerant(index);
            }
        }
        Ok(index)
    }
}
