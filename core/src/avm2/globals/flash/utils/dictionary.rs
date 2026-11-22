use crate::avm2::function::FunctionArgs;
use crate::avm2::{Activation, Error, Value};

pub use crate::avm2::object::dictionary_allocator;

pub fn use_weak_keys<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: FunctionArgs<'_, 'gc>,
) -> Result<Value<'gc>, Error<'gc>> {
    let dictionary = this.as_object().unwrap().as_dictionary_object().unwrap();
    dictionary.use_weak_keys();
    Ok(Value::Undefined)
}
