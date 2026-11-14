//! A narrow execution summary for already-verified fixed-slot getters.
//! It never stores a returned value: every invocation reads the actual receiver.

use super::activation::Activation;
use super::error::Error;
use super::function::FunctionArgs;
use super::method::{Method, MethodKind};
use super::object::TObject;
use super::op::Op;
use super::traits::TraitKind;
use super::value::Value;

pub(super) fn classify(method: Method<'_>, code: &[Op<'_>], no_exceptions: bool) -> Option<usize> {
    let [
        Op::GetLocal { index: 0 },
        Op::GetSlot { index },
        Op::ReturnValue { return_type: None },
    ] = code
    else {
        return None;
    };
    if !no_exceptions
        || !method.signature().is_empty()
        || method.is_variadic()
        || method.sets_dxns()
        || method.is_interpreted()
    {
        return None;
    }
    // Restrict this experiment to actual getters. A matching body on a method
    // or free function keeps its existing execution path.
    if !method.bound_class()?.traits().iter().any(|entry| {
        matches!(entry.kind(), TraitKind::Getter { method: getter, .. } if *getter == method)
    }) {
        return None;
    }
    #[cfg(feature = "lifecycle_counts")]
    ruffle_render::profile::lifecycle::add(
        ruffle_render::profile::lifecycle::Count::FastAccessorCandidate,
        1,
    );
    Some(*index)
}

pub(super) fn try_exec<'gc>(
    method: Method<'gc>,
    receiver: Value<'gc>,
    arguments: FunctionArgs<'_, 'gc>,
    activation: &mut Activation<'_, 'gc>,
) -> Option<Result<Value<'gc>, Error<'gc>>> {
    if cfg!(feature = "avm_debug")
        || !ruffle_render::profile::engine_optimized()
        || !arguments.is_empty()
    {
        return None;
    }
    let MethodKind::Bytecode { verified_info } = method.method_kind() else {
        return None;
    };
    // The first invocation must perform normal resolution and verification,
    // including their errors and initialization effects.
    let slot = verified_info.get()?.fast_accessor?;
    let object = receiver.as_object()?;

    // Reserve exactly the original frame size before the receiver assertion,
    // preserving the stack bounds check and clearing previously held values.
    let stack = activation.context.avm2.stack;
    let frame = stack.get_frame_for_method(method);
    if let Some(class) = method.bound_class() {
        assert!(receiver.is_of_type(class));
    }

    #[cfg(feature = "tracy_avm")]
    let _span = {
        let mut name = crate::string::WString::new();
        super::function::display_function(&mut name, method);
        let span = tracy_client::Client::running()
            .expect("tracy_client should be running")
            .span_alloc(
                None,
                &name.to_utf8_lossy(),
                method.owner_movie().url(),
                line!(),
                0,
            );
        span.emit_color(0x425fa1);
        span
    };

    let mc = activation.gc();
    activation.context.avm2.push_call(mc, method);
    // A straight-line interpreted body performs one timeout check at entry.
    // Keep that check even on an error and balance both stacks before returning.
    let result = activation.timeout_check().map(|()| object.get_slot(slot));
    stack.dispose_stack_frame(frame);
    activation.context.avm2.pop_call(mc);
    #[cfg(feature = "lifecycle_counts")]
    ruffle_render::profile::lifecycle::add(
        ruffle_render::profile::lifecycle::Count::FastAccessorHit,
        1,
    );
    Some(result)
}
