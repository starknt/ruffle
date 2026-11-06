//! `flash.utils.Timer` native methods

use crate::avm2::Error;
use crate::avm2::activation::Activation;
use crate::avm2::function::FunctionArgs;
use crate::avm2::globals::slots::flash_utils_timer as slots;
use crate::avm2::object::TObject;
use crate::avm2::value::Value;
use crate::timer::TimerCallback;

/// Implements `Timer.stop`
pub fn stop<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: FunctionArgs<'_, 'gc>,
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    let id = this.get_slot(slots::_TIMER_ID).as_i32();

    if id != -1 {
        activation.context.timers.remove(id);
        this.set_slot(slots::_TIMER_ID, (-1).into(), activation)?;
    }

    Ok(Value::Undefined)
}

/// Implements `Timer.start`
pub fn start<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: FunctionArgs<'_, 'gc>,
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    let id = this.get_slot(slots::_TIMER_ID).as_i32();

    let delay = this.get_slot(slots::_DELAY).as_f64() as i32;

    if id == -1 {
        let on_update = this
            .get_slot(slots::_ON_UPDATE_CLOSURE)
            .as_object()
            .and_then(|o| o.as_function_object())
            .expect("Internal function is function");

        // Note - we deliberately do *not* check if currentCount is less than repeatCount.
        // Calling 'start' on a timer that has currentCount >= repeatCount will tick exactly
        // once, and then stop immediately. This is handled by Timer.onUpdate
        let mut wall = activation.context.in_wall_clock_method();
        if !activation.context.playback.wall_classes.is_empty() {
            let list = crate::avm2::globals::flash::events::event_dispatcher::dispatch_list(
                activation, this,
            )?;
            let (wall_listener, game_listener) = list
                .as_dispatch_mut(activation.gc())
                .expect("event dispatcher")
                .timer_clocks();
            wall |= wall_listener;
            if wall && game_listener {
                activation.context.playback.mixed_timer.set(true);
            }
        }
        let id = activation.context.timers.add_for_clock(
            TimerCallback::Avm2Callback {
                closure: Some(on_update),
                params: vec![],
            },
            delay,
            false,
            wall,
        );
        this.set_slot(slots::_TIMER_ID, id.into(), activation)?;
    }
    Ok(Value::Undefined)
}

/// Implements `Timer.updateDelay`
pub fn update_delay<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: FunctionArgs<'_, 'gc>,
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    let id = this.get_slot(slots::_TIMER_ID).as_i32();

    let delay = this.get_slot(slots::_DELAY).as_f64() as i32;

    if id != -1 {
        activation.context.timers.set_delay(id, delay);
    }
    Ok(Value::Undefined)
}
