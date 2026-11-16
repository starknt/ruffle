use crate::PlayerBuilder;
use crate::avm2::{Activation, Value};
use crate::backend::log::LogBackend;
use crate::display_object::TDisplayObject;
use crate::limits::ExecutionLimit;
use crate::string::{AvmString, WString};
use crate::tag_utils::SwfMovieData;
use ruffle_render::profile::lifecycle::{Count, Phase};
use ruffle_render::profile::{RenderMetrics, enter};
use std::cell::RefCell;
use std::rc::Rc;

struct Trace(Rc<RefCell<Vec<String>>>);

impl LogBackend for Trace {
    fn avm_trace(&self, message: &str) {
        self.0.borrow_mut().push(message.to_owned());
    }

    fn avm_warning(&self, _message: &str) {}
}

#[test]
fn verified_accessors_read_current_slots_and_preserve_dispatch_and_errors() {
    let metrics = RenderMetrics::default();
    let _profile = enter(Some(&metrics));
    let traces = Rc::new(RefCell::new(Vec::new()));
    let movie = SwfMovieData::from_data(
        include_bytes!("../../../tests/tests/swfs/avm2/fast_accessor/test.swf"),
        "file:///fast-accessor-test.swf".into(),
        None,
        None,
    )
    .unwrap();
    let owner = PlayerBuilder::new()
        .with_movie(movie)
        .with_log(Trace(traces.clone()))
        .with_autoplay(true)
        .build();
    let mut player = owner.lock().unwrap();
    assert!(player.preload(&mut ExecutionLimit::none()));
    player.run_frame();
    assert_eq!(
        traces.borrow().join("\n"),
        include_str!("../../../tests/tests/swfs/avm2/fast_accessor/output.txt").trim_end()
    );
    let hits =
        metrics.snapshot().lifecycle.values[Phase::Frame as usize][Count::FastAccessorHit as usize];
    if !cfg!(feature = "avm_debug") {
        assert!(
            hits >= 3,
            "verified getter repeats must use the fast path, got {hits}"
        );
    } else {
        assert_eq!(hits, 0, "stock/debug must preserve the interpreter path");
    }
    player.mutate_with_update_context(|context| {
        let root: Value<'_> = context.stage.root_clip().unwrap().object2().unwrap().into();
        let probe_name = AvmString::new_utf8(context.gc(), "probe");
        let value_name = AvmString::new_utf8(context.gc(), "freshValue");
        let field_name = AvmString::new_utf8(context.gc(), "field");
        let mut activation = Activation::from_nothing(context);
        let probe = root
            .get_public_property(probe_name, &mut activation)
            .unwrap();
        let hits = || {
            metrics.snapshot().lifecycle.values[Phase::Frame as usize]
                [Count::FastAccessorHit as usize]
        };
        let before = hits();
        assert!(matches!(
            probe
                .get_public_property(value_name, &mut activation)
                .unwrap(),
            Value::Integer(19)
        ));
        assert_eq!(
            hits(),
            before,
            "cold getter must use ordinary verification/execution"
        );
        probe
            .set_public_property(field_name, Value::Integer(37), &mut activation)
            .unwrap();
        assert!(matches!(
            probe
                .get_public_property(value_name, &mut activation)
                .unwrap(),
            Value::Integer(37)
        ));
        let enabled = !cfg!(feature = "avm_debug");
        assert_eq!(
            hits() - before,
            u64::from(enabled),
            "only this warmed getter counts"
        );

        let mut before_stack = WString::new();
        activation
            .context
            .avm2
            .capture_call_stack()
            .display(&mut before_stack);
        let scope_depth = activation.context.avm2.scope_stack.len();
        let max_duration = activation.context.max_execution_duration;
        activation.context.max_execution_duration = std::time::Duration::ZERO;
        *activation.context.actions_since_timeout_check = 9999;
        assert!(
            probe
                .get_public_property(value_name, &mut activation)
                .is_err()
        );
        activation.context.max_execution_duration = max_duration;
        let mut after_stack = WString::new();
        activation
            .context
            .avm2
            .capture_call_stack()
            .display(&mut after_stack);
        assert_eq!(
            after_stack, before_stack,
            "timeout must balance the call stack"
        );
        assert_eq!(activation.context.avm2.scope_stack.len(), scope_depth);
        assert!(matches!(
            probe
                .get_public_property(value_name, &mut activation)
                .unwrap(),
            Value::Integer(37)
        ));
    });
}
