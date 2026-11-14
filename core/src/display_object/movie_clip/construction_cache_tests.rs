use super::*;
use crate::display_object::Graphic;
use crate::player::PlayerBuilder;
use crate::tag_utils::SwfMovieData;
use std::cell::Cell;

thread_local! {
    static VISITS: Cell<usize> = const { Cell::new(0) };
    static SCRIPT_VISITS: Cell<usize> = const { Cell::new(0) };
    static ENTER_VISITS: Cell<usize> = const { Cell::new(0) };
}

pub(super) fn record_visit() {
    VISITS.set(VISITS.get() + 1);
}

pub(super) fn record_script_visit() {
    SCRIPT_VISITS.set(SCRIPT_VISITS.get() + 1);
}

pub(super) fn record_enter_visit() {
    ENTER_VISITS.set(ENTER_VISITS.get() + 1);
}

fn with_tree(test: impl for<'gc> FnOnce(MovieClip<'gc>, MovieClip<'gc>, &mut UpdateContext<'gc>)) {
    let mut bytes = Vec::new();
    swf::write_swf(
        &swf::Header::default_with_swf_version(10),
        &[
            swf::Tag::FileAttributes(swf::FileAttributes::IS_ACTION_SCRIPT_3),
            swf::Tag::ShowFrame,
        ],
        &mut bytes,
    )
    .unwrap();
    let movie = SwfMovieData::from_data(
        &bytes,
        "file:///construction-cache-test.swf".into(),
        None,
        None,
    )
    .unwrap();
    let player = PlayerBuilder::new().with_movie(movie).build();
    let _scope = ruffle_render::profile::enter(None, true);
    player
        .lock()
        .unwrap()
        .mutate_with_update_context(|context| {
            *context.frame_phase = FramePhase::Construct;
            let movie = context.stage.root_clip().unwrap().movie();
            let root = MovieClip::new(movie, context.gc());
            let child = MovieClip::new(movie, context.gc());
            root.replace_at_depth(context, child.into(), 1);
            let leaf = Graphic::empty(context);
            child.replace_at_depth(context, leaf.into(), 1);
            // Constructors may dirty their ancestors; a second pass settles the tree.
            root.construct_frame(context);
            root.construct_frame(context);
            assert!(root.object2().is_some());
            assert!(child.child_by_index(0).unwrap().object2().is_some());
            test(root, child, context);
        });
}

#[test]
fn settled_construction_subtree_avoids_revisiting_descendants() {
    with_tree(|root, _, context| {
        VISITS.set(0);
        for _ in 0..10 {
            root.construct_frame(context);
        }
        assert_eq!(
            VISITS.get(),
            0,
            "settled static tree should perform no construction visits"
        );
    });
}

#[test]
fn settled_scriptless_subtree_avoids_frame_script_walks() {
    with_tree(|root, _, context| {
        *context.frame_phase = FramePhase::FrameScripts;
        SCRIPT_VISITS.set(0);
        for _ in 0..10 {
            root.run_frame_scripts(context);
        }
        assert_eq!(SCRIPT_VISITS.get(), 0);

        let _scope = ruffle_render::profile::enter_engine(false);
        root.run_frame_scripts(context);
        assert_eq!(SCRIPT_VISITS.get(), 2, "stock must retain the full walk");
    });
}

#[test]
fn stopped_subtree_avoids_enter_frame_walks() {
    with_tree(|root, child, context| {
        assert!(!root.playing() && !child.playing());
        *context.frame_phase = FramePhase::Enter;
        ENTER_VISITS.set(0);
        for _ in 0..10 {
            root.enter_frame(context);
        }
        assert_eq!(
            ENTER_VISITS.get(),
            0,
            "stopped settled subtrees have no tick work"
        );

        let _scope = ruffle_render::profile::enter_engine(false);
        root.enter_frame(context);
        assert_eq!(ENTER_VISITS.get(), 2, "stock must retain the full walk");
    });
}

#[test]
fn stopped_scripted_subtree_skips_ticks_but_still_runs_frame_scripts() {
    with_tree(|root, child, context| {
        child.register_frame_script(
            child.current_frame(),
            Some(context.avm2.classes().object.into()),
            context,
        );
        root.construct_frame(context);
        assert!(child.has_pending_script());
        *context.frame_phase = FramePhase::Enter;
        ENTER_VISITS.set(0);
        root.enter_frame(context);
        assert_eq!(
            ENTER_VISITS.get(),
            0,
            "stopped scripts do not require timeline ticks"
        );
        *context.frame_phase = FramePhase::FrameScripts;
        root.run_frame_scripts(context);
        assert_eq!(
            child.last_queued_script_frame(),
            Some(child.current_frame())
        );
        assert!(!child.has_pending_script());
        // The parent and child have scripts below them, so construction is
        // still dirty. play must nevertheless invalidate the idle ancestors.
        child.play();
        *context.frame_phase = FramePhase::Enter;
        ENTER_VISITS.set(0);
        root.enter_frame(context);
        assert_eq!(
            ENTER_VISITS.get(),
            2,
            "dirty construction must not stop idle invalidation"
        );
    });
}

#[test]
fn playing_descendant_wakes_stopped_ancestors() {
    with_tree(|root, child, context| {
        child.replace_with_movie(context, Some(root.movie()), false, None);
        assert!(child.preload(context, &mut ExecutionLimit::none()));
        child.stop(context);
        root.construct_frame(context);
        root.construct_frame(context);
        assert_eq!(child.current_frame(), 0);
        {
            let _scope = ruffle_render::profile::enter_engine(false);
            child.play();
        }
        root.construct_frame(context);
        *context.frame_phase = FramePhase::Enter;
        ENTER_VISITS.set(0);
        root.enter_frame(context);
        assert_eq!(ENTER_VISITS.get(), 2, "play must reach the descendant");
        assert_eq!(
            child.current_frame(),
            1,
            "the resumed timeline must advance"
        );
    });
}

#[test]
fn descendant_skip_flag_is_consumed_under_a_stopped_ancestor() {
    with_tree(|root, child, context| {
        child.base().set_skip_next_enter_frame(true);
        root.construct_frame(context);
        *context.frame_phase = FramePhase::Enter;
        root.enter_frame(context);
        assert!(!child.base().should_skip_next_enter_frame());
        assert!(
            child
                .child_by_index(0)
                .unwrap()
                .base()
                .should_skip_next_enter_frame(),
            "the existing skip propagation to inert leaves must remain observable"
        );
    });
}

#[test]
fn parent_skip_flag_still_propagates_through_stopped_subtrees() {
    with_tree(|root, child, context| {
        root.base().set_skip_next_enter_frame(true);
        *context.frame_phase = FramePhase::Enter;
        root.enter_frame(context);
        assert!(!root.base().should_skip_next_enter_frame());
        assert!(!child.base().should_skip_next_enter_frame());
        assert!(
            child
                .child_by_index(0)
                .unwrap()
                .base()
                .should_skip_next_enter_frame()
        );
    });
}

#[test]
fn reparented_playing_descendant_wakes_a_stopped_tree() {
    with_tree(|root, child, context| {
        let added = MovieClip::new(root.movie(), context.gc());
        added.replace_with_movie(context, Some(root.movie()), false, None);
        assert!(added.preload(context, &mut ExecutionLimit::none()));
        added.stop(context);
        added.construct_frame(context);
        added.construct_frame(context);
        added.play();
        child.replace_at_depth(context, added.into(), 2);
        root.construct_frame(context);
        *context.frame_phase = FramePhase::Enter;
        root.enter_frame(context);
        assert_eq!(
            added.current_frame(),
            1,
            "newly attached active children must still tick"
        );
    });
}

#[test]
fn stopping_an_active_descendant_makes_the_tree_idle_again() {
    with_tree(|root, child, context| {
        child.play();
        root.construct_frame(context);
        child.stop(context);
        root.construct_frame(context);
        *context.frame_phase = FramePhase::Enter;
        ENTER_VISITS.set(0);
        root.enter_frame(context);
        assert_eq!(ENTER_VISITS.get(), 0);
    });
}

#[test]
fn replacing_a_cached_movie_preserves_the_new_timeline_tick() {
    with_tree(|root, child, context| {
        child.replace_with_movie(context, Some(root.movie()), false, None);
        assert!(child.preload(context, &mut ExecutionLimit::none()));
        root.construct_frame(context);
        *context.frame_phase = FramePhase::Enter;
        root.enter_frame(context);
        assert_eq!(child.current_frame(), 1);
    });
}

#[test]
#[ignore = "diagnostic stopped-subtree traversal benchmark; no FPS assertion"]
fn stopped_enter_frame_benchmark() {
    use std::hint::black_box;
    use std::time::Instant;
    with_tree(|root, _, context| {
        for index in 2..102 {
            let branch = MovieClip::new(root.movie(), context.gc());
            let mut parent = branch;
            for _ in 0..10 {
                let child = MovieClip::new(root.movie(), context.gc());
                parent.replace_at_depth(context, child.into(), 1);
                parent = child;
            }
            root.replace_at_depth(context, branch.into(), index);
        }
        root.construct_frame(context);
        root.construct_frame(context);
        *context.frame_phase = FramePhase::Enter;
        let mut samples = [Vec::new(), Vec::new()];
        for sample in 0..31 {
            for offset in 0..2 {
                let mode = (sample + offset) % 2;
                let _scope = ruffle_render::profile::enter_engine(mode == 1);
                ENTER_VISITS.set(0);
                let start = Instant::now();
                for _ in 0..100 {
                    black_box(root).enter_frame(context);
                }
                samples[mode].push(start.elapsed().as_nanos() / 100);
                assert_eq!(ENTER_VISITS.get(), if mode == 1 { 0 } else { 110_200 });
            }
        }
        for sample in &mut samples {
            sample.sort_unstable();
        }
        println!(
            "stock_ns_p50={} stock_ns_p95={} optimized_ns_p50={} optimized_ns_p95={}",
            samples[0][15], samples[0][29], samples[1][15], samples[1][29]
        );
    });
}

#[test]
fn queued_goto_survives_construction_before_frame_scripts() {
    with_tree(|root, child, context| {
        // Exercise the enqueue path used while an AS3 frame script is active.
        child.play();
        child
            .0
            .set_flag(MovieClipFlags::EXECUTING_AVM2_FRAME_SCRIPT, true);
        child.goto_frame(
            context,
            GotoInfo {
                frame: child.current_frame(),
                stop_or_play: StopOrPlay::Stop,
            },
        );
        child
            .0
            .set_flag(MovieClipFlags::EXECUTING_AVM2_FRAME_SCRIPT, false);
        assert!(child.playing(), "goto must remain deferred");
        root.construct_frame(context);
        *context.frame_phase = FramePhase::FrameScripts;
        root.run_frame_scripts(context);
        assert!(!child.playing(), "the queued goto must execute");
        assert!(child.0.queued_goto.get().is_none());
    });
}

#[test]
fn nested_insertion_constructs_new_child_after_tree_settles() {
    with_tree(|root, child, context| {
        let leaf = Graphic::empty(context);
        assert!(leaf.object2().is_none());
        child.replace_at_depth(context, leaf.into(), 2);
        root.construct_frame(context);
        assert!(
            leaf.object2().is_some(),
            "a new grandchild must invalidate the root"
        );
    });
}

#[test]
fn loader_style_mutation_outside_optimized_scope_invalidates_cached_tree() {
    with_tree(|root, child, context| {
        let leaf = Graphic::empty(context);
        {
            let _scope = ruffle_render::profile::enter_engine(false);
            child.replace_at_depth(context, leaf.into(), 2);
        }
        root.construct_frame(context);
        assert!(leaf.object2().is_some());
    });
}

#[test]
fn stock_traversal_does_not_use_an_existing_clean_cache() {
    with_tree(|root, _, context| {
        let _scope = ruffle_render::profile::enter_engine(false);
        VISITS.set(0);
        root.construct_frame(context);
        root.construct_frame(context);
        assert_eq!(VISITS.get(), 4);
    });
}

#[test]
fn new_frame_script_is_queued_after_static_tree_was_cached() {
    with_tree(|root, child, context| {
        // Classes are callable in AVM2; Object() provides a real no-op script.
        let callable = context.avm2.classes().object.into();
        child.register_frame_script(child.current_frame(), Some(callable), context);
        child.set_has_pending_script(false);
        root.construct_frame(context);
        assert!(child.has_pending_script());
        *context.frame_phase = FramePhase::FrameScripts;
        root.run_frame_scripts(context);
        assert_eq!(
            child.last_queued_script_frame(),
            Some(child.current_frame())
        );
        assert!(!child.has_pending_script());
    });
}

#[test]
fn dirty_subtree_reparenting_invalidates_previously_settled_parent() {
    with_tree(|mut old_root, child, context| {
        let new_root = MovieClip::new(old_root.movie(), context.gc());
        new_root.construct_frame(context);
        new_root.construct_frame(context);
        let leaf = Graphic::empty(context);
        child.replace_at_depth(context, leaf.into(), 2);
        old_root.remove_child(context, child.into());
        new_root.replace_at_depth(context, child.into(), 1);
        new_root.construct_frame(context);
        assert!(leaf.object2().is_some());
        assert!(old_root.is_empty());
        assert_eq!(child.parent().unwrap().as_ptr(), new_root.as_ptr());
    });
}

#[test]
fn reparented_pending_script_runs_under_a_previously_settled_parent() {
    with_tree(|mut root, child, context| {
        let new_root = MovieClip::new(root.movie(), context.gc());
        new_root.construct_frame(context);
        new_root.construct_frame(context);
        let callable = context.avm2.classes().object.into();
        child.register_frame_script(child.current_frame(), Some(callable), context);
        root.remove_child(context, child.into());
        new_root.replace_at_depth(context, child.into(), 1);
        *context.frame_phase = FramePhase::FrameScripts;
        new_root.run_frame_scripts(context);
        assert_eq!(
            child.last_queued_script_frame(),
            Some(child.current_frame())
        );
        assert!(!child.has_pending_script());
    });
}

#[test]
#[ignore = "diagnostic frame-script traversal benchmark; no FPS assertion"]
fn settled_frame_scripts_benchmark() {
    use std::hint::black_box;
    use std::time::Instant;
    with_tree(|root, _, context| {
        for index in 2..102 {
            let branch = MovieClip::new(root.movie(), context.gc());
            let mut parent = branch;
            for _ in 0..10 {
                let child = MovieClip::new(root.movie(), context.gc());
                parent.replace_at_depth(context, child.into(), 1);
                parent = child;
            }
            root.replace_at_depth(context, branch.into(), index);
        }
        root.construct_frame(context);
        root.construct_frame(context);
        *context.frame_phase = FramePhase::FrameScripts;
        let mut samples = [Vec::new(), Vec::new()];
        for sample in 0..31 {
            for offset in 0..2 {
                let mode = (sample + offset) % 2;
                let _scope = ruffle_render::profile::enter_engine(mode == 1);
                SCRIPT_VISITS.set(0);
                let start = Instant::now();
                for _ in 0..100 {
                    black_box(root).run_frame_scripts(context);
                }
                samples[mode].push(start.elapsed().as_nanos() / 100);
                assert_eq!(SCRIPT_VISITS.get(), if mode == 1 { 0 } else { 110_200 });
            }
        }
        for sample in &mut samples {
            sample.sort_unstable();
        }
        println!(
            "stock_ns_p50={} stock_ns_p95={} optimized_ns_p50={} optimized_ns_p95={}",
            samples[0][15], samples[0][29], samples[1][15], samples[1][29]
        );
    });
}

#[test]
#[ignore = "diagnostic construction benchmark; no machine-dependent FPS assertion"]
fn settled_construction_benchmark() {
    use std::hint::black_box;
    use std::time::Instant;
    with_tree(|root, _, context| {
        for index in 2..102 {
            let branch = MovieClip::new(root.movie(), context.gc());
            for depth in 1..11 {
                let leaf = Graphic::empty(context);
                branch.replace_at_depth(context, leaf.into(), depth);
            }
            root.replace_at_depth(context, branch.into(), index);
        }
        root.construct_frame(context);
        root.construct_frame(context);
        let mut stock = Vec::new();
        let mut optimized = Vec::new();
        for sample in 0..31 {
            for offset in 0..2 {
                let enabled = (sample + offset) % 2 == 0;
                let _scope = ruffle_render::profile::enter_engine(enabled);
                VISITS.set(0);
                let start = Instant::now();
                for _ in 0..100 {
                    black_box(root).construct_frame(context);
                }
                let nanos = start.elapsed().as_nanos() / 100;
                if enabled {
                    assert_eq!(VISITS.get(), 0);
                    optimized.push(nanos);
                } else {
                    assert_eq!(VISITS.get(), 10200);
                    stock.push(nanos);
                }
            }
        }
        stock.sort_unstable();
        optimized.sort_unstable();
        println!(
            "stock_ns_p50={} stock_ns_p95={} optimized_ns_p50={} optimized_ns_p95={}",
            stock[15], stock[29], optimized[15], optimized[29]
        );
    });
}

#[cfg(feature = "scripted_construction_cache")]
fn settle_scripted_tree<'gc>(
    root: MovieClip<'gc>,
    child: MovieClip<'gc>,
    context: &mut UpdateContext<'gc>,
) {
    child.register_frame_script(
        child.current_frame(),
        Some(context.avm2.classes().object.into()),
        context,
    );
    for _ in 0..3 {
        *context.frame_phase = FramePhase::Construct;
        root.construct_frame(context);
        *context.frame_phase = FramePhase::FrameScripts;
        root.run_frame_scripts(context);
    }
    assert_eq!(
        child.last_queued_script_frame(),
        Some(child.current_frame())
    );
}

#[test]
#[cfg(feature = "scripted_construction_cache")]
fn consumed_scripted_subtree_avoids_repeated_nested_frame_walks() {
    with_tree(|root, child, context| {
        settle_scripted_tree(root, child, context);
        VISITS.set(0);
        SCRIPT_VISITS.set(0);
        for _ in 0..10 {
            *context.frame_phase = FramePhase::Construct;
            root.construct_frame(context);
            *context.frame_phase = FramePhase::FrameScripts;
            root.run_frame_scripts(context);
        }
        assert_eq!(
            VISITS.get(),
            0,
            "consumed scripts must not force repeated construction"
        );
        assert_eq!(
            SCRIPT_VISITS.get(),
            0,
            "already consumed scripts have no remaining work"
        );
        let _scope = ruffle_render::profile::enter_engine(false);
        *context.frame_phase = FramePhase::Construct;
        root.construct_frame(context);
        *context.frame_phase = FramePhase::FrameScripts;
        root.run_frame_scripts(context);
        assert_eq!(VISITS.get(), 2, "stock still constructs both clips");
        assert_eq!(
            SCRIPT_VISITS.get(),
            2,
            "stock still visits both script nodes"
        );
    });
}

#[test]
#[cfg(feature = "scripted_construction_cache")]
fn rearming_consumed_script_invalidates_ancestors_without_a_new_registration() {
    with_tree(|root, child, context| {
        settle_scripted_tree(root, child, context);
        assert!(child.has_pending_script());
        child.set_last_queued_script_frame(None);
        SCRIPT_VISITS.set(0);
        root.run_frame_scripts(context);
        assert_eq!(
            child.last_queued_script_frame(),
            Some(child.current_frame()),
            "rearmed script was hidden by an idle ancestor"
        );
        assert!(!child.has_pending_script());
        assert_eq!(SCRIPT_VISITS.get(), 2);
    });
}

#[test]
#[cfg(feature = "scripted_construction_cache")]
fn scripted_tree_constructed_outside_construct_still_checks_pending_scripts() {
    with_tree(|root, child, context| {
        child.register_frame_script(
            child.current_frame(),
            Some(context.avm2.classes().object.into()),
            context,
        );
        child.set_has_pending_script(false);
        *context.frame_phase = FramePhase::FrameScripts;
        for _ in 0..3 {
            root.construct_frame(context);
            root.run_frame_scripts(context);
        }
        assert!(!root.base().construction_clean());
        assert_eq!(child.last_queued_script_frame(), None);
        *context.frame_phase = FramePhase::Construct;
        root.construct_frame(context);
        assert!(child.has_pending_script());
        *context.frame_phase = FramePhase::FrameScripts;
        root.run_frame_scripts(context);
        assert_eq!(
            child.last_queued_script_frame(),
            Some(child.current_frame())
        );
    });
}

#[test]
#[ignore = "diagnostic settled scripted-subtree benchmark; no FPS assertion"]
#[cfg(feature = "scripted_construction_cache")]
fn settled_scripted_subtree_benchmark() {
    use std::hint::black_box;
    use std::time::Instant;

    with_tree(|root, child, context| {
        for depth in 2..102 {
            let branch = MovieClip::new(root.movie(), context.gc());
            root.replace_at_depth(context, branch.into(), depth);
            root.construct_frame(context);
            branch.register_frame_script(
                branch.current_frame(),
                Some(context.avm2.classes().object.into()),
                context,
            );
        }
        settle_scripted_tree(root, child, context);
        let mut stock = Vec::new();
        let mut optimized = Vec::new();
        for sample in 0..31 {
            for offset in 0..2 {
                let enabled = (sample + offset) % 2 == 0;
                let _scope = ruffle_render::profile::enter_engine(enabled);
                VISITS.set(0);
                SCRIPT_VISITS.set(0);
                let start = Instant::now();
                for _ in 0..100 {
                    *context.frame_phase = FramePhase::Construct;
                    black_box(root).construct_frame(context);
                    *context.frame_phase = FramePhase::FrameScripts;
                    black_box(root).run_frame_scripts(context);
                }
                let nanos = start.elapsed().as_nanos() / 100;
                if enabled {
                    assert_eq!(VISITS.get(), 0);
                    assert_eq!(SCRIPT_VISITS.get(), 0);
                    optimized.push(nanos);
                } else {
                    assert_eq!(VISITS.get(), 10200);
                    assert_eq!(SCRIPT_VISITS.get(), 10200);
                    stock.push(nanos);
                }
            }
        }
        stock.sort_unstable();
        optimized.sort_unstable();
        println!(
            "stock_ns_p50={} stock_ns_p95={} optimized_ns_p50={} optimized_ns_p95={}",
            stock[15], stock[29], optimized[15], optimized[29]
        );
    });
}
