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
    let _scope = ruffle_render::profile::enter(None);
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
fn settled_stage_avoids_revisiting_root_children() {
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
        "file:///construction-cache-stage-test.swf".into(),
        None,
        None,
    )
    .unwrap();
    let player = PlayerBuilder::new().with_movie(movie).build();
    let _scope = ruffle_render::profile::enter(None);
    player
        .lock()
        .unwrap()
        .mutate_with_update_context(|context| {
            *context.frame_phase = FramePhase::Construct;
            context.stage.construct_frame(context);
            context.stage.construct_frame(context);
            assert!(
                context.stage.base().construction_clean(),
                "stage should settle after its initial constructions"
            );
            VISITS.set(0);
            context.stage.construct_frame(context);
            assert_eq!(
                VISITS.get(),
                0,
                "a settled stage should not revisit its root MovieClip"
            );
            let root = context.stage.root_clip().unwrap();
            root.invalidate_construction();
            context.stage.construct_frame(context);
            assert_eq!(
                VISITS.get(),
                1,
                "stage invalidation should restore the root construction visit"
            );
        });
}

#[test]
fn settled_stage_avoids_revisiting_frame_scripts() {
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
        "file:///construction-cache-stage-script-test.swf".into(),
        None,
        None,
    )
    .unwrap();
    let player = PlayerBuilder::new().with_movie(movie).build();
    let _scope = ruffle_render::profile::enter(None);
    player
        .lock()
        .unwrap()
        .mutate_with_update_context(|context| {
            *context.frame_phase = FramePhase::Construct;
            context.stage.construct_frame(context);
            context.stage.construct_frame(context);
            *context.frame_phase = FramePhase::FrameScripts;
            context.stage.run_frame_scripts(context);
            assert!(
                context.stage.base().frame_scripts_idle(),
                "stage should record an idle frame-script proof"
            );
            SCRIPT_VISITS.set(0);
            context.stage.run_frame_scripts(context);
            assert_eq!(
                SCRIPT_VISITS.get(),
                0,
                "an idle stage should not revisit frame-script children"
            );
            let root = context.stage.root_clip().unwrap();
            root.invalidate_construction();
            context.stage.run_frame_scripts(context);
            assert_eq!(
                SCRIPT_VISITS.get(),
                1,
                "stage invalidation should restore frame-script traversal"
            );
        });
}

#[test]
fn settled_orphan_worklist_avoids_even_root_callbacks() {
    use crate::orphan_manager::OrphanManager;
    with_tree(|root, _, context| {
        *context.orphan_manager = OrphanManager::default();
        context.orphan_manager.add_orphan_obj(root.into());
        let mut calls = 0;
        for phase in [FramePhase::Construct, FramePhase::FrameScripts] {
            *context.frame_phase = phase;
            OrphanManager::each_orphan_obj(context, |_, _| calls += 1);
        }
        assert_eq!(
            calls, 0,
            "settled roots must not reach the lifecycle callback"
        );
    });
}

#[test]
fn observing_an_orphan_does_not_make_it_a_hidden_button_state() {
    with_tree(|root, _, _| {
        root.base().observe_orphan_worklist();
        let revision = crate::display_object::button_state_revision();
        root.invalidate_construction();
        assert_eq!(crate::display_object::button_state_revision(), revision);
    });
}

#[test]
fn orphan_worklist_wakes_later_members_without_repeating_earlier_members() {
    use crate::orphan_manager::OrphanManager;
    with_tree(|root, _, context| {
        let later = MovieClip::new(root.movie(), context.gc());
        later.construct_frame(context);
        later.construct_frame(context);
        *context.orphan_manager = OrphanManager::default();
        context.orphan_manager.add_orphan_obj(root.into());
        context.orphan_manager.add_orphan_obj(later.into());
        OrphanManager::each_orphan_obj(context, |_, _| panic!("settled fixture"));
        root.invalidate_construction();
        let mut order = Vec::new();
        OrphanManager::each_orphan_obj(context, |dobj, context| {
            order.push(dobj.as_ptr());
            dobj.construct_frame(context);
            if dobj.as_ptr() == root.as_ptr() {
                later.invalidate_construction();
            } else {
                root.invalidate_construction();
            }
        });
        assert_eq!(order, [root.as_ptr(), later.as_ptr()]);
        let mut next = Vec::new();
        OrphanManager::each_orphan_obj(context, |dobj, context| {
            next.push(dobj.as_ptr());
            dobj.construct_frame(context);
        });
        assert_eq!(next, [root.as_ptr()]);
    });
}

#[test]
fn orphan_worklist_additions_wait_for_the_next_snapshot() {
    use crate::orphan_manager::OrphanManager;
    with_tree(|root, _, context| {
        let added = MovieClip::new(root.movie(), context.gc());
        *context.orphan_manager = OrphanManager::default();
        context.orphan_manager.add_orphan_obj(root.into());
        root.invalidate_construction();
        let mut order = Vec::new();
        OrphanManager::each_orphan_obj(context, |dobj, context| {
            order.push(dobj.as_ptr());
            dobj.construct_frame(context);
            context.orphan_manager.add_orphan_obj(added.into());
        });
        assert_eq!(order, [root.as_ptr()]);
        let mut next = Vec::new();
        OrphanManager::each_orphan_obj(context, |dobj, context| {
            next.push(dobj.as_ptr());
            dobj.construct_frame(context);
        });
        assert_eq!(next, [added.as_ptr()]);
    });
}

#[test]
fn nested_orphan_cleanup_does_not_remove_members_from_an_outer_snapshot() {
    use crate::orphan_manager::OrphanManager;
    with_tree(|root, _, context| {
        let later = MovieClip::new(root.movie(), context.gc());
        later.construct_frame(context);
        later.construct_frame(context);
        root.base().set_placed_by_avm2_script(true);
        *context.orphan_manager = OrphanManager::default();
        context.orphan_manager.add_orphan_obj(root.into());
        context.orphan_manager.add_orphan_obj(later.into());
        OrphanManager::each_orphan_obj(context, |_, _| panic!("settled fixture"));
        root.invalidate_construction();
        let mut order = Vec::new();
        OrphanManager::each_orphan_obj(context, |dobj, context| {
            order.push(dobj.as_ptr());
            dobj.construct_frame(context);
            if dobj.as_ptr() == root.as_ptr() {
                later.set_parent(context, Some(root.into()));
                context.orphan_manager.cleanup_dead_orphans(context.gc());
                OrphanManager::each_orphan_obj(context, |nested, context| {
                    assert_eq!(nested.as_ptr(), root.as_ptr());
                    nested.construct_frame(context);
                });
                later.set_parent(context, None);
                later.invalidate_construction();
            }
        });
        assert_eq!(order, [root.as_ptr(), later.as_ptr()]);
    });
}

#[test]
fn orphan_worklist_hidden_state_and_reload_wake_parked_roots() {
    use crate::orphan_manager::OrphanManager;
    with_button_tree(|root, hidden, _, context| {
        *context.orphan_manager = OrphanManager::default();
        context.orphan_manager.add_orphan_obj(root.into());
        *context.frame_phase = FramePhase::Construct;
        OrphanManager::each_orphan_obj(context, |_, _| panic!("settled fixture"));
        hidden.replace_with_movie(context, Some(root.movie()), false, None);
        let mut calls = 0;
        OrphanManager::each_orphan_obj(context, |dobj, context| {
            calls += 1;
            dobj.construct_frame(context);
        });
        assert_eq!(
            calls, 1,
            "hidden state replacement must invalidate the proof"
        );
        root.replace_with_movie(context, Some(root.movie()), false, None);
        root.base().set_construction_clean(true);
        OrphanManager::each_orphan_obj(context, |_, _| panic!("settled fixture"));
        root.invalidate_construction();
        let mut reloaded = 0;
        OrphanManager::each_orphan_obj(context, |_, _| reloaded += 1);
        assert_eq!(reloaded, 1, "movie load must retain the worklist observer");
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
        child.play();
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
        let mut samples = Vec::new();
        for _ in 0..62 {
            ENTER_VISITS.set(0);
            let start = Instant::now();
            for _ in 0..100 {
                black_box(root).enter_frame(context);
            }
            samples.push(start.elapsed().as_nanos() / 100);
            assert_eq!(ENTER_VISITS.get(), 0);
        }
        samples.sort_unstable();
        println!(
            "optimized_ns_p50={} optimized_ns_p95={}",
            samples[31], samples[58]
        );
    });
}

#[test]
fn empty_clip_can_seek_before_and_after_its_first_frame() {
    with_tree(|_, child, context| {
        assert_eq!(child.current_frame(), 0);
        for frame in [1, 2] {
            child.goto_frame(
                context,
                GotoInfo {
                    frame,
                    stop_or_play: StopOrPlay::Stop,
                },
            );
            assert_eq!(child.current_frame(), 1);
        }
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
        child.replace_at_depth(context, leaf.into(), 2);
        root.construct_frame(context);
        assert!(leaf.object2().is_some());
    });
}

#[test]
fn clean_cache_is_reused_by_the_default_traversal() {
    with_tree(|root, _, context| {
        VISITS.set(0);
        root.construct_frame(context);
        root.construct_frame(context);
        assert_eq!(VISITS.get(), 0);
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
        let mut samples = Vec::new();
        for _ in 0..62 {
            SCRIPT_VISITS.set(0);
            let start = Instant::now();
            for _ in 0..100 {
                black_box(root).run_frame_scripts(context);
            }
            samples.push(start.elapsed().as_nanos() / 100);
            assert_eq!(SCRIPT_VISITS.get(), 0);
        }
        samples.sort_unstable();
        println!(
            "optimized_ns_p50={} optimized_ns_p95={}",
            samples[31], samples[58]
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
        let mut optimized = Vec::new();
        for _ in 0..62 {
            VISITS.set(0);
            let start = Instant::now();
            for _ in 0..100 {
                black_box(root).construct_frame(context);
            }
            let nanos = start.elapsed().as_nanos() / 100;
            assert_eq!(VISITS.get(), 0);
            optimized.push(nanos);
        }
        optimized.sort_unstable();
        println!(
            "optimized_ns_p50={} optimized_ns_p95={}",
            optimized[31], optimized[58]
        );
    });
}

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
    });
}

#[test]
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
        let mut optimized = Vec::new();
        for _ in 0..62 {
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
            assert_eq!(VISITS.get(), 0);
            assert_eq!(SCRIPT_VISITS.get(), 0);
            optimized.push(nanos);
        }
        optimized.sort_unstable();
        println!(
            "optimized_ns_p50={} optimized_ns_p95={}",
            optimized[31], optimized[58]
        );
    });
}

#[test]
#[cfg(feature = "lifecycle_counts")]
fn lifecycle_counts_distinguish_cache_hits_from_pending_script_work() {
    use ruffle_render::profile::lifecycle::{Count, Phase, scope};
    use ruffle_render::profile::{RenderMetrics, enter};
    with_tree(|root, child, context| {
        let metrics = RenderMetrics::default();
        let _player = enter(Some(&metrics));
        {
            let _phase = scope(Phase::OrphanConstruct);
            root.construct_frame(context);
            child.register_frame_script(
                child.current_frame(),
                Some(context.avm2.classes().object.into()),
                context,
            );
            root.construct_frame(context);
        }
        assert!(child.has_pending_script());
        {
            let _phase = scope(Phase::OrphanScripts);
            *context.frame_phase = FramePhase::FrameScripts;
            root.run_frame_scripts(context);
        }
        assert_eq!(
            child.last_queued_script_frame(),
            Some(child.current_frame())
        );
        let counts = metrics.snapshot().lifecycle;
        let construct = counts.values[Phase::OrphanConstruct as usize];
        assert_eq!(construct[Count::ConstructVisit as usize], 3);
        assert_eq!(construct[Count::ConstructHit as usize], 1);
        assert_eq!(construct[Count::RegisteredScripts as usize], 1);
        let scripts = counts.values[Phase::OrphanScripts as usize];
        assert_eq!(scripts[Count::ScriptsVisit as usize], 2);
        assert_eq!(scripts[Count::ScriptsHit as usize], 0);
        assert_eq!(scripts[Count::ScriptsPending as usize], 0);
    });
}

fn with_button_tree(
    test: impl for<'gc> FnOnce(
        MovieClip<'gc>,
        MovieClip<'gc>,
        crate::display_object::Avm2Button<'gc>,
        &mut UpdateContext<'gc>,
    ),
) {
    with_tree(|root, up, context| {
        let hidden = MovieClip::new(root.movie(), context.gc());
        hidden.construct_frame(context);
        hidden.construct_frame(context);
        let button = crate::display_object::Avm2Button::empty_button(context);
        let object = Avm2StageObject::for_display_object(
            context.gc(),
            button.into(),
            context.avm2.classes().simplebutton,
        );
        button.set_object2(context, object);
        button.set_state_child(context, swf::ButtonState::UP, Some(up.into()));
        button.set_state_child(context, swf::ButtonState::OVER, Some(hidden.into()));
        root.replace_at_depth(context, button.into(), 1);
        for _ in 0..3 {
            *context.frame_phase = FramePhase::Construct;
            root.construct_frame(context);
            *context.frame_phase = FramePhase::FrameScripts;
            root.run_frame_scripts(context);
        }
        assert!(
            hidden.parent().is_none(),
            "test must cover a foreign state root"
        );
        test(root, hidden, button, context);
    });
}

#[test]
fn settled_button_states_allow_ancestor_walks_to_stop() {
    with_button_tree(|root, _, _, context| {
        VISITS.set(0);
        SCRIPT_VISITS.set(0);
        for _ in 0..5 {
            *context.frame_phase = FramePhase::Construct;
            root.construct_frame(context);
            *context.frame_phase = FramePhase::FrameScripts;
            root.run_frame_scripts(context);
        }
        assert_eq!(
            VISITS.get(),
            0,
            "buttons must not permanently block settled ancestors"
        );
        assert_eq!(SCRIPT_VISITS.get(), 0);
    });
}

#[test]
fn hidden_button_state_script_wakes_a_cached_ancestor() {
    with_button_tree(|root, hidden, _, context| {
        hidden.register_frame_script(
            hidden.current_frame(),
            Some(context.avm2.classes().object.into()),
            context,
        );
        *context.frame_phase = FramePhase::Construct;
        root.construct_frame(context);
        *context.frame_phase = FramePhase::FrameScripts;
        root.run_frame_scripts(context);
        assert_eq!(
            hidden.last_queued_script_frame(),
            Some(hidden.current_frame())
        );
        assert!(!hidden.has_pending_script());
    });
}

#[test]
fn hidden_button_state_play_wakes_cached_enter_frame() {
    with_button_tree(|root, hidden, _, context| {
        hidden.play();
        assert!(hidden.playing());
        ENTER_VISITS.set(0);
        *context.frame_phase = FramePhase::Enter;
        root.enter_frame(context);
        assert_eq!(
            ENTER_VISITS.get(),
            2,
            "the ancestor and hidden playing state must tick"
        );
    });
}

#[test]
fn replacing_an_inactive_button_state_keeps_its_pending_script() {
    with_button_tree(|root, _, button, context| {
        let replacement = MovieClip::new(root.movie(), context.gc());
        *context.frame_phase = FramePhase::Construct;
        replacement.construct_frame(context);
        replacement.register_frame_script(
            replacement.current_frame(),
            Some(context.avm2.classes().object.into()),
            context,
        );
        button.set_state_child(context, swf::ButtonState::OVER, Some(replacement.into()));
        root.construct_frame(context);
        *context.frame_phase = FramePhase::FrameScripts;
        root.run_frame_scripts(context);
        assert_eq!(
            replacement.last_queued_script_frame(),
            Some(replacement.current_frame())
        );
    });
}

#[test]
fn reloaded_hidden_button_state_still_invalidates_its_observers() {
    with_button_tree(|root, hidden, _, context| {
        hidden.replace_with_movie(context, Some(root.movie()), false, None);
        assert!(hidden.preload(context, &mut ExecutionLimit::none()));
        *context.frame_phase = FramePhase::Enter;
        hidden.enter_frame(context);
        hidden.set_initialized();
        hidden.stop(context);
        for _ in 0..3 {
            *context.frame_phase = FramePhase::Construct;
            root.construct_frame(context);
            *context.frame_phase = FramePhase::FrameScripts;
            root.run_frame_scripts(context);
        }
        assert!(root.enter_frame_is_idle());
        hidden.play();
        ENTER_VISITS.set(0);
        root.enter_frame(context);
        assert_eq!(
            ENTER_VISITS.get(),
            2,
            "reloading must preserve hidden-state invalidation"
        );
    });
}

#[test]
fn nested_shared_button_states_settle_and_wake_after_descendant_mutation() {
    with_button_tree(|root, hidden, button, context| {
        let nested = crate::display_object::Avm2Button::empty_button(context);
        let object = Avm2StageObject::for_display_object(
            context.gc(),
            nested.into(),
            context.avm2.classes().simplebutton,
        );
        nested.set_object2(context, object);
        let state = MovieClip::new(root.movie(), context.gc());
        let descendant = MovieClip::new(root.movie(), context.gc());
        state.replace_at_depth(context, descendant.into(), 1);
        nested.set_state_child(context, swf::ButtonState::OVER, Some(state.into()));
        // Also observed directly by another button: there is no single owner.
        button.set_state_child(context, swf::ButtonState::DOWN, Some(state.into()));
        hidden.replace_at_depth(context, nested.into(), 1);
        for _ in 0..5 {
            *context.frame_phase = FramePhase::Construct;
            root.construct_frame(context);
            *context.frame_phase = FramePhase::FrameScripts;
            root.run_frame_scripts(context);
        }
        VISITS.set(0);
        root.construct_frame(context);
        assert_eq!(VISITS.get(), 0, "nested state epochs must converge");
        descendant.register_frame_script(
            descendant.current_frame(),
            Some(context.avm2.classes().object.into()),
            context,
        );
        *context.frame_phase = FramePhase::Construct;
        root.construct_frame(context);
        *context.frame_phase = FramePhase::FrameScripts;
        root.run_frame_scripts(context);
        assert_eq!(
            descendant.last_queued_script_frame(),
            Some(descendant.current_frame())
        );
        for _ in 0..5 {
            *context.frame_phase = FramePhase::Construct;
            root.construct_frame(context);
            *context.frame_phase = FramePhase::FrameScripts;
            root.run_frame_scripts(context);
        }
        VISITS.set(0);
        SCRIPT_VISITS.set(0);
        root.construct_frame(context);
        root.run_frame_scripts(context);
        assert_eq!((VISITS.get(), SCRIPT_VISITS.get()), (0, 0));
    });
}
