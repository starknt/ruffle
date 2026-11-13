use super::*;
use crate::display_object::Graphic;
use crate::player::PlayerBuilder;
use crate::tag_utils::SwfMovieData;
use std::cell::Cell;

thread_local! {
    static VISITS: Cell<usize> = const { Cell::new(0) };
}

pub(super) fn record_visit() {
    VISITS.set(VISITS.get() + 1);
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
