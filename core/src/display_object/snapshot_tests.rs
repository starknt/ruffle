use super::*;
use crate::player::PlayerBuilder;
use ruffle_render::profile::{RenderMetrics, enter};

#[test]
fn static_swf_containers_are_admitted_to_snapshot_lookup() {
    let movie = crate::tag_utils::SwfMovieData::from_data(
        include_bytes!("../../../tests/tests/swfs/native_perf/static_tree/test.swf"),
        "file:///static-tree.swf".into(),
        None,
        None,
    )
    .unwrap();
    let player = PlayerBuilder::new()
        .with_movie(movie)
        .with_viewport_dimensions(1280, 720, 1.0)
        .with_autoplay(true)
        .build();
    let mut player = player.lock().unwrap();
    assert!(player.preload(&mut crate::limits::ExecutionLimit::none()));
    player.run_frame();
    let metrics = RenderMetrics::default();
    let _scope = enter(Some(&metrics));
    player.render().unwrap();
    assert!(metrics.snapshot().work.snapshot_misses > 0);
}

#[test]
fn retained_stage_subtree_reuses_and_immediately_invalidates_after_render() {
    let player = PlayerBuilder::new()
        .with_viewport_dimensions(64, 64, 1.0)
        .with_scale_mode(crate::display_object::StageScaleMode::NoScale, true)
        .build();
    let mut player = player.lock().unwrap();
    player.mutate_with_update_context(|context| {
        let movie = context.stage.movie();
        let parent = MovieClip::new(movie, context.gc());
        for depth in 1..=16 {
            let child = Graphic::empty(context);
            parent.replace_at_depth(context, child.into(), depth);
        }
        context.stage.replace_at_depth(context, parent.into(), 1);
    });
    let metrics = RenderMetrics::default();
    let _scope = enter(Some(&metrics));
    player.render().unwrap();
    let first = metrics.snapshot().work;
    player.render().unwrap();
    let second = metrics.snapshot().work.delta(first);
    assert!(second.snapshot_hits > 0);
    assert!(second.tree_visits < 8);
    // Two changes in the same logical frame must each invalidate, even though
    // cacheAsBitmap's once-per-frame invalidation flag may already be set.
    for x in [17, 29] {
        player.mutate_with_update_context(|context| {
            let parent = context.stage.child_by_index(1).unwrap();
            parent
                .as_container()
                .unwrap()
                .child_by_index(0)
                .unwrap()
                .set_matrix(Matrix::translate(Twips::new(x), Twips::ZERO));
        });
        let before = metrics.snapshot().work;
        player.render().unwrap();
        let delta = metrics.snapshot().work.delta(before);
        assert!(
            delta.tree_visits < 8,
            "single leaf expanded {} nodes",
            delta.tree_visits
        );
    }
}

#[test]
fn wide_edit_keeps_old_order_and_copies_only_affected_pages() {
    let player = PlayerBuilder::new().build();
    player
        .lock()
        .unwrap()
        .mutate_with_update_context(|context| {
            let mut parent = MovieClip::new(context.stage.movie(), context.gc());
            for depth in 1..=512 {
                let child = Graphic::empty(context);
                parent.replace_at_depth(context, child.into(), depth);
            }
            let first = parent.child_by_index(0).unwrap();
            let last = parent.child_by_index(511).unwrap();
            let mut old = parent.iter_render_list();
            let metrics = RenderMetrics::default();
            let _scope = enter(Some(&metrics));
            parent.swap_at_index(context, 0, 511);
            assert!(DisplayObject::ptr_eq(old.next().unwrap(), first));
            assert!(DisplayObject::ptr_eq(old.next_back().unwrap(), last));
            assert!(DisplayObject::ptr_eq(
                parent.child_by_index(0).unwrap(),
                last
            ));
            assert!(DisplayObject::ptr_eq(
                parent.child_by_index(511).unwrap(),
                first
            ));
            assert!(
                metrics.snapshot().work.child_slots_copied <= 128,
                "copied {} child slots",
                metrics.snapshot().work.child_slots_copied
            );
        });
}

#[test]
fn moving_a_drawn_leaf_updates_its_instance_without_rebuilding_geometry() {
    let movie = crate::tag_utils::SwfMovieData::from_data(
        include_bytes!("../../../tests/tests/swfs/native_perf/static_tree/test.swf"),
        "file:///instances.swf".into(),
        None,
        None,
    )
    .unwrap();
    let player = PlayerBuilder::new()
        .with_movie(movie)
        .with_viewport_dimensions(1280, 720, 1.0)
        .with_autoplay(true)
        .build();
    let mut player = player.lock().unwrap();
    assert!(player.preload(&mut crate::limits::ExecutionLimit::none()));
    player.run_frame();
    for _ in 0..3 {
        player.render().unwrap();
    }
    player.mutate_with_update_context(|ctx| {
        let mut stack: Vec<_> = ctx.stage.iter_render_list().collect();
        while let Some(obj) = stack.pop() {
            if let Some(container) = obj.as_container() {
                if container.num_children() == 40 {
                    container
                        .child_by_index(0)
                        .unwrap()
                        .set_x(Twips::from_pixels(170.));
                    return;
                }
                stack.extend(container.iter_render_list());
            }
        }
        panic!("missing fixture group")
    });
    let metrics = RenderMetrics::default();
    let _scope = enter(Some(&metrics));
    player.render().unwrap();
    assert!(metrics.snapshot().work.render_instances_updated > 0);
}

#[test]
fn reordering_a_wide_subtree_reuses_unchanged_child_draws() {
    let player = PlayerBuilder::new()
        .with_viewport_dimensions(128, 128, 1.0)
        .with_scale_mode(StageScaleMode::NoScale, true)
        .build();
    let mut player = player.lock().unwrap();
    player.mutate_with_update_context(|ctx| {
        let parent = MovieClip::new(ctx.stage.movie(), ctx.gc());
        for depth in 1..=256 {
            let child = Graphic::empty(ctx);
            parent.replace_at_depth(ctx, child.into(), depth);
        }
        ctx.stage.replace_at_depth(ctx, parent.into(), 1);
    });
    player.render().unwrap();
    player.render().unwrap();
    player.mutate_with_update_context(|ctx| {
        let mut parent = ctx.stage.child_by_depth(1).unwrap().as_container().unwrap();
        parent.swap_at_index(ctx, 0, 255);
    });
    let metrics = RenderMetrics::default();
    let _scope = enter(Some(&metrics));
    player.render().unwrap();
    assert!(
        metrics.snapshot().work.tree_visits < 8,
        "expanded {} nodes",
        metrics.snapshot().work.tree_visits
    );
}

#[test]
fn clean_bounds_skip_descendants_and_world_matrix_observes_ancestor_changes() {
    let player = PlayerBuilder::new().build();
    player.lock().unwrap().mutate_with_update_context(|ctx| {
        let parent = MovieClip::new(ctx.stage.movie(), ctx.gc());
        for depth in 1..=20 {
            let child = Graphic::empty(ctx);
            parent.replace_at_depth(ctx, child.into(), depth);
        }
        let child = parent.child_by_index(0).unwrap();
        let mut cache = snapshot::Cache::default();
        parent.base().snapshots.node(&mut cache).unwrap();
        child.base().snapshots.node(&mut cache).unwrap();
        let matrix = Matrix::IDENTITY;
        parent.render_bounds_with_transform(&matrix, true, &matrix);
        child.local_to_global_matrix_without_own_scroll_rect();
        let metrics = RenderMetrics::default();
        let _scope = enter(Some(&metrics));
        parent.render_bounds_with_transform(&matrix, true, &matrix);
        child.local_to_global_matrix_without_own_scroll_rect();
        assert_eq!(metrics.snapshot().work.render_bounds_nodes, 0);
        assert_eq!(metrics.snapshot().work.world_matrix_builds, 0);
        for x in [5, 19] {
            parent.set_x(Twips::new(x));
            assert_eq!(
                child.local_to_global_matrix_without_own_scroll_rect().tx,
                Twips::new(x)
            );
        }
        for x in [7, 23] {
            let before = metrics.snapshot().work;
            child.set_x(Twips::new(x));
            parent.render_bounds_with_transform(&matrix, true, &matrix);
            assert!(metrics.snapshot().work.delta(before).render_bounds_nodes > 0);
        }
    });
}

#[test]
fn stable_external_stencil_reuses_its_geometry_and_rechecks_moved_ancestors() {
    let player = PlayerBuilder::new()
        .with_viewport_dimensions(128, 128, 1.0)
        .with_scale_mode(StageScaleMode::NoScale, true)
        .build();
    let mut player = player.lock().unwrap();
    player.mutate_with_update_context(|ctx| {
        let parent = MovieClip::new(ctx.stage.movie(), ctx.gc());
        let mask = MovieClip::new(ctx.stage.movie(), ctx.gc());
        let subject = MovieClip::new(ctx.stage.movie(), ctx.gc());
        for depth in 1..=64 {
            let g = Graphic::empty(ctx);
            mask.replace_at_depth(ctx, g.into(), depth);
        }
        ctx.stage.replace_at_depth(ctx, parent.into(), 1);
        ctx.stage.replace_at_depth(ctx, subject.into(), 2);
        parent.replace_at_depth(ctx, mask.into(), 1);
        subject.set_mask(Some(mask.into()), ctx.gc());
    });
    player.render().unwrap();
    let metrics = RenderMetrics::default();
    let _scope = enter(Some(&metrics));
    player.render().unwrap();
    assert!(
        metrics.snapshot().work.tree_visits < 14,
        "stable stencil expanded {} nodes",
        metrics.snapshot().work.tree_visits
    );
    player.mutate_with_update_context(|ctx| {
        ctx.stage.child_by_depth(1).unwrap().set_x(Twips::new(40));
    });
    let before = metrics.snapshot().work;
    player.render().unwrap();
    assert!(metrics.snapshot().work.delta(before).tree_visits >= 64);
}
