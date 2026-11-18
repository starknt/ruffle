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
        assert!(metrics.snapshot().work.delta(before).tree_visits >= 16);
    }
}
