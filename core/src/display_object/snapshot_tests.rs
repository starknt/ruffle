use super::*;
use crate::player::PlayerBuilder;
use ruffle_render::profile::{RenderMetrics, enter};

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
    player.render();
    let first = metrics.snapshot().work;
    player.render();
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
        player.render();
        assert!(metrics.snapshot().work.delta(before).tree_visits >= 16);
    }
}
