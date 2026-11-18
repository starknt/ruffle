use super::*;
use crate::player::PlayerBuilder;
use crate::tag_utils::SwfMovieData;
use ruffle_render::shape_utils::DrawCommand;
use std::cell::Cell;

thread_local! {
    static BOUNDS_VISITS: Cell<usize> = const { Cell::new(0) };
}

pub(super) fn record_bounds_visit() {
    BOUNDS_VISITS.set(BOUNDS_VISITS.get() + 1);
}

fn with_tree(test: impl for<'gc> FnOnce(MovieClip<'gc>, &mut UpdateContext<'gc>)) {
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
    let movie =
        SwfMovieData::from_data(&bytes, "file:///mouse-pick-test.swf".into(), None, None).unwrap();
    let player = PlayerBuilder::new().with_movie(movie).build();
    player
        .lock()
        .unwrap()
        .mutate_with_update_context(|context| {
            let movie = context.stage.root_clip().unwrap().movie();
            let root = MovieClip::new(movie, context.gc());
            test(root, context);
        });
}

fn rectangle(clip: MovieClip<'_>) {
    let mut drawing = clip.drawing_mut();
    drawing.set_fill_style(Some(swf::FillStyle::Color(swf::Color::RED)));
    drawing.draw_command(DrawCommand::MoveTo(Point::ZERO));
    for (x, y) in [(200, 0), (200, 200), (0, 200), (0, 0)] {
        drawing.draw_command(DrawCommand::LineTo(Point::new(
            Twips::new(x),
            Twips::new(y),
        )));
    }
    drawing.set_fill_style(None);
}

#[test]
fn missing_drawings_do_not_repeat_descendant_bounds_walks() {
    with_tree(|root, context| {
        let mut parent = root;
        for _ in 0..32 {
            let child = MovieClip::new(root.movie(), context.gc());
            parent.replace_at_depth(context, child.into(), 1);
            parent = child;
        }
        BOUNDS_VISITS.set(0);
        assert!(matches!(
            root.mouse_pick_avm2(context, Point::ZERO, false),
            Avm2MousePick::Miss
        ));
        assert_eq!(
            BOUNDS_VISITS.get(),
            0,
            "empty containers must not recursively measure descendants after a miss"
        );
    });
}

#[test]
fn nested_child_and_late_parent_graphics_remain_pickable() {
    with_tree(|root, context| {
        let child = MovieClip::new(root.movie(), context.gc());
        root.replace_at_depth(context, child.into(), 1);
        rectangle(child);
        let point = Point::new(Twips::new(100), Twips::new(100));
        assert!(
            matches!(root.mouse_pick_avm2(context, point, false), Avm2MousePick::Hit(hit) if hit == child.into())
        );
        child.set_visible(context, false);
        assert!(matches!(
            root.mouse_pick_avm2(context, point, false),
            Avm2MousePick::Miss
        ));
        rectangle(root);
        assert!(
            matches!(root.mouse_pick_avm2(context, point, false), Avm2MousePick::Hit(hit) if hit == root.into())
        );
        root.drawing_mut().clear();
        assert!(matches!(
            root.mouse_pick_avm2(context, point, false),
            Avm2MousePick::Miss
        ));
    });
}

#[test]
fn timeline_mask_clips_a_script_added_child_until_a_later_timeline_depth() {
    with_tree(|root, context| {
        let button = MovieClip::new(root.movie(), context.gc());
        rectangle(button);
        root.replace_at_depth(context, button.into(), 1);
        let mut card = MovieClip::new(root.movie(), context.gc());
        root.replace_at_depth(context, card.into(), 2);

        let mask = MovieClip::new(root.movie(), context.gc());
        rectangle(mask);
        mask.set_x(Twips::new(100));
        mask.set_clip_depth(3);
        card.replace_at_depth(context, mask.into(), 1);
        let placeholder = MovieClip::new(root.movie(), context.gc());
        card.replace_at_depth(context, placeholder.into(), 2);
        let portrait = MovieClip::new(root.movie(), context.gc());
        rectangle(portrait);
        card.insert_at_index(context, portrait.into(), card.num_children());

        assert_pick(root, context, 150, portrait);
        assert_pick(root, context, 50, button);

        // A later timeline placement ends the mask before the next addChild.
        let boundary = MovieClip::new(root.movie(), context.gc());
        card.replace_at_depth(context, boundary.into(), 4);
        let overlay = MovieClip::new(root.movie(), context.gc());
        rectangle(overlay);
        card.insert_at_index(context, overlay.into(), card.num_children());
        assert_pick(root, context, 50, overlay);
    });
}

#[test]
fn nested_timeline_masks_both_apply_at_the_inclusive_end_depth() {
    with_tree(|root, context| {
        let button = MovieClip::new(root.movie(), context.gc());
        rectangle(button);
        root.replace_at_depth(context, button.into(), 1);
        let outer = MovieClip::new(root.movie(), context.gc());
        rectangle(outer);
        outer.set_x(Twips::new(100));
        outer.set_clip_depth(5);
        root.replace_at_depth(context, outer.into(), 2);
        let inner = MovieClip::new(root.movie(), context.gc());
        rectangle(inner);
        inner.set_clip_depth(4);
        root.replace_at_depth(context, inner.into(), 3);
        let child = MovieClip::new(root.movie(), context.gc());
        rectangle(child);
        root.replace_at_depth(context, child.into(), 4);

        assert_pick(root, context, 150, child);
        assert_pick(root, context, 50, button);

        let sibling = MovieClip::new(root.movie(), context.gc());
        rectangle(sibling);
        sibling.set_x(Twips::new(100));
        root.replace_at_depth(context, sibling.into(), 5);
        assert_pick(root, context, 250, sibling);
    });
}

#[test]
fn timeline_mask_still_applies_when_picking_non_interactive_children() {
    with_tree(|root, context| {
        let button = MovieClip::new(root.movie(), context.gc());
        rectangle(button);
        root.replace_at_depth(context, button.into(), 1);
        let card = MovieClip::new(root.movie(), context.gc());
        root.replace_at_depth(context, card.into(), 2);

        // The interactive pass visits this empty sibling before the bitmap.
        let empty = MovieClip::new(root.movie(), context.gc());
        card.replace_at_depth(context, empty.into(), 1);
        let mask = MovieClip::new(root.movie(), context.gc());
        rectangle(mask);
        mask.set_x(Twips::new(100));
        mask.set_clip_depth(3);
        card.replace_at_depth(context, mask.into(), 2);
        let pixels = ruffle_render::bitmap::Bitmap::new(
            10,
            10,
            ruffle_render::bitmap::BitmapFormat::Rgb,
            vec![255; 300],
        );
        let bitmap = crate::display_object::Bitmap::new(context.gc(), 0, pixels, root.movie());
        card.replace_at_depth(context, bitmap.into(), 3);

        assert_pick(root, context, 150, card);
        assert_pick(root, context, 50, button);
    });
}

fn assert_pick<'gc>(
    root: MovieClip<'gc>,
    context: &mut UpdateContext<'gc>,
    x: i32,
    expected: MovieClip<'gc>,
) {
    let point = Point::new(Twips::new(x), Twips::new(100));
    let result = root.mouse_pick_avm2(context, point, false);
    assert!(
        matches!(result, Avm2MousePick::Hit(hit) if hit == expected.into()),
        "at {point:?}, expected {expected:?}, got {result:?}"
    );
}

#[test]
#[ignore = "diagnostic mouse-pick benchmark; no machine-dependent FPS assertion"]
fn empty_container_mouse_pick_benchmark() {
    use std::{hint::black_box, time::Instant};
    with_tree(|root, context| {
        for depth in 1..101 {
            let mut parent = MovieClip::new(root.movie(), context.gc());
            root.replace_at_depth(context, parent.into(), depth);
            for _ in 0..10 {
                let child = MovieClip::new(root.movie(), context.gc());
                parent.replace_at_depth(context, child.into(), 1);
                parent = child;
            }
        }
        let mut samples = Vec::new();
        for sample in 0..31 {
            for offset in 0..2 {
                let start = Instant::now();
                for _ in 0..20 {
                    black_box(root.mouse_pick_avm2(context, Point::ZERO, false));
                }
                let ns = start.elapsed().as_nanos() / 20;
                samples.push(ns);
            }
        }
        samples.sort_unstable();
        println!(
            "optimized_ns_p50={} optimized_ns_p95={}",
            samples[15], samples[29]
        );
    });
}
