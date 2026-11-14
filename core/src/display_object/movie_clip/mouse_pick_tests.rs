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
        for optimized in [false, true] {
            let _scope = ruffle_render::profile::enter_engine(optimized);
            BOUNDS_VISITS.set(0);
            assert!(matches!(
                root.mouse_pick_avm2(context, Point::ZERO, false),
                Avm2MousePick::Miss
            ));
            if optimized {
                assert_eq!(
                    BOUNDS_VISITS.get(),
                    0,
                    "empty containers must not recursively measure descendants after a miss"
                );
            } else {
                assert_eq!(
                    BOUNDS_VISITS.get(),
                    561,
                    "stock control visits every ancestor's subtree"
                );
            }
        }
    });
}

#[test]
fn nested_child_and_late_parent_graphics_remain_pickable() {
    with_tree(|root, context| {
        let child = MovieClip::new(root.movie(), context.gc());
        root.replace_at_depth(context, child.into(), 1);
        rectangle(child);
        let point = Point::new(Twips::new(100), Twips::new(100));
        for optimized in [false, true] {
            let _scope = ruffle_render::profile::enter_engine(optimized);
            assert!(
                matches!(root.mouse_pick_avm2(context, point, false), Avm2MousePick::Hit(hit) if hit == child.into())
            );
        }
        child.set_visible(context, false);
        for optimized in [false, true] {
            let _scope = ruffle_render::profile::enter_engine(optimized);
            assert!(matches!(
                root.mouse_pick_avm2(context, point, false),
                Avm2MousePick::Miss
            ));
        }
        rectangle(root);
        for optimized in [false, true] {
            let _scope = ruffle_render::profile::enter_engine(optimized);
            assert!(
                matches!(root.mouse_pick_avm2(context, point, false), Avm2MousePick::Hit(hit) if hit == root.into())
            );
        }
        root.drawing_mut().clear();
        for optimized in [false, true] {
            let _scope = ruffle_render::profile::enter_engine(optimized);
            assert!(matches!(
                root.mouse_pick_avm2(context, point, false),
                Avm2MousePick::Miss
            ));
        }
    });
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
        let mut stock = Vec::new();
        let mut optimized = Vec::new();
        for sample in 0..31 {
            for offset in 0..2 {
                let enabled = (sample + offset) % 2 == 0;
                let _scope = ruffle_render::profile::enter_engine(enabled);
                let start = Instant::now();
                for _ in 0..20 {
                    black_box(root.mouse_pick_avm2(context, Point::ZERO, false));
                }
                let ns = start.elapsed().as_nanos() / 20;
                if enabled {
                    optimized.push(ns);
                } else {
                    stock.push(ns);
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
