//! Exact bounds and stage-local transforms. The reference multiplication order
//! matters: twip translations round after each multiplication.
use crate::display_object::{BoundsMode, DisplayObject, TDisplayObject};
use ruffle_render::matrix::Matrix;
use std::cell::Cell;
use swf::{Rectangle, Twips};

#[derive(Clone, Copy, PartialEq)]
pub(super) struct Key {
    matrix: Matrix,
    kind: Kind,
}

#[derive(Clone, Copy, PartialEq)]
pub(in crate::display_object) enum Kind {
    Geometry(BoundsMode),
    Render { filters: bool, scale: [f32; 2] },
}

thread_local! {
    static TRANSFORM_CLOCK: Cell<u64> = const { Cell::new(0) };
    static VOLATILE_BOUNDS: Cell<u64> = const { Cell::new(0) };
}

pub(super) fn next_transform_stamp() -> u64 {
    let next = TRANSFORM_CLOCK
        .get()
        .checked_add(1)
        .expect("transform revision exhausted");
    TRANSFORM_CLOCK.set(next);
    next
}

fn tracks_bounds(object: DisplayObject<'_>) -> bool {
    matches!(
        object,
        DisplayObject::Graphic(_)
            | DisplayObject::MovieClip(_)
            | DisplayObject::LoaderDisplay(_)
            | DisplayObject::Bitmap(_)
    )
}

pub(in crate::display_object) fn observe_child(object: DisplayObject<'_>) {
    // Buttons can override the bounds walk and text or video can change size
    // outside the display-list invalidation path.
    if !tracks_bounds(object) {
        VOLATILE_BOUNDS.set(VOLATILE_BOUNDS.get().wrapping_add(1));
    }
}

pub(in crate::display_object) fn measure(
    object: DisplayObject<'_>,
    matrix: Matrix,
    kind: Kind,
    calculate: impl FnOnce() -> Rectangle<Twips>,
) -> Rectangle<Twips> {
    let base = object.base();
    let node = base.snapshots.node.borrow().clone();
    let key = Key { matrix, kind };
    if let Some((stored, bounds)) = node.as_ref().and_then(|node| node.bounds.get())
        && key == stored
    {
        return bounds;
    }
    let before = VOLATILE_BOUNDS.get();
    let bounds = calculate();
    if tracks_bounds(object) && VOLATILE_BOUNDS.get() == before {
        // A parent may observe this result even when this object has no cache.
        base.snapshots.bounds_observed.set(true);
        if let Some(node) = node {
            node.bounds.set(Some((key, bounds)));
        }
    }
    bounds
}

#[derive(Clone, Copy)]
pub(super) struct World {
    checked: u64,
    path_stamp: u64,
    matrix: Matrix,
}

pub(in crate::display_object) fn world(
    object: DisplayObject<'_>,
    calculate: impl FnOnce() -> Matrix,
) -> Matrix {
    let Some(node) = object.base().snapshots.node.borrow().clone() else {
        return calculate();
    };
    let clock = TRANSFORM_CLOCK.get();
    let cached = node.world.get();
    if let Some(entry) = cached
        && entry.checked == clock
    {
        return entry.matrix;
    }
    let mut stamp = object.base().snapshots.transform_stamp.get();
    let mut parent = object.parent();
    while let Some(ancestor) = parent {
        if ancestor.as_stage().is_some() {
            break;
        }
        let base = ancestor.base();
        stamp = stamp.max(base.snapshots.transform_stamp.get());
        parent = base.parent();
    }
    let matrix = cached
        .filter(|e| e.path_stamp == stamp)
        .map_or_else(calculate, |e| e.matrix);
    node.world.set(Some(World {
        checked: clock,
        path_stamp: stamp,
        matrix,
    }));
    matrix
}
