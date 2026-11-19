//! Exact bounds and stage-local transforms. The reference multiplication order
//! matters: twip translations round after each multiplication.
use crate::display_object::{BoundsMode, DisplayObject, TDisplayObject};
use ruffle_render::matrix::Matrix;
use std::cell::Cell;
use swf::{Rectangle, Twips};

// Leaf instances do not need the much larger container directory/entry node.
pub(super) struct Spatial {
    bounds: Cell<Option<(Key, Rectangle<Twips>)>>,
    world: Cell<Option<World>>,
    path: Cell<(u64, u64)>,
    _charge: super::NodeCharge,
}

impl Spatial {
    pub(super) fn new(charge: super::NodeCharge) -> Self {
        Self {
            bounds: Default::default(),
            world: Default::default(),
            path: Default::default(),
            _charge: charge,
        }
    }
    pub(super) fn clear_bounds(&self) {
        self.bounds.set(None);
    }
}

#[derive(Clone, Copy, PartialEq)]
struct Key {
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
    let key = Key { matrix, kind };
    if let Some((stored, bounds)) = base
        .snapshots
        .spatial
        .borrow()
        .as_ref()
        .and_then(|space| space.bounds.get())
        && key == stored
    {
        return bounds;
    }
    let before = VOLATILE_BOUNDS.get();
    let bounds = calculate();
    if tracks_bounds(object) && VOLATILE_BOUNDS.get() == before {
        // A parent may observe this result even when this object has no cache.
        base.snapshots.bounds_observed.set(true);
        if let Some(space) = base.snapshots.spatial.borrow().as_ref() {
            space.bounds.set(Some((key, bounds)));
        }
    }
    bounds
}

#[derive(Clone, Copy)]
struct World {
    checked: u64,
    path_stamp: u64,
    matrix: Matrix,
}

pub(in crate::display_object) fn world(
    object: DisplayObject<'_>,
    calculate: impl FnOnce() -> Matrix,
) -> Matrix {
    let base = object.base();
    let storage = base.snapshots.spatial.borrow();
    let Some(space) = storage.as_ref() else {
        return calculate();
    };
    let clock = TRANSFORM_CLOCK.get();
    let cached = space.world.get();
    if let Some(entry) = cached
        && entry.checked == clock
    {
        return entry.matrix;
    }
    let mut stamp = base.snapshots.transform_stamp.get();
    if let Some(parent) = object.parent().filter(|p| p.as_stage().is_none()) {
        let parent_base = parent.base();
        let parent_space = parent_base.snapshots.spatial.borrow();
        let parent_stamp = if let Some((checked, stamp)) =
            parent_space.as_ref().map(|s| s.path.get())
            && checked == clock
        {
            stamp
        } else {
            let mut ancestor_stamp = parent_base.snapshots.transform_stamp.get();
            let mut next = parent.parent();
            while let Some(ancestor) = next {
                if ancestor.as_stage().is_some() {
                    break;
                }
                let base = ancestor.base();
                if let Some((checked, prior)) = base
                    .snapshots
                    .spatial
                    .borrow()
                    .as_ref()
                    .map(|s| s.path.get())
                    && checked == clock
                {
                    ancestor_stamp = ancestor_stamp.max(prior);
                    break;
                }
                ancestor_stamp = ancestor_stamp.max(base.snapshots.transform_stamp.get());
                next = base.parent();
            }
            if let Some(s) = parent_space.as_ref() {
                s.path.set((clock, ancestor_stamp));
            }
            ancestor_stamp
        };
        stamp = stamp.max(parent_stamp);
    }
    space.path.set((clock, stamp));
    let matrix = cached
        .filter(|e| e.path_stamp == stamp)
        .map_or_else(calculate, |e| e.matrix);
    space.world.set(Some(World {
        checked: clock,
        path_stamp: stamp,
        matrix,
    }));
    matrix
}
