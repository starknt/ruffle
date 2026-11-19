//! Index logical child identities into balanced, immutable command blocks. The
//! index contains no GC edges. A structural epoch owns each mapping exclusively.
use super::{Change, Entry, Key, Node, NodeCharge};
use crate::context::RenderContext;
use crate::display_object::{
    DisplayObject, DisplayObjectContainer, TDisplayObject, TDisplayObjectContainer,
};
use fnv::FnvHashMap;
use ruffle_render::commands::{Command, CommandList};
use ruffle_render::{
    backend::{ShapeHandle, ShapeHandleImpl},
    transform::Transform,
};
use std::{collections::BTreeSet, ops::Range, rc::Rc};

const WIDTH: usize = 16;
type Block = Rc<CommandList>;

enum SlotEdit {
    Draw(Command),
    Color(swf::ColorTransform),
}

struct Instance {
    shape: std::sync::Weak<dyn ShapeHandleImpl>,
    basis: [f32; 4],
    graphic: bool,
}
impl Instance {
    fn from_draw<'gc>(
        child: DisplayObject<'gc>,
        block: &Block,
        inherited: &Transform,
    ) -> Option<Self> {
        let graphic = matches!(child, DisplayObject::Graphic(_));
        if !graphic && !matches!(child, DisplayObject::MovieClip(mc) if mc.num_children() == 0) {
            return None;
        }
        let [Command::RenderShape { shape, transform }] = block.iter().as_slice() else {
            return None;
        };
        let local = child.base().transform(true);
        if inherited.concat(&local) != *transform {
            return None;
        }
        let m = transform.matrix;
        Some(Self {
            shape: std::sync::Arc::downgrade(&shape.0),
            basis: [m.a, m.b, m.c, m.d],
            graphic,
        })
    }
    fn update<'gc>(
        &self,
        child: DisplayObject<'gc>,
        reason: Change,
        parent_change: Change,
        inherited: &Transform,
        context: &mut RenderContext<'_, 'gc>,
        empty: &Block,
    ) -> Option<SlotEdit> {
        if !(reason - (Change::TRANSFORM | Change::COLOR | Change::VISIBILITY)).is_empty() {
            return None;
        }
        if !reason.is_empty() {
            child.pre_render(context);
        }
        if ((reason | parent_change) - Change::COLOR).is_empty() {
            // Visibility and coverage are unchanged, even for a culled slot.
            let color = inherited.color_transform * child.base().color_transform();
            ruffle_render::profile::work(|w| w.render_instances_updated += 1);
            return Some(SlotEdit::Color(color));
        }
        if !child.visible() {
            ruffle_render::profile::work(|w| w.render_instances_updated += 1);
            return Some(SlotEdit::Draw(Command::Shared(empty.clone())));
        }
        let local = child.base().transform(true);
        #[cfg(test)]
        tests::INSTANCE_MATRICES.update(|count| count + 1);
        let world = inherited.concat(&local);
        let m = world.matrix;
        // Timeline shapes choose their tessellation using the linear matrix.
        // Script graphics use the Drawing's existing, scale-independent handle.
        if world.tz != 0.
            || local.perspective_projection.is_some()
            || self.graphic && self.basis != [m.a, m.b, m.c, m.d]
        {
            return None;
        }
        let command = if !self.graphic
            || child
                .world_bounds(super::super::BoundsMode::Engine)
                .intersects(&context.stage.view_bounds())
        {
            Command::RenderShape {
                shape: ShapeHandle(self.shape.upgrade()?),
                transform: world,
            }
        } else {
            Command::Shared(empty.clone())
        };
        ruffle_render::profile::work(|w| w.render_instances_updated += 1);
        Some(SlotEdit::Draw(command))
    }
}

pub(in crate::display_object) struct Children {
    layout_revision: u64,
    ids: FnvHashMap<u64, usize>,
    scopes: Vec<Range<usize>>,
    dirty: Vec<usize>,
    reasons: Vec<Change>,
    instances: Vec<Option<Instance>>,
    live: BTreeSet<usize>,
    owner: Rc<Node>,
    stride: usize,
    _metadata: NodeCharge,
}

impl super::State {
    pub(in crate::display_object) fn resize_children(&self, count: usize) {
        if let Some(node) = self.node.borrow().as_ref() {
            let mut children = node.children.borrow_mut();
            if children
                .as_ref()
                .is_some_and(|index| count < 2 || index.ids.len() > count.saturating_mul(2))
            {
                children.take();
            }
        }
    }
}

impl Children {
    pub(super) fn mark(&mut self, child: u64, reason: Change) {
        if let Some(index) = self.ids.get(&child) {
            if self.reasons[*index].is_empty() {
                self.dirty.push(*index);
            }
            self.reasons[*index] |= reason;
        }
    }
}

// Keep one painter-order slot per child or complete mask scope. Simple
// geometry lives directly in its leaf page; only compound slots need a block.
fn slot_command(block: Block) -> Command {
    match block.iter().as_slice() {
        [
            cmd @ (Command::RenderShape { .. }
            | Command::DrawRect { .. }
            | Command::DrawLine { .. }
            | Command::DrawLineRect { .. }
            | Command::Shared(_)),
        ] => cmd.clone(),
        _ => Command::Shared(block),
    }
}

fn chunk(items: impl ExactSizeIterator<Item = Command>, stride: usize) -> Block {
    let slots = items.map(|command| {
        let region = command.region();
        (command, region)
    });
    Rc::new(CommandList::from_render_page(slots, stride == 1))
}

fn pack(mut level: Vec<Command>) -> (Block, usize) {
    let mut stride = 1;
    while level.len() > WIDTH {
        let mut items = level.into_iter();
        let mut next = Vec::with_capacity(items.len().div_ceil(WIDTH));
        while items.len() != 0 {
            next.push(Command::Shared(chunk((&mut items).take(WIDTH), stride)));
        }
        level = next;
        stride *= WIDTH;
    }
    (chunk(level.into_iter(), stride), stride)
}

// Stable siblings keep their pages. Only pages intersecting a patch are copied.
fn patch(root: &Block, stride: usize, changes: Vec<(usize, SlotEdit)>) -> Block {
    type Patches = std::iter::Peekable<std::vec::IntoIter<(usize, SlotEdit)>>;
    fn update(root: &Block, stride: usize, offset: usize, changes: &mut Patches) -> Block {
        let end = offset + root.iter().len() * stride;
        if changes.peek().is_none_or(|(slot, _)| *slot >= end) {
            return root.clone();
        }
        let slots = root.iter().enumerate().map(|(i, command)| {
            let start = offset + i * stride;
            if changes
                .peek()
                .is_none_or(|(slot, _)| *slot >= start + stride)
            {
                return (command.clone(), root.command_region(i));
            }
            let command = if stride == 1 {
                match changes.next().unwrap().1 {
                    SlotEdit::Draw(draw) => draw,
                    SlotEdit::Color(color) => {
                        let mut draw = command.clone();
                        match &mut draw {
                            Command::RenderShape { transform, .. } => {
                                transform.color_transform = color
                            }
                            Command::Shared(block) => debug_assert!(block.is_empty()),
                            _ => unreachable!("only a simple shape instance can be recolored"),
                        }
                        return (draw, root.command_region(i));
                    }
                }
            } else {
                let Command::Shared(child) = command else {
                    unreachable!("invalid chunk directory")
                };
                Command::Shared(update(child, stride / WIDTH, start, changes))
            };
            let region = command.region();
            (command, region)
        });
        Rc::new(CommandList::from_render_page(slots, stride == 1))
    }
    update(root, stride, 0, &mut changes.into_iter().peekable())
}

fn slot_at(root: &Block, mut stride: usize, mut index: usize) -> Command {
    let mut node = root;
    loop {
        let command = &node.iter().as_slice()[index / stride];
        if stride == 1 {
            return command.clone();
        }
        let Command::Shared(child) = command else {
            unreachable!("invalid chunk directory")
        };
        index %= stride;
        stride /= WIDTH;
        node = child;
    }
}

fn layout<'gc>(
    container: DisplayObjectContainer<'gc>,
    children: &mut Children,
    cache: &mut super::Cache,
) -> bool {
    let mut depths = Vec::new();
    for (i, child) in container.iter_render_list().enumerate() {
        while depths.last().is_some_and(|end| child.depth() > *end) {
            depths.pop();
        }
        if depths.is_empty() {
            children.scopes.push(i..i + 1);
        }
        let index = children.scopes.len() - 1;
        children.scopes[index].end = i + 1;
        child.base().snapshots.observe_spatial(cache);
        if children
            .ids
            .insert(child.base().snapshots.id(), index)
            .is_some()
        {
            return false;
        }
        if child.clip_depth() > 0 {
            depths.push(child.clip_depth());
        }
    }
    true
}

fn capture<'gc>(
    container: DisplayObjectContainer<'gc>,
    range: Range<usize>,
    context: &mut RenderContext<'_, 'gc>,
) -> (Block, bool) {
    let before = context.snapshot_barriers;
    let original = std::mem::take(&mut context.commands);
    let recording = std::mem::replace(&mut context.snapshot_recording, true);
    super::super::container::render_order(
        range.map(|i| container.child_by_index(i).unwrap()),
        context,
    );
    context.snapshot_recording = recording;
    let commands = Rc::new(std::mem::replace(&mut context.commands, original));
    let stable = before == context.snapshot_barriers && commands.is_retainable();
    if !stable && before == context.snapshot_barriers {
        context.snapshot_barriers += 1;
    }
    (commands, stable)
}

pub(in crate::display_object) fn render<'gc>(
    container: DisplayObjectContainer<'gc>,
    count: usize,
    context: &mut RenderContext<'_, 'gc>,
) -> bool {
    if !context.snapshot_allowed || context.is_offscreen || context.commands.drawing_mask() {
        return false;
    }
    let Some(cache) = context.snapshots.as_deref_mut() else {
        return false;
    };
    let object: DisplayObject<'gc> = container.into();
    let base = object.base();
    let Some(node) = base.snapshots.node(cache) else {
        return false;
    };
    if node.live_until.get() > context.render_epoch {
        context.snapshot_barriers += 1;
        return false;
    }
    if node.dense.get() && node.dirty_children.replace(0) > count / 2 {
        return false;
    }
    node.dense.set(false);
    if node.live_until.get() == 0
        && node.children.borrow().is_none()
        && matches!(container.child_by_index(0), Some(DisplayObject::Graphic(g)) if g.needs_first_tessellation())
    {
        // The first real timeline frame of a loaded movie can appear after
        // the Loader itself has already drawn. Stream cold shape handles
        // directly and allow admission again on the next stage submission.
        node.live_until.set(context.render_epoch.saturating_add(1));
        context.snapshot_barriers += 1;
        return false;
    }
    if node.children.borrow().is_none()
        && container.raw_container().iter_render_list().all(|child| {
            matches!(child, DisplayObject::Video(_) | DisplayObject::EditText(_))
                || child.is_bitmap_cached()
                || child.masker().is_some()
                || child.maskee().is_some()
        })
    {
        node.live_until.set(
            context
                .render_epoch
                .saturating_add(super::LIVE_RETRY_FRAMES),
        );
        context.snapshot_barriers += 1;
        return false;
    }
    let dimensions = context.renderer.viewport_dimensions();
    let key = Key {
        mask_geometry: false,
        generation: container.raw_container().children_epoch(),
        inherited: context.transform_stack.transform(),
        own: Default::default(),
        viewport: (
            dimensions.width,
            dimensions.height,
            dimensions.scale_factor.to_bits(),
        ),
        view: context.stage.view_bounds(),
        quality: context.stage.quality(),
    };
    // Take the index out before rendering children. No RefCell borrow spans a draw.
    let mut index = node.children.borrow_mut().take();
    let previous = index.as_ref().and_then(|index| {
        index
            .owner
            .entry
            .borrow()
            .as_ref()
            .filter(|entry| {
                index.layout_revision == node.layout_revision.get()
                    && entry.key.generation == key.generation
                    && entry.key.viewport == key.viewport
                    && entry.key.view == key.view
                    && entry.key.quality == key.quality
            })
            .map(|entry| {
                let prior = &entry.key.inherited;
                let current = &key.inherited;
                let mut inherited = Change::empty();
                inherited.set(
                    Change::TRANSFORM,
                    prior.matrix != current.matrix
                        || prior.tz != current.tz
                        || prior.perspective_projection != current.perspective_projection,
                );
                inherited.set(
                    Change::COLOR,
                    prior.color_transform != current.color_transform,
                );
                (entry.commands.clone(), inherited)
            })
    });
    let empty = cache.empty();
    let (index, stable_root, output) = if let Some((old, inherited)) = previous {
        let mut index = index.take().unwrap();
        if count >= 128 && index.dirty.len() > index.scopes.len() / 2 {
            node.dense.set(true);
            return false;
        }
        let mut work = std::mem::take(&mut index.dirty);
        if !inherited.is_empty() {
            work.clear();
            work.extend(0..index.scopes.len());
        } else {
            work.extend(index.live.iter().copied());
            work.sort_unstable();
            if !index.live.is_empty() {
                work.dedup();
            }
        }
        let mut stable_patches = Vec::with_capacity(work.len());
        let mut live_patches = Vec::new();
        ruffle_render::profile::work(|w| w.child_slots_examined += work.len() as u64);
        for slot in work.drain(..) {
            let scope = index.scopes[slot].clone();
            let reason = std::mem::take(&mut index.reasons[slot]);
            let child = container.child_by_index(scope.start).unwrap();
            let instance = index.instances[slot]
                .as_ref()
                .and_then(|i| i.update(child, reason, inherited, &key.inherited, context, &empty));
            let (block, stable) = if let Some(block) = instance {
                (block, true)
            } else {
                let one = scope.len() == 1;
                let (block, stable) = capture(container, scope, context);
                index.instances[slot] = if stable && one {
                    Instance::from_draw(child, &block, &key.inherited)
                } else {
                    None
                };
                (SlotEdit::Draw(slot_command(block)), stable)
            };
            if stable {
                index.live.remove(&slot);
                stable_patches.push((slot, block));
            } else {
                if index.live.insert(slot) {
                    stable_patches.push((slot, SlotEdit::Draw(Command::Shared(empty.clone()))));
                }
                live_patches.push((slot, block));
            }
        }
        index.dirty = work;
        if stable_patches.is_empty() {
            ruffle_render::profile::work(|w| w.snapshot_hits += 1);
        }
        let root = patch(&old, index.stride, stable_patches);
        let output = patch(&root, index.stride, live_patches);
        let changed = !inherited.is_empty() || !Rc::ptr_eq(&root, &old);
        (index, changed.then_some(root), output)
    } else {
        let mut old = index.take().and_then(|previous| {
            let root = previous
                .owner
                .entry
                .borrow()
                .as_ref()
                .filter(|e| {
                    e.key.viewport == key.viewport
                        && e.key.view == key.view
                        && e.key.quality == key.quality
                        && e.key.inherited == key.inherited
                })
                .map(|e| e.commands.clone())?;
            Some((previous, root))
        });
        // Charge worst-case directory, hash, and dirty/live tree storage in advance.
        let Some(metadata) = cache.charge(count.saturating_mul(384).saturating_add(256)) else {
            return false;
        };
        let Some(owner) = cache.new_node() else {
            return false;
        };
        let mut index = Children {
            layout_revision: node.layout_revision.get(),
            ids: Default::default(),
            scopes: Vec::new(),
            dirty: Default::default(),
            reasons: Vec::new(),
            instances: Vec::new(),
            live: Default::default(),
            owner,
            stride: 1,
            _metadata: metadata,
        };
        if !layout(container, &mut index, cache) {
            return false;
        }
        let mut stable = Vec::with_capacity(index.scopes.len());
        let mut live = Vec::new();
        index.reasons.resize(index.scopes.len(), Change::empty());
        ruffle_render::profile::work(|w| w.child_slots_examined += count as u64);
        for (slot, range) in index.scopes.iter().enumerate() {
            if range.len() == 1
                && let Some((old, root)) = old.as_mut()
            {
                let id = container
                    .child_by_index(range.start)
                    .unwrap()
                    .base()
                    .snapshots
                    .id();
                if let Some(&prior) = old.ids.get(&id)
                    && old.scopes[prior].len() == 1
                    && !old.live.contains(&prior)
                    && old.reasons[prior].is_empty()
                {
                    stable.push(slot_at(root, old.stride, prior));
                    index.instances.push(old.instances[prior].take());
                    continue;
                }
            }
            let (block, reusable) = capture(container, range.clone(), context);
            let instance = if reusable && range.len() == 1 {
                Instance::from_draw(
                    container.child_by_index(range.start).unwrap(),
                    &block,
                    &key.inherited,
                )
            } else {
                None
            };
            index.instances.push(instance);
            if reusable {
                stable.push(slot_command(block));
            } else {
                stable.push(Command::Shared(empty.clone()));
                live.push((slot, SlotEdit::Draw(slot_command(block))));
                index.live.insert(slot);
            }
        }
        let (root, stride) = pack(stable);
        index.stride = stride;
        let output = patch(&root, stride, live);
        (index, Some(root), output)
    };
    // A fully live painter order has no reusable geometry to hide its
    // recording cost. Stream it directly until an edit or a periodic probe.
    if index.live.len() == index.scopes.len() && stable_root.is_some() {
        node.live_until.set(
            context
                .render_epoch
                .saturating_add(super::LIVE_RETRY_FRAMES),
        );
        context.commands.push(Command::Shared(output));
        return true;
    }
    let cache = context.snapshots.as_deref_mut().unwrap();
    let stored = if let Some(commands) = stable_root {
        index.owner.entry.borrow_mut().take();
        ruffle_render::profile::work(|w| w.render_chunks_rebuilt += 1);
        cache.insert(&index.owner, Entry { key, commands })
    } else {
        cache.touch(&index.owner);
        true
    };
    context.commands.push(Command::Shared(output));
    if stored {
        *node.children.borrow_mut() = Some(index);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::display_object::{Graphic, MovieClip, StageScaleMode, Video};
    use crate::player::PlayerBuilder;

    thread_local! {
        pub(super) static INSTANCE_MATRICES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    #[test]
    fn alpha_only_edits_keep_leaf_matrices_and_coverage() {
        let movie = crate::tag_utils::SwfMovieData::from_data(
            include_bytes!("../../../../tests/tests/swfs/native_perf/static_tree/test.swf"),
            "file:///colors.swf".into(),
            None,
            None,
        )
        .unwrap();
        let player = PlayerBuilder::new()
            .with_movie(movie)
            .with_viewport_dimensions(1280, 720, 1.)
            .with_autoplay(true)
            .build();
        let mut player = player.lock().unwrap();
        assert!(player.preload(&mut crate::limits::ExecutionLimit::none()));
        player.run_frame();
        player.render().unwrap();
        fn group<'gc>(root: DisplayObject<'gc>) -> Option<DisplayObjectContainer<'gc>> {
            let c = root.as_container()?;
            if c.num_children() == 40 {
                return Some(c);
            }
            c.iter_render_list().find_map(group)
        }
        let sample = |ctx: &mut crate::context::UpdateContext<'_>| {
            let c = group(ctx.stage.into()).unwrap();
            let node = DisplayObject::from(c)
                .base()
                .snapshots
                .node
                .borrow()
                .clone()
                .unwrap();
            let index = node.children.borrow();
            let index = index.as_ref().unwrap();
            let entry = index.owner.entry.borrow();
            let root = &entry.as_ref().unwrap().commands;
            let Command::Shared(page) = &root.iter().as_slice()[0] else {
                panic!("missing leaf page")
            };
            let Command::RenderShape { transform, .. } = &page.iter().as_slice()[0] else {
                panic!("missing leaf draw")
            };
            (transform.clone(), page.command_region(0))
        };
        let (before, region) = player.mutate_with_update_context(sample);
        for inherited in [false, true] {
            player.mutate_with_update_context(|ctx| {
                let g = group(ctx.stage.into()).unwrap();
                let object = if inherited {
                    g.into()
                } else {
                    g.child_by_index(0).unwrap()
                };
                object.set_alpha(0.5);
            });
            INSTANCE_MATRICES.set(0);
            player.render().unwrap();
            let (after, coverage) = player.mutate_with_update_context(sample);
            assert_eq!(after.matrix, before.matrix);
            assert_eq!(coverage, region);
            assert_eq!(
                after.color_transform.a_multiply.to_f32(),
                if inherited { 0.25 } else { 0.5 }
            );
            assert_eq!(
                INSTANCE_MATRICES.get(),
                0,
                "color change rebuilt a world matrix"
            );
        }
    }

    #[test]
    fn first_draw_of_untessellated_timeline_children_streams_their_order_once() {
        use ruffle_render::profile::{RenderMetrics, enter};
        let movie = crate::tag_utils::SwfMovieData::from_data(
            include_bytes!("../../../../tests/tests/swfs/native_perf/load_unload/child.swf"),
            "file:///timeline-child.swf".into(),
            None,
            None,
        )
        .unwrap();
        let player = PlayerBuilder::new()
            .with_movie(movie)
            .with_viewport_dimensions(1280, 720, 1.)
            .with_autoplay(true)
            .build();
        let mut player = player.lock().unwrap();
        assert!(player.preload(&mut crate::limits::ExecutionLimit::none()));
        player.run_frame();
        let metrics = RenderMetrics::default();
        let _scope = enter(Some(&metrics));
        player.render().unwrap();
        let first = metrics.snapshot().work;
        assert!(first.tree_visits >= 40);
        assert!(
            first.child_slots_examined <= first.tree_visits + 12,
            "visited {} tree nodes, examined {} slots",
            first.tree_visits,
            first.child_slots_examined
        );
    }

    #[test]
    fn patching_one_draw_does_not_measure_its_stable_siblings() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        #[derive(Debug, Default)]
        struct Shape(AtomicUsize);
        impl ShapeHandleImpl for Shape {
            fn bounds(&self) -> Option<[f64; 4]> {
                self.0.fetch_add(1, Ordering::Relaxed);
                Some([0., 0., 1., 1.])
            }
        }
        let shape = Arc::new(Shape::default());
        let command = Command::RenderShape {
            shape: ShapeHandle(shape.clone()),
            transform: Default::default(),
        };
        let (root, stride) = pack(vec![command; 66]);
        shape.0.store(0, Ordering::Relaxed);
        let replacement = Command::DrawRect {
            color: swf::Color::WHITE,
            matrix: Default::default(),
        };
        let updated = patch(
            &root,
            stride,
            vec![
                (1, SlotEdit::Draw(replacement)),
                (17, SlotEdit::Color(Default::default())),
            ],
        );
        assert_eq!(root.region(), updated.region());
        assert_eq!(shape.0.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn wholly_live_video_children_are_examined_once_per_draw() {
        use ruffle_render::profile::{RenderMetrics, enter};
        let examined = |count| {
            let player = PlayerBuilder::new()
                .with_viewport_dimensions(128, 128, 1.)
                .build();
            let mut player = player.lock().unwrap();
            player.mutate_with_update_context(|ctx| {
                let root = MovieClip::new(ctx.stage.movie(), ctx.gc());
                for depth in 1..=count {
                    let video = Video::new(ctx.gc(), ctx.stage.movie(), 32, 32, None);
                    root.replace_at_depth(ctx, video.into(), depth);
                }
                ctx.stage.replace_at_depth(ctx, root.into(), 1);
            });
            let metrics = RenderMetrics::default();
            let _scope = enter(Some(&metrics));
            player.render().unwrap();
            let first = metrics.snapshot().work;
            player.render().unwrap();
            (
                first.child_slots_examined,
                metrics.snapshot().work.delta(first).child_slots_examined,
            )
        };
        let small = examined(16);
        let large = examined(32);
        assert_eq!((large.0 - small.0, large.1 - small.1), (16, 16));
    }

    #[test]
    fn simple_draws_share_one_allocation_per_leaf_page() {
        let movie = crate::tag_utils::SwfMovieData::from_data(
            include_bytes!("../../../../tests/tests/swfs/native_perf/static_tree/test.swf"),
            "file:///chunks.swf".into(),
            None,
            None,
        )
        .unwrap();
        let player = PlayerBuilder::new()
            .with_movie(movie)
            .with_autoplay(true)
            .with_viewport_dimensions(1280, 720, 1.0)
            .build();
        let mut player = player.lock().unwrap();
        assert!(player.preload(&mut crate::limits::ExecutionLimit::none()));
        player.run_frame();
        player.render().unwrap();
        player.mutate_with_update_context(|ctx| {
            let mut stack = vec![DisplayObject::from(ctx.stage)];
            while let Some(object) = stack.pop() {
                if let Some(c) = object.as_container() {
                    if c.num_children() == 40 {
                        let node = object.base().snapshots.node.borrow().clone().unwrap();
                        let index = node.children.borrow();
                        let entry = index.as_ref().unwrap().owner.entry.borrow();
                        let root = &entry.as_ref().unwrap().commands;
                        let Command::Shared(page) = &root.iter().as_slice()[0] else {
                            panic!("expected page")
                        };
                        assert!(
                            page.iter()
                                .all(|c| matches!(c, Command::RenderShape { .. }))
                        );
                        return;
                    }
                    stack.extend(c.iter_render_list());
                }
            }
            panic!("fixture group missing");
        });
    }

    #[test]
    fn empty_video_remains_live_until_its_first_decoded_frame() {
        let player = PlayerBuilder::new()
            .with_viewport_dimensions(128, 128, 1.0)
            .with_scale_mode(StageScaleMode::NoScale, true)
            .build();
        let mut player = player.lock().unwrap();
        player.mutate_with_update_context(|ctx| {
            let root = MovieClip::new(ctx.stage.movie(), ctx.gc());
            let video = Video::new(ctx.gc(), ctx.stage.movie(), 32, 32, None);
            root.replace_at_depth(ctx, video.into(), 1);
            for i in 2..=16 {
                let g = Graphic::empty(ctx);
                root.replace_at_depth(ctx, g.into(), i);
            }
            ctx.stage.replace_at_depth(ctx, root.into(), 1);
        });
        player.render().unwrap();
        player.mutate_with_update_context(|ctx| {
            let root = ctx.stage.child_by_depth(1).unwrap();
            let node = root.base().snapshots.node.borrow().clone().unwrap();
            assert!(
                node.live_until.get() > 0
                    || node
                        .children
                        .borrow()
                        .as_ref()
                        .is_some_and(|index| index.live.contains(&0))
            );
        });
    }

    #[test]
    fn emptying_a_wide_container_releases_its_index_before_the_next_frame() {
        let player = PlayerBuilder::new()
            .with_viewport_dimensions(128, 128, 1.0)
            .with_scale_mode(StageScaleMode::NoScale, true)
            .build();
        let mut player = player.lock().unwrap();
        player.mutate_with_update_context(|ctx| {
            let root = MovieClip::new(ctx.stage.movie(), ctx.gc());
            for i in 1..=256 {
                let g = Graphic::empty(ctx);
                root.replace_at_depth(ctx, g.into(), i);
            }
            ctx.stage.replace_at_depth(ctx, root.into(), 1);
        });
        player.render().unwrap();
        player.mutate_with_update_context(|ctx| {
            let root = ctx.stage.child_by_depth(1).unwrap();
            let node = root.base().snapshots.node.borrow().clone().unwrap();
            assert!(node.children.borrow().is_some());
            let container = root.as_container().unwrap();
            for child in container.iter_render_list() {
                container.remove_child_directly(ctx, child);
            }
            assert!(node.children.borrow().is_none());
        });
    }
}
