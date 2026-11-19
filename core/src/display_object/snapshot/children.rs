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

struct Instance {
    shape: std::sync::Weak<dyn ShapeHandleImpl>,
    local: Transform,
    basis: [f32; 4],
    graphic: bool,
    visible: bool,
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
            local,
            basis: [m.a, m.b, m.c, m.d],
            graphic,
            visible: child.visible(),
        })
    }
    fn update<'gc>(
        &mut self,
        child: DisplayObject<'gc>,
        reason: Change,
        context: &mut RenderContext<'_, 'gc>,
        empty: &Block,
    ) -> Option<Command> {
        if !(reason - (Change::TRANSFORM | Change::COLOR | Change::VISIBILITY)).is_empty() {
            return None;
        }
        if !reason.is_empty() {
            child.pre_render(context);
            self.local = child.base().transform(true);
            self.visible = child.visible();
        }
        let world = context.transform_stack.transform().concat(&self.local);
        let m = world.matrix;
        // Timeline shapes choose their tessellation using the linear matrix.
        // Script graphics use the Drawing's existing, scale-independent handle.
        if world.tz != 0.
            || self.local.perspective_projection.is_some()
            || self.graphic && self.basis != [m.a, m.b, m.c, m.d]
        {
            return None;
        }
        let command = if self.visible
            && (!self.graphic
                || child
                    .world_bounds(super::super::BoundsMode::Engine)
                    .intersects(&context.stage.view_bounds()))
        {
            Command::RenderShape {
                shape: ShapeHandle(self.shape.upgrade()?),
                transform: world,
            }
        } else {
            Command::Shared(empty.clone())
        };
        ruffle_render::profile::work(|w| w.render_instances_updated += 1);
        Some(command)
    }
}

pub(in crate::display_object) struct Children {
    layout_revision: u64,
    ids: FnvHashMap<u64, usize>,
    scopes: Vec<Range<usize>>,
    dirty: BTreeSet<usize>,
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
            self.reasons[*index] |= reason;
            self.dirty.insert(*index);
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

fn chunk(items: impl ExactSizeIterator<Item = Command>) -> Block {
    let mut list = CommandList::with_capacity(items.len());
    for item in items {
        list.push(item);
    }
    Rc::new(list)
}

fn pack(mut level: Vec<Command>) -> (Block, usize) {
    let mut stride = 1;
    while level.len() > WIDTH {
        let mut items = level.into_iter();
        let mut next = Vec::with_capacity(items.len().div_ceil(WIDTH));
        while items.len() != 0 {
            next.push(Command::Shared(chunk((&mut items).take(WIDTH))));
        }
        level = next;
        stride *= WIDTH;
    }
    (chunk(level.into_iter()), stride)
}

// Stable siblings keep their pages. Only pages intersecting a patch are copied.
fn patch(root: &Block, stride: usize, offset: usize, changes: &[(usize, Command)]) -> Block {
    if changes.is_empty() {
        return root.clone();
    }
    let mut rest = changes;
    chunk(root.iter().enumerate().map(|(i, command)| {
        let start = offset + i * stride;
        let n = rest.partition_point(|(slot, _)| *slot < start + stride);
        let (here, tail) = rest.split_at(n);
        rest = tail;
        if here.is_empty() {
            return command.clone();
        }
        if stride == 1 {
            return here[0].1.clone();
        }
        let Command::Shared(child) = command else {
            unreachable!("invalid chunk directory")
        };
        Command::Shared(patch(child, stride / WIDTH, start, here))
    }))
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

fn layout<'gc>(container: DisplayObjectContainer<'gc>, children: &mut Children) -> bool {
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
    context: &mut RenderContext<'_, 'gc>,
) -> bool {
    let count = container.num_children();
    if count < 2
        || !context.snapshot_allowed
        || context.is_offscreen
        || context.commands.drawing_mask()
    {
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
    if node.dense.get() && node.dirty_children.replace(0) > count / 2 {
        return false;
    }
    node.dense.set(false);
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
            .map(|entry| (entry.commands.clone(), entry.key.inherited != key.inherited))
    });
    let empty = cache.empty();
    let (index, stable_root, output) = if let Some((old, inherited_changed)) = previous {
        let mut index = index.take().unwrap();
        if count >= 128 && index.dirty.len() > index.scopes.len() / 2 {
            node.dense.set(true);
            return false;
        }
        let dirty = std::mem::take(&mut index.dirty);
        let work: Vec<_> = if inherited_changed {
            (0..index.scopes.len()).collect()
        } else {
            dirty.union(&index.live).copied().collect()
        };
        let mut stable_patches = Vec::with_capacity(work.len());
        let mut live_patches = Vec::new();
        ruffle_render::profile::work(|w| w.child_slots_examined += work.len() as u64);
        for slot in work {
            let scope = index.scopes[slot].clone();
            let reason = std::mem::take(&mut index.reasons[slot]);
            let child = container.child_by_index(scope.start).unwrap();
            let instance = index.instances[slot]
                .as_mut()
                .and_then(|i| i.update(child, reason, context, &empty));
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
                (slot_command(block), stable)
            };
            if stable {
                index.live.remove(&slot);
                stable_patches.push((slot, block));
            } else {
                if index.live.insert(slot) {
                    stable_patches.push((slot, Command::Shared(empty.clone())));
                }
                live_patches.push((slot, block));
            }
        }
        if stable_patches.is_empty() {
            ruffle_render::profile::work(|w| w.snapshot_hits += 1);
        }
        let root = patch(&old, index.stride, 0, &stable_patches);
        let output = patch(&root, index.stride, 0, &live_patches);
        let changed = inherited_changed || !Rc::ptr_eq(&root, &old);
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
        if !layout(container, &mut index) {
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
                    && !old.dirty.contains(&prior)
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
                live.push((slot, slot_command(block)));
                index.live.insert(slot);
            }
        }
        let (root, stride) = pack(stable);
        index.stride = stride;
        let output = patch(&root, stride, 0, &live);
        (index, Some(root), output)
    };
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
            assert!(node.children.borrow().as_ref().unwrap().live.contains(&0));
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
