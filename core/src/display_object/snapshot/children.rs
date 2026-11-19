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
    ) -> Option<Block> {
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
        let mut list = CommandList::new();
        if self.visible
            && (!self.graphic
                || child
                    .world_bounds(super::super::BoundsMode::Engine)
                    .intersects(&context.stage.view_bounds()))
        {
            list.push(Command::RenderShape {
                shape: ShapeHandle(self.shape.upgrade()?),
                transform: world,
            });
        }
        ruffle_render::profile::work(|w| w.render_instances_updated += 1);
        Some(Rc::new(list))
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

impl Children {
    pub(super) fn mark(&mut self, child: u64, reason: Change) {
        if let Some(index) = self.ids.get(&child) {
            self.reasons[*index] |= reason;
            self.dirty.insert(*index);
        }
    }
}

fn join(items: impl IntoIterator<Item = Block>) -> Block {
    let mut list = CommandList::new();
    for item in items {
        list.push(Command::Shared(item));
    }
    Rc::new(list)
}

fn pack(mut level: Vec<Block>) -> (Block, usize) {
    let mut stride = 1;
    while level.len() > WIDTH {
        level = level
            .chunks(WIDTH)
            .map(|group| join(group.iter().cloned()))
            .collect();
        stride *= WIDTH;
    }
    (join(level), stride)
}

// Each patch copies at most WIDTH command references per affected tree level.
fn patch(root: &Block, stride: usize, offset: usize, changes: &[(usize, Block)]) -> Block {
    if changes.is_empty() {
        return root.clone();
    }
    let mut rest = changes;
    join(root.iter().enumerate().map(|(i, command)| {
        let Command::Shared(child) = command else {
            unreachable!("invalid chunk directory")
        };
        let start = offset + i * stride;
        let n = rest.partition_point(|(slot, _)| *slot < start + stride);
        let (here, tail) = rest.split_at(n);
        rest = tail;
        if here.is_empty() {
            child.clone()
        } else if stride == 1 {
            here[0].1.clone()
        } else {
            patch(child, stride / WIDTH, start, here)
        }
    }))
}

fn slot_at(root: &Block, mut stride: usize, mut index: usize) -> Block {
    let mut node = root;
    loop {
        let Command::Shared(child) = &node.iter().as_slice()[index / stride] else {
            unreachable!("invalid chunk directory")
        };
        if stride == 1 {
            return child.clone();
        }
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
    let stable = before == context.snapshot_barriers && commands.retained_bytes().is_some();
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
    if count < 8
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
    let empty = Rc::new(CommandList::new());
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
        let mut stable_patches = Vec::new();
        let mut live_patches = Vec::new();
        ruffle_render::profile::work(|w| w.child_slots_examined += work.len() as u64);
        for slot in work {
            let scope = index.scopes[slot].clone();
            let reason = std::mem::take(&mut index.reasons[slot]);
            let child = container.child_by_index(scope.start).unwrap();
            let instance = index.instances[slot]
                .as_mut()
                .and_then(|i| i.update(child, reason, context));
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
                (block, stable)
            };
            if stable {
                index.live.remove(&slot);
                stable_patches.push((slot, block));
            } else {
                if index.live.insert(slot) {
                    stable_patches.push((slot, empty.clone()));
                }
                live_patches.push((slot, block));
            }
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
                stable.push(block);
            } else {
                stable.push(empty.clone());
                live.push((slot, block));
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
