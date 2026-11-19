//! Stage command snapshots. No GC edges are retained; display objects own the
//! entries and the player keeps only weak references for accounting/eviction.
pub(super) mod bounds;
pub(super) mod children;
pub(super) mod stencil;
use ruffle_render::{commands::CommandList, quality::StageQuality, transform::Transform};
use std::{
    cell::{Cell, RefCell},
    rc::{Rc, Weak},
};
use swf::{Rectangle, Twips};

bitflags::bitflags! {
    #[derive(Clone, Copy, Default, PartialEq, Eq)]
    pub(super) struct Change: u8 {
        const TRANSFORM = 1;
        const COLOR = 2;
        const VISUAL = 4;
        const MASK = 8;
        const FILTER = 16;
        const VISIBILITY = 32;
        const CHILDREN = 64;
    }
}

#[derive(Default)]
struct RenderGeneration {
    // Source stamps are allocated with an observed node. World transforms
    // incorporate inherited stamps at use time.
    revisions: [Cell<u64>; 7],
}
impl RenderGeneration {
    fn advance(&self, change: Change) {
        for flag in change.iter() {
            let revision = &self.revisions[flag.bits().trailing_zeros() as usize];
            revision.set(revision.get().wrapping_add(1));
        }
    }
}

#[derive(Clone, PartialEq)]
pub struct Key {
    pub mask_geometry: bool,
    pub generation: u64,
    pub inherited: Transform,
    pub own: Transform,
    pub viewport: (u32, u32, u64),
    pub view: Rectangle<Twips>,
    pub quality: StageQuality,
}

pub struct Entry {
    pub key: Key,
    pub commands: Rc<CommandList>,
}

#[derive(Default)]
pub struct Node {
    layout_revision: Cell<u64>,
    revisions: RenderGeneration,
    children: RefCell<Option<children::Children>>,
    dense: Cell<bool>,
    live_until: Cell<u64>,
    dirty_children: Cell<usize>,
    pub entry: RefCell<Option<Entry>>,
    pub used: Cell<u64>,
    pub observed: RefCell<Option<Key>>,
    _charge: Option<NodeCharge>,
}

struct NodeCharge {
    total: Rc<Cell<usize>>,
    used: Rc<Cell<usize>>,
    bytes: usize,
}
impl Drop for NodeCharge {
    fn drop(&mut self) {
        self.used.set(self.used.get() - self.bytes);
        self.total.set(self.total.get() - self.bytes);
    }
}

thread_local! { static NEXT_ID: Cell<u64> = const { Cell::new(1) }; }

#[derive(Default)]
pub struct State {
    spatial: RefCell<Option<Box<bounds::Spatial>>>,
    id: Cell<u64>,
    pub(super) transform_stamp: Cell<u64>,
    pub(super) bounds_observed: Cell<bool>,
    pub generation: Cell<u64>,
    pub node: RefCell<Option<Rc<Node>>>,
    invalidated: Cell<bool>,
}

// Instantiated display objects must never share a cache or invalidation state.
impl Clone for State {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl State {
    /// Coalesce until the next render traversal, not until the next frame.
    /// A second mutation after a render in the same logical frame propagates.
    pub(super) fn id(&self) -> u64 {
        let id = self.id.get();
        if id != 0 {
            return id;
        }
        let id = NEXT_ID.get();
        NEXT_ID.set(
            id.checked_add(1)
                .expect("display object identity exhausted"),
        );
        self.id.set(id);
        id
    }
    pub(super) fn observed_id(&self) -> u64 {
        self.id.get()
    }
    pub(super) fn revision(&self, component: Change) -> u64 {
        debug_assert_eq!(component.bits().count_ones(), 1);
        self.node.borrow().as_ref().map_or(0, |node| {
            node.revisions.revisions[component.bits().trailing_zeros() as usize].get()
        })
    }
    pub(super) fn changed(&self, change: Change) {
        if change.contains(Change::TRANSFORM) {
            self.transform_stamp.set(bounds::next_transform_stamp());
        }
        if let Some(node) = self.node.borrow().as_ref() {
            node.revisions.advance(change);
            if change.contains(Change::CHILDREN) {
                node.live_until.set(0);
            }
        }
    }
    pub(super) fn invalidate_bounds(&self) -> bool {
        if !self.bounds_observed.replace(false) {
            return false;
        }
        if let Some(space) = self.spatial.borrow().as_ref() {
            space.clear_bounds();
        }
        true
    }
    pub(super) fn mark_child(&self, id: u64, change: Change, first: bool) {
        if let Some(node) = self.node.borrow().as_ref() {
            if node.live_until.get() != 0
                && !(change - (Change::TRANSFORM | Change::COLOR)).is_empty()
            {
                node.live_until.set(0);
            }
            if first && node.dense.get() {
                node.dirty_children
                    .set(node.dirty_children.get().saturating_add(1));
            }
            if let Some(children) = node.children.borrow_mut().as_mut() {
                children.mark(id, change);
            }
        }
    }
    pub(super) fn invalidate_child_layout(&self) {
        if let Some(node) = self.node.borrow().as_ref() {
            node.layout_revision
                .set(node.layout_revision.get().wrapping_add(1));
        }
    }

    pub fn invalidate(&self) -> bool {
        if self.invalidated.replace(true) {
            return false;
        }
        self.generation.set(self.generation.get().wrapping_add(1));
        if let Some(node) = self.node.borrow().as_ref() {
            node.entry.borrow_mut().take();
        }
        true
    }
    pub fn visited(&self) {
        self.invalidated.set(false);
    }
    pub fn clear(&self) {
        self.invalidate();
        self.node.borrow_mut().take();
        self.spatial.borrow_mut().take();
    }
    fn observe_spatial(&self, cache: &mut Cache) {
        let mut space = self.spatial.borrow_mut();
        if space.is_none()
            && let Some(charge) = cache.charge(size_of::<bounds::Spatial>() + 32)
        {
            *space = Some(Box::new(bounds::Spatial::new(charge)));
        }
    }
    pub fn node(&self, cache: &mut Cache) -> Option<Rc<Node>> {
        let mut node = self.node.borrow_mut();
        if node.is_none() {
            *node = Some(cache.new_node()?);
            self.observe_spatial(cache);
        }
        node.clone()
    }
}

struct Record {
    node: Weak<Node>,
    commands: Weak<CommandList>,
    bytes: usize,
}

// Weak<Node> keeps the Rc allocation, including Node's inline storage, alive
// after its last strong owner drops. Charge that lifetime with the block too.
const ENTRY_OVERHEAD: usize = std::mem::size_of::<Node>() + 2 * std::mem::size_of::<Record>() + 256;

pub struct Cache {
    empty: std::cell::OnceCell<Rc<CommandList>>,
    records: Vec<Record>,
    limit: usize,
    bytes: usize,
    node_bytes: Rc<Cell<usize>>,
    used_bytes: Rc<Cell<usize>>,
    clock: u64,
}

impl Default for Cache {
    fn default() -> Self {
        Self::new(24 * 1024 * 1024)
    }
}

impl Cache {
    fn new(limit: usize) -> Self {
        Self {
            empty: Default::default(),
            records: Vec::new(),
            limit,
            bytes: 0,
            node_bytes: Rc::new(Cell::new(0)),
            used_bytes: Rc::new(Cell::new(0)),
            clock: 0,
        }
    }
    fn empty(&self) -> Rc<CommandList> {
        self.empty.get_or_init(Default::default).clone()
    }
    pub fn touch(&mut self, node: &Node) {
        self.clock = self.clock.wrapping_add(1);
        node.used.set(self.clock);
    }
    pub fn prune(&mut self) {
        self.records.retain(|r| {
            if r.commands.strong_count() == 0 {
                self.bytes -= r.bytes;
                self.used_bytes.set(self.used_bytes.get() - r.bytes);
                false
            } else {
                true
            }
        });
        // Entry charges cover spare capacity up to twice the live record count.
        // Do not retain an old large registry alongside a new large block.
        if self.records.capacity() > self.records.len().saturating_mul(2) {
            self.records.shrink_to_fit();
        }
    }
    fn charge(&mut self, bytes: usize) -> Option<NodeCharge> {
        if !self.make_room(bytes) {
            return None;
        }
        self.node_bytes.set(self.node_bytes.get() + bytes);
        self.used_bytes.set(self.used_bytes.get() + bytes);
        Some(NodeCharge {
            total: self.used_bytes.clone(),
            used: self.node_bytes.clone(),
            bytes,
        })
    }
    fn new_node(&mut self) -> Option<Rc<Node>> {
        // Includes Rc/allocator overhead; observation keys live inside Node.
        let bytes = std::mem::size_of::<Node>() + 128;
        let charge = self.charge(bytes)?;
        Some(Rc::new(Node {
            _charge: Some(charge),
            ..Default::default()
        }))
    }
    fn make_room(&mut self, bytes: usize) -> bool {
        if bytes > self.limit {
            return false;
        }
        if self.used_bytes.get() + bytes <= self.limit {
            return true;
        }
        self.prune();
        while self.used_bytes.get() + bytes > self.limit {
            let oldest = self
                .records
                .iter()
                .filter_map(|r| {
                    let node = r.node.upgrade()?;
                    let owns = node
                        .entry
                        .borrow()
                        .as_ref()
                        .is_some_and(|e| Rc::as_ptr(&e.commands) == r.commands.as_ptr());
                    owns.then_some(node)
                })
                .min_by_key(|n| n.used.get());
            let Some(oldest) = oldest else {
                return false;
            };
            oldest.entry.borrow_mut().take();
            ruffle_render::profile::work(|w| w.snapshot_evictions += 1);
            // A parent block or this frame may still pin the evicted block.
            // Keep charging it until the LAST strong reference disappears.
            self.prune();
        }
        true
    }
    pub fn insert(&mut self, node: &Rc<Node>, entry: Entry) -> bool {
        // Charge vector spare capacity and weak/control-block overhead as well
        // as command/resource storage. Candidate Node has a separate lifetime.
        let Some(bytes) = entry
            .commands
            .additional_bytes(&self.used_bytes)
            .and_then(|b| b.checked_add(ENTRY_OVERHEAD))
        else {
            return false;
        };
        if !self.make_room(bytes) {
            return false;
        }
        self.touch(node);
        entry.commands.charge_storage(&self.used_bytes, self.limit);
        self.used_bytes.set(self.used_bytes.get() + ENTRY_OVERHEAD);
        self.bytes += ENTRY_OVERHEAD;
        self.records.push(Record {
            node: Rc::downgrade(node),
            commands: Rc::downgrade(&entry.commands),
            bytes: ENTRY_OVERHEAD,
        });
        *node.entry.borrow_mut() = Some(entry);
        // Cumulative admitted bytes (not a live-size gauge); delta-safe.
        ruffle_render::profile::work(|w| w.snapshot_bytes += bytes as u64);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn entry() -> Entry {
        Entry {
            key: Key {
                mask_geometry: false,
                generation: 0,
                inherited: Transform::default(),
                own: Transform::default(),
                viewport: (1, 1, 0),
                view: Rectangle::default(),
                quality: StageQuality::Low,
            },
            commands: Rc::new(CommandList::new()),
        }
    }
    #[test]
    fn pinned_blocks_charge_weak_node_storage_and_prune_releases_registry_capacity() {
        let mut cache = Cache::default();
        let mut frames = Vec::new();
        for _ in 0..65 {
            let state = State::default();
            let node = state.node(&mut cache).unwrap();
            let block = entry();
            frames.push(block.commands.clone());
            assert!(cache.insert(&node, block));
            state.clear();
        }
        assert_eq!(cache.node_bytes.get(), 0);
        let actual = frames
            .iter()
            .map(|c| c.retained_bytes().unwrap())
            .sum::<usize>()
            + cache.records.capacity() * std::mem::size_of::<Record>()
            + frames.len() * (std::mem::size_of::<Node>() + 4 * std::mem::size_of::<usize>());
        assert!(
            cache.used_bytes.get() >= actual,
            "{} charged for at least {actual} retained bytes",
            cache.bytes
        );
        frames.clear();
        cache.prune();
        assert_eq!(cache.records.capacity(), 0);
    }

    #[test]
    fn rejected_candidates_cannot_retain_unbounded_metadata() {
        let mut cache = Cache::new(4096);
        let states: Vec<_> = (0..200).map(|_| State::default()).collect();
        for state in &states {
            let Some(node) = state.node(&mut cache) else {
                continue;
            };
            let mut block = entry();
            let mut commands = CommandList::new();
            for _ in 0..1000 {
                commands.push(ruffle_render::commands::Command::DrawRect {
                    color: swf::Color::RED,
                    matrix: Default::default(),
                });
            }
            block.commands = Rc::new(commands);
            assert!(!cache.insert(&node, block));
        }
        let metadata = states.iter().filter(|s| s.node.borrow().is_some()).count()
            * std::mem::size_of::<Node>();
        assert!(
            metadata <= cache.limit,
            "metadata {metadata} exceeds {}",
            cache.limit
        );
        assert!(cache.node_bytes.get() >= metadata);
        // Removed objects may remain referenced by AS, but their cache state
        // must release its observation allocation, not just its commands.
        for state in &states {
            state.clear();
        }
        assert_eq!(cache.node_bytes.get(), 0);
    }
    #[test]
    fn cloned_display_object_state_does_not_share_cache_or_generation() {
        let state = State::default();
        let mut cache = Cache::default();
        let node = state.node(&mut cache).unwrap();
        *node.entry.borrow_mut() = Some(entry());
        let copy = state.clone();
        copy.invalidate();
        assert!(node.entry.borrow().is_some());
        assert!(copy.node.borrow().is_none());
        assert_eq!(state.generation.get(), 0);
    }

    #[test]
    fn eviction_keeps_charging_blocks_pinned_by_current_frame() {
        let cost = entry().commands.retained_bytes().unwrap() + ENTRY_OVERHEAD;
        let mut cache = Cache::new(cost);
        let a = Rc::new(Node::default());
        let b = Rc::new(Node::default());
        let first = entry();
        let frame = first.commands.clone();
        assert!(cache.insert(&a, first));
        assert!(!cache.insert(&b, entry()));
        assert_eq!(cache.used_bytes.get(), cost);
        drop(frame);
        assert!(cache.insert(&b, entry()));
        drop(b);
        cache.prune();
        assert_eq!(cache.bytes, 0);
    }
}
