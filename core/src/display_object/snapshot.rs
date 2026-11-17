//! Stage command snapshots. No GC edges are retained; display objects own the
//! entries and the player keeps only weak references for accounting/eviction.
use ruffle_render::{commands::CommandList, quality::StageQuality, transform::Transform};
use std::{
    cell::{Cell, RefCell},
    rc::{Rc, Weak},
};
use swf::{Rectangle, Twips};

#[derive(Clone, PartialEq)]
pub struct Key {
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
    pub entry: RefCell<Option<Entry>>,
    pub used: Cell<u64>,
    pub observed: RefCell<Option<Key>>,
    _charge: Option<NodeCharge>,
}

struct NodeCharge {
    used: Rc<Cell<usize>>,
    bytes: usize,
}
impl Drop for NodeCharge {
    fn drop(&mut self) {
        self.used.set(self.used.get() - self.bytes);
    }
}

#[derive(Default)]
pub struct State {
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
    }
    pub fn node(&self, cache: &mut Cache) -> Option<Rc<Node>> {
        let mut node = self.node.borrow_mut();
        if node.is_none() {
            *node = Some(cache.new_node()?);
        }
        node.clone()
    }
}

struct Record {
    node: Weak<Node>,
    commands: Weak<CommandList>,
    bytes: usize,
}

pub struct Cache {
    records: Vec<Record>,
    limit: usize,
    bytes: usize,
    node_bytes: Rc<Cell<usize>>,
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
            records: Vec::new(),
            limit,
            bytes: 0,
            node_bytes: Rc::new(Cell::new(0)),
            clock: 0,
        }
    }
    pub fn touch(&mut self, node: &Node) {
        self.clock = self.clock.wrapping_add(1);
        node.used.set(self.clock);
    }
    pub fn prune(&mut self) {
        self.records.retain(|r| {
            if r.commands.strong_count() == 0 {
                self.bytes -= r.bytes;
                false
            } else {
                true
            }
        });
    }
    fn new_node(&mut self) -> Option<Rc<Node>> {
        // Includes Rc/allocator overhead; observation keys live inside Node.
        let bytes = std::mem::size_of::<Node>() + 128;
        if !self.make_room(bytes) {
            return None;
        }
        self.node_bytes.set(self.node_bytes.get() + bytes);
        Some(Rc::new(Node {
            _charge: Some(NodeCharge {
                used: self.node_bytes.clone(),
                bytes,
            }),
            ..Default::default()
        }))
    }
    fn make_room(&mut self, bytes: usize) -> bool {
        if bytes > self.limit {
            return false;
        }
        self.prune();
        while self.bytes + self.node_bytes.get() + bytes > self.limit {
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
            .retained_bytes()
            .and_then(|b| b.checked_add(512))
        else {
            return false;
        };
        if !self.make_room(bytes) {
            return false;
        }
        self.touch(node);
        self.bytes += bytes;
        self.records.push(Record {
            node: Rc::downgrade(node),
            commands: Rc::downgrade(&entry.commands),
            bytes,
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
        let cost = entry().commands.retained_bytes().unwrap() + 512;
        let mut cache = Cache::new(cost);
        let a = Rc::new(Node::default());
        let b = Rc::new(Node::default());
        let first = entry();
        let frame = first.commands.clone();
        assert!(cache.insert(&a, first));
        assert!(!cache.insert(&b, entry()));
        assert_eq!(cache.bytes, cost);
        drop(frame);
        assert!(cache.insert(&b, entry()));
        drop(b);
        cache.prune();
        assert_eq!(cache.bytes, 0);
    }
}
