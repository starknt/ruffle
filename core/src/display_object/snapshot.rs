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
}

#[derive(Default)]
pub struct State {
    pub generation: Cell<u64>,
    pub node: RefCell<Option<Rc<Node>>>,
}

// Instantiated display objects must never share a cache or invalidation state.
impl Clone for State {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl State {
    pub fn invalidate(&self) {
        self.generation.set(self.generation.get().wrapping_add(1));
        if let Some(node) = self.node.borrow().as_ref() {
            node.entry.borrow_mut().take();
        }
    }
    pub fn node(&self) -> Rc<Node> {
        self.node
            .borrow_mut()
            .get_or_insert_with(Default::default)
            .clone()
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
    pub fn insert(&mut self, node: &Rc<Node>, entry: Entry) -> bool {
        let Some(bytes) = entry
            .commands
            .retained_bytes()
            .and_then(|b| b.checked_add(512))
        else {
            return false;
        };
        if bytes > self.limit {
            return false;
        }
        self.prune();
        while self.bytes + bytes > self.limit {
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
    fn cloned_display_object_state_does_not_share_cache_or_generation() {
        let state = State::default();
        let node = state.node();
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
