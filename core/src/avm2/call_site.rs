//! GC-traced, weakly guarded lookup caches, separate from immutable bytecode.
use super::{
    property::Property,
    vtable::{VTable, WeakVTable},
};
use gc_arena::{Collect, Gc, GcWeak, Mutation, lock::RefLock};
use std::{cell::Cell, rc::Rc};

#[derive(Clone, Copy, Collect)]
#[collect(no_drop)]
struct Entry<'gc> {
    table: WeakVTable<'gc>,
    property: Property,
}

#[derive(Default, Collect)]
#[collect(no_drop)]
pub struct Site<'gc> {
    entries: [Option<Entry<'gc>>; 4],
    pub megamorphic: bool,
}

impl<'gc> Site<'gc> {
    pub fn get(&self, table: VTable<'gc>) -> Option<Property> {
        self.entries
            .iter()
            .flatten()
            .find(|e| e.table.matches(table))
            .map(|e| e.property)
    }
    pub fn insert(&mut self, table: VTable<'gc>, property: Property, mc: &Mutation<'gc>) {
        if self.megamorphic {
            return;
        }
        if let Some(slot) = self
            .entries
            .iter_mut()
            .find(|slot| slot.is_none_or(|e| e.table.upgrade(mc).is_none()))
        {
            *slot = Some(Entry {
                table: table.downgrade(),
                property,
            });
        } else {
            self.megamorphic = true;
            self.entries = [None; 4];
        }
    }
}

// Static accounting token. It contains no GC pointer and refunds storage when
// a method is collected as well as when a live table is evicted.
struct Charge {
    used: Rc<Cell<usize>>,
    bytes: usize,
}
impl Drop for Charge {
    fn drop(&mut self) {
        self.used.set(self.used.get() - self.bytes);
    }
}

#[derive(Default, Collect)]
#[collect(no_drop)]
pub struct Data<'gc> {
    pub sites: Box<[Site<'gc>]>,
    #[collect(require_static)]
    _charge: Option<Charge>,
}

#[derive(Default, Collect)]
#[collect(no_drop)]
pub struct Table<'gc> {
    pub data: RefLock<Data<'gc>>,
    used_at: Cell<u64>,
    // Method's OnceLock keeps this allocation after its sites are evicted.
    #[collect(require_static)]
    _metadata_charge: Option<Charge>,
}

impl<'gc> Table<'gc> {
    fn clear(this: Gc<'gc, Self>, mc: &Mutation<'gc>) {
        gc_arena::barrier::unlock!(Gc::write(mc, this), Self, data).replace(Data::default());
    }
}

#[derive(Collect)]
#[collect(no_drop)]
pub struct Budget<'gc> {
    tables: Vec<GcWeak<'gc, Table<'gc>>>,
    #[collect(require_static)]
    used: Rc<Cell<usize>>,
    limit: usize,
    clock: u64,
}

impl Default for Budget<'_> {
    fn default() -> Self {
        Self {
            tables: Vec::new(),
            used: Rc::new(Cell::new(0)),
            limit: 8 * 1024 * 1024,
            clock: 0,
        }
    }
}

impl<'gc> Budget<'gc> {
    fn metadata_bytes() -> usize {
        // GC header, allocator overhead, weak registry and its spare capacity.
        std::mem::size_of::<Table<'gc>>() + 512
    }
    fn site_bytes(count: usize) -> Option<usize> {
        count
            .checked_mul(std::mem::size_of::<Site<'gc>>())?
            .checked_add(64)
    }
    pub fn new_table(&mut self, count: usize, mc: &Mutation<'gc>) -> Option<Gc<'gc, Table<'gc>>> {
        let bytes = Self::metadata_bytes().checked_add(Self::site_bytes(count)?)?;
        if !self.make_room(bytes, mc) {
            return None;
        }
        let metadata = Self::metadata_bytes();
        self.used.set(self.used.get() + metadata);
        let table = Gc::new(
            mc,
            Table {
                _metadata_charge: Some(Charge {
                    used: self.used.clone(),
                    bytes: metadata,
                }),
                ..Default::default()
            },
        );
        self.tables.push(Gc::downgrade(table));
        // Space for both lifetimes was reserved before allocating either one.
        assert!(self.allocate(table, count, mc));
        Some(table)
    }
    pub fn touch(&mut self, table: Gc<'gc, Table<'gc>>) {
        self.clock = self.clock.wrapping_add(1);
        table.used_at.set(self.clock);
    }
    pub fn allocate(
        &mut self,
        table: Gc<'gc, Table<'gc>>,
        count: usize,
        mc: &Mutation<'gc>,
    ) -> bool {
        let Some(bytes) = Self::site_bytes(count) else {
            return false;
        };
        if !self.make_room(bytes, mc) {
            return false;
        }
        self.used.set(self.used.get() + bytes);
        let data = Data {
            sites: (0..count).map(|_| Site::default()).collect(),
            _charge: Some(Charge {
                used: self.used.clone(),
                bytes,
            }),
        };
        gc_arena::barrier::unlock!(Gc::write(mc, table), Table, data).replace(data);
        self.touch(table);
        true
    }
    fn make_room(&mut self, bytes: usize, mc: &Mutation<'gc>) -> bool {
        if bytes > self.limit {
            return false;
        }
        self.tables.retain(|w| w.upgrade(mc).is_some());
        if self.tables.capacity() > self.tables.len().saturating_mul(2) {
            self.tables.shrink_to_fit();
        }
        while self.used.get() + bytes > self.limit {
            let oldest = self
                .tables
                .iter()
                .enumerate()
                .filter_map(|(i, w)| {
                    w.upgrade(mc)
                        .filter(|t| !t.data.borrow().sites.is_empty())
                        .map(|t| (i, t))
                })
                .min_by_key(|(_, t)| t.used_at.get());
            let Some((_, old)) = oldest else {
                return false;
            };
            Table::clear(old, mc);
            // Keep the weak registry and lifetime charge for the empty table.
        }
        true
    }
}

#[derive(Default)]
pub struct Counters {
    pub hits: u64,
    pub misses: u64,
    pub megamorphic: u64,
}
impl Drop for Counters {
    fn drop(&mut self) {
        if self.hits | self.misses | self.megamorphic != 0 {
            ruffle_render::profile::work(|w| {
                w.pic_hits += self.hits;
                w.pic_misses += self.misses;
                w.pic_megamorphic += self.megamorphic;
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gc_arena::{Arena, Rootable};
    #[test]
    fn dead_table_registry_does_not_keep_its_peak_capacity() {
        let mut arena = Arena::<Rootable![(Budget<'_>, Vec<Gc<'_, Table<'_>>>)]>::new(|mc| {
            let mut budget = Budget::default();
            let tables = (0..500).map(|_| budget.new_table(1, mc).unwrap()).collect();
            (budget, tables)
        });
        arena.mutate_root(|_, (_, tables)| tables.clear());
        arena.finish_cycle();
        arena.mutate_root(|mc, (budget, tables)| {
            tables.push(budget.new_table(1, mc).unwrap());
            assert!(
                budget.tables.capacity() <= 4,
                "retained {} registry slots for one table",
                budget.tables.capacity()
            );
        });
    }

    #[test]
    fn live_evicted_tables_keep_their_metadata_charged() {
        let mut arena = Arena::<Rootable![(Budget<'_>, Vec<Gc<'_, Table<'_>>>)]>::new(|mc| {
            let mut budget = Budget {
                limit: 4096,
                ..Default::default()
            };
            let mut tables = Vec::new();
            for _ in 0..200 {
                if let Some(table) = budget.new_table(1, mc) {
                    tables.push(table);
                }
            }
            (budget, tables)
        });
        arena.mutate_root(|mc, (budget, tables)| {
            let metadata = tables.len() * std::mem::size_of::<Table>();
            assert!(
                budget.used.get() >= metadata,
                "only {} bytes charged for {metadata} bytes of live tables",
                budget.used.get()
            );
            assert!(budget.used.get() <= budget.limit);
            assert!(budget.new_table(usize::MAX, mc).is_none());
        });
    }
    #[test]
    fn weak_guards_do_not_retain_classes_or_match_new_allocations() {
        let mut arena = Arena::<Rootable![Site<'_>]>::new(|mc| {
            let mut site = Site::default();
            let table = VTable::empty(mc);
            site.insert(table, Property::Slot { slot_id: 17 }, mc);
            assert!(matches!(
                site.get(table),
                Some(Property::Slot { slot_id: 17 })
            ));
            site
        });
        arena.finish_cycle();
        arena.mutate_root(|mc, site| {
            assert!(site.entries[0].unwrap().table.upgrade(mc).is_none());
            for _ in 0..1024 {
                assert!(site.get(VTable::empty(mc)).is_none());
            }
        });
    }
    #[test]
    fn lru_eviction_and_gc_refund_the_shared_budget() {
        let used = Rc::new(Cell::new(0));
        let mut arena = Arena::<Rootable![(Budget<'_>, Vec<Gc<'_, Table<'_>>>)]>::new(|mc| {
            let mut budget = Budget {
                used: used.clone(),
                limit: 3 * Budget::metadata_bytes() + 2 * Budget::site_bytes(1).unwrap(),
                ..Default::default()
            };
            let mut tables = vec![
                budget.new_table(1, mc).unwrap(),
                budget.new_table(1, mc).unwrap(),
            ];
            budget.touch(tables[0]);
            tables.push(budget.new_table(1, mc).unwrap());
            assert!(tables[1].data.borrow().sites.is_empty());
            assert!(!tables[0].data.borrow().sites.is_empty());
            assert!(used.get() <= budget.limit);
            (budget, tables)
        });
        arena.mutate_root(|_, root| root.1.clear());
        arena.finish_cycle();
        assert_eq!(used.get(), 0);
    }
}
