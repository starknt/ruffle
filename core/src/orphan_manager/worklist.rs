//! Ordered phase worklists. Object addresses are opaque lookup keys only:
//! each is backed by the original snapshot's GcWeak, never dereferenced here.
//! No strong GC edges or collection scheduling changes are introduced.

use super::{DisplayObject, DisplayObjectWeak, TDisplayObject, valid_orphan};
use crate::context::UpdateContext;
use crate::display_object::button_state_revision;
use crate::frame_lifecycle::FramePhase;
use gc_arena::{Collect, Mutation};
use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

const BUDGET: usize = 64 * 1024 * 1024;
const MAX_WATCHERS: usize = 256;
const WATCHER_BYTES: usize = MAX_WATCHERS * std::mem::size_of::<Weak<State>>() + 1024;

thread_local! {
    // Usually one current snapshot. Reentrant member changes can keep older
    // snapshots alive; every one must hear invalidations until its walk ends.
    static WATCHERS: RefCell<Vec<Weak<State>>> = const { RefCell::new(Vec::new()) };
    static RETAINED_BYTES: Cell<usize> = const { Cell::new(0) };
}

struct Storage(usize);

impl Drop for Storage {
    fn drop(&mut self) {
        let _ = RETAINED_BYTES.try_with(|bytes| bytes.set(bytes.get() - self.0));
    }
}

struct State {
    // Sorted by identity for invalidation; each value is the original ordered
    // snapshot index. The lookup never determines visitation order.
    keys: Vec<(usize, usize)>,
    active: [Vec<Cell<u64>>; 2],
    button_revision: Cell<u64>,
    _storage: Storage,
}

pub(crate) fn wake(identity: usize) {
    WATCHERS.with_borrow_mut(|watchers| {
        watchers.retain(|weak| {
            let Some(state) = weak.upgrade() else {
                return false;
            };
            if let Ok(key) = state.keys.binary_search_by_key(&identity, |entry| entry.0) {
                let index = state.keys[key].1;
                for bits in &state.active {
                    bits[index / 64].set(bits[index / 64].get() | (1 << (index % 64)));
                }
                #[cfg(feature = "lifecycle_counts")]
                ruffle_render::profile::lifecycle::add(
                    ruffle_render::profile::lifecycle::Count::WorklistWake,
                    1,
                );
            }
            true
        });
    });
}

impl State {
    fn refresh_buttons(&self) {
        let revision = button_state_revision();
        if self.button_revision.replace(revision) != revision {
            // Hidden and shared button states lack a unique parent chain.
            // Wake both phases conservatively, including older active snapshots.
            for bits in &self.active {
                for word in bits {
                    word.set(u64::MAX);
                }
            }
            #[cfg(feature = "lifecycle_counts")]
            ruffle_render::profile::lifecycle::add(
                ruffle_render::profile::lifecycle::Count::WorklistButtonWake,
                1,
            );
        }
    }

    fn next(&self, phase: usize, cursor: usize, len: usize) -> Option<usize> {
        self.refresh_buttons();
        let bits = &self.active[phase];
        let mut word_index = cursor / 64;
        let mut mask = u64::MAX << (cursor % 64);
        while let Some(word) = bits.get(word_index) {
            let active = word.get() & mask;
            if active != 0 {
                let index = word_index * 64 + active.trailing_zeros() as usize;
                return (index < len).then_some(index);
            }
            word_index += 1;
            mask = u64::MAX;
        }
        None
    }

    fn park(&self, phase: usize, index: usize) {
        let word = &self.active[phase][index / 64];
        word.set(word.get() & !(1 << (index % 64)));
    }
}

#[derive(Collect)]
#[collect(no_drop)]
pub(super) struct Worklist<'gc> {
    members: Rc<Vec<DisplayObjectWeak<'gc>>>,
    #[collect(require_static)]
    state: Rc<State>,
}

impl<'gc> Worklist<'gc> {
    fn new(members: Rc<Vec<DisplayObjectWeak<'gc>>>, mc: &Mutation<'gc>) -> Option<Self> {
        let len = members.len();
        let words = len.div_ceil(64);
        // Charge the retained member allocation too, even though it is shared
        // with OrphanManager. The slack includes Rc/Vec headers and a watcher.
        let bytes = members
            .capacity()
            .checked_mul(std::mem::size_of::<DisplayObjectWeak>())?
            .checked_add(len.checked_mul(std::mem::size_of::<(usize, usize)>())?)?
            .checked_add(words.checked_mul(2 * std::mem::size_of::<Cell<u64>>())?)?
            .checked_add(1024)?;
        if RETAINED_BYTES.get().checked_add(bytes)? > BUDGET - WATCHER_BYTES {
            return None;
        }
        let registered = WATCHERS.with_borrow_mut(|watchers| {
            watchers.retain(|weak| weak.strong_count() != 0);
            if watchers.len() == MAX_WATCHERS {
                return false;
            }
            if watchers.capacity() < MAX_WATCHERS
                && watchers
                    .try_reserve_exact(MAX_WATCHERS - watchers.len())
                    .is_err()
            {
                return false;
            }
            true
        });
        if !registered {
            return None;
        }
        let mut keys = Vec::new();
        keys.try_reserve_exact(len).ok()?;
        let mut active = [Vec::new(), Vec::new()];
        for bits in &mut active {
            bits.try_reserve_exact(words).ok()?;
            bits.resize_with(words, || Cell::new(u64::MAX));
        }
        for (index, member) in members.iter().enumerate() {
            if let Some(dobj) = member.upgrade(mc) {
                keys.push((dobj.base().observe_orphan_worklist(), index));
            }
        }
        keys.sort_unstable_by_key(|key| key.0);
        RETAINED_BYTES.set(RETAINED_BYTES.get() + bytes);
        let state = Rc::new(State {
            keys,
            active,
            button_revision: Cell::new(button_state_revision()),
            _storage: Storage(bytes),
        });
        WATCHERS.with_borrow_mut(|watchers| {
            watchers.retain(|weak| weak.strong_count() != 0);
            watchers.push(Rc::downgrade(&state));
        });
        Some(Self { members, state })
    }

    pub(super) fn each(
        context: &mut UpdateContext<'gc>,
        f: &mut impl FnMut(DisplayObject<'gc>, &mut UpdateContext<'gc>),
    ) -> bool {
        let phase = match context.frame_phase {
            FramePhase::Construct => 0,
            FramePhase::FrameScripts => 1,
            _ => return false,
        };
        let manager = &mut context.orphan_manager;
        if manager
            .worklist
            .as_ref()
            .is_none_or(|list| !Rc::ptr_eq(&list.members, &manager.orphans))
        {
            // Release an obsolete cached snapshot before checking the budget;
            // reentrant walks keep their own snapshots alive independently.
            manager.worklist = None;
            context.orphan_manager.worklist =
                Self::new(context.orphan_manager.orphans.clone(), context.gc()).map(Rc::new);
        }
        let Some(list) = context.orphan_manager.worklist.clone() else {
            #[cfg(feature = "lifecycle_counts")]
            ruffle_render::profile::lifecycle::add(
                ruffle_render::profile::lifecycle::Count::WorklistFallback,
                1,
            );
            return false;
        };
        #[cfg(feature = "lifecycle_counts")]
        ruffle_render::profile::lifecycle::add(
            ruffle_render::profile::lifecycle::Count::WorklistEntries,
            list.members.len() as u64,
        );
        let mut cursor = 0;
        #[cfg(feature = "lifecycle_counts")]
        let mut visits = 0;
        while let Some(index) = list.state.next(phase, cursor, list.members.len()) {
            cursor = index + 1;
            let Some(dobj) = valid_orphan(list.members[index], context.gc()) else {
                list.state.park(phase, index);
                continue;
            };
            if phase_idle(dobj, phase) {
                list.state.park(phase, index);
                continue;
            }
            #[cfg(feature = "lifecycle_counts")]
            {
                visits += 1;
            }
            f(dobj, context);
            if phase_idle(dobj, phase) {
                list.state.park(phase, index);
            }
        }
        #[cfg(feature = "lifecycle_counts")]
        {
            use ruffle_render::profile::lifecycle::{Count, add};
            add(Count::WorklistVisit, visits);
            add(Count::WorklistSkipped, list.members.len() as u64 - visits);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::display_object::MovieClip;
    use crate::orphan_manager::OrphanManager;
    use crate::tag_utils::{SwfMovie, SwfMovieData};
    use gc_arena::{Arena, Rootable};

    #[test]
    fn worklist_snapshot_keeps_only_weak_movie_references() {
        let mut arena = Arena::<Rootable![OrphanManager<'_>]>::new(|mc| {
            let movie = SwfMovie::new(mc, SwfMovieData::empty(10, None));
            let clip = MovieClip::new(movie, mc);
            let mut manager = OrphanManager::default();
            manager.add_orphan_obj(clip.into());
            manager.worklist = Worklist::new(manager.orphans.clone(), mc).map(Rc::new);
            assert!(manager.worklist.is_some());
            manager
        });
        arena.finish_cycle();
        arena.mutate(|mc, manager| {
            assert!(manager.orphans[0].upgrade(mc).is_none());
            assert!(
                manager.worklist.as_ref().unwrap().members[0]
                    .upgrade(mc)
                    .is_none()
            );
        });
    }

    #[test]
    fn exhausted_auxiliary_budget_declines_a_new_worklist() {
        let before = RETAINED_BYTES.get();
        let reservation = Storage(BUDGET - before);
        RETAINED_BYTES.set(BUDGET);
        let arena = Arena::<Rootable![()]>::new(|mc| {
            assert!(Worklist::new(Rc::new(Vec::new()), mc).is_none());
        });
        drop(arena);
        drop(reservation);
        assert_eq!(RETAINED_BYTES.get(), before);
    }

    #[test]
    fn bitmaps_keep_order_across_word_boundaries_and_callback_wakes() {
        let state = State {
            keys: Vec::new(),
            active: [vec![Cell::new(0), Cell::new(1 << 1)], vec![]],
            button_revision: Cell::new(button_state_revision()),
            _storage: Storage(0),
        };
        assert_eq!(state.next(0, 0, 128), Some(65));
        state.active[0][0].set(1 << 4);
        state.active[0][1].set((1 << 1) | (1 << 6));
        assert_eq!(state.next(0, 66, 128), Some(70));
        assert_eq!(state.next(0, 0, 128), Some(4));
        assert_eq!(state.next(0, 71, 128), None);
    }
}

fn phase_idle(dobj: DisplayObject<'_>, phase: usize) -> bool {
    // Only MovieClip has a persistent, invalidated proof. Other kinds keep
    // their original callback, including first-time construction of leaves.
    dobj.as_movie_clip().is_some()
        && if phase == 0 {
            dobj.construction_is_settled()
        } else {
            dobj.frame_scripts_are_idle()
        }
}
