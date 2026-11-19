//! Persistent display-list order. A small container occupies one contiguous leaf.
//! Interior counts allow indexing without renumbering children in other leaves.
use super::DisplayObject;
use gc_arena::Collect;
use std::ops::Index;
use std::rc::Rc;

const LEAF: usize = 64;
const FANOUT: usize = 16;

#[derive(Clone, Collect)]
#[collect(no_drop)]
pub(super) struct Order<'gc>(Rc<Page<'gc>>);

#[derive(Clone)]
enum Page<'gc> {
    Leaf(Vec<DisplayObject<'gc>>),
    Branch {
        len: usize,
        pages: Vec<Rc<Page<'gc>>>,
    },
}

// SAFETY: leaves trace every display object and branches trace every child page.
// Page has no destructor which can observe GC data. An explicit impl breaks
// the recursive NEEDS_TRACE associated-constant dependency of Rc<Page>.
unsafe impl<'gc> Collect<'gc> for Page<'gc> {
    fn trace<C: gc_arena::collect::Trace<'gc>>(&self, cc: &mut C) {
        match self {
            Self::Leaf(items) => cc.trace(items),
            Self::Branch { pages, .. } => cc.trace(pages),
        }
    }
}

impl<'gc> Page<'gc> {
    fn len(&self) -> usize {
        match self {
            Self::Leaf(items) => items.len(),
            Self::Branch { len, .. } => *len,
        }
    }

    fn weight(&self) -> (usize, usize) {
        match self {
            Self::Leaf(items) => (items.len(), LEAF),
            Self::Branch { pages, .. } => (pages.len(), FANOUT),
        }
    }

    // At the right boundary, insertion belongs to the last page.
    fn locate(pages: &[Rc<Self>], mut index: usize) -> (usize, usize) {
        for (slot, page) in pages.iter().enumerate() {
            if index < page.len() || slot + 1 == pages.len() {
                return (slot, index);
            }
            index -= page.len();
        }
        unreachable!("empty interior display-list page")
    }

    fn leaf_at(&self, index: usize) -> (&[DisplayObject<'gc>], usize) {
        match self {
            Self::Leaf(items) => (items, index),
            Self::Branch { pages, .. } => {
                let (slot, offset) = Self::locate(pages, index);
                pages[slot].leaf_at(offset)
            }
        }
    }

    fn edit(page: &mut Rc<Self>) -> &mut Self {
        if Rc::strong_count(page) > 1
            && let Self::Leaf(items) = &**page
        {
            ruffle_render::profile::work(|w| w.child_slots_copied += items.len() as u64);
        }
        Rc::make_mut(page)
    }

    fn insert(&mut self, index: usize, value: DisplayObject<'gc>) -> Option<Rc<Self>> {
        match self {
            Self::Leaf(items) => {
                items.insert(index, value);
                (items.len() > LEAF).then(|| Rc::new(Self::Leaf(items.split_off(LEAF / 2))))
            }
            Self::Branch { len, pages } => {
                let (slot, offset) = Self::locate(pages, index);
                if let Some(right) = Self::edit(&mut pages[slot]).insert(offset, value) {
                    pages.insert(slot + 1, right);
                }
                *len += 1;
                if pages.len() <= FANOUT {
                    return None;
                }
                let right = pages.split_off(FANOUT / 2);
                let right_len = right.iter().map(|p| p.len()).sum::<usize>();
                *len -= right_len;
                Some(Rc::new(Self::Branch {
                    len: right_len,
                    pages: right,
                }))
            }
        }
    }

    fn replace(&mut self, index: usize, value: DisplayObject<'gc>) {
        match self {
            Self::Leaf(items) => items[index] = value,
            Self::Branch { pages, .. } => {
                let (slot, offset) = Self::locate(pages, index);
                Self::edit(&mut pages[slot]).replace(offset, value);
            }
        }
    }

    fn remove(&mut self, index: usize) -> DisplayObject<'gc> {
        match self {
            Self::Leaf(items) => items.remove(index),
            Self::Branch { len, pages } => {
                let (slot, offset) = Self::locate(pages, index);
                let value = Self::edit(&mut pages[slot]).remove(offset);
                *len -= 1;
                let (weight, capacity) = pages[slot].weight();
                if weight == 0 {
                    pages.remove(slot);
                } else if weight < capacity / 2 && pages.len() > 1 {
                    let left = slot.min(pages.len() - 2);
                    if pages[left].weight().0 + pages[left + 1].weight().0 <= capacity {
                        let right = pages.remove(left + 1);
                        if Rc::strong_count(&right) > 1
                            && let Self::Leaf(items) = &*right
                        {
                            ruffle_render::profile::work(|w| {
                                w.child_slots_copied += items.len() as u64
                            });
                        }
                        match (Self::edit(&mut pages[left]), Rc::unwrap_or_clone(right)) {
                            (Self::Leaf(a), Self::Leaf(b)) => a.extend(b),
                            (Self::Branch { len, pages }, Self::Branch { len: n, pages: b }) => {
                                *len += n;
                                pages.extend(b);
                            }
                            _ => unreachable!("siblings have equal height"),
                        }
                    }
                }
                value
            }
        }
    }
}

/// An owning iterator may outlive a logical edit. Hold only its current leaf
/// in addition to the stable root, and descend again at a leaf boundary.
#[derive(Default)]
pub(super) struct Cursor<'gc> {
    page: Option<Rc<Page<'gc>>>,
    start: usize,
}

impl<'gc> Cursor<'gc> {
    pub(super) fn at(&mut self, order: &Order<'gc>, index: usize) -> DisplayObject<'gc> {
        if let Page::Leaf(items) = &*order.0 {
            return items[index];
        }
        if let Some(page) = &self.page
            && let Page::Leaf(items) = &**page
            && let Some(offset) = index.checked_sub(self.start)
            && let Some(child) = items.get(offset)
        {
            return *child;
        }
        let mut page = &order.0;
        let mut offset = index;
        while let Page::Branch { pages, .. } = &**page {
            let (slot, next) = Page::locate(pages, offset);
            page = &pages[slot];
            offset = next;
        }
        let Page::Leaf(items) = &**page else {
            unreachable!()
        };
        self.start = index - offset;
        self.page = Some(page.clone());
        items[offset]
    }
}

impl<'gc> Order<'gc> {
    pub(super) fn new() -> Self {
        Self(Rc::new(Page::Leaf(Vec::new())))
    }
    pub(super) fn len(&self) -> usize {
        self.0.len()
    }
    pub(super) fn position(&self, child: DisplayObject<'gc>) -> Option<usize> {
        self.iter()
            .position(|other| DisplayObject::ptr_eq(*other, child))
    }
    pub(super) fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub(super) fn get(&self, index: usize) -> Option<&DisplayObject<'gc>> {
        if index >= self.len() {
            return None;
        }
        let (page, offset) = self.0.leaf_at(index);
        page.get(offset)
    }
    pub(super) fn iter(&self) -> impl Iterator<Item = &DisplayObject<'gc>> {
        // Walk contiguous runs in order. Descend from the root only between runs.
        let mut index = 0;
        std::iter::from_fn(move || {
            if index == self.len() {
                return None;
            }
            let (leaf, offset) = self.0.leaf_at(index);
            let rest = &leaf[offset..];
            index += rest.len();
            Some(rest)
        })
        .flatten()
    }
    pub(super) fn insert(&mut self, index: usize, value: DisplayObject<'gc>) {
        debug_assert!(index <= self.len());
        if let Some(right) = Page::edit(&mut self.0).insert(index, value) {
            self.0 = Rc::new(Page::Branch {
                len: self.len() + right.len(),
                pages: vec![self.0.clone(), right],
            });
        }
    }
    pub(super) fn push(&mut self, value: DisplayObject<'gc>) {
        self.insert(self.len(), value);
    }
    pub(super) fn replace(&mut self, index: usize, value: DisplayObject<'gc>) {
        Page::edit(&mut self.0).replace(index, value);
    }
    pub(super) fn remove(&mut self, index: usize) -> DisplayObject<'gc> {
        let value = Page::edit(&mut self.0).remove(index);
        while let Page::Branch { pages, .. } = &*self.0 {
            match pages.as_slice() {
                [only] => self.0 = only.clone(),
                [] => self.0 = Rc::new(Page::Leaf(Vec::new())),
                _ => break,
            }
        }
        value
    }
    pub(super) fn swap(&mut self, a: usize, b: usize) {
        if a != b {
            let (first, second) = (self[a], self[b]);
            self.replace(a, second);
            self.replace(b, first);
        }
    }
    pub(super) fn move_to(&mut self, from: usize, to: usize) {
        let to = to.min(self.len() - 1);
        if from != to {
            let value = self.remove(from);
            self.insert(to, value);
        }
    }
}

impl<'gc> Index<usize> for Order<'gc> {
    type Output = DisplayObject<'gc>;
    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("invalid display-list index")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::display_object::{Graphic, TDisplayObject};
    use crate::player::PlayerBuilder;

    #[test]
    fn promoted_order_matches_vec_across_splits_merges_and_held_versions() {
        let player = PlayerBuilder::new().build();
        player
            .lock()
            .unwrap()
            .mutate_with_update_context(|context| {
                let mut order = Order::new();
                let mut expected = Vec::new();
                let mut random = 42u32;
                let mut rank = |limit| {
                    random = random.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (random as usize) % limit
                };
                for _ in 0..1400 {
                    let position = rank(expected.len() + 1);
                    let value = Graphic::empty(context).into();
                    order.insert(position, value);
                    expected.insert(position, value);
                }
                let old = order.clone();
                let old_expected = expected.clone();
                for _ in 0..3200 {
                    let from = rank(expected.len());
                    let to = rank(expected.len());
                    if rank(2) == 0 {
                        order.swap(from, to);
                        expected.swap(from, to);
                    } else {
                        order.move_to(from, to);
                        let value = expected.remove(from);
                        expected.insert(to, value);
                    }
                }
                let mut cursor = Cursor::default();
                for (i, child) in expected.iter().enumerate() {
                    assert!(DisplayObject::ptr_eq(cursor.at(&order, i), *child));
                }
                while !expected.is_empty() {
                    let index = rank(expected.len());
                    assert!(DisplayObject::ptr_eq(
                        order.remove(index),
                        expected.remove(index)
                    ));
                }
                let mut cursor = Cursor::default();
                for (i, child) in old_expected.iter().enumerate().rev() {
                    assert_eq!(cursor.at(&old, i).as_ptr(), child.as_ptr());
                }
            });
    }
}
