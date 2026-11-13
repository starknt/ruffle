//! Container mix-in for display objects

use crate::avm1::{Activation, ActivationIdentifier};
use crate::avm2::{
    Activation as Avm2Activation, Avm2, EventObject as Avm2EventObject, Multiname as Avm2Multiname,
    Value as Avm2Value,
};
use crate::context::{RenderContext, UpdateContext};
use crate::display_object::avm1_button::Avm1Button;
use crate::display_object::loader_display::LoaderDisplay;
use crate::display_object::movie_clip::MovieClip;
use crate::display_object::stage::Stage;
use crate::display_object::{Depth, DisplayObject, TDisplayObject, TInteractiveObject};
use crate::focus_tracker::TabOrder;
use crate::string::WStr;
use crate::tag_utils::SwfMovieData;
use gc_arena::{Collect, Mutation};
use ruffle_macros::{enum_trait_object, istr};
use ruffle_render::commands::CommandHandler;
#[cfg(any(feature = "lifecycle_projection", test))]
use std::cell::OnceCell;
use std::cell::{Ref, RefMut};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::ops::Bound;
use std::rc::Rc;

/// Dispatch the `removedFromStage` event on a child and all of it's
/// grandchildren, recursively.
pub fn dispatch_removed_from_stage_event<'gc>(
    child: DisplayObject<'gc>,
    context: &mut UpdateContext<'gc>,
) {
    if let Some(object) = child.object2() {
        let removed_evt = Avm2EventObject::bare_default_event(context, "removedFromStage");
        Avm2::dispatch_event(context, removed_evt, object.into());
    }

    if let Some(child_container) = child.as_container() {
        for grandchild in child_container.iter_render_list() {
            dispatch_removed_from_stage_event(grandchild, context)
        }
    }
}

/// Dispatch the `removed` event on a child and log any errors encountered
/// whilst doing so.
pub fn dispatch_removed_event<'gc>(child: DisplayObject<'gc>, context: &mut UpdateContext<'gc>) {
    if let Some(object) = child.object2() {
        let removed_evt = Avm2EventObject::bare_event(context, "removed", true, false);
        Avm2::dispatch_event(context, removed_evt, object.into());

        if child.is_on_stage(context) {
            dispatch_removed_from_stage_event(child, context)
        }
    }
}

/// Dispatch the `addedToStage` event on a child, ignoring it's grandchildren.
pub fn dispatch_added_to_stage_event_only<'gc>(
    child: DisplayObject<'gc>,
    context: &mut UpdateContext<'gc>,
) {
    if let Some(object) = child.object2() {
        let added_evt = Avm2EventObject::bare_default_event(context, "addedToStage");
        Avm2::dispatch_event(context, added_evt, object.into());
    }
}

/// Dispatch the `addedToStage` event on a child and all of it's grandchildren,
/// recursively.
pub fn dispatch_added_to_stage_event<'gc>(
    child: DisplayObject<'gc>,
    context: &mut UpdateContext<'gc>,
) {
    dispatch_added_to_stage_event_only(child, context);

    if let Some(child_container) = child.as_container() {
        for grandchild in child_container.iter_render_list() {
            dispatch_added_to_stage_event(grandchild, context)
        }
    }
    if let Some(button) = child.as_avm2_button()
        && let Some(child) = button.get_state_child(button.state().into())
    {
        dispatch_added_to_stage_event(child, context);
    }
}

/// Dispatch an `added` event to one object, and log any errors encountered
/// whilst doing so.
pub fn dispatch_added_event_only<'gc>(child: DisplayObject<'gc>, context: &mut UpdateContext<'gc>) {
    if let Some(object) = child.object2() {
        let added_evt = Avm2EventObject::bare_event(context, "added", true, false);
        Avm2::dispatch_event(context, added_evt, object.into());
    }
}

/// Dispatch the `added` event on a child and log any errors encountered
/// whilst doing so.
///
/// `child_was_on_stage` should be the result of calling `child.is_on_stage()`
/// before any container manipulation has been made. The `added` event is
/// generally fired after the container manipulation has been made.
pub fn dispatch_added_event<'gc>(
    parent: DisplayObject<'gc>,
    child: DisplayObject<'gc>,
    child_was_on_stage: bool,
    context: &mut UpdateContext<'gc>,
) {
    dispatch_added_event_only(child, context);

    if parent.is_on_stage(context) && !child_was_on_stage {
        dispatch_added_to_stage_event(child, context);
    }
}

#[enum_trait_object(
    #[derive(Clone, Collect, Copy, Debug)]
    #[collect(no_drop)]
    pub enum DisplayObjectContainer<'gc> {
        Stage(Stage<'gc>),
        Avm1Button(Avm1Button<'gc>),
        MovieClip(MovieClip<'gc>),
        LoaderDisplay(LoaderDisplay<'gc>),
    }
)]
pub trait TDisplayObjectContainer<'gc>:
    'gc
    + Clone
    + Copy
    + Collect<'gc>
    + Debug
    + Into<DisplayObjectContainer<'gc>>
    + Into<DisplayObject<'gc>>
{
    /// Get read-only access to the raw container.
    fn raw_container(&self) -> Ref<'_, ChildContainer<'gc>>;

    /// Get mutable access to the raw container.
    fn raw_container_mut(&self, gc_context: &Mutation<'gc>) -> RefMut<'_, ChildContainer<'gc>>;

    /// Get a child display object by its position in the render list.
    ///
    /// The `index` provided here should not be confused with the `Depth`s used
    /// to index the depth list.
    #[no_dynamic]
    fn child_by_index(self, index: usize) -> Option<DisplayObject<'gc>> {
        self.raw_container().get_id(index)
    }

    /// Get a child display object by its position in the depth list.
    ///
    /// The `Depth` provided here should not be confused with the `index`s used
    /// to index the render list.
    #[no_dynamic]
    fn child_by_depth(self, depth: Depth) -> Option<DisplayObject<'gc>> {
        self.raw_container().get_depth(depth)
    }

    /// Checks if there's a child at a given depth.
    #[no_dynamic]
    fn has_child_at_depth(self, depth: Depth) -> bool {
        self.raw_container().has_depth(depth)
    }

    /// Get a child display object by its instance/timeline name.
    ///
    /// The `case_sensitive` parameter determines if we should consider
    /// children with different capitalizations as being distinct names.
    ///
    /// If multiple children with the same name exist, the one with the lowest
    /// depth wins. Children not on the depth list will not be accessible via
    /// this mechanism.
    #[no_dynamic]
    fn child_by_name(self, name: &WStr, case_sensitive: bool) -> Option<DisplayObject<'gc>> {
        self.raw_container().get_name(name, case_sensitive)
    }

    /// Returns the number of children on the render list.
    #[no_dynamic]
    fn num_children(self) -> usize {
        self.raw_container().num_children()
    }

    /// Returns the highest depth among children.
    #[no_dynamic]
    fn highest_depth(self) -> Depth {
        self.raw_container().highest_depth()
    }

    /// Insert a child display object into the container at a specific position
    /// in the depth list, removing any child already at that position.
    ///
    /// After inserting the child into the depth list, we will attempt to
    /// assign it a render list position one after the previous item in the
    /// depth list. The position that children are placed into the render list
    /// matches Flash Player behavior.
    ///
    /// Any child removed from the depth list will also be removed from the
    /// render list if and only if the child was not flagged as
    /// being placed by script. If such a child was removed from said lists, it
    /// will be returned here. Otherwise, this method returns `None`.
    ///
    /// Note: this method specifically does *not* dispatch events on any
    /// children it modifies. You must do this yourself.
    #[no_dynamic]
    fn replace_at_depth(
        self,
        context: &mut UpdateContext<'gc>,
        child: DisplayObject<'gc>,
        depth: Depth,
    ) -> Option<DisplayObject<'gc>> {
        let removed_child = self
            .raw_container_mut(context.gc())
            .replace_at_depth(child, depth);

        child.set_parent(context, Some(self.into()));
        child.set_place_frame(0);
        child.set_depth(depth);

        if let Some(removed_child) = removed_child
            && !self.raw_container().is_action_script_3()
        {
            // The parent of `removed_child` should NOT be set to `None` here.
            // It's theoretically impossible for an AVM2 child to end up here(?),
            // and doing so for AVM1 can e.g. break the logic of the audio
            // manager's `stop_sounds_on_parent_and_children` method.
            // (See https://github.com/ruffle-rs/ruffle/issues/21500)
            removed_child.avm1_unload(context);
        }

        let this: DisplayObject<'_> = self.into();
        this.invalidate_cached_bitmap();

        removed_child
    }

    /// Move a child display object around in the container's depth list.
    ///
    /// Any child already at the desired position will move back to the new
    /// child's former position. The render list positions of each child will
    /// also be swapped. If no child has been displaced by the swap operation,
    /// then the render list position of the child will be determined in the same
    /// way as `replace_at_depth`.
    #[no_dynamic]
    fn swap_at_depth(
        &mut self,
        context: &mut UpdateContext<'gc>,
        child: DisplayObject<'gc>,
        depth: Depth,
    ) {
        let this: DisplayObject<'_> = (*self).into();
        // Verify this is actually our child.
        // TODO: This seems unnecessary (especially since AS3 MovieClips
        // are allowed to be used in ways that would trip this assert).
        debug_assert!(DisplayObject::ptr_eq(child.parent().unwrap(), this));

        self.raw_container_mut(context.gc())
            .swap_at_depth(context, this, child, depth);

        this.invalidate_cached_bitmap();
    }

    /// Insert a child display object into the container at a specific position
    /// in the render list.
    ///
    /// Callers of this method should be aware that reordering items onto or off of the
    /// render list can make further depth list manipulations (e.g. from the
    /// timeline) produce unusual results.
    #[no_dynamic]
    fn insert_at_index(
        &mut self,
        context: &mut UpdateContext<'gc>,
        child: DisplayObject<'gc>,
        index: usize,
    ) {
        let this: DisplayObject<'_> = (*self).into();
        let parent_changed = if let Some(old_parent) = child.parent() {
            if !DisplayObject::ptr_eq(old_parent, this) {
                if let Some(mut old_parent) = old_parent.as_container() {
                    old_parent.remove_child(context, child);
                }

                true
            } else {
                false
            }
        } else {
            true
        };

        let child_was_on_stage = child.is_on_stage(context);

        child.set_place_frame(0);
        child.set_parent(context, Some(this));
        if !self.raw_container().is_action_script_3() {
            child.set_avm1_removed(false);
        }

        self.raw_container_mut(context.gc())
            .insert_at_id(child, index);

        if parent_changed {
            dispatch_added_event(this, child, child_was_on_stage, context);
        }

        this.invalidate_cached_bitmap();
    }

    /// Swap two children in the render list.
    ///
    /// No changes to the depth or render lists are made by this function.
    #[no_dynamic]
    fn swap_at_index(&mut self, context: &mut UpdateContext<'gc>, index1: usize, index2: usize) {
        self.raw_container_mut(context.gc())
            .swap_at_id(index1, index2);
        let this: DisplayObject<'_> = (*self).into();
        this.invalidate_cached_bitmap();
    }

    /// Remove (and unloads) a child display object from this container's render and depth lists.
    ///
    /// Will also handle AVM1 delayed clip removal, when a unload listener is present
    #[no_dynamic]
    fn remove_child(&mut self, context: &mut UpdateContext<'gc>, child: DisplayObject<'gc>) {
        let this: DisplayObject<'_> = (*self).into();

        // We should always be the parent of this child
        debug_assert!(DisplayObject::ptr_eq(
            child.parent().unwrap(),
            (*self).into()
        ));

        // Check if this child should have delayed removal (AVM1 only)
        if !self.raw_container().is_action_script_3() {
            let should_delay_removal = {
                let mut activation = Activation::from_nothing(
                    context,
                    ActivationIdentifier::root("[Unload Handler Check]"),
                    this.avm1_root(),
                );

                ChildContainer::should_delay_removal(&mut activation, child)
            };

            if should_delay_removal {
                let mut raw_container = self.raw_container_mut(context.gc());

                // Remove the child from the depth list, before moving it to a negative depth
                raw_container.remove_child_from_depth_list(child);

                // Enqueue for removal
                ChildContainer::queue_removal(child, context);

                // Mark that we have a pending removal
                raw_container.set_pending_removals(true);

                // Re-Insert the child at the new depth
                raw_container.insert_child_into_depth_list(child.depth(), child);

                drop(raw_container);
                this.invalidate_cached_bitmap();

                return;
            }
        }

        self.remove_child_directly(context, child);
    }

    /// Remove (and unloads) a child display object from this container's render and depth lists.
    #[no_dynamic]
    fn remove_child_directly(&self, context: &mut UpdateContext<'gc>, child: DisplayObject<'gc>) {
        dispatch_removed_event(child, context);

        // Event listeners might have moved the child to another container,
        // in which case it's removed from there instead. They might also
        // have removed it already, in which case there's nothing left to do.
        let Some(this) = child.parent().and_then(|parent| parent.as_container()) else {
            return;
        };
        this.raw_container_mut(context.gc())
            .remove_child_from_depth_list(child);

        let removed_from_render_list =
            ChildContainer::remove_child_from_render_list(this, child, context);

        if removed_from_render_list {
            if !this.raw_container().is_action_script_3() {
                child.avm1_unload(context);
            } else if child.object2().is_some() {
                //TODO: This is an awful, *awful* hack to deal with the fact
                //that unloaded AVM1 clips see their parents, while AVM2 clips
                //don't.
                child.set_parent(context, None);
            }

            let this: DisplayObject<'_> = this.into();
            this.invalidate_cached_bitmap();
        }
    }

    /// Insert a child directly into this container's depth list.
    #[no_dynamic]
    fn insert_child_into_depth_list(
        &mut self,
        context: &mut UpdateContext<'gc>,
        depth: Depth,
        child: DisplayObject<'gc>,
    ) {
        let this: DisplayObject<'_> = (*self).into();

        child.set_depth(depth);
        child.set_parent(context, Some(this));
        self.raw_container_mut(context.gc())
            .insert_child_into_depth_list(depth, child);

        this.invalidate_cached_bitmap();
    }

    /// Removes (without unloading) a child display object from this container's depth list.
    #[no_dynamic]
    fn remove_child_from_depth_list(
        &mut self,
        context: &mut UpdateContext<'gc>,
        child: DisplayObject<'gc>,
    ) {
        debug_assert!(DisplayObject::ptr_eq(
            child.parent().unwrap(),
            (*self).into()
        ));

        self.raw_container_mut(context.gc())
            .remove_child_from_depth_list(child);

        let this: DisplayObject<'_> = (*self).into();
        this.invalidate_cached_bitmap();
    }

    /// Determine if the container is empty.
    #[no_dynamic]
    fn is_empty(self) -> bool {
        self.raw_container().is_empty()
    }

    /// Iterates over the children of this display object in render order.
    ///
    /// This yields an iterator that does *not* lock the parent and can be
    /// safely held in situations where display objects need to be unlocked.
    /// This will iterate over a snapshot of the render list as it was at
    /// the time `iter_render_list()` was called.
    ///
    /// The iterator's concrete type is stated here due to Rust language
    /// limitations.
    #[no_dynamic]
    fn iter_render_list(self) -> RenderIter<'gc> {
        RenderIter::from_container(self)
    }

    /// Snapshot of children with non-empty enter-frame or frame-script work.
    /// Construction and render traversal must still visit every child. This
    /// filters only immutable display-object types, never playback state.
    #[cfg(any(feature = "lifecycle_projection", test))]
    #[no_dynamic]
    fn iter_lifecycle_list(self) -> LifecycleIter<'gc> {
        let container = self.raw_container();
        let indices = container.lifecycle_indices.get_or_init(|| {
            let mut indices: Option<Vec<usize>> = None;
            for (index, child) in container.render_list.iter().enumerate() {
                let active = match child {
                    DisplayObject::Stage(_)
                    | DisplayObject::MovieClip(_)
                    | DisplayObject::LoaderDisplay(_)
                    | DisplayObject::Avm1Button(_)
                    | DisplayObject::Avm2Button(_) => true,
                    DisplayObject::Bitmap(_)
                    | DisplayObject::EditText(_)
                    | DisplayObject::TextLine(_)
                    | DisplayObject::Graphic(_)
                    | DisplayObject::MorphShape(_)
                    | DisplayObject::Text(_)
                    | DisplayObject::Video(_) => false,
                };
                if active {
                    if let Some(indices) = &mut indices {
                        indices.push(index);
                    }
                } else if indices.is_none() {
                    // No extra allocation for an entirely active list.
                    indices = Some((0..index).collect::<Vec<_>>());
                }
            }
            indices.map(Rc::new)
        });
        let src = container.render_list.clone();
        let end = indices.as_ref().map_or(src.len(), |indices| indices.len());
        LifecycleIter {
            src,
            indices: indices.clone(),
            range: 0..end,
        }
    }

    fn is_tab_children_avm1(self, _context: &mut UpdateContext<'gc>) -> bool {
        true
    }

    /// The property `tabChildren` allows changing the behavior of
    /// tab ordering hierarchically.
    /// When set to `false`, it excludes the whole subtree represented by
    /// the container from tab ordering.
    ///
    /// _NOTE:_
    /// According to the AS2 documentation, it should affect only automatic tab ordering.
    /// However, that does not seem to be the case, as it also affects custom ordering.
    #[no_dynamic]
    fn is_tab_children(&self, context: &mut UpdateContext<'gc>) -> bool {
        let container = self.raw_container();
        if container.is_action_script_3() {
            container.tab_children
        } else {
            self.is_tab_children_avm1(context)
        }
    }

    #[no_dynamic]
    fn set_tab_children(&self, context: &mut UpdateContext<'gc>, value: bool) {
        let this: DisplayObject<'_> = (*self).into();
        if self.raw_container().is_action_script_3() {
            self.raw_container_mut(context.gc()).tab_children = value;
        } else {
            this.set_avm1_property(istr!(context, "tabChildren"), value.into(), context);
        }
    }

    #[no_dynamic]
    fn fill_tab_order(&self, tab_order: &mut TabOrder<'gc>, context: &mut UpdateContext<'gc>) {
        if !self.is_tab_children(context) {
            // AS3 docs say that objects with custom ordering (tabIndex set)
            // are included even when tabChildren is false.
            // Do not be fooled for that is untrue!
            return;
        }

        for child in self.iter_render_list() {
            if !child.visible() {
                // Non-visible objects and their children are excluded from tab ordering.
                continue;
            }
            if let Some(child) = child.as_interactive()
                && child.is_tabbable(context)
            {
                tab_order.add_object(child);
            }
            if let Some(container) = child.as_container() {
                container.fill_tab_order(tab_order, context);
            }
        }
    }

    /// Renders the children of this container in render list order.
    #[no_dynamic]
    fn render_children(self, context: &mut RenderContext<'_, 'gc>) {
        let mut clip_depth = 0;
        let mut clip_depth_stack: Vec<(Depth, DisplayObject<'_>)> = vec![];
        for child in self.iter_render_list() {
            let depth = child.depth();

            child.pre_render(context);

            // Check if we need to pop off a mask.
            // This must be a while loop because multiple masks can be popped
            // at the same depth.
            while clip_depth > 0 && depth > clip_depth {
                // Clear the mask stencil and pop the mask.
                let (prev_clip_depth, clip_child) = clip_depth_stack.pop().unwrap();
                clip_depth = prev_clip_depth;
                context.commands.deactivate_mask();
                clip_child.render(context);
                context.commands.pop_mask();
            }
            if child.clip_depth() > 0 && child.allow_as_mask() {
                // Push and render the mask.
                clip_depth_stack.push((clip_depth, child));
                clip_depth = child.clip_depth();
                context.commands.push_mask();
                child.render(context);
                context.commands.activate_mask();
            } else if child.visible() || context.commands.drawing_mask() {
                // Either a normal visible child, or a descendant of a mask object
                // that we're drawing. The 'visible' flag is ignored for all descendants
                // of a mask.
                child.render(context);
            }
        }

        // Pop any remaining masks.
        for (_, clip_child) in clip_depth_stack.into_iter().rev() {
            context.commands.deactivate_mask();
            clip_child.render(context);
            context.commands.pop_mask();
        }
    }

    #[cfg(not(feature = "avm_debug"))]
    #[no_dynamic]
    fn recurse_render_tree(&self, _depth: usize) {}

    #[cfg(feature = "avm_debug")]
    #[no_dynamic]
    fn recurse_render_tree(&self, depth: usize) {
        for child in self.iter_render_list() {
            child.display_render_tree(depth);
        }
    }
}

impl<'gc> From<DisplayObjectContainer<'gc>> for DisplayObject<'gc> {
    #[inline(always)]
    fn from(obj: DisplayObjectContainer<'gc>) -> Self {
        match obj {
            DisplayObjectContainer::Stage(o) => DisplayObject::Stage(o),
            DisplayObjectContainer::Avm1Button(o) => DisplayObject::Avm1Button(o),
            DisplayObjectContainer::MovieClip(o) => DisplayObject::MovieClip(o),
            DisplayObjectContainer::LoaderDisplay(o) => DisplayObject::LoaderDisplay(o),
        }
    }
}

/// A structure that stores child display objects.
///
/// Child display objects are stored in two lists: a render list and a depth
/// list. The latter references display objects by their chosen depth; while
/// the render list represents the order in which those children should be
/// rendered. Not all children have a position on this depth.
#[derive(Clone, Collect)]
#[collect(no_drop)]
pub struct ChildContainer<'gc> {
    /// The list of all children in render order.
    ///
    /// This list is the actual list used to render children. All other list
    /// manipulations intended to change the order children are rendered in
    /// must adjust this list as well. All children must be present on the
    /// render list.
    ///
    /// In AVM1, the depth and render lists are identical; AS1/2 code interacts
    /// exclusively with the depth list. However, AS3 instead references clips
    /// by render list indexes and does not manipulate the depth list.
    ///
    /// We store an `Rc` to support iterating over the render list
    /// when modifications may occur during iteration.
    /// Whenever we modify the render list, we call `Rc::make_mut`.
    /// Normally, there will only be one strong reference to this `Rc`,
    /// so we will end up modifying the list in place. However, if
    /// any `RenderIter`s exist (which hold a strong reference to the `Rc`),
    /// modifying the `ChildContainer` will cause `Rc::make_mut` to clone
    /// the underlying data, and store a new `Rc` allocation. This will
    /// cause the `RenderIter` to continue iterating over the old list,
    /// while consumers of the `ChildContainer` will immediately see the
    /// updated list.
    render_list: Rc<Vec<DisplayObject<'gc>>>,

    // Indices only: no extra GC roots, and old iterators retain the matching
    // COW render-list snapshot. Every structural edit invalidates this cache.
    #[cfg(any(feature = "lifecycle_projection", test))]
    #[collect(require_static)]
    lifecycle_indices: OnceCell<Option<Rc<Vec<usize>>>>,

    /// The mapping from timeline Depths to child display objects.
    ///
    /// This list is the list used to map depths to actual display objects.
    /// It does not affect render order. Unlike the render list, children may
    /// or may not live on the depth list.
    ///
    /// In AVM1, the depth and render lists are identical; AS1/2 code interacts
    /// exclusively with the depth list. However, AS3 instead references clips
    /// by render list indexes and does not manipulate the depth list.
    depth_list: BTreeMap<Depth, DisplayObject<'gc>>,

    /// Does this container have any AVM1 objects that are pending removal
    ///
    /// Objects that are pending removal are placed at a negative depth in the depth list,
    /// because accessing children exclusively interacts with the render list, which cannot handle
    /// negative render depths, we need to check both lists when we have something pending removal.
    ///
    /// This should be more efficient than switching the render list to a `BTreeMap`,
    /// as it will usually be false
    has_pending_removals: bool,

    mouse_children: bool,

    /// Specifies whether this container uses AVM1 or AVM2 semantics.
    is_action_script_3: bool,

    /// Specifies whether children are present in the tab ordering.
    tab_children: bool,
}

impl<'gc> ChildContainer<'gc> {
    pub fn new(movie: &SwfMovieData) -> Self {
        Self {
            render_list: Rc::new(Vec::new()),
            #[cfg(any(feature = "lifecycle_projection", test))]
            lifecycle_indices: OnceCell::new(),
            depth_list: BTreeMap::new(),
            has_pending_removals: false,
            mouse_children: true,
            is_action_script_3: movie.is_action_script_3(),
            tab_children: true,
        }
    }

    /// Add a child to the depth list.
    ///
    /// This returns the child that was previously at that particular depth, if
    /// such a child exists. If so, that constitutes removing the child from
    /// the depth list.
    fn insert_child_into_depth_list(
        &mut self,
        depth: Depth,
        child: DisplayObject<'gc>,
    ) -> Option<DisplayObject<'gc>> {
        self.depth_list.insert(depth, child)
    }

    /// Remove a child from the depth list.
    ///
    /// This returns `true` if the child was successfully removed, and `false`
    /// if no list alterations were made.
    fn remove_child_from_depth_list(&mut self, child: DisplayObject<'gc>) -> bool {
        if let Some(other_child) = self.depth_list.get(&child.depth()) {
            DisplayObject::ptr_eq(*other_child, child)
                && self.depth_list.remove(&child.depth()).is_some()
        } else {
            false
        }
    }

    /// Remove a child from the render list.
    ///
    /// This returns `true` if the child was successfully removed, and `false`
    /// if no list alterations were made.
    ///
    /// This must be called *before* setting the child's parent field to `None`.
    /// Note: This cannot be a normal method that takes &self, since we need to call
    /// `parent.object2()` on our own `DisplayObject` (which is borrowed mutably
    /// by `raw_container_mut`)
    fn remove_child_from_render_list(
        container: DisplayObjectContainer<'gc>,
        child: DisplayObject<'gc>,
        context: &mut UpdateContext<'gc>,
    ) -> bool {
        let mut this = container.raw_container_mut(context.gc());

        let render_list_position = this
            .render_list
            .iter()
            .position(|x| DisplayObject::ptr_eq(*x, child));
        if let Some(position) = render_list_position {
            this.render_list_mut().remove(position);
            drop(this);

            // Only set the parent's field to 'null' if the child was not placed/modified
            // on the render list by AVM2 code.
            if !child.placed_by_avm2_script() {
                let parent = child.parent().expect(
                    "Parent must be removed *after* calling `remove_child_from_render_list`",
                );
                if child.has_explicit_name()
                    && let Some(name) = child.name()
                    && let Some(parent_obj) = parent.object2()
                    && child.movie().is_action_script_3()
                {
                    let parent_obj = Avm2Value::from(parent_obj);

                    let mut activation = Avm2Activation::from_nothing(context);
                    let multiname =
                        Avm2Multiname::new(activation.avm2().find_public_namespace(), name);
                    let current_val = parent_obj.get_property(&multiname, &mut activation);
                    match current_val {
                        Ok(Avm2Value::Null | Avm2Value::Undefined) => {
                            // When the `get_property` returns null
                            // or undefined, FP doesn't attempt to
                            // set it to `null`. This is observable:
                            // a setter method won't get called.
                        }
                        Ok(_other) => {
                            let res = parent_obj.set_property(
                                &multiname,
                                Avm2Value::Null,
                                &mut activation,
                            );
                            if let Err(err) = res {
                                Avm2::uncaught_error(
                                    &mut activation,
                                    Some(child),
                                    err,
                                    &format!("Error setting AVM2 child named \"{}\" to null", name),
                                );
                            }
                        }
                        Err(_) => {
                            // In FP, errors when accessing the
                            // original value are completely
                            // swallowed. They don't make it to
                            // flashlog or to uncaught error events.
                        }
                    }
                }
            }
            true
        } else {
            false
        }
    }

    /// Insert a child display object into the container at a specific position
    /// in the depth list, removing any child already at that position.
    ///
    /// After inserting the child into the depth list, we will attempt to
    /// assign it a render list position one after the previous item in the
    /// depth list. The position that children are placed into the render list
    /// matches Flash Player behavior.
    ///
    /// Any child removed from the depth list will also be removed from the
    /// render list if and only if the child was not flagged as
    /// being placed by script. If such a child was removed from said lists, it
    /// will be returned here. Otherwise, this method returns `None`.
    fn replace_at_depth(
        &mut self,
        child: DisplayObject<'gc>,
        depth: Depth,
    ) -> Option<DisplayObject<'gc>> {
        let prev_child = self.insert_child_into_depth_list(depth, child);
        if let Some(prev_child) = prev_child {
            if let Some(position) = self
                .render_list
                .iter()
                .position(|x| DisplayObject::ptr_eq(*x, prev_child))
            {
                if !prev_child.placed_by_avm2_script() {
                    self.replace_id(position, child);
                    Some(prev_child)
                } else {
                    self.insert_id(position + 1, child);
                    None
                }
            } else {
                tracing::error!(
                    "ChildContainer::replace_at_depth: Previous child is not in render list"
                );
                self.push_id(child);
                None
            }
        } else {
            let above = self
                .depth_list
                .range((Bound::Excluded(depth), Bound::Unbounded))
                .map(|(_, v)| *v)
                .next();

            if let Some(above_child) = above {
                if let Some(position) = self
                    .render_list
                    .iter()
                    .position(|x| DisplayObject::ptr_eq(*x, above_child))
                {
                    self.insert_id(position, child);
                } else {
                    self.push_id(child);
                }
            } else {
                self.push_id(child);
            }

            None
        }
    }

    /// Returns the highest depth among children.
    fn highest_depth(&self) -> Depth {
        self.depth_list.keys().next_back().copied().unwrap_or(0)
    }

    /// Determine if the render list is empty.
    fn is_empty(&self) -> bool {
        self.render_list.is_empty()
    }

    /// Get a child at a given depth.
    fn get_depth(&self, depth: Depth) -> Option<DisplayObject<'gc>> {
        self.depth_list.get(&depth).copied()
    }

    /// Checks if there's a child at a given depth.
    fn has_depth(&self, depth: Depth) -> bool {
        self.depth_list.contains_key(&depth)
    }

    /// Get a child by it's instance/timeline name.
    ///
    /// The `case_sensitive` parameter determines if we should consider
    /// children with different capitalizations as being distinct names.
    ///
    /// If multiple children with the same name exist, the one that occurs
    /// first in the render list wins.
    fn get_name(&self, name: &WStr, case_sensitive: bool) -> Option<DisplayObject<'gc>> {
        if self.has_pending_removals {
            // Find matching children by searching the depth list
            let mut matching_render_children = if case_sensitive {
                self.depth_list
                    .iter()
                    .filter(|(_, child)| child.name().is_some_and(|n| n == name))
                    .collect::<Vec<_>>()
            } else {
                self.depth_list
                    .iter()
                    .filter(|(_, child)| child.name().is_some_and(|n| n.eq_ignore_case(name)))
                    .collect::<Vec<_>>()
            };

            // Sort so we can get the lowest depth child
            matching_render_children.sort_by_key(|&(depth, _child)| *depth);

            // First child will have the lowest depth
            matching_render_children
                .first()
                .map(|&(_depth, child)| child)
                .copied()
        } else {
            // TODO: Make a HashMap from name -> child?
            // But need to handle conflicting names (lowest in depth order takes priority).
            if case_sensitive {
                self.render_list
                    .iter()
                    .copied()
                    .find(|child| child.name().is_some_and(|n| n == name))
            } else {
                self.render_list
                    .iter()
                    .copied()
                    .find(|child| child.name().is_some_and(|n| n.eq_ignore_case(name)))
            }
        }
    }

    /// Get a child by it's render list position (ID).
    fn get_id(&self, id: usize) -> Option<DisplayObject<'gc>> {
        self.render_list.get(id).copied()
    }

    /// Replace a child in the render list with another child in the same
    /// position.
    fn replace_id(&mut self, id: usize, child: DisplayObject<'gc>) {
        self.render_list_mut()[id] = child;
    }

    /// Insert a child into the render list at a particular position.
    fn insert_id(&mut self, id: usize, child: DisplayObject<'gc>) {
        self.render_list_mut().insert(id, child);
    }

    /// Push a child onto the end of the render list.
    fn push_id(&mut self, child: DisplayObject<'gc>) {
        self.render_list_mut().push(child);
    }

    /// Get the number of children on the render list.
    fn num_children(&self) -> usize {
        self.render_list.len()
    }

    pub fn mouse_children(&self) -> bool {
        self.mouse_children
    }

    pub fn set_mouse_children(&mut self, mouse_children: bool) {
        self.mouse_children = mouse_children;
    }

    pub fn is_action_script_3(&self) -> bool {
        self.is_action_script_3
    }

    pub fn set_is_action_script_3(&mut self, is_action_script_3: bool) {
        self.is_action_script_3 = is_action_script_3;
    }

    /// Insert a child at a given render list position.
    ///
    /// If the child is already a child of another container, you must remove
    /// it from that container before calling this method. If it's already a
    /// member of this container, do not remove it, as we will need to take
    /// care of that. In that case, note that all children after the old
    /// position will be shifted back by one, which must be taken into account
    /// when calculating future insertion IDs.
    ///
    /// `parent` should be the display object that owns this container.
    ///
    /// All children at or after the given ID will be shifted down in the
    /// render list. The child will *not* be put onto the depth list.
    fn insert_at_id(&mut self, child: DisplayObject<'gc>, id: usize) {
        if let Some(old_id) = self
            .render_list
            .iter()
            .position(|x| DisplayObject::ptr_eq(*x, child))
        {
            match old_id.cmp(&id) {
                Ordering::Less if id < self.render_list.len() => {
                    self.render_list_mut()[old_id..=id].rotate_left(1)
                }
                Ordering::Less => self.render_list_mut()[old_id..id].rotate_left(1),
                Ordering::Greater if old_id < self.render_list.len() => {
                    self.render_list_mut()[id..=old_id].rotate_right(1)
                }
                Ordering::Greater => self.render_list_mut()[id..old_id].rotate_right(1),
                Ordering::Equal => {}
            }
        } else {
            self.render_list_mut().insert(id, child);
        }
    }

    /// Swap two children in the render list.
    ///
    /// No changes to the depth or render lists are made by this function.
    fn swap_at_id(&mut self, id1: usize, id2: usize) {
        self.render_list_mut().swap(id1, id2);
    }

    /// Move an already-inserted child to a new location on the depth list.
    ///
    /// If another child already exists at the target depth, it will be moved
    /// to the current depth of the given child. Their relative positions in
    /// the render list will also be swapped. If the target depth is empty, the
    /// same steps occur, but the child will merely be removed and reinserted
    /// within the render list at a position after the closest previous child
    /// in the depth list.
    ///
    /// `parent` should be the display object that owns this container.
    fn swap_at_depth(
        &mut self,
        context: &mut UpdateContext<'gc>,
        parent: DisplayObject<'gc>,
        child: DisplayObject<'gc>,
        depth: Depth,
    ) {
        let prev_depth = child.depth();
        child.set_depth(depth);
        child.set_parent(context, Some(parent));

        if let Some(prev_child) = self.depth_list.insert(depth, child) {
            child.set_clip_depth(0);
            prev_child.set_depth(prev_depth);
            prev_child.set_clip_depth(0);
            prev_child.set_transformed_by_script(true);
            self.depth_list.insert(prev_depth, prev_child);

            let prev_position = self
                .render_list
                .iter()
                .position(|x| DisplayObject::ptr_eq(*x, prev_child));
            let next_position = self
                .render_list
                .iter()
                .position(|x| DisplayObject::ptr_eq(*x, child));

            match (prev_position, next_position) {
                (Some(prev_position), Some(next_position)) => {
                    self.render_list_mut().swap(prev_position, next_position);
                }
                _ => {
                    tracing::error!(
                        "ChildContainer::swap_at_depth: A child is missing in the render list"
                    );
                }
            }
        } else {
            self.depth_list.remove(&prev_depth);

            if let Some(old_position) = self
                .render_list
                .iter()
                .position(|x| DisplayObject::ptr_eq(*x, child))
            {
                self.render_list_mut().remove(old_position);
            }

            if let Some((_, below_child)) = self.depth_list.range(..depth).next_back() {
                let new_position = self
                    .render_list
                    .iter()
                    .position(|x| DisplayObject::ptr_eq(*x, *below_child))
                    .map(|pos| pos + 1)
                    .unwrap_or_else(|| {
                        tracing::error!(
                            "ChildContainer::swap_at_depth: below_child is missing in the render list"
                        );
                        0
                    });
                self.render_list_mut().insert(new_position, child);
            } else {
                self.render_list_mut().insert(0, child);
            }
        }
    }

    /// Yield children in the order they are rendered.
    pub fn iter_render_list<'a>(&'a self) -> impl 'a + Iterator<Item = DisplayObject<'gc>> {
        self.render_list.iter().copied()
    }

    /// Check for pending removals and update the pending removals flag
    pub fn update_pending_removals(&mut self) {
        self.has_pending_removals = self.depth_list.values().any(|c| c.avm1_pending_removal());
    }

    /// Set the pending_removals flag
    pub fn set_pending_removals(&mut self, pending: bool) {
        self.has_pending_removals = pending;
    }

    /// Should the removal of this clip be delayed to the start of the next frame
    ///
    /// Checks recursively for unload handlers
    pub fn should_delay_removal(
        activation: &mut Activation<'_, 'gc>,
        child: DisplayObject<'gc>,
    ) -> bool {
        // Do we have an unload event handler
        if let Some(mc) = child.as_movie_clip() {
            // If we have an unload handler, we need the delay
            if mc.has_unload_handler() {
                return true;
            // otherwise, check for a dynamic unload handler
            } else if let Some(obj) = child.object1()
                && obj.has_property(activation, istr!("onUnload"))
            {
                return true;
            }
        }

        // Otherwise, check children if we have them
        if let Some(c) = child.as_container() {
            for child in c.iter_render_list() {
                if Self::should_delay_removal(activation, child) {
                    return true;
                }
            }
        }

        false
    }

    /// Enqueue the given child and all sub-children for delayed removal at the start of the next frame
    ///
    /// This just moves the children to a negative depth
    /// Will also fire unload events, as they should occur when the removal is queued, not when it actually occurs
    fn queue_removal(child: DisplayObject<'gc>, context: &mut UpdateContext<'gc>) {
        if let Some(c) = child.as_container() {
            for child in c.iter_render_list() {
                Self::queue_removal(child, context);
            }
        }

        let cur_depth = child.depth();
        // Note that the depth returned by AS will be offset by the `AVM_DEPTH_BIAS`, so this is really `-(cur_depth+1+AVM_DEPTH_BIAS)`
        child.set_depth(-cur_depth - 1);
        child.set_avm1_pending_removal(true);

        if let Some(mc) = child.as_movie_clip() {
            // Clip events should still fire
            mc.event_dispatch(context, crate::events::ClipEvent::Unload);
        }
    }

    fn render_list_mut(&mut self) -> &mut Vec<DisplayObject<'gc>> {
        #[cfg(any(feature = "lifecycle_projection", test))]
        self.lifecycle_indices.take();
        Rc::make_mut(&mut self.render_list)
    }
}

pub struct RenderIter<'gc> {
    // We store an `Rc` cloned from the original `DisplayObjectContainer`.
    // Any modifications to the render list will call `Rc::make_mut` on
    // the `Rc` field stored by `ChildContainer`, which will leave this
    // `Rc` unaffected.
    src: Rc<Vec<DisplayObject<'gc>>>,
    i: usize,
    neg_i: usize,
}

impl<'gc> RenderIter<'gc> {
    fn from_container(src: DisplayObjectContainer<'gc>) -> Self {
        let src = src.raw_container().render_list.clone();
        let neg_i = src.len();
        Self { src, i: 0, neg_i }
    }
}

impl<'gc> Iterator for RenderIter<'gc> {
    type Item = DisplayObject<'gc>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i == self.neg_i {
            return None;
        }

        let this = self.src.get(self.i).copied();

        self.i += 1;

        this
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.neg_i - self.i;
        (len, Some(len))
    }
}

impl<'gc> DoubleEndedIterator for RenderIter<'gc> {
    fn next_back(&mut self) -> Option<DisplayObject<'gc>> {
        if self.i == self.neg_i {
            return None;
        }

        let this = self.src.get(self.neg_i - 1).copied();

        self.neg_i -= 1;

        this
    }
}

impl ExactSizeIterator for RenderIter<'_> {}

/// Lifecycle projection tied to one immutable render-list snapshot.
#[cfg(any(feature = "lifecycle_projection", test))]
pub struct LifecycleIter<'gc> {
    src: Rc<Vec<DisplayObject<'gc>>>,
    indices: Option<Rc<Vec<usize>>>,
    range: std::ops::Range<usize>,
}

#[cfg(any(feature = "lifecycle_projection", test))]
impl<'gc> From<RenderIter<'gc>> for LifecycleIter<'gc> {
    fn from(iter: RenderIter<'gc>) -> Self {
        Self {
            src: iter.src,
            indices: None,
            range: iter.i..iter.neg_i,
        }
    }
}

#[cfg(any(feature = "lifecycle_projection", test))]
impl<'gc> LifecycleIter<'gc> {
    #[inline]
    pub fn is_projected(&self) -> bool {
        self.indices.is_some()
    }
}

#[cfg(any(feature = "lifecycle_projection", test))]
impl<'gc> Iterator for LifecycleIter<'gc> {
    type Item = DisplayObject<'gc>;

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.range.next()?;
        match &self.indices {
            Some(indices) => Some(self.src[indices[index]]),
            None => Some(self.src[index]),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

#[cfg(any(feature = "lifecycle_projection", test))]
impl<'gc> DoubleEndedIterator for LifecycleIter<'gc> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let index = self.range.next_back()?;
        match &self.indices {
            Some(indices) => Some(self.src[indices[index]]),
            None => Some(self.src[index]),
        }
    }
}

#[cfg(any(feature = "lifecycle_projection", test))]
impl ExactSizeIterator for LifecycleIter<'_> {}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;
    use crate::display_object::{avm2_button::Avm2Button, graphic::Graphic};
    use crate::player::PlayerBuilder;

    #[test]
    fn lifecycle_snapshot_skips_leaves_but_keeps_buttons_and_clips() {
        let player = PlayerBuilder::new().build();
        player
            .lock()
            .unwrap()
            .mutate_with_update_context(|context| {
                let movie = context.stage.movie();
                let parent = MovieClip::new(movie, context.gc());
                let leaf: DisplayObject<'_> = Graphic::empty(context).into();
                let clip: DisplayObject<'_> = MovieClip::new(movie, context.gc()).into();
                let button: DisplayObject<'_> = Avm2Button::empty_button(context).into();
                parent.replace_at_depth(context, leaf, 1);
                parent.replace_at_depth(context, clip, 2);
                parent.replace_at_depth(context, button, 3);
                let mut snapshot = parent.iter_lifecycle_list();
                assert!(snapshot.is_projected());
                assert_eq!(snapshot.len(), 2);
                assert_eq!(
                    snapshot.next_back().map(|child| child.as_ptr()),
                    Some(button.as_ptr())
                );
                assert_eq!(
                    snapshot.next().map(|child| child.as_ptr()),
                    Some(clip.as_ptr())
                );
                assert_eq!(snapshot.next().map(|child| child.as_ptr()), None);
                assert_eq!(snapshot.next_back().map(|child| child.as_ptr()), None);
                assert_eq!(
                    parent
                        .iter_render_list()
                        .map(|child| child.as_ptr())
                        .collect::<Vec<_>>(),
                    vec![leaf.as_ptr(), clip.as_ptr(), button.as_ptr()]
                );
            });
    }

    #[test]
    fn lifecycle_snapshot_uses_direct_path_for_all_active_children() {
        let player = PlayerBuilder::new().build();
        player
            .lock()
            .unwrap()
            .mutate_with_update_context(|context| {
                let movie = context.stage.movie();
                let parent = MovieClip::new(movie, context.gc());
                let clip: DisplayObject<'_> = MovieClip::new(movie, context.gc()).into();
                let button: DisplayObject<'_> = Avm2Button::empty_button(context).into();
                parent.replace_at_depth(context, clip, 1);
                parent.replace_at_depth(context, button, 2);
                let snapshot = parent.iter_lifecycle_list();
                assert!(!snapshot.is_projected());
                assert_eq!(snapshot.len(), 2);
            });
    }

    #[test]
    fn lifecycle_snapshot_survives_replacement_and_reordering() {
        let player = PlayerBuilder::new().build();
        player
            .lock()
            .unwrap()
            .mutate_with_update_context(|context| {
                let movie = context.stage.movie();
                let mut parent = MovieClip::new(movie, context.gc());
                let leaf: DisplayObject<'_> = Graphic::empty(context).into();
                let a: DisplayObject<'_> = MovieClip::new(movie, context.gc()).into();
                let b: DisplayObject<'_> = MovieClip::new(movie, context.gc()).into();
                let c: DisplayObject<'_> = MovieClip::new(movie, context.gc()).into();
                parent.replace_at_depth(context, a, 1);
                parent.replace_at_depth(context, leaf, 2);
                parent.replace_at_depth(context, b, 3);
                let old = parent.iter_lifecycle_list();
                parent.replace_at_depth(context, c, 1);
                parent.swap_at_index(context, 0, 2);
                assert_eq!(
                    old.map(|child| child.as_ptr()).collect::<Vec<_>>(),
                    vec![a.as_ptr(), b.as_ptr()]
                );
                assert_eq!(
                    parent
                        .iter_lifecycle_list()
                        .map(|child| child.as_ptr())
                        .collect::<Vec<_>>(),
                    vec![b.as_ptr(), c.as_ptr()]
                );
                parent.remove_child(context, c);
                assert_eq!(
                    parent
                        .iter_lifecycle_list()
                        .map(|child| child.as_ptr())
                        .collect::<Vec<_>>(),
                    vec![b.as_ptr()]
                );
            });
    }

    #[test]
    fn lifecycle_snapshot_invalidates_empty_and_all_active_lists() {
        let player = PlayerBuilder::new().build();
        player
            .lock()
            .unwrap()
            .mutate_with_update_context(|context| {
                let movie = context.stage.movie();
                let parent = MovieClip::new(movie, context.gc());
                let leaf: DisplayObject<'_> = Graphic::empty(context).into();
                let clip: DisplayObject<'_> = MovieClip::new(movie, context.gc()).into();
                assert_eq!(parent.iter_lifecycle_list().len(), 0);
                parent.replace_at_depth(context, leaf, 1);
                assert_eq!(parent.iter_lifecycle_list().len(), 0);
                parent.replace_at_depth(context, clip, 1);
                assert_eq!(
                    parent
                        .iter_lifecycle_list()
                        .map(|child| child.as_ptr())
                        .collect::<Vec<_>>(),
                    vec![clip.as_ptr()]
                );
                parent.replace_at_depth(context, leaf, 1);
                assert_eq!(parent.iter_lifecycle_list().len(), 0);
            });
    }

    #[test]
    #[ignore = "diagnostic traversal benchmark; prints timings without an FPS threshold"]
    fn lifecycle_snapshot_benchmark() {
        use std::hint::black_box;
        use std::time::Instant;
        let player = PlayerBuilder::new().build();
        player.lock().unwrap().mutate_with_update_context(|context| {
            for active_every in [1, 16, 0] {
                let movie = context.stage.movie();
                let parent = MovieClip::new(movie, context.gc());
                for index in 0..1024 {
                    let child = if active_every > 0 && index % active_every == 0 {
                        MovieClip::new(movie, context.gc()).into()
                    } else {
                        Graphic::empty(context).into()
                    };
                    parent.replace_at_depth(context, child, index + 1);
                }
                // Warm the projection independently of the measured steady state.
                black_box(parent.iter_lifecycle_list().count());
                let mut full = Vec::new();
                let mut projected = Vec::new();
                for sample in 0..31 {
                    // Alternate order to reduce systematic warm-cache bias.
                    for offset in 0..2 {
                        let filtered = (sample + offset) % 2 == 0;
                        let start = Instant::now();
                        for _ in 0..100 {
                            if filtered {
                                for child in parent.iter_lifecycle_list() {
                                    black_box(child);
                                }
                            } else {
                                for child in parent.iter_render_list() {
                                    black_box(child);
                                }
                            }
                        }
                        let samples = if filtered { &mut projected } else { &mut full };
                        samples.push(start.elapsed().as_nanos() / 100);
                    }
                }
                full.sort_unstable();
                projected.sort_unstable();
                println!("active_every={active_every} full_ns_p50={} full_ns_p95={} projected_ns_p50={} projected_ns_p95={}", full[15], full[29], projected[15], projected[29]);
            }
        });
    }
}
