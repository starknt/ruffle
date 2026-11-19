//! Reference masks may sit outside the subject's subtree. Observe both world
//! transforms at the scope boundary and separately retain the mask's geometry.
use super::{Entry, Key};
use crate::{
    context::RenderContext,
    display_object::{DisplayObject, TDisplayObject},
};
use ruffle_render::commands::Command;
use std::rc::Rc;

pub(in crate::display_object) fn draw<'gc>(
    mask: DisplayObject<'gc>,
    context: &mut RenderContext<'_, 'gc>,
) {
    if context.is_offscreen
        || !context.snapshot_allowed
        || !matches!(
            mask,
            DisplayObject::MovieClip(_) | DisplayObject::Graphic(_)
        )
    {
        mask.render_self(context);
        return;
    }
    let Some(recorder) = context.commands.stencil_recorder() else {
        mask.render_self(context);
        return;
    };
    let base = mask.base();
    let Some(cache) = context.snapshots.as_deref_mut() else {
        mask.render_self(context);
        return;
    };
    let Some(node) = base.snapshots.node(cache) else {
        mask.render_self(context);
        return;
    };
    let dimensions = context.renderer.viewport_dimensions();
    let key = Key {
        mask_geometry: true,
        generation: base.snapshots.generation.get(),
        inherited: context.transform_stack.transform(),
        own: ruffle_render::transform::Transform {
            matrix: mask.local_to_global_matrix(),
            ..Default::default()
        },
        viewport: (
            dimensions.width,
            dimensions.height,
            dimensions.scale_factor.to_bits(),
        ),
        view: context.stage.view_bounds(),
        quality: context.stage.quality(),
    };
    if let Some(entry) = node.entry.borrow().as_ref()
        && entry.key == key
    {
        cache.touch(&node);
        context
            .commands
            .push(Command::Shared(entry.commands.clone()));
        return;
    }
    base.snapshots.visited();
    let before = context.snapshot_barriers;
    let original = std::mem::replace(&mut context.commands, recorder);
    let recording = std::mem::replace(&mut context.snapshot_recording, true);
    mask.render_self(context);
    context.snapshot_recording = recording;
    let commands = Rc::new(std::mem::replace(&mut context.commands, original));
    if before == context.snapshot_barriers {
        context.snapshots.as_deref_mut().unwrap().insert(
            &node,
            Entry {
                key,
                commands: commands.clone(),
            },
        );
    }
    context.commands.push(Command::Shared(commands));
}
