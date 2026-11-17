use crate::backend::ShapeHandle;
use crate::bitmap::{BitmapHandle, PixelRegion, PixelSnapping};
use crate::matrix::Matrix;
use crate::pixel_bender::PixelBenderShaderHandle;
use crate::region::RenderRegion;
use crate::transform::Transform;
use std::rc::Rc;
use swf::{BlendMode, Color};

pub trait CommandHandler {
    fn render_shared(&mut self, commands: Rc<CommandList>) {
        commands.execute_ref(self);
    }
    fn render_bitmap(
        &mut self,
        bitmap: BitmapHandle,
        transform: Transform,
        smoothing: bool,
        pixel_snapping: PixelSnapping,
        region: PixelRegion,
    );
    fn render_stage3d(&mut self, bitmap: BitmapHandle, transform: Transform);
    fn render_shape(&mut self, shape: ShapeHandle, transform: Transform);
    fn render_alpha_mask(&mut self, maskee_commands: CommandList, mask_commands: CommandList);
    fn draw_rect(&mut self, color: Color, matrix: Matrix);
    fn draw_line(&mut self, color: Color, matrix: Matrix);
    fn draw_line_rect(&mut self, color: Color, matrix: Matrix);
    fn push_mask(&mut self);
    fn activate_mask(&mut self);
    fn deactivate_mask(&mut self);
    fn pop_mask(&mut self);

    fn blend(&mut self, commands: CommandList, blend_mode: RenderBlendMode);
}

/// Holds either a normal BlendMode, or the shader for BlendMode.SHADER.
///
/// We cannot store the `PixelBenderShaderHandle` directly in `ExtendedBlendMode`,
/// since we need to remember the shader even if the blend mode is changed
/// to something else (so that the shader will still be used if we switch back)
#[derive(Debug, Clone)]
pub enum RenderBlendMode {
    Builtin(BlendMode),
    Shader(PixelBenderShaderHandle),
}

#[derive(Debug, Default, Clone)]
pub struct CommandList {
    commands: Vec<Command>,
    region: RenderRegion,
    temporary_inputs: usize,
    samples_ancestor_layer: bool,

    /// The number of mask regions in the process of being drawn.
    /// This is used to discard drawing commands of nested maskers, which Flash does not support.
    maskers_in_progress: u32,
}

impl CommandList {
    pub fn temporary_inputs(&self) -> usize {
        self.temporary_inputs
    }
    pub fn samples_ancestor_layer(&self) -> bool {
        self.samples_ancestor_layer
    }
    pub fn append(&mut self, mut other: CommandList) {
        debug_assert_eq!(other.maskers_in_progress, 0);
        self.region = self.region.union(other.region);
        self.temporary_inputs = self.temporary_inputs.saturating_add(other.temporary_inputs);
        self.samples_ancestor_layer |= other.samples_ancestor_layer;
        self.commands.append(&mut other.commands);
    }

    /// Counts command storage and all resources it can keep resident. Shared
    /// resources are intentionally charged once per reference (an upper bound).
    pub fn retained_bytes(&self) -> Option<usize> {
        let mut bytes =
            std::mem::size_of::<Self>() + self.commands.capacity() * std::mem::size_of::<Command>();
        for command in &self.commands {
            bytes = bytes.checked_add(match command {
                Command::RenderShape { shape, .. } => shape.0.retained_bytes()?,
                Command::Shared(commands) => commands.retained_bytes()?,
                Command::DrawRect { .. }
                | Command::DrawLine { .. }
                | Command::DrawLineRect { .. } => 0,
                _ => return None,
            })?;
        }
        Some(bytes)
    }

    fn execute_ref(&self, handler: &mut (impl CommandHandler + ?Sized)) {
        for command in &self.commands {
            // Do not materialize a cloned (large) Command enum on the stack.
            // Retained blocks contain only leaf draws or shared blocks.
            match command {
                Command::Shared(commands) => handler.render_shared(commands.clone()),
                Command::RenderShape { shape, transform } => {
                    handler.render_shape(shape.clone(), transform.clone())
                }
                Command::DrawRect { color, matrix } => handler.draw_rect(*color, *matrix),
                Command::DrawLine { color, matrix } => handler.draw_line(*color, *matrix),
                Command::DrawLineRect { color, matrix } => handler.draw_line_rect(*color, *matrix),
                // Shared lists from other producers still preserve all commands.
                other => other.clone().execute(handler),
            }
        }
    }
    pub fn iter(&self) -> std::slice::Iter<'_, Command> {
        self.commands.iter()
    }
    pub fn into_commands(self) -> Vec<Command> {
        self.commands
    }
    pub fn region(&self) -> RenderRegion {
        self.region
    }
    pub fn push(&mut self, command: Command) {
        self.region = self.region.union(command.region());
        self.temporary_inputs = self
            .temporary_inputs
            .saturating_add(command.temporary_inputs());
        self.samples_ancestor_layer |= match &command {
            Command::Blend(_, RenderBlendMode::Builtin(BlendMode::Alpha | BlendMode::Erase)) => {
                true
            }
            Command::Blend(children, _) => children.samples_ancestor_layer,
            Command::Shared(children) => children.samples_ancestor_layer,
            Command::RenderAlphaMask {
                maskee_commands,
                mask_commands,
            } => maskee_commands.samples_ancestor_layer || mask_commands.samples_ancestor_layer,
            _ => false,
        };
        self.commands.push(command);
    }
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    pub fn execute(self, handler: &mut (impl CommandHandler + ?Sized)) {
        for command in self.commands {
            command.execute(handler);
        }
    }

    pub fn drawing_mask(&self) -> bool {
        self.maskers_in_progress > 0
    }
}

impl Command {
    pub fn temporary_inputs(&self) -> usize {
        match self {
            Self::Blend(..) => 1,
            Self::RenderAlphaMask { .. } => 2,
            Self::Shared(commands) => commands.temporary_inputs,
            _ => 0,
        }
    }
    pub fn execute(self, handler: &mut (impl CommandHandler + ?Sized)) {
        match self {
            Command::Shared(commands) => handler.render_shared(commands),
            Command::RenderBitmap {
                bitmap,
                transform,
                smoothing,
                pixel_snapping,
                region,
            } => handler.render_bitmap(bitmap, transform, smoothing, pixel_snapping, region),
            Command::RenderShape { shape, transform } => handler.render_shape(shape, transform),
            Command::RenderStage3D { bitmap, transform } => {
                handler.render_stage3d(bitmap, transform)
            }
            Command::DrawRect { color, matrix } => handler.draw_rect(color, matrix),
            Command::DrawLine { color, matrix } => handler.draw_line(color, matrix),
            Command::DrawLineRect { color, matrix } => handler.draw_line_rect(color, matrix),
            Command::PushMask => handler.push_mask(),
            Command::ActivateMask => handler.activate_mask(),
            Command::DeactivateMask => handler.deactivate_mask(),
            Command::PopMask => handler.pop_mask(),
            Command::Blend(commands, blend_mode) => handler.blend(commands, blend_mode),
            Command::RenderAlphaMask {
                maskee_commands,
                mask_commands,
            } => handler.render_alpha_mask(maskee_commands, mask_commands),
        }
    }
}

impl CommandHandler for CommandList {
    #[inline]
    fn render_bitmap(
        &mut self,
        bitmap: BitmapHandle,
        transform: Transform,
        smoothing: bool,
        pixel_snapping: PixelSnapping,
        region: PixelRegion,
    ) {
        if self.maskers_in_progress <= 1 {
            self.push(Command::RenderBitmap {
                bitmap,
                transform,
                smoothing,
                pixel_snapping,
                region,
            });
        }
    }

    #[inline]
    fn render_stage3d(&mut self, bitmap: BitmapHandle, transform: Transform) {
        if self.maskers_in_progress <= 1 {
            self.push(Command::RenderStage3D { bitmap, transform });
        }
    }

    #[inline]
    fn render_shape(&mut self, shape: ShapeHandle, transform: Transform) {
        if self.maskers_in_progress <= 1 {
            self.push(Command::RenderShape { shape, transform });
        }
    }

    fn render_alpha_mask(&mut self, maskee_commands: CommandList, mask_commands: CommandList) {
        if self.maskers_in_progress <= 1 {
            self.push(Command::RenderAlphaMask {
                maskee_commands,
                mask_commands,
            });
        }
    }

    #[inline]
    fn draw_rect(&mut self, color: Color, matrix: Matrix) {
        if self.maskers_in_progress <= 1 {
            self.push(Command::DrawRect { color, matrix });
        }
    }

    #[inline]
    fn draw_line(&mut self, color: Color, matrix: Matrix) {
        if self.maskers_in_progress <= 1 {
            self.push(Command::DrawLine { color, matrix });
        }
    }

    #[inline]
    fn draw_line_rect(&mut self, color: Color, matrix: Matrix) {
        if self.maskers_in_progress <= 1 {
            self.push(Command::DrawLineRect { color, matrix });
        }
    }

    #[inline]
    fn push_mask(&mut self) {
        if self.maskers_in_progress == 0 {
            self.push(Command::PushMask);
        }
        self.maskers_in_progress += 1;
    }

    #[inline]
    fn activate_mask(&mut self) {
        self.maskers_in_progress -= 1;
        if self.maskers_in_progress == 0 {
            self.push(Command::ActivateMask);
        }
    }

    #[inline]
    fn deactivate_mask(&mut self) {
        if self.maskers_in_progress == 0 {
            self.push(Command::DeactivateMask);
        }
        self.maskers_in_progress += 1;
    }

    #[inline]
    fn pop_mask(&mut self) {
        self.maskers_in_progress -= 1;
        if self.maskers_in_progress == 0 {
            self.push(Command::PopMask);
        }
    }

    #[inline]
    fn blend(&mut self, commands: CommandList, blend_mode: RenderBlendMode) {
        if self.maskers_in_progress <= 1 {
            self.push(Command::Blend(commands, blend_mode));
        }
    }
}

#[derive(Debug, Clone)]
pub enum Command {
    Shared(Rc<CommandList>),
    RenderBitmap {
        bitmap: BitmapHandle,
        transform: Transform,
        smoothing: bool,
        pixel_snapping: PixelSnapping,
        region: PixelRegion,
    },
    RenderStage3D {
        bitmap: BitmapHandle,
        transform: Transform,
    },
    RenderShape {
        shape: ShapeHandle,
        transform: Transform,
    },
    RenderAlphaMask {
        maskee_commands: CommandList,
        mask_commands: CommandList,
    },
    DrawRect {
        color: Color,
        matrix: Matrix,
    },
    DrawLine {
        color: Color,
        matrix: Matrix,
    },
    DrawLineRect {
        color: Color,
        matrix: Matrix,
    },
    PushMask,
    ActivateMask,
    DeactivateMask,
    PopMask,
    Blend(CommandList, RenderBlendMode),
}

impl Command {
    pub fn region(&self) -> RenderRegion {
        match self {
            Self::Shared(commands) => commands.region(),
            Self::RenderBitmap {
                transform,
                pixel_snapping,
                region,
                ..
            } => {
                if transform.perspective_projection.is_some() || transform.tz != 0.0 {
                    return RenderRegion::Full;
                }
                let mut matrix = transform.matrix;
                pixel_snapping.apply(&mut matrix);
                RenderRegion::transformed(
                    [0.0, 0.0, region.width().into(), region.height().into()],
                    &matrix,
                )
            }
            Self::RenderShape { shape, transform } => {
                if transform.perspective_projection.is_some() || transform.tz != 0.0 {
                    return RenderRegion::Full;
                }
                shape.0.bounds().map_or(RenderRegion::Full, |b| {
                    RenderRegion::transformed(b, &transform.matrix)
                })
            }
            Self::DrawRect { matrix, .. }
            | Self::DrawLine { matrix, .. }
            | Self::DrawLineRect { matrix, .. } => {
                RenderRegion::transformed([0.0, 0.0, 1.0, 1.0], matrix)
            }
            Self::RenderStage3D { .. }
            | Self::Blend(_, RenderBlendMode::Shader(_))
            | Self::Blend(_, RenderBlendMode::Builtin(BlendMode::Alpha | BlendMode::Erase)) => {
                RenderRegion::Full
            }
            Self::Blend(commands, _) => commands.region(),
            // A union is deliberately conservative, and propagates dependencies
            // even when one input is empty or disjoint.
            Self::RenderAlphaMask {
                maskee_commands,
                mask_commands,
            } => maskee_commands.region().union(mask_commands.region()),
            Self::PushMask | Self::ActivateMask | Self::DeactivateMask | Self::PopMask => {
                RenderRegion::Empty
            }
        }
    }
}
