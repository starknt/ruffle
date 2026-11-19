use crate::backend::ShapeHandle;
use crate::bitmap::{BitmapHandle, PixelRegion, PixelSnapping};
use crate::matrix::Matrix;
use crate::pixel_bender::PixelBenderShaderHandle;
use crate::region::RenderRegion;
use crate::transform::Transform;
use std::{
    cell::{Cell, OnceCell, RefCell},
    rc::Rc,
};
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

#[derive(Debug, Default)]
pub struct CommandList {
    commands: Vec<Command>,
    has_commands: bool,
    retention: OnceCell<Box<Retention>>,
    region: RenderRegion,
    temporary_inputs: usize,
    samples_ancestor_layer: bool,

    /// The number of mask regions in the process of being drawn.
    /// This is used to discard drawing commands of nested maskers, which Flash does not support.
    maskers_in_progress: u32,
}

// Live command lists and inline blend/mask operands pay one pointer.
// Allocate this state only when observing a retained block.
#[derive(Default, Debug)]
struct Retention {
    regions: OnceCell<Box<[RenderRegion]>>,
    total_bytes: Cell<Option<usize>>,
    local_bytes: Cell<Option<usize>>,
    device: Cell<Option<u64>>,
    charge: RefCell<Option<StorageCharge>>,
    prepared: RefCell<Option<PreparedData>>,
    observed: Cell<bool>,
    required_bytes: Cell<usize>,
}

#[derive(Debug)]
struct PreparedData {
    value: Rc<dyn std::any::Any>,
    _charge: StorageCharge,
}

#[derive(Debug)]
struct StorageCharge {
    limit: usize,
    budget: Rc<Cell<usize>>,
    bytes: usize,
}
impl Drop for StorageCharge {
    fn drop(&mut self) {
        self.budget.set(self.budget.get() - self.bytes);
    }
}

impl Clone for CommandList {
    fn clone(&self) -> Self {
        Self {
            commands: self.commands.clone(),
            has_commands: self.has_commands,
            region: self.region,
            temporary_inputs: self.temporary_inputs,
            samples_ancestor_layer: self.samples_ancestor_layer,
            maskers_in_progress: self.maskers_in_progress,
            ..Self::default()
        }
    }
}

impl CommandList {
    fn retention(&self) -> &Retention {
        self.retention.get_or_init(Default::default)
    }
    /// A positive certificate covers only immutable resident draws on one device.
    /// Mutation of the command list revokes it, including after Rc::make_mut.
    pub fn certify_immutable(&self, device: u64, check: impl FnOnce() -> bool) -> bool {
        if self
            .retention
            .get()
            .is_some_and(|state| state.device.get() == Some(device))
        {
            return true;
        }
        if !check() {
            return false;
        }
        let state = self.retention();
        state.device.set(Some(device));
        state.required_bytes.set(0);
        true
    }
    pub fn prepared_draws<T: std::any::Any>(&self) -> Option<Rc<T>> {
        self.retention
            .get()?
            .prepared
            .borrow()
            .as_ref()?
            .value
            .clone()
            .downcast::<T>()
            .ok()
    }
    pub fn may_prepare_draws(&self, device: u64) -> bool {
        let Some(state) = self.retention.get() else {
            return false;
        };
        state.device.get() == Some(device)
            && state.observed.replace(true)
            && state.charge.borrow().as_ref().is_some_and(|charge| {
                state.required_bytes.get() <= charge.limit.saturating_sub(charge.budget.get())
            })
    }

    pub fn decline_draw_packet(&self) {
        self.retention().required_bytes.set(usize::MAX);
    }

    pub fn store_draws<T: std::any::Any>(&self, value: Rc<T>, bytes: usize) {
        let state = self.retention();
        state.prepared.borrow_mut().take();
        let charge = state.charge.borrow();
        let Some(charge) = charge.as_ref() else {
            return;
        };
        let bytes = bytes.saturating_add(size_of::<PreparedData>() + 64);
        if bytes > charge.limit.saturating_sub(charge.budget.get()) {
            state.required_bytes.set(bytes);
            return;
        }
        charge.budget.set(charge.budget.get() + bytes);
        state.required_bytes.set(0);
        *state.prepared.borrow_mut() = Some(PreparedData {
            value,
            _charge: StorageCharge {
                budget: charge.budget.clone(),
                bytes,
                limit: charge.limit,
            },
        });
    }
    pub fn temporary_inputs(&self) -> usize {
        self.temporary_inputs
    }
    pub fn samples_ancestor_layer(&self) -> bool {
        self.samples_ancestor_layer
    }
    pub fn append(&mut self, mut other: CommandList) {
        self.retention.take();
        debug_assert_eq!(other.maskers_in_progress, 0);
        self.region = self.region.union(other.region);
        self.temporary_inputs = self.temporary_inputs.saturating_add(other.temporary_inputs);
        self.samples_ancestor_layer |= other.samples_ancestor_layer;
        self.has_commands |= other.has_commands;
        self.commands.append(&mut other.commands);
    }

    fn local_bytes(&self) -> Option<usize> {
        if let Some(bytes) = self
            .retention
            .get()
            .and_then(|state| state.local_bytes.get())
        {
            return Some(bytes);
        }
        let mut bytes = size_of::<Self>()
            + size_of::<Retention>()
            + 32
            + self.commands.capacity() * size_of::<Command>()
            + self
                .retention
                .get()
                .and_then(|s| s.regions.get())
                .map_or(0, |regions| size_of_val(&**regions));
        for command in &self.commands {
            bytes = bytes.checked_add(command.pinned_bytes()?)?;
        }
        self.retention().local_bytes.set(Some(bytes));
        Some(bytes)
    }

    /// Stage admission can stop at a previously validated shared page.
    pub fn is_retainable(&self) -> bool {
        if self
            .retention
            .get()
            .is_some_and(|s| s.charge.borrow().is_some() || s.total_bytes.get().is_some())
        {
            return true;
        }
        self.commands.iter().all(|cmd| match cmd {
            Command::Shared(page) => page.is_retainable(),
            _ => cmd.pinned_bytes().is_some(),
        })
    }

    pub fn retained_bytes(&self) -> Option<usize> {
        if let Some(bytes) = self
            .retention
            .get()
            .and_then(|state| state.total_bytes.get())
        {
            return Some(bytes);
        }
        let mut bytes = self.local_bytes()?;
        for cmd in &self.commands {
            if let Command::Shared(child) = cmd {
                bytes = bytes.checked_add(child.retained_bytes()?)?;
            }
        }
        self.retention().total_bytes.set(Some(bytes));
        Some(bytes)
    }

    /// Additional storage this player would retain. Already charged shared
    /// blocks are constant-time stops, including blocks pinned by older frames.
    pub fn additional_bytes(&self, budget: &Rc<Cell<usize>>) -> Option<usize> {
        if let Some(state) = self.retention.get()
            && let Some(charge) = state.charge.borrow().as_ref()
        {
            return Rc::ptr_eq(&charge.budget, budget).then_some(0);
        }
        let mut bytes = self.local_bytes()?;
        for command in &self.commands {
            if let Command::Shared(child) = command {
                bytes = bytes.checked_add(child.additional_bytes(budget)?)?;
            }
        }
        Some(bytes)
    }

    /// Called after the complete graph and its size have been validated.
    pub fn charge_storage(&self, budget: &Rc<Cell<usize>>, limit: usize) {
        let state = self.retention();
        if state.charge.borrow().is_some() {
            return;
        }
        let bytes = self
            .local_bytes()
            .expect("validated retained command block");
        budget.set(budget.get() + bytes);
        *state.charge.borrow_mut() = Some(StorageCharge {
            budget: budget.clone(),
            bytes,
            limit,
        });
        for command in &self.commands {
            if let Command::Shared(child) = command {
                child.charge_storage(budget, limit);
            }
        }
    }

    pub fn execute_ref(&self, handler: &mut (impl CommandHandler + ?Sized)) {
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
    /// A retained leaf page remembers exact per-slot coverage. Directory
    /// children already carry their aggregate region on the shared block.
    pub fn from_render_page(
        items: impl ExactSizeIterator<Item = (Command, RenderRegion)>,
        leaf: bool,
    ) -> Self {
        let mut list = Self::with_capacity(items.len());
        let mut regions = leaf.then(|| Vec::with_capacity(items.len()));
        for (command, region) in items {
            if let Some(regions) = &mut regions {
                regions.push(region);
            }
            list.push_with_region(command, region);
        }
        if let Some(regions) = regions {
            let _ = list.retention().regions.set(regions.into_boxed_slice());
        }
        list
    }

    pub fn command_region(&self, index: usize) -> RenderRegion {
        self.retention
            .get()
            .and_then(|s| s.regions.get())
            .map_or_else(|| self.commands[index].region(), |regions| regions[index])
    }

    pub fn push(&mut self, command: Command) {
        self.retention.take();
        let region = command.region();
        self.push_with_region(command, region);
    }

    fn push_with_region(&mut self, command: Command, region: RenderRegion) {
        self.region = self.region.union(region);
        self.temporary_inputs = self
            .temporary_inputs
            .saturating_add(command.temporary_inputs());
        self.samples_ancestor_layer |= match &command {
            Command::Blend(_, RenderBlendMode::Builtin(BlendMode::Alpha | BlendMode::Erase)) => {
                true
            }
            Command::Blend(children, _) => children.samples_ancestor_layer,
            Command::Shared(children) => children.samples_ancestor_layer,
            Command::RenderAlphaMask(inputs) => {
                inputs.maskee.samples_ancestor_layer || inputs.mask.samples_ancestor_layer
            }
            _ => false,
        };
        self.has_commands |= !matches!(&command, Command::Shared(children) if children.is_empty());
        self.commands.push(command);
    }
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            commands: Vec::with_capacity(capacity),
            ..Default::default()
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        !self.has_commands
    }

    pub fn execute(self, handler: &mut (impl CommandHandler + ?Sized)) {
        for command in self.commands {
            command.execute(handler);
        }
    }

    /// Preserve Flash's nested-mask suppression while recording a stencil input.
    pub fn stencil_recorder(&self) -> Option<Self> {
        (self.maskers_in_progress == 1).then(|| Self {
            maskers_in_progress: 1,
            ..Default::default()
        })
    }

    pub fn drawing_mask(&self) -> bool {
        self.maskers_in_progress > 0
    }
}

impl Command {
    // Storage pinned directly by this command. Shared children are charged
    // separately when the page graph is admitted to its player's budget.
    fn pinned_bytes(&self) -> Option<usize> {
        match self {
            Command::RenderShape { shape, .. } => shape.0.retained_bytes(),
            Command::RenderBitmap { bitmap, .. } => bitmap.0.retained_bytes(),
            Command::PopMask
            | Command::DeactivateMask
            | Command::PushMask
            | Command::ActivateMask
            | Command::Shared(_)
            | Command::DrawRect { .. }
            | Command::DrawLine { .. }
            | Command::DrawLineRect { .. } => Some(0),
            _ => None,
        }
    }

    pub fn temporary_inputs(&self) -> usize {
        match self {
            Self::Blend(..) => 1,
            Self::RenderAlphaMask(_) => 2,
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
            Command::RenderAlphaMask(inputs) => {
                handler.render_alpha_mask(inputs.maskee, inputs.mask)
            }
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
            self.push(Command::RenderAlphaMask(Box::new(AlphaMaskInputs {
                maskee: maskee_commands,
                mask: mask_commands,
            })));
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

// Store the pair together. Ordinary draw slots only carry the size of their
// largest leaf or blend operand, including in short-lived and retained pages.
#[derive(Debug, Clone)]
pub struct AlphaMaskInputs {
    pub maskee: CommandList,
    pub mask: CommandList,
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
    RenderAlphaMask(Box<AlphaMaskInputs>),
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
            Self::RenderAlphaMask(inputs) => inputs.maskee.region().union(inputs.mask.region()),
            Self::PushMask | Self::ActivateMask | Self::DeactivateMask | Self::PopMask => {
                RenderRegion::Empty
            }
        }
    }
}

#[cfg(test)]
mod prepared_budget_tests {
    use super::*;

    #[test]
    fn draw_slots_do_not_reserve_space_for_two_effect_lists() {
        assert!(
            size_of::<Command>() < size_of::<[CommandList; 2]>(),
            "draw slot occupies {} bytes",
            size_of::<Command>()
        );
    }

    #[test]
    fn exhausted_packet_budget_retries_only_when_space_is_available() {
        let mut block = CommandList::new();
        block.draw_rect(swf::Color::WHITE, Matrix::IDENTITY);
        let storage = block.retained_bytes().unwrap();
        let used = Rc::new(Cell::new(2048));
        block.charge_storage(&used, storage + 2048);
        assert!(block.certify_immutable(1, || true));
        assert!(!block.may_prepare_draws(1));
        assert!(block.may_prepare_draws(1));
        block.store_draws(Rc::new(()), 1024);
        assert!(!block.may_prepare_draws(1));
        used.set(used.get() - 2048);
        assert!(block.may_prepare_draws(1));
    }
}
