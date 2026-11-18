use crate::graph_encoder::{PlannedCommand, PlannedList};

// Small regions should not force repeated resolves of the entire destination.
// Storage is bounded by the byte watermark; this independent cap bounds metadata.
const MAX_INPUTS: usize = 64;

/// Consume graph scopes before preparing the next sibling set. Region-aware
/// storage costs include MSAA/resolve; a small object cap also bounds bindings.
/// Ancestor-layer sampling keeps its established preparation boundary intact.
pub(super) struct CommandBatches {
    unbatched: Option<PlannedList>,
    commands: std::iter::Peekable<std::vec::IntoIter<PlannedCommand>>,
    watermark: u64,
}
impl CommandBatches {
    pub fn new(commands: PlannedList, watermark: u64) -> Self {
        let needs_batches =
            commands.temporary_inputs() > MAX_INPUTS || commands.temporary_bytes() > watermark;
        let opaque_layer = needs_batches && commands.samples_ancestor_layer();
        let (unbatched, commands) = if !needs_batches || opaque_layer {
            (Some(commands), Vec::new())
        } else {
            (None, commands.into_commands())
        };
        if opaque_layer {
            ruffle_render::profile::blend_batch_fallback();
        }
        Self {
            unbatched,
            commands: commands.into_iter().peekable(),
            watermark,
        }
    }
}
impl Iterator for CommandBatches {
    type Item = PlannedList;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(commands) = self.unbatched.take() {
            return Some(commands);
        }
        let mut batch = PlannedList::default();
        while let Some(command) = self.commands.peek() {
            if command.inputs != 0
                && batch.temporary_inputs() != 0
                && (batch.temporary_inputs().saturating_add(command.inputs) > MAX_INPUTS
                    || batch
                        .temporary_bytes()
                        .saturating_add(command.transient_bytes)
                        > self.watermark)
            {
                ruffle_render::profile::blend_batch_split();
                break;
            }
            if let Some(command) = self.commands.next() {
                batch.push(command);
            }
        }
        (!batch.is_empty()).then_some(batch)
    }
}
