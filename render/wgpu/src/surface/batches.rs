use ruffle_render::commands::{Command, CommandList, RenderBlendMode};
use swf::BlendMode;

/// Consume intermediate blend/mask textures before building the next batch.
/// This bounds sibling intermediates, not recursion depth or an individual target.
pub(super) struct CommandBatches {
    unbatched: Option<CommandList>,
    commands: std::iter::Peekable<std::vec::IntoIter<Command>>,
    max_inputs: usize,
}

impl CommandBatches {
    pub fn new(commands: CommandList, width: u32, height: u32) -> Self {
        // Alpha/Erase can read an enclosing Layer while child commands are being
        // prepared. Keep the original preparation order for that entire list.
        // Shader blends receive their foreground/background at consumption time.
        let rgba_bytes = u64::from(width).max(1) * u64::from(height).max(1) * 4;
        let limit = ((32 * 1024 * 1024 / rgba_bytes) as usize).clamp(1, 8);
        let mut inputs = 0;
        let needs_batches = commands.iter().any(|command| {
            inputs += temporary_inputs(command);
            inputs > limit
        });
        let fallback = needs_batches && samples_ancestor_layer(&commands);
        let max_inputs = if needs_batches && !fallback {
            limit
        } else {
            usize::MAX
        };
        // Move the original vector through unchanged for stock and fallback.
        let (unbatched, commands) = if max_inputs == usize::MAX {
            (Some(commands), Vec::new())
        } else {
            (None, commands.into_commands())
        };
        if fallback {
            ruffle_render::profile::blend_batch_fallback();
        }
        Self {
            unbatched,
            commands: commands.into_iter().peekable(),
            max_inputs,
        }
    }
}

impl Iterator for CommandBatches {
    type Item = CommandList;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(commands) = self.unbatched.take() {
            return Some(commands);
        }
        let mut batch = CommandList::new();
        let mut inputs: usize = 0;
        while let Some(command) = self.commands.peek() {
            let cost = temporary_inputs(command);
            if cost != 0 && inputs != 0 && inputs.saturating_add(cost) > self.max_inputs {
                ruffle_render::profile::blend_batch_split();
                break;
            }
            inputs += cost;
            batch.push(self.commands.next().unwrap());
        }
        (!batch.is_empty()).then_some(batch)
    }
}

fn temporary_inputs(command: &Command) -> usize {
    match command {
        Command::Shared(commands) => commands.iter().map(temporary_inputs).sum(),
        Command::Blend(..) => 1,
        Command::RenderAlphaMask { .. } => 2,
        _ => 0,
    }
}

fn samples_ancestor_layer(commands: &CommandList) -> bool {
    commands.iter().any(|command| match command {
        Command::Blend(_, RenderBlendMode::Builtin(BlendMode::Alpha | BlendMode::Erase)) => true,
        Command::Shared(children) => samples_ancestor_layer(children),
        Command::Blend(children, _) => samples_ancestor_layer(children),
        Command::RenderAlphaMask {
            maskee_commands,
            mask_commands,
        } => samples_ancestor_layer(maskee_commands) || samples_ancestor_layer(mask_commands),
        _ => false,
    })
}
