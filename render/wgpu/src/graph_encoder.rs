//! Main-thread lowering of an immutable graph. Only this boundary reattaches
//! backend handles; preparation workers never receive them.
use crate::gpu_runtime::{RenderFailure, RenderFailureKind};
use crate::pass_graph::{GraphResource, Load, PassGraph, PassKind};
use crate::render_snapshot::{MaskCommand, PreparedSnapshot};
use ruffle_render::{
    commands::{Command, CommandList},
    region::RenderRect,
};
use std::collections::HashMap;

/// Native lowering retains graph-selected origins and lifetime costs through
/// nested composition; the surface never recaptures a display command list.
pub struct PlannedCommand {
    pub op: PlannedOp,
    pub region: ruffle_render::region::RenderRegion,
    pub transient_bytes: u64,
    pub inputs: usize,
    reads_layer: bool,
}
pub enum PlannedOp {
    Leaf(Command),
    Blend(
        PlannedList,
        ruffle_render::commands::RenderBlendMode,
        RenderRect,
    ),
    AlphaMask(PlannedList, PlannedList, RenderRect),
}
#[derive(Default)]
pub struct PlannedList {
    commands: Vec<PlannedCommand>,
    region: ruffle_render::region::RenderRegion,
    transient_bytes: u64,
    inputs: usize,
    reads_layer: bool,
}
impl PlannedList {
    pub fn push(&mut self, command: PlannedCommand) {
        self.region = self.region.union(command.region);
        self.transient_bytes = self.transient_bytes.saturating_add(command.transient_bytes);
        self.inputs = self.inputs.saturating_add(command.inputs);
        self.reads_layer |= command.reads_layer;
        self.commands.push(command);
    }
    pub fn region(&self) -> ruffle_render::region::RenderRegion {
        self.region
    }
    pub fn temporary_bytes(&self) -> u64 {
        self.transient_bytes
    }
    pub fn temporary_inputs(&self) -> usize {
        self.inputs
    }
    pub fn samples_ancestor_layer(&self) -> bool {
        self.reads_layer
    }
    pub fn into_commands(self) -> Vec<PlannedCommand> {
        self.commands
    }
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }
}

pub struct GraphEncoder {
    pub graph: PassGraph,
    prepared: PreparedSnapshot,
}
impl GraphEncoder {
    pub fn prepare<T: crate::target::RenderTarget>(
        backend: &mut crate::backend::WgpuRenderBackend<T>,
        commands: &CommandList,
        bounds: RenderRect,
        samples: u32,
        load: Load,
    ) -> Result<Self, RenderFailure> {
        let prepared = PreparedSnapshot::capture(commands)?;
        let graph = PassGraph::compile(&prepared.snapshot, bounds, samples, load)?;
        // All numeric versions are checked before the first upload/allocation.
        prepared.resolve(backend)?;
        Ok(Self { graph, prepared })
    }
    /// Lower stable dependency scopes to the existing adjacent chunk builder.
    /// The builder retains ownership of stencil/ancestor-Layer raster semantics;
    /// no GC traversal or recapture occurs here.
    pub fn into_commands(self) -> Result<PlannedList, RenderFailure> {
        let mut targets = HashMap::<GraphResource, PlannedList>::new();
        let invalid = || {
            RenderFailure::new(
                RenderFailureKind::Validation,
                "invalid graph target dependency",
            )
        };
        let costs: HashMap<_, _> = self
            .graph
            .lifetimes()
            .into_iter()
            .map(|life| (life.resource, life.bytes))
            .collect();
        for (index, pass) in self.graph.passes.iter().enumerate() {
            if pass
                .dependencies
                .iter()
                .any(|dependency| *dependency >= index)
            {
                return Err(invalid());
            }
            let Some(attachment) = &pass.attachment else {
                match pass.kind {
                    PassKind::Upload(_) | PassKind::Prepare(_) => continue,
                    _ => return Err(invalid()),
                }
            };
            let mut region = ruffle_render::region::RenderRegion::Empty;
            let mut transient_bytes = 0u64;
            let mut inputs = 0;
            let mut reads_layer = false;
            let op = match &pass.kind {
                PassKind::Clear => {
                    targets.entry(attachment.resource).or_default();
                    continue;
                }
                PassKind::Draw(node) | PassKind::Opaque(node) => {
                    region = node.region;
                    PlannedOp::Leaf(self.prepared.leaf(node)?)
                }
                PassKind::Mask(mask) => PlannedOp::Leaf(match mask {
                    MaskCommand::Push => Command::PushMask,
                    MaskCommand::Activate => Command::ActivateMask,
                    MaskCommand::Deactivate => Command::DeactivateMask,
                    MaskCommand::Pop => Command::PopMask,
                }),
                PassKind::Blend(mode) => {
                    let input = pass
                        .reads
                        .iter()
                        .find(|r| {
                            r.resource != attachment.resource
                                && matches!(r.resource, GraphResource::Temporary(_))
                        })
                        .ok_or_else(invalid)?;
                    let children = targets.remove(&input.resource).ok_or_else(invalid)?;
                    reads_layer = children.samples_ancestor_layer()
                        || matches!(
                            mode,
                            crate::render_snapshot::SnapshotBlend::Builtin(
                                swf::BlendMode::Alpha | swf::BlendMode::Erase
                            )
                        );
                    transient_bytes = costs.get(&input.resource).copied().unwrap_or(0);
                    inputs = 1;
                    region = rect_region(pass.bounds);
                    PlannedOp::Blend(children, self.prepared.blend_mode(mode)?, pass.bounds)
                }
                PassKind::AlphaMask => {
                    let mut input_resources = pass.reads.iter().filter(|r| {
                        r.resource != attachment.resource
                            && matches!(r.resource, GraphResource::Temporary(_))
                    });
                    let content = input_resources.next().ok_or_else(invalid)?.resource;
                    let mask = input_resources.next().ok_or_else(invalid)?.resource;
                    let content_commands = targets.remove(&content).ok_or_else(invalid)?;
                    let mask_commands = targets.remove(&mask).ok_or_else(invalid)?;
                    reads_layer = content_commands.samples_ancestor_layer()
                        || mask_commands.samples_ancestor_layer();
                    transient_bytes = costs
                        .get(&content)
                        .copied()
                        .unwrap_or(0)
                        .saturating_add(costs.get(&mask).copied().unwrap_or(0));
                    inputs = 2;
                    region = rect_region(pass.bounds);
                    PlannedOp::AlphaMask(content_commands, mask_commands, pass.bounds)
                }
                _ => return Err(invalid()),
            };
            targets
                .entry(attachment.resource)
                .or_default()
                .push(PlannedCommand {
                    op,
                    region,
                    transient_bytes,
                    inputs,
                    reads_layer,
                });
        }
        let commands = targets.remove(&GraphResource::Output).unwrap_or_default();
        if !targets.is_empty() {
            return Err(invalid());
        }
        Ok(commands)
    }
}

fn rect_region(rect: RenderRect) -> ruffle_render::region::RenderRegion {
    ruffle_render::region::RenderRegion::Finite([
        f64::from(rect.x),
        f64::from(rect.y),
        f64::from(rect.x) + f64::from(rect.width),
        f64::from(rect.y) + f64::from(rect.height),
    ])
}
