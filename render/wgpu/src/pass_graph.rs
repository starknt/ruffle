//! Stable native pass dependencies. Versions describe logical content; physical
//! temporary lifetimes span every version and are never shortened by a write.

use crate::gpu_runtime::{RenderFailure, RenderFailureKind};
use crate::render_snapshot::{
    MaskCommand, RenderSnapshot, ResourceKey, ResourceRecipe, ResourceVersion, SnapshotBlend,
    SnapshotCommand, SnapshotNode,
};
use ruffle_render::region::RenderRect;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use swf::{BlendMode, Color};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GraphResource {
    Output,
    External(ResourceKey),
    Temporary(u32),
    Stencil(u32),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Version {
    pub resource: GraphResource,
    pub version: u64,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Load {
    Clear(Color),
    Load,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureKey {
    pub width: u32,
    pub height: u32,
    pub samples: u32,
    pub format: wgpu::TextureFormat,
    pub usage: wgpu::TextureUsages,
}
impl TextureKey {
    fn bytes(self) -> Result<u64, RenderFailure> {
        let bytes = u64::from(self.width)
            .checked_mul(u64::from(self.height))
            .and_then(|v| v.checked_mul(u64::from(self.format.block_copy_size(None).unwrap_or(4))))
            .and_then(|v| v.checked_mul(u64::from(self.samples) + u64::from(self.samples > 1)))
            .ok_or_else(|| invalid("graph texture size overflow"))?;
        Ok(bytes)
    }
}
#[derive(Clone, Debug)]
pub struct Attachment {
    pub resource: GraphResource,
    pub bounds: RenderRect,
    pub load: Load,
    pub store: bool,
    pub samples: u32,
    pub resolve: bool,
    pub stencil: Option<GraphResource>,
    pub stencil_depth: u32,
}
#[derive(Clone, Debug)]
pub enum PassKind {
    Upload(ResourceRecipe),
    Prepare(ResourceRecipe),
    Clear,
    Draw(SnapshotNode),
    Mask(MaskCommand),
    Blend(SnapshotBlend),
    AlphaMask,
    Copy,
    Readback,
    Filter(usize),
    Opaque(SnapshotNode),
}
#[derive(Clone, Debug)]
pub struct Pass {
    pub kind: PassKind,
    pub reads: Vec<Version>,
    pub writes: Vec<Version>,
    pub dependencies: Vec<usize>,
    pub attachment: Option<Attachment>,
    pub bounds: RenderRect,
    pub submit_after: bool,
}
#[derive(Clone, Debug)]
pub struct Lifetime {
    pub resource: GraphResource,
    pub first: usize,
    pub last: usize,
    pub key: Option<TextureKey>,
    pub bytes: u64,
}
impl Lifetime {
    pub fn can_alias(&self, other: &Self) -> bool {
        matches!(
            (self.resource, other.resource),
            (GraphResource::Temporary(_), GraphResource::Temporary(_))
        ) && self.key.is_some()
            && self.key == other.key
            && (self.last < other.first || other.last < self.first)
    }
}
#[derive(Clone, Debug, Default)]
struct AccessState {
    version: u64,
    writer: Option<usize>,
    readers: Vec<usize>,
}
#[derive(Clone, Debug)]
pub struct PassGraph {
    pub passes: Vec<Pass>,
    pub commands: Arc<[SnapshotNode]>,
    states: HashMap<GraphResource, AccessState>,
    textures: HashMap<GraphResource, TextureKey>,
    shape_inputs: HashMap<ResourceKey, Vec<ResourceVersion>>,
    next_temporary: u32,
    next_stencil: u32,
    direct: bool,
}
fn invalid(message: &str) -> RenderFailure {
    RenderFailure::new(RenderFailureKind::Validation, message)
}
impl PassGraph {
    pub fn compile(
        snapshot: &RenderSnapshot,
        bounds: RenderRect,
        samples: u32,
        load: Load,
    ) -> Result<Self, RenderFailure> {
        if bounds.width == 0 || bounds.height == 0 || samples == 0 {
            return Err(invalid("empty graph attachment"));
        }
        let mut graph = Self {
            passes: Vec::new(),
            commands: snapshot.commands.clone(),
            states: HashMap::new(),
            textures: HashMap::new(),
            shape_inputs: HashMap::new(),
            next_temporary: 0,
            next_stencil: 0,
            direct: true,
        };
        for recipe in &snapshot.recipes {
            let resource = match recipe {
                ResourceRecipe::Bitmap { resource, .. }
                | ResourceRecipe::Shape { resource, .. }
                | ResourceRecipe::Resident(resource) => *resource,
            };
            graph
                .states
                .entry(GraphResource::External(resource.key))
                .or_default()
                .version = resource.version;
            if let ResourceRecipe::Shape {
                resource, bitmaps, ..
            } = recipe
            {
                graph
                    .shape_inputs
                    .insert(resource.key, bitmaps.iter().map(|(_, r)| *r).collect());
            }
        }
        // CPU recipes are an explicit preparation prefix, before any attachment writes.
        for recipe in &snapshot.recipes {
            match recipe {
                ResourceRecipe::Bitmap {
                    resource,
                    pixels,
                    coverage,
                } => {
                    let _ = pixels.data();
                    let _ = coverage;
                    graph
                        .states
                        .entry(GraphResource::External(resource.key))
                        .or_default()
                        .version = resource
                        .version
                        .checked_sub(1)
                        .ok_or_else(|| invalid("zero upload version"))?;
                    graph.add(
                        PassKind::Upload(recipe.clone()),
                        &[],
                        &[GraphResource::External(resource.key)],
                        None,
                        bounds,
                        true,
                    )?;
                }
                ResourceRecipe::Shape {
                    resource,
                    geometry: Some(_),
                    bitmaps,
                    ..
                } => {
                    graph
                        .states
                        .entry(GraphResource::External(resource.key))
                        .or_default()
                        .version = resource
                        .version
                        .checked_sub(1)
                        .ok_or_else(|| invalid("zero shape version"))?;
                    graph.add(
                        PassKind::Prepare(recipe.clone()),
                        &bitmaps
                            .iter()
                            .map(|(_, r)| GraphResource::External(r.key))
                            .collect::<Vec<_>>(),
                        &[GraphResource::External(resource.key)],
                        None,
                        bounds,
                        true,
                    )?;
                }
                _ => {}
            }
        }
        graph.list(
            &snapshot.commands,
            GraphResource::Output,
            bounds,
            samples,
            load,
            None,
        )?;
        Ok(graph)
    }
    pub fn direct_attachment(&self) -> bool {
        self.direct
    }
    pub fn temporary(
        &mut self,
        bounds: RenderRect,
        samples: u32,
    ) -> Result<GraphResource, RenderFailure> {
        let resource = GraphResource::Temporary(self.next_temporary);
        self.next_temporary = self
            .next_temporary
            .checked_add(1)
            .ok_or_else(|| invalid("too many graph temporaries"))?;
        let key = TextureKey {
            width: bounds.width,
            height: bounds.height,
            samples,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
        };
        key.bytes()?;
        self.textures.insert(resource, key);
        Ok(resource)
    }
    fn add(
        &mut self,
        kind: PassKind,
        reads: &[GraphResource],
        writes: &[GraphResource],
        attachment: Option<Attachment>,
        bounds: RenderRect,
        submit_after: bool,
    ) -> Result<usize, RenderFailure> {
        let index = self.passes.len();
        let mut dependencies = BTreeSet::new();
        // Never material-sort independent Flash commands. The hazard edges are
        // recorded separately so later merging cannot erase a WAR or WAW edge.
        if index != 0 {
            dependencies.insert(index - 1);
        }
        let mut read_versions = Vec::new();
        for &resource in reads {
            if read_versions
                .iter()
                .any(|v: &Version| v.resource == resource)
            {
                continue;
            }
            let state = self.states.entry(resource).or_default();
            if let Some(writer) = state.writer {
                dependencies.insert(writer);
            }
            state.readers.push(index);
            read_versions.push(Version {
                resource,
                version: state.version,
            });
        }
        let mut write_versions = Vec::new();
        for &resource in writes {
            if write_versions
                .iter()
                .any(|v: &Version| v.resource == resource)
            {
                continue;
            }
            let state = self.states.entry(resource).or_default();
            if let Some(writer) = state.writer {
                dependencies.insert(writer);
            }
            dependencies.extend(state.readers.drain(..).filter(|reader| *reader != index));
            state.version = state
                .version
                .checked_add(1)
                .ok_or_else(|| invalid("graph content version exhausted"))?;
            state.writer = Some(index);
            write_versions.push(Version {
                resource,
                version: state.version,
            });
        }
        self.passes.push(Pass {
            kind,
            reads: read_versions,
            writes: write_versions,
            dependencies: dependencies.into_iter().collect(),
            attachment,
            bounds,
            submit_after,
        });
        Ok(index)
    }
    fn list(
        &mut self,
        commands: &[SnapshotNode],
        target: GraphResource,
        bounds: RenderRect,
        samples: u32,
        load: Load,
        ancestor: Option<GraphResource>,
    ) -> Result<(), RenderFailure> {
        let stencil = GraphResource::Stencil(self.next_stencil);
        self.next_stencil = self
            .next_stencil
            .checked_add(1)
            .ok_or_else(|| invalid("too many stencil attachments"))?;
        let mut attachment = Attachment {
            resource: target,
            bounds,
            load,
            store: true,
            samples,
            resolve: samples > 1,
            stencil: None,
            stencil_depth: 0,
        };
        if matches!(load, Load::Clear(_)) {
            self.add(
                PassKind::Clear,
                &[],
                &[target],
                Some(attachment.clone()),
                bounds,
                true,
            )?;
        }
        attachment.load = Load::Load;
        self.commands_in(commands, &mut attachment, stencil, ancestor)?;
        if attachment.stencil_depth != 0 {
            return Err(invalid("unbalanced mask commands"));
        }
        Ok(())
    }
    fn commands_in(
        &mut self,
        commands: &[SnapshotNode],
        attachment: &mut Attachment,
        stencil: GraphResource,
        ancestor: Option<GraphResource>,
    ) -> Result<(), RenderFailure> {
        for node in commands {
            let target = attachment.resource;
            let mut reads = vec![target];
            let mut writes = vec![target];
            let mut bounds = node
                .region
                .clip(attachment.bounds)
                .unwrap_or(attachment.bounds);
            let kind = match &node.command {
                SnapshotCommand::Shared(children) => {
                    self.commands_in(children, attachment, stencil, ancestor)?;
                    continue;
                }
                SnapshotCommand::Mask(command) => {
                    match command {
                        MaskCommand::Push => {
                            attachment.stencil_depth = attachment
                                .stencil_depth
                                .checked_add(1)
                                .ok_or_else(|| invalid("stencil depth overflow"))?
                        }
                        MaskCommand::Pop => {
                            attachment.stencil_depth = attachment
                                .stencil_depth
                                .checked_sub(1)
                                .ok_or_else(|| invalid("mask pop without push"))?
                        }
                        _ if attachment.stencil_depth == 0 => {
                            return Err(invalid("mask phase without push"));
                        }
                        _ => {}
                    }
                    attachment.stencil = Some(stencil);
                    self.textures.entry(stencil).or_insert(TextureKey {
                        width: attachment.bounds.width,
                        height: attachment.bounds.height,
                        samples: attachment.samples,
                        format: wgpu::TextureFormat::Stencil8,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    });
                    reads = vec![stencil];
                    writes = vec![stencil];
                    PassKind::Mask(*command)
                }
                SnapshotCommand::Blend { children, mode } => {
                    self.direct = false;
                    // Ancestor-layer and custom-shader sampling cannot be bounded
                    // from the foreground's rectangle alone.
                    if matches!(
                        mode,
                        SnapshotBlend::Shader(_)
                            | SnapshotBlend::Builtin(
                                BlendMode::Alpha | BlendMode::Erase | BlendMode::Layer
                            )
                    ) || samples_ancestor(children)
                    {
                        bounds = attachment.bounds;
                    }
                    let foreground = self.temporary(bounds, attachment.samples)?;
                    let layer = if matches!(mode, SnapshotBlend::Builtin(BlendMode::Layer)) {
                        Some(foreground)
                    } else {
                        ancestor
                    };
                    self.list(
                        children,
                        foreground,
                        bounds,
                        attachment.samples,
                        Load::Clear(Color::TRANSPARENT),
                        layer,
                    )?;
                    reads.push(foreground);
                    match mode {
                        SnapshotBlend::Shader(shader) => {
                            reads.push(GraphResource::External(shader.key))
                        }
                        SnapshotBlend::Builtin(BlendMode::Alpha | BlendMode::Erase) => {
                            if let Some(layer) = ancestor {
                                reads.push(layer);
                            }
                        }
                        _ => {}
                    }
                    PassKind::Blend(mode.clone())
                }
                SnapshotCommand::AlphaMask { maskee, mask } => {
                    self.direct = false;
                    if samples_ancestor(maskee) || samples_ancestor(mask) {
                        bounds = attachment.bounds;
                    }
                    let content = self.temporary(bounds, attachment.samples)?;
                    let mask_target = self.temporary(bounds, attachment.samples)?;
                    self.list(
                        maskee,
                        content,
                        bounds,
                        attachment.samples,
                        Load::Clear(Color::TRANSPARENT),
                        ancestor,
                    )?;
                    self.list(
                        mask,
                        mask_target,
                        bounds,
                        attachment.samples,
                        Load::Clear(Color::TRANSPARENT),
                        ancestor,
                    )?;
                    reads.extend([content, mask_target]);
                    PassKind::AlphaMask
                }
                SnapshotCommand::Bitmap { resource, .. }
                | SnapshotCommand::Shape { resource, .. } => {
                    reads.push(GraphResource::External(resource.key));
                    if let Some(inputs) = self.shape_inputs.get(&resource.key) {
                        reads.extend(inputs.iter().map(|r| GraphResource::External(r.key)));
                    }
                    PassKind::Draw(node.clone())
                }
                SnapshotCommand::Stage3D { resource, .. } => {
                    reads.push(GraphResource::External(resource.key));
                    PassKind::Opaque(node.clone())
                }
                _ => PassKind::Draw(node.clone()),
            };
            if attachment.stencil_depth > 0 {
                reads.push(stencil);
            }
            self.add(
                kind,
                &reads,
                &writes,
                Some(attachment.clone()),
                bounds,
                attachment.stencil_depth == 0,
            )?;
        }
        Ok(())
    }
    pub fn copy(
        &mut self,
        source: GraphResource,
        destination: GraphResource,
        bounds: RenderRect,
    ) -> Result<usize, RenderFailure> {
        self.add(
            PassKind::Copy,
            &[source],
            &[destination],
            None,
            bounds,
            true,
        )
    }
    pub fn readback(
        &mut self,
        source: GraphResource,
        bounds: RenderRect,
    ) -> Result<usize, RenderFailure> {
        self.add(PassKind::Readback, &[source], &[], None, bounds, true)
    }
    pub fn lifetimes(&self) -> Vec<Lifetime> {
        let mut slots = HashMap::new();
        let mut lives: Vec<Lifetime> = Vec::new();
        for (index, pass) in self.passes.iter().enumerate() {
            for version in pass.reads.iter().chain(&pass.writes) {
                let slot = *slots.entry(version.resource).or_insert_with(|| {
                    let key = self.textures.get(&version.resource).copied();
                    lives.push(Lifetime {
                        resource: version.resource,
                        first: index,
                        last: index,
                        key,
                        bytes: key.and_then(|k| k.bytes().ok()).unwrap_or(0),
                    });
                    lives.len() - 1
                });
                lives[slot].last = index;
            }
        }
        lives
    }
    pub fn peak_temporary_bytes(&self) -> Result<u64, RenderFailure> {
        let lives = self.lifetimes();
        let mut peak = 0;
        for index in 0..self.passes.len() {
            let bytes = lives
                .iter()
                .filter(|l| l.first <= index && l.last >= index)
                .try_fold(0u64, |sum, life| {
                    sum.checked_add(life.bytes)
                        .ok_or_else(|| invalid("graph live bytes overflow"))
                })?;
            peak = peak.max(bytes);
        }
        Ok(peak)
    }
}
fn samples_ancestor(commands: &[SnapshotNode]) -> bool {
    commands.iter().any(|node| match &node.command {
        SnapshotCommand::Blend {
            mode: SnapshotBlend::Builtin(BlendMode::Alpha | BlendMode::Erase),
            ..
        } => true,
        SnapshotCommand::Blend { children, .. } | SnapshotCommand::Shared(children) => {
            samples_ancestor(children)
        }
        SnapshotCommand::AlphaMask { maskee, mask } => {
            samples_ancestor(maskee) || samples_ancestor(mask)
        }
        _ => false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_snapshot::{MaskCommand, SnapshotBlend, SnapshotCommand, SnapshotNode};
    use ruffle_render::{matrix::Matrix, region::RenderRegion};
    use std::sync::Arc;
    use swf::{BlendMode, Color};

    fn rect() -> SnapshotNode {
        SnapshotNode {
            command: SnapshotCommand::Rect {
                color: Color::RED,
                matrix: Matrix::IDENTITY,
            },
            region: RenderRegion::Finite([3.0, 5.0, 11.0, 13.0]),
        }
    }
    fn viewport() -> RenderRect {
        RenderRect {
            x: 0,
            y: 0,
            width: 32,
            height: 32,
        }
    }

    #[test]
    fn pass_graph_nested_masks_keep_order_and_temporary_lifetimes() {
        let mask = |command| SnapshotNode {
            command: SnapshotCommand::Mask(command),
            region: RenderRegion::Empty,
        };
        let blend = SnapshotNode {
            command: SnapshotCommand::Blend {
                children: Arc::from([rect()]),
                mode: SnapshotBlend::Builtin(BlendMode::Multiply),
            },
            region: RenderRegion::Finite([3.0, 5.0, 11.0, 13.0]),
        };
        let commands = Arc::from([
            mask(MaskCommand::Push),
            rect(),
            mask(MaskCommand::Activate),
            SnapshotNode {
                command: SnapshotCommand::AlphaMask {
                    maskee: Arc::from([blend]),
                    mask: Arc::from([rect()]),
                },
                region: RenderRegion::Finite([3.0, 5.0, 11.0, 13.0]),
            },
            mask(MaskCommand::Deactivate),
            rect(),
            mask(MaskCommand::Pop),
        ]);
        let snapshot = RenderSnapshot {
            commands,
            recipes: vec![],
        };
        let mut graph =
            PassGraph::compile(&snapshot, viewport(), 4, Load::Clear(Color::BLACK)).unwrap();
        let last_write = graph.passes.len() - 1;
        let read = graph.readback(GraphResource::Output, viewport()).unwrap();
        assert!(graph.passes[read].dependencies.contains(&last_write));
        assert!(
            graph
                .passes
                .iter()
                .enumerate()
                .all(|(i, pass)| pass.dependencies.iter().all(|p| *p < i))
        );
        assert!(
            graph
                .passes
                .iter()
                .any(|p| matches!(p.kind, PassKind::AlphaMask)
                    && p.attachment.as_ref().unwrap().stencil_depth == 1)
        );
        let mut temporaries: Vec<_> = graph
            .lifetimes()
            .into_iter()
            .filter(|l| matches!(l.resource, GraphResource::Temporary(_)))
            .collect();
        temporaries.sort_by_key(|l| l.first);
        assert_eq!(temporaries.len(), 3);
        // The two inputs of the alpha mask coexist through its compositing pass.
        assert!(!temporaries[0].can_alias(&temporaries[2]));
        assert!(
            graph
                .passes
                .iter()
                .filter_map(|p| p.attachment.as_ref())
                .any(|a| a.bounds
                    == RenderRect {
                        x: 3,
                        y: 5,
                        width: 8,
                        height: 8
                    })
        );
        assert!(!graph.direct_attachment());
    }

    #[test]
    fn pass_graph_copy_readback_write_tracks_raw_war_and_waw() {
        let snapshot = RenderSnapshot {
            commands: Arc::from([]),
            recipes: vec![],
        };
        let mut graph =
            PassGraph::compile(&snapshot, viewport(), 1, Load::Clear(Color::BLACK)).unwrap();
        let temporary = graph.temporary(viewport(), 1).unwrap();
        let copy = graph
            .copy(GraphResource::Output, temporary, viewport())
            .unwrap();
        let read = graph.readback(temporary, viewport()).unwrap();
        let overwrite = graph
            .copy(GraphResource::Output, temporary, viewport())
            .unwrap();
        assert!(graph.passes[overwrite].dependencies.contains(&copy));
        assert!(graph.passes[overwrite].dependencies.contains(&read));
        assert_eq!(graph.passes[copy].writes[0].version, 1);
        assert_eq!(graph.passes[overwrite].writes[0].version, 2);
        assert_eq!(graph.passes[read].reads[0].version, 1);
        let life = graph
            .lifetimes()
            .into_iter()
            .find(|l| l.resource == temporary)
            .unwrap();
        assert_eq!((life.first, life.last, life.bytes), (copy, overwrite, 4096));
    }

    #[test]
    fn pass_graph_only_exact_disjoint_temporaries_alias() {
        let snapshot = RenderSnapshot {
            commands: Arc::from([]),
            recipes: vec![],
        };
        let mut graph =
            PassGraph::compile(&snapshot, viewport(), 1, Load::Clear(Color::BLACK)).unwrap();
        let a = graph.temporary(viewport(), 1).unwrap();
        let b = graph.temporary(viewport(), 1).unwrap();
        graph.copy(GraphResource::Output, a, viewport()).unwrap();
        graph.readback(a, viewport()).unwrap();
        graph.copy(GraphResource::Output, b, viewport()).unwrap();
        let lifetimes = graph.lifetimes();
        let a = lifetimes.iter().find(|l| l.resource == a).unwrap();
        let b = lifetimes.iter().find(|l| l.resource == b).unwrap();
        assert!(a.can_alias(b));
        assert_eq!(graph.peak_temporary_bytes().unwrap(), 4096);
    }
}
