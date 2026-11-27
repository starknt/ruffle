//! Immutable preparation input. Only numeric identities and owned CPU data are
//! Send; the side table with renderer handles stays on the player thread.
use crate::{
    bitmap_recipe::BitmapRecipe,
    gpu_runtime::{RenderFailure, RenderFailureKind, ResourceId, ResourceOwner},
    shape_recipe::ShapeRecipe,
};
use fnv::FnvHashMap as HashMap;
use ruffle_render::{
    backend::ShapeHandle,
    bitmap::{Bitmap, BitmapHandle, PixelRegion, PixelSnapping},
    commands::{Command, CommandList, RenderBlendMode},
    matrix::Matrix,
    pixel_bender::PixelBenderShaderHandle,
    region::RenderRegion,
    shape_utils::OwnedDistilledShape,
    transform::Transform,
};
use std::{any::Any, rc::Rc, sync::Arc};
use swf::{BlendMode, Color};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResourceKey {
    Recipe { device: u64, owner: ResourceOwner },
    Native(ResourceId),
}
impl ResourceKey {
    fn device(self) -> u64 {
        match self {
            Self::Recipe { device, .. } => device,
            Self::Native(id) => id.device_generation(),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResourceVersion {
    pub key: ResourceKey,
    pub version: u64,
}
#[derive(Clone, Debug)]
pub enum ResourceRecipe {
    Bitmap {
        resource: ResourceVersion,
        pixels: Arc<Bitmap<'static>>,
        coverage: ruffle_render::bitmap::coverage::Coverage,
    },
    Shape {
        resource: ResourceVersion,
        static_definition: bool,
        geometry: Option<Arc<OwnedDistilledShape>>,
        scale: f32,
        bitmap_sizes: Vec<(u16, ruffle_render::bitmap::BitmapSize)>,
        bitmaps: Vec<(u16, ResourceVersion)>,
    },
    Resident(ResourceVersion),
}
#[derive(Clone, Debug, PartialEq)]
pub enum SnapshotBlend {
    Builtin(BlendMode),
    Shader(ResourceVersion),
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MaskCommand {
    Push,
    Activate,
    Deactivate,
    Pop,
}
#[derive(Clone, Debug, PartialEq)]
pub enum SnapshotCommand {
    Bitmap {
        resource: ResourceVersion,
        transform: Transform,
        smoothing: bool,
        pixel_snapping: PixelSnapping,
        region: PixelRegion,
    },
    Shape {
        resource: ResourceVersion,
        transform: Transform,
    },
    Stage3D {
        resource: ResourceVersion,
        transform: Transform,
    },
    Rect {
        color: Color,
        matrix: Matrix,
    },
    Line {
        color: Color,
        matrix: Matrix,
        rectangle: bool,
    },
    Mask(MaskCommand),
    Shared(Arc<[SnapshotNode]>),
    /// Index into the player-thread immutable block table for this capture.
    Retained(usize),
    Blend {
        children: Arc<[SnapshotNode]>,
        mode: SnapshotBlend,
    },
    AlphaMask {
        maskee: Arc<[SnapshotNode]>,
        mask: Arc<[SnapshotNode]>,
    },
}
#[derive(Clone, Debug, PartialEq)]
pub struct SnapshotNode {
    pub command: SnapshotCommand,
    pub region: RenderRegion,
}
#[derive(Clone, Debug)]
pub struct RenderSnapshot {
    pub commands: Arc<[SnapshotNode]>,
    pub recipes: Vec<ResourceRecipe>,
}
#[derive(Clone)]
enum Handle {
    Bitmap(BitmapHandle),
    Shape(ShapeHandle),
    Shader(PixelBenderShaderHandle),
}
pub(crate) fn bitmap_version(handle: &BitmapHandle) -> Result<ResourceVersion, RenderFailure> {
    Handle::Bitmap(handle.clone()).version()
}
impl Handle {
    fn version(&self) -> Result<ResourceVersion, RenderFailure> {
        match self {
            Self::Bitmap(handle) => {
                let value = &*handle.0 as &dyn Any;
                if let Some(recipe) = value.downcast_ref::<BitmapRecipe>() {
                    Ok(ResourceVersion {
                        key: ResourceKey::Recipe {
                            device: recipe.device_generation,
                            owner: recipe.owner,
                        },
                        version: recipe.version(),
                    })
                } else if let Some(cache) = value.downcast_ref::<crate::bitmap_cache::CacheImage>()
                {
                    Ok(ResourceVersion {
                        key: ResourceKey::Recipe {
                            device: cache.device_generation,
                            owner: cache.owner,
                        },
                        version: cache.version(),
                    })
                } else if let Some(texture) = value.downcast_ref::<crate::Texture>() {
                    Ok(ResourceVersion {
                        key: ResourceKey::Native(texture.texture.lease().id()),
                        version: texture.version.load(std::sync::atomic::Ordering::Relaxed),
                    })
                } else {
                    Err(foreign())
                }
            }
            Self::Shape(handle) => {
                let value = &*handle.0 as &dyn Any;
                if let Some(recipe) = value.downcast_ref::<ShapeRecipe>() {
                    Ok(ResourceVersion {
                        key: ResourceKey::Recipe {
                            device: recipe.device_generation,
                            owner: recipe.owner,
                        },
                        version: 1,
                    })
                } else if let Some(mesh) = value.downcast_ref::<crate::mesh::Mesh>() {
                    Ok(ResourceVersion {
                        key: ResourceKey::Native(mesh.vertex_buffer.lease().id()),
                        version: 0,
                    })
                } else {
                    Err(foreign())
                }
            }
            Self::Shader(handle) => Ok(ResourceVersion {
                key: ResourceKey::Native(
                    crate::pixel_bender::as_cache_holder(handle).resource_id(),
                ),
                version: 0,
            }),
        }
    }
}
fn foreign() -> RenderFailure {
    RenderFailure::new(
        RenderFailureKind::Validation,
        "snapshot resource belongs to another renderer",
    )
}

pub struct PreparedSnapshot {
    pub snapshot: RenderSnapshot,
    resources: Vec<(ResourceVersion, Handle)>,
    resource_indices: HashMap<ResourceVersion, usize>,
    retained: Vec<Rc<CommandList>>,
}
#[derive(Default)]
struct Capture {
    device: Option<u64>,
    retained: Vec<Rc<CommandList>>,
    known: HashMap<ResourceVersion, usize>,
    resources: Vec<(ResourceVersion, Handle)>,
    recipes: Vec<ResourceRecipe>,
}
impl Capture {
    fn immutable(commands: &CommandList, device: u64) -> bool {
        commands.certify_immutable(device, || {
            commands.iter().all(|cmd| match cmd {
                Command::DrawRect { .. }
                | Command::DrawLine { .. }
                | Command::DrawLineRect { .. } => true,
                Command::Shared(block) => Self::immutable(block, device),
                Command::RenderShape { shape, .. } => {
                    let handle = &*shape.0 as &dyn Any;
                    if let Some(recipe) = handle.downcast_ref::<ShapeRecipe>() {
                        recipe.device_generation == device
                            && recipe.bitmaps.0.is_empty()
                            && recipe
                                .mesh
                                .get()
                                .is_some_and(|m| m.retained_bytes.is_some())
                    } else if let Some(mesh) = handle.downcast_ref::<crate::mesh::Mesh>() {
                        mesh.vertex_buffer.lease().id().device_generation() == device
                            && mesh.retained_bytes.is_some()
                    } else {
                        false
                    }
                }
                _ => false,
            })
        })
    }
    fn resource(&mut self, handle: Handle) -> Result<ResourceVersion, RenderFailure> {
        let resource = handle.version()?;
        // Register before dependencies to make cyclic references bounded.
        match self.known.entry(resource) {
            std::collections::hash_map::Entry::Occupied(_) => return Ok(resource),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(usize::MAX);
            }
        }
        let recipe = match &handle {
            Handle::Bitmap(handle) => (&*handle.0 as &dyn Any)
                .downcast_ref::<BitmapRecipe>()
                .and_then(BitmapRecipe::pending_upload)
                .map(|(pixels, coverage)| ResourceRecipe::Bitmap {
                    resource,
                    pixels,
                    coverage,
                })
                .unwrap_or(ResourceRecipe::Resident(resource)),
            Handle::Shape(handle) => {
                // Published solid meshes need only their resource identity.
                // Textured meshes still carry the current bitmap versions.
                if let Some(shape) = (&*handle.0 as &dyn Any).downcast_ref::<ShapeRecipe>()
                    && (shape.mesh.get().is_none() || !shape.bitmaps.0.is_empty())
                {
                    let mut bitmaps = Vec::new();
                    let mut sources: Vec<_> = shape.bitmaps.0.iter().collect();
                    sources.sort_by_key(|(id, _)| **id);
                    for (&id, (_, bitmap)) in sources {
                        if let Some(bitmap) = bitmap {
                            bitmaps.push((id, self.resource(Handle::Bitmap(bitmap.clone()))?));
                        }
                    }
                    ResourceRecipe::Shape {
                        resource,
                        static_definition: shape.static_definition.get(),
                        geometry: shape.geometry.borrow().clone(),
                        scale: shape.scale,
                        bitmap_sizes: shape
                            .bitmaps
                            .0
                            .iter()
                            .map(|(id, (size, _))| (*id, *size))
                            .collect(),
                        bitmaps,
                    }
                } else {
                    ResourceRecipe::Resident(resource)
                }
            }
            Handle::Shader(_) => ResourceRecipe::Resident(resource),
        };
        self.recipes.push(recipe);
        *self.known.get_mut(&resource).unwrap() = self.resources.len();
        self.resources.push((resource, handle));
        Ok(resource)
    }
    fn commands(&mut self, commands: &CommandList) -> Result<Arc<[SnapshotNode]>, RenderFailure> {
        let mut nodes = Vec::with_capacity(commands.iter().len());
        self.commands_into(commands, &mut nodes)?;
        Ok(nodes.into())
    }

    fn commands_into(
        &mut self,
        commands: &CommandList,
        nodes: &mut Vec<SnapshotNode>,
    ) -> Result<(), RenderFailure> {
        let mut leaves = 0;
        for (index, command) in commands.iter().enumerate() {
            let region = commands.command_region(index);
            let command = match command {
                Command::RenderBitmap {
                    bitmap,
                    transform,
                    smoothing,
                    pixel_snapping,
                    region,
                } => SnapshotCommand::Bitmap {
                    resource: self.resource(Handle::Bitmap(bitmap.clone()))?,
                    transform: transform.clone(),
                    smoothing: *smoothing,
                    pixel_snapping: *pixel_snapping,
                    region: *region,
                },
                Command::RenderShape { shape, transform } => SnapshotCommand::Shape {
                    resource: self.resource(Handle::Shape(shape.clone()))?,
                    transform: transform.clone(),
                },
                Command::RenderStage3D { bitmap, transform } => SnapshotCommand::Stage3D {
                    resource: self.resource(Handle::Bitmap(bitmap.clone()))?,
                    transform: transform.clone(),
                },
                Command::DrawRect { color, matrix } => SnapshotCommand::Rect {
                    color: *color,
                    matrix: *matrix,
                },
                Command::DrawLine { color, matrix } | Command::DrawLineRect { color, matrix } => {
                    SnapshotCommand::Line {
                        color: *color,
                        matrix: *matrix,
                        rectangle: matches!(command, Command::DrawLineRect { .. }),
                    }
                }
                Command::Shared(children)
                    if self.device.is_some_and(|d| Self::immutable(children, d)) =>
                {
                    let index = self.retained.len();
                    self.retained.push(children.clone());
                    ruffle_render::profile::work(|w| w.prepared_chunk_hits += 1);
                    SnapshotCommand::Retained(index)
                }
                Command::Shared(children) if self.device.is_some() => {
                    // An unprepared directory is just painter order. Expanding it
                    // into this array avoids a worker allocation and scratch
                    // reservation per logical slot on cold or volatile frames.
                    self.commands_into(children, nodes)?;
                    continue;
                }
                Command::Shared(children) => SnapshotCommand::Shared(self.commands(children)?),
                Command::Blend(children, mode) => SnapshotCommand::Blend {
                    children: self.commands(children)?,
                    mode: match mode {
                        RenderBlendMode::Builtin(mode) => SnapshotBlend::Builtin(*mode),
                        RenderBlendMode::Shader(shader) => {
                            SnapshotBlend::Shader(self.resource(Handle::Shader(shader.clone()))?)
                        }
                    },
                },
                Command::RenderAlphaMask(inputs) => SnapshotCommand::AlphaMask {
                    maskee: self.commands(&inputs.maskee)?,
                    mask: self.commands(&inputs.mask)?,
                },
                Command::PushMask => SnapshotCommand::Mask(MaskCommand::Push),
                Command::ActivateMask => SnapshotCommand::Mask(MaskCommand::Activate),
                Command::DeactivateMask => SnapshotCommand::Mask(MaskCommand::Deactivate),
                Command::PopMask => SnapshotCommand::Mask(MaskCommand::Pop),
            };
            leaves += u64::from(!matches!(
                command,
                SnapshotCommand::Retained(_)
                    | SnapshotCommand::Shared(_)
                    | SnapshotCommand::Blend { .. }
                    | SnapshotCommand::AlphaMask { .. }
            ));
            nodes.push(SnapshotNode { command, region });
        }
        ruffle_render::profile::work(|w| w.captured_command_leaves += leaves);
        Ok(())
    }
}
impl PreparedSnapshot {
    /// A filtered tile can be replayed only if this version describes every
    /// pixel input. Textured meshes may refer to mutable BitmapData.
    pub(super) fn single_tile_draw(&self) -> Option<&SnapshotCommand> {
        let [node] = &*self.snapshot.commands else {
            return None;
        };
        let [(_, handle)] = self.resources.as_slice() else {
            return None;
        };
        match (&node.command, handle) {
            (SnapshotCommand::Shape { .. }, Handle::Shape(shape)) => {
                let raw = &*shape.0 as &dyn Any;
                let immutable = raw
                    .downcast_ref::<ShapeRecipe>()
                    .is_some_and(|r| r.bitmaps.0.is_empty());
                immutable.then_some(&node.command)
            }
            (SnapshotCommand::Bitmap { .. }, Handle::Bitmap(bitmap))
                if (&*bitmap.0 as &dyn Any).is::<BitmapRecipe>() =>
            {
                Some(&node.command)
            }
            _ => None,
        }
    }

    fn handle(&self, resource: ResourceVersion) -> Result<&Handle, RenderFailure> {
        self.resource_indices
            .get(&resource)
            .map(|index| &self.resources[*index].1)
            .ok_or_else(|| {
                RenderFailure::new(
                    RenderFailureKind::Validation,
                    "graph refers to an absent snapshot resource",
                )
            })
    }
    pub fn blend_mode(&self, mode: &SnapshotBlend) -> Result<RenderBlendMode, RenderFailure> {
        Ok(match mode {
            SnapshotBlend::Builtin(mode) => RenderBlendMode::Builtin(*mode),
            SnapshotBlend::Shader(resource) => match self.handle(*resource)? {
                Handle::Shader(shader) => RenderBlendMode::Shader(shader.clone()),
                _ => return Err(foreign()),
            },
        })
    }
    pub fn leaf(&self, node: &SnapshotNode) -> Result<Command, RenderFailure> {
        Ok(match &node.command {
            SnapshotCommand::Retained(index) => Command::Shared(self.retained[*index].clone()),
            SnapshotCommand::Bitmap {
                resource,
                transform,
                smoothing,
                pixel_snapping,
                region,
            } => match self.handle(*resource)? {
                Handle::Bitmap(bitmap) => Command::RenderBitmap {
                    bitmap: bitmap.clone(),
                    transform: transform.clone(),
                    smoothing: *smoothing,
                    pixel_snapping: *pixel_snapping,
                    region: *region,
                },
                _ => return Err(foreign()),
            },
            SnapshotCommand::Stage3D {
                resource,
                transform,
            } => match self.handle(*resource)? {
                Handle::Bitmap(bitmap) => Command::RenderStage3D {
                    bitmap: bitmap.clone(),
                    transform: transform.clone(),
                },
                _ => return Err(foreign()),
            },
            SnapshotCommand::Shape {
                resource,
                transform,
            } => match self.handle(*resource)? {
                Handle::Shape(shape) => Command::RenderShape {
                    shape: shape.clone(),
                    transform: transform.clone(),
                },
                _ => return Err(foreign()),
            },
            SnapshotCommand::Rect { color, matrix } => Command::DrawRect {
                color: *color,
                matrix: *matrix,
            },
            SnapshotCommand::Line {
                color,
                matrix,
                rectangle,
            } => {
                if *rectangle {
                    Command::DrawLineRect {
                        color: *color,
                        matrix: *matrix,
                    }
                } else {
                    Command::DrawLine {
                        color: *color,
                        matrix: *matrix,
                    }
                }
            }
            _ => {
                return Err(RenderFailure::new(
                    RenderFailureKind::Validation,
                    "composite used as a graph leaf",
                ));
            }
        })
    }

    pub fn capture(commands: &CommandList) -> Result<Self, RenderFailure> {
        Self::capture_for_device(commands, None)
    }

    /// A dense frame can change every transform while all geometry is already
    /// resident. Move a long, dependency-free command stream to the owner
    /// table for this submission. The next submission supplies fresh commands.
    pub(super) fn capture_owned(commands: CommandList, device: u64) -> Result<Self, RenderFailure> {
        if commands.iter().len() >= 64 && Capture::immutable(&commands, device) {
            let node = SnapshotNode {
                command: SnapshotCommand::Retained(0),
                region: commands.region(),
            };
            return Ok(Self {
                snapshot: RenderSnapshot {
                    commands: Arc::from([node]),
                    recipes: Vec::new(),
                },
                resources: Vec::new(),
                resource_indices: Default::default(),
                retained: vec![Rc::new(commands)],
            });
        }
        Self::capture_for_device(&commands, Some(device))
    }

    pub(super) fn capture_for_device(
        commands: &CommandList,
        device: Option<u64>,
    ) -> Result<Self, RenderFailure> {
        let capacity = commands.iter().len().min(64);
        let mut capture = Capture {
            device,
            retained: Vec::new(),
            known: HashMap::with_capacity_and_hasher(capacity, Default::default()),
            resources: Vec::with_capacity(capacity),
            recipes: Vec::with_capacity(capacity),
        };
        let commands = capture.commands(commands)?;
        Ok(Self {
            snapshot: RenderSnapshot {
                commands,
                recipes: capture.recipes,
            },
            resource_indices: capture.known,
            resources: capture.resources,
            retained: capture.retained,
        })
    }
    fn validate<T: crate::target::RenderTarget>(
        &self,
        backend: &crate::backend::WgpuRenderBackend<T>,
    ) -> Result<(), RenderFailure> {
        // Reject the whole snapshot before any candidate uploads or allocations.
        backend.descriptors.runtime.check_health()?;
        for (version, handle) in &self.resources {
            if version.key.device() != backend.descriptors.runtime.generation()
                || handle.version()? != *version
            {
                return Err(RenderFailure::new(
                    RenderFailureKind::Validation,
                    "snapshot resource version changed before preparation",
                ));
            }
        }
        Ok(())
    }
    pub fn prepare<T: crate::target::RenderTarget>(
        &self,
        backend: &mut crate::backend::WgpuRenderBackend<T>,
        bounds: ruffle_render::region::RenderRect,
        samples: u32,
        load: crate::pass_graph::Load,
    ) -> Result<crate::pass_graph::PassGraph, RenderFailure> {
        use crate::pass_graph::PassGraph;
        use crate::prepare::{
            Work,
            mesh::{ShapeInput, input_vertices},
        };
        self.validate(backend)?;
        enum Input<'a> {
            Graph(&'a RenderSnapshot),
            Shapes(Vec<(ResourceVersion, ShapeInput)>, u64),
        }
        enum Output {
            Graph(PassGraph),
            Shapes(
                Vec<(ResourceVersion, crate::prepare::mesh::PreparedMesh)>,
                Vec<(ResourceVersion, ShapeInput)>,
            ),
        }
        fn command_count(nodes: &[SnapshotNode]) -> usize {
            nodes
                .iter()
                .map(|node| {
                    1usize.saturating_add(match &node.command {
                        SnapshotCommand::Shared(children)
                        | SnapshotCommand::Blend { children, .. } => command_count(children),
                        SnapshotCommand::AlphaMask { maskee, mask } => {
                            command_count(maskee).saturating_add(command_count(mask))
                        }
                        _ => 0,
                    })
                })
                .fold(0usize, usize::saturating_add)
        }
        let commands = command_count(&self.snapshot.commands);
        let vertices = self
            .snapshot
            .recipes
            .iter()
            .filter_map(|r| match r {
                ResourceRecipe::Shape {
                    geometry: Some(geometry),
                    ..
                } => Some(input_vertices(geometry)),
                _ => None,
            })
            .fold(0usize, usize::saturating_add);
        let parallel = crate::prepare::parallel_workload(commands, vertices);
        let output_budget =
            (backend.descriptors.runtime.config().staging_limit / 4).clamp(4096, 16 * 1024 * 1024);
        let mesh_store = backend.descriptors.device.mesh_store().cloned();
        let mut shapes = self.snapshot.recipes.iter().filter_map(|r| match r {
            ResourceRecipe::Shape {
                resource,
                geometry: Some(geometry),
                scale,
                static_definition,
                bitmap_sizes,
                ..
            } => Some((
                *resource,
                ShapeInput {
                    cache: if *static_definition {
                        mesh_store.clone()
                    } else {
                        None
                    },
                    geometry: geometry.clone(),
                    scale: *scale,
                    bitmap_sizes: bitmap_sizes.clone(),
                },
            )),
            _ => None,
        });
        let jobs = std::iter::once(Work {
            bytes: (commands as u64).saturating_mul(4096).saturating_add(4096),
            value: Input::Graph(&self.snapshot),
        })
        .chain(std::iter::from_fn(|| {
            // Serial batches share the same bounded tessellator and validate
            // once at publication, just as a partition dispatched to a worker.
            let partition: Vec<_> = shapes.by_ref().take(32).collect();
            if partition.is_empty() {
                return None;
            }
            let input = partition.iter().fold(0u64, |bytes, (_, input)| {
                bytes.saturating_add((input_vertices(&input.geometry) as u64).saturating_mul(1024))
            });
            // Reserve proportionally for small partitions so four tasks can
            // share the device allowance. An underestimated tessellation still
            // takes the existing exclusive, bounded path after the wave joins.
            let budget = output_budget.min(input.saturating_add(4096));
            Some(Work {
                bytes: input.saturating_add(budget),
                value: Input::Shapes(partition, budget),
            })
        }));
        let preparation = backend.descriptors.runtime.preparation().clone();
        let generation = backend.preparation_generation.capture();
        let mut graph = None;
        let mut oversized = Vec::new();
        let current = preparation.for_each_batch(
            &generation,
            parallel,
            jobs,
            |input| match input {
                Input::Graph(snapshot) => {
                    PassGraph::compile(snapshot, bounds, samples, load).map(Output::Graph)
                }
                Input::Shapes(inputs, mut budget) => {
                    let mut prepared = Vec::with_capacity(inputs.len());
                    let mut remaining = inputs.into_iter();
                    // The partition's scratch permit bounds this tessellator.
                    // Reuse Lyon's internal work storage within the partition;
                    // drop it before publishing/releasing that permit.
                    let mut tessellator = ruffle_render::tessellator::ShapeTessellator::new();
                    while let Some((resource, input)) = remaining.next() {
                        match input.clone().prepare_with(&mut tessellator, budget) {
                            Ok(mesh) if mesh.retained_bytes() <= budget => {
                                budget -= mesh.retained_bytes();
                                prepared.push((resource, mesh));
                            }
                            Ok(_) => {
                                return Ok(Output::Shapes(
                                    prepared,
                                    std::iter::once((resource, input))
                                        .chain(remaining)
                                        .collect(),
                                ));
                            }
                            Err(error) if error.kind() == RenderFailureKind::BudgetExceeded => {
                                // The partition estimate was too small. Publish
                                // completed meshes, then prepare the remainder
                                // after every wave has released its outputs.
                                return Ok(Output::Shapes(
                                    prepared,
                                    std::iter::once((resource, input))
                                        .chain(remaining)
                                        .collect(),
                                ));
                            }
                            Err(error) => return Err(error),
                        }
                    }
                    Ok(Output::Shapes(prepared, Vec::new()))
                }
            },
            |batch| {
                // Recheck after join before GPU publication; only this closure
                // can access backend handles, bindings, or the device.
                self.validate(backend)?;
                for result in batch {
                    match result {
                        Output::Graph(value) => graph = Some(value),
                        Output::Shapes(meshes, deferred) => {
                            let mut publish = |resource, mesh| {
                                let Handle::Shape(handle) = self.handle(resource)? else {
                                    return Err(foreign());
                                };
                                let recipe = (&*handle.0 as &dyn Any)
                                    .downcast_ref::<ShapeRecipe>()
                                    .ok_or_else(foreign)?;
                                recipe.publish(backend, mesh)
                            };
                            for (resource, mesh) in meshes {
                                publish(resource, mesh)?;
                            }
                            oversized.extend(deferred);
                        }
                    }
                }
                Ok(())
            },
        )?;
        if !current {
            return Err(RenderFailure::new(
                RenderFailureKind::Validation,
                "stale native preparation generation",
            ));
        }
        for (resource, input) in oversized {
            let current = preparation.run_exclusive(
                &generation,
                |budget| input.prepare_bounded(budget),
                |mesh| {
                    self.validate(backend)?;
                    let Handle::Shape(handle) = self.handle(resource)? else {
                        return Err(foreign());
                    };
                    let recipe = (&*handle.0 as &dyn Any)
                        .downcast_ref::<ShapeRecipe>()
                        .ok_or_else(foreign)?;
                    recipe.publish(backend, mesh)
                },
            )?;
            if !current {
                return Err(RenderFailure::new(
                    RenderFailureKind::Validation,
                    "stale native preparation generation",
                ));
            }
        }
        for (_, handle) in &self.resources {
            if let Handle::Bitmap(handle) = handle {
                crate::as_texture(handle, &backend.descriptors.device)?;
            }
        }
        backend.descriptors.runtime.check_health()?;
        graph.ok_or_else(|| {
            RenderFailure::new(
                RenderFailureKind::Internal,
                "native preparation produced no graph",
            )
        })
    }
}

#[cfg(test)]
mod stage_capture_tests {
    use super::*;
    use ruffle_render::commands::CommandHandler;

    #[test]
    fn unprepared_stage_blocks_flatten_in_painter_order_for_workers() {
        let mut mixed = CommandList::new();
        mixed.push_mask();
        mixed.draw_rect(swf::Color::WHITE, Default::default());
        mixed.activate_mask();
        mixed.deactivate_mask();
        mixed.draw_rect(swf::Color::BLACK, Default::default());
        mixed.pop_mask();
        let mut stage = CommandList::new();
        stage.push(Command::Shared(Rc::new(mixed)));
        let prepared = PreparedSnapshot::capture_for_device(&stage, Some(1)).unwrap();
        assert_eq!(prepared.snapshot.commands.len(), 6);
        assert!(matches!(
            prepared.snapshot.commands[0].command,
            SnapshotCommand::Mask(MaskCommand::Push)
        ));
        assert!(matches!(
            prepared.snapshot.commands[5].command,
            SnapshotCommand::Mask(MaskCommand::Pop)
        ));
    }
}
