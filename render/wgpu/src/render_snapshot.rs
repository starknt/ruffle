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
}
// Only the VM cache carries local resource bindings. The shared node array
// can outlive the originating page or any previous frame that used it.
struct FrozenPage {
    device: u64,
    nodes: Arc<[SnapshotNode]>,
    shapes: Vec<(ResourceVersion, Handle)>,
}
#[derive(Default)]
struct Capture {
    device: Option<u64>,
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
    fn frozen_page(
        &mut self,
        page: &CommandList,
        device: u64,
    ) -> Result<Option<Arc<[SnapshotNode]>>, RenderFailure> {
        let frozen = if let Some(cached) = page.cached_snapshot::<FrozenPage>()
            && cached.device == device
        {
            ruffle_render::profile::work(|w| w.prepared_chunk_hits += 1);
            cached
        } else {
            // Cache leaf pages. A changed directory can still refer to the
            // unchanged pages on either side of the edit.
            if page.iter().any(|cmd| matches!(cmd, Command::Shared(_)))
                || !Self::immutable(page, device)
            {
                return Ok(None);
            }
            let mut local = Self::default();
            let nodes = local.commands(page)?;
            let bytes = size_of::<FrozenPage>()
                + size_of_val(&*nodes)
                + local.resources.capacity() * size_of::<(ResourceVersion, Handle)>();
            let frozen = Rc::new(FrozenPage {
                device,
                nodes,
                shapes: local.resources,
            });
            page.store_snapshot(frozen.clone(), bytes);
            frozen
        };
        // Certification admits only resident, untextured geometry and
        // primitives. Every shape therefore has exactly one external input.
        for (version, handle) in &frozen.shapes {
            if let std::collections::hash_map::Entry::Vacant(entry) = self.known.entry(*version) {
                entry.insert(self.resources.len());
                self.resources.push((*version, handle.clone()));
                self.recipes.push(ResourceRecipe::Resident(*version));
            }
        }
        Ok(Some(frozen.nodes.clone()))
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
                Command::Shared(children) if let Some(device) = self.device => {
                    if let Some(page) = self.frozen_page(children, device)? {
                        SnapshotCommand::Shared(page)
                    } else {
                        self.commands_into(children, nodes)?;
                        continue;
                    }
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
                SnapshotCommand::Shared(_)
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

    /// The synchronous encoder can consume certified draws directly at the
    /// requested stage quality. This list never enters a queued snapshot.
    pub(super) fn direct_resident_draws(commands: &CommandList, device: u64) -> bool {
        !commands.is_empty()
            && (commands.iter().len() >= 64
                || commands.iter().all(|cmd| matches!(cmd, Command::Shared(_))))
            && Capture::immutable(commands, device)
    }

    pub(super) fn capture_for_device(
        commands: &CommandList,
        device: Option<u64>,
    ) -> Result<Self, RenderFailure> {
        let capacity = commands.iter().len().min(64);
        let mut capture = Capture {
            device,
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
        let mut shapes = self.snapshot.recipes.iter().filter_map(|r| match r {
            ResourceRecipe::Shape {
                resource,
                geometry: Some(geometry),
                scale,
                bitmap_sizes,
                ..
            } => Some((
                *resource,
                ShapeInput {
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
    fn replacement_frame_carries_immutable_draws_after_the_vm_and_first_capture_drop() {
        let mut block = CommandList::new();
        block.draw_rect(Color::RED, Matrix::IDENTITY);
        let block = Rc::new(block);
        let weak_source = Rc::downgrade(&block);
        let mut frame = CommandList::new();
        frame.push(Command::Shared(block.clone()));
        drop(block);
        let first = PreparedSnapshot::capture_for_device(&frame, Some(1)).unwrap();
        frame.draw_rect(Color::BLUE, Matrix::IDENTITY);
        let replacement = PreparedSnapshot::capture_for_device(&frame, Some(1)).unwrap();
        drop(first);
        drop(frame);
        assert!(
            weak_source.upgrade().is_none(),
            "the frame still pins a VM command list"
        );
        let snapshot = replacement.snapshot;
        std::thread::spawn(move || {
            fn colors(nodes: &[SnapshotNode], out: &mut Vec<Color>) {
                for node in nodes {
                    match &node.command {
                        SnapshotCommand::Rect { color, .. } => out.push(*color),
                        SnapshotCommand::Shared(children) => colors(children, out),
                        _ => panic!("the captured rectangle cannot be decoded independently"),
                    }
                }
            }
            let mut drawn = Vec::new();
            colors(&snapshot.commands, &mut drawn);
            assert_eq!(drawn, vec![Color::RED, Color::BLUE]);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn mixed_frames_share_the_owned_page_within_the_display_tree_budget() {
        let mut page = CommandList::new();
        page.draw_rect(Color::RED, Matrix::IDENTITY);
        let page = Rc::new(page);
        let budget = Rc::new(std::cell::Cell::new(0usize));
        page.charge_storage(&budget, 1 << 20);
        let make_frame = || {
            let mut frame = CommandList::new();
            frame.push(Command::Shared(page.clone()));
            frame.push_mask();
            frame.activate_mask();
            frame.deactivate_mask();
            frame.pop_mask();
            PreparedSnapshot::capture_for_device(&frame, Some(3))
                .unwrap()
                .snapshot
        };
        let first = make_frame();
        let before = budget.get();
        let second = make_frame();
        let nodes = |s: &RenderSnapshot| match &s.commands[0].command {
            SnapshotCommand::Shared(nodes) => nodes.clone(),
            _ => panic!("expected an owned shared page"),
        };
        assert!(Arc::ptr_eq(&nodes(&first), &nodes(&second)));
        assert_eq!(budget.get(), before);
        drop(page);
        assert_eq!(budget.get(), 0);
    }

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
