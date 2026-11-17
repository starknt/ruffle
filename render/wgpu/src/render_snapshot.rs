//! Immutable preparation input. Only numeric identities and owned CPU data are
//! Send; the side table with renderer handles stays on the player thread.
use crate::{
    bitmap_recipe::BitmapRecipe,
    gpu_runtime::{RenderFailure, RenderFailureKind, ResourceId, ResourceOwner},
    shape_recipe::ShapeRecipe,
};
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
use std::{
    any::Any,
    collections::{HashMap, HashSet},
    sync::Arc,
};
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
#[derive(Default)]
struct Capture {
    known: HashSet<ResourceVersion>,
    resources: Vec<(ResourceVersion, Handle)>,
    recipes: Vec<ResourceRecipe>,
}
impl Capture {
    fn resource(&mut self, handle: Handle) -> Result<ResourceVersion, RenderFailure> {
        let resource = handle.version()?;
        if self.known.contains(&resource) {
            return Ok(resource);
        }
        // Register before dependencies to make cyclic references bounded.
        self.known.insert(resource);
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
                if let Some(shape) = (&*handle.0 as &dyn Any).downcast_ref::<ShapeRecipe>() {
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
                        bitmaps,
                    }
                } else {
                    ResourceRecipe::Resident(resource)
                }
            }
            Handle::Shader(_) => ResourceRecipe::Resident(resource),
        };
        self.recipes.push(recipe);
        self.resources.push((resource, handle));
        Ok(resource)
    }
    fn commands(&mut self, commands: &CommandList) -> Result<Arc<[SnapshotNode]>, RenderFailure> {
        commands
            .iter()
            .map(|command| {
                let region = command.region();
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
                    Command::DrawLine { color, matrix }
                    | Command::DrawLineRect { color, matrix } => SnapshotCommand::Line {
                        color: *color,
                        matrix: *matrix,
                        rectangle: matches!(command, Command::DrawLineRect { .. }),
                    },
                    Command::Shared(children) => SnapshotCommand::Shared(self.commands(children)?),
                    Command::Blend(children, mode) => SnapshotCommand::Blend {
                        children: self.commands(children)?,
                        mode: match mode {
                            RenderBlendMode::Builtin(mode) => SnapshotBlend::Builtin(*mode),
                            RenderBlendMode::Shader(shader) => SnapshotBlend::Shader(
                                self.resource(Handle::Shader(shader.clone()))?,
                            ),
                        },
                    },
                    Command::RenderAlphaMask {
                        maskee_commands,
                        mask_commands,
                    } => SnapshotCommand::AlphaMask {
                        maskee: self.commands(maskee_commands)?,
                        mask: self.commands(mask_commands)?,
                    },
                    Command::PushMask => SnapshotCommand::Mask(MaskCommand::Push),
                    Command::ActivateMask => SnapshotCommand::Mask(MaskCommand::Activate),
                    Command::DeactivateMask => SnapshotCommand::Mask(MaskCommand::Deactivate),
                    Command::PopMask => SnapshotCommand::Mask(MaskCommand::Pop),
                };
                Ok(SnapshotNode { command, region })
            })
            .collect()
    }
}
impl PreparedSnapshot {
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
        let mut capture = Capture::default();
        let commands = capture.commands(commands)?;
        Ok(Self {
            snapshot: RenderSnapshot {
                commands,
                recipes: capture.recipes,
            },
            resource_indices: capture
                .resources
                .iter()
                .enumerate()
                .map(|(index, (version, _))| (*version, index))
                .collect(),
            resources: capture.resources,
        })
    }
    pub fn resolve<T: crate::target::RenderTarget>(
        &self,
        backend: &mut crate::backend::WgpuRenderBackend<T>,
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
        for (_, handle) in &self.resources {
            match handle {
                Handle::Shape(handle) => {
                    if let Some(recipe) = (&*handle.0 as &dyn Any).downcast_ref::<ShapeRecipe>() {
                        recipe.resolve(backend)?;
                    }
                }
                Handle::Bitmap(handle) => {
                    crate::as_texture(handle, &backend.descriptors.device)?;
                }
                Handle::Shader(_) => {}
            }
        }
        backend.descriptors.runtime.check_health()
    }
}
