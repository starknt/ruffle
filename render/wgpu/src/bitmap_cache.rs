//! Reconstructible cacheAsBitmap targets. Main-thread handles remain thread
//! confined; reclamation and history own native storage and numeric CPU data only.
use crate::{
    Texture,
    gpu_resources::{Device, try_sync_cell},
    gpu_runtime::{OwnerKind, RenderFailure, RenderFailureKind, ResourceOwner},
    render_snapshot::{
        RenderSnapshot, ResourceRecipe, ResourceVersion, SnapshotCommand, SnapshotNode,
    },
    resource_scheduler::{CacheTicket, Reclaim, ReclaimClass, ResourceScheduler},
};
use ruffle_render::{
    backend::BitmapCacheLease,
    bitmap::{BitmapHandle, BitmapHandleImpl, BitmapInfo},
    filters::Filter,
    quality::StageQuality,
    region::{RenderRect, RenderRegion},
};
use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex, OnceLock, Weak,
        atomic::{AtomicU64, Ordering},
    },
};

#[derive(Debug)]
pub struct CacheImage {
    pub owner: ResourceOwner,
    pub device_generation: u64,
    width: u32,
    height: u32,
    weak_self: Weak<Self>,
    version: AtomicU64,
    resident: OnceLock<Texture>,
    history: Mutex<Option<History>>,
}
impl BitmapHandleImpl for CacheImage {}
impl CacheImage {
    pub fn new(width: u32, height: u32, device: &Device) -> Result<Arc<Self>, RenderFailure> {
        device.runtime().check_health()?;
        if width == 0
            || height == 0
            || width > device.limits().max_texture_dimension_2d
            || height > device.limits().max_texture_dimension_2d
        {
            return Err(RenderFailure::new(
                RenderFailureKind::Validation,
                "bitmap cache exceeds device texture limits",
            ));
        }
        let owner = device.runtime().issue_owner(OwnerKind::BitmapCache)?;
        Ok(Arc::new_cyclic(|weak| Self {
            owner,
            device_generation: device.runtime().generation(),
            width,
            height,
            weak_self: weak.clone(),
            version: AtomicU64::new(1),
            resident: OnceLock::new(),
            history: Mutex::new(None),
        }))
    }
    pub fn resolve(&self, device: &Device) -> Result<&Texture, RenderFailure> {
        device.runtime().check_health()?;
        if self.device_generation != device.runtime().generation() {
            return Err(RenderFailure::new(
                RenderFailureKind::Validation,
                "cache belongs to another device",
            ));
        }
        try_sync_cell(&self.resident, || {
            let texture =
                device
                    .with_owner(self.owner)
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some("cacheAsBitmap"),
                        size: wgpu::Extent3d {
                            width: self.width,
                            height: self.height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING
                            | wgpu::TextureUsages::COPY_SRC
                            | wgpu::TextureUsages::COPY_DST
                            | wgpu::TextureUsages::RENDER_ATTACHMENT,
                        view_formats: &[],
                    })?;
            Ok(Texture {
                _allocation: ruffle_render::profile::GpuBitmap::new(
                    u64::from(self.width) * u64::from(self.height) * 4,
                ),
                view: texture.create_view(&Default::default())?,
                texture,
                repeating_linear: Default::default(),
                repeating_nearest: Default::default(),
                clamped_linear: Default::default(),
                clamped_nearest: Default::default(),
                copy_count: Default::default(),
                version: Default::default(),
            })
        })
    }
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Relaxed)
    }
    pub fn written(&self) -> Result<(), RenderFailure> {
        self.version
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
            .map(|_| ())
            .map_err(|_| {
                RenderFailure::new(
                    RenderFailureKind::Internal,
                    "cache content version exhausted",
                )
            })
    }
    pub fn retain(&self, scheduler: &ResourceScheduler) -> Arc<dyn BitmapCacheLease> {
        let bytes = u64::from(self.width) * u64::from(self.height) * 4;
        let entry = scheduler
            .admit(ReclaimClass::Cache, bytes)
            .and_then(|ticket| Some((self.weak_self.upgrade()?, ticket, scheduler.tick())));
        let lease = Arc::new(Retained {
            entry: Mutex::new(entry),
            scheduler: scheduler.clone(),
        });
        scheduler.register(ReclaimClass::Cache, lease.clone());
        lease
    }
    pub fn prepare_redraw(
        &self,
        snapshot: &RenderSnapshot,
        clear: swf::Color,
        quality: StageQuality,
        filters: &[Filter],
        scheduler: &ResourceScheduler,
    ) -> (RenderRegion, Option<History>) {
        let mut next = History::capture(snapshot, clear, quality, filters);
        let previous = self
            .history
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        let dirty = match (previous.as_ref(), next.as_ref()) {
            (Some(old), Some(new)) => old.changed(new),
            _ => RenderRegion::Full,
        };
        drop(previous);
        if let Some(history) = next.as_mut() {
            history.ticket = scheduler.admit(ReclaimClass::Cache, history.bytes());
            if history.ticket.is_none() {
                next = None;
            }
        }
        (dirty, next)
    }
    pub fn finish_redraw(&self, history: Option<History>) {
        *self
            .history
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = history;
    }
}

#[derive(Debug)]
struct Retained {
    entry: Mutex<Option<(Arc<CacheImage>, CacheTicket, u64)>>,
    scheduler: ResourceScheduler,
}
impl BitmapCacheLease for Retained {
    fn bitmap(&self) -> Option<BitmapInfo> {
        let mut entry = self
            .entry
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (image, _, age) = entry.as_mut()?;
        *age = self.scheduler.tick();
        Some(BitmapInfo {
            width: image.width,
            height: image.height,
            handle: BitmapHandle(image.clone()),
        })
    }
}
impl Reclaim for Retained {
    fn oldest(&self) -> Option<u64> {
        self.entry
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .map(|e| e.2)
    }
    fn evict_oldest(&self) -> bool {
        let entry = self
            .entry
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        entry.is_some()
    }
}

const MAX_HISTORY_NODES: usize = 4096;
const HISTORY_BYTES: u64 = 2 * 1024 * 1024;
#[derive(Clone, Debug, PartialEq)]
enum FilterKey {
    Matrix(swf::ColorMatrixFilter),
    Blur(swf::BlurFilter),
    Glow(swf::GlowFilter),
    Shadow(swf::DropShadowFilter),
    Bevel(swf::BevelFilter),
}
#[derive(Debug)]
pub struct History {
    ticket: Option<CacheTicket>,
    nodes: Vec<SnapshotNode>,
    dependencies: HashMap<ResourceVersion, Vec<ResourceVersion>>,
    clear: swf::Color,
    quality: StageQuality,
    filters: Vec<FilterKey>,
}
impl History {
    fn capture(
        snapshot: &RenderSnapshot,
        clear: swf::Color,
        quality: StageQuality,
        filters: &[Filter],
    ) -> Option<Self> {
        let mut nodes = Vec::new();
        fn flatten(input: &[SnapshotNode], nodes: &mut Vec<SnapshotNode>) -> Option<()> {
            for node in input {
                if nodes.len() >= MAX_HISTORY_NODES || node.region == RenderRegion::Full {
                    return None;
                }
                match &node.command {
                    SnapshotCommand::Shared(children) => flatten(children, nodes)?,
                    SnapshotCommand::Bitmap { .. }
                    | SnapshotCommand::Shape { .. }
                    | SnapshotCommand::Rect { .. }
                    | SnapshotCommand::Line { .. } => nodes.push(node.clone()),
                    // Stencil, ancestor Layer, shaders and Stage3D are indivisible
                    // dependency scopes. Preserve their entire valid coverage.
                    _ => return None,
                }
            }
            Some(())
        }
        flatten(&snapshot.commands, &mut nodes)?;
        let filters = filters
            .iter()
            .map(|filter| {
                Some(match filter {
                    Filter::ColorMatrixFilter(f) => FilterKey::Matrix(f.clone()),
                    Filter::BlurFilter(f) => FilterKey::Blur(f.clone()),
                    Filter::GlowFilter(f) => FilterKey::Glow(f.clone()),
                    Filter::DropShadowFilter(f) => FilterKey::Shadow(f.clone()),
                    Filter::BevelFilter(f) => FilterKey::Bevel(f.clone()),
                    _ => return None,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let dependencies: HashMap<_, _> = snapshot
            .recipes
            .iter()
            .filter_map(|recipe| match recipe {
                ResourceRecipe::Shape {
                    resource, bitmaps, ..
                } => Some((
                    *resource,
                    bitmaps
                        .iter()
                        .map(|(_, version)| *version)
                        .collect::<Vec<_>>(),
                )),
                _ => None,
            })
            .collect();
        let entries = nodes.len()
            + dependencies.len()
            + dependencies.values().map(Vec::len).sum::<usize>()
            + filters.len();
        if entries > MAX_HISTORY_NODES {
            return None;
        }
        let history = Self {
            ticket: None,
            nodes,
            dependencies,
            clear,
            quality,
            filters,
        };
        (history.bytes() <= HISTORY_BYTES).then_some(history)
    }
    fn bytes(&self) -> u64 {
        (std::mem::size_of::<Self>()
            + self.nodes.capacity() * std::mem::size_of::<SnapshotNode>()
            + self.filters.capacity() * std::mem::size_of::<FilterKey>()
            + self.dependencies.capacity()
                * (std::mem::size_of::<(ResourceVersion, Vec<ResourceVersion>)>() + 16)
            + self
                .dependencies
                .values()
                .map(|v| v.capacity() * std::mem::size_of::<ResourceVersion>())
                .sum::<usize>()) as u64
    }
    fn changed(&self, next: &Self) -> RenderRegion {
        if self.clear != next.clear || self.quality != next.quality || self.filters != next.filters
        {
            return RenderRegion::Full;
        }
        let mut dirty = RenderRegion::Empty;
        for index in 0..self.nodes.len().max(next.nodes.len()) {
            let old = self.nodes.get(index);
            let new = next.nodes.get(index);
            let dependency_changed = new.is_some_and(|node| match node.command {
                SnapshotCommand::Shape { resource, .. } => {
                    self.dependencies.get(&resource) != next.dependencies.get(&resource)
                }
                _ => false,
            });
            if old != new || dependency_changed {
                if let Some(node) = old {
                    dirty = dirty.union(node.region);
                }
                if let Some(node) = new {
                    dirty = dirty.union(node.region);
                }
            }
        }
        dirty
    }
}

pub fn filter_regions(
    dirty: RenderRect,
    valid: RenderRect,
    filters: &[Filter],
) -> (RenderRect, RenderRect) {
    let array = |r: RenderRect| {
        [
            i64::from(r.x),
            i64::from(r.y),
            i64::from(r.x + r.width),
            i64::from(r.y + r.height),
        ]
    };
    let rect = |r: [i64; 4]| RenderRect {
        x: r[0] as u32,
        y: r[1] as u32,
        width: (r[2] - r[0]) as u32,
        height: (r[3] - r[1]) as u32,
    };
    let graph = crate::filters::graph::FilterGraph::new(filters, true);
    let output = graph.affected_output(array(dirty), array(valid));
    let input = graph.required_input(output, array(valid));
    (rect(input), rect(output))
}
