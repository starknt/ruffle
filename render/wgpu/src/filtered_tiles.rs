//! Exact post-filter pixels for repeating small, single-draw animations.
//! The atlas owns no Flash objects, meshes, source bitmaps or writable targets.
use crate::{
    bitmap_cache::FilterKey,
    gpu_resources::{CommandEncoder, Device, Texture},
    gpu_runtime::RenderFailure,
    render_snapshot::{PreparedSnapshot, ResourceVersion, SnapshotCommand},
    resource_scheduler::{CacheTicket, Reclaim, ReclaimClass, ResourceScheduler},
};
use lru::LruCache;
use parking_lot::Mutex;
use ruffle_render::{filters::Filter, quality::StageQuality};
use std::sync::Arc;

const TILE: u32 = 32;
const SIDE: u32 = 2048;
const COLUMNS: u32 = SIDE / TILE;
const SLOTS: usize = (COLUMNS * COLUMNS) as usize;
const MAX_FILTERS: usize = 3;
const BUSY_DRAWS: usize = 8;
const QUIET_FRAMES: u32 = 120;

// An index collision only replaces a slot. Full command and filter equality
// below decides whether it is safe to copy the pixels.
#[derive(Clone, Copy, Hash, Eq, PartialEq)]
struct Placement {
    resource: ResourceVersion,
    linear: [u32; 4],
    translation: [i32; 2],
    size: [u32; 2],
}

#[derive(PartialEq)]
pub(super) struct Input {
    placement: Placement,
    command: SnapshotCommand,
    filters: Vec<FilterKey>,
    clear: swf::Color,
    quality: StageQuality,
    samples: u32,
}

impl Input {
    pub fn capture(
        prepared: &PreparedSnapshot,
        filters: &[Filter],
        clear: swf::Color,
        quality: StageQuality,
        samples: u32,
        width: u32,
        height: u32,
    ) -> Option<Self> {
        if width > TILE || height > TILE || !(1..=MAX_FILTERS).contains(&filters.len()) {
            return None;
        }
        let command = prepared.single_tile_draw()?;
        let (resource, transform) = match command {
            SnapshotCommand::Shape {
                resource,
                transform,
            }
            | SnapshotCommand::Bitmap {
                resource,
                transform,
                ..
            } => (*resource, transform),
            _ => return None,
        };
        if transform.perspective_projection.is_some() || transform.tz != 0.0 {
            return None;
        }
        let m = transform.matrix;
        Some(Self {
            placement: Placement {
                resource,
                linear: [m.a, m.b, m.c, m.d].map(f32::to_bits),
                translation: [m.tx.get(), m.ty.get()],
                size: [width, height],
            },
            command: command.clone(),
            filters: filters
                .iter()
                .map(FilterKey::capture)
                .collect::<Option<_>>()?,
            clear,
            quality,
            samples,
        })
    }
}

struct Atlas {
    texture: Texture,
    slots: LruCache<Placement, (Input, usize, u64)>,
    last_use: u64,
    _budget: CacheTicket,
}

impl Reclaim for Mutex<Option<Atlas>> {
    fn oldest(&self) -> Option<u64> {
        self.lock().as_ref().map(|a| a.last_use)
    }
    fn evict_oldest(&self) -> bool {
        self.lock().take().is_some()
    }
}

pub(super) struct FilteredTiles {
    atlas: Arc<Mutex<Option<Atlas>>>,
    scheduler: ResourceScheduler,
    draws: usize,
    quiet: u32,
}

impl FilteredTiles {
    pub fn new(scheduler: &ResourceScheduler) -> Self {
        let atlas = Arc::new(Mutex::new(None));
        scheduler.register(ReclaimClass::Cache, atlas.clone());
        Self {
            atlas,
            scheduler: scheduler.clone(),
            draws: 0,
            quiet: 0,
        }
    }

    pub fn begin_frame(&mut self) {
        if self.draws >= BUSY_DRAWS {
            self.quiet = 0;
        } else {
            self.quiet = self.quiet.saturating_add(1);
        }
        if self.quiet == QUIET_FRAMES {
            self.atlas.lock().take();
        }
        self.draws = 0;
    }

    pub fn replay(
        &mut self,
        input: &Input,
        destination: &Texture,
        encoder: &mut CommandEncoder,
    ) -> Option<u64> {
        self.draws += 1;
        let mut state = self.atlas.lock();
        let atlas = state.as_mut()?;
        let (stored, slot, id) = atlas.slots.get(&input.placement)?;
        if stored != input {
            return None;
        }
        copy(
            encoder,
            &atlas.texture,
            origin(*slot),
            destination,
            wgpu::Origin3d::ZERO,
            input.placement.size,
        );
        atlas.last_use = self.scheduler.tick();
        Some(*id)
    }

    pub fn store(
        &mut self,
        input: Input,
        source: &Texture,
        device: &Device,
        encoder: &mut CommandEncoder,
    ) -> Result<Option<u64>, RenderFailure> {
        if self.atlas.lock().is_none() {
            if self.draws < BUSY_DRAWS || device.limits().max_texture_dimension_2d < SIDE {
                return Ok(None);
            }
            // Admission may invoke our own eviction hook. Never hold the atlas
            // lock across admission or device allocation.
            let metadata = SLOTS
                * (size_of::<Input>()
                    + size_of::<Placement>()
                    + MAX_FILTERS * size_of::<FilterKey>()
                    + 128);
            let bytes = u64::from(SIDE) * u64::from(SIDE) * 4 + metadata as u64;
            let Some(ticket) = self.scheduler.admit(ReclaimClass::Cache, bytes) else {
                return Ok(None);
            };
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Small filtered draws"),
                size: wgpu::Extent3d {
                    width: SIDE,
                    height: SIDE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })?;
            *self.atlas.lock() = Some(Atlas {
                texture,
                slots: LruCache::unbounded(),
                last_use: self.scheduler.tick(),
                _budget: ticket,
            });
        }
        let mut state = self.atlas.lock();
        let Some(atlas) = state.as_mut() else {
            return Ok(None);
        };
        let slot = if let Some((_, slot, _)) = atlas.slots.pop(&input.placement) {
            slot
        } else if atlas.slots.len() == SLOTS {
            atlas.slots.pop_lru().unwrap().1.1
        } else {
            atlas.slots.len()
        };
        copy(
            encoder,
            source,
            wgpu::Origin3d::ZERO,
            &atlas.texture,
            origin(slot),
            input.placement.size,
        );
        let id = self
            .scheduler
            .tick()
            .checked_add(1)
            .expect("tile identity exhausted");
        atlas.slots.put(input.placement, (input, slot, id));
        atlas.last_use = self.scheduler.tick();
        Ok(Some(id))
    }
}

fn origin(slot: usize) -> wgpu::Origin3d {
    wgpu::Origin3d {
        x: slot as u32 % COLUMNS * TILE,
        y: slot as u32 / COLUMNS * TILE,
        z: 0,
    }
}
fn copy(
    encoder: &mut CommandEncoder,
    source: &Texture,
    from: wgpu::Origin3d,
    destination: &Texture,
    to: wgpu::Origin3d,
    size: [u32; 2],
) {
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            origin: from,
            ..source.as_image_copy()
        },
        wgpu::TexelCopyTextureInfo {
            origin: to,
            ..destination.as_image_copy()
        },
        wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
    );
}
