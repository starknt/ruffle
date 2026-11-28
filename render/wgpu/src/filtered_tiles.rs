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
const PAGE_SIDE: u32 = 512;
const COLUMNS: u32 = PAGE_SIDE / TILE;
const SLOTS_PER_PAGE: usize = (COLUMNS * COLUMNS) as usize;
const MAX_PAGES: usize = 16;
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

struct Page {
    texture: Texture,
    _budget: CacheTicket,
}

impl Page {
    fn try_new(
        device: &Device,
        scheduler: &ResourceScheduler,
    ) -> Result<Option<Self>, RenderFailure> {
        if device.limits().max_texture_dimension_2d < PAGE_SIDE {
            return Ok(None);
        }
        let texture_bytes = u64::from(PAGE_SIDE) * u64::from(PAGE_SIDE) * 4;
        let metadata = SLOTS_PER_PAGE
            * (size_of::<Input>()
                + size_of::<Placement>()
                + MAX_FILTERS * size_of::<FilterKey>()
                + 128);
        let Some(ticket) =
            scheduler.admit_optional(ReclaimClass::Cache, texture_bytes + metadata as u64)
        else {
            return Ok(None);
        };
        let texture = device.try_create_cache_texture(&wgpu::TextureDescriptor {
            label: Some("Small filtered draws"),
            size: wgpu::Extent3d {
                width: PAGE_SIDE,
                height: PAGE_SIDE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })?;
        Ok(texture.map(|texture| Self {
            texture,
            _budget: ticket,
        }))
    }
}

struct Atlas {
    pages: Vec<Page>,
    slots: LruCache<Placement, (Input, usize, u64)>,
    last_use: u64,
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
    attempted_page: bool,
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
            attempted_page: false,
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
        self.attempted_page = false;
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
            &atlas.pages[*slot / SLOTS_PER_PAGE].texture,
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
        let mut state = self.atlas.lock();
        if self.draws >= BUSY_DRAWS
            && !self.attempted_page
            && state.as_ref().is_none_or(|atlas| {
                atlas.pages.len() < MAX_PAGES
                    && atlas.slots.len() == atlas.pages.len() * SLOTS_PER_PAGE
                    && !atlas.slots.contains(&input.placement)
            })
        {
            self.attempted_page = true;
            // A trim can release the atlas during allocation. Recheck under the lock.
            drop(state);
            let page = Page::try_new(device, &self.scheduler)?;
            state = self.atlas.lock();
            if let Some(page) = page {
                state
                    .get_or_insert_with(|| Atlas {
                        pages: Vec::new(),
                        slots: LruCache::unbounded(),
                        last_use: self.scheduler.tick(),
                    })
                    .pages
                    .push(page);
            }
        }
        let Some(atlas) = state.as_mut() else {
            return Ok(None);
        };
        let slot = if let Some((_, slot, _)) = atlas.slots.pop(&input.placement) {
            slot
        } else if atlas.slots.len() == atlas.pages.len() * SLOTS_PER_PAGE {
            atlas.slots.pop_lru().unwrap().1.1
        } else {
            atlas.slots.len()
        };
        copy(
            encoder,
            source,
            wgpu::Origin3d::ZERO,
            &atlas.pages[slot / SLOTS_PER_PAGE].texture,
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
        y: (slot % SLOTS_PER_PAGE) as u32 / COLUMNS * TILE,
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
