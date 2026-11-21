//! Reuse the exact resolved RGBA8 input of an isolated blend. The blend with
//! the current backdrop still runs each time. Keys own no display/GPU handles.
use crate::{
    bitmap_cache::CacheImage,
    gpu_resources::{Texture, TextureView},
    gpu_runtime::ResourceOwner,
    graph_encoder::{PlannedList, PlannedOp},
    render_snapshot::{ResourceVersion, bitmap_version},
    resource_scheduler::{CacheTicket, Reclaim, ReclaimClass, ResourceScheduler},
    shape_recipe::ShapeRecipe,
};
use lru::LruCache;
use parking_lot::Mutex;
use ruffle_render::{
    commands::{Command, RenderBlendMode},
    matrix::Matrix,
    quality::StageQuality,
    region::RenderRect,
    transform::Transform,
};
use std::{
    any::Any,
    num::NonZeroUsize,
    sync::{Arc, atomic::Ordering},
};
use swf::BlendMode;

const MAX_OPS: usize = 32;
const MAX_DIMENSION: u32 = 256;
const MAX_ENTRIES: usize = 2048;
const QUIET_FRAMES: u64 = 120;

#[derive(Hash, PartialEq, Eq)]
struct Placement {
    linear: [u32; 4],
    translation: [i32; 2],
}

impl From<Matrix> for Placement {
    fn from(m: Matrix) -> Self {
        Self {
            linear: [m.a, m.b, m.c, m.d].map(f32::to_bits),
            translation: [m.tx.get(), m.ty.get()],
        }
    }
}

#[derive(Hash, PartialEq, Eq)]
struct DrawTransform(Placement, [i16; 8]);
impl DrawTransform {
    fn capture(t: &Transform) -> Option<Self> {
        if t.perspective_projection.is_some() || t.tz != 0.0 {
            return None;
        }
        let c = t.color_transform;
        Some(Self(
            t.matrix.into(),
            [
                c.r_multiply.get(),
                c.g_multiply.get(),
                c.b_multiply.get(),
                c.a_multiply.get(),
                c.r_add,
                c.g_add,
                c.b_add,
                c.a_add,
            ],
        ))
    }
}

#[derive(Hash, PartialEq, Eq)]
enum Pixels {
    Bitmap(ResourceVersion),
    Filtered(u64),
}

#[derive(Hash, PartialEq, Eq)]
enum Op {
    Bitmap(Pixels, DrawTransform, bool, u8, [u32; 4]),
    Shape(ResourceOwner, DrawTransform),
    Rect(u32, Placement),
    Blend(u8, [u32; 4]),
    End,
}

fn rect(r: RenderRect) -> [u32; 4] {
    [r.x, r.y, r.width, r.height]
}

#[derive(Hash, PartialEq, Eq)]
pub(super) struct Key {
    region: [u32; 4],
    quality: u8,
    ops: Vec<Op>,
}
impl Key {
    pub fn capture(list: &PlannedList, region: RenderRect, quality: StageQuality) -> Option<Self> {
        if region.width > MAX_DIMENSION
            || region.height > MAX_DIMENSION
            || list.samples_ancestor_layer()
        {
            return None;
        }
        let mut key = Self {
            region: rect(region),
            quality: quality as u8,
            ops: Vec::new(),
        };
        key.append(list)?;
        Some(key)
    }

    fn append(&mut self, list: &PlannedList) -> Option<()> {
        for draw in list.commands() {
            if self.ops.len() >= MAX_OPS {
                return None;
            }
            let op = match &draw.op {
                PlannedOp::Leaf(Command::RenderBitmap {
                    bitmap,
                    transform,
                    smoothing,
                    pixel_snapping,
                    region,
                }) => {
                    let handle = &*bitmap.0 as &dyn Any;
                    let id = handle
                        .downcast_ref::<CacheImage>()
                        .map_or(0, |c| c.tile_id.load(Ordering::Relaxed));
                    let source = if id != 0 {
                        Pixels::Filtered(id)
                    } else if handle.is::<CacheImage>()
                        || handle.is::<crate::bitmap_recipe::BitmapRecipe>()
                    {
                        Pixels::Bitmap(bitmap_version(bitmap).ok()?)
                    } else {
                        return None;
                    };
                    Op::Bitmap(
                        source,
                        DrawTransform::capture(transform)?,
                        *smoothing,
                        *pixel_snapping as u8,
                        [region.x_min, region.y_min, region.x_max, region.y_max],
                    )
                }
                PlannedOp::Leaf(Command::RenderShape { shape, transform }) => {
                    let recipe = (&*shape.0 as &dyn Any).downcast_ref::<ShapeRecipe>()?;
                    if !recipe.bitmaps.0.is_empty() {
                        return None;
                    }
                    Op::Shape(recipe.owner, DrawTransform::capture(transform)?)
                }
                PlannedOp::Leaf(Command::DrawRect { color, matrix }) => {
                    Op::Rect(color.to_rgba(), (*matrix).into())
                }
                PlannedOp::Blend(child, RenderBlendMode::Builtin(mode), bounds)
                    if !matches!(mode, BlendMode::Alpha | BlendMode::Erase) =>
                {
                    self.ops.push(Op::Blend(*mode as u8, rect(*bounds)));
                    self.append(child)?;
                    Op::End
                }
                _ => return None,
            };
            self.ops.push(op);
        }
        Some(())
    }
}

type Raster = (Texture, TextureView);
struct Entry {
    raster: Raster,
    used: u64,
    frame: u64,
    _ticket: CacheTicket,
}
struct State {
    entries: LruCache<Key, Entry>,
    frame: u64,
}
pub(super) struct BlendInputs {
    state: Mutex<State>,
    budget: ResourceScheduler,
}

impl BlendInputs {
    pub fn new(budget: &ResourceScheduler) -> Arc<Self> {
        let result = Arc::new(Self {
            state: Mutex::new(State {
                entries: LruCache::new(NonZeroUsize::new(MAX_ENTRIES).unwrap()),
                frame: 0,
            }),
            budget: budget.clone(),
        });
        budget.register(ReclaimClass::Cache, result.clone());
        result
    }
    pub fn begin_frame(&self) {
        let mut state = self.state.lock();
        state.frame += 1;
        while state
            .entries
            .peek_lru()
            .is_some_and(|(_, entry)| state.frame - entry.frame >= QUIET_FRAMES)
        {
            state.entries.pop_lru();
        }
    }
    pub fn lookup(&self, key: &Key) -> Option<Raster> {
        let mut state = self.state.lock();
        let frame = state.frame;
        let entry = state.entries.get_mut(key)?;
        entry.frame = frame;
        entry.used = self.budget.tick();
        Some(entry.raster.clone())
    }
    pub fn admit(&self, key: &Key) -> Option<CacheTicket> {
        let bytes = u64::from(key.region[2]) * u64::from(key.region[3]) * 4
            + (key.ops.capacity() * size_of::<Op>() + size_of::<Key>() + 512) as u64;
        // Admission may reclaim older entries through this same hook.
        self.budget.admit(ReclaimClass::Cache, bytes)
    }
    pub fn store(&self, key: Key, raster: Raster, ticket: CacheTicket) {
        let mut state = self.state.lock();
        let frame = state.frame;
        state.entries.put(
            key,
            Entry {
                raster,
                frame,
                used: self.budget.tick(),
                _ticket: ticket,
            },
        );
    }
}
impl Reclaim for BlendInputs {
    fn oldest(&self) -> Option<u64> {
        self.state
            .lock()
            .entries
            .peek_lru()
            .map(|(_, entry)| entry.used)
    }
    fn evict_oldest(&self) -> bool {
        self.state.lock().entries.pop_lru().is_some()
    }
}
