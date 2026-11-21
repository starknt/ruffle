//! Opt-in Linux/NVIDIA experiment. Only fully retired command pools are lent
//! to the device. A later retirement can reclaim a different, parked encoder.
use alloc::{
    sync::{Arc, Weak},
    vec::Vec,
};
use ash::vk;
use core::{
    mem,
    sync::atomic::{AtomicU64, Ordering},
};
use parking_lot::Mutex;
use std::time::{Duration, Instant};

const BUSY: usize = 768;
const KEEP: usize = 512;
// Normal room traffic uses about 300 buffers per encoder. Preserve a larger
// set of already recorded buffers for the first frame when a portrait reopens.
const PRIMARY_BUFFERS: usize = 1280;
const BATCH: usize = 1024;
const QUIET: Duration = Duration::from_secs(4);
const INTERVAL: Duration = Duration::from_millis(150);

#[derive(Debug, Clone, Copy, PartialEq)]
enum Mode {
    Reset,
    Buffers,
    Pool,
    Split,
}

/// The encoder owns both pools while recording. The registry can access them
/// only after reset_all has returned every buffer and parked this partition.
#[derive(Debug)]
pub(super) struct Partition {
    family: u32,
    primary: Vec<vk::CommandBuffer>,
    overflow: Option<vk::CommandPool>,
}

impl Partition {
    pub fn allocate(
        &mut self,
        device: &ash::Device,
        mut info: vk::CommandBufferAllocateInfo<'_>,
    ) -> Result<Vec<vk::CommandBuffer>, vk::Result> {
        let remaining = PRIMARY_BUFFERS - self.primary.len();
        if remaining != 0 {
            info.command_buffer_count = info.command_buffer_count.min(remaining as u32);
            let buffers = unsafe { device.allocate_command_buffers(&info) }?;
            self.primary.extend_from_slice(&buffers);
            return Ok(buffers);
        }
        if let Some(pool) = self.overflow {
            info.command_pool = pool;
            return unsafe { device.allocate_command_buffers(&info) };
        }
        let desc = vk::CommandPoolCreateInfo::default()
            .queue_family_index(self.family)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let pool = unsafe { device.create_command_pool(&desc, None) }?;
        info.command_pool = pool;
        match unsafe { device.allocate_command_buffers(&info) } {
            Ok(buffers) => {
                self.overflow = Some(pool);
                Ok(buffers)
            }
            Err(error) => {
                unsafe { device.destroy_command_pool(pool, None) };
                Err(error)
            }
        }
    }
}

#[derive(Debug)]
struct Parked {
    buffers: Vec<vk::CommandBuffer>,
    partition: Option<Partition>,
}

#[derive(Debug)]
struct Retired {
    raw: vk::CommandPool,
    family: u32,
    free: Option<Parked>,
    armed: bool,
}

#[derive(Debug)]
pub(super) struct Slot {
    state: Arc<Mutex<Retired>>,
    parked: bool,
    pub(super) partition: Option<Partition>,
}
impl Slot {
    pub fn overflow_pool(&self) -> Option<vk::CommandPool> {
        self.partition.as_ref()?.overflow
    }

    pub fn checkout(&mut self, raw: &mut vk::CommandPool, free: &mut Vec<vk::CommandBuffer>) {
        if self.parked {
            let mut state = self.state.lock();
            *raw = state.raw;
            let owned = state
                .free
                .take()
                .expect("parked encoder owns its buffer list");
            *free = owned.buffers;
            self.partition = owned.partition;
            self.parked = false;
        }
    }
}

#[derive(Debug)]
struct Registry {
    next_scan: Instant,
    slots: Vec<Weak<Mutex<Retired>>>,
}

#[derive(Debug)]
pub(super) struct Trimmer {
    mode: Mode,
    registry: Mutex<Registry>,
    resets: AtomicU64,
    freed: AtomicU64,
    replaced: AtomicU64,
    reserved: AtomicU64,
}

impl Trimmer {
    pub fn from_env(vendor: u32) -> Option<Self> {
        if !cfg!(target_os = "linux") || vendor != crate::auxil::db::nvidia::VENDOR {
            return None;
        }
        let mode = match std::env::var("RUFFLE_VK_POOL_TRIM").as_deref() {
            Ok("reset") => Mode::Reset,
            Ok("buffers") => Mode::Buffers,
            Ok("pool") => Mode::Pool,
            Ok("split") => Mode::Split,
            _ => return None,
        };
        Some(Self {
            mode,
            registry: Mutex::new(Registry {
                next_scan: Instant::now() + QUIET,
                slots: Vec::new(),
            }),
            resets: AtomicU64::new(0),
            freed: AtomicU64::new(0),
            replaced: AtomicU64::new(0),
            reserved: AtomicU64::new(0),
        })
    }

    pub fn register(&self, raw: vk::CommandPool, family: u32) -> Slot {
        let state = Arc::new(Mutex::new(Retired {
            raw,
            family,
            free: None,
            armed: false,
        }));
        let mut registry = self.registry.lock();
        registry.slots.retain(|slot| slot.strong_count() != 0);
        registry.slots.push(Arc::downgrade(&state));
        Slot {
            state,
            parked: false,
            partition: (self.mode == Mode::Split).then(|| Partition {
                family,
                primary: Vec::new(),
                overflow: None,
            }),
        }
    }

    pub fn retired(
        &self,
        slot: &mut Slot,
        free: &mut Vec<vk::CommandBuffer>,
        used: usize,
        device: &ash::Device,
    ) {
        let now = Instant::now();
        let excess = slot
            .partition
            .as_ref()
            .map_or(free.len() > KEEP, |p| p.overflow.is_some());
        if excess {
            let mut state = slot.state.lock();
            state.armed |= used >= BUSY;
            state.free = Some(Parked {
                buffers: mem::take(free),
                partition: slot.partition.take(),
            });
            slot.parked = true;
        }
        let mut registry = self.registry.lock();
        if used >= BUSY {
            registry.next_scan = now + QUIET;
            return;
        }
        if now < registry.next_scan {
            return;
        }
        registry.next_scan = now + INTERVAL;
        let slots = registry.slots.clone();
        drop(registry);
        for weak in &slots {
            let Some(state) = weak.upgrade() else {
                continue;
            };
            let Some(mut state) = state.try_lock() else {
                continue;
            };
            if !state.armed {
                continue;
            }
            let raw = state.raw;
            let family = state.family;
            let Some(parked) = &mut state.free else {
                continue;
            };
            let buffers = &mut parked.buffers;
            // reset_all returned every outstanding command buffer, and both
            // begin_encoding and destruction take this list under the same
            // mutex before either may touch the Vulkan pool.
            if let Some(partition) = &mut parked.partition {
                let Some(overflow) = partition.overflow.take() else {
                    continue;
                };
                // All the primary handles are still valid in their original
                // pool. A future burst creates another overflow pool on demand.
                unsafe { device.destroy_command_pool(overflow, None) };
                buffers.clear();
                buffers.extend_from_slice(&partition.primary);
                self.reserved
                    .fetch_add(buffers.len() as u64, Ordering::Relaxed);
                self.replaced.fetch_add(1, Ordering::Relaxed);
                state.armed = false;
                break;
            }
            if self.mode == Mode::Pool {
                let desc = vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT);
                // Acquire the replacement before destroying the old pool so a
                // failed allocation leaves the parked encoder usable.
                let Ok(replacement) = (unsafe { device.create_command_pool(&desc, None) }) else {
                    break;
                };
                let allocate = vk::CommandBufferAllocateInfo::default()
                    .command_pool(replacement)
                    .command_buffer_count(KEEP as u32);
                let Ok(reserve) = (unsafe { device.allocate_command_buffers(&allocate) }) else {
                    unsafe { device.destroy_command_pool(replacement, None) };
                    break;
                };
                let reserved = reserve.len();
                unsafe { device.destroy_command_pool(raw, None) };
                *buffers = reserve;
                state.raw = replacement;
                state.armed = false;
                self.reserved.fetch_add(reserved as u64, Ordering::Relaxed);
                self.replaced.fetch_add(1, Ordering::Relaxed);
                break;
            }
            if self.mode == Mode::Buffers {
                let release = buffers.len().saturating_sub(KEEP).min(BATCH);
                if release != 0 {
                    let keep = buffers.len() - release;
                    unsafe {
                        device.free_command_buffers(raw, &buffers[keep..]);
                    }
                    buffers.truncate(keep);
                    self.freed.fetch_add(release as u64, Ordering::Relaxed);
                    if keep > KEEP {
                        break;
                    }
                }
            }
            if unsafe {
                device.reset_command_pool(raw, vk::CommandPoolResetFlags::RELEASE_RESOURCES)
            }
            .is_ok()
            {
                self.resets.fetch_add(1, Ordering::Relaxed);
                state.armed = false;
            }
            break;
        }
    }
}

impl super::Device {
    /// Totals for release resets, freed handles, reclaimed whole pools, and
    /// buffers immediately reusable by the corresponding encoders after reclaim.
    pub fn command_pool_trim_counts(&self) -> Option<[u64; 4]> {
        let trim = self.shared.pool_trim.as_ref()?;
        Some([
            trim.resets.load(Ordering::Relaxed),
            trim.freed.load(Ordering::Relaxed),
            trim.replaced.load(Ordering::Relaxed),
            trim.reserved.load(Ordering::Relaxed),
        ])
    }
}
