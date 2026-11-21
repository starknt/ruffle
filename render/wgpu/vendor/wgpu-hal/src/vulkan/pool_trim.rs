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
const BATCH: usize = 1024;
const QUIET: Duration = Duration::from_secs(4);
const INTERVAL: Duration = Duration::from_millis(150);

#[derive(Debug, Clone, Copy, PartialEq)]
enum Mode {
    Reset,
    Buffers,
    Pool,
}

#[derive(Debug)]
struct Retired {
    raw: vk::CommandPool,
    family: u32,
    free: Option<Vec<vk::CommandBuffer>>,
    armed: bool,
}

#[derive(Debug)]
pub(super) struct Slot {
    state: Arc<Mutex<Retired>>,
    parked: bool,
}
impl Slot {
    pub fn checkout(&mut self, raw: &mut vk::CommandPool, free: &mut Vec<vk::CommandBuffer>) {
        if self.parked {
            let mut state = self.state.lock();
            *raw = state.raw;
            *free = state
                .free
                .take()
                .expect("parked encoder owns its buffer list");
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
        if free.len() > KEEP {
            let mut state = slot.state.lock();
            state.armed |= used >= BUSY;
            state.free = Some(mem::take(free));
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
            let Some(buffers) = &mut state.free else {
                continue;
            };
            // reset_all returned every outstanding command buffer, and both
            // begin_encoding and destruction take this list under the same
            // mutex before either may touch the Vulkan pool.
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
    /// Totals for release resets, freed handles, replaced pools, and handles
    /// reserved in the replacement pools before the encoders become busy again.
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
