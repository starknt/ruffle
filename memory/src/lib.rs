//! Native CPU accounting. A lease belongs to the allocation, not to each reference to it.
//! GPU allocations stay on their device ledger; its independent pressure signal uses the same policy.
mod buffer;
mod device;
pub use device::{AllocatorSnapshot, DeviceMemory, DeviceSnapshot, ResourceSnapshot, SymbolId};
mod platform;
mod pressure;
pub mod private_cache;
mod resources;
pub use buffer::MemoryBuffer;
pub use pressure::{PressureLevel, PressureReading, PressureState};
pub use resources::{ContentId, ResourceManager};
use serde::{Deserialize, Serialize};
use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MemoryKind {
    GcHeap,
    AvmStack,
    BitmapPixels,
    EncodedBitmap,
    ByteArray,
    MovieBytes,
    MappedFile,
    Network,
    Decode,
    Pending,
    FrameScratch,
    Audio,
    Video,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScopeSnapshot {
    pub id: u64,
    pub active: BTreeMap<MemoryKind, u64>,
    pub admission_bytes: BTreeMap<MemoryKind, u64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub layout_samples: BTreeMap<String, [u64; 6]>,
    pub reserved_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemorySnapshot {
    pub budget_bytes: u64,
    pub active_bytes: u64,
    pub reserved_bytes: u64,
    pub peak_bytes: u64,
    pub rss_bytes: Option<u64>,
    pub pressure: PressureReading,
    pub devices: Vec<DeviceSnapshot>,
    pub shared: ScopeSnapshot,
    pub players: Vec<ScopeSnapshot>,
}

#[derive(Debug, Default)]
struct Ledger {
    next_player: u64,
    collected_epoch: u64,
    players: BTreeMap<u64, ScopeSnapshot>,
    shared: ScopeSnapshot,
    active: u64,
    reserved: u64,
    mapped: u64,
    peak: u64,
    rss: Option<(Instant, u64)>,
    hint: Option<(Instant, PressureLevel)>,
    pressure: PressureState,
}

impl Ledger {
    fn scope(&mut self, id: u64) -> &mut ScopeSnapshot {
        if id == 0 {
            &mut self.shared
        } else {
            self.players.get_mut(&id).expect("live memory scope")
        }
    }

    fn rss(&self, now: Instant) -> Option<u64> {
        self.rss
            .filter(|(at, _)| now.duration_since(*at) <= Duration::from_secs(2))
            .map(|(_, bytes)| bytes)
    }

    fn usage(&self, now: Instant) -> u64 {
        // RSS includes resident mapped pages. Outstanding reservations haven't been allocated yet.
        (self.active - self.mapped)
            .max(self.rss(now).unwrap_or(0))
            .saturating_add(self.reserved)
    }

    fn pressure(&mut self, budget: u64, now: Instant) -> PressureReading {
        let used = self.usage(now);
        let floor = self
            .hint
            .filter(|(until, _)| now < *until)
            .map_or(PressureLevel::Normal, |(_, floor)| floor);
        self.pressure.update(used, budget, floor);
        self.pressure.reading()
    }
}

#[derive(Debug)]
struct Inner {
    budget: u64,
    ledger: Mutex<Ledger>,
    devices: Mutex<BTreeMap<u64, std::sync::Weak<dyn DeviceMemory>>>,
    stores: Mutex<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>,
}

#[derive(Clone, Debug)]
pub struct MemoryManager(Arc<Inner>);

impl MemoryManager {
    pub fn process() -> &'static Self {
        static PROCESS: OnceLock<MemoryManager> = OnceLock::new();
        PROCESS.get_or_init(|| Self::new(platform::cpu_budget()))
    }

    pub fn new(budget_bytes: u64) -> Self {
        Self(Arc::new(Inner {
            budget: budget_bytes.max(1),
            ledger: Mutex::default(),
            stores: Mutex::default(),
            devices: Mutex::default(),
        }))
    }

    pub fn budget_bytes(&self) -> u64 {
        self.0.budget
    }

    pub fn resources<T: Send + Sync + 'static>(&self) -> Arc<ResourceManager<T>> {
        let mut stores = self
            .0
            .stores
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let store = stores
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Arc::new(ResourceManager::<T>::default()));
        store
            .clone()
            .downcast()
            .expect("resource store type matches its key")
    }

    fn ledger(&self) -> MutexGuard<'_, Ledger> {
        self.0
            .ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub fn player(&self) -> MemoryScope {
        let mut ledger = self.ledger();
        ledger.next_player = ledger
            .next_player
            .checked_add(1)
            .expect("memory player identity exhausted");
        let id = ledger.next_player;
        ledger.players.insert(
            id,
            ScopeSnapshot {
                id,
                ..Default::default()
            },
        );
        MemoryScope(Arc::new(Scope {
            manager: self.clone(),
            id,
        }))
    }

    /// Backings that may outlive or be shared by several independent movies or players.
    pub fn shared(&self) -> MemoryScope {
        MemoryScope(Arc::new(Scope {
            manager: self.clone(),
            id: 0,
        }))
    }

    pub fn sample_rss(&self, bytes: Option<u64>) {
        self.ledger().rss = bytes.map(|value| (Instant::now(), value));
    }

    pub fn notify_pressure(&self, level: PressureLevel) {
        let mut ledger = self.ledger();
        let now = Instant::now();
        let old = ledger
            .hint
            .filter(|(until, _)| now < *until)
            .map(|(_, level)| level);
        ledger.hint = Some((
            now + Duration::from_secs(5),
            old.unwrap_or_default().max(level),
        ));
        ledger.pressure(self.0.budget, now);
    }

    pub fn pressure(&self) -> PressureReading {
        self.ledger().pressure(self.0.budget, Instant::now())
    }

    pub fn register_device(&self, device: Arc<dyn DeviceMemory>) {
        let mut devices = self
            .0
            .devices
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        devices.retain(|_, weak| weak.strong_count() != 0);
        devices.insert(device.id(), Arc::downgrade(&device));
    }

    fn devices(&self) -> Vec<Arc<dyn DeviceMemory>> {
        self.0
            .devices
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .values()
            .filter_map(std::sync::Weak::upgrade)
            .collect()
    }

    /// Enumerating individual resources is reserved for explicit exports.
    pub fn device_resources(&self) -> BTreeMap<u64, Vec<ResourceSnapshot>> {
        self.devices()
            .into_iter()
            .map(|device| (device.id(), device.resources()))
            .collect()
    }

    pub fn snapshot(&self) -> MemorySnapshot {
        let devices = self
            .devices()
            .iter()
            .map(|device| device.snapshot())
            .collect();
        let mut ledger = self.ledger();
        let now = Instant::now();
        MemorySnapshot {
            devices,
            budget_bytes: self.0.budget,
            pressure: ledger.pressure(self.0.budget, now),
            active_bytes: ledger.active,
            reserved_bytes: ledger.reserved,
            peak_bytes: ledger.peak,
            rss_bytes: ledger.rss(now),
            shared: ledger.shared.clone(),
            players: ledger.players.values().cloned().collect(),
        }
    }
}

#[derive(Debug)]
struct Scope {
    manager: MemoryManager,
    id: u64,
}
impl Drop for Scope {
    fn drop(&mut self) {
        if self.id != 0 {
            self.manager.ledger().players.remove(&self.id);
        }
    }
}

#[derive(Clone, Debug)]
pub struct MemoryScope(Arc<Scope>);

thread_local! {
    static CURRENT: RefCell<Option<MemoryScope>> = const { RefCell::new(None) };
}

struct Restore(Option<MemoryScope>);
impl Drop for Restore {
    fn drop(&mut self) {
        CURRENT.with(|current| current.replace(self.0.take()));
    }
}

impl MemoryScope {
    pub fn manager(&self) -> &MemoryManager {
        &self.0.manager
    }
    pub fn id(&self) -> u64 {
        self.0.id
    }
    pub fn current() -> Self {
        CURRENT
            .with(|scope| scope.borrow().clone())
            .unwrap_or_else(|| MemoryManager::process().shared())
    }

    /// Covers legacy synchronous constructors on the current player thread. Workers carry an explicit scope.
    pub fn enter<R>(&self, f: impl FnOnce() -> R) -> R {
        let _restore = Restore(CURRENT.with(|current| current.replace(Some(self.clone()))));
        f()
    }

    /// Sample only this arena's collectable external backings on the GC path.
    pub fn gc_pressure(&self) -> (u64, PressureReading, bool) {
        let mut ledger = self.manager().ledger();
        let pressure = ledger.pressure(self.manager().0.budget, Instant::now());
        let emergency = pressure.level == PressureLevel::Critical
            && ledger.collected_epoch != pressure.critical_epoch;
        if emergency {
            ledger.collected_epoch = pressure.critical_epoch;
        }
        let active = &ledger.scope(self.id()).active;
        let external = [
            MemoryKind::BitmapPixels,
            MemoryKind::ByteArray,
            MemoryKind::Audio,
            MemoryKind::Video,
        ]
        .iter()
        .filter_map(|kind| active.get(kind))
        .sum();
        (external, pressure, emergency)
    }

    /// Sample one in 64 events in each distribution. Buckets are 0, 1, 2–4, 5–16, 17–64, and 65+.
    pub fn sample_layout(category: &'static str, count: usize) {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        if !*ENABLED.get_or_init(|| std::env::var_os("RUFFLE_MEMORY_LAYOUT").is_some()) {
            return;
        }
        thread_local! { static COUNTS: RefCell<HashMap<&'static str, u64>> = RefCell::default(); }
        if !COUNTS.with_borrow_mut(|counts| {
            let n = counts.entry(category).or_default();
            *n += 1;
            (*n - 1).is_multiple_of(64)
        }) {
            return;
        }
        let scope = Self::current();
        let mut ledger = scope.manager().ledger();
        let samples = &mut ledger.scope(scope.id()).layout_samples;
        let bucket = match count {
            0 => 0,
            1 => 1,
            2..=4 => 2,
            5..=16 => 3,
            17..=64 => 4,
            _ => 5,
        };
        if let Some(histogram) = samples.get_mut(category) {
            histogram[bucket] += 1;
        } else {
            let mut histogram = [0; 6];
            histogram[bucket] = 1;
            samples.insert(category.to_owned(), histogram);
        }
    }

    pub fn admission(&self, kind: MemoryKind, bytes: u64) -> Admission {
        let mut credit = Admission {
            scope: self.clone(),
            kind,
            bytes: 0,
        };
        credit.resize(bytes);
        credit
    }

    pub fn track(&self, kind: MemoryKind, bytes: u64) -> MemoryLease {
        let mut lease = MemoryLease {
            scope: self.clone(),
            kind,
            bytes: 0,
            reserved: false,
        };
        lease.resize(bytes);
        lease
    }

    /// Optional work must fit before it starts. Mandatory Flash allocations retain their own error semantics.
    pub fn try_reserve(&self, kind: MemoryKind, bytes: u64) -> Option<MemoryLease> {
        let mut ledger = self.manager().ledger();
        let now = Instant::now();
        let budget = self.manager().budget_bytes();
        if ledger.pressure(budget, now).level == PressureLevel::Critical
            || ledger.usage(now).checked_add(bytes)? > budget
        {
            return None;
        }
        ledger.reserved += bytes;
        ledger.scope(self.id()).reserved_bytes += bytes;
        ledger.peak = ledger.peak.max(ledger.active + ledger.reserved);
        Some(MemoryLease {
            scope: self.clone(),
            kind,
            bytes,
            reserved: true,
        })
    }
}

/// A loader quota may cover a shared backing, so it is reported separately from allocated bytes.
pub struct Admission {
    scope: MemoryScope,
    kind: MemoryKind,
    bytes: u64,
}
impl Admission {
    pub fn resize(&mut self, bytes: u64) {
        if bytes == self.bytes {
            return;
        }
        let mut ledger = self.scope.manager().ledger();
        let total = ledger
            .scope(self.scope.id())
            .admission_bytes
            .entry(self.kind)
            .or_default();
        *total = *total - self.bytes + bytes;
        self.bytes = bytes;
    }
}
impl Drop for Admission {
    fn drop(&mut self) {
        self.resize(0);
    }
}

#[derive(Debug)]
pub struct MemoryLease {
    scope: MemoryScope,
    kind: MemoryKind,
    bytes: u64,
    reserved: bool,
}
impl MemoryLease {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn scope(&self) -> &MemoryScope {
        &self.scope
    }
    pub fn resize(&mut self, actual: u64) {
        if actual != self.bytes {
            self.transition(actual, self.reserved);
        }
    }
    pub fn commit(mut self, actual: u64) -> Self {
        self.transition(actual, false);
        self
    }

    fn transition(&mut self, bytes: u64, reserved: bool) {
        let mut ledger = self.scope.manager().ledger();
        let id = self.scope.id();
        if self.reserved {
            ledger.reserved -= self.bytes;
            ledger.scope(id).reserved_bytes -= self.bytes;
        } else if self.bytes != 0 {
            ledger.active -= self.bytes;
            *ledger
                .scope(id)
                .active
                .get_mut(&self.kind)
                .expect("live memory kind") -= self.bytes;
            if self.kind == MemoryKind::MappedFile {
                ledger.mapped -= self.bytes;
            }
        }
        if reserved {
            ledger.reserved += bytes;
            ledger.scope(id).reserved_bytes += bytes;
        } else if bytes != 0 {
            ledger.active += bytes;
            *ledger.scope(id).active.entry(self.kind).or_default() += bytes;
            if self.kind == MemoryKind::MappedFile {
                ledger.mapped += bytes;
            }
        }
        ledger.peak = ledger.peak.max(ledger.active + ledger.reserved);
        self.bytes = bytes;
        self.reserved = reserved;
    }
}
impl Drop for MemoryLease {
    fn drop(&mut self) {
        self.resize(0);
    }
}
#[cfg(test)]
mod tests;
