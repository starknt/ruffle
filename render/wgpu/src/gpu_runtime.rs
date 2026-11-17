//! Device-scoped resource accounting independent of optional profiling.
//!
//! The registry owns metadata only. Leases accompany the actual resource owners
//! and binding dependencies. Once the last lease dies, bytes remain charged until
//! the last recorded submission completes, even though wgpu owns GPU retirement.

use fnv::FnvHashMap;
pub use ruffle_render::error::{RenderFailure, RenderFailureKind, RendererHealth};
use std::sync::{
    Arc, Mutex, MutexGuard,
    atomic::{AtomicU64, Ordering},
};

const MIB: u64 = 1024 * 1024;
static NEXT_DEVICE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug)]
pub struct GpuBudgetConfig {
    pub hard_limit: u64,
    pub soft_limit: u64,
    pub cache_limit: u64,
    pub idle_limit: u64,
    pub staging_limit: u64,
    pub transient_watermark: u64,
    /// Also bounds metadata for zero-storage views, bindings and samplers.
    pub resource_limit: usize,
}

impl GpuBudgetConfig {
    pub fn for_adapter(adapter: wgpu::DeviceType, android: bool) -> Self {
        let (hard, cache, idle, staging, transient, resources) = if android {
            (512, 64, 16, 16, 32, 65_536)
        } else if adapter == wgpu::DeviceType::DiscreteGpu {
            (2048, 256, 64, 64, 128, 262_144)
        } else {
            (1024, 128, 32, 32, 64, 131_072)
        };
        Self {
            hard_limit: hard * MIB,
            soft_limit: hard * MIB / 4 * 3,
            cache_limit: cache * MIB,
            idle_limit: idle * MIB,
            staging_limit: staging * MIB,
            transient_watermark: transient * MIB,
            resource_limit: resources,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(usize)]
pub enum ResourceKind {
    Texture,
    Buffer,
    TextureView,
    BindGroup,
    Sampler,
    Pipeline,
    ShapeMesh,
    GlyphMesh,
    OffscreenTarget,
    ReadbackBuffer,
    StagingBuffer,
    Stage3D,
    Shader,
    HostTarget,
    QuerySet,
    CommandEncoder,
    DynamicBuffer,
}

impl ResourceKind {
    pub const COUNT: usize = 17;
    pub const NAMES: [&'static str; Self::COUNT] = [
        "texture",
        "buffer",
        "textureView",
        "bindGroup",
        "sampler",
        "pipeline",
        "shapeMesh",
        "glyphMesh",
        "offscreenTarget",
        "readbackBuffer",
        "stagingBuffer",
        "stage3d",
        "shader",
        "hostTarget",
        "querySet",
        "commandEncoder",
        "dynamicBuffer",
    ];
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum OwnerKind {
    #[default]
    Renderer,
    Movie,
    BitmapData,
    Graphic,
    Glyph,
    Filter,
    BitmapCache,
    ShaderJob,
    Scene,
    Host,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct ResourceOwner {
    pub kind: OwnerKind,
    pub id: u64,
}

impl ResourceOwner {
    pub const fn new(kind: OwnerKind, id: u64) -> Self {
        Self { kind, id }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ResourceId {
    device: u64,
    index: u64,
}

impl ResourceId {
    pub fn device_generation(self) -> u64 {
        self.device
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SubmissionSerial {
    device: u64,
    index: u64,
}

impl SubmissionSerial {
    pub(crate) fn sequence(self) -> u64 {
        self.index
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MemoryPressure {
    #[default]
    Normal,
    Soft,
    Hard,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceState {
    Reserved,
    Active,
    Idle,
    Retiring,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourcePriority {
    Visible,
    Reconstructible,
    Idle,
}

#[derive(Clone, Debug)]
pub struct ResourceInfo {
    visited: u64,
    pub id: ResourceId,
    pub owner: ResourceOwner,
    pub kind: ResourceKind,
    pub bytes: u64,
    pub state: ResourceState,
    pub created_submission: u64,
    pub last_submission: u64,
    pub dependencies: Vec<ResourceId>,
    pub priority: ResourcePriority,
    pub reconstructible: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GpuMemoryStats {
    pub reserved_bytes: u64,
    pub active_bytes: u64,
    pub idle_bytes: u64,
    pub retiring_bytes: u64,
    pub total_bytes: u64,
    pub peak_bytes: u64,
    pub cache_retained_bytes: u64,
    pub pool_retained_bytes: u64,
    /// Includes reservations, so pending zero-byte objects cannot bypass limits.
    pub resources: usize,
    pub kind_bytes: [u64; ResourceKind::COUNT],
    pub kind_counts: [usize; ResourceKind::COUNT],
    pub pressure: MemoryPressure,
    pub submitted: u64,
    pub completed: u64,
    pub budget_rejections: u64,
    pub creation_retries: u64,
}

#[derive(Default)]
struct Ledger {
    visit: u64,
    retiring: Vec<ResourceId>,
    records: FnvHashMap<ResourceId, ResourceInfo>,
    stats: GpuMemoryStats,
    next_resource: u64,
    next_owner: u64,
}

struct Inner {
    preparation: crate::prepare::NativePreparation,
    scheduler: crate::resource_scheduler::ResourceScheduler,
    device: u64,
    config: GpuBudgetConfig,
    ledger: Mutex<Ledger>,
    failure: RendererHealth,
}

impl Inner {
    fn ledger(&self) -> MutexGuard<'_, Ledger> {
        self.ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn healthy(&self) -> Result<(), RenderFailure> {
        self.failure.check()
    }
}

#[derive(Clone)]
pub struct GpuRuntime(Arc<Inner>);

impl std::fmt::Debug for GpuRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuRuntime")
            .field("device", &self.0.device)
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

fn failure(kind: RenderFailureKind, message: impl Into<Arc<str>>) -> RenderFailure {
    RenderFailure::new(kind, message)
}

impl GpuRuntime {
    /// Called once by the native device owner, before any renderer resources.
    /// Shared hosts pass this runtime to Descriptors instead of installing
    /// callbacks again for each movie.
    pub fn install_device_callbacks(&self, device: &wgpu::Device) {
        self.install_device_callbacks_with_sink(device, |_| {});
    }

    pub fn install_device_callbacks_with_sink(
        &self,
        device: &wgpu::Device,
        sink: impl Fn(&RenderFailure) + Send + Sync + 'static,
    ) {
        let sink = Arc::new(sink);
        let errors = sink.clone();
        let health = self.health().clone();
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            let error = Self::wgpu_failure(error);
            if health.fail(error.clone()) {
                errors(&error);
            }
        }));
        let health = self.health().clone();
        device.set_device_lost_callback(move |reason, message| {
            let error = RenderFailure::new(
                RenderFailureKind::DeviceLost,
                format!("{reason:?}: {message}"),
            );
            if health.fail(error.clone()) {
                sink(&error);
            }
        });
    }

    pub fn wgpu_failure(error: wgpu::Error) -> RenderFailure {
        let kind = match &error {
            wgpu::Error::OutOfMemory { .. } => RenderFailureKind::OutOfMemory,
            wgpu::Error::Validation { .. } => RenderFailureKind::Validation,
            wgpu::Error::Internal { .. } => RenderFailureKind::Internal,
        };
        RenderFailure::new(kind, error.to_string())
    }

    pub fn new(config: GpuBudgetConfig) -> Result<Self, RenderFailure> {
        if config.soft_limit > config.hard_limit || config.resource_limit == 0 {
            return Err(failure(
                RenderFailureKind::Validation,
                "invalid GPU budget configuration",
            ));
        }
        let device = NEXT_DEVICE
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
            .map_err(|_| {
                failure(
                    RenderFailureKind::Internal,
                    "device identity space exhausted",
                )
            })?;
        let failure = RendererHealth::default();
        Ok(Self(Arc::new(Inner {
            preparation: crate::prepare::NativePreparation::for_device(
                config.staging_limit,
                failure.clone(),
            ),
            scheduler: crate::resource_scheduler::ResourceScheduler::new(
                config.idle_limit,
                config.cache_limit,
            ),
            device,
            config,
            ledger: Mutex::default(),
            failure,
        })))
    }

    pub(crate) fn scheduler(&self) -> &crate::resource_scheduler::ResourceScheduler {
        &self.0.scheduler
    }

    pub(crate) fn preparation(&self) -> &crate::prepare::NativePreparation {
        &self.0.preparation
    }

    pub fn config(&self) -> GpuBudgetConfig {
        self.0.config
    }

    pub fn generation(&self) -> u64 {
        self.0.device
    }

    pub fn health(&self) -> &RendererHealth {
        &self.0.failure
    }

    pub fn failure(&self) -> Option<RenderFailure> {
        self.0.failure.failure()
    }

    /// Preserve the initiating cause rather than cascading invalid-resource errors.
    pub fn fail(&self, error: RenderFailure) -> bool {
        self.0.failure.fail(error)
    }

    pub fn check_health(&self) -> Result<(), RenderFailure> {
        self.0.healthy()
    }

    pub fn issue_owner(&self, kind: OwnerKind) -> Result<ResourceOwner, RenderFailure> {
        self.check_health()?;
        let mut ledger = self.0.ledger();
        ledger.next_owner = ledger.next_owner.checked_add(1).ok_or_else(|| {
            failure(
                RenderFailureKind::Internal,
                "resource owner identity space exhausted",
            )
        })?;
        Ok(ResourceOwner::new(kind, ledger.next_owner))
    }

    pub fn reserve(
        &self,
        kind: ResourceKind,
        owner: ResourceOwner,
        bytes: u64,
    ) -> Result<ResourceReservation, RenderFailure> {
        let mut ledger = self.0.ledger();
        self.0.healthy()?;
        let total = ledger
            .stats
            .total_bytes
            .checked_add(bytes)
            .filter(|total| *total <= self.0.config.hard_limit);
        if total.is_none() || ledger.stats.resources >= self.0.config.resource_limit {
            ledger.stats.budget_rejections += 1;
            return Err(failure(
                RenderFailureKind::BudgetExceeded,
                format!(
                    "GPU {:?} request {bytes} bytes exceeds budget ({} used, {} limit, {} resources)",
                    kind,
                    ledger.stats.total_bytes,
                    self.0.config.hard_limit,
                    ledger.stats.resources
                ),
            ));
        }
        if kind == ResourceKind::StagingBuffer
            && ledger.stats.kind_bytes[kind as usize]
                .checked_add(bytes)
                .is_none_or(|size| size > self.0.config.staging_limit)
        {
            ledger.stats.budget_rejections += 1;
            return Err(failure(
                RenderFailureKind::BudgetExceeded,
                "GPU staging budget exceeded",
            ));
        }
        ledger.next_resource = ledger.next_resource.checked_add(1).ok_or_else(|| {
            failure(
                RenderFailureKind::Internal,
                "resource identity space exhausted",
            )
        })?;
        let id = ResourceId {
            device: self.0.device,
            index: ledger.next_resource,
        };
        let created_submission = ledger.stats.submitted;
        ledger.insert(ResourceInfo {
            visited: 0,
            id,
            owner,
            kind,
            bytes,
            state: ResourceState::Reserved,
            created_submission,
            last_submission: 0,
            dependencies: Vec::new(),
            priority: ResourcePriority::Visible,
            reconstructible: false,
        });
        Ok(ResourceReservation {
            runtime: self.clone(),
            id: Some(id),
        })
    }

    pub(crate) fn record_creation_retry(&self) {
        self.0.ledger().stats.creation_retries += 1;
    }
    /// Explicit acceptance/teardown purge. In-flight and recorded references
    /// retain their storage; callers can drain first to make all idle pages eligible.
    pub fn purge_idle_and_cached_resources(&self) {
        self.0.scheduler.reclaim();
    }

    pub fn diagnostic_snapshot(&self) -> ruffle_render::profile::NativeResources {
        let stats = self.stats();
        let work = self.preparation().stats();
        ruffle_render::profile::NativeResources {
            device_generation: self.generation(),
            logical_bytes: stats.total_bytes,
            active_bytes: stats.active_bytes,
            reserved_bytes: stats.reserved_bytes,
            idle_bytes: stats.idle_bytes,
            retiring_bytes: stats.retiring_bytes,
            peak_bytes: stats.peak_bytes,
            cache_retained_bytes: stats.cache_retained_bytes,
            pool_retained_bytes: stats.pool_retained_bytes,
            resources: stats.resources,
            classes: ResourceKind::NAMES
                .iter()
                .enumerate()
                .map(|(i, name)| (*name, stats.kind_bytes[i], stats.kind_counts[i]))
                .collect(),
            submitted: stats.submitted,
            completed: stats.completed,
            budget_rejections: stats.budget_rejections,
            creation_retries: stats.creation_retries,
            terminal_failure: self.failure().map(|error| format!("{:?}", error.kind())),
            worker_active: work.active,
            worker_bytes: work.bytes,
            worker_peak_active: work.peak_active,
            worker_peak_bytes: work.peak_bytes,
            worker_submitted: work.submitted,
        }
    }

    pub fn stats(&self) -> GpuMemoryStats {
        let mut stats = self.0.ledger().stats;
        stats.cache_retained_bytes = self
            .scheduler()
            .bytes(crate::resource_scheduler::ReclaimClass::Cache);
        stats.pool_retained_bytes = self
            .scheduler()
            .bytes(crate::resource_scheduler::ReclaimClass::Idle);
        stats.pressure = if stats.total_bytes >= self.0.config.hard_limit {
            MemoryPressure::Hard
        } else if stats.total_bytes >= self.0.config.soft_limit {
            MemoryPressure::Soft
        } else {
            MemoryPressure::Normal
        };
        stats
    }

    pub fn owner_bytes(&self, owner: ResourceOwner) -> u64 {
        self.0
            .ledger()
            .records
            .values()
            .filter(|r| r.owner == owner)
            .map(|r| r.bytes)
            .sum()
    }

    /// Detailed attribution is opt-in; frame gauges do not clone the registry.
    pub fn resources(&self) -> Vec<ResourceInfo> {
        self.0.ledger().records.values().cloned().collect()
    }

    /// Call under the queue's submission ordering gate. Include all resources
    /// recorded in the encoder; binding dependencies are visited transitively.
    pub fn register_submission(
        &self,
        resources: &[ResourceLease],
    ) -> Result<SubmissionSerial, RenderFailure> {
        self.0.healthy()?;
        if resources
            .iter()
            .any(|r| !Arc::ptr_eq(&r.0.runtime.0, &self.0))
        {
            return Err(failure(
                RenderFailureKind::Validation,
                "submission references another device",
            ));
        }
        let mut ledger = self.0.ledger();
        if ledger.stats.submitted - ledger.stats.completed >= 2 {
            return Err(failure(
                RenderFailureKind::BudgetExceeded,
                "two Ruffle submissions are already in flight",
            ));
        }

        let serial = ledger.stats.submitted.checked_add(1).ok_or_else(|| {
            failure(
                RenderFailureKind::Internal,
                "submission identity space exhausted",
            )
        })?;
        // Use a unique traversal stamp rather than allocating a second identity
        // set for every submission. Attempts have separate stamps, including a
        // failed attempt that did not issue its proposed submission serial.
        ledger.visit = ledger.visit.checked_add(1).ok_or_else(|| {
            failure(
                RenderFailureKind::Internal,
                "submission visit space exhausted",
            )
        })?;
        let visit = ledger.visit;
        let mut pending: Vec<_> = resources.iter().map(ResourceLease::id).collect();
        while let Some(id) = pending.pop() {
            let record = ledger.records.get_mut(&id).ok_or_else(|| {
                failure(
                    RenderFailureKind::Validation,
                    "submission references a retired resource",
                )
            })?;
            if record.visited != visit {
                record.visited = visit;
                record.last_submission = serial;
                pending.extend_from_slice(&record.dependencies);
            }
        }
        ledger.stats.submitted = serial;
        Ok(SubmissionSerial {
            device: self.0.device,
            index: serial,
        })
    }

    /// Queue completion is a prefix: completion of N also completes older work.
    /// Reject foreign or unissued tokens, including stale device callbacks.
    pub fn complete_submission(&self, serial: SubmissionSerial) -> bool {
        if serial.device != self.0.device {
            return false;
        }
        let mut ledger = self.0.ledger();
        if serial.index <= ledger.stats.completed || serial.index > ledger.stats.submitted {
            return false;
        }
        ledger.stats.completed = serial.index;
        // Completion only needs resources whose last owning lease has gone.
        // A large live movie must not turn every completion into a full scan.
        let mut index = 0;
        while index < ledger.retiring.len() {
            let id = ledger.retiring[index];
            if ledger
                .records
                .get(&id)
                .is_none_or(|r| r.last_submission <= serial.index)
            {
                ledger.retiring.swap_remove(index);
                ledger.remove(id);
            } else {
                index += 1;
            }
        }
        true
    }
}

impl Ledger {
    fn bucket(&mut self, state: ResourceState) -> &mut u64 {
        match state {
            ResourceState::Reserved => &mut self.stats.reserved_bytes,
            ResourceState::Active => &mut self.stats.active_bytes,
            ResourceState::Idle => &mut self.stats.idle_bytes,
            ResourceState::Retiring => &mut self.stats.retiring_bytes,
        }
    }

    fn insert(&mut self, record: ResourceInfo) {
        *self.bucket(record.state) += record.bytes;
        self.stats.total_bytes += record.bytes;
        self.stats.peak_bytes = self.stats.peak_bytes.max(self.stats.total_bytes);
        self.stats.resources += 1;
        self.stats.kind_bytes[record.kind as usize] += record.bytes;
        self.stats.kind_counts[record.kind as usize] += 1;
        self.records.insert(record.id, record);
    }

    fn transition(&mut self, id: ResourceId, state: ResourceState) {
        if let Some(record) = self.records.get_mut(&id) {
            let old = record.state;
            let bytes = record.bytes;
            record.state = state;
            *self.bucket(old) -= bytes;
            *self.bucket(state) += bytes;
        }
    }

    fn remove(&mut self, id: ResourceId) {
        if let Some(record) = self.records.remove(&id) {
            *self.bucket(record.state) -= record.bytes;
            self.stats.total_bytes -= record.bytes;
            self.stats.resources -= 1;
            self.stats.kind_bytes[record.kind as usize] -= record.bytes;
            self.stats.kind_counts[record.kind as usize] -= 1;
        }
    }
}

#[derive(Debug)]
pub struct ResourceReservation {
    runtime: GpuRuntime,
    id: Option<ResourceId>,
}

impl ResourceReservation {
    pub fn commit(
        mut self,
        dependencies: &[ResourceLease],
    ) -> Result<ResourceLease, RenderFailure> {
        self.runtime.check_health()?;
        if dependencies
            .iter()
            .any(|r| !Arc::ptr_eq(&r.0.runtime.0, &self.runtime.0))
        {
            return Err(failure(
                RenderFailureKind::Validation,
                "binding depends on another device",
            ));
        }
        // A reservation is consumed once; its ID remains present until this call
        // succeeds, so every error path is rolled back by Drop.
        let id = self
            .id
            .ok_or_else(|| failure(RenderFailureKind::Internal, "reservation already consumed"))?;
        let (transient_bytes, staging_bytes, upload_page) = {
            let mut ledger = self.runtime.0.ledger();
            ledger.transition(id, ResourceState::Active);
            let record = ledger.records.get_mut(&id).ok_or_else(|| {
                failure(RenderFailureKind::Internal, "missing resource reservation")
            })?;
            record.dependencies = dependencies.iter().map(ResourceLease::id).collect();
            let transient = if matches!(
                record.kind,
                ResourceKind::OffscreenTarget
                    | ResourceKind::StagingBuffer
                    | ResourceKind::ReadbackBuffer
                    | ResourceKind::DynamicBuffer
            ) {
                record.bytes
            } else {
                0
            };
            (
                transient,
                if record.kind == ResourceKind::StagingBuffer {
                    record.bytes
                } else {
                    0
                },
                matches!(
                    record.kind,
                    ResourceKind::StagingBuffer | ResourceKind::DynamicBuffer
                ),
            )
        };
        self.id = None;
        Ok(ResourceLease(Arc::new(Allocation {
            runtime: self.runtime.clone(),
            id,
            transient_bytes,
            staging_bytes,
            upload_page,
            unsubmitted: Default::default(),
            _dependencies: dependencies.to_vec(),
        })))
    }
}

impl Drop for ResourceReservation {
    fn drop(&mut self) {
        if let Some(id) = self.id.take() {
            self.runtime.0.ledger().remove(id);
        }
    }
}

struct Allocation {
    runtime: GpuRuntime,
    id: ResourceId,
    transient_bytes: u64,
    staging_bytes: u64,
    upload_page: bool,
    unsubmitted: std::sync::atomic::AtomicUsize,
    // These mirror the owning GPU object's references, not registry ownership.
    _dependencies: Vec<ResourceLease>,
}

impl Drop for Allocation {
    fn drop(&mut self) {
        let mut ledger = self.runtime.0.ledger();
        let pending = ledger
            .records
            .get(&self.id)
            .is_some_and(|r| r.last_submission > ledger.stats.completed);
        if pending {
            ledger.transition(self.id, ResourceState::Retiring);
            ledger.retiring.push(self.id);
        } else {
            ledger.remove(self.id);
        }
        // Release the mutex before fields (and their dependency leases) drop.
    }
}

#[derive(Clone)]
pub struct ResourceLease(Arc<Allocation>);

impl std::fmt::Debug for ResourceLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ResourceLease").field(&self.id()).finish()
    }
}

pub(crate) struct UnsubmittedUse(Arc<Allocation>);
impl Drop for UnsubmittedUse {
    fn drop(&mut self) {
        self.0
            .unsubmitted
            .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}

impl ResourceLease {
    pub(crate) fn check_health(&self) -> Result<(), RenderFailure> {
        self.0.runtime.check_health()
    }
    pub(crate) fn staging_bytes(&self) -> u64 {
        self.0.staging_bytes
    }
    pub(crate) fn unsubmitted_uses(&self) -> usize {
        self.0
            .unsubmitted
            .load(std::sync::atomic::Ordering::Acquire)
    }
    pub(crate) fn recording_use(&self) -> Option<UnsubmittedUse> {
        if !self.0.upload_page {
            return None;
        }
        self.0
            .unsubmitted
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        Some(UnsubmittedUse(self.0.clone()))
    }
    pub(crate) fn last_submission(&self) -> u64 {
        self.0
            .runtime
            .0
            .ledger()
            .records
            .get(&self.id())
            .map_or(0, |record| record.last_submission)
    }
    /// Conservative cache-retention weight, not a second storage charge.
    pub(crate) fn retained_storage_bytes(&self) -> u64 {
        let mut pending = vec![self];
        let mut ids = std::collections::HashSet::new();
        while let Some(lease) = pending.pop() {
            if ids.insert(lease.id()) {
                pending.extend(lease.dependencies());
            }
        }
        let ledger = self.0.runtime.0.ledger();
        ids.into_iter()
            .filter_map(|id| ledger.records.get(&id))
            .map(|record| record.bytes)
            .sum()
    }
    pub(crate) fn dependencies(&self) -> &[ResourceLease] {
        &self.0._dependencies
    }

    pub(crate) fn transient_bytes(&self) -> u64 {
        self.0.transient_bytes
    }
    pub fn id(&self) -> ResourceId {
        self.0.id
    }

    pub fn set_reconstructible(&self, reconstructible: bool) {
        if let Some(record) = self.0.runtime.0.ledger().records.get_mut(&self.id()) {
            record.reconstructible = reconstructible;
            record.priority = if reconstructible {
                ResourcePriority::Reconstructible
            } else {
                ResourcePriority::Visible
            };
        }
    }
    pub fn set_idle(&self, idle: bool) {
        let mut ledger = self.0.runtime.0.ledger();
        if let Some(record) = ledger.records.get_mut(&self.id()) {
            record.priority = if idle {
                ResourcePriority::Idle
            } else {
                ResourcePriority::Visible
            };
        }
        ledger.transition(
            self.id(),
            if idle {
                ResourceState::Idle
            } else {
                ResourceState::Active
            },
        );
    }

    pub fn is_reusable(&self) -> bool {
        let ledger = self.0.runtime.0.ledger();
        ledger
            .records
            .get(&self.id())
            .is_some_and(|r| r.last_submission <= ledger.stats.completed)
    }
}

/// Conservative storage accounting, not the driver's allocation/block overhead.
pub fn texture_storage_bytes(desc: &wgpu::TextureDescriptor<'_>) -> Result<u64, RenderFailure> {
    let invalid = || {
        failure(
            RenderFailureKind::Validation,
            "invalid or overflowing texture extent",
        )
    };
    let size = desc.size;
    if size.width == 0
        || size.height == 0
        || size.depth_or_array_layers == 0
        || desc.mip_level_count == 0
        || desc.sample_count == 0
        || !desc.sample_count.is_power_of_two()
    {
        return Err(invalid());
    }
    let max_dimension = match desc.dimension {
        wgpu::TextureDimension::D1 => size.width,
        wgpu::TextureDimension::D2 => size.width.max(size.height),
        wgpu::TextureDimension::D3 => size.width.max(size.height).max(size.depth_or_array_layers),
    };
    if desc.mip_level_count > max_dimension.ilog2() + 1 {
        return Err(invalid());
    }
    let block_bytes = match desc.format {
        wgpu::TextureFormat::Depth24Plus => 4,
        wgpu::TextureFormat::Depth24PlusStencil8 | wgpu::TextureFormat::Depth32FloatStencil8 => 8,
        format => format.block_copy_size(None).ok_or_else(invalid)?,
    };
    let (block_width, block_height) = desc.format.block_dimensions();
    let mut total = 0u64;
    for level in 0..desc.mip_level_count {
        let width = u64::from((size.width >> level).max(1)).div_ceil(u64::from(block_width));
        let height = u64::from((size.height >> level).max(1)).div_ceil(u64::from(block_height));
        let depth = if desc.dimension == wgpu::TextureDimension::D3 {
            (size.depth_or_array_layers >> level).max(1)
        } else {
            size.depth_or_array_layers
        };
        let bytes = width
            .checked_mul(height)
            .and_then(|v| v.checked_mul(u64::from(depth)))
            .and_then(|v| v.checked_mul(u64::from(block_bytes)))
            .and_then(|v| v.checked_mul(u64::from(desc.sample_count)))
            .ok_or_else(invalid)?;
        total = total.checked_add(bytes).ok_or_else(invalid)?;
    }
    Ok(total)
}

#[cfg(test)]
mod tests;
