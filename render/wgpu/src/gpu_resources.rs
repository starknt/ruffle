//! Native allocation boundary. GPU objects carry a shared lease, while this
//! device's lookup table contains weak references only. Bindings retain leases
//! for their parents, even after the caller releases its texture/buffer handle.
//!
//! Encoders retain only resources referenced by actual commands and their
//! dependency leases. Submission completion releases those exact use sets.
mod coverage_shader;
pub(crate) mod diagnostics;
mod encoding;
mod history;
mod submission_order;
use crate::gpu_runtime::{
    GpuRuntime, RenderFailure, RenderFailureKind, ResourceId, ResourceKind, ResourceLease,
    ResourceOwner, texture_storage_bytes,
};
use crate::pipeline_cache::{PipelineCacheState, PipelineCacheStatus};
use crate::pipeline_profile::{Family, Kind, ObjectInfo, Profile, Signature, Variant, signature};
pub use diagnostics::PassDiagnostic;
pub use encoding::{ComputePass, RenderPass};
use fnv::FnvHashMap;
use std::{
    any::{Any, TypeId},
    fmt::Debug,
    future::Future,
    hash::{Hash, Hasher},
    ops::Deref,
    path::Path,
    sync::{
        Arc, Mutex, OnceLock, Weak,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll},
    time::{Duration, Instant},
};
pub use submission_order::ReservedSubmission;

pub struct ExternalHistory(Arc<history::Observation>);

#[derive(Hash, PartialEq, Eq)]
struct PipelineUseKey {
    device: u64,
    epoch: u64,
    frame: Option<crate::pipeline_profile::Frame>,
    observation: usize,
    variant: usize,
}

struct PipelineUse {
    observation: Option<Arc<history::Observation>>,
    variant: Option<Arc<Variant>>,
    count: u64,
}

type UseCounts = indexmap::IndexMap<PipelineUseKey, PipelineUse, fnv::FnvBuildHasher>;

/// Draw and dispatch counts for one command recording. Dropping it cancels the observations.
#[derive(Default)]
pub struct PipelineUses {
    entries: Option<Box<UseCounts>>,
}
impl PipelineUses {
    fn record(
        &mut self,
        device: &Device,
        observation: Option<&Arc<history::Observation>>,
        variant: Option<&Arc<Variant>>,
    ) {
        if observation.is_none() && variant.is_none() {
            return;
        }
        let generation = device.runtime().generation();
        let key = PipelineUseKey {
            device: generation,
            epoch: observation
                .and_then(|_| device.history())
                .map_or(0, history::History::epoch),
            frame: crate::pipeline_profile::current_frame().filter(|f| f.device == generation),
            observation: observation.map_or(0, |v| Arc::as_ptr(v) as usize),
            variant: variant.map_or(0, |v| Arc::as_ptr(v) as usize),
        };
        let entry = self
            .entries
            .get_or_insert_with(Default::default)
            .entry(key)
            .or_insert_with(|| PipelineUse {
                observation: observation.cloned(),
                variant: variant.cloned(),
                count: 0,
            });
        entry.count = entry.count.saturating_add(1);
    }

    /// Raw wgpu callers own their queue. The encoder drops this callback if it is abandoned.
    pub fn observe_raw_completion(self, encoder: &wgpu::CommandEncoder, device: &Device) {
        if self.entries.is_some() {
            let device = device.clone();
            encoder.on_submitted_work_done(move || {
                if device.runtime().check_health().is_ok() {
                    self.submitted(&device);
                }
            });
        }
    }

    /// Publish only after this recording's checked queue submission succeeds.
    pub fn submitted(self, device: &Device) {
        if let Some(entries) = self.entries {
            for (key, use_) in *entries {
                if key.device != device.runtime().generation() {
                    continue;
                }
                if let Some(obs) = use_.observation
                    && let Some(h) = device.history()
                {
                    h.used(&obs, key.epoch, key.frame, use_.count);
                }
                if let Some(v) = use_.variant {
                    v.used(key.frame, use_.count);
                }
            }
        }
    }
}

pub type Buffer = Resource<wgpu::Buffer>;
pub type Texture = Resource<wgpu::Texture>;
pub type TextureView = Resource<wgpu::TextureView>;
pub type BindGroup = Resource<wgpu::BindGroup>;
pub type BindGroupLayout = Resource<wgpu::BindGroupLayout>;
pub type PipelineLayout = Resource<wgpu::PipelineLayout>;
pub type RenderPipeline = Resource<wgpu::RenderPipeline>;
pub type ComputePipeline = Resource<wgpu::ComputePipeline>;
pub type ShaderModule = Resource<wgpu::ShaderModule>;
pub type Sampler = Resource<wgpu::Sampler>;
pub type QuerySet = Resource<wgpu::QuerySet>;

type ObjectKey = (TypeId, u64);
type UseSet = FnvHashMap<ResourceId, ResourceLease>;

/// A closed set of recorded resources, including every binding dependency.
/// Only this module can construct one; submission need not traverse it again.
#[derive(Default)]
pub(crate) struct RecordedUses(UseSet);
impl RecordedUses {
    fn from_roots<'a>(roots: impl Iterator<Item = &'a ResourceLease>) -> Self {
        fn retain(uses: &mut UseSet, lease: &ResourceLease) {
            if let std::collections::hash_map::Entry::Vacant(entry) = uses.entry(lease.id()) {
                entry.insert(lease.clone());
                for dependency in lease.dependencies() {
                    retain(uses, dependency);
                }
            }
        }
        let mut uses = Self::default();
        for lease in roots {
            retain(&mut uses.0, lease);
        }
        uses
    }
    pub(crate) fn leases(&self) -> impl Iterator<Item = &ResourceLease> {
        self.0.values()
    }
}
#[derive(Default)]
struct Recording {
    pipeline_uses: PipelineUses,
    uses: UseSet,
    objects: FnvHashMap<ObjectKey, smallvec::SmallVec<[Arc<dyn Indexed>; 1]>>,
    transient_bytes: u64,
    staging_bytes: u64,
    pending: Vec<crate::gpu_runtime::UnsubmittedUse>,
}
impl Recording {
    fn retain(&mut self, lease: &ResourceLease) {
        if let std::collections::hash_map::Entry::Vacant(entry) = self.uses.entry(lease.id()) {
            self.transient_bytes += lease.transient_bytes();
            self.staging_bytes += lease.staging_bytes();
            self.pending.extend(lease.recording_use());
            entry.insert(lease.clone());
            for dependency in lease.dependencies() {
                self.retain(dependency);
            }
        }
    }
}

trait Indexed: Any + Send + Sync {
    fn matches(&self, raw: &dyn Any) -> bool;
    fn lease(&self) -> &ResourceLease;
    fn profile_info(&self) -> Option<&ObjectInfo>;
    fn layout(&self) -> Option<&history::Layout>;
    fn observation(&self) -> Option<&Arc<history::Observation>>;
}
#[derive(Default)]
struct Metadata {
    profile: OnceLock<Arc<ObjectInfo>>,
    observation: OnceLock<Arc<history::Observation>>,
    layout: OnceLock<Box<history::Layout>>,
}
struct Object<T> {
    metadata: OnceLock<Box<Metadata>>,
    raw: T,
    lease: ResourceLease,
    device: Device,
}
impl<T: Any + Eq + Send + Sync> Indexed for Object<T> {
    fn layout(&self) -> Option<&history::Layout> {
        self.metadata.get()?.layout.get().map(Box::as_ref)
    }
    fn observation(&self) -> Option<&Arc<history::Observation>> {
        self.metadata.get()?.observation.get()
    }
    fn profile_info(&self) -> Option<&ObjectInfo> {
        self.metadata.get()?.profile.get().map(Arc::as_ref)
    }
    fn matches(&self, raw: &dyn Any) -> bool {
        raw.downcast_ref::<T>().is_some_and(|raw| self.raw == *raw)
    }
    fn lease(&self) -> &ResourceLease {
        &self.lease
    }
}

pub struct Resource<T>(Arc<Object<T>>);
impl<T> Resource<T> {
    fn meta(&self) -> &Metadata {
        self.0.metadata.get_or_init(Default::default)
    }
    fn set_profile(&self, key: Signature, variant: Option<Arc<Variant>>) {
        let _ = self.meta().profile.set(Arc::new(ObjectInfo {
            key,
            instance: variant.map(|v| v.instance()),
        }));
    }
    /// Add a raw pass's draw or dispatch to the observations for this submission.
    pub fn record_pipeline_use(&self, uses: &mut PipelineUses) {
        let Some(meta) = self.0.metadata.get() else {
            return;
        };
        let variant = meta
            .profile
            .get()
            .and_then(|info| info.instance.as_ref())
            .map(|i| i.variant());
        uses.record(&self.0.device, meta.observation.get(), variant);
    }
    pub fn profile_hit(&self) {
        if let Some(instance) = self
            .0
            .metadata
            .get()
            .and_then(|m| m.profile.get())
            .and_then(|o| o.instance.as_ref())
        {
            instance.variant().hit();
        }
    }

    pub fn device(&self) -> &Device {
        &self.0.device
    }

    pub fn lease(&self) -> &ResourceLease {
        &self.0.lease
    }
    /// Borrow for native interop. An exported raw clone must retain this lease
    /// through the external consumer's completion callback.
    pub fn raw(&self) -> &T {
        &self.0.raw
    }
}
impl<T> Clone for Resource<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<T> Deref for Resource<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0.raw
    }
}
impl<T: Debug> Debug for Resource<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Resource")
            .field("id", &self.lease().id())
            .field("raw", self.raw())
            .finish()
    }
}
impl<T: PartialEq> PartialEq for Resource<T> {
    fn eq(&self, other: &Self) -> bool {
        self.raw() == other.raw()
    }
}
impl<T: Eq> Eq for Resource<T> {}
impl<T: Hash> Hash for Resource<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.raw().hash(state);
    }
}

struct DeviceInner {
    decoded: OnceLock<Option<crate::pipeline_cache::decoded::Store>>,
    meshes: OnceLock<Option<Arc<crate::pipeline_cache::mesh::MeshCache>>>,
    diagnostics: Option<Arc<diagnostics::Diagnostics>>,
    pipeline_profile: OnceLock<Profile>,
    movie_session: std::sync::atomic::AtomicU64,
    history: OnceLock<Option<history::History>>,
    raw: wgpu::Device,
    queue: wgpu::Queue,
    runtime: GpuRuntime,
    pipeline_cache: OnceLock<PipelineCacheState>,
    pipeline_compile_stats: parking_lot::Mutex<(u64, u64)>,
    index: Mutex<FnvHashMap<ObjectKey, Vec<Weak<dyn Indexed>>>>,
    pending_writes: Mutex<UseSet>,
    upload_sealers: Mutex<Vec<Weak<dyn crate::frame_upload::SealUploads>>>,
    submission_gate: Mutex<()>,
    submission_order: submission_order::SubmissionOrder,
    external_submission_lock: Mutex<Option<Arc<Mutex<()>>>>,
    pending_submissions: Mutex<
        std::collections::VecDeque<(crate::gpu_runtime::SubmissionSerial, wgpu::SubmissionIndex)>,
    >,
}
#[derive(Clone)]
pub struct Device(Arc<dyn DeviceAccess>, ResourceOwner);
// Erasing the concrete native device keeps downstream auto-trait evaluation
// independent of wgpu-core's deep backend types. No unsafe Send/Sync assertion.
trait DeviceAccess: Send + Sync {
    fn inner(&self) -> &DeviceInner;
}
impl DeviceAccess for DeviceInner {
    fn inner(&self) -> &DeviceInner {
        self
    }
}

impl Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrackedDevice")
            .field("runtime", &self.0.inner().runtime)
            .finish()
    }
}
impl Deref for Device {
    type Target = wgpu::Device;
    fn deref(&self) -> &Self::Target {
        &self.0.inner().raw
    }
}
fn object_key<T: Any + Hash>(raw: &T) -> ObjectKey {
    let mut hash = fnv::FnvHasher::default();
    raw.hash(&mut hash);
    (TypeId::of::<T>(), hash.finish())
}

/// Pop every scope on the creating thread, including when creation fails. Native
/// scopes normally resolve immediately; a backend fault still has a bounded wait.
fn resolve_scope(
    future: impl Future<Output = Option<wgpu::Error>>,
    device: &wgpu::Device,
) -> Result<Option<wgpu::Error>, RenderFailure> {
    let started = Instant::now();
    let mut future = std::pin::pin!(future);
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    loop {
        if let Poll::Ready(result) = future.as_mut().poll(&mut context) {
            return Ok(result);
        }
        if started.elapsed() >= Duration::from_secs(2) {
            return Err(RenderFailure::new(
                RenderFailureKind::Timeout,
                "GPU error scope timed out",
            ));
        }
        device
            .poll(wgpu::PollType::Poll)
            .map_err(crate::utils::poll_failure)?;
        std::thread::yield_now();
    }
}

fn retryable(error: &RenderFailure) -> bool {
    matches!(
        error.kind(),
        RenderFailureKind::BudgetExceeded | RenderFailureKind::OutOfMemory
    )
}

impl Device {
    /// The caller installs device callbacks exactly once before constructing this.
    pub fn new(raw: wgpu::Device, queue: wgpu::Queue, runtime: GpuRuntime) -> Self {
        Self(
            Arc::new(DeviceInner {
                diagnostics: diagnostics::Diagnostics::new(&raw),
                pipeline_profile: Default::default(),
                movie_session: Default::default(),
                decoded: Default::default(),
                meshes: Default::default(),
                history: Default::default(),
                raw,
                queue,
                runtime,
                pipeline_cache: OnceLock::new(),
                pipeline_compile_stats: Default::default(),
                index: Default::default(),
                pending_writes: Default::default(),
                upload_sealers: Default::default(),
                submission_gate: Default::default(),
                submission_order: Default::default(),
                external_submission_lock: Default::default(),
                pending_submissions: Default::default(),
            }),
            ResourceOwner::default(),
        )
    }
    pub(crate) fn register_upload_sealer(&self, sealer: Arc<dyn crate::frame_upload::SealUploads>) {
        let mut sealers = self
            .0
            .inner()
            .upload_sealers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        sealers.retain(|weak| weak.strong_count() != 0);
        sealers.push(Arc::downgrade(&sealer));
    }
    fn seal_uploads(&self, used: &UseSet) -> Result<(), RenderFailure> {
        let sealers: Vec<_> = self
            .0
            .inner()
            .upload_sealers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .filter_map(Weak::upgrade)
            .collect();
        for sealer in sealers {
            sealer.seal(&|id| used.contains_key(&id))?;
        }
        Ok(())
    }
    pub fn runtime(&self) -> &GpuRuntime {
        &self.0.inner().runtime
    }
    pub fn with_owner(&self, owner: ResourceOwner) -> Self {
        Self(self.0.clone(), owner)
    }
    pub fn raw(&self) -> &wgpu::Device {
        &self.0.inner().raw
    }

    /// Configure the shared cache before creating pipelines. Every owner uses the
    /// first call's configuration and status, including when opening it failed.
    /// Bump `host_abi` when the host's shaders or pipeline layouts change.
    ///
    /// # Safety
    /// `dir` must be dedicated to this installation's native pipeline cache. The
    /// caller must control its ancestry, permissions and contents across runs.
    /// Its parent directory must already exist.
    pub unsafe fn enable_pipeline_cache(&self, dir: &Path, host_abi: &str) -> PipelineCacheStatus {
        self.0
            .inner()
            .meshes
            .get_or_init(|| crate::pipeline_cache::mesh::MeshCache::from_env(dir).map(Arc::new));
        self.0
            .inner()
            .decoded
            .get_or_init(|| crate::pipeline_cache::decoded::Store::from_env(dir));
        self.0
            .inner()
            .history
            .get_or_init(|| history::History::open(self, dir, host_abi));
        self.0
            .inner()
            .pipeline_cache
            .get_or_init(|| {
                // SAFETY: the caller guarantees the cache directory's provenance.
                unsafe { PipelineCacheState::open(self.raw(), dir, host_abi) }
            })
            .status()
    }

    pub(crate) fn mesh_store(&self) -> Option<&Arc<crate::pipeline_cache::mesh::MeshCache>> {
        self.0.inner().meshes.get()?.as_ref()
    }
    pub fn static_mesh_cache_stats(&self) -> Option<(u64, u64)> {
        self.mesh_store().map(|s| s.stats())
    }
    pub(crate) fn decoded_store(&self) -> Option<&crate::pipeline_cache::decoded::Store> {
        self.0.inner().decoded.get()?.as_ref()
    }
    pub fn bitmap_decode_cache_stats(&self) -> Option<(u64, u64)> {
        self.decoded_store().map(|s| s.stats())
    }
    pub fn register_history_kernel(
        &self,
        name: &'static str,
        shader: &'static str,
        layout: Vec<wgpu::BindGroupLayoutEntry>,
    ) {
        if let Some(h) = self.history() {
            h.kernel(self, name, shader, layout)
        }
    }
    pub fn pipeline_history_enabled(&self) -> bool {
        self.history().is_some()
    }
    pub fn pipeline_history_presentation(&self, settings: impl FnOnce() -> String) {
        if let Some(h) = self.history() {
            h.presentation(self, settings());
        }
    }
    /// The native compositor builds its mandatory pipeline while attaching its window.
    pub fn ready_pipeline_history(
        &self,
        family: Family,
        key: impl FnOnce() -> Signature,
    ) -> Option<ExternalHistory> {
        self.history().map(|_| {
            ExternalHistory(history::Observation::new(
                history::Recipe::ready(family, key()),
                0,
                None,
            ))
        })
    }
    pub fn record_external_pipeline_use(
        &self,
        uses: &mut PipelineUses,
        history: Option<&ExternalHistory>,
        profile: Option<&crate::pipeline_profile::Instance>,
    ) {
        uses.record(self, history.map(|o| &o.0), profile.map(|p| p.variant()));
    }
    pub fn pipeline_history_mode(&self) -> &'static str {
        self.history().map_or("off", history::History::mode)
    }
    fn history(&self) -> Option<&history::History> {
        self.0.inner().history.get()?.as_ref()
    }
    pub(crate) fn begin_movie_session(
        &self,
        data: &[u8],
        header: &swf::HeaderExt,
        quality: ruffle_render::quality::StageQuality,
    ) -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let session = NEXT.fetch_add(1, Ordering::Relaxed);
        self.0
            .inner()
            .movie_session
            .store(session, Ordering::Release);
        if let Some(history) = self.history() {
            history.begin(
                self,
                session,
                data,
                &format!("{:?}", header.swf_header()),
                &format!("{quality:?}"),
            );
        }
        session
    }
    pub(crate) fn pipeline_history_quality(
        &self,
        session: u64,
        quality: ruffle_render::quality::StageQuality,
    ) {
        if let Some(h) = self.history() {
            h.quality(self, session, format!("{quality:?}"));
        }
    }
    pub(crate) fn end_movie_session(&self, session: u64) {
        use std::sync::atomic::Ordering;
        if session != 0
            && self
                .0
                .inner()
                .movie_session
                .compare_exchange(session, 0, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            && let Some(h) = self.history()
        {
            h.end(Some(session))
        }
    }
    pub fn background_pipeline_cache(&self) {
        self.schedule_pipeline_cache_save();
        if let Some(h) = self.history() {
            h.save();
        }
    }
    fn speculation_fits(&self) -> bool {
        self.runtime().check_health().is_ok()
            && self.runtime().stats().total_bytes.saturating_add(64 * 1024)
                < self.runtime().config().soft_limit
            && self.runtime().stats().resources.saturating_add(8)
                < self.runtime().config().resource_limit
            && self.runtime().config().cache_limit >= 64 * 1024
    }
    pub fn pipeline_frame(
        &self,
        frame_id: Option<u64>,
        packet_id: Option<u64>,
    ) -> Option<crate::pipeline_profile::FrameScope> {
        (self.pipeline_profile().is_some() || self.history().is_some()).then(|| {
            crate::pipeline_profile::frame_scope(crate::pipeline_profile::Frame {
                device: self.runtime().generation(),
                session: self
                    .0
                    .inner()
                    .movie_session
                    .load(std::sync::atomic::Ordering::Acquire),
                frame_id,
                packet_id,
            })
        })
    }
    pub fn enable_pipeline_profile(&self) {
        self.0
            .inner()
            .pipeline_profile
            .get_or_init(Profile::default);
    }

    pub fn pipeline_profile(&self) -> Option<&Profile> {
        self.0.inner().pipeline_profile.get()
    }

    pub fn profile_variant(
        &self,
        family: Family,
        kind: Kind,
        key: Signature,
    ) -> Option<Arc<Variant>> {
        self.pipeline_profile()?
            .variant(self.runtime().generation(), family, kind, key)
    }

    /// Observe a host-owned native object without changing the legacy tracked-pipeline counters.
    pub fn create_profiled_external<T>(
        &self,
        family: Family,
        kind: Kind,
        key: impl FnOnce() -> Signature,
        create: impl FnOnce() -> T,
    ) -> (T, Option<crate::pipeline_profile::Instance>) {
        let variant = self
            .pipeline_profile()
            .and_then(|_| self.profile_variant(family, kind, key()));
        match variant {
            Some(v) => (v.measure(create), Some(v.instance())),
            None => (create(), None),
        }
    }

    fn resource_signature<T: Any + Eq + Hash>(&self, raw: &T) -> Option<Signature> {
        self.lookup_object(raw).ok()?.profile_info().map(|v| v.key)
    }

    pub fn pipeline_cache(&self) -> Option<&wgpu::PipelineCache> {
        self.0
            .inner()
            .pipeline_cache
            .get()
            .and_then(PipelineCacheState::handle)
    }

    pub fn pipeline_cache_status(&self) -> PipelineCacheStatus {
        self.0
            .inner()
            .pipeline_cache
            .get()
            .map_or(PipelineCacheStatus::Disabled, PipelineCacheState::status)
    }

    pub fn pipeline_cache_dirty(&self) {
        if let Some(c) = self.0.inner().pipeline_cache.get() {
            c.dirty()
        }
    }
    pub fn schedule_pipeline_cache_save(&self) {
        if let Some(c) = self.0.inner().pipeline_cache.get() {
            c.schedule()
        }
    }

    pub fn flush_pipeline_cache(&self) -> bool {
        if let Some(h) = self.history() {
            h.save();
        }
        self.0
            .inner()
            .pipeline_cache
            .get()
            .is_some_and(PipelineCacheState::flush)
    }

    /// Number of tracked native render/compute creation calls (including retries)
    /// and their summed wall-clock time in nanoseconds, across all device owners.
    pub fn pipeline_compile_stats(&self) -> (u64, u64) {
        *self.0.inner().pipeline_compile_stats.lock()
    }

    fn time_pipeline_creation<T>(&self, create: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let pipeline = create();
        let nanos = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let mut stats = self.0.inner().pipeline_compile_stats.lock();
        stats.0 = stats.0.saturating_add(1);
        stats.1 = stats.1.saturating_add(nanos);
        pipeline
    }

    pub fn queue(&self) -> Queue {
        Queue(self.clone())
    }

    pub(crate) fn lookup<T: Any + Eq + Hash>(
        &self,
        raw: &T,
    ) -> Result<ResourceLease, RenderFailure> {
        self.lookup_object(raw).map(|object| object.lease().clone())
    }
    fn lookup_object<T: Any + Eq + Hash>(
        &self,
        raw: &T,
    ) -> Result<Arc<dyn Indexed>, RenderFailure> {
        let index = self
            .0
            .inner()
            .index
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        index
            .get(&object_key(raw))
            .and_then(|bucket| {
                bucket
                    .iter()
                    .filter_map(Weak::upgrade)
                    .find(|object| object.matches(raw))
            })
            .ok_or_else(|| {
                RenderFailure::new(
                    RenderFailureKind::Validation,
                    "untracked or foreign GPU dependency",
                )
            })
    }
    fn wrap<T: Any + Eq + Hash + Send + Sync>(&self, raw: T, lease: ResourceLease) -> Resource<T> {
        let key = object_key(&raw);
        let object = Arc::new(Object {
            metadata: OnceLock::new(),
            raw,
            lease,
            device: self.clone(),
        });
        let erased: Arc<dyn Indexed> = object.clone();
        let mut index = self
            .0
            .inner()
            .index
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Stale weak entries must not form an unbounded second metadata cache.
        if index.len() >= self.runtime().stats().resources.saturating_add(256) {
            index.retain(|_, bucket| {
                bucket.retain(|v| v.strong_count() > 0);
                !bucket.is_empty()
            });
        }
        index.entry(key).or_default().push(Arc::downgrade(&erased));
        drop(index);
        Resource(object)
    }
    fn create<T: Any + Eq + Hash + Send + Sync>(
        &self,
        kind: ResourceKind,
        bytes: u64,
        dependencies: &[ResourceLease],
        create: impl FnMut(&wgpu::Device) -> Result<T, RenderFailure>,
    ) -> Result<Resource<T>, RenderFailure> {
        let (raw, lease) = self.allocate(kind, bytes, dependencies, create)?;
        Ok(self.wrap(raw, lease))
    }
    fn allocate<T>(
        &self,
        kind: ResourceKind,
        bytes: u64,
        dependencies: &[ResourceLease],
        mut create: impl FnMut(&wgpu::Device) -> Result<T, RenderFailure>,
    ) -> Result<(T, ResourceLease), RenderFailure> {
        if history::speculative() {
            if !self.speculation_fits() {
                return Err(RenderFailure::new(
                    RenderFailureKind::BudgetExceeded,
                    "prewarm skipped under pressure",
                ));
            }
            let reservation = self.runtime().reserve(kind, self.1, bytes)?;
            let raw = self.scoped(create)?;
            return Ok((raw, reservation.commit(dependencies)?));
        }
        self.reclaim_soft(bytes);
        self.recover(|| {
            let reservation = self.runtime().reserve(kind, self.1, bytes)?;
            let raw = self.scoped(&mut create)?;
            Ok((raw, reservation.commit(dependencies)?))
        })
    }
    fn recover<T>(
        &self,
        mut attempt: impl FnMut() -> Result<T, RenderFailure>,
    ) -> Result<T, RenderFailure> {
        match attempt() {
            Err(error) if retryable(&error) => {
                self.runtime().record_creation_retry();
                self.reclaim_and_drain()?;
                attempt()
            }
            result => result,
        }
    }

    fn reclaim_soft(&self, bytes: u64) {
        use crate::resource_scheduler::ReclaimClass;
        let scheduler = self.runtime().scheduler();
        while self.runtime().stats().total_bytes.saturating_add(bytes)
            > self.runtime().config().soft_limit
        {
            if !scheduler.evict_one(ReclaimClass::Idle) && !scheduler.evict_one(ReclaimClass::Cache)
            {
                break;
            }
        }
    }
    fn reclaim_and_drain(&self) -> Result<(), RenderFailure> {
        self.runtime().check_health()?;
        self.runtime().scheduler().reclaim();
        let deadline = Instant::now() + Duration::from_secs(2);
        // Wait through the oldest own submission first. Consumer completion on
        // the shared queue may still retain imported stage slots after that.
        if self.runtime().stats().submitted > self.runtime().stats().completed {
            self.wait_for_oldest(deadline)?;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(RenderFailure::new(
                RenderFailureKind::Timeout,
                "GPU reclamation timed out",
            ));
        }
        // No submission gate, cache mutex, or shared UI lock is held while polling.
        self.raw()
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(remaining),
            })
            .map_err(crate::utils::poll_failure)?;
        self.runtime().check_health()?;
        // The wait can make idle pages eligible for the first time. Reclaim
        // those completed pages before the single allocation retry.
        self.runtime().scheduler().reclaim();
        Ok(())
    }
    fn reserve_with_reclaim(
        &self,
        kind: ResourceKind,
        bytes: u64,
    ) -> Result<crate::gpu_runtime::ResourceReservation, RenderFailure> {
        self.reclaim_soft(bytes);
        match self.runtime().reserve(kind, self.1, bytes) {
            Err(error) if retryable(&error) => {
                self.runtime().record_creation_retry();
                self.reclaim_and_drain()?;
                self.runtime().reserve(kind, self.1, bytes)
            }
            result => result,
        }
    }
    pub fn set_submission_lock(&self, lock: Arc<Mutex<()>>) {
        *self
            .0
            .inner()
            .external_submission_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(lock);
    }
    fn submission_lock(&self) -> Option<Arc<Mutex<()>>> {
        self.0
            .inner()
            .external_submission_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
    fn wait_for_oldest(&self, deadline: Instant) -> Result<(), RenderFailure> {
        self.runtime().check_health()?;
        let completed = self.runtime().stats().completed;
        let oldest = {
            let mut pending = self
                .0
                .inner()
                .pending_submissions
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            while pending
                .front()
                .is_some_and(|(serial, _)| serial.sequence() <= completed)
            {
                pending.pop_front();
            }
            pending.front().map(|(_, index)| index.clone())
        };
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(RenderFailure::new(
                RenderFailureKind::Timeout,
                "GPU submission limit wait timed out",
            ));
        }
        self.raw()
            .poll(wgpu::PollType::Wait {
                submission_index: oldest,
                timeout: Some(remaining),
            })
            .map_err(crate::utils::poll_failure)?;
        self.runtime().check_health()
    }

    fn scoped<T>(
        &self,
        create: impl FnOnce(&wgpu::Device) -> Result<T, RenderFailure>,
    ) -> Result<T, RenderFailure> {
        self.runtime().check_health()?;
        let oom = self.raw().push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal = self.raw().push_error_scope(wgpu::ErrorFilter::Internal);
        let validation = self.raw().push_error_scope(wgpu::ErrorFilter::Validation);
        let value = create(self.raw());
        // Pop first, then await, so no future can leave another scope installed.
        let validation = validation.pop();
        let internal = internal.pop();
        let oom = oom.pop();
        let validation = resolve_scope(validation, self.raw());
        let internal = resolve_scope(internal, self.raw());
        let oom = resolve_scope(oom, self.raw());
        self.runtime().check_health()?;
        for result in [oom, internal, validation] {
            if let Some(error) = result? {
                return Err(GpuRuntime::wgpu_failure(error));
            }
        }
        value
    }

    pub fn configure_surface(
        &self,
        surface: &wgpu::Surface<'_>,
        config: &wgpu::SurfaceConfiguration,
    ) -> Result<(), RenderFailure> {
        self.configure_surface_with_lock(surface, config, None)
    }

    /// Serializes raw configuration attempts with an embedding frontend's
    /// queue submissions. Error-scope polling and retries run outside the lock.
    pub fn configure_surface_with_lock(
        &self,
        surface: &wgpu::Surface<'_>,
        config: &wgpu::SurfaceConfiguration,
        submission_lock: Option<&Mutex<()>>,
    ) -> Result<(), RenderFailure> {
        self.configure_with(submission_lock, |device| {
            surface.configure(device, config);
            Ok(())
        })
    }

    fn configure_with(
        &self,
        submission_lock: Option<&Mutex<()>>,
        mut configure: impl FnMut(&wgpu::Device) -> Result<(), RenderFailure>,
    ) -> Result<(), RenderFailure> {
        self.recover(|| {
            self.scoped(|device| {
                // Exclude concurrent submits only while wgpu needs the queue
                // idle. Error-scope polling and OOM recovery may run callbacks.
                let _guard = submission_lock.map(|lock| {
                    lock.lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                });
                configure(device)
            })
        })
    }

    pub fn create_buffer(
        &self,
        desc: &wgpu::BufferDescriptor<'_>,
    ) -> Result<Buffer, RenderFailure> {
        let kind = if desc.usage.contains(wgpu::BufferUsages::MAP_READ) {
            ResourceKind::ReadbackBuffer
        } else if desc.usage.contains(wgpu::BufferUsages::MAP_WRITE) {
            ResourceKind::StagingBuffer
        } else {
            ResourceKind::Buffer
        };
        self.create_buffer_with_kind(desc, kind)
    }
    pub(crate) fn create_buffer_with_kind(
        &self,
        desc: &wgpu::BufferDescriptor<'_>,
        kind: ResourceKind,
    ) -> Result<Buffer, RenderFailure> {
        self.create(
            kind,
            desc.size,
            &[],
            |device| Ok(device.create_buffer(desc)),
        )
    }

    pub fn create_buffer_init(
        &self,
        desc: &wgpu::util::BufferInitDescriptor<'_>,
    ) -> Result<Buffer, RenderFailure> {
        let len = desc.contents.len() as u64;
        let size = len
            .checked_add(wgpu::COPY_BUFFER_ALIGNMENT - 1)
            .map(|v| v & !(wgpu::COPY_BUFFER_ALIGNMENT - 1))
            .ok_or_else(|| {
                RenderFailure::new(
                    RenderFailureKind::BudgetExceeded,
                    "buffer alignment overflow",
                )
            })?;
        self.create(ResourceKind::Buffer, size, &[], |device| {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: desc.label,
                size,
                usage: desc.usage,
                mapped_at_creation: size > 0,
            });
            if size > 0 {
                let mapped = buffer.get_mapped_range_mut(..);
                match mapped {
                    Ok(mut mapped) => {
                        mapped
                            .slice(..desc.contents.len())
                            .copy_from_slice(desc.contents);
                        drop(mapped);
                        buffer.unmap();
                    }
                    Err(error) => {
                        return Err(RenderFailure::new(
                            RenderFailureKind::MapFailed,
                            error.to_string(),
                        ));
                    }
                }
            }
            Ok(buffer)
        })
    }
    pub fn create_texture(
        &self,
        desc: &wgpu::TextureDescriptor<'_>,
    ) -> Result<Texture, RenderFailure> {
        self.create_texture_with_kind(desc, ResourceKind::Texture)
    }
    pub fn create_texture_with_kind(
        &self,
        desc: &wgpu::TextureDescriptor<'_>,
        kind: ResourceKind,
    ) -> Result<Texture, RenderFailure> {
        self.create(kind, texture_storage_bytes(desc)?, &[], |device| {
            Ok(device.create_texture(desc))
        })
    }
    pub fn import_texture(
        &self,
        texture: wgpu::Texture,
        kind: ResourceKind,
    ) -> Result<Texture, RenderFailure> {
        if let Ok(lease) = self.lookup(&texture) {
            return Ok(self.wrap(texture, lease));
        }
        let bytes = texture_storage_bytes(&wgpu::TextureDescriptor {
            label: None,
            size: texture.size(),
            mip_level_count: texture.mip_level_count(),
            sample_count: texture.sample_count(),
            dimension: texture.dimension(),
            format: texture.format(),
            usage: texture.usage(),
            view_formats: &[],
        })?;
        let reservation = self.reserve_with_reclaim(kind, bytes)?;
        Ok(self.wrap(texture, reservation.commit(&[])?))
    }
    pub fn create_sampler(
        &self,
        desc: &wgpu::SamplerDescriptor<'_>,
    ) -> Result<Sampler, RenderFailure> {
        self.create(ResourceKind::Sampler, 0, &[], |device| {
            Ok(device.create_sampler(desc))
        })
    }
    pub fn create_bind_group_layout(
        &self,
        desc: &wgpu::BindGroupLayoutDescriptor<'_>,
    ) -> Result<BindGroupLayout, RenderFailure> {
        let layout = self.create(ResourceKind::Pipeline, 0, &[], |device| {
            Ok(device.create_bind_group_layout(desc))
        })?;
        if self.pipeline_profile().is_some() || self.history().is_some() {
            let mut entries = desc.entries.to_vec();
            entries.sort_unstable_by_key(|e| e.binding);
            let key = signature(format_args!("{entries:?}"));
            layout.set_profile(key, None);
            if let Some(h) = self.history() {
                h.group(self, key, &layout)
            }
        }
        Ok(layout)
    }
    pub fn create_bind_group(
        &self,
        desc: &wgpu::BindGroupDescriptor<'_>,
    ) -> Result<BindGroup, RenderFailure> {
        let mut dependencies = vec![self.lookup(desc.layout)?];
        for entry in desc.entries {
            match &entry.resource {
                wgpu::BindingResource::Buffer(buffer) => {
                    dependencies.push(self.lookup(buffer.buffer)?)
                }
                wgpu::BindingResource::BufferArray(buffers) => {
                    for buffer in *buffers {
                        dependencies.push(self.lookup(buffer.buffer)?);
                    }
                }
                wgpu::BindingResource::Sampler(sampler) => {
                    dependencies.push(self.lookup(*sampler)?)
                }
                wgpu::BindingResource::SamplerArray(samplers) => {
                    for sampler in *samplers {
                        dependencies.push(self.lookup(*sampler)?);
                    }
                }
                wgpu::BindingResource::TextureView(view) => dependencies.push(self.lookup(*view)?),
                wgpu::BindingResource::TextureViewArray(views) => {
                    for view in *views {
                        dependencies.push(self.lookup(*view)?);
                    }
                }
                _ => {
                    return Err(RenderFailure::new(
                        RenderFailureKind::Validation,
                        "unsupported native binding resource",
                    ));
                }
            }
        }
        self.create(ResourceKind::BindGroup, 0, &dependencies, |device| {
            Ok(device.create_bind_group(desc))
        })
    }
    pub fn create_pipeline_layout(
        &self,
        desc: &wgpu::PipelineLayoutDescriptor<'_>,
    ) -> Result<PipelineLayout, RenderFailure> {
        let dependencies: Result<Vec<_>, _> = desc
            .bind_group_layouts
            .iter()
            .flatten()
            .map(|v| self.lookup(*v))
            .collect();
        let layout = self.create(ResourceKind::Pipeline, 0, &dependencies?, |device| {
            Ok(device.create_pipeline_layout(desc))
        })?;
        if self.pipeline_profile().is_some() || self.history().is_some() {
            let groups: Vec<_> = desc
                .bind_group_layouts
                .iter()
                .map(|g| g.map(|g| self.resource_signature(g)))
                .collect();
            if self.history().is_some()
                && let Some(groups) = groups
                    .iter()
                    .map(|g| match g {
                        Some(v) => v.map(Some),
                        None => Some(None),
                    })
                    .collect::<Option<Vec<_>>>()
            {
                let _ = layout.meta().layout.set(Box::new(history::Layout {
                    groups,
                    immediate: desc.immediate_size,
                }));
            }
            layout.set_profile(
                signature(format_args!("{groups:?} {}", desc.immediate_size)),
                None,
            );
        }
        Ok(layout)
    }
    pub fn create_shader_module(
        &self,
        desc: wgpu::ShaderModuleDescriptor<'_>,
    ) -> Result<ShaderModule, RenderFailure> {
        self.create_shader_module_for(Family::TwoD, desc)
    }
    pub fn create_shader_module_for(
        &self,
        family: Family,
        desc: wgpu::ShaderModuleDescriptor<'_>,
    ) -> Result<ShaderModule, RenderFailure> {
        let profile = (self.pipeline_profile().is_some() || self.history().is_some()).then(|| {
            let key = signature(format_args!("{:?}", desc.source));
            (key, self.profile_variant(family, Kind::Shader, key))
        });
        let module = self.create(ResourceKind::Shader, 0, &[], |device| {
            let create = || device.create_shader_module(desc.clone());
            Ok(match &profile {
                Some((_, Some(v))) => v.measure(create),
                _ => create(),
            })
        })?;
        if let Some((key, v)) = profile {
            module.set_profile(key, v);
            if let Some(h) = self.history() {
                h.shader(self, key, &module)
            }
        }
        if let Some(diagnostics) = &self.0.inner().diagnostics {
            diagnostics.shader(self, &desc, &module);
        }
        Ok(module)
    }
    pub fn create_render_pipeline(
        &self,
        desc: &wgpu::RenderPipelineDescriptor<'_>,
    ) -> Result<RenderPipeline, RenderFailure> {
        self.create_render_pipeline_for(Family::TwoD, desc)
    }
    pub(crate) fn register_agal(&self, program: &Arc<crate::context3d::shader_pair::AgalProgram>) {
        if let Some(h) = self.history() {
            h.agal(self, program)
        }
    }
    pub(crate) fn create_agal_pipeline(
        &self,
        content: Signature,
        spec: &crate::context3d::shader_pair::ShaderCompileData,
        desc: &wgpu::RenderPipelineDescriptor<'_>,
    ) -> Result<RenderPipeline, RenderFailure> {
        if let Some(h) = self.history()
            && let Some(mut recipe) = history::Recipe::render(self, Family::Stage3d, desc)
        {
            recipe.agal = Some((content, spec.clone()));
            return h.render(self, Family::Stage3d, desc, recipe);
        }
        self.create_render_direct(Family::Stage3d, desc)
    }
    pub fn create_render_pipeline_for(
        &self,
        family: Family,
        desc: &wgpu::RenderPipelineDescriptor<'_>,
    ) -> Result<RenderPipeline, RenderFailure> {
        if let Some(h) = self.history()
            && let Some(recipe) = history::Recipe::render(self, family, desc)
        {
            return h.render(self, family, desc, recipe);
        }
        self.create_render_direct(family, desc)
    }
    fn create_render_direct(
        &self,
        family: Family,
        desc: &wgpu::RenderPipelineDescriptor<'_>,
    ) -> Result<RenderPipeline, RenderFailure> {
        let mut dependencies = vec![self.lookup(desc.vertex.module)?];
        if let Some(layout) = desc.layout {
            dependencies.push(self.lookup(layout)?);
        }
        if let Some(fragment) = &desc.fragment {
            dependencies.push(self.lookup(fragment.module)?);
        }
        let profile = self.pipeline_profile().map(|_| {
            let fragment = desc.fragment.as_ref().map(|f| {
                (
                    self.resource_signature(f.module),
                    f.entry_point,
                    &f.compilation_options,
                    f.targets,
                )
            });
            let key = signature(format_args!(
                "{:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?}",
                desc.layout.map(|l| self.resource_signature(l)),
                self.resource_signature(desc.vertex.module),
                desc.vertex.entry_point,
                desc.vertex.compilation_options,
                desc.vertex.buffers,
                fragment,
                desc.primitive,
                desc.depth_stencil,
                desc.multisample,
                desc.multiview_mask
            ));
            (key, self.profile_variant(family, Kind::Render, key))
        });
        let desc = wgpu::RenderPipelineDescriptor {
            cache: desc.cache.or_else(|| self.pipeline_cache()),
            ..desc.clone()
        };
        let pipeline = self.create(ResourceKind::Pipeline, 0, &dependencies, |device| {
            let create = || self.time_pipeline_creation(|| device.create_render_pipeline(&desc));
            Ok(match &profile {
                Some((_, Some(v))) => v.measure(create),
                _ => create(),
            })
        })?;
        if let Some((key, v)) = profile {
            pipeline.set_profile(key, v);
        }
        self.pipeline_cache_dirty();
        if let Some(diagnostics) = &self.0.inner().diagnostics {
            diagnostics.pipeline(self, &desc, &pipeline);
        }
        Ok(pipeline)
    }
    pub fn create_compute_pipeline(
        &self,
        desc: &wgpu::ComputePipelineDescriptor<'_>,
    ) -> Result<ComputePipeline, RenderFailure> {
        self.create_compute_pipeline_for(Family::Filter, desc)
    }
    pub fn create_compute_pipeline_for(
        &self,
        family: Family,
        desc: &wgpu::ComputePipelineDescriptor<'_>,
    ) -> Result<ComputePipeline, RenderFailure> {
        if let Some(h) = self.history()
            && let Some(recipe) = history::Recipe::compute(self, family, desc)
        {
            return h.compute(self, family, desc, recipe);
        }
        self.create_compute_direct(family, desc)
    }
    fn create_compute_direct(
        &self,
        family: Family,
        desc: &wgpu::ComputePipelineDescriptor<'_>,
    ) -> Result<ComputePipeline, RenderFailure> {
        let mut dependencies = vec![self.lookup(desc.module)?];
        if let Some(layout) = desc.layout {
            dependencies.push(self.lookup(layout)?);
        }
        let profile = self.pipeline_profile().map(|_| {
            let key = signature(format_args!(
                "{:?} {:?} {:?} {:?}",
                desc.layout.map(|l| self.resource_signature(l)),
                self.resource_signature(desc.module),
                desc.entry_point,
                desc.compilation_options
            ));
            (key, self.profile_variant(family, Kind::Compute, key))
        });
        let desc = wgpu::ComputePipelineDescriptor {
            cache: desc.cache.or_else(|| self.pipeline_cache()),
            ..desc.clone()
        };
        let pipeline = self.create(ResourceKind::Pipeline, 0, &dependencies, |device| {
            let create = || self.time_pipeline_creation(|| device.create_compute_pipeline(&desc));
            Ok(match &profile {
                Some((_, Some(v))) => v.measure(create),
                _ => create(),
            })
        })?;
        if let Some((key, v)) = profile {
            pipeline.set_profile(key, v);
        }
        self.pipeline_cache_dirty();
        Ok(pipeline)
    }
    pub fn create_query_set(
        &self,
        desc: &wgpu::QuerySetDescriptor<'_>,
    ) -> Result<QuerySet, RenderFailure> {
        self.create(
            ResourceKind::QuerySet,
            u64::from(desc.count) * 8,
            &[],
            |device| Ok(device.create_query_set(desc)),
        )
    }

    pub fn create_command_encoder(
        &self,
        desc: &wgpu::CommandEncoderDescriptor<'_>,
    ) -> Result<CommandEncoder, RenderFailure> {
        let (raw, lease) = self.allocate(ResourceKind::CommandEncoder, 0, &[], |device| {
            Ok(device.create_command_encoder(desc))
        })?;
        let uses = Arc::new(Mutex::new(Recording {
            pipeline_uses: Default::default(),
            uses: UseSet::default(),
            objects: Default::default(),
            transient_bytes: 0,
            staging_bytes: 0,
            pending: Vec::new(),
        }));
        {
            let mut recording = uses
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            recording.retain(&lease);
        }
        Ok(CommandEncoder {
            raw,
            uses,
            debug_groups: Vec::new(),
            diagnostics: Vec::new(),
            bitmap_writes: None,
            device: self.clone(),
        })
    }
}
impl Texture {
    pub fn create_view(
        &self,
        desc: &wgpu::TextureViewDescriptor<'_>,
    ) -> Result<TextureView, RenderFailure> {
        self.0.device.create(
            ResourceKind::TextureView,
            0,
            &[self.lease().clone()],
            |_| Ok(self.raw().create_view(desc)),
        )
    }
}

impl TextureView {
    /// Recover the tracked parent even when only a surface view remains alive.
    /// This aliases its lease and preserves the view's explicit color format.
    pub fn texture_handle(&self) -> Result<Texture, RenderFailure> {
        let parent = self.lease().dependencies().first().ok_or_else(|| {
            RenderFailure::new(
                RenderFailureKind::Validation,
                "texture view has no tracked parent",
            )
        })?;
        Ok(self
            .0
            .device
            .wrap(self.raw().texture().clone(), parent.clone()))
    }
}

pub struct CommandEncoder {
    diagnostics: Vec<diagnostics::Pending>,
    raw: wgpu::CommandEncoder,
    debug_groups: Vec<String>,
    bitmap_writes: Option<Arc<AtomicBool>>,
    uses: Arc<Mutex<Recording>>,
    device: Device,
}
impl CommandEncoder {
    /// Reserve this device's queue order before handing a stage to its executor.
    pub fn into_stage_work(self) -> Result<StageWork, RenderFailure> {
        let turn = self.device.queue().reserve_submission()?;
        Ok(StageWork {
            encoder: self,
            turn,
        })
    }

    /// Bitmap readers on another session can proceed once this encoder is submitted.
    pub(crate) fn pending_bitmap_write(&mut self) -> Arc<AtomicBool> {
        self.bitmap_writes
            .get_or_insert_with(Default::default)
            .clone()
    }

    pub fn transient_bytes(&self) -> u64 {
        self.uses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .transient_bytes
    }
    /// Called only between passes with no mapped upload views held. Submitting
    /// the already recorded prefix never replays a draw or an ActionScript write.
    pub(crate) fn flush_uploads_if_needed(&mut self, incoming: u64) -> Result<(), RenderFailure> {
        let staging = self
            .uses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .staging_bytes;
        let config = self.device.runtime().config();
        if staging == 0
            || (staging.saturating_add(incoming) <= config.staging_limit / 2
                && self.transient_bytes() < config.transient_watermark)
        {
            return Ok(());
        }
        {
            let recording = self
                .uses
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            self.device.seal_uploads(&recording.uses)?;
        }
        let next = self.device.create_command_encoder(&Default::default())?;
        let mut previous = std::mem::replace(self, next);
        let groups = std::mem::take(&mut previous.debug_groups);
        for _ in &groups {
            previous.raw.pop_debug_group();
        }
        for group in &groups {
            self.raw.push_debug_group(group);
        }
        self.debug_groups = groups;
        let commands = previous.finish()?;
        self.device.queue().submit([commands])?;
        Ok(())
    }
    pub fn finish(mut self) -> Result<CommandBuffer, RenderFailure> {
        for pass in self.diagnostics.drain(..) {
            pass.finish(&self.device, &mut self.raw);
        }
        let (uses, pending, pipeline_uses) = {
            let mut recording = self
                .uses
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            (
                std::mem::take(&mut recording.uses),
                std::mem::take(&mut recording.pending),
                std::mem::take(&mut recording.pipeline_uses),
            )
        };
        self.device.runtime().check_health()?;
        let raw = self.device.scoped(|_| Ok(self.raw.finish()))?;
        // Recording::retain already closes this set transitively.
        Ok(CommandBuffer {
            raw,
            uses: RecordedUses(uses),
            pipeline_uses,
            bitmap_writes: self.bitmap_writes,
            pending,
        })
    }
}
/// Owned stage commands and their queue position. Native wgpu-core lowers
/// the recorded passes to the driver inside `finish` on the executing thread.
#[must_use = "dropping stage work cancels its unsubmitted queue turn"]
pub struct StageWork {
    encoder: CommandEncoder,
    turn: ReservedSubmission,
}

impl StageWork {
    /// Append the target readback and completion callbacks before dispatch.
    pub fn encoder(&mut self) -> &mut CommandEncoder {
        &mut self.encoder
    }

    pub fn submit(self) -> Result<wgpu::SubmissionIndex, RenderFailure> {
        self.submit_until_closed(&crossbeam_channel::never())
    }

    /// Disconnecting `closed` releases this stage's unsubmitted turn while a peer holds queue order.
    pub fn submit_until_closed(
        self,
        closed: &crossbeam_channel::Receiver<std::convert::Infallible>,
    ) -> Result<wgpu::SubmissionIndex, RenderFailure> {
        use ruffle_render::profile::{NativePhase, native_time};
        let commands = {
            let _timer = native_time(NativePhase::EncoderFinish);
            self.encoder.finish()?
        };
        let _timer = native_time(NativePhase::QueueSubmit);
        self.turn.submit_until_closed([commands], closed)
    }
}

pub struct CommandBuffer {
    pipeline_uses: PipelineUses,
    raw: wgpu::CommandBuffer,
    bitmap_writes: Option<Arc<AtomicBool>>,
    uses: RecordedUses,
    pending: Vec<crate::gpu_runtime::UnsubmittedUse>,
}

/// All writes and submissions use the same device ledger and ordering gate.
#[derive(Clone, Debug)]
pub struct Queue(Device);
impl Deref for Queue {
    type Target = wgpu::Queue;
    fn deref(&self) -> &Self::Target {
        &self.0.0.inner().queue
    }
}
fn lock_for_submission(lock: &Mutex<()>) -> std::sync::MutexGuard<'_, ()> {
    let _timer =
        ruffle_render::profile::native_time(ruffle_render::profile::NativePhase::SubmitLock);
    lock.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl Queue {
    fn order(&self) -> &submission_order::SubmissionOrder {
        &self.0.0.inner().submission_order
    }
    pub fn reserve_submission(&self) -> Result<ReservedSubmission, RenderFailure> {
        self.order().reserve(self)
    }

    pub fn pending_upload_bytes(&self) -> u64 {
        self.0
            .0
            .inner()
            .pending_writes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .values()
            .map(ResourceLease::transient_bytes)
            .sum()
    }
    pub fn raw(&self) -> &wgpu::Queue {
        &self.0.0.inner().queue
    }

    fn prepare_upload(&self, bytes: u64) -> Result<(), RenderFailure> {
        let limit = self.0.runtime().config().staging_limit;
        if bytes > limit {
            return Err(RenderFailure::new(
                RenderFailureKind::BudgetExceeded,
                "upload slice exceeds staging budget",
            ));
        }
        let pending = self.pending_upload_bytes();
        if pending != 0 && pending.saturating_add(bytes) > limit / 2 {
            self.submit(std::iter::empty())?;
        }
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let stats = self.0.runtime().stats();
            if stats.kind_bytes[ResourceKind::StagingBuffer as usize].saturating_add(bytes) <= limit
                || stats.submitted == stats.completed
            {
                break;
            }
            self.0.wait_for_oldest(deadline)?;
        }
        Ok(())
    }
    fn write(
        &self,
        bytes: u64,
        destination: ResourceLease,
        write: impl FnOnce(&wgpu::Queue),
    ) -> Result<(), RenderFailure> {
        self.order().wait_pending(self)?;
        self.prepare_upload(bytes)?;
        let staging = self
            .0
            .reserve_with_reclaim(ResourceKind::StagingBuffer, bytes)?
            .commit(&[destination])?;
        self.0
            .scoped(|_| {
                let _gate = lock_for_submission(&self.0.0.inner().submission_gate);
                self.0.runtime().check_health()?;
                self.0
                    .0
                    .inner()
                    .pending_writes
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .insert(staging.id(), staging);
                write(self.raw());
                Ok(())
            })
            .map_err(|error| self.drain_after_failure(error, UseSet::default()))
    }

    fn drain_after_failure(&self, error: RenderFailure, submitting: UseSet) -> RenderFailure {
        let error = self.0.runtime().health().record(error);
        let gate = lock_for_submission(&self.0.0.inner().submission_gate);
        let pending = std::mem::take(
            &mut *self
                .0
                .0
                .inner()
                .pending_writes
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        // A rejected native write or submit may leave staging in wgpu's queue.
        // Normal submission is now disabled. Keep the charges through its fence.
        if !pending.is_empty() || !submitting.is_empty() {
            let index = {
                let shared = self.0.submission_lock();
                let _shared = shared.as_deref().map(lock_for_submission);
                let index = self.raw().submit([]);
                self.raw()
                    .on_submitted_work_done(move || drop((pending, submitting)));
                index
            };
            drop(gate);
            // The progress worker exits on failure. Drive this fence outside
            // the locks; a timeout leaves the callback owning the charges.
            let _ = self.0.poll(wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: Some(Duration::from_secs(2)),
            });
        }
        error
    }

    pub fn write_buffer(
        &self,
        buffer: &wgpu::Buffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), RenderFailure> {
        let lease = self.0.lookup(buffer)?;
        if !offset.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)
            || !(data.len() as u64).is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)
            || offset
                .checked_add(data.len() as u64)
                .is_none_or(|end| end > buffer.size())
        {
            return Err(RenderFailure::new(
                RenderFailureKind::Validation,
                "invalid buffer upload range",
            ));
        }
        let capacity = (self.0.runtime().config().staging_limit / 2).min(4 * 1024 * 1024) / 4 * 4;
        if capacity == 0 {
            return Err(RenderFailure::new(
                RenderFailureKind::BudgetExceeded,
                "staging budget cannot hold a buffer copy",
            ));
        }
        for (index, chunk) in data.chunks(capacity as usize).enumerate() {
            self.write(chunk.len() as u64, lease.clone(), |queue| {
                queue.write_buffer(buffer, offset + index as u64 * capacity, chunk)
            })?;
        }
        self.0.runtime().check_health()
    }
    pub fn write_texture(
        &self,
        destination: wgpu::TexelCopyTextureInfo<'_>,
        data: &[u8],
        layout: wgpu::TexelCopyBufferLayout,
        size: wgpu::Extent3d,
    ) -> Result<(), RenderFailure> {
        self.write_texture_observed(destination, data, layout, size, |_, _| {})
    }

    /// Reports payload and charged staging bytes after each successful native write.
    pub(crate) fn write_texture_observed(
        &self,
        destination: wgpu::TexelCopyTextureInfo<'_>,
        data: &[u8],
        layout: wgpu::TexelCopyBufferLayout,
        size: wgpu::Extent3d,
        mut on_write: impl FnMut(u64, u64),
    ) -> Result<(), RenderFailure> {
        let lease = self.0.lookup(destination.texture)?;
        self.0.runtime().check_health()?;
        if size.width == 0 || size.height == 0 || size.depth_or_array_layers == 0 {
            return Ok(());
        }
        let invalid = || {
            RenderFailure::new(
                RenderFailureKind::Validation,
                "invalid texture upload layout or extent",
            )
        };
        let format = destination.texture.format();
        let block = u64::from(
            format
                .block_copy_size(Some(destination.aspect))
                .ok_or_else(invalid)?,
        );
        let (bw, bh) = format.block_dimensions();
        let row = u64::from(size.width.div_ceil(bw)) * block;
        let rows = size.height.div_ceil(bh);
        let stride = match layout.bytes_per_row {
            Some(stride) if u64::from(stride) >= row && u64::from(stride).is_multiple_of(block) => {
                u64::from(stride)
            }
            None if rows == 1 && size.depth_or_array_layers == 1 => row,
            _ => return Err(invalid()),
        };
        let image_rows = match layout.rows_per_image {
            Some(rows_per_image) if rows_per_image >= rows => u64::from(rows_per_image),
            None if size.depth_or_array_layers == 1 => u64::from(rows),
            _ => return Err(invalid()),
        };
        let image_bytes = stride.checked_mul(image_rows).ok_or_else(invalid)?;
        let end = image_bytes
            .checked_mul(u64::from(size.depth_or_array_layers - 1))
            .and_then(|v| v.checked_add(stride * u64::from(rows - 1)))
            .and_then(|v| v.checked_add(row))
            .and_then(|v| v.checked_add(layout.offset))
            .ok_or_else(invalid)?;
        if end > data.len() as u64 || destination.mip_level >= destination.texture.mip_level_count()
        {
            return Err(invalid());
        }
        let extent = destination
            .texture
            .size()
            .mip_level_size(destination.mip_level, destination.texture.dimension())
            .physical_size(format);
        if destination
            .origin
            .x
            .checked_add(size.width)
            .is_none_or(|end| end > extent.width)
            || destination
                .origin
                .y
                .checked_add(size.height)
                .is_none_or(|end| end > extent.height)
            || destination
                .origin
                .z
                .checked_add(size.depth_or_array_layers)
                .is_none_or(|end| end > extent.depth_or_array_layers)
        {
            return Err(invalid());
        }
        let padded = row.div_ceil(u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT))
            * u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let capacity = (self.0.runtime().config().staging_limit / 2)
            .min(4 * 1024 * 1024)
            .max(padded);
        let rows_per_slice = (capacity / padded).max(1).min(u64::from(rows)) as u32;
        for layer in 0..size.depth_or_array_layers {
            let mut first_row = 0;
            while first_row < rows {
                let count = rows_per_slice.min(rows - first_row);
                let mut target = destination;
                target.origin.y += first_row * bh;
                target.origin.z += layer;
                let source = wgpu::TexelCopyBufferLayout {
                    offset: layout.offset
                        + u64::from(layer) * image_bytes
                        + u64::from(first_row) * stride,
                    bytes_per_row: Some(stride as u32),
                    rows_per_image: None,
                };
                let extent = wgpu::Extent3d {
                    width: size.width,
                    height: (count * bh).min(size.height - first_row * bh),
                    depth_or_array_layers: 1,
                };
                let staging_bytes = padded * u64::from(count);
                self.write(staging_bytes, lease.clone(), |queue| {
                    queue.write_texture(target, data, source, extent)
                })?;
                on_write(row * u64::from(count), staging_bytes);
                first_row += count;
            }
        }
        Ok(())
    }
    pub fn submit_external(
        &self,
        commands: impl IntoIterator<Item = wgpu::CommandBuffer>,
        resources: &[ResourceLease],
    ) -> Result<wgpu::SubmissionIndex, RenderFailure> {
        self.order().wait_pending(self)?;
        let commands = commands.into_iter().map(|raw| CommandBuffer {
            raw,
            bitmap_writes: None,
            pipeline_uses: Default::default(),
            uses: RecordedUses::default(),
            pending: Vec::new(),
        });
        self.submit_ordered(commands, resources)
    }
    pub fn submit(
        &self,
        commands: impl IntoIterator<Item = CommandBuffer>,
    ) -> Result<wgpu::SubmissionIndex, RenderFailure> {
        self.order().wait_pending(self)?;
        self.submit_ordered(commands, &[])
    }

    fn submit_ordered(
        &self,
        commands: impl IntoIterator<Item = CommandBuffer>,
        external_resources: &[ResourceLease],
    ) -> Result<wgpu::SubmissionIndex, RenderFailure> {
        let deadline = Instant::now() + Duration::from_secs(2);
        let gate = loop {
            self.0.runtime().check_health()?;
            let gate = lock_for_submission(&self.0.0.inner().submission_gate);
            let stats = self.0.runtime().stats();
            if stats.submitted - stats.completed < 2 {
                break gate;
            }
            drop(gate);
            self.0
                .wait_for_oldest(deadline)
                .map_err(|error| self.0.runtime().health().record(error))?;
        };
        let mut writes = UseSet::default();
        let mut observations = smallvec::SmallVec::<[PipelineUses; 1]>::new();
        // Scopes resolve after the closure has released both submission locks.
        self.0
            .scoped(|_| {
                let shared = self.0.submission_lock();
                let shared = shared.as_deref().map(lock_for_submission);
                let mut raw = smallvec::SmallVec::<[wgpu::CommandBuffer; 1]>::new();
                writes = std::mem::take(
                    &mut *self
                        .0
                        .0
                        .inner()
                        .pending_writes
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner),
                );
                let mut uses = RecordedUses::from_roots(writes.values().chain(external_resources));
                let mut pending = Vec::new();
                let mut bitmap_writes = smallvec::SmallVec::<[Arc<AtomicBool>; 1]>::new();
                for command in commands {
                    if command.pipeline_uses.entries.is_some() {
                        observations.push(command.pipeline_uses);
                    }
                    bitmap_writes.extend(command.bitmap_writes);
                    pending.extend(command.pending);
                    raw.push(command.raw);
                    if uses.0.is_empty() {
                        uses = command.uses;
                    } else {
                        uses.0.extend(command.uses.0);
                    }
                }
                let serial = self.0.runtime().register_recorded_submission(&uses)?;
                drop(pending);
                let index = self.raw().submit(raw);
                for submitted in bitmap_writes {
                    submitted.store(true, Ordering::Release);
                }
                {
                    let completed = self.0.runtime().stats().completed;
                    let mut pending = self
                        .0
                        .0
                        .inner()
                        .pending_submissions
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    while pending
                        .front()
                        .is_some_and(|(serial, _)| serial.sequence() <= completed)
                    {
                        pending.pop_front();
                    }
                    pending.push_back((serial, index.clone()));
                }
                let runtime = self.0.runtime().clone();
                self.raw().on_submitted_work_done(move || {
                    runtime.complete_submission(serial);
                });
                drop(shared);
                drop(gate);
                Ok(index)
            })
            .map_err(|error| self.drain_after_failure(error, writes))
            .inspect(|_| {
                for recorded in observations {
                    recorded.submitted(&self.0);
                }
            })
    }
}

impl RenderPipeline {
    pub fn get_bind_group_layout(&self, index: u32) -> Result<BindGroupLayout, RenderFailure> {
        self.0
            .device
            .create(ResourceKind::Pipeline, 0, &[self.lease().clone()], |_| {
                Ok(self.raw().get_bind_group_layout(index))
            })
    }
}
impl ComputePipeline {
    pub fn get_bind_group_layout(&self, index: u32) -> Result<BindGroupLayout, RenderFailure> {
        self.0
            .device
            .create(ResourceKind::Pipeline, 0, &[self.lease().clone()], |_| {
                Ok(self.raw().get_bind_group_layout(index))
            })
    }
}

#[cfg(test)]
mod tests;

#[test]
#[ignore = "requires a Vulkan adapter"]
fn disabled_cache_allows_pipeline_creation_and_shares_compile_stats() {
    use crate::gpu_runtime::{GpuBudgetConfig, OwnerKind};
    use crate::pipeline_cache::PipelineCacheStatus;
    use futures::executor::block_on;

    let mut options = wgpu::InstanceDescriptor::new_without_display_handle();
    options.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(options);
    let adapter = block_on(instance.request_adapter(&Default::default())).unwrap();
    let (raw, queue) = block_on(adapter.request_device(&Default::default())).unwrap();
    let runtime = GpuRuntime::new(GpuBudgetConfig::for_adapter(
        adapter.get_info().device_type,
        false,
    ))
    .unwrap();
    runtime.install_device_callbacks(&raw);
    let device = Device::new(raw, queue, runtime);
    let owner = device.with_owner(device.runtime().issue_owner(OwnerKind::Host).unwrap());
    assert_eq!(
        device.pipeline_cache_status(),
        PipelineCacheStatus::Disabled
    );
    assert!(owner.pipeline_cache().is_none());
    assert!(!owner.flush_pipeline_cache());

    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("pipeline-cache");
    // SAFETY: this fresh directory is reserved for this test's native cache.
    let status = unsafe { owner.enable_pipeline_cache(&dir, "test-v1") };
    assert_eq!(status, PipelineCacheStatus::Unsupported);
    assert_eq!(device.pipeline_cache_status(), status);
    assert!(device.pipeline_cache().is_none());
    assert!(!device.flush_pipeline_cache());

    let module = owner
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()),
        })
        .unwrap();
    assert_eq!(device.pipeline_compile_stats(), (0, 0));
    owner
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
        .unwrap();
    let (count, nanos) = device.pipeline_compile_stats();
    assert_eq!(count, 1);
    assert!(nanos > 0);
}

/// Native callers share the submission-aware paged upload implementation.
pub type StagingBelt = crate::frame_upload::FrameUploadArena;

impl Device {
    pub fn create_texture_with_data(
        &self,
        queue: &Queue,
        desc: &wgpu::TextureDescriptor<'_>,
        order: wgpu::util::TextureDataOrder,
        data: &[u8],
    ) -> Result<Texture, RenderFailure> {
        let mut desc = desc.clone();
        desc.usage |= wgpu::TextureUsages::COPY_DST;
        let texture = self.create_texture(&desc)?;
        let invalid = || {
            RenderFailure::new(
                RenderFailureKind::Validation,
                "invalid texture initialization data",
            )
        };
        let block = desc.format.block_copy_size(None).ok_or_else(invalid)?;
        let (bw, bh) = desc.format.block_dimensions();
        let layers = desc.array_layer_count();
        use wgpu::util::TextureDataOrder;
        let (outer_count, inner_count) = match order {
            TextureDataOrder::LayerMajor => (layers, desc.mip_level_count),
            TextureDataOrder::MipMajor => (desc.mip_level_count, layers),
        };
        let mut offset = 0usize;
        for outer in 0..outer_count {
            for inner in 0..inner_count {
                let (layer, mip) = match order {
                    TextureDataOrder::LayerMajor => (outer, inner),
                    TextureDataOrder::MipMajor => (inner, outer),
                };
                let mut extent = desc.mip_level_size(mip).ok_or_else(invalid)?;
                if desc.dimension != wgpu::TextureDimension::D3 {
                    extent.depth_or_array_layers = 1;
                }
                let physical = extent.physical_size(desc.format);
                let row = (physical.width / bw)
                    .checked_mul(block)
                    .ok_or_else(invalid)?;
                let rows = physical.height / bh;
                let len = u64::from(row)
                    .checked_mul(u64::from(rows))
                    .and_then(|v| v.checked_mul(u64::from(extent.depth_or_array_layers)))
                    .and_then(|v| usize::try_from(v).ok())
                    .ok_or_else(invalid)?;
                let end = offset.checked_add(len).ok_or_else(invalid)?;
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: mip,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: layer,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    data.get(offset..end).ok_or_else(invalid)?,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(row),
                        rows_per_image: Some(rows),
                    },
                    physical,
                )?;
                offset = end;
            }
        }
        Ok(texture)
    }
}

/// A failed construction never populates a cache entry.
pub(crate) fn try_cached<K: Eq + Hash, V, S: std::hash::BuildHasher, E>(
    cache: &mut std::collections::HashMap<K, V, S>,
    key: K,
    create: impl FnOnce() -> Result<V, E>,
) -> Result<&mut V, E> {
    Ok(match cache.entry(key) {
        std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
        std::collections::hash_map::Entry::Vacant(entry) => entry.insert(create()?),
    })
}

pub(crate) fn try_cell<T, E>(
    cell: &std::cell::OnceCell<T>,
    init: impl FnOnce() -> Result<T, E>,
) -> Result<&T, E> {
    if let Some(value) = cell.get() {
        return Ok(value);
    }
    let value = init()?;
    Ok(cell.get_or_init(|| value))
}
pub(crate) fn try_sync_cell<T, E>(
    cell: &std::sync::OnceLock<T>,
    init: impl FnOnce() -> Result<T, E>,
) -> Result<&T, E> {
    if let Some(value) = cell.get() {
        return Ok(value);
    }
    let value = init()?;
    Ok(cell.get_or_init(|| value))
}
pub(crate) fn try_option<T, E>(
    slot: &mut Option<T>,
    init: impl FnOnce() -> Result<T, E>,
) -> Result<&mut T, E> {
    if slot.is_none() {
        *slot = Some(init()?);
    }
    Ok(slot
        .as_mut()
        .expect("successful initializer set the option"))
}
impl ruffle_render::backend::RawTexture for Texture {
    fn equals(&self, other: &dyn ruffle_render::backend::RawTexture) -> bool {
        (other as &dyn Any)
            .downcast_ref::<Self>()
            .is_some_and(|other| self == other)
    }
}
