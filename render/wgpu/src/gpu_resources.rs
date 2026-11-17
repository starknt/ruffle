//! Native allocation boundary. GPU objects carry a shared lease, while this
//! device's lookup table contains weak references only. Bindings retain leases
//! for their parents, even after the caller releases its texture/buffer handle.
//!
//! Legacy encoders conservatively retain all live resources and resources
//! created while recording. The pass graph supplies precise use sets later;
//! until then this deliberately overestimates lifetimes, never underestimates.
use crate::gpu_runtime::{
    GpuRuntime, RenderFailure, RenderFailureKind, ResourceId, ResourceKind, ResourceLease,
    ResourceOwner, texture_storage_bytes,
};
use fnv::FnvHashMap;
use std::{
    any::{Any, TypeId},
    fmt::Debug,
    future::Future,
    hash::{Hash, Hasher},
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex, Weak},
    task::{Context, Poll},
    time::{Duration, Instant},
};

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
#[derive(Default)]
struct Recording {
    uses: UseSet,
    open: bool,
    transient_bytes: u64,
}
impl Recording {
    fn retain(&mut self, lease: ResourceLease) {
        if let std::collections::hash_map::Entry::Vacant(entry) = self.uses.entry(lease.id()) {
            self.transient_bytes += lease.transient_bytes();
            entry.insert(lease);
        }
    }
}

trait Indexed: Any + Send + Sync {
    fn matches(&self, raw: &dyn Any) -> bool;
    fn lease(&self) -> &ResourceLease;
}
struct Object<T> {
    raw: T,
    lease: ResourceLease,
    device: Device,
}
impl<T: Any + Eq + Send + Sync> Indexed for Object<T> {
    fn matches(&self, raw: &dyn Any) -> bool {
        raw.downcast_ref::<T>().is_some_and(|raw| self.raw == *raw)
    }
    fn lease(&self) -> &ResourceLease {
        &self.lease
    }
}

pub struct Resource<T>(Arc<Object<T>>);
impl<T> Resource<T> {
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
    raw: wgpu::Device,
    queue: wgpu::Queue,
    runtime: GpuRuntime,
    index: Mutex<FnvHashMap<ObjectKey, Vec<Weak<dyn Indexed>>>>,
    recordings: Mutex<Vec<Weak<Mutex<Recording>>>>,
    pending_writes: Mutex<UseSet>,
    submission_gate: Mutex<()>,
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
                raw,
                queue,
                runtime,
                index: Default::default(),
                recordings: Default::default(),
                pending_writes: Default::default(),
                submission_gate: Default::default(),
                external_submission_lock: Default::default(),
                pending_submissions: Default::default(),
            }),
            ResourceOwner::default(),
        )
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

    pub fn queue(&self) -> Queue {
        Queue(self.clone())
    }

    fn lookup<T: Any + Eq + Hash>(&self, raw: &T) -> Result<ResourceLease, RenderFailure> {
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
                    .map(|object| object.lease().clone())
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
        let mut recordings = self
            .0
            .inner()
            .recordings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        recordings.retain(|weak| {
            if let Some(recording) = weak.upgrade() {
                let mut recording = recording
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if recording.open {
                    recording.retain(object.lease.clone());
                }
                recording.open
            } else {
                false
            }
        });
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
        self.runtime().check_health()
    }
    fn reserve_with_reclaim(
        &self,
        kind: ResourceKind,
        bytes: u64,
    ) -> Result<crate::gpu_runtime::ResourceReservation, RenderFailure> {
        self.reclaim_soft(bytes);
        match self.runtime().reserve(kind, self.1, bytes) {
            Err(error) if retryable(&error) => {
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
        self.recover(|| {
            self.scoped(|device| {
                surface.configure(device, config);
                Ok(())
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
        self.create(ResourceKind::Pipeline, 0, &[], |device| {
            Ok(device.create_bind_group_layout(desc))
        })
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
        self.create(ResourceKind::Pipeline, 0, &dependencies?, |device| {
            Ok(device.create_pipeline_layout(desc))
        })
    }
    pub fn create_shader_module(
        &self,
        desc: wgpu::ShaderModuleDescriptor<'_>,
    ) -> Result<ShaderModule, RenderFailure> {
        self.create(ResourceKind::Shader, 0, &[], |device| {
            Ok(device.create_shader_module(desc.clone()))
        })
    }
    pub fn create_render_pipeline(
        &self,
        desc: &wgpu::RenderPipelineDescriptor<'_>,
    ) -> Result<RenderPipeline, RenderFailure> {
        let mut dependencies = vec![self.lookup(desc.vertex.module)?];
        if let Some(layout) = desc.layout {
            dependencies.push(self.lookup(layout)?);
        }
        if let Some(fragment) = &desc.fragment {
            dependencies.push(self.lookup(fragment.module)?);
        }
        self.create(ResourceKind::Pipeline, 0, &dependencies, |device| {
            Ok(device.create_render_pipeline(desc))
        })
    }
    pub fn create_compute_pipeline(
        &self,
        desc: &wgpu::ComputePipelineDescriptor<'_>,
    ) -> Result<ComputePipeline, RenderFailure> {
        let mut dependencies = vec![self.lookup(desc.module)?];
        if let Some(layout) = desc.layout {
            dependencies.push(self.lookup(layout)?);
        }
        self.create(ResourceKind::Pipeline, 0, &dependencies, |device| {
            Ok(device.create_compute_pipeline(desc))
        })
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

    fn live_uses(&self) -> UseSet {
        let mut uses = UseSet::default();
        let mut index = self
            .0
            .inner()
            .index
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        index.retain(|_, bucket| {
            bucket.retain(|weak| {
                if let Some(resource) = weak.upgrade() {
                    uses.insert(resource.lease().id(), resource.lease().clone());
                    true
                } else {
                    false
                }
            });
            !bucket.is_empty()
        });
        uses
    }
    pub fn create_command_encoder(
        &self,
        desc: &wgpu::CommandEncoderDescriptor<'_>,
    ) -> Result<CommandEncoder, RenderFailure> {
        let (raw, lease) = self.allocate(ResourceKind::CommandEncoder, 0, &[], |device| {
            Ok(device.create_command_encoder(desc))
        })?;
        let uses = Arc::new(Mutex::new(Recording {
            uses: UseSet::default(),
            open: true,
            transient_bytes: 0,
        }));
        {
            let mut recordings = self
                .0
                .inner()
                .recordings
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            recordings.retain(|recording| recording.strong_count() > 0);
            recordings.push(Arc::downgrade(&uses));
        }
        {
            let mut recording = uses
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for lease in self.live_uses().into_values() {
                recording.retain(lease);
            }
            recording.retain(lease);
        }
        Ok(CommandEncoder {
            raw,
            uses,
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

pub struct CommandEncoder {
    raw: wgpu::CommandEncoder,
    uses: Arc<Mutex<Recording>>,
    device: Device,
}
impl CommandEncoder {
    pub fn transient_bytes(&self) -> u64 {
        self.uses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .transient_bytes
    }
    pub fn finish(self) -> Result<CommandBuffer, RenderFailure> {
        let uses = {
            let mut recording = self
                .uses
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            recording.open = false;
            std::mem::take(&mut recording.uses)
        };
        let raw = self.device.scoped(|_| Ok(self.raw.finish()))?;
        Ok(CommandBuffer {
            raw,
            uses: uses.into_values().collect(),
        })
    }
}
impl Deref for CommandEncoder {
    type Target = wgpu::CommandEncoder;
    fn deref(&self) -> &Self::Target {
        &self.raw
    }
}
impl DerefMut for CommandEncoder {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.raw
    }
}
pub struct CommandBuffer {
    raw: wgpu::CommandBuffer,
    uses: Vec<ResourceLease>,
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
impl Queue {
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

    fn write(
        &self,
        bytes: u64,
        destination: ResourceLease,
        write: impl FnOnce(&wgpu::Queue),
    ) -> Result<(), RenderFailure> {
        let staging = self
            .0
            .reserve_with_reclaim(ResourceKind::StagingBuffer, bytes)?
            .commit(&[destination])?;
        self.0
            .scoped(|_| {
                let _gate = self
                    .0
                    .0
                    .inner()
                    .submission_gate
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
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
            .map_err(|error| self.0.runtime().health().record(error))
    }

    pub fn write_buffer(
        &self,
        buffer: &wgpu::Buffer,
        offset: wgpu::BufferAddress,
        data: &[u8],
    ) -> Result<(), RenderFailure> {
        self.write(data.len() as u64, self.0.lookup(buffer)?, |queue| {
            queue.write_buffer(buffer, offset, data)
        })
    }
    pub fn write_texture(
        &self,
        destination: wgpu::TexelCopyTextureInfo<'_>,
        data: &[u8],
        layout: wgpu::TexelCopyBufferLayout,
        size: wgpu::Extent3d,
    ) -> Result<(), RenderFailure> {
        let format = destination.texture.format();
        let block = format
            .block_copy_size(Some(destination.aspect))
            .ok_or_else(|| {
                RenderFailure::new(
                    RenderFailureKind::Validation,
                    "texture upload has no copyable aspect",
                )
            })?;
        let (bw, bh) = format.block_dimensions();
        let row = u64::from(size.width.div_ceil(bw)) * u64::from(block);
        let padded = row.div_ceil(u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT))
            * u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let bytes = padded
            .checked_mul(u64::from(size.height.div_ceil(bh)))
            .and_then(|v| v.checked_mul(u64::from(size.depth_or_array_layers)))
            .ok_or_else(|| {
                RenderFailure::new(
                    RenderFailureKind::BudgetExceeded,
                    "texture upload size overflow",
                )
            })?;
        self.write(bytes, self.0.lookup(destination.texture)?, |queue| {
            queue.write_texture(destination, data, layout, size)
        })
    }
    pub fn submit_external(
        &self,
        commands: impl IntoIterator<Item = wgpu::CommandBuffer>,
        resources: &[ResourceLease],
    ) -> Result<wgpu::SubmissionIndex, RenderFailure> {
        self.0
            .scoped(|_| {
                let _gate = self
                    .0
                    .0
                    .inner()
                    .submission_gate
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let shared = self.0.submission_lock();
                let _shared = shared.as_ref().map(|lock| {
                    lock.lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                });
                let mut retained = std::mem::take(
                    &mut *self
                        .0
                        .0
                        .inner()
                        .pending_writes
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner),
                );
                retained.extend(resources.iter().map(|lease| (lease.id(), lease.clone())));
                let index = self.raw().submit(commands);
                self.raw().on_submitted_work_done(move || drop(retained));
                Ok(index)
            })
            .map_err(|error| self.0.runtime().health().record(error))
    }
    pub fn submit(
        &self,
        commands: impl IntoIterator<Item = CommandBuffer>,
    ) -> Result<wgpu::SubmissionIndex, RenderFailure> {
        let deadline = Instant::now() + Duration::from_secs(2);
        let gate = loop {
            self.0.runtime().check_health()?;
            let gate = self
                .0
                .0
                .inner()
                .submission_gate
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let stats = self.0.runtime().stats();
            if stats.submitted - stats.completed < 2 {
                break gate;
            }
            drop(gate);
            self.0
                .wait_for_oldest(deadline)
                .map_err(|error| self.0.runtime().health().record(error))?;
        };
        // Scopes resolve after the closure has released both submission locks.
        self.0
            .scoped(|_| {
                let shared = self.0.submission_lock();
                let shared = shared.as_ref().map(|lock| {
                    lock.lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                });
                let mut raw = Vec::new();
                let mut uses = std::mem::take(
                    &mut *self
                        .0
                        .0
                        .inner()
                        .pending_writes
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner),
                );
                for command in commands {
                    raw.push(command.raw);
                    uses.extend(command.uses.into_iter().map(|lease| (lease.id(), lease)));
                }
                let leases: Vec<_> = uses.into_values().collect();
                let serial = self.0.runtime().register_submission(&leases)?;
                let index = self.raw().submit(raw);
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
            .map_err(|error| self.0.runtime().health().record(error))
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

/// Tracked staging foundation. Pages and completion-aware reuse are supplied by
/// the frame upload arena; until then every admitted slice has an exact lease.
#[derive(Debug)]
pub struct StagingBelt {
    device: Device,
    active: Vec<Buffer>,
}
impl StagingBelt {
    pub fn new(device: Device, _chunk_size: u64) -> Self {
        Self {
            device,
            active: Vec::new(),
        }
    }
    pub fn write_buffer(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::Buffer,
        offset: u64,
        size: wgpu::BufferSize,
    ) -> Result<wgpu::BufferViewMut, RenderFailure> {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tracked upload slice"),
            size: size.get(),
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        })?;
        let view = buffer
            .get_mapped_range_mut(..)
            .map_err(|e| RenderFailure::new(RenderFailureKind::MapFailed, e.to_string()))?;
        encoder.copy_buffer_to_buffer(&buffer, 0, target, offset, size.get());
        self.active.push(buffer);
        Ok(view)
    }
    pub fn finish(&mut self) {
        for buffer in self.active.drain(..) {
            buffer.unmap();
        }
    }
    pub fn recall(&mut self) {}
}

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
