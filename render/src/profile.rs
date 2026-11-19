//! Per-player instrumentation. A host scopes synchronous AVM/renderer work to its own
//! session. Allocation guards retain the owning counters across GC and delayed drops.
use std::{cell::RefCell, rc::Rc};
use web_time::Instant;

macro_rules! work_counters {
    ($($name:ident),* $(,)?) => {
        #[derive(Debug, Default, Clone, Copy)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize))]
        pub struct RenderWork { $(pub $name: u64,)* }
        impl RenderWork {
            pub fn entries(&self) -> impl Iterator<Item = (&'static str, u64)> {
                [$( (stringify!($name), self.$name),)*].into_iter()
            }
            pub fn delta(self, previous: Self) -> Self {
                Self { $($name: self.$name.saturating_sub(previous.$name),)* }
            }
            pub fn add(&mut self, other: Self) {
                $(self.$name = self.$name.saturating_add(other.$name);)*
            }
        }
    }
}
// `pass_pixels` sums color attachment areas per encoded pass (before scissor),
// not shaded fragments. `snapshot_bytes` counts cumulative admissions, not
// current resident bytes. `mask_passes` counts passes using stencil/alpha masks.
work_counters!(
    draw_calls,
    render_passes,
    blend_passes,
    mask_passes,
    filter_passes,
    offscreen_targets,
    offscreen_pixels,
    pass_pixels,
    copy_bytes,
    pipeline_changes,
    material_binds,
    cache_hits,
    cache_redraws,
    tree_visits,
    render_bounds_nodes,
    world_matrix_builds,
    child_slots_examined,
    child_slots_copied,
    render_instances_updated,
    render_chunks_rebuilt,
    prepared_chunk_hits,
    prepared_draw_replays,
    captured_command_leaves,
    snapshot_hits,
    snapshot_misses,
    snapshot_evictions,
    snapshot_bytes,
    pic_hits,
    pic_misses,
    pic_megamorphic,
    catch_up_frames,
    discarded_frame_debt_us,
    native_pipeline_changes,
    native_bind_groups,
    native_vertex_buffers,
    native_index_buffers,
    native_scissors,
    native_stencil_references,
    native_viewports,
);

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct GpuSample {
    pub frame: Option<u64>,
    pub nanos: u64,
    /// Inclusive named query durations. Nested names must not be summed as total frame time.
    pub phases: Vec<(String, u64)>,
}

#[derive(Debug, Clone, Copy)]
pub enum NativePhase {
    EncoderFinish,
    SubmitLock,
    QueueSubmit,
    SurfaceAcquire,
    Present,
    QueueThrottle,
    Preparation,
    CommandPreparation,
}
impl NativePhase {
    pub const NAMES: [&'static str; 8] = [
        "encoderFinish",
        "submitLock",
        "queueSubmit",
        "surfaceAcquire",
        "present",
        "queueThrottle",
        "preparation",
        "commandPreparation",
    ];
}

/// Aggregate locally in renderer hot loops, then publish once per chunk.
pub fn work(f: impl FnOnce(&mut RenderWork)) {
    CURRENT.with_borrow(|current| {
        if let Some(metrics) = current {
            f(&mut metrics.0.borrow_mut().work);
        }
    });
}

pub struct NativeTimer(Option<(RenderMetrics, NativePhase, Instant)>);
pub fn native_time(phase: NativePhase) -> NativeTimer {
    NativeTimer(CURRENT.with_borrow(|m| m.clone().map(|m| (m, phase, Instant::now()))))
}
impl Drop for NativeTimer {
    fn drop(&mut self) {
        if let Some((metrics, phase, start)) = &self.0 {
            let mut counters = metrics.0.borrow_mut();
            let timing = &mut counters.native_timings[*phase as usize];
            let nanos = start.elapsed().as_nanos() as u64;
            timing.samples += 1;
            timing.nanos += nanos;
            timing.max_nanos = timing.max_nanos.max(nanos);
        }
    }
}

#[cfg(feature = "lifecycle_counts")]
pub mod lifecycle;

#[derive(Debug, Clone, Copy)]
pub enum Phase {
    Logic,
    Gc,
    Decode,
    Upload,
    Readback,
    Gpu,
    Executor,
    StageCpu,
    HostAction,
    Preload,
    Avm2,
    Avm1,
    Audio,
    PostFrame,
    Connections,
    AudioSkew,
    Socket,
    NetConnection,
    WallTimers,
    Timers,
    Streams,
    AudioTick,
    SocketReceive,
    SocketConnect,
    SocketData,
    SocketClose,
    Wake,
    Input,
    Composite,
    RenderTree,
    RenderBackend,
    RenderSubmit,
}
impl Phase {
    pub const NAMES: [&'static str; 32] = [
        "logic",
        "gc",
        "decode",
        "upload",
        "readback",
        "gpu",
        "executor",
        "stageCpu",
        "hostAction",
        "preload",
        "avm2",
        "avm1",
        "audio",
        "postFrame",
        "connections",
        "audioSkew",
        "socket",
        "netConnection",
        "wallTimers",
        "timers",
        "streams",
        "audioTick",
        "socketReceive",
        "socketConnect",
        "socketData",
        "socketClose",
        "wake",
        "input",
        "composite",
        "renderTree",
        "renderBackend",
        "renderSubmit",
    ];
}
#[derive(Debug, Clone, Copy, Default)]
pub struct Timing {
    pub samples: u64,
    pub nanos: u64,
    pub max_nanos: u64,
}
/// Numeric-only evidence for long collector pauses; never roots arena objects.
#[derive(Debug, Clone, Copy)]
pub struct GcSample {
    pub nanos: u64,
    pub phase_before: &'static str,
    pub phase_after: &'static str,
    pub objects_before: usize,
    pub objects_after: usize,
    pub debt_before: f64,
    pub debt_after: f64,
}
#[derive(Debug, Clone)]
pub struct AsyncEvent {
    pub kind: &'static str,
    pub generation: u64,
    pub queue_depth: u64,
    pub duration_ns: u64,
    pub bytes: u64,
    pub error: Option<&'static str>,
}

impl Default for AsyncEvent {
    fn default() -> Self {
        Self {
            kind: "unknown",
            generation: 0,
            queue_depth: 0,
            duration_ns: 0,
            bytes: 0,
            error: None,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct PublishedCodeStats {
    pub methods: u64,
    pub input_storage_bytes: u64,
    pub output_storage_bytes: u64,
    pub instructions: u64,
    pub trace_items: u64,
}

#[derive(Debug, Default, Clone)]
pub struct Counters {
    pub work: RenderWork,
    pub native_timings: [Timing; NativePhase::NAMES.len()],
    pub native_resources: Option<NativeResources>,
    pub gpu_samples: Vec<GpuSample>,
    pub dropped_gpu_samples: u64,
    pub gc_samples: Vec<GcSample>,
    pub dropped_gc_samples: u64,
    #[cfg(feature = "lifecycle_counts")]
    pub lifecycle: lifecycle::Counters,
    pub timings: [Timing; 32],
    pub logical_frames: u64,
    pub socket_events: [u64; 3],
    pub socket_data_bytes: u64,
    pub cpu_bitmap_bytes: u64,
    pub peak_cpu_bitmap_bytes: u64,
    pub cpu_bitmaps: u64,
    pub stage_submissions: u64,
    pub gpu_bitmaps: u64,
    pub gpu_bitmap_bytes: u64,
    pub peak_gpu_bitmap_bytes: u64,
    pub gpu_meshes: u64,
    pub gpu_mesh_bytes: u64,
    pub peak_gpu_mesh_bytes: u64,
    pub gpu_frame: Option<u64>,
    pub bitmap_readbacks: u64,
    pub bitmap_readback_bytes: u64,
    pub bitmap_readback_nanos: u64,
    pub bitmap_gpu_copies: u64,
    pub bitmap_gpu_copy_fallbacks: u64,
    pub loader_bytes: [u64; 3],
    pub idle: [u64; 2],
    pub peak_idle: [u64; 2],
    pub peak_pool: u64,
    pub upload_bytes: u64,
    pub executor_queue_depth: u64,
    pub blend_batch_splits: u64,
    pub blend_batch_fallbacks: u64,
    pub bitmap_residency: Option<BitmapResidencyStats>,
    /// Cumulative publication totals, not currently live allocations or RSS.
    pub published_code: PublishedCodeStats,
    pub async_events: Vec<AsyncEvent>,
}

/// Device-wide logical accounting. Driver allocations and process RSS are
/// different measurements and are deliberately not folded into these bytes.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct NativeResources {
    pub device_generation: u64,
    pub logical_bytes: u64,
    pub active_bytes: u64,
    pub reserved_bytes: u64,
    pub idle_bytes: u64,
    pub retiring_bytes: u64,
    pub peak_bytes: u64,
    pub cache_retained_bytes: u64,
    pub pool_retained_bytes: u64,
    pub resources: usize,
    pub classes: Vec<(&'static str, u64, usize)>,
    pub submitted: u64,
    pub completed: u64,
    pub budget_rejections: u64,
    pub creation_retries: u64,
    pub terminal_failure: Option<String>,
    pub worker_active: usize,
    pub worker_bytes: u64,
    pub worker_peak_active: usize,
    pub worker_peak_bytes: u64,
    pub worker_submitted: u64,
}
pub fn record_native_resources(stats: NativeResources) {
    CURRENT.with_borrow(|current| {
        if let Some(metrics) = current {
            metrics.0.borrow_mut().native_resources = Some(stats);
        }
    });
}

#[derive(Clone, Copy, Default, Debug)]
pub struct BitmapResidencyStats {
    pub retained_bytes: u64,
    pub retained_entries: usize,
    pub hits: u64,
    pub insertions: u64,
    pub evictions: u64,
}

pub fn record_bitmap_residency(stats: BitmapResidencyStats) {
    CURRENT.with_borrow(|current| {
        if let Some(metrics) = current {
            metrics.0.borrow_mut().bitmap_residency = Some(stats);
        }
    });
}
#[derive(Clone, Default, Debug)]
pub struct RenderMetrics(
    Rc<RefCell<Counters>>,
    std::sync::Arc<std::sync::Mutex<GpuBitmapStats>>,
);
thread_local! {
    static CURRENT: RefCell<Option<RenderMetrics>> = const { RefCell::new(None) };
}
pub fn enabled() -> bool {
    CURRENT.with_borrow(Option::is_some)
}
pub struct Scope(Option<RenderMetrics>);
pub fn enter(metrics: Option<&RenderMetrics>) -> Scope {
    Scope(CURRENT.replace(metrics.cloned()))
}
impl Drop for Scope {
    fn drop(&mut self) {
        CURRENT.set(self.0.take());
    }
}
impl RenderMetrics {
    pub fn take_gc_samples(&self) -> (Vec<GcSample>, u64) {
        let mut counters = self.0.borrow_mut();
        (
            std::mem::take(&mut counters.gc_samples),
            std::mem::take(&mut counters.dropped_gc_samples),
        )
    }
    pub fn take_gpu_samples(&self) -> Vec<GpuSample> {
        std::mem::take(&mut self.0.borrow_mut().gpu_samples)
    }
    pub fn snapshot(&self) -> Counters {
        let mut snapshot = self.0.borrow().clone();
        let bitmaps = self
            .1
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        snapshot.gpu_bitmaps = bitmaps.count;
        snapshot.gpu_bitmap_bytes = bitmaps.bytes;
        snapshot.peak_gpu_bitmap_bytes = bitmaps.peak;
        snapshot
    }

    pub fn take_async_events(&self) -> Vec<AsyncEvent> {
        std::mem::take(&mut self.0.borrow_mut().async_events)
    }
    fn record(&self, phase: Phase, nanos: u64) {
        let mut all = self.0.borrow_mut();
        let t = &mut all.timings[phase as usize];
        t.samples += 1;
        t.nanos += nanos;
        t.max_nanos = t.max_nanos.max(nanos);
    }
}
pub fn next_frame() -> Option<u64> {
    CURRENT.with_borrow(|m| m.as_ref().map(|m| m.0.borrow().stage_submissions + 1))
}
pub fn record_gpu(nanos: u64, frame: Option<u64>) {
    record_gpu_sample(GpuSample {
        frame,
        nanos,
        phases: Vec::new(),
    });
}
pub fn record_gpu_sample(sample: GpuSample) {
    let (frame, nanos) = (sample.frame, sample.nanos);
    CURRENT.with_borrow(|m| {
        if let Some(m) = m {
            m.record(Phase::Gpu, nanos);
            let mut counters = m.0.borrow_mut();
            counters.gpu_frame = frame;
            if counters.gpu_samples.len() < 256 {
                counters.gpu_samples.push(sample);
            } else {
                counters.dropped_gpu_samples += 1;
            }
        }
    });
}
pub fn submitted() {
    CURRENT.with_borrow(|m| {
        if let Some(m) = m {
            m.0.borrow_mut().stage_submissions += 1;
        }
    });
}

pub fn logical_frame() {
    CURRENT.with_borrow(|m| {
        if let Some(m) = m {
            m.0.borrow_mut().logical_frames += 1;
        }
    });
}

pub fn record_host_duration(phase: Phase, nanos: u64) {
    CURRENT.with_borrow(|m| {
        if let Some(m) = m {
            m.record(phase, nanos);
        }
    });
}

pub fn record_gc(sample: GcSample) {
    if sample.nanos < 8_000_000 {
        return;
    }
    CURRENT.with_borrow(|current| {
        if let Some(metrics) = current {
            let mut counters = metrics.0.borrow_mut();
            if counters.gc_samples.len() < 32 {
                counters.gc_samples.push(sample);
            } else {
                counters.dropped_gc_samples += 1;
            }
        }
    });
}
pub fn uploaded(bytes: u64) {
    CURRENT.with_borrow(|m| {
        if let Some(m) = m {
            m.0.borrow_mut().upload_bytes += bytes;
        }
    });
}

pub fn bitmap_readback(bytes: u64, nanos: u64) {
    CURRENT.with_borrow(|m| {
        if let Some(m) = m {
            let mut counters = m.0.borrow_mut();
            counters.bitmap_readbacks += 1;
            counters.bitmap_readback_bytes += bytes;
            counters.bitmap_readback_nanos += nanos;
        }
    });
}

pub fn bitmap_gpu_copy(supported: bool) {
    CURRENT.with_borrow(|m| {
        if let Some(m) = m {
            let mut counters = m.0.borrow_mut();
            if supported {
                counters.bitmap_gpu_copies += 1;
            } else {
                counters.bitmap_gpu_copy_fallbacks += 1;
            }
        }
    });
}

pub fn blend_batch_split() {
    CURRENT.with_borrow(|m| {
        if let Some(m) = m {
            m.0.borrow_mut().blend_batch_splits += 1;
        }
    });
}

pub fn code_published(
    input_bytes: usize,
    output_bytes: usize,
    instructions: usize,
    trace_items: usize,
) {
    CURRENT.with_borrow(|m| {
        if let Some(m) = m {
            let mut counters = m.0.borrow_mut();
            let stats = &mut counters.published_code;
            stats.methods += 1;
            stats.input_storage_bytes += input_bytes as u64;
            stats.output_storage_bytes += output_bytes as u64;
            stats.instructions += instructions as u64;
            stats.trace_items += trace_items as u64;
        }
    });
}

pub fn blend_batch_fallback() {
    CURRENT.with_borrow(|m| {
        if let Some(m) = m {
            m.0.borrow_mut().blend_batch_fallbacks += 1;
        }
    });
}
pub fn socket_connect() {
    CURRENT.with_borrow(|m| {
        if let Some(m) = m {
            m.0.borrow_mut().socket_events[0] += 1;
        }
    });
}
pub fn socket_data(bytes: usize) {
    CURRENT.with_borrow(|m| {
        if let Some(m) = m {
            let mut counters = m.0.borrow_mut();
            counters.socket_events[1] += 1;
            counters.socket_data_bytes += bytes as u64;
        }
    });
}
pub fn socket_close() {
    CURRENT.with_borrow(|m| {
        if let Some(m) = m {
            m.0.borrow_mut().socket_events[2] += 1;
        }
    });
}
pub struct Timer(Option<(RenderMetrics, Phase, Instant)>);
pub fn time(phase: Phase) -> Timer {
    Timer(CURRENT.with_borrow(|m| m.clone().map(|m| (m, phase, Instant::now()))))
}
impl Drop for Timer {
    fn drop(&mut self) {
        if let Some((m, p, s)) = &self.0 {
            m.record(*p, s.elapsed().as_nanos() as u64);
        }
    }
}
#[derive(Debug, Default)]
pub struct BitmapAllocation(Option<(RenderMetrics, u64)>);
impl BitmapAllocation {
    pub fn new(bytes: usize) -> Self {
        Self(CURRENT.with_borrow(|m| {
            m.clone().filter(|_| bytes > 0).map(|m| {
                let bytes = bytes as u64;
                {
                    let mut s = m.0.borrow_mut();
                    s.cpu_bitmap_bytes += bytes;
                    s.cpu_bitmaps += 1;
                    s.peak_cpu_bitmap_bytes = s.peak_cpu_bitmap_bytes.max(s.cpu_bitmap_bytes);
                }
                (m, bytes)
            })
        }))
    }
}
impl Drop for BitmapAllocation {
    fn drop(&mut self) {
        if let Some((m, b)) = &self.0 {
            let mut s = m.0.borrow_mut();
            s.cpu_bitmap_bytes -= *b;
            s.cpu_bitmaps -= 1;
        }
    }
}

pub fn pool(idle: [u64; 2], peak: [u64; 2], combined: u64) {
    CURRENT.with_borrow(|m| {
        if let Some(m) = m {
            let mut c = m.0.borrow_mut();
            c.idle = idle;
            c.peak_idle = peak;
            c.peak_pool = combined;
        }
    });
}

#[derive(Debug, Default)]
struct GpuBitmapStats {
    count: u64,
    bytes: u64,
    peak: u64,
}
#[derive(Debug, Default)]
pub struct GpuBitmap(Option<(std::sync::Arc<std::sync::Mutex<GpuBitmapStats>>, u64)>);
impl GpuBitmap {
    pub fn new(bytes: u64) -> Self {
        Self(CURRENT.with_borrow(|current| {
            current.as_ref().map(|metrics| {
                let mut counts = metrics
                    .1
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                counts.count += 1;
                counts.bytes += bytes;
                counts.peak = counts.peak.max(counts.bytes);
                (metrics.1.clone(), bytes)
            })
        }))
    }
}
impl Drop for GpuBitmap {
    fn drop(&mut self) {
        if let Some((metrics, bytes)) = &self.0 {
            let mut counts = metrics
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            counts.count -= 1;
            counts.bytes -= bytes;
        }
    }
}

#[derive(Debug, Default)]
pub struct GpuMesh(Option<(RenderMetrics, u64)>);
impl GpuMesh {
    pub fn new(bytes: u64) -> Self {
        Self(CURRENT.with_borrow(|m| {
            m.clone().map(|m| {
                {
                    let mut c = m.0.borrow_mut();
                    c.gpu_meshes += 1;
                    c.gpu_mesh_bytes += bytes;
                    c.peak_gpu_mesh_bytes = c.peak_gpu_mesh_bytes.max(c.gpu_mesh_bytes);
                }
                (m, bytes)
            })
        }))
    }
}
impl Drop for GpuMesh {
    fn drop(&mut self) {
        if let Some((m, b)) = &self.0 {
            let mut c = m.0.borrow_mut();
            c.gpu_meshes -= 1;
            c.gpu_mesh_bytes -= b;
        }
    }
}

pub fn async_event(
    kind: &'static str,
    generation: u64,
    queue_depth: usize,
    duration_ns: u64,
    bytes: usize,
    error: Option<&'static str>,
) {
    CURRENT.with_borrow(|m| {
        if let Some(m) = m {
            m.0.borrow_mut().async_events.push(AsyncEvent {
                kind,
                generation,
                queue_depth: queue_depth as u64,
                duration_ns,
                bytes: bytes as u64,
                error,
            });
        }
    });
}

pub fn loader_bytes(index: usize, bytes: usize) {
    CURRENT.with_borrow(|m| {
        if let Some(m) = m {
            m.0.borrow_mut().loader_bytes[index] += bytes as u64;
        }
    });
}

pub fn executor_queue_depth(depth: usize) {
    CURRENT.with_borrow(|m| {
        if let Some(m) = m {
            m.0.borrow_mut().executor_queue_depth = depth as u64;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::{
        RenderMetrics, bitmap_gpu_copy, bitmap_readback, enabled, enter, executor_queue_depth,
    };

    #[test]
    fn slow_gc_diagnostics_are_scoped_bounded_and_drained_once() {
        let metrics = RenderMetrics::default();
        let sample = super::GcSample {
            nanos: 10_000_000,
            phase_before: "marking",
            phase_after: "sweeping",
            objects_before: 100,
            objects_after: 80,
            debt_before: 500.0,
            debt_after: 0.0,
        };
        {
            let _scope = enter(Some(&metrics));
            super::record_gc(super::GcSample {
                nanos: 100,
                ..sample
            });
            for i in 0..40 {
                super::record_gc(super::GcSample {
                    objects_before: 100 + i,
                    ..sample
                });
            }
        }
        super::record_gc(sample);
        let (samples, dropped) = metrics.take_gc_samples();
        assert_eq!((samples.len(), dropped), (32, 8));
        assert_eq!(samples.first().unwrap().objects_before, 100);
        assert_eq!(samples.last().unwrap().objects_before, 131);
        assert_eq!(samples[0].phase_after, "sweeping");
        assert_eq!(samples[0].debt_before, 500.0);
        assert_eq!(samples[0].objects_after, 80);
        let (samples, dropped) = metrics.take_gc_samples();
        assert!(samples.is_empty());
        assert_eq!(dropped, 0);
    }

    #[test]
    fn completed_gpu_samples_preserve_each_frame_and_drain_once() {
        let metrics = RenderMetrics::default();
        let _scope = enter(Some(&metrics));
        super::record_gpu(11, Some(3));
        super::record_gpu(17, Some(4));
        let samples = metrics.take_gpu_samples();
        assert_eq!(
            samples
                .iter()
                .map(|s| (s.frame, s.nanos))
                .collect::<Vec<_>>(),
            [(Some(3), 11), (Some(4), 17)]
        );
        assert!(metrics.take_gpu_samples().is_empty());
    }

    #[test]
    fn work_counters_belong_to_the_scoped_player() {
        let a = RenderMetrics::default();
        let b = RenderMetrics::default();
        {
            let _scope = enter(Some(&a));
            super::work(|w| w.draw_calls += 2);
            {
                let _scope = enter(Some(&b));
                super::work(|w| w.draw_calls += 7);
            }
            super::work(|w| w.draw_calls += 3);
        }
        assert_eq!(a.snapshot().work.draw_calls, 5);
        assert_eq!(b.snapshot().work.draw_calls, 7);
    }

    #[test]
    fn executor_queue_depth_is_recorded_in_the_current_metrics() {
        let metrics = RenderMetrics::default();
        let _scope = enter(Some(&metrics));

        executor_queue_depth(3);

        assert_eq!(metrics.snapshot().executor_queue_depth, 3);
    }

    #[test]
    fn profiling_is_enabled_only_inside_an_active_scope() {
        assert!(!enabled());
        let metrics = RenderMetrics::default();
        {
            let _scope = enter(Some(&metrics));
            assert!(enabled());
        }
        assert!(!enabled());
    }

    #[test]
    fn bitmap_readback_records_count_bytes_and_time() {
        let metrics = RenderMetrics::default();
        {
            let _scope = enter(Some(&metrics));
            bitmap_readback(4096, 12);
            bitmap_readback(1024, 7);
        }

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.bitmap_readbacks, 2);
        assert_eq!(snapshot.bitmap_readback_bytes, 5120);
        assert_eq!(snapshot.bitmap_readback_nanos, 19);
    }

    #[test]
    fn bitmap_gpu_copy_records_success_and_fallbacks() {
        let metrics = RenderMetrics::default();
        {
            let _scope = enter(Some(&metrics));
            bitmap_gpu_copy(true);
            bitmap_gpu_copy(false);
            bitmap_gpu_copy(true);
        }

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.bitmap_gpu_copies, 2);
        assert_eq!(snapshot.bitmap_gpu_copy_fallbacks, 1);
    }

    #[test]
    fn gpu_mesh_allocation_tracks_live_and_peak_bytes() {
        let metrics = RenderMetrics::default();
        {
            let _scope = enter(Some(&metrics));
            let first = super::GpuMesh::new(128);
            let second = super::GpuMesh::new(256);
            let snapshot = metrics.snapshot();
            assert_eq!(snapshot.gpu_meshes, 2);
            assert_eq!(snapshot.gpu_mesh_bytes, 384);
            assert_eq!(snapshot.peak_gpu_mesh_bytes, 384);
            drop(first);
            drop(second);
        }
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.gpu_meshes, 0);
        assert_eq!(snapshot.gpu_mesh_bytes, 0);
        assert_eq!(snapshot.peak_gpu_mesh_bytes, 384);
    }

    #[test]
    #[cfg(feature = "lifecycle_counts")]
    fn lifecycle_counts_restore_nested_phases_and_remain_per_player() {
        use super::lifecycle::{Count, Phase, add, scope};
        let first = RenderMetrics::default();
        let second = RenderMetrics::default();
        // No active player means no counter is retained.
        add(Count::ConstructVisit, 100);
        {
            let _player = enter(Some(&first));
            let _phase = scope(Phase::OrphanConstruct);
            add(Count::ConstructVisit, 1);
            {
                let _nested = scope(Phase::StageScripts);
                add(Count::ScriptsVisit, 2);
            }
            {
                let _other_player = enter(Some(&second));
                add(Count::ConstructVisit, 4);
            }
            add(Count::ConstructVisit, 8);
        }
        let first = first.snapshot().lifecycle;
        let second = second.snapshot().lifecycle;
        assert_eq!(
            first.values[Phase::OrphanConstruct as usize][Count::ConstructVisit as usize],
            9
        );
        assert_eq!(
            first.values[Phase::StageScripts as usize][Count::ScriptsVisit as usize],
            2
        );
        assert_eq!(
            second.values[Phase::OrphanConstruct as usize][Count::ConstructVisit as usize],
            4
        );
        assert_eq!(first.values.iter().flatten().sum::<u64>(), 11);
        assert_eq!(second.values.iter().flatten().sum::<u64>(), 4);
    }
}

#[cfg(test)]
mod gpu_bitmap_thread_tests {
    #[test]
    fn cache_eviction_updates_bitmap_metrics_from_the_device_owner() {
        let metrics = super::RenderMetrics::default();
        let _scope = super::enter(Some(&metrics));
        let allocation = super::GpuBitmap::new(64);
        assert_eq!(metrics.snapshot().gpu_bitmap_bytes, 64);
        std::thread::spawn(move || drop(allocation)).join().unwrap();
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.gpu_bitmap_bytes, 0);
        assert_eq!(snapshot.gpu_bitmaps, 0);
        assert_eq!(snapshot.peak_gpu_bitmap_bytes, 64);
    }
}
