//! Per-player instrumentation. A host scopes synchronous AVM/renderer work to its own
//! session. Allocation guards retain the owning counters across GC and delayed drops.
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
use web_time::Instant;

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
#[derive(Debug, Clone, Copy, Default)]
pub struct Timing {
    pub samples: u64,
    pub nanos: u64,
    pub max_nanos: u64,
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
    pub gpu_frame: Option<u64>,
    pub loader_bytes: [u64; 3],
    pub idle: [u64; 2],
    pub peak_idle: [u64; 2],
    pub peak_pool: u64,
    pub upload_bytes: u64,
    pub blend_batch_splits: u64,
    pub blend_batch_fallbacks: u64,
    pub bitmap_residency: Option<BitmapResidencyStats>,
    /// Cumulative publication totals, not currently live allocations or RSS.
    pub published_code: PublishedCodeStats,
    pub async_events: Vec<AsyncEvent>,
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
pub struct RenderMetrics(Rc<RefCell<Counters>>);
thread_local! {
    static CURRENT: RefCell<Option<RenderMetrics>> = const { RefCell::new(None) };
    static OPTIMIZED: Cell<bool> = const { Cell::new(false) };
    static ENGINE_OPTIMIZED: Cell<Option<bool>> = const { Cell::new(None) };
}
pub fn optimized() -> bool {
    OPTIMIZED.get()
}
pub fn engine_optimized() -> bool {
    ENGINE_OPTIMIZED.get().unwrap_or_else(optimized)
}

pub struct EngineScope(Option<bool>, std::marker::PhantomData<Rc<()>>);

pub fn enter_engine(enabled: bool) -> EngineScope {
    EngineScope(
        ENGINE_OPTIMIZED.replace(Some(enabled)),
        std::marker::PhantomData,
    )
}

impl Drop for EngineScope {
    fn drop(&mut self) {
        ENGINE_OPTIMIZED.set(self.0);
    }
}
pub fn enabled() -> bool {
    CURRENT.with_borrow(Option::is_some)
}
pub struct Scope(Option<RenderMetrics>, bool);
pub fn enter(metrics: Option<&RenderMetrics>, optimized: bool) -> Scope {
    Scope(
        CURRENT.replace(metrics.cloned()),
        OPTIMIZED.replace(optimized),
    )
}
impl Drop for Scope {
    fn drop(&mut self) {
        CURRENT.set(self.0.take());
        OPTIMIZED.set(self.1);
    }
}
impl RenderMetrics {
    pub fn snapshot(&self) -> Counters {
        self.0.borrow().clone()
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
    CURRENT.with_borrow(|m| {
        if let Some(m) = m {
            m.record(Phase::Gpu, nanos);
            m.0.borrow_mut().gpu_frame = frame;
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
pub fn uploaded(bytes: u64) {
    CURRENT.with_borrow(|m| {
        if let Some(m) = m {
            m.0.borrow_mut().upload_bytes += bytes;
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
pub struct GpuBitmap(Option<(RenderMetrics, u64)>);
impl GpuBitmap {
    pub fn new(bytes: u64) -> Self {
        Self(CURRENT.with_borrow(|m| {
            m.clone().map(|m| {
                {
                    let mut c = m.0.borrow_mut();
                    c.gpu_bitmaps += 1;
                    c.gpu_bitmap_bytes += bytes;
                    c.peak_gpu_bitmap_bytes = c.peak_gpu_bitmap_bytes.max(c.gpu_bitmap_bytes);
                }
                (m, bytes)
            })
        }))
    }
}
impl Drop for GpuBitmap {
    fn drop(&mut self) {
        if let Some((m, b)) = &self.0 {
            let mut c = m.0.borrow_mut();
            c.gpu_bitmaps -= 1;
            c.gpu_bitmap_bytes -= b;
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

#[cfg(test)]
mod tests {
    use super::{RenderMetrics, enabled, enter};

    #[test]
    fn profiling_is_enabled_only_inside_an_active_scope() {
        assert!(!enabled());
        let metrics = RenderMetrics::default();
        {
            let _scope = enter(Some(&metrics), false);
            assert!(enabled());
        }
        assert!(!enabled());
    }
}
