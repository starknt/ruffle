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
}
#[derive(Debug, Clone, Copy, Default)]
pub struct Timing {
    pub samples: u64,
    pub nanos: u64,
    pub max_nanos: u64,
}
#[derive(Debug, Default, Clone)]
pub struct Counters {
    pub timings: [Timing; 9],
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
}
#[derive(Clone, Default, Debug)]
pub struct RenderMetrics(Rc<RefCell<Counters>>);
thread_local! {
    static CURRENT: RefCell<Option<RenderMetrics>> = const { RefCell::new(None) };
    static OPTIMIZED: Cell<bool> = const { Cell::new(false) };
}
pub fn optimized() -> bool {
    OPTIMIZED.get()
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
pub fn uploaded(bytes: u64) {
    CURRENT.with_borrow(|m| {
        if let Some(m) = m {
            m.0.borrow_mut().upload_bytes += bytes;
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

pub fn loader_bytes(index: usize, bytes: usize) {
    CURRENT.with_borrow(|m| {
        if let Some(m) = m {
            m.0.borrow_mut().loader_bytes[index] += bytes as u64;
        }
    });
}
