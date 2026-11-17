//! Three submission slots of bounded, reclaimable upload pages. A page can be
//! reused only after its actual last submission completes, never by frame age.
use crate::gpu_resources::{Buffer, CommandEncoder, Device};
use crate::gpu_runtime::{GpuRuntime, RenderFailure, RenderFailureKind, ResourceKind};
use crate::resource_scheduler::{CacheTicket, Reclaim, ReclaimClass};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

#[derive(Debug)]
struct ActivePage {
    buffer: Buffer,
    used: u64,
    mapped: bool,
    after_serial: u64,
}
#[derive(Debug)]
struct RetiredPage {
    buffer: Buffer,
    age: u64,
    after_serial: u64,
    _ticket: CacheTicket,
}
#[derive(Debug)]
struct Pages {
    slots: [Vec<RetiredPage>; 3],
    runtime: GpuRuntime,
}
impl Pages {
    fn completed(&self, page: &RetiredPage) -> bool {
        let serial = page.buffer.lease().last_submission();
        serial > page.after_serial
            && serial <= self.runtime.stats().completed
            && page.buffer.lease().unsubmitted_uses() == 0
    }
    fn oldest_index(&self) -> Option<(usize, usize)> {
        self.slots
            .iter()
            .enumerate()
            .flat_map(|(slot, pages)| {
                pages
                    .iter()
                    .enumerate()
                    .map(move |(index, page)| (slot, index, page))
            })
            .filter(|(_, _, page)| self.completed(page))
            .min_by_key(|(_, _, page)| page.age)
            .map(|(slot, index, _)| (slot, index))
    }
}
impl Reclaim for Mutex<Pages> {
    fn oldest(&self) -> Option<u64> {
        let pages = self
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        pages
            .oldest_index()
            .map(|(slot, index)| pages.slots[slot][index].age)
    }
    fn evict_oldest(&self) -> bool {
        let page = {
            let mut pages = self
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            pages
                .oldest_index()
                .map(|(slot, index)| pages.slots[slot].swap_remove(index))
        };
        page.is_some()
    }
}
pub(crate) trait SealUploads: Send + Sync {
    fn seal(&self) -> Result<(), RenderFailure>;
}
#[derive(Debug)]
struct ActiveUploads {
    buffers: Mutex<Vec<ActivePage>>,
    views: AtomicUsize,
    slot: AtomicUsize,
    pages: Arc<Mutex<Pages>>,
    runtime: GpuRuntime,
}
impl SealUploads for ActiveUploads {
    fn seal(&self) -> Result<(), RenderFailure> {
        let active = {
            let mut buffers = self
                .buffers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if self.views.load(Ordering::Acquire) != 0 {
                return Err(RenderFailure::new(
                    RenderFailureKind::Validation,
                    "upload view still held at submission boundary",
                ));
            }
            std::mem::take(&mut *buffers)
        };
        for page in active {
            if page.mapped {
                page.buffer.unmap();
            }
            if let Some(ticket) = self
                .runtime
                .scheduler()
                .admit(ReclaimClass::Idle, page.buffer.size())
            {
                page.buffer.lease().set_idle(true);
                self.pages
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .slots[self.slot.load(Ordering::Relaxed)]
                .push(RetiredPage {
                    after_serial: page.after_serial,
                    buffer: page.buffer,
                    age: self.runtime.scheduler().tick(),
                    _ticket: ticket,
                });
            }
        }
        Ok(())
    }
}
pub struct UploadViewMut {
    view: Option<wgpu::BufferViewMut>,
    owner: Arc<ActiveUploads>,
}
impl std::ops::Deref for UploadViewMut {
    type Target = wgpu::BufferViewMut;
    fn deref(&self) -> &Self::Target {
        self.view.as_ref().expect("live upload view")
    }
}
impl std::ops::DerefMut for UploadViewMut {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.view.as_mut().expect("live upload view")
    }
}
impl Drop for UploadViewMut {
    fn drop(&mut self) {
        drop(self.view.take());
        self.owner.views.fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Debug)]
pub struct UploadAllocation {
    pub buffer: Buffer,
    pub offset: u64,
    pub size: u64,
}
#[derive(Debug)]
pub struct FrameUploadArena {
    device: Device,
    page_size: u64,
    active: Arc<ActiveUploads>,
    pages: Arc<Mutex<Pages>>,
}
impl FrameUploadArena {
    pub fn new(device: Device, page_size: u64) -> Self {
        let runtime = device.runtime().clone();
        let limit = (runtime.config().staging_limit / 3)
            .min(runtime.config().hard_limit / 3)
            .min(device.limits().max_buffer_size)
            .max(8);
        let page_size = page_size.max(8).min(limit) / 8 * 8;
        let pages = Arc::new(Mutex::new(Pages {
            slots: Default::default(),
            runtime: runtime.clone(),
        }));
        runtime
            .scheduler()
            .register(ReclaimClass::Idle, pages.clone());
        let active = Arc::new(ActiveUploads {
            buffers: Default::default(),
            views: Default::default(),
            slot: Default::default(),
            pages: pages.clone(),
            runtime,
        });
        device.register_upload_sealer(active.clone());
        Self {
            device,
            page_size,
            active,
            pages,
        }
    }
    pub fn page_size(&self) -> u64 {
        self.page_size
    }
    fn reserve(
        &mut self,
        size: u64,
        binding_size: u64,
        alignment: u64,
        usage: wgpu::BufferUsages,
        mapped: bool,
    ) -> Result<UploadAllocation, RenderFailure> {
        self.device.runtime().check_health()?;
        // Dynamic pages can be submitted by any backend synchronization barrier.
        // Observe their actual serial before appending another logical frame.
        if self
            .active
            .buffers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .any(|page| page.buffer.lease().last_submission() > page.after_serial)
        {
            self.finish();
            self.recall();
        }
        if size == 0 || binding_size < size || alignment == 0 || !alignment.is_power_of_two() {
            return Err(RenderFailure::new(
                RenderFailureKind::Validation,
                "invalid upload slice alignment/length",
            ));
        }
        let aligned = |used: u64| {
            used.checked_add(alignment - 1)
                .map(|value| value & !(alignment - 1))
        };
        for page in self
            .active
            .buffers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter_mut()
            .rev()
            .filter(|page| page.buffer.usage() == usage)
        {
            if let Some(offset) = aligned(page.used)
                && offset
                    .checked_add(binding_size)
                    .is_some_and(|end| end <= page.buffer.size())
            {
                page.used = offset + size;
                return Ok(UploadAllocation {
                    buffer: page.buffer.clone(),
                    offset,
                    size,
                });
            }
        }
        let capacity = self
            .page_size
            .max(binding_size)
            .checked_add(7)
            .map(|v| v / 8 * 8)
            .ok_or_else(|| {
                RenderFailure::new(
                    RenderFailureKind::BudgetExceeded,
                    "upload page size overflow",
                )
            })?;
        let reused = {
            let mut pages = self
                .pages
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let found = (0..3).find_map(|distance| {
                let slot = (self.active.slot.load(Ordering::Relaxed) + distance) % 3;
                pages.slots[slot]
                    .iter()
                    .position(|page| {
                        page.buffer.usage() == usage
                            && page.buffer.size() >= capacity
                            && pages.completed(page)
                    })
                    .map(|index| (slot, index))
            });
            found.map(|(slot, index)| pages.slots[slot].swap_remove(index))
        };
        let buffer = if let Some(page) = reused {
            let buffer = page.buffer;
            buffer.lease().set_idle(false);
            if mapped {
                self.map_page(&buffer)?;
            }
            buffer
        } else {
            self.device.create_buffer_with_kind(
                &wgpu::BufferDescriptor {
                    label: Some("Frame upload page"),
                    size: capacity,
                    usage,
                    mapped_at_creation: mapped,
                },
                if mapped {
                    ResourceKind::StagingBuffer
                } else {
                    ResourceKind::DynamicBuffer
                },
            )?
        };
        self.active
            .buffers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(ActivePage {
                after_serial: buffer.lease().last_submission(),
                buffer: buffer.clone(),
                used: size,
                mapped,
            });
        Ok(UploadAllocation {
            buffer,
            offset: 0,
            size,
        })
    }
    fn map_page(&self, buffer: &Buffer) -> Result<(), RenderFailure> {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        buffer.map_async(wgpu::MapMode::Write, .., move |result| {
            let _ = tx.send(result);
        });
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            self.device.runtime().check_health()?;
            match rx.try_recv() {
                Ok(result) => {
                    return result.map_err(|error| {
                        RenderFailure::new(RenderFailureKind::MapFailed, error.to_string())
                    });
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Err(RenderFailure::new(
                        RenderFailureKind::MapFailed,
                        "upload map callback dropped",
                    ));
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
            if Instant::now() >= deadline {
                buffer.unmap();
                return Err(RenderFailure::new(
                    RenderFailureKind::Timeout,
                    "upload page map timed out",
                ));
            }
            self.device
                .poll(wgpu::PollType::Poll)
                .map_err(crate::utils::poll_failure)?;
            std::thread::yield_now();
        }
    }
    pub fn allocate_dynamic(
        &mut self,
        size: u64,
        alignment: u64,
    ) -> Result<UploadAllocation, RenderFailure> {
        self.reserve(
            size,
            size,
            alignment,
            wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::INDEX,
            false,
        )
    }
    /// The shader may bind a fixed array but only reads the uploaded instances.
    /// Its unused tail can overlap later allocations; the used bytes never do.
    pub fn allocate_uniform(
        &mut self,
        used: u64,
        binding_size: u64,
        alignment: u64,
    ) -> Result<UploadAllocation, RenderFailure> {
        self.reserve(
            used.max(4),
            binding_size,
            alignment,
            wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::INDEX,
            false,
        )
    }

    pub fn write_buffer(
        &mut self,
        encoder: &mut CommandEncoder,
        target: &wgpu::Buffer,
        offset: u64,
        size: wgpu::BufferSize,
    ) -> Result<UploadViewMut, RenderFailure> {
        if !offset.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)
            || !size.get().is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)
            || offset
                .checked_add(size.get())
                .is_none_or(|end| end > target.size())
        {
            return Err(RenderFailure::new(
                RenderFailureKind::Validation,
                "invalid staged buffer copy",
            ));
        }
        let allocation = self.reserve(
            size.get(),
            size.get(),
            wgpu::MAP_ALIGNMENT,
            wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            true,
        )?;
        let guard = self
            .active
            .buffers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let view = allocation
            .buffer
            .get_mapped_range_mut(allocation.offset..allocation.offset + allocation.size)
            .map_err(|error| RenderFailure::new(RenderFailureKind::MapFailed, error.to_string()))?;
        self.active.views.fetch_add(1, Ordering::AcqRel);
        drop(guard);
        let view = UploadViewMut {
            view: Some(view),
            owner: self.active.clone(),
        };
        encoder.copy_buffer_to_buffer(
            &allocation.buffer,
            allocation.offset,
            target,
            offset,
            allocation.size,
        );
        Ok(view)
    }
    pub fn finish(&mut self) {
        if let Err(error) = self.active.seal() {
            self.device.runtime().fail(error);
        }
    }
    pub fn recall(&mut self) {
        let slot = self.active.slot.load(Ordering::Relaxed);
        self.active.slot.store((slot + 1) % 3, Ordering::Relaxed);
    }
}
