use super::{RenderFailure, RenderFailureKind, RendererHealth};
use crossbeam_channel::{Receiver, Sender, TrySendError};
use std::sync::OnceLock;
use std::thread::JoinHandle;
use std::time::Duration;

/// Held by native device owners and renderer descriptors. GPU completion
/// callbacks may retain the runtime and resources, but do not own this thread.
pub struct GpuProgress {
    device: wgpu::Device,
    health: RendererHealth,
    worker: OnceLock<Result<Worker, RenderFailure>>,
}

struct Worker {
    wake: Sender<()>,
    thread: JoinHandle<()>,
}

impl GpuProgress {
    pub(super) fn new(device: wgpu::Device, health: RendererHealth) -> Self {
        Self {
            device,
            health,
            worker: Default::default(),
        }
    }

    pub(crate) fn request(&self) -> Result<(), RenderFailure> {
        self.health.check()?;
        let worker = self
            .worker
            .get_or_init(|| {
                let (wake, work) = crossbeam_channel::bounded(1);
                let device = self.device.clone();
                let health = self.health.clone();
                let thread = std::thread::Builder::new()
                    .name("Ruffle GPU progress".into())
                    .spawn(move || poll_registered_work(&device, &health, work))
                    .map_err(|error| {
                        self.health.record(RenderFailure::new(
                            RenderFailureKind::Internal,
                            error.to_string(),
                        ))
                    })?;
                Ok(Worker { wake, thread })
            })
            .as_ref()
            .map_err(Clone::clone)?;
        match worker.wake.try_send(()) {
            Ok(()) | Err(TrySendError::Full(())) => Ok(()),
            Err(TrySendError::Disconnected(())) => Err(self.health.record(RenderFailure::new(
                RenderFailureKind::Internal,
                "GPU progress worker exited",
            ))),
        }
    }
}

impl Drop for GpuProgress {
    fn drop(&mut self) {
        if let Some(Ok(Worker { wake, thread })) = self.worker.take() {
            drop(wake);
            if thread.thread().id() != std::thread::current().id() {
                // Keep our native device alive until its final callbacks and
                // the worker's device handle have both been released.
                let _ = thread.join();
            }
        }
    }
}

fn poll_registered_work(device: &wgpu::Device, health: &RendererHealth, work: Receiver<()>) {
    let failed = health.failure_signal();
    let mut busy = false;
    loop {
        let kick = if busy {
            crossbeam_channel::select! {
                recv(failed) -> _ => return,
                recv(work) -> result => result,
                default(Duration::from_millis(1)) => Ok(()),
            }
        } else {
            crossbeam_channel::select! {
                recv(failed) -> _ => return,
                recv(work) -> result => result,
            }
        };
        if kick.is_err() {
            if busy
                && health.check().is_ok()
                && let Err(error) = device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_secs(2)),
                })
            {
                health.record(crate::utils::poll_failure(error));
            }
            return;
        }
        match device.poll(wgpu::PollType::Poll) {
            Ok(status) => busy = !status.is_queue_empty(),
            Err(error) => {
                health.record(crate::utils::poll_failure(error));
                return;
            }
        }
    }
}
