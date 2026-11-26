use crate::buffer_pool::PoolEntry;
use crate::gpu_resources::{Buffer, Device};
use crate::gpu_runtime::{RenderFailure, RenderFailureKind};
use crate::utils::{BufferDimensions, poll_failure};
use crate::{
    Descriptors, QueueSyncHandle, backend::ActiveFrame, render_snapshot::OperationDependency,
};
use crossbeam_channel::{Receiver, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

const MAP_TIMEOUT: Duration = Duration::from_secs(2);

pub(crate) enum ReadbackBuffer {
    Pooled(PoolEntry<Buffer, BufferDimensions>),
    Shared(Buffer),
}

struct MappedReadback(ReadbackBuffer);
type Completion = Receiver<Result<MappedReadback, RenderFailure>>;

impl ReadbackBuffer {
    fn buffer(&self) -> &Buffer {
        match self {
            Self::Pooled(lease) => lease,
            Self::Shared(buffer) => buffer,
        }
    }

    fn map(self, device: &Device, size: u64) -> Result<Completion, RenderFailure> {
        device.runtime().check_health()?;
        let (sender, completion) = crossbeam_channel::bounded(1);
        let raw = self.buffer().raw().clone();
        raw.slice(..size)
            .map_async(wgpu::MapMode::Read, move |result| {
                // Cancellation may race the driver before the callback is delivered.
                // Keep the pool lease here until the map has a definite outcome.
                let mapped = match result {
                    Ok(()) => Ok(MappedReadback(self)),
                    Err(error) => {
                        if let Self::Pooled(lease) = self {
                            drop(lease.into_owned());
                        }
                        Err(RenderFailure::new(
                            RenderFailureKind::MapFailed,
                            error.to_string(),
                        ))
                    }
                };
                let _ = sender.send(mapped);
            });
        device.runtime().request_poll(device)?;
        Ok(completion)
    }

    pub(crate) fn capture<R>(
        self,
        device: &Device,
        dimensions: &BufferDimensions,
        dependency: Option<wgpu::SubmissionIndex>,
        read: impl FnOnce(&[u8], u32) -> R,
    ) -> Result<R, RenderFailure> {
        let runtime = device.runtime();
        runtime.check_health()?;
        if self.buffer().lease().id().device_generation() != runtime.generation() {
            return Err(RenderFailure::new(
                RenderFailureKind::Validation,
                "readback buffer belongs to another device",
            ));
        }
        if let Some(index) = dependency {
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(index),
                    timeout: Some(MAP_TIMEOUT),
                })
                .map_err(poll_failure)?;
        }
        let completion = self.map(device, dimensions.size())?;
        let failed = runtime.health().failure_signal();
        let mapped = crossbeam_channel::select_biased! {
            recv(failed) -> _ => return Err(runtime.failure().expect("failed device signal")),
            recv(completion) -> result => result.map_err(|_| disconnected())??,
            default(MAP_TIMEOUT) => return Err(RenderFailure::new(RenderFailureKind::Timeout, "readback map callback timed out")),
        };
        runtime.check_health()?;
        mapped.read(dimensions, read)
    }
}

fn disconnected() -> RenderFailure {
    RenderFailure::new(
        RenderFailureKind::MapFailed,
        "buffer map callback disconnected",
    )
}

impl MappedReadback {
    fn read<R>(
        &self,
        dimensions: &BufferDimensions,
        read: impl FnOnce(&[u8], u32) -> R,
    ) -> Result<R, RenderFailure> {
        let pixels = self
            .0
            .buffer()
            .slice(..dimensions.size())
            .get_mapped_range()
            .map_err(|error| RenderFailure::new(RenderFailureKind::MapFailed, error.to_string()))?;
        Ok(read(&pixels, dimensions.padded_bytes_per_row))
    }
}

impl Drop for MappedReadback {
    fn drop(&mut self) {
        self.0.buffer().unmap();
    }
}

pub struct AsyncReadback {
    dependency: OperationDependency,
    completion: Result<Option<Completion>, RenderFailure>,
    dimensions: BufferDimensions,
    descriptors: Arc<Descriptors>,
    started: Instant,
}

impl AsyncReadback {
    pub(crate) fn new(
        buffer: PoolEntry<Buffer, BufferDimensions>,
        dimensions: BufferDimensions,
        descriptors: Arc<Descriptors>,
        dependency: OperationDependency,
    ) -> Result<Self, RenderFailure> {
        let completion =
            ReadbackBuffer::Pooled(buffer).map(&descriptors.device, dimensions.size())?;
        Ok(Self {
            dependency,
            completion: Ok(Some(completion)),
            dimensions,
            descriptors,
            started: Instant::now(),
        })
    }

    pub fn poll(&mut self) -> Result<Option<(Vec<u8>, u32)>, RenderFailure> {
        let Some(completion) = self.completion.as_ref().map_err(Clone::clone)? else {
            return Ok(None);
        };
        let result = self.poll_mapping(completion);
        match &result {
            Ok(Some(_)) => self.completion = Ok(None),
            Err(error) => self.completion = Err(error.clone()),
            Ok(None) => (),
        }
        result
    }

    fn poll_mapping(
        &self,
        completion: &Completion,
    ) -> Result<Option<(Vec<u8>, u32)>, RenderFailure> {
        let health = self.descriptors.runtime.health();
        health.check()?;
        self.dependency.check_open()?;
        self.descriptors
            .device
            .poll(wgpu::PollType::Poll)
            .map_err(|error| health.record(poll_failure(error)))?;
        let result = match completion.try_recv() {
            Ok(result) => result,
            Err(TryRecvError::Empty) if self.started.elapsed() < MAP_TIMEOUT => return Ok(None),
            Err(TryRecvError::Empty) => Err(RenderFailure::new(
                RenderFailureKind::Timeout,
                "asynchronous readback timed out",
            )),
            Err(TryRecvError::Disconnected) => Err(disconnected()),
        };
        result
            .and_then(|mapped| {
                mapped.read(&self.dimensions, |bytes, stride| {
                    Some((bytes.to_vec(), stride))
                })
            })
            .map_err(|error| health.record(error))
            .and_then(|data| {
                health.check()?;
                self.dependency.check_open()?;
                Ok(data)
            })
    }
}

pub(crate) enum ShaderReadback {
    Queued(QueueSyncHandle),
    Mapping(AsyncReadback),
    Failed(RenderFailure),
    Done,
}

impl ShaderReadback {
    pub(crate) fn poll(
        &mut self,
        frame: &mut ActiveFrame,
    ) -> Result<Option<(Vec<u8>, u32)>, RenderFailure> {
        let dependency = match self {
            Self::Queued(handle) => &handle.dependency,
            Self::Mapping(readback) => &readback.dependency,
            Self::Failed(error) => return Err(error.clone()),
            Self::Done => return Ok(None),
        };
        if let Err(error) = dependency.check(&frame.operations) {
            if error.kind() == RenderFailureKind::Canceled {
                *self = Self::Failed(error.clone());
            }
            return Err(error);
        }
        let result = match self {
            Self::Mapping(readback) => readback.poll(),
            _ => {
                let Self::Queued(handle) = std::mem::replace(self, Self::Done) else {
                    unreachable!()
                };
                handle.prepare_async(frame).map(|readback| {
                    *self = Self::Mapping(readback);
                    None
                })
            }
        };
        match &result {
            Ok(Some(_)) => *self = Self::Done,
            Err(error) => *self = Self::Failed(error.clone()),
            Ok(None) => (),
        }
        result
    }
}
