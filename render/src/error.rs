use std::borrow::Cow;
use std::sync::{Arc, OnceLock};

use thiserror::Error;

use crate::bitmap::BitmapHandle;

/// Failures that stop native rendering instead of changing Flash's pixels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderFailureKind {
    Canceled,
    BudgetExceeded,
    OutOfMemory,
    DeviceLost,
    Validation,
    Internal,
    MapFailed,
    Timeout,
}

/// Cheaply cloned first-failure information shared with the host and VM.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("{kind:?}: {message}")]
pub struct RenderFailure {
    kind: RenderFailureKind,
    message: std::sync::Arc<str>,
}

impl RenderFailure {
    pub fn new(kind: RenderFailureKind, message: impl Into<std::sync::Arc<str>>) -> Self {
        let message = message.into();
        let message = if message.len() > 4096 {
            let mut end = 4096;
            while !message.is_char_boundary(end) {
                end -= 1;
            }
            Arc::from(&message[..end])
        } else {
            message
        };
        Self { kind, message }
    }

    pub fn kind(&self) -> RenderFailureKind {
        self.kind
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

/// Shared first-fatal latch. Callbacks only record diagnostics; the owner thread
/// observes it at execution and publication boundaries and stops the player.
#[derive(Clone, Debug, Default)]
pub struct RendererHealth(Arc<HealthState>);

#[derive(Debug)]
struct HealthState {
    failure: OnceLock<RenderFailure>,
    // Disconnecting the only sender broadcasts to every pending receiver.
    sender: std::sync::Mutex<Option<crossbeam_channel::Sender<std::convert::Infallible>>>,
    failed: crossbeam_channel::Receiver<std::convert::Infallible>,
}

impl Default for HealthState {
    fn default() -> Self {
        let (sender, failed) = crossbeam_channel::bounded(0);
        Self {
            failure: OnceLock::new(),
            sender: std::sync::Mutex::new(Some(sender)),
            failed,
        }
    }
}

impl RendererHealth {
    pub fn fail(&self, error: RenderFailure) -> bool {
        if self.0.failure.set(error).is_err() {
            return false;
        }
        self.0
            .sender
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        true
    }

    /// Select alongside a data or completion channel. Disconnection means the
    /// first error is already available, including to a late subscriber.
    pub fn failure_signal(&self) -> crossbeam_channel::Receiver<std::convert::Infallible> {
        self.0.failed.clone()
    }

    pub fn failure(&self) -> Option<RenderFailure> {
        self.0.failure.get().cloned()
    }

    #[inline]
    pub fn check(&self) -> Result<(), RenderFailure> {
        self.0
            .failure
            .get()
            .map_or(Ok(()), |error| Err(error.clone()))
    }

    pub fn record(&self, error: RenderFailure) -> RenderFailure {
        self.fail(error);
        // A failed set means the initiating cause was already recorded.
        self.0.failure.get().unwrap().clone()
    }
}

#[derive(Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Render(#[from] RenderFailure),
    #[error("Bitmap texture is larger than the rendering device supports")]
    TooLarge,

    #[error("Unknown bitmap format")]
    UnknownType,

    #[error("Invalid ZLIB compression: {0}")]
    InvalidZlibCompression(std::io::Error),

    #[error("Invalid JPEG")]
    InvalidJpeg(#[from] jpeg_decoder::Error),

    #[cfg(feature = "jpegxr")]
    #[error("Invalid JPEG-XR: {0}")]
    InvalidJpegXr(#[from] jpegxr::JXRError),

    #[cfg(feature = "jpegxr")]
    #[error("Error decoding TIFF from JPEG-XR: {0}")]
    MalformedTiffFromJpegXr(#[from] image::ImageError),

    #[error("Invalid PNG")]
    InvalidPng(#[from] png::DecodingError),

    #[error("Invalid GIF")]
    InvalidGif(#[from] gif::DecodingError),

    #[error("Empty GIF")]
    EmptyGif,

    #[error("Unknown handle {0:?}")]
    UnknownHandle(BitmapHandle),

    #[error("Not yet implemented: {0}")]
    Unimplemented(Cow<'static, str>),
}

impl Error {
    /// Errors in mandatory rendering/synchronization cannot become unsupported
    /// capability fallbacks. Preserve a backend's structured cause when present.
    pub fn into_render_failure(self) -> RenderFailure {
        match self {
            Self::Render(error) => error,
            other => RenderFailure::new(RenderFailureKind::Internal, other.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_failure_wakes_all_current_and_late_gpu_waiters() {
        let health = RendererHealth::default();
        let other_device = RendererHealth::default();
        let first = RenderFailure::new(RenderFailureKind::DeviceLost, "test removal");
        let waiters = (0..2).map(|_| health.failure_signal()).collect::<Vec<_>>();
        let scoped = std::sync::Barrier::new(3);
        std::thread::scope(|scope| {
            for waiter in waiters {
                let (scoped, health, first) = (&scoped, &health, &first);
                scope.spawn(move || {
                    scoped.wait();
                    assert!(matches!(
                        waiter.recv_timeout(std::time::Duration::from_secs(1)),
                        Err(crossbeam_channel::RecvTimeoutError::Disconnected)
                    ));
                    assert_eq!(health.check().unwrap_err(), *first);
                });
            }
            scoped.wait();
            assert!(health.fail(first.clone()));
            assert!(!health.fail(RenderFailure::new(RenderFailureKind::Timeout, "later")));
        });
        assert!(matches!(
            health.failure_signal().try_recv(),
            Err(crossbeam_channel::TryRecvError::Disconnected)
        ));
        assert!(matches!(
            other_device.failure_signal().try_recv(),
            Err(crossbeam_channel::TryRecvError::Empty)
        ));
    }

    #[test]
    fn renderer_health_preserves_first_failure_across_threads() {
        let health = RendererHealth::default();
        let callback = health.clone();
        std::thread::spawn(move || {
            callback.fail(RenderFailure::new(
                RenderFailureKind::DeviceLost,
                "lost device",
            ));
        })
        .join()
        .unwrap();
        health.fail(RenderFailure::new(
            RenderFailureKind::Validation,
            "invalid after loss",
        ));
        assert_eq!(
            health.check().unwrap_err().kind(),
            RenderFailureKind::DeviceLost
        );
        assert_eq!(health.failure().unwrap().message(), "lost device");
    }
}
