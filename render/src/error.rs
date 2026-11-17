use std::borrow::Cow;
use std::sync::{Arc, OnceLock};

use thiserror::Error;

use crate::bitmap::BitmapHandle;

/// Failures that stop native rendering instead of changing Flash's pixels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderFailureKind {
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
pub struct RendererHealth(Arc<OnceLock<RenderFailure>>);

impl RendererHealth {
    pub fn fail(&self, error: RenderFailure) -> bool {
        self.0.set(error).is_ok()
    }

    pub fn failure(&self) -> Option<RenderFailure> {
        self.0.get().cloned()
    }

    pub fn check(&self) -> Result<(), RenderFailure> {
        self.0.get().map_or(Ok(()), |error| Err(error.clone()))
    }

    pub fn record(&self, error: RenderFailure) -> RenderFailure {
        self.fail(error);
        // A failed set means the initiating cause was already recorded.
        self.0.get().unwrap().clone()
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

    #[cfg(feature = "web")]
    #[error("Javascript error")]
    JavascriptError(wasm_bindgen::JsValue),

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
