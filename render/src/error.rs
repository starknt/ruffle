use std::borrow::Cow;

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
#[derive(Clone, Debug, Error)]
#[error("{kind:?}: {message}")]
pub struct RenderFailure {
    kind: RenderFailureKind,
    message: std::sync::Arc<str>,
}

impl RenderFailure {
    pub fn new(kind: RenderFailureKind, message: impl Into<std::sync::Arc<str>>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn kind(&self) -> RenderFailureKind {
        self.kind
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

#[derive(Error, Debug)]
pub enum Error {
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
