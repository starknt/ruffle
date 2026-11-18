/// Errors that can occur during profiler creation.
#[cfg_attr(not(feature = "profile-with-tracy"), derive(PartialEq))]
#[derive(thiserror::Error, Debug)]
pub enum CreationError {
    #[error(transparent)]
    Resource(#[from] crate::gpu_runtime::RenderFailure),
    #[error(transparent)]
    InvalidSettings(#[from] SettingsError),

    #[cfg(feature = "profile-with-tracy")]
    #[error("Tracy client doesn't run yet.")]
    TracyClientNotRunning,

    #[cfg(feature = "profile-with-tracy")]
    #[error("Failed to create Tracy GPU context: {0}")]
    TracyGpuContextCreationError(#[from] tracy_client::GpuContextCreationError),

    #[cfg(feature = "profile-with-tracy")]
    #[error("Failed to read back initial GPU timestamp for Tracy: {0}")]
    TracyGpuTimestampReadback(#[from] wgpu::MapRangeError),
}

#[cfg(feature = "profile-with-tracy")]
impl PartialEq for CreationError {
    fn eq(&self, other: &Self) -> bool {
        match self {
            CreationError::Resource(left) => {
                matches!(other, CreationError::Resource(right) if left == right)
            }
            CreationError::InvalidSettings(left) => match other {
                CreationError::InvalidSettings(right) => left == right,
                _ => false,
            },
            CreationError::TracyClientNotRunning => {
                matches!(other, CreationError::TracyClientNotRunning)
            }
            CreationError::TracyGpuContextCreationError(left) => match left {
                tracy_client::GpuContextCreationError::TooManyContextsCreated => matches!(
                    other,
                    CreationError::TracyGpuContextCreationError(
                        tracy_client::GpuContextCreationError::TooManyContextsCreated
                    )
                ),
            },
            CreationError::TracyGpuTimestampReadback(left) => match other {
                CreationError::TracyGpuTimestampReadback(right) => {
                    left.to_string() == right.to_string()
                }
                _ => false,
            },
        }
    }
}

impl Eq for CreationError {}

/// Errors that can occur during settings validation and change.
#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum SettingsError {
    #[error("GpuProfilerSettings::max_num_pending_frames must be at least 1.")]
    InvalidMaxNumPendingFrames,
}

/// Errors that can occur during [`crate::gpu_profiler::GpuProfiler::end_frame`].
#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum EndFrameError {
    #[error(transparent)]
    Resource(#[from] crate::gpu_runtime::RenderFailure),
    #[error(
        "All profiling queries need to be closed before ending a frame. There were still {0} open queries."
    )]
    UnclosedQueries(u32),

    #[error(
        "Not all queries were resolved before ending a frame.\n
Call `GpuProfiler::resolve_queries` after all profiling queries have been closed and before ending the frame.\n
There were still {0} queries unresolved."
    )]
    UnresolvedQueries(u32),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_runtime::{RenderFailure, RenderFailureKind};

    #[test]
    fn frame_failure_preserves_the_resource_cause() {
        let cause = RenderFailure::new(RenderFailureKind::Timeout, "query drain");
        let reported: RenderFailure = EndFrameError::Resource(cause.clone()).into();
        assert_eq!(reported, cause);
    }
}

impl From<EndFrameError> for crate::gpu_runtime::RenderFailure {
    fn from(error: EndFrameError) -> Self {
        match error {
            EndFrameError::Resource(cause) => cause,
            error => Self::new(
                crate::gpu_runtime::RenderFailureKind::Internal,
                format!("GPU profiler: {error}"),
            ),
        }
    }
}
