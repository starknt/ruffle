//! Native adaptation of wgpu-profiler 0.28.0 (https://github.com/Wumpf/wgpu-profiler).
//! The upstream licenses are retained alongside this module. Its scope/tree and
//! Tracy behavior are preserved, with managed fallible allocation, bounded maps,
//! and completion-based retirement through the renderer's device runtime.
mod errors;
mod profiler;
mod profiler_command_recorder;
mod profiler_query;
mod profiler_settings;
mod scope;
#[cfg(feature = "profile-with-tracy")]
mod tracy;
pub use errors::{CreationError, EndFrameError, SettingsError};
pub use profiler::GpuProfiler;
pub use profiler_command_recorder::ProfilerCommandRecorder;
pub use profiler_query::{GpuProfilerQuery, GpuTimerQueryResult};
pub use profiler_settings::GpuProfilerSettings;
pub use scope::{ManualOwningScope, OwningScope, Scope};
