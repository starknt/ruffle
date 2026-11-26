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

pub(crate) fn time_range_ns(
    ticks: std::ops::Range<u64>,
    period: f32,
) -> Option<std::ops::Range<u64>> {
    let ns = |value| {
        if period == 1.0 {
            value
        } else {
            (value as f64 * f64::from(period)).round() as u64
        }
    };
    let start = ns(ticks.start);
    let duration = ns(ticks.end.checked_sub(ticks.start)?);
    Some(start..start.checked_add(duration)?)
}
