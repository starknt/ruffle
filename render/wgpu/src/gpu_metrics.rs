//! Timestamp identities and inclusive named phases, independent of CPU timers.
use crate::gpu_profiler::GpuTimerQueryResult;
use ruffle_render::profile::GpuSample;

pub fn sample_queries(queries: &[GpuTimerQueryResult]) -> Option<GpuSample> {
    let mut frame = None;
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut phases = std::collections::BTreeMap::<String, u64>::new();
    fn visit(
        queries: &[GpuTimerQueryResult],
        phases: &mut std::collections::BTreeMap<String, u64>,
    ) -> Option<()> {
        for query in queries {
            if let Some(time) = &query.time {
                if !time.start.is_finite() || !time.end.is_finite() || time.end < time.start {
                    return None;
                }
                let label = if query.label.starts_with("Stage ") {
                    "Stage"
                } else {
                    &query.label
                };
                let duration = ((time.end - time.start) * 1e9) as u64;
                let total = phases.entry(label.to_owned()).or_default();
                *total = total.saturating_add(duration);
            }
            visit(&query.nested_queries, phases)?;
        }
        Some(())
    }
    for query in queries {
        if let Some(id) = query
            .label
            .strip_prefix("Stage ")
            .and_then(|id| id.parse::<u64>().ok())
        {
            if frame.is_some_and(|previous| previous != id) {
                return None;
            }
            frame = Some(id);
        }
        if let Some(time) = &query.time {
            if !time.start.is_finite() || !time.end.is_finite() || time.end < time.start {
                return None;
            }
            lo = lo.min(time.start);
            hi = hi.max(time.end);
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        return None;
    }
    visit(queries, &mut phases)?;
    Some(GpuSample {
        frame,
        nanos: ((hi - lo) * 1e9) as u64,
        phases: phases.into_iter().collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn query(label: &str, start: f64, end: f64) -> GpuTimerQueryResult {
        GpuTimerQueryResult {
            label: label.into(),
            pid: 1,
            tid: std::thread::current().id(),
            time: Some(start..end),
            nested_queries: vec![],
        }
    }
    #[test]
    fn native_gpu_metrics_preserve_frame_identity_and_inclusive_phase_times() {
        let mut stage = query("Stage 42", 2., 6.);
        stage.nested_queries.push(query("Shape", 3., 5.));
        let sample = sample_queries(&[query("Draw to CAB", 1., 2.), stage]).unwrap();
        assert_eq!(sample.frame, Some(42));
        assert_eq!(sample.nanos, 5_000_000_000);
        assert!(sample.phases.contains(&("Shape".into(), 2_000_000_000)));
        assert!(sample.phases.contains(&("Stage".into(), 4_000_000_000)));
        assert!(sample_queries(&[query("Stage 42", f64::NAN, 3.)]).is_none());
        assert!(sample_queries(&[query("Stage 42", 1., 2.), query("Stage 43", 2., 3.)]).is_none());
    }
}
