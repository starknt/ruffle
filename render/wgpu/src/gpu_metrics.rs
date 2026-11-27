//! Timestamp identities and inclusive named phases, independent of CPU timers.
use crate::gpu_profiler::GpuTimerQueryResult;
use ruffle_render::profile::GpuSample;

pub fn sample_queries(queries: &[GpuTimerQueryResult]) -> Option<GpuSample> {
    let mut frame = None;
    let mut stage_end: Option<u64> = None;
    let mut phases = std::collections::BTreeMap::<String, u64>::new();
    fn visit(
        queries: &[GpuTimerQueryResult],
        phases: &mut std::collections::BTreeMap<String, u64>,
        ranges: &mut Vec<(u64, u64)>,
        ancestor_timed: bool,
    ) -> Option<()> {
        for query in queries {
            if let Some(time) = &query.time_ns {
                let duration = time.end.checked_sub(time.start)?;
                if !ancestor_timed {
                    ranges.push((time.start, time.end));
                }
                let label = if query.label.starts_with("Stage ") {
                    "Stage"
                } else {
                    &query.label
                };
                let total = phases.entry(label.to_owned()).or_default();
                *total = total.saturating_add(duration);
            }
            visit(
                &query.nested_queries,
                phases,
                ranges,
                ancestor_timed || query.time_ns.is_some(),
            )?;
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
            if let Some(time) = &query.time_ns {
                stage_end = Some(stage_end.map_or(time.end, |end| end.max(time.end)));
            }
        }
    }
    let mut ranges_ns = Vec::new();
    visit(queries, &mut phases, &mut ranges_ns, false)?;
    let first = ranges_ns.iter().map(|r| r.0).min()?;
    let last = ranges_ns.iter().map(|r| r.1).max()?;
    Some(GpuSample {
        frame,
        nanos: last - first,
        stage_end_ns: stage_end,
        ranges_ns,
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
            time_ns: (start.is_finite() && end.is_finite())
                .then(|| (start * 1e9).round() as u64..(end * 1e9).round() as u64),
            nested_queries: vec![],
        }
    }
    #[test]
    fn native_gpu_metrics_preserve_frame_identity_and_inclusive_phase_times() {
        let mut stage = query("Stage 42", 2., 6.);
        stage.nested_queries.push(query("Shape", 3., 5.));
        let sample = sample_queries(&[query("Draw to CAB", 0., 1.), stage]).unwrap();
        assert_eq!(sample.frame, Some(42));
        assert_eq!(sample.nanos, 6_000_000_000);
        assert_eq!(
            sample.ranges_ns,
            [(0, 1_000_000_000), (2_000_000_000, 6_000_000_000)]
        );
        assert_eq!(sample.stage_end_ns, Some(6_000_000_000));
        assert!(sample.phases.contains(&("Shape".into(), 2_000_000_000)));
        assert!(sample.phases.contains(&("Stage".into(), 4_000_000_000)));
        assert!(sample_queries(&[query("Stage 42", f64::NAN, 3.)]).is_none());
        assert!(sample_queries(&[query("Stage 42", 1., 2.), query("Stage 43", 2., 3.)]).is_none());
        let epoch = 1_795_624_717_765_783_297;
        let mut submicrosecond = query("Stage 43", 0., 0.);
        submicrosecond.time_ns = Some(epoch..epoch + 37);
        let sample = sample_queries(&[submicrosecond]).unwrap();
        assert_eq!(sample.ranges_ns, [(epoch, epoch + 37)]);
        assert_eq!(sample.nanos, 37);
    }
}
