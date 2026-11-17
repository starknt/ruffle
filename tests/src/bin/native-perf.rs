//! Offline native frame service benchmark. Captures and trace comparison belong
//! to the regression runner; this runner measures GPU-completed work, not FPS.
#![recursion_limit = "256"]
use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use ruffle_render::profile::{self, NativePhase, RenderMetrics};
use ruffle_render_wgpu::backend::{WgpuRenderBackend, request_adapter_and_device};
use ruffle_render_wgpu::{descriptors::Descriptors, target::TextureTarget, wgpu};
use ruffle_test_framework::environment::{
    CompileMode, Environment, RenderBackend, RenderInterface,
};
use ruffle_test_framework::options::{RenderOptions, TestOptions};
use ruffle_test_framework::runner::TestStatus;
use ruffle_test_framework::test::Test;
use ruffle_test_framework::vfs::{PhysicalFS, VfsPath};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::any::Any;
use std::collections::{HashSet, VecDeque};
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

// Keep the measurement writer allocation-light on both frozen builds. These
// records retain the existing schema, real counters and measured interval.
struct TimingDeltas<'a> {
    names: &'static [&'static str],
    current: &'a [profile::Timing],
    previous: &'a [profile::Timing],
}
impl serde::Serialize for TimingDeltas<'_> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(self.names.len()))?;
        for (i, name) in self.names.iter().enumerate() {
            map.serialize_entry(
                name,
                &self.current[i].nanos.saturating_sub(self.previous[i].nanos),
            )?;
        }
        map.end()
    }
}
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct FrameRecord<'a, Resources: serde::Serialize> {
    event: &'static str,
    frame_id: u64,
    cpu_ns: u64,
    phase_ns: TimingDeltas<'a>,
    native_ns: TimingDeltas<'a>,
    work: profile::RenderWork,
    rss_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    native_resources: Option<&'a Resources>,
    allocator: Option<serde_json::Value>,
}
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct GpuRecord<'a> {
    event: &'static str,
    frame_id: Option<u64>,
    ns: u64,
    phase_ns: &'a [(String, u64)],
}
fn write_record(writer: &mut impl Write, record: &impl serde::Serialize) -> Result<()> {
    serde_json::to_writer(&mut *writer, record)?;
    writer.write_all(b"\n")?;
    Ok(())
}
fn write_gpu_record(writer: &mut impl Write, sample: &profile::GpuSample) -> Result<()> {
    write_record(
        writer,
        &GpuRecord {
            event: "gpu",
            frame_id: sample.frame,
            ns: sample.nanos,
            phase_ns: &sample.phases,
        },
    )
}

#[derive(Parser)]
struct Opt {
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long)]
    case: String,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 120)]
    warmup: u32,
    #[arg(long, default_value_t = 1200)]
    frames: u32,
    #[arg(long, default_value = "vulkan")]
    backend: String,
    #[arg(long)]
    compile: bool,
    /// Pace a long-lived correctness/memory soak; never use for throughput comparisons.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=1000))]
    pace_fps: Option<u32>,
    /// Request full GC work at existing frame boundaries, drain, and purge caches.
    /// Memory-soak checkpoints are deliberately unavailable in throughput runs.
    #[arg(long, requires = "pace_fps", value_parser = clap::value_parser!(u32).range(1..))]
    reclaim_every: Option<u32>,
    /// Measure instrumentation-disabled overhead. Optimizations stay enabled.
    #[arg(long)]
    profile_off: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    case: Vec<Case>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    path: PathBuf,
}
impl Manifest {
    fn parse(text: &str) -> Result<Self> {
        let manifest: Self = toml::from_str(text)?;
        ensure!(!manifest.case.is_empty(), "empty workload manifest");
        let mut names = HashSet::new();
        for case in &manifest.case {
            ensure!(
                !case.name.is_empty() && names.insert(&case.name),
                "empty or duplicate case name"
            );
            ensure!(!case.path.as_os_str().is_empty(), "empty workload path");
        }
        Ok(manifest)
    }
}
fn validate_frames(warmup: u32, frames: u32) -> Result<u32> {
    ensure!(frames > 0, "frames must be positive");
    warmup
        .checked_add(frames)
        .and_then(|n| n.checked_add(2))
        .context("frame count overflow")
}

struct PerfEnvironment {
    descriptors: Arc<Descriptors>,
    compile: bool,
    profiling: bool,
}
impl Environment for PerfEnvironment {
    fn is_render_supported(&self, _: &RenderOptions) -> bool {
        true
    }
    fn compile_mode(&self) -> CompileMode {
        if self.compile {
            CompileMode::CompileSilently
        } else {
            CompileMode::UsePrecompiled
        }
    }
    fn create_renderer(
        &self,
        width: u32,
        height: u32,
    ) -> Option<(Box<dyn RenderInterface>, Box<dyn RenderBackend>)> {
        let target =
            TextureTarget::new_without_readback(&self.descriptors.device, (width, height)).ok()?;
        let renderer =
            WgpuRenderBackend::with_options(self.descriptors.clone(), target, self.profiling)
                .ok()?;
        Some((Box::new(NoCaptures), Box::new(renderer)))
    }
}
struct NoCaptures;
impl RenderInterface for NoCaptures {
    fn name(&self) -> String {
        "native-perf".into()
    }
    fn capture(&self, _: &mut dyn RenderBackend) -> image::RgbaImage {
        panic!("performance workloads must not request image captures")
    }
}

fn wait(descriptors: &Descriptors, submission: Option<wgpu::SubmissionIndex>) -> Result<()> {
    let _timer = profile::native_time(NativePhase::QueueThrottle);
    descriptors.runtime.check_health()?;
    descriptors
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: submission,
            timeout: Some(Duration::from_secs(30)),
        })
        .context("GPU completion timed out")?;
    descriptors.runtime.check_health()?;
    Ok(())
}

fn resident_memory() -> Option<(u64, u64)> {
    // Linux supplies KiB regardless of the machine's page size. Other native
    // systems explicitly report missing RSS rather than an invented zero.
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let field = |prefix| {
        status
            .lines()
            .find_map(|line| line.strip_prefix(prefix))?
            .split_whitespace()
            .next()?
            .parse::<u64>()
            .ok()
            .map(|kib| kib * 1024)
    };
    Some((field("VmRSS:")?, field("VmHWM:")?))
}

fn workload_digest(path: &std::path::Path) -> Result<String> {
    fn visit(root: &std::path::Path, path: &std::path::Path, hash: &mut Sha256) -> Result<()> {
        let mut entries = fs::read_dir(path)?.collect::<std::io::Result<Vec<_>>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            if entry.file_type()?.is_dir() {
                visit(root, &path, hash)?;
            } else if matches!(
                path.extension().and_then(|s| s.to_str()),
                Some(
                    "swf"
                        | "toml"
                        | "json"
                        | "jsonl"
                        | "pbj"
                        | "png"
                        | "jpg"
                        | "jpeg"
                        | "mp3"
                        | "flv"
                        | "txt"
                )
            ) {
                let name = path.strip_prefix(root)?.to_string_lossy();
                let bytes = fs::read(&path)?;
                hash.update((name.len() as u64).to_le_bytes());
                hash.update(name.as_bytes());
                hash.update((bytes.len() as u64).to_le_bytes());
                hash.update(bytes);
            }
        }
        Ok(())
    }
    let mut hash = Sha256::new();
    visit(path, path, &mut hash)?;
    Ok(format!("{:x}", hash.finalize()))
}
fn allocator_stats(device: &wgpu::Device) -> serde_json::Value {
    device.generate_allocator_report().map_or(serde_json::Value::Null, |report|
        json!({"allocatedBytes":report.total_allocated_bytes,"reservedBytes":report.total_reserved_bytes}))
}

fn main() -> Result<()> {
    let opt = Opt::parse();
    let total = validate_frames(opt.warmup, opt.frames)?;
    let manifest = Manifest::parse(&fs::read_to_string(&opt.manifest)?)?;
    let case = manifest
        .case
        .iter()
        .find(|case| case.name == opt.case)
        .context("unknown workload")?;
    let backends = match opt.backend.as_str() {
        "vulkan" => wgpu::Backends::VULKAN,
        "metal" => wgpu::Backends::METAL,
        "dx12" => wgpu::Backends::DX12,
        "gl" => wgpu::Backends::GL,
        _ => bail!("backend must be vulkan, metal, dx12 or gl"),
    };
    let setup = Instant::now();
    let instance = ruffle_render_wgpu::backend::create_wgpu_instance(
        backends,
        wgpu::BackendOptions::default(),
        None,
    );
    let (adapter, device, queue) = futures::executor::block_on(request_adapter_and_device(
        backends,
        &instance,
        None,
        wgpu::PowerPreference::HighPerformance,
    ))
    .map_err(|e| anyhow::anyhow!("adapter/device: {e}"))?;
    let info = adapter.get_info();
    let environment = PerfEnvironment {
        descriptors: Arc::new(Descriptors::new(instance, adapter, device, queue)?),
        compile: opt.compile,
        profiling: !opt.profile_off,
    };
    let path = opt
        .manifest
        .parent()
        .context("manifest parent")?
        .join(&case.path)
        .canonicalize()?;
    let root: VfsPath = PhysicalFS::new(&path).into();
    let mut options = TestOptions::read(&root.join("test.toml")?)?;
    options.num_frames = Some(total);
    options.num_ticks = None;
    options.sleep_to_meet_frame_rate = false;
    options.image_comparisons.clear();
    options.audio_assertions.clear();
    let test = Test::from_options(options, root, case.name.clone())?;
    let metrics = RenderMetrics::default();
    let _scope = profile::enter((!opt.profile_off).then_some(&metrics));
    let mut runner = test.create_test_runner(&environment)?;
    while !runner.is_preloaded() {
        ensure!(setup.elapsed() < Duration::from_secs(60), "preload timeout");
        ensure!(
            !matches!(runner.tick()?, TestStatus::Finished),
            "workload ended during preload"
        );
    }
    ensure!(
        <dyn Any>::is::<WgpuRenderBackend<TextureTarget>>(
            runner.player().lock().unwrap().renderer()
        ),
        "workload must enable with_renderer; renderer creation must succeed"
    );
    let mut output = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&opt.output)?,
    );
    let settings = {
        let player = runner.player().lock().unwrap();
        let renderer =
            <dyn Any>::downcast_ref::<WgpuRenderBackend<TextureTarget>>(player.renderer()).unwrap();
        use ruffle_render::backend::RenderBackend as _;
        let viewport = renderer.viewport_dimensions();
        json!({"quality":renderer.quality().to_string(), "samples":renderer.sample_count(),
            "viewport":[viewport.width,viewport.height], "scaleFactor":viewport.scale_factor,
            "readback":false,"inFlightLimit":2,"debugAssertions":cfg!(debug_assertions),"avmDebug":cfg!(feature="avm_debug"),
            "timestampPeriod":environment.descriptors.queue.get_timestamp_period(),
            "deviceFeatures":format!("{:?}", environment.descriptors.device.features())})
    };
    let run_id = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    );
    writeln!(
        output,
        "{}",
        json!({"event":"session", "schemaVersion":2, "case":case.name,
        "backend":format!("{:?}", info.backend), "device":info.name, "driver":info.driver_info,
        "warmup":opt.warmup, "frames":opt.frames, "inFlightLimit":2, "readback":false, "paceFps":opt.pace_fps,
        "profilingEnabled":!opt.profile_off, "runId":run_id, "settings":settings,
        "reclaimEvery":opt.reclaim_every,
        "workloadSha256":workload_digest(&path)?})
    )?;
    let mut inflight = VecDeque::new();
    let mut measured_start = None;
    let mut descriptor_baseline = HashSet::new();
    let mut previous = metrics.snapshot();
    let mut last_rss = Instant::now();
    let mut frame = 0;
    let pace_start = Instant::now();
    while frame < opt.warmup + opt.frames {
        ensure!(
            setup.elapsed() < Duration::from_secs(3600),
            "workload timeout"
        );
        if inflight.len() >= 2 {
            wait(&environment.descriptors, inflight.pop_front())?;
        }
        if frame == opt.warmup && runner.is_preloaded() && measured_start.is_none() {
            wait(&environment.descriptors, None)?;
            inflight.clear();
            let mut player = runner.player().lock().unwrap();
            let renderer =
                <dyn Any>::downcast_mut::<WgpuRenderBackend<TextureTarget>>(player.renderer_mut())
                    .unwrap();
            renderer.collect_gpu_samples();
            metrics.take_gpu_samples();
            descriptor_baseline = environment
                .descriptors
                .runtime
                .resources()
                .into_iter()
                .filter(|r| r.owner == environment.descriptors.descriptor_owner)
                .map(|r| r.id)
                .collect();
            previous = metrics.snapshot();
            writeln!(
                output,
                "{}",
                json!({"event":"warmup", "setupNs":setup.elapsed().as_nanos() as u64,
                "firstMeasuredFrame":previous.stage_submissions+1})
            )?;
            measured_start = Some(Instant::now());
        }
        let reclaim_gc_before = if frame >= opt.warmup
            && opt
                .reclaim_every
                .is_some_and(|interval| (frame - opt.warmup + 1) % interval == 0)
        {
            let mut player = runner.player().lock().unwrap();
            Some(player.mutate_with_update_context(|context| {
                let gc = context.gc().metrics();
                let count = gc.total_gc_count();
                // With Player's normal pacing, all work per object is below one
                // unit. This pays the current partial cycle, the sleep threshold,
                // and a complete cycle in the upcoming normal Player::update.
                // No extra VM update or production-only cleanup API is needed.
                gc.adjust_debt(count as f64 * 4.0 + 1024.0);
                count
            }))
        } else {
            None
        };
        let start = Instant::now();
        ensure!(
            !matches!(runner.tick()?, TestStatus::Finished),
            "workload ended before measurement completed"
        );
        let cpu_ns = start.elapsed().as_nanos() as u64;
        if !runner.is_preloaded() {
            continue;
        }
        let mut player = runner.player().lock().unwrap();
        let renderer =
            <dyn Any>::downcast_mut::<WgpuRenderBackend<TextureTarget>>(player.renderer_mut())
                .unwrap();
        if let Some(submission) = renderer.last_submission() {
            inflight.push_back(submission);
        }
        renderer.collect_gpu_samples();
        drop(player);
        let current = metrics.snapshot();
        if frame >= opt.warmup {
            let rss = (last_rss.elapsed() >= Duration::from_millis(100))
                .then(|| {
                    last_rss = Instant::now();
                    resident_memory()
                })
                .flatten();
            write_record(
                &mut output,
                &FrameRecord {
                    event: "frame",
                    frame_id: if opt.profile_off {
                        u64::from(frame) + 1
                    } else {
                        current.stage_submissions
                    },
                    cpu_ns,
                    phase_ns: TimingDeltas {
                        names: &profile::Phase::NAMES,
                        current: &current.timings,
                        previous: &previous.timings,
                    },
                    native_ns: TimingDeltas {
                        names: &NativePhase::NAMES,
                        current: &current.native_timings,
                        previous: &previous.native_timings,
                    },
                    work: current.work.delta(previous.work),
                    rss_bytes: rss.map(|r| r.0),
                    native_resources: current.native_resources.as_ref(),
                    allocator: rss.map(|_| allocator_stats(&environment.descriptors.device)),
                },
            )?;
        }
        for sample in metrics.take_gpu_samples() {
            if measured_start.is_some() {
                write_gpu_record(&mut output, &sample)?;
            }
        }
        if let Some(gc_before) = reclaim_gc_before {
            wait(&environment.descriptors, None)?;
            inflight.clear();
            let gc_after = runner.player().lock().unwrap().gc_object_count();
            environment
                .descriptors
                .runtime
                .purge_idle_and_cached_resources();
            wait(&environment.descriptors, None)?;
            writeln!(
                output,
                "{}",
                json!({
                    "event":"reclaim", "measuredFrame":frame - opt.warmup + 1,
                    "gcBefore":gc_before, "gcAfter":gc_after,
                    "resources":environment.descriptors.runtime.diagnostic_snapshot(),
                    "allocator":allocator_stats(&environment.descriptors.device),
                    "rssBytes":resident_memory().map(|r| r.0),
                })
            )?;
            output.flush()?;
        }
        previous = current;
        frame += 1;
        if let Some(fps) = opt.pace_fps {
            let deadline = pace_start + Duration::from_secs_f64(f64::from(frame) / f64::from(fps));
            std::thread::sleep(deadline.saturating_duration_since(Instant::now()));
        }
    }
    wait(&environment.descriptors, None)?;
    let completed_ns = measured_start
        .context("no measured frames")?
        .elapsed()
        .as_nanos() as u64;
    let mut player = runner.player().lock().unwrap();
    <dyn Any>::downcast_mut::<WgpuRenderBackend<TextureTarget>>(player.renderer_mut())
        .unwrap()
        .collect_gpu_samples();
    for sample in metrics.take_gpu_samples() {
        write_gpu_record(&mut output, &sample)?;
    }
    let resources_after_drain = environment.descriptors.runtime.diagnostic_snapshot();
    let allocator_after_drain = allocator_stats(&environment.descriptors.device);
    drop(player);
    drop(runner); // Drops the movie GC arena and every display-owned lease.
    wait(&environment.descriptors, None)?;
    environment
        .descriptors
        .runtime
        .purge_idle_and_cached_resources();
    wait(&environment.descriptors, None)?;
    let unloaded_owner_resources: Vec<_> = environment
        .descriptors
        .runtime
        .resources()
        .into_iter()
        .filter(|r| r.owner != environment.descriptors.descriptor_owner)
        .collect();
    ensure!(
        unloaded_owner_resources.is_empty(),
        "unloaded owners retain resources: {unloaded_owner_resources:?}"
    );
    let remaining: HashSet<_> = environment
        .descriptors
        .runtime
        .resources()
        .iter()
        .map(|r| r.id)
        .collect();
    ensure!(
        remaining == descriptor_baseline,
        "shared descriptor resource identities changed after warmup; the warmed baseline must cover the full workload"
    );
    let memory = resident_memory();
    writeln!(
        output,
        "{}",
        json!({"event":"summary","frames":opt.frames,"completedNs":completed_ns,
        "peakRssBytes":memory.map(|r|r.1),"rssBytes":memory.map(|r|r.0),
        "droppedGpuSamples":metrics.snapshot().dropped_gpu_samples,
        "resourcesAfterDrain":resources_after_drain, "allocatorAfterDrain":allocator_after_drain,
        "resourcesAfterUnload":environment.descriptors.runtime.diagnostic_snapshot(),
        "allocatorAfterUnload":allocator_stats(&environment.descriptors.device),
        "unloadedOwnerResources":unloaded_owner_resources.len(),
        "descriptorBaselineResources":descriptor_baseline.len()})
    )?;
    output.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_trace_writer_preserves_exact_counters_and_gpu_identity() {
        let current = [
            profile::Timing {
                nanos: u64::MAX,
                ..Default::default()
            },
            profile::Timing {
                nanos: 3,
                ..Default::default()
            },
        ];
        let previous = [
            profile::Timing {
                nanos: 7,
                ..Default::default()
            },
            profile::Timing {
                nanos: 9,
                ..Default::default()
            },
        ];
        let mut output = Vec::new();
        write_record(
            &mut output,
            &FrameRecord {
                event: "frame",
                frame_id: 42,
                cpu_ns: u64::MAX,
                phase_ns: TimingDeltas {
                    names: &["first", "second"],
                    current: &current,
                    previous: &previous,
                },
                native_ns: TimingDeltas {
                    names: &[],
                    current: &[],
                    previous: &[],
                },
                work: profile::RenderWork {
                    draw_calls: 13,
                    ..Default::default()
                },
                rss_bytes: Some(1234),
                native_resources: Some(&json!({"active_bytes":456})),
                allocator: None,
            },
        )
        .unwrap();
        write_gpu_record(
            &mut output,
            &profile::GpuSample {
                frame: Some(42),
                nanos: 9001,
                phases: vec![("render".into(), 7000)],
            },
        )
        .unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.ends_with('\n'));
        let records: Vec<serde_json::Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0]["frameId"], 42);
        assert_eq!(records[0]["cpuNs"], u64::MAX);
        assert_eq!(
            records[0]["phaseNs"],
            json!({"first":u64::MAX-7,"second":0})
        );
        assert_eq!(records[0]["work"]["draw_calls"], 13);
        assert_eq!(records[0]["nativeResources"]["active_bytes"], 456);
        assert_eq!(records[0]["rssBytes"], 1234);
        assert_eq!(
            records[1],
            json!({"event":"gpu","frameId":42,"ns":9001,"phaseNs":[["render",7000]]})
        );
    }

    #[test]
    fn rejects_duplicate_case_names_and_zero_frames() {
        let duplicate = "[[case]]\nname='same'\npath='a'\n[[case]]\nname='same'\npath='b'";
        assert!(Manifest::parse(duplicate).is_err());
        assert!(validate_frames(120, 0).is_err());
        assert!(validate_frames(u32::MAX, 10).is_err());
        assert_eq!(validate_frames(120, 1200).unwrap(), 1322);
    }

    #[test]
    fn accepts_independent_workload_paths() {
        let manifest =
            Manifest::parse("[[case]]\nname='sparse-blend'\npath='../workload'").unwrap();
        assert_eq!(manifest.case[0].name, "sparse-blend");
    }

    #[test]
    fn reclamation_checkpoints_require_a_paced_soak() {
        let args = [
            "native-perf",
            "--manifest",
            "soak.toml",
            "--case",
            "cache_soak",
            "--output",
            "soak.jsonl",
            "--reclaim-every",
            "3600",
        ];
        assert!(Opt::try_parse_from(args).is_err());
        assert!(Opt::try_parse_from(args.into_iter().chain(["--pace-fps", "60"])).is_ok());
    }
}
