# Native performance corpus

Build `native-perf` with `cargo build -p tests --no-default-features --features
imgtests --bin native-perf --profile dist`. Compile fixtures separately using
`cargo run -p tests --bin tests -- compile native_perf`. Keep the resulting
executables, their hashes, manifest and SWFs fixed during comparisons.

Example, from the Ruffle root:

```sh
target/dist/native-perf --manifest tests/perf/native.toml --case sparse_blend \
  --output /tmp/sparse-blend.jsonl
python3 tools/native_perf.py --baseline /absolute/baseline --candidate /absolute/candidate \
  --manifest tests/perf/native.toml --output /absolute/new-results --case sparse_blend
```

Each invocation runs one workload in a fresh process. The target has no capture
buffer, the queue is bounded to two in-flight frames, and `completedNs` includes
the final GPU wait. GPU samples retain their source frame IDs. This is frame
service time/throughput, not display presentation FPS. Setup, warmup and measured
work are separate. Raster correctness belongs to the image regression suite.

`--backend` selects vulkan/metal/dx12/gl; unsupported devices fail explicitly.
RSS is currently collected on Linux; missing RSS on other systems is null and
cannot pass the memory acceptance gate. `phaseNs` values overlap and must not be
summed. No profiling results are an ActionScript compatibility reference.

Five alternating AB/BA pairs are required by the evaluator. Use `--phase
renderTree` for snapshot work, `--phase avm2` for PIC work, and `--control` for
workloads which should not speed up but must not regress. Existing image and
AVM regression fixtures provide independent semantic references.

For Linux Vulkan and GL runs, the evaluator gives each process an empty NVIDIA
and Mesa shader disk cache. This includes cold shader compilation in both peak
RSS measurements. The cache policy is recorded in `builds.json`.

The load/unload case performs one cycle every 12 frames, so the default 1200
measured frames include 100 cycles. Its tiny `child.swf` is the repository's
existing `visual/blend_modes/multiply/test.swf`, not an external download.

For a 30-minute same-process cache lifetime soak, compile
`native_perf/cache_soak` and run:

```sh
target/dist/native-perf --manifest tests/perf/soak.toml --case cache_soak \
  --warmup 120 --frames 108000 --pace-fps 60 --output /absolute/soak.jsonl
```

The loader's child supplies a static subtree, sealed property calls, blend
targets and an alpha mask. On each load the parent displays the static subtree
beside the Loader so it can enter the retained-command cache, and detaches it
before reloading every 120 frames (900 measured cycles). Pacing
is outside the per-frame CPU timer. The evaluator rejects paced traces for
throughput comparisons. Validate that snapshot/PIC/mask/blend counters are
nonzero, GPU samples have full coverage, and RSS reaches a stable plateau.

`--profile-off` disables measurement hooks in the runner while retaining all
engine optimizations. Both binaries must use the same instrumentation mode.
Comparing profiling on versus off measures enabled instrumentation cost; it
does not prove the cost of disabled hooks. That requires an otherwise identical
reference build with the hooks removed and both runners using `--profile-off`.

Native WGPU acceptance uses schema 2 traces. The session records a unique process
identity, driver, viewport, scale, quality/MSAA, timestamp features and a SHA-256
of the executed workload assets and settings. Comparisons require matching
settings on the same driver and adapter, release builds without AVM debug,
at least 120 warmup and 1,200 measured frames, five fresh AB/BA pairs, complete
GPU frame coverage and process peak RSS. For each process the evaluator orders
GPU stage completion timestamps (`stageEndNs`) by source frame ID. With N
intervals, 1% Low FPS equals one billion divided by the mean nanoseconds of the
longest ceil(N / 100) intervals. A 1,200-frame run supplies 1,199 intervals and
12 tail samples. This includes idle time between stages.

A candidate qualifies when its paired median completed-frame throughput and
1% Low FPS rise and its paired median frame-interval P95 falls. Any positive
improvement counts. The report includes the target phase P95, CPU and GPU
tails and worst paired RSS growth. `--phase gpu` selects whole-frame GPU work,
and `--phase preparation` selects owner-thread preparation. Historical traces
without stage end times remain readable and cannot qualify a candidate.
The separate `--control` regression check applies its 3% tolerance to frame
interval P95 growth and 1% Low FPS loss as well as its other timing limits.
It also checks the 5% peak RSS bound and requires complete interval measurements.
Use `--compare-bitmap-bindings --case ordered_bitmaps --phase gpu` to compare
explicit slots and binding arrays with the same device features and executable.
Absent, orphan or duplicate GPU identities never become zero-time measurements.

`nativeNs.preparation` measures owner-thread CPU preparation and joining, excluding
GPU publication; it does not sum worker durations. `nativeResources` reports the
shared device's logical states, resource classes, retry/failure and worker counts.
Driver allocator allocated/reserved bytes are sampled separately under `allocator`
and `allocatorAfterDrain`; unsupported allocator reports remain null. Named GPU
`phaseNs` query durations are inclusive, and nested scopes must not be summed as
frame time. `work.native_*` counts calls actually forwarded after state suppression.
