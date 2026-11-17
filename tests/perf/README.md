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

The load/unload case performs one cycle every 12 frames, so the default 1200
measured frames include 100 cycles. Its tiny `child.swf` is the repository's
existing `visual/blend_modes/multiply/test.swf`, not an external download.
