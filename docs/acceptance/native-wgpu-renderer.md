# Native WGPU renderer acceptance

Status: **INCOMPLETE — do not merge on the basis of this report.** Tasks1–15 are implemented; Task16 performance and live-resource-plateau gates remain open. This is local work on `codex/native-wgpu-renderer`, with no push or merge.

## Scope and provenance

The approved contract is [the native renderer design](../ryugi/specs/2026-11-17-native-wgpu-renderer-design.md). Desktop and Android use native wgpu30.0.1; Web/wasm compatibility is excluded. Resource admission failures terminate through structured renderer errors; no replay, omitted effects, replacement pixels or reduced resolution is authorized.

Frozen instrumentation baseline: Ruffle `989100c1353391f0abe756802752080bf96bedba`, parent `68325477e6739ed7f47410d23fdd2ba92e051b05`. The completed pixel/performance matrix below used candidate `6145313999`. Subsequent review fixes are identified separately; they do not inherit old performance acceptance.

Evidence directory: `.ryugi/sdd/2026-11-17-native-wgpu-renderer-design/` in the implementation worktree. It contains raw JSONL, commands/logs, workload/build hashes, frozen executables, baseline PNG comparisons and the review report. It must be retained while acceptance remains incomplete.

Hardware: NVIDIA RTX5060 Laptop (Vulkan/GL620.70.06) and Radeon780M (RADV Mesa26.2.6-arch3.1). Local native GL exposed3.3. Android ARM64 has compile evidence only; no Android device, Windows/DX12 or macOS/Metal runtime evidence is available.

## Correctness and safety

Candidate614531 final402-case sweeps:

| Backend | Pass | Fail | Ignored |
| --- | ---: | ---: | ---: |
| NVIDIA Vulkan |393|4|5|
| Radeon Vulkan |394|3|5|
| Native GL |396|1|5|

Every remaining failure has unchanged tolerances and byte-identical actual baseline output. NVIDIA has four displacement-map cases; Radeon has `avm2/pixelbender_sign`, `visual/edittext/edittext_caret_empty` and `visual/edittext/edittext_gutter`; GL has `visual/edittext/edittext_bounds_vs_position`. Five ignored cases comprise recorded known exclusions and unavailable freetype variants. These are compatibility gaps, not passes.

GL comparison requires the common capability fix on both versions: published wgpu-hal30.0.1 is vendored with one authored predicate correction, using Naga's explicit-binding support for `SHADER_BINDING_LAYOUT`. The original GL baseline rendered blank. Patch provenance/removal condition: [RUFFLE-PATCH.md](../../render/wgpu/vendor/wgpu-hal/RUFFLE-PATCH.md). GL actual PNG hash for the remaining difference: `67b699fa68a35bfa5264f78543b7fa063a5883c53c63e968ecbc1d3b730cafe7`.

The fresh whole-branch review found four Important issues and no Critical or deferred Minor issues. One author fix pass addresses ShaderData allocation panic, Stage3D terminal latching, bounded Stage3D staging, and resource ownership/unload validation. Corrected RED→GREEN tests plus suites pass: core245 (9 existing ignored), real GPU107/107, native-perf3/3, host real surface5/5 on each of GL and Vulkan. ShaderData's first fixture failed class lookup and is not valid RED evidence; `task-16-shader-data-red.log` uses the resolved real playerglobals class and reproduces the allocation panic. Post-review fixes have no second independent source review.

The owner oracle now permits only exact warmed shared Descriptor ResourceIds. Context/shader resources use distinct Scene/ShaderJob owners, including objects intentionally surviving their context. A blanket Renderer/Host exemption is removed. The host uses the production native instance initializer; its earlier real GL device-loss test is now green.

Android ARM64 and desktop/Tracy/exporter/image-test checks passed at614531. Final affected platform/pixel checks after review/performance fixes remain pending.

## Performance: failed

Each group uses120 warmup +1200 measured frames,5 fresh-process AB/BA pairs, full frame-linked GPU samples, final-drain throughput, identical workload/settings and no screenshot readback. No builds, other renderer tests or soak ran concurrently; normal host desktop applications remained running. Only2/16 groups passed, both dynamic controls. No optimization speedup or overall performance acceptance is claimed.

| Device | Case | Throughput gain | CPU P95 regression | Target phase P95 gain | Worst-pair peak RSS growth | Accepted |
| --- | --- | ---: | ---: | ---: | ---: | --- |
|nvidia|sparse_blend|-26.0%|+48.4%|+1.4%|+49.7%|no|
|nvidia|alpha_mask|-27.7%|+28.6%|+0.2%|+0.4%|no|
|nvidia|filter_cache|-32.6%|+63.3%|+29.7%|+0.4%|no|
|nvidia|worker_prepare|-34.3%|+59.9%|-25.8%|+0.9%|no|
|nvidia|static_tree|-60.9%|+117.6%|+1.4%|+3.3%|no|
|nvidia|dirty_1_percent|-65.8%|+189.4%|+1.3%|+4.2%|no|
|nvidia|dirty_all|-56.4%|+101.8%|+1.8%|+1.7%|no|
|nvidia|dynamic_control|+0.2%|-9.1%|+36.0%|-0.0%|yes|
|radeon|sparse_blend|-15.5%|+19.4%|-16.0%|+1.5%|no|
|radeon|alpha_mask|-26.5%|+27.8%|-39.0%|+0.8%|no|
|radeon|filter_cache|-29.2%|+24.9%|+12.1%|+0.8%|no|
|radeon|worker_prepare|-35.1%|+81.6%|-39.5%|+7.5%|no|
|radeon|static_tree|-60.0%|+138.1%|-19.5%|+10.2%|no|
|radeon|dirty_1_percent|-63.1%|+154.9%|-28.6%|+11.3%|no|
|radeon|dirty_all|-52.7%|+84.2%|-16.5%|+11.7%|no|
|radeon|dynamic_control|+1.7%|-3.8%|+46.9%|+0.1%|yes|

Target phase is GPU except `worker_prepare`, where it is CPU preparation. GPU targets require10% phase P95 and5% throughput improvement; preparation target requires10% preparation P95 and5% throughput improvement. Controls permit at most3% CPU/GPU P95/P99 regression. All phases permit at most5% cumulative peak RSS growth. The comparator and thresholds remain unchanged.

## Lifetime and memory: incomplete

The earlier100 real GPU load/unload cycles pass with exact warmed baseline identities. The old30-minute paced soak completed108000 measured frames,100% GPU coverage and no terminal error; complete player drop/drain/purge reclaimed nonpersistent owners. However, live mesh buffer counts grew between natural GC cycles, so that evidence does **not** prove a live-player plateau. The earlier blanket-owner unload assertion is also superseded.

The revised runner offers paced-only GC/reclaim checkpoints through normal Player update boundaries, without extra ActionScript updates. Final short-checkpoint investigation,30-minute final-candidate soak and100-cycle proof with the strengthened oracle remain pending.

Logical reserved/active/idle/retiring capacity estimates, driver allocator allocated/reserved storage and process RSS are separate measurements. Neither logical bytes nor driver reports are hardware capacity. OS process OOM survival and exact driver-private allocation accounting are outside the controlled GPU-resource contract.

## Open work

- Attribute and correct measured CPU/throughput regressions; rerun the full isolated matrix on a frozen final candidate.
- Prove live post-GC/purge resource plateau and final unload identities on the final candidate.
- Rerun affected native/pixel checks after the final edits and record final commits/build hashes.
- Android device, DX12 and Metal runtime remain explicitly unverified until hardware is available.
