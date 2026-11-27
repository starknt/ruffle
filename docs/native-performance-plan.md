# Native performance implementation, 2026-11-17

Approved in the conversation. Scope: portable native Ruffle core and WGPU.
Keep ActionScript, timeline,
events, GC and pixel semantics. Optimizations are automatic, not user switches.

## Deliverables

1. P0: extend existing RenderMetrics with draw/pass/region/copy/state work,
   asynchronous GPU frame identities, cache work and catch-up diagnostics.
   Add an offline native performance runner with a no-readback render target,
   bounded in-flight submissions, buffered JSONL and GPU-completed throughput.
   Freeze an instrumentation-only baseline before adopting optimizations.
2. P1: conservative Empty/Finite/Full regions and coordinate origins, smaller
   built-in blend/independent alpha-mask targets and copies. Preserve full-target
   behavior for ancestor Layer sampling, shaders, perspective and unknown bounds.
   Keep filtering, cacheAsBitmap and submission-safe resource reuse semantics.
3. P2: retained shared command blocks for static 2D stage subtrees only. Exclude
   masks, filters, scrollRect, bitmap caches, dynamic text, video, Stage3D,
   offscreen and non-default render options. Independent render invalidation;
   complete inherited transform/color/viewport/quality keys. Do not skip logic
   or pre-render state commits. Release on unload and count pinned resources.
4. P3: lazy per-method, GC-traced call-site PICs for static names and sealed
   ordinary objects: get/set/callproperty/callpropvoid. Weak VTable identities,
   trait descriptors, at most four receiver types; preserve conversion, errors,
   accessors, stacks and timeout checks. Published opcodes stay immutable.
   Dynamic objects, Proxy, XML, special indices and scope/domain lookup bypass.

## Budgets and acceptance

Each Player has a 24 MiB render cache and an 8 MiB PIC cache. Both allocate
lazily and evict by LRU. Existing GPU resource budgets apply to both paths.

Run five alternating AB/BA pairs per GPU in fresh processes with matched
settings and input. Warm up for 120 frames, measure 1,200, allow at most two
frames in flight, and include the final GPU drain in completed throughput.
Measure cold starts and capture correctness images in separate runs.

For each GPU, take the median paired change in each of three metrics. A
candidate qualifies when completed throughput and **1% low FPS** increase
and the P95 interval between consecutive GPU stage completions decreases.
Any positive improvement qualifies. For N valid intervals, take the longest
ceil(N/100), average their lengths in seconds, and take the reciprocal.
Report N and the tail sample count. Report target-phase P95, CPU/GPU P95 and
P99, and worst-pair peak RSS growth as costs. The separate control comparison
retains its 3% timing and 5% RSS tolerances. [Runner and trace fields](../tests/perf/README.md)

Retain image tolerances and check the production AVM and timeline paths,
native renderer regressions, 30-minute playback and 100 load/unload cycles.
Record which platforms ran and which had only compile coverage.

Six offline workload groups: sparse blend/mask, transformed filter caches,
static/1%/100%-dirty display trees, mono/poly/megamorphic traits, dynamic-object
negative controls, load/unload ownership. Existing regression SWFs supplement
synthetic workloads. Aola is supplementary, not a native acceptance prerequisite.

## Later work (research only in this implementation)

Adjacent pass merging requires matching attachment/load/store/stencil semantics
and no dependencies. Native preparation workers require >=20% CPU time in two
workload groups and immutable inputs, with no added frame latency. Upload
scheduling only delays invisible reconstructible prefetch. Catch-up changes
need independent Timer/audio/event design. PGO uses separate training/holdout
sets and the existing dist profile; portable builds do not use target-cpu=native.
Dynamic shape/epoch PICs, broader snapshots and JIT/IR require a separate design.
