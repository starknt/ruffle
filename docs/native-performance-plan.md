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

New per-Player caches: 24 MiB render, 8 MiB PIC; lazy, bounded, LRU. Total peak
RSS increase <=5% against the common baseline, not +5% per stage. Existing GPU
budgets do not increase. Capacity is not proof of RSS acceptance.

Default performance protocol: 120 warmup +1200 measured frames, five AB/BA
pairs, fresh processes and identical builds/settings/input. Cold start is
separate. GPU-completed throughput includes the final drain; CPU/GPU samples
remain separate. At most two in-flight frames. Correctness captures do not
run inside the measured section. For new target comparisons, either GPU-completed throughput or target-phase P95
must improve on each card, with the other metric at least level. CPU and GPU
P95/P99 regressions must stay within 3% and worst-pair peak RSS growth within 5%. Preserve image
tolerances. Test default/non-avm_debug AVM and timeline paths, native renderer
regressions, 30-minute loops and 100 load/unload cycles. Missing platform/device
coverage is explicitly not-run, never inferred from compilation.

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
