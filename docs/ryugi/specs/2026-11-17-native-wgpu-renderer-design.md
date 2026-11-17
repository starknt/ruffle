# Native WGPU renderer design

Approved by the user in chat on 2026-11-17. This is the implementation contract,
not a report that the features or acceptance already pass.

## Intent and decisions

Implement the entire proposed renderer program: resource safety, an explicit
pass graph, smaller offscreen work, batching and uploads, BitmapData residency,
adaptive cache retention, native CPU preparation workers, and backend validation.
The user chose WGPU-native-only (desktop and Android), pixel fidelity with safe
termination on unrecoverable errors, all stages, and automatic device budget tiers.

Preserve Flash draw order, pixels, ActionScript/event timing and synchronous
pixel observability. Do not degrade resolution, omit filters, substitute images,
restart the movie, or rebuild a lost device transparently. Optimizations are
automatic, without user-facing experiment/fallback switches. Resource errors must
not become panics, silent no-ops, or successful reads of stale pixels. This is a
controlled GPU resource-failure contract, not a promise to survive OS process OOM
or arbitrary programming bugs.

Work on the existing implementation rather than replacing already useful bounded
idle pools, immutable bitmap residency, regional blend/masks, instancing, bindings,
stage snapshots and native performance instrumentation. Preserve preexisting work.
Default delivery is local staged commits, without a push or merge.

## Native boundary

Keep wgpu 30.0.1. Remove WGPU's canvas entry point, wasm branches/dependencies,
browser backend and webgl feature. Select native std/parking_lot/wgsl/naga-ir,
Vulkan/DX12/Metal/GLES features explicitly. Preserve native GL/GLES and negotiate
adapter capabilities; a GLES requirement is not removed merely because an API
uses the word WebGL. Make native methods/futures unconditional and reject wasm
explicitly at the WGPU crate boundary.

Remove WGPU registrations/dependencies/features from the Web frontend to avoid
dangling Cargo feature references. Keep independent canvas/WebGL source and
unrelated wasm code. Mechanically adapt shared interfaces when necessary, but
remaining Web build/runtime compatibility is explicitly unverified. No wasm gate.

## Device runtime, accounting and policy

A device-scoped GpuRuntime is shared by Descriptors and the host. It owns the
resource registry, budget governor, device-health/error sink and submission
tracker. Register device callbacks once, before GPU initialization, in the shared
context or standalone constructor; do not overwrite callbacks per renderer.

Resources have unique IDs/device generations, opaque monotonically issued
Movie/object/job owner IDs, kind, checked storage size (including mips, blocks,
samples, layers and alignment), creation/last-use information, dependency IDs,
priority and reconstructibility. Metadata never owns GC objects or strong GPU
handles. Charge storage once; views/bind groups track references/counts rather
than charging parent storage again. Include mesh/glyph, Stage3D, PixelBender,
filters, staging/readback and imported host stage slots.

Separate reserved-for-allocation, active, idle and retiring/in-flight bytes.
Accounting is always enabled; profiling only samples it. Report logical estimates,
driver allocated/reserved and RSS separately. Shared UI allocations not controlled
by Ruffle are external observations, not a second charge of tracked resources.
Allocated/reserved totals are not hardware capacity.

Initial defaults (Android takes precedence over adapter type):

| Tier | Hard total | Shared reconstructible cache | Idle pool | In-flight staging |
| --- | ---: | ---: | ---: | ---: |
| Desktop discrete | 2 GiB | 256 MiB | 64 MiB | 64 MiB |
| Desktop integrated/other | 1 GiB | 128 MiB | 32 MiB | 32 MiB |
| Android | 512 MiB | 64 MiB | 16 MiB | 16 MiB |

Soft total is 75% of hard. All category budgets are subsets of total. Transient
target scheduling watermarks are 128/64/32 MiB respectively. Internal host/test
constructor overrides are allowed, no user tuning switches. Existing immutable
bitmap retention joins the shared reconstructible cache budget. Where supported,
set wgpu native resource-creation pressure threshold to 90%; do not configure a
forced device-loss threshold. Other backends use registry admission and errors.

Reserve checked capacity before allocation. Under soft pressure evict idle then
oldest reconstructible caches. Under hard pressure also submit at safe dependency
boundaries and wait for the oldest submission, with bounded polls and a total
2-second reclamation deadline. Never evict mutable GPU-only state or live owners.
Unloading respects intentional surviving AS references. Submission completion,
not frame N+2 or Rust handle drop, governs retirement and cross-submission pool
reuse. Within an encoder, reuse requires graph-proven nonoverlapping lifetimes.
Invalidate dependent binding caches by resource generation. Replace the arbitrary
100-draw flush rule with resource bytes and dependencies.

## Failure contract and interfaces

Central fallible GPU creation functions use wgpu 30 ErrorScopeGuard::pop on the
owning thread; callbacks record bounded error state without GC/UI reentrancy.
Use structured errors for budget, OOM, device loss, validation/internal errors,
map failure and timeout. Constructors, target resize/submission/readback and
allocation-bearing RenderBackend calls return Result. Unsupported capability is
Ok(None), not an error; a resource failure is never swallowed into CPU/uncached
fallback. BitmapData synchronization/handle access propagate errors through core.

Player latches fatal failure, aborts the current internal VM operation without
manufacturing an AS-visible success, and the host stops tick/render/publication.
Poll health at execution and publication boundaries for asynchronous faults.
Reclaim and retry a resource creation once, only before externally visible GPU
writes. Drop failed encoders/output; never replay submitted BitmapData/Stage3D
writes or re-run ActionScript. Submission errors and device loss are terminal.
Invalidate host slot generations, cancel maps/jobs, and use the existing host
error/log channel (do not require a lost GPU to render an error dialog).

Ruffle has at most two own in-flight submissions, including direct/offscreen/stage.
Slint submissions are independent; preserve its three-slot bridge and track both
producer and consumer completion/generation. Never wait/poll while holding the
shared submission lock.

## Snapshot, pass graph and scheduling

Pipeline: DisplayObject -> immutable RenderSnapshot -> ordered RenderCommandList
-> PassGraph -> ResourceScheduler -> wgpu encoder/queue.

Main/player thread finishes AVM/GC/pre-render events before capturing transforms,
bounds/material/mask/filter state and numeric resource/version IDs. Register shapes
and bitmaps as lazy CPU recipes; display traversal does not allocate GPU resources.
DistilledShape borrows styles: freeze owned styles/path data once. Reuse shared
immutable pixels where possible; dynamic data uses copy-on-write. Existing static
command snapshots feed this pipeline, not another competing cache.

Graph nodes explicitly represent uploads, draws, masks, blends, filters, copies
and readbacks, including versions/read-write dependencies, origins/bounds and
clear/load/store/resolve/stencil semantics. Stage3D and shader operations are opaque
dependency barriers with managed resources. Stable topology preserves Flash order;
no global material sorting. Schedule based on lifetimes/peak live set, exact texture
keys and legal submit boundaries; reserve an operation's outputs before writes.

Eligible normal runs render directly to the final attachment (MSAA resolve when
needed), avoiding redundant copies. Complex work uses bounded offscreen regions.
Propagate empty/finite/full bounds, old/new dirty extents, filter halos/backward
sampling requirements, mask intersections and target clipping with correct origins.
Unknown shader/perspective/ancestor-Layer reads retain full valid coverage.

Merge only adjacent passes with compatible attachments, load/store/stencil and
dependencies. Eligible 1:1 pointwise ColorMatrix chains may fuse while preserving
each stage's premultiplication, clamp and RGBA8 quantization. Blur/glow/bevel keep
their algorithm order and reuse ping-pong intermediates.

## Batching and uploads

Extend existing adjacent instancing/geometry merging with full pipeline/material/
texture/sampler/blend/mask/stencil/scissor/vertex/index compatibility. Boundaries
and read-write hazards cannot be crossed. Keep four per-texture bitmap bindings;
bound temporary copy/filter/material reuse by resource generation/layout and LRU.
Suppress redundant pipeline/bind/vertex/index/scissor state changes per pass.

A three-slot paged FrameUploadArena and reusable CPU scratch replace per-chunk
fresh vectors and growth-only dynamic buffers. Cover dynamic transforms, vertices,
indices and staging; immutable meshes remain persistent rather than reuploaded.
Retire by submissions, honor alignment/block rows, and bound input/output storage.
Visible mandatory uploads finish in the same logical frame through bounded slices
and submissions. Only invisible reconstructible prefetch can defer.

## BitmapData and cacheAsBitmap

Use one authoritative CPU/GPU version/valid-region model. Metadata does not sync;
pixel reads sync required dirty regions, full overwrite skips readback, GPU draw
uploads only CPU changes. Fragmented regions conservatively synchronize before
coalescing so no valid writes are lost. A failed sync never marks stale data clean.

Extend exact GPU copyPixels to chained GPU destinations and self-overlap via a
budgeted temporary. Alpha/merge GPU shaders must preserve original rounding and
premultiplication; otherwise retain the correct CPU algorithm. Actual GPU errors
are terminal, not unsupported. All operations remain observable in AS order.

Track content/transform/alpha/filter/mask/blend cache invalidation. Translation and
alpha can reuse; scale/rotation follow current raster-space semantics. Partial
redraw clears dirty area and redraws ALL intersecting contributors in order,
including old/new extents and filter halo; unknown dependencies redraw fully.

Adaptive retention does not disable required cache/filter rasterization or pixel
snapping: after 8 consecutive dirty frames keep only a transient frame result;
after 30 unchanged frames retain across frames again. Retention shares the global
budget and unload ownership. Never drop filters due to budget pressure.

## Native preparation workers

Use the existing Rayon facility with at most min(4, max(1, CPU count - 2))
outstanding tasks per shared runtime. Workers receive only Send immutable CPU data
and IDs, never GC/DisplayObject/Rc/GPU handles or renderer ownership. Do not add
unsafe Send implementations or loosen existing backend handle traits.

Parallelize pure tessellation, bounds/transforms with preserved arithmetic order,
packing and graph preparation in deterministic partitions. Fewer than 256 commands
AND fewer than 8192 input vertices uses serial processing. Bound queues/scratch
storage and join in the same frame before encoding, with no extra frame latency.
Generation invalidates stale results on unload/resize/failure. AS BitmapData and
Stage3D synchronous barriers stay ordered on the owner thread.

## Verification and acceptance

Freeze a common instrumentation baseline at implementation start, including needed
preexisting work. Fault tests cover admission/rollback, dependencies/aliases, mip
and alignment accounting, submission retirement, creation retry, terminal submit/
map/device loss, stale jobs and bridge generations; do not exhaust physical VRAM.
Pixel regressions cover nested masks/alpha/erase/Layer, all filters and shaders,
fractional coordinates/MSAA/quality, overlapping copies/alpha, cache dirty regions,
Stage3D and host resize. Do not relax established image tolerances.

Run 100 load/unload cycles and 30-minute paced stress. After drain, GC and explicit
cache purge, unloaded owners have no orphan resources; live/retiring capacity
plateaus. Report allocator allocated/reserved and RSS separately.

Performance protocol: 120 warmup + 1200 measured frames, five AB/BA pairs, fresh
processes, identical build/settings/input. No screenshot readback in measurement;
completed throughput includes final drain. GPU target groups require >=10% actual
frame-linked GPU P95 gain and >=5% completed-throughput gain. Worker groups require
>=10% CPU preparation P95 gain and >=5% throughput. Control CPU/GPU P95/P99 regression
<=3%, cumulative total peak RSS growth <=5% for all phases. Safety phase needs
correctness and no material regression, not forced speedup. Missing GPU samples
cannot be zero or accepted; extend native_perf's current CPU-only gate.

Primary local runtime targets: RTX 5060/Vulkan and Radeon 780M/Vulkan. Exercise
available native GL and Android builds/real hardware; add Windows/DX12 and
macOS/Metal native build/runtime evidence when available. Missing hardware remains
explicitly not verified; compilation is not runtime acceptance. Stages missing
acceptance remain incomplete, even if draw counts or a single FPS run look better.

Final whole-branch fresh review and verification report are required. The initial
design received a fresh builder check with no remaining implementation-blocking
questions. Remaining Web compatibility is deliberately not an acceptance claim.
