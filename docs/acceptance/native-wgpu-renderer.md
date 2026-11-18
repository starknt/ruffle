# Native WGPU renderer acceptance

Status: **Implementation delivered locally; resource-lifetime and native consumer checks pass. Final performance is 15/16, with NVIDIA worker preparation/tail gates unmet. Golden-image and unavailable-platform limitations also remain open. Task 16 is incomplete.** No push or merge.

## Scope and provenance

The contract is [the approved native renderer design](../ryugi/specs/2026-11-17-native-wgpu-renderer-design.md). Tasks 1–15 are implemented. Task 16 has final-source build, resource, host, image-comparison, performance and paced lifetime evidence, but remains open for the unmet performance gates and unwaived golden-image/platform gaps below.

WGPU is native-only for desktop and Android, pinned to wgpu 30.0.1. Independent Web/wasm frontends are outside this verification. Controlled GPU resource failures terminate through typed errors; no replay, omitted effects, replacement pixels, reduced resolution or silent device recreation is authorized. OS process OOM survival is outside this contract.

Frozen renderer source: `624fdb4510c21c94fa9c4854b229452b1a7d2176`. Test-only commit `e7e90c5655` subsequently extends owner/parallel pixel coverage to 320 overlapping shapes; the runtime source is byte-identical. Both adapters pass that expanded test after restoring the 32-shape implementation. Parent integration: `5c360b10d6d1b033a5fab5cee885f9d29784c0c6`. The later report-only commits do not change this tested source. The parent build includes the independent, unstaged 21-line `render_profile.rs` addition, whose SHA256 remains `2d39c29e551ef09eb58a39013b097120cc33f14deed08104bc7b1c2036740fb3`; it is excluded from task commits.

Original baseline: Ruffle `989100c1353391f0abe756802752080bf96bedba`, parent `68325477e6739ed7f47410d23fdd2ba92e051b05`. Common instrumentation patches add identical recursive GPU-phase aggregation and direct trace serialization to the archived baseline. Missing baseline resource gauges remain absent, never fabricated as zero. The native GL baseline also receives the same capability predicate correction documented in [RUFFLE-PATCH.md](../../render/wgpu/vendor/wgpu-hal/RUFFLE-PATCH.md).

Evidence lives under `.ryugi/sdd/2026-11-17-native-wgpu-renderer-design/`, including commands, raw traces/logs, frozen executables, source/workload hashes, PNGs and review findings. Preserve it with this worktree. Final series prefix: `final-reclaim-20261118`.

Candidate native-perf SHA256: `30aff5b47527db290f0f6135eb0f19bb2e113999240e7a840d8a9678bb814df3`. Baseline native-perf SHA256: `328835a9c94a70d02c3cfa2b5ed856e3a1158c9d22f8d422d0235f4a8773fb51`. Frozen at 2026-11-17T23:16:12.608534+00:00.

Hardware: NVIDIA RTX 5060 Laptop / Vulkan 620.70.06; Radeon 780M / RADV Mesa 26.2.6-arch3.1; available native GL. Toolchain/host and empty Android device inventory are captured in `final-reclaim-20261118-environment.json`; the final empty device inventory is also retained in `final-reclaim-20261118-final-device-inventory.json`.

## Implementation

- Native device-scoped budgets, checked allocations, resource/owner IDs, shared health latch, bounded retries and submission-completion retirement cover render, Stage3D, PixelBender, uploads and host stage slots.
- Immutable snapshots and ordered pass graphs retain draw/dependency order. Regional compositing, filter planning, exact dirty coverage, adjacent batching, dynamic bindings and bounded material reuse reduce work without changing sampling rules.
- Paged uploads, asynchronous staging-page mapping, submission use-set ownership and cache eviction remain bounded by shared policy.
- BitmapData keeps synchronous pixel observability; adaptive cache retention and collectible movie/library roots release unloaded resources while respecting surviving AS references.
- Bounded native workers prepare immutable CPU data and join before owner-thread publication. Hot VM checks use a live shared latch without reducing check frequency.
- Completed GPU timestamp query pools now participate in the shared idle budget, including query, resolve and readback storage. Pending/active pools remain protected, and disabling timing clears the actual cache. Paced checkpoints collect every completed sample before purging.
- Imported host targets use `TextureTarget::from_texture`, preserving tracked storage, identity and owner while initializing the private view cache without an extra allocation.

## Correctness, safety and consumers

Final renderer suites: **120/120 on each Vulkan adapter**, including actual GPU, native GL pixel, fault-injection and 100-cycle exact-owner unload tests. Core: **248 passed, 9 preexisting diagnostic tests ignored**. Render: **64 passed**. Native-perf runner: **4 passed**. Comparator: **12 passed**.

Desktop/Tracy/exporter/image-test, Android ARM64 renderer and parent aola-ui compilation pass. Host surface suites pass **5/5 on Vulkan and GL**, including real GPU resize and consumer-completion retirement. Commands and per-check timestamps are in `final-reclaim-20261118-verification-exit.json`. Builds have warnings; warning-free/strict-Clippy acceptance is not claimed.

Two final-reclaim wrapper invocations initially failed before testing (Cargo feature package selection and Python sibling-module import). Their original logs and commands are retained in `final-reclaim-20261118-invocation-errors.json`; corrected invocations pass. The initial final-health invocation without `wgpu/counters` failed three counter-oracle tests; the correctly configured final commands pass. The actual host integration compile failure from a private cached-view field was fixed with the tracked-texture constructor before this final freeze. Both failed records are retained.

The fresh whole-branch review found four Important issues: ShaderData allocation panic, Stage3D terminal propagation, bounded Stage3D staging, and the overly broad unload-owner oracle. All were fixed and covered by regression tests. Later performance/interface edits have test evidence but no second independent source review. The original review remains in `final-review.md`.

## Image and semantic regressions

Each backend ran 489 selected cases, covering native performance fixtures, loaders/imports/SymbolClass, mixed AVM lifetimes, BitmapData, filters/shaders, masks/blends, cache and Stage3D. Existing tolerances were unchanged. Prior generated actual PNGs were preserved and removed before each new run; baseline outputs were also regenerated.

| Backend | Passed | Failed | Ignored |
| --- | ---: | ---: | ---: |
| NVIDIA Vulkan | 478 | 4 | 7 |
| Radeon Vulkan | 479 | 3 | 7 |
| Native GL | 480 | 2 | 7 |

**These sweeps are not all green. Every observed failed output in this final sweep matches a freshly run frozen baseline exactly.** The original expected-failure-set predicate is **FALSE** for this final sweep; it is separate from the all-green golden-image gate. Exact current failures are recorded in `final-reclaim-20261118-image-assessment.json` and the per-adapter baseline comparisons.

The previous bc72719557 sweep also exposed a variable Radeon `avm2/stage3d_bitmap` counter-character difference. Its candidate and fresh baseline produced PNG SHA256 `a962b3839bb5d2d55a45db75f70f52517178f6f7e98c44cc63b29ab3228580ed`: 10 pixels / 30 RGB channels in one character exceed the original tolerance (4; maximum 24 outliers). A fixed three-pair follow-up had baseline 3 passes and candidate 2 passes / 1 identical failure. The pairwise pass/fail assertion **failed**, and remains retained in `final-native-20261118-stage3d-control/`. The SWF contains getTimer/FPS/counter logic, but its precise timing cause remains unresolved. A later passing run does not erase this limitation. No tolerance, test fixture or rendering behavior was changed to mask it.

In the final GL sweep, `stage3d_bitmap` also failed with 30 channel outliers. Candidate and freshly run GL baseline both produced PNG SHA256 `a08867d2a2335c7d560907b9a3885f797e887284d0405a22efbba9c7477b9ab5`. This remains an additional failure against the original GL failure set; it is not silently added to that expected set.

This is baseline-output comparison evidence, not complete Flash compatibility or a waiver of golden-image failures.

Seven ignored cases remain: bitmap_data_thorough/pixelDissolve, bitmap_pixelsnapping, loader_jpegxr, loader_jpegxr_alpha, and three freetype/default-font variants. The optional JPEG-XR/freetype paths and known exclusions are not validated by these runs.

## Performance

**15/16 groups meet the unchanged gates; NVIDIA worker_prepare fails.** The final series declares 140 fresh-process AB/BA pairs / 280 processes from the outset: **20 pairs per adapter for both dynamic_control and filter_cache**, and 5 pairs for every other case. No extension or early stopping is used for this source. Each run uses 120 warmup + 1200 measured frames, full frame-linked GPU coverage, identical settings/workloads/features and completed throughput including final drain. No builds, tests or soak from this task ran concurrently; ordinary desktop applications remained running. Other applications were not continuously monitored throughout the matrix. These are deterministic native-perf workloads, not official-server game acceptance.

GPU targets require at least 10% GPU P95 and 5% throughput improvement. Worker targets require at least 10% preparation P95 and 5% throughput improvement. CPU/GPU P95/P99 regressions must be at most 3%; every pair's total peak RSS growth must be at most 5%. All original failed runs remain retained; no outliers were removed.

| Device / case | Throughput gain | CPU P95 / P99 regression | GPU P95 / P99 regression | Preparation P95 gain | Worst-pair peak RSS growth | Passed |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| nvidia / dynamic_control | +1.72% | -2.46% / -1.33% | -16.97% / -18.43% | — | +0.70% | yes |
| nvidia / sparse_blend | +221.09% | -71.72% / -72.22% | -86.64% / -74.84% | — | -15.00% | yes |
| nvidia / alpha_mask | +73.83% | -39.47% / -38.02% | -59.28% / -32.57% | — | -2.80% | yes |
| nvidia / filter_cache | +9.29% | -3.29% / -2.18% | -35.04% / -10.46% | — | +1.31% | yes |
| nvidia / worker_prepare | +10.81% | -0.76% / +12.63% | -19.61% / -75.66% | +8.72% | -0.16% | NO |
| nvidia / static_tree | +169.15% | -60.49% / -57.22% | -68.40% / -13.03% | — | -7.20% | yes |
| nvidia / dirty_1_percent | +174.71% | -64.33% / -63.01% | -68.77% / -13.89% | — | -7.16% | yes |
| nvidia / dirty_all | +95.82% | -38.27% / -36.32% | -67.85% / -18.18% | — | -5.72% | yes |
| radeon / dynamic_control | +1.69% | -1.55% / -1.27% | -47.37% / -48.96% | — | +1.56% | yes |
| radeon / sparse_blend | +183.09% | -59.94% / -54.44% | -70.15% / -70.30% | — | -2.90% | yes |
| radeon / alpha_mask | +103.01% | -52.10% / -42.70% | -52.75% / -54.91% | — | -6.57% | yes |
| radeon / filter_cache | +10.42% | -12.50% / -15.31% | -36.45% / -41.85% | — | +1.30% | yes |
| radeon / worker_prepare | +15.77% | -12.43% / -8.58% | -11.56% / -12.16% | +16.92% | +4.81% | yes |
| radeon / static_tree | +90.51% | -47.07% / -41.64% | -66.78% / -66.22% | — | -3.17% | yes |
| radeon / dirty_1_percent | +65.41% | -38.86% / -36.07% | -66.13% / -65.58% | — | -2.90% | yes |
| radeon / dirty_all | +53.29% | -26.68% / -19.94% | -66.71% / -65.74% | — | -2.13% | yes |

The earlier 5778e349 matrix passed 14/16; its fixed 20-pair extension still failed NVIDIA dynamic_control CPU P99. Caching the live health latch at d9fd2d8 produced a 16/16 matrix. The later bc72719557 matrix originally passed 15/16, then passed all groups with a declared fixed filter extension retaining all original samples. Its full 30-minute soak subsequently exposed unreclaimed profiler pools. The final matrix above reruns all groups after fixing that defect; earlier successful results are not substituted for final-source evidence. The final NVIDIA worker group has +10.81% throughput but only +8.72% preparation P95 improvement and +12.63% CPU P99 regression; its unchanged thresholds require >=10% preparation gain and <=3% CPU P99 regression. This failure is not waived.

A single predeclared 64-shape partition experiment retained all samples and worsened preparation P95 by 20.40%/26.20% against the 32-shape source; it was rejected and production restored exactly. A separate frozen before/after-profiler comparison showed severe drift in both sources (old NVIDIA completed-frame time 3.764–8.852 ms). CPU pressure and rustdoc activity were observed immediately after that collection, but process ownership/start time was unavailable; these observations neither establish the cause nor erase the failed performance results. All diagnostic binaries, patches and raw traces remain available. No further parameter sweep or sample extension was used.

## Lifetime and memory

Final paced load/unload soak: **108060 measured frames, 1801.000 seconds, 100% GPU coverage, zero dropped samples and no terminal failure**, using the frozen final executable. It includes **600 normal-update GC/reclaim checkpoints** with the player, renderer and cache owners still alive.

At every checkpoint, active bytes are exactly **3754092**, live resource count exactly **305**, and reserved/idle/retiring bytes are zero. Checkpoint RSS ranges from 201658368 to 293113856 bytes. The kernel-reported process VmHWM is 292093952 bytes; procfs RSS/HWM sampling is approximate and is not exact driver accounting. The exact live-resource plateau assertion passes.

After full unload/drain/purge, unloaded-owner resources are zero; remaining ResourceIds exactly equal the warmed Descriptor baseline (295 resources, 1004 logical bytes). The old full-drop-only soak is not used as proof of a live-player plateau.

The earlier bc72719557 soak completed 108060 frames / 1800.999905990 seconds with full GPU coverage and exact final unloaded identities, but **failed** the live plateau gate: at frame 63540, one cached profiler pool added 1920 bytes / 3 resources. That failure and its raw traces are retained. The live-profiler purge regression failed first, then passed after routing completed pools through the shared idle governor. The final soak above verifies the fix over a fresh complete interval.

Logical storage estimates, driver allocated/reserved bytes and RSS are reported separately in the raw soak analysis. Neither logical accounting nor allocator reports are hardware capacity, and allocator reservation is not treated as an orphan leak.

## Remaining limitations

- NVIDIA worker preparation and CPU P99 performance gates fail in the final matrix; reliable acceptance of this group remains open.
- Golden-image mismatches and the variable Stage3D counter case above remain unresolved baseline compatibility/fixture limitations. Seven selected cases are ignored.
- Android ARM64 has compilation evidence only. Android real hardware, Windows/DX12 and macOS/Metal runtime/build acceptance are unavailable; no device recreation/fallback claim covers those gaps.
- Remaining Web compatibility is deliberately outside scope. Official-server Aola gameplay performance and universal OS-OOM survival are not claimed.

All delivery commits remain on `codex/native-wgpu-renderer`; no merge or push has been performed.

## Recorded decisions and deferred findings

Chronological ledger entries follow; later entries supersede earlier implementation-stage choices. The fresh review reported no additional actionable Minor finding.

- Ruling: Use isolated managed parent worktree and referenced local submodule clone — original checkout has overlapping edits and active performance validation — integration must preserve independent commits.
- Ruling: Baseline Ruffle is 989100c1353391f0abe756802752080bf96bedba, parent68325477e6739ed7f47410d23fdd2ba92e051b05; preexisting parent render_profile diff is copied for context and not committed as our work — freeze current state without mutating the other task — later parent integration must stage only our changes.
- Ruling: Add structured RenderFailure types to ruffle_render during Task2, before propagation in Task3 — the registry needs the same portable error identity as core/host — no new runtime behavior in other backends yet.
- Ruling: Bound metadata as well as bytes with262144/131072/65536 device-tier resource records, including reservations — zero-byte GPU views/bindings otherwise evade byte limits — unusually object-heavy scenes can reach this admission cap first.
- Ruling: Keep existing void display traversal interfaces during error propagation, record terminal allocation failures in shared RendererHealth, and reject the entire command list before submission — rewriting every display visitor before the approved snapshot stage would add unnecessary churn; explicit BitmapData/VM operations still return errors immediately — cost if wrong: traversal can spend CPU until the frame boundary after a fatal error, but cannot publish a partial frame.
- Ruling: Remove WGPU's legacy oversize bitmap rescaling and reject oversize viewport/bitmap requests — the approved pixel-fidelity contract forbids silently shrinking images — formerly degraded oversized content now terminates with diagnostics.
- Ruling: Adapt wgpu-profiler0.28 source into a licensed native module — upstream hides raw query/readback allocation and silently drops pending frames, violating accounting and sample completeness — maintenance now includes tracking upstream profiler fixes; Tracy remains optional and compiles.
- Ruling: During allocation integration, conservative encoder leases retain all live resources and newly created objects until submission; exact graph use sets replace this in Tasks7–8 — prevents premature refund during migration — temporary retention overestimates memory until graph integration.
- Ruling: Extend Task4 to Stage3D index/vertex creation and desktop Tracy/surface failure propagation — these were remaining allocation/panic boundaries uncovered by the resource audit — shared interface churn and optional Tracy uses bounded capture storage.
- Ruling: Make only the concrete native bitmap cache payload and its optional bitmap counters synchronized; keep public BitmapHandle/ShapeHandle traits and renderer/GC ownership unchanged — device-wide weak reclaim hooks must safely evict caches across native contexts — adds narrow synchronization to four lazy bindings and bitmap counters, without allowing worker access to renderer handles.
- Ruling: Capture the existing post-pre-render CommandList into owned numeric RenderSnapshot with a separate main-thread handle table — this boundary already follows AVM/GC events and preserves static command snapshots, so no second display-tree walk/cache is added — Task7 must consume this prepared snapshot, rather than recapturing GC state.
- Ruling: Use exact bit coverage for fragmented deferred bitmap patches rather than a bounding-box upload — a union rectangle could overwrite authoritative GPU pixels with stale CPU holes — fragmented patterns cost at most one bit per pixel and more upload regions until Task11 slicing.
- Ruling: Preserve ancestor-Layer sampling scopes as indivisible preparation groups while bounding their admission with the hard device budget — existing Alpha/Erase pixel regressions depend on this preparation order; splitting it would change pixels — those uncommon groups can retain more intermediates and fail admission earlier than independent sibling groups.
- Ruling: Fuse ColorMatrix chains only when every eliminated intermediate is an exact RGB channel permutation with unchanged alpha; retain all other attachment boundaries — real Vulkan half-step and low-alpha regressions prove WGSL pack4x8unorm cannot reproduce native attachment conversion, and portable rounding is not guaranteed — fewer chains fuse, while zero-tolerance pixels are preserved. The earlier test's blanket-finite eligibility assertion and guessed half-step constant were corrected; differential pixel assertions remain exact.
- Ruling: Use the existing adjacent chunk encoder as compatible draw-pass merging; filter sampling boundaries remain distinct except proven pointwise groups — a filter output is the next filter's sampled input and cannot be merged merely because dimensions match — no general render-pass concatenation across hazards.
- Ruling: Bound temporary materials to128 entries and charge their unique transitive storage as a conservative cache-retention weight; clear shared material bindings when a renderer is dropped — existing teardown tests caught cached bindings retaining orphaned textures in longer-lived Descriptors — shared sibling renderers may rebuild bindings after a peer unload, but no GPU contents or submitted work are evicted.
- Ruling: Existing staging accounting test now expects the full64-byte page for a16-byte slice and explicitly purges the completed idle page before expecting zero — paged storage must charge actual capacity and remain reusable after completion — retained idle pages raise the active baseline within the same shared idle/staging budgets.
- Ruling: Share exact pixel coverage between lazy native uploads and BitmapValidity, promoting a rectangle to a bounded one-bit-per-pixel mask only on fragmentation — this avoids coalescing CPU holes over authoritative GPU pixels, so no synchronization is needed merely to merge dirty regions — fragmented full reads/uploads may require several region operations; coverage storage is bounded by bitmap dimensions.
- Ruling: Keep Flash's down-left overlapping self-copy on the existing CPU row-order implementation; use budgeted native temporaries only for overlap cases with exact memmove semantics — the existing Flash-compatible CPU implementation intentionally reads overwritten rows in this quadrant — these unusual copies retain CPU readback cost rather than changing observable pixels.
- Ruling: Cache rasterization uses native device/budget admission instead of the previous2/4-million-pixel uncached fallback — required filters and cache pixel snapping must not disappear under pressure — an oversized required raster now terminates through the typed renderer error instead of displaying altered pixels.
- Ruling: Retain at most4096 numeric cache-history entries and2MiB of history, charged to the shared cache governor; unknown/full bounds, stencil/ancestor-Layer/Stage3D/shader scopes and unsupported filter dependencies redraw the entire valid cache — only independently bounded dependencies are cropped — these complex scopes and large histories retain full redraw cost.
- Ruling: Native preparation reserves bounded waves on the existing shared Rayon pool and caps generated mesh output before Lyon buffer growth (including spare capacity and packed vertices); per-shape allowance is one quarter of native staging policy, capped at16MiB — generated curve/intersection output cannot be bounded from input counts alone — pathological mandatory geometry exceeding that allowance returns typed BudgetExceeded, never a partial mesh. Pure CPU preparation remains owner-thread serial when the workload is small or no shared permit fits. Third-party tessellator internal sweep scratch is estimated from immutable input, not reported as exact allocator accounting.
- Ruling: Performance admission requires full frame-linked GPU coverage, driver/workload/settings identity and unique fresh-process identities for every pair; any pair's process peak-RSS growth above5% fails rather than disappearing in a median — incomplete or heterogeneous traces are diagnostic only — noisy outlier pairs can require a fresh controlled experiment, never a relaxed gate.
- Ruling: Partition at most16 small native shapes per CPU task and share one output budget across the partition; if a partition fills, publish its completed results and prepare the remainder individually on the owner — the real worker corpus exposed avoidable per-shape Rayon wave overhead — unusually large shapes retain serial preparation cost but never change pixels or GPU publication order.
- Ruling: Keep established displacement-map tolerances unchanged and classify four NVIDIA/Vulkan differences as preexisting compatibility gaps — exact baseline989100c135 also fails all four cases and produces byte-identical actual PNG files (task-16-baseline-displacement.json plus image-evidence/) — full Flash pixel conformity for those four cases is not claimed. Additional native backends will be exercised separately.
- Ruling: Replace the per-shape quarter-staging limit with exclusive access to the full existing preparation allowance after ordinary worker waves finish — a legitimate existing SWF exceeded8MiB output on integrated GPUs — unusually large meshes take serial preparation and may still return BudgetExceeded at the full allowance; third-party internal scratch remains an input-based estimate.
- Ruling: Vendor the unmodified published wgpu-hal30.0.1 package plus one native capability correction, applied by both Cargo roots — apitrace proves no glUniformBlockBinding calls although GLSL330 omits explicit bindings; compute extensions incorrectly enabled SHADER_BINDING_LAYOUT — use Naga's actual supports_explicit_locations() predicate, preserving wgpu30.0.1 and shader arithmetic; maintain this small dependency patch until upstream includes it. Provenance and removal condition are in render/wgpu/vendor/wgpu-hal/RUFFLE-PATCH.md.
- Final: Ruling: Pristine vendored wgpu-hal code remains the upstream published30.0.1 implementation — only the authored capability predicate/provenance/Cargo patch is in review scope — upstream defects remain dependency risk, not a fresh full-vendor audit.
- Final: Ruling: Preserve the21 independent unstaged parent render_profile.rs lines and exclude them from commits — they are unrelated user work — no review/acceptance claim for those lines.
- Final: Ruling: Remaining Web compatibility stays unverified — user explicitly requested native-only — this branch cannot be represented as a supported Web renderer.
- Final: Ruling: Android real-device, DX12 and Metal runtime remain not verified because hardware is unavailable — cross-compilation does not establish runtime behavior — those platforms need future hardware evidence before acceptance.
- Final: Ruling: OS process OOM survival and exact driver allocation remain outside the contract — checked logical GPU admission and typed failures cannot control OS termination or all driver storage — report allocator/RSS separately without promising universal OOM survival.
- Final: Ruling: Retain documented baseline-identical Flash failures with unchanged tolerances — exact reference binaries/PNG hashes are preserved and source review does not regenerate them — no claim of full Flash compatibility for those cases.
- Final: Ruling: Post-review changes are covered by author RED→GREEN plus affected suites, not another review — executing-specs requires one fix pass and forbids re-review — subsequent edits have no independent source-review claim.
- Ruling: Replace global strong movie/library and SymbolClass indexes with GC-owned edges and weak lookup indexes — the explicit live-load reclamation test proved that the global maps rooted otherwise unreachable movies/classes — intentional retained classes still retain their defining movie; wrong edge migration could lose definitions, so real Arena tests and the expanded loader/import/mixed-AVM suite cover the transition.
- Ruling: Native shape partitions now hold32 small shapes and estimate each partition output as min(output_budget, input_estimate/4+4096), with the existing exclusive owner-thread fallback after all waves — attribution showed scheduling/over-reservation dominated small mesh preparation — underestimates cost serial re-preparation but retain checked output limits and exact pixels. This replaces the earlier16-shape partition decision.
- Ruling: Let the owner thread execute one admitted preparation task and use Rayon in_place_scope for the rest — the owner otherwise blocked while the scope body moved to a worker — all tasks join in-frame and permit ownership persists through publication; dispatch counters now count only actually spawned tasks.
- Ruling: Start each new formal comparison series with fresh frozen executable paths for BOTH baseline and candidate — prior experiments reused a warmed baseline name against a cold candidate name — preserve all earlier failed runs and all first-process samples, keeping the original every-pair5% peakRSS gate. This controls the demonstrated naming/cache asymmetry without changing drivers, global cache settings or thresholds; subsequent cases share the same binary paths equally.
- Ruling: Raise small-partition output allowance from input_estimate/4+4096 to input_estimate+4096 under the same hard output and shared scratch caps — timers showed330 tessellations for320 shapes per frame, with every partition overflowing and serially redoing its tail — a larger estimate can reduce concurrency on large inputs, but bounded fallback and exact output remain unchanged.
- Ruling: Defer MSAA restoration until geometry/stencil/shader work needs sample storage, retaining the exact list of changed rectangles — resolving the full stage for every tiny pointwise blend dominates sparse GPU time — restoring a bounding union would corrupt unrelated AA samples; exact Low/High pixel comparison against the stencil path covers overlapping blends and later translucent geometry. Performance RED is the existing failed sparse matrix; the new pixel guard is supporting correctness, not falsely claimed as a feature RED.
- Ruling: Intern at most1024 small solid immutable meshes (exact encoded key<=512bytes), with weak mesh references and periodic expired-key pruning — independent identical quads otherwise cannot enter existing adjacent instance batches — reject larger/gradient/bitmap meshes to bound CPU identity storage and hashing cost; index entries never keep GPU storage alive. Resource ownership token belongs to the shared immutable graphic allocation while any referencing recipe is alive; final unload still requires exact warmed IDs.
- Ruling: Apply the identical gpu_metrics.rs, GpuSample phase storage/recording and native-perf GPU phase serialization to the archived baseline before the final comparison — the contract requires common measurement instrumentation — this is an additive harness-only baseline correction, not an optimization or tolerance change. Keep original frozen baseline and all old evidence, save baseline-common-gpu-phases.patch, and rerun fresh-process pairs with both executable paths cold. No renderer algorithms or workload changed in the baseline.
- Ruling: Cache at most64 fixed-binding bitmap groups under the shared cache byte governor and source-lifetime invalidation — global strong material entries keep evicted images resident; per-source weak registrations immediately drop cached bindings when any source expires, while recorded command clones remain valid — adds bounded key/token metadata, no cross-frame image ownership.
- Ruling: Reuse exact single-solid-mesh raster inputs within one bounded sibling set (at most64 keys); key includes immutable vertex/index identities and ranges, complete target-local transform, clear value and target size — repeated equivalent blend inputs need only one rasterization, with every ordered composite preserved — excludes shaders/perspective/non-solid/multiple draws, never persists across frames, and shared pool ownership prevents reuse before the last command.


## Manual-session fixes and investigation (2026-11-18)

Evidence is under `.ryugi/sdd/2026-11-17-native-wgpu-renderer-design/manual-fixes-20261118/`.
This follow-up repairs the scalar readback amplification and Linux/Android RSS reader. It does not close the original Task16 performance, image or platform gaps, and does not establish the same official-game interaction sequence after the fix.

### Scalar readback

`getPixel` and `getPixel32` retain exact synchronization for the first isolated dirty pixel. A second dirty read in the same 32x32 tile enables bounded prefetch, clipped to the bitmap. Only GPU-authoritative coverage is copied, preserving CPU-written holes; a GPU write resets the locality hint. Each tile contains at most 4 KiB of logical pixel data. Other area reads, aliased hit tests and copy operations retain their exact coverage behavior. Calls that need GPU data are still synchronous; this amortizes their queue/map waits.

The patterned 130x126 core regression failed with 16380 synchronizations before the change and passes after it, checking every pixel, the synchronization bound, and the exact 65520-byte payload. The second regression checks isolated reads, the 4 KiB bound, CPU overwrites, distant untouched pixels and subsequent GPU writes. An early broader implementation broke two existing aliasing/hit-test tests; restricting prefetch to scalar getters fixed them without changing their assertions.

The real SWF scanner verifies all 16380 pixels against four changing quadrant colors. Its constructor consumes the first regression frame, so the fixture runs four frames to preserve all three expected scan results. Twelve selected bitmap/readback/copy/mask regression cases pass on each actual NVIDIA and Radeon Vulkan adapter, with unchanged existing image tolerances.

The actual Slint host completed 361 submitted frames / 360 scans. Every scan used **22 readbacks and 65520 bytes**, with no failed pixel oracle in the regression runner. Completed-frame diagnostic comparisons use identical settings and input digest `888755a29b9cb51d99423d6a06816458edf5da9f584e47370806264066d3e813`, two warmup and eight measured frames per process:

| Actual Vulkan adapter | Retained pairs | Baseline mean ms/frame | Candidate mean ms/frame |
| --- | ---: | ---: | ---: |
| NVIDIA RTX 5060 Laptop | 4 | 820.628552 | 4.221764 |
| Radeon 780M / RADV | 2 | 1106.009009 | 6.204547 |

These are short targeted diagnostics, not the formal performance matrix or official-game FPS. All measured GPU samples match their frame IDs and none were dropped. Logical pixel copy volume is unchanged. The cached readback-buffer storage after drain increases from 1024 to 16896 bytes in this fixture; all readback buffers are reclaimed after unload/purge.

Measurement corrections are retained: the initial candidate builds enabled the tests package's default `avm_debug` feature and are excluded from equal-setting comparisons. The final candidate uses `--no-default-features`. `WGPU_ADAPTER_NAME` did not select the requested adapter in this harness; four runs labelled Radeon actually used NVIDIA and remain included as NVIDIA data. The real Radeon AB/BA runs use `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json`, with device identity asserted from the session record.

Frozen baseline SHA256: `30aff5b47527db290f0f6135eb0f19bb2e113999240e7a840d8a9678bb814df3`.
Frozen candidate SHA256: `7a301fc5105593b3af9a93616ea73ba0da3f19c0eff49ac38d9fe047fe9c8073`.

### RSS and slow-GC telemetry

Linux/Android now reads approximate `/proc/self/status` `VmRSS` rather than scanning `smaps`. It validates the field, units, duplicates and overflow; missing/unreadable values stay missing, with no expensive fallback. The shared background sampler, 500 ms cadence and nonblocking shutdown remain. Session metadata records `rssSource=procStatusVmRss`; other platforms retain `memory-stats`.

Two isolated local Slint-host runs produced 24 RSS samples: sampling medians 61/68 microseconds, maximum 103 microseconds. Sample peaks were 387084288 and 634331136 bytes, both within independently polled kernel VmHWM. These are small controlled runs, not an equivalent-load comparison with the earlier 507-second official session; VmRSS remains approximate. The desktop binary SHA256 is `eb5493a973b17029948a8443919f1e53f0d8c27fe3f1d9abf49519cc5455e5a4`, including the preserved independent parent profiling diff.

GC pauses of at least 8 ms now produce bounded numeric-only samples containing before/after phase, object count and allocation debt. At most 32 samples await draining; overflow is counted. Profiling-disabled execution does not collect these samples. The host emits frame-associated `gc` events and drains again before its end record. No collector scheduling or pacing was changed.

An isolated allocation/release SWF completed 181 stage submissions, producing ten slow-GC samples without drops. The longest collection took **200.814034 ms**, transitioned marking to sweeping, reduced allocated GC objects from 1125500 to 698551, and paid debt from 399993 to zero. The first diagnostic version stopped mutating visible content and could not satisfy the host's submitted-frame exit condition; its timed-out, truncated trace is retained separately. The final fixture keeps a small shape animated and exits normally.

`gc-arena 0.7.0::Arena::collect_debt()` continues until allocation debt is paid, without a time deadline; one VM update can therefore impose a long collection. Large `ByteArrayStorage::from_vec` inputs also add debt. The diagnostic confirms the mechanism, not the cause of the earlier game's individual pauses. GC object counts include allocations not yet freed, not only reachable objects; debt-driven partial collection can retain garbage after allocation stops. A time-budgeted collector would require a separate correctness/memory tradeoff, not simply clearing debt.

### Resource holding and validation

Source inspection confirms weak movie/library indexes, weak network transfer groups with removal on completion/drop, and a 32 MiB shared network-body allowance. Application-domain definitions/classes intentionally retain scripts, translation units and defining movies; `ClassData::symbol` also retains its defining movie. These are candidate holding paths for the official game's live movies, not evidence of an erroneous root. Existing real-Arena tests prove both unreachable movie reclamation and intentional class retention.

A fresh paced NVIDIA load/unload diagnostic ran **1800 measured frames / 29.999722 seconds**, with 1800 matching GPU samples and ten explicit GC/drain/purge checkpoints. Every checkpoint has exactly 305 resources / 3754092 active bytes, with reserved/idle/retiring zero. After dropping the player and purging, unloaded-owner resources are zero and the remaining 295 resource identities exactly match warmed descriptors. This is a short follow-up with explicit reclamation, not a new 30-minute soak or proof of natural official-game unload.

Validation: core 250 passed / 9 existing ignored, render 65 passed, host 30 passed / 1 existing ignored; the stricter 4 KiB scalar-read regressions pass separately. The desktop release builds successfully. Twelve selected native bitmap regressions pass per GPU. Existing warnings remain; strict Clippy and other platforms were not rerun. The unrelated 21-line parent `render_profile.rs` diff remains untouched and excluded from commits.

Remaining: the official game's repeated-scene ownership baseline and post-fix frame/RSS trace; its long GC and socket-handler pauses; the original Task16 acceptance gaps. No new engine leak or safe GC scheduling repair is claimed by this follow-up.
