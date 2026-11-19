# Incremental stage display-list rendering

At `bdd30fe1f`, AMD Radeon 780M passed 5/8 performance cases and NVIDIA GeForce RTX 5060 Laptop passed 5/8 performance cases. Stable frames recaptured zero command leaves, down from 2,000. The filter-cache, alpha-mask and load/unload controls exceeded timing limits on both adapters. Both matched the reference images. The 30-minute resource test passed on Radeon.

The stage publishes a render snapshot after the Flash render event. ActionScript child edits take effect immediately. Event handlers observe the current logical order through `numChildren`, `getChildAt()` and `getChildIndex()`, including during reentrant `ADDED` and `REMOVED` callbacks. An iterator holds the order it started with. Lists up to 64 children fit in one leaf. Larger lists copy the affected path through a ranked tree with up to 16 children per internal page.

The render state records local transform, color, visual content, visibility, mask, filter and child-order changes. An epoch identifies the child order used by each container index. A separate subtree generation invalidates enclosing snapshots. Exact world matrices retain Flash's twip rounding. Sparse changes patch painter-ordered pages, including the complete affected timeline-mask scope. Retained leaves keep their device-space coverage, so rebuilding a page can reuse its unchanged siblings' bounds.

Native WGPU replays validated resident geometry and material draws with per-submission uniform ranges. A draw spanning a uniform-window boundary keeps its painter order. Raster cache wrappers check residency and mutable bitmap inputs are prepared for the current frame. Dynamic text, video, Stage3D, custom shaders and independent offscreen rendering use the live path as needed. The stage checks a bounded number of weak cache records on warm frames and can sweep the complete registry under budget pressure. Retained command storage, spatial and directory metadata and native prepared packets share the 24 MiB stage allowance.

Local and inherited color edits update the leaf color while preserving its world matrix and exact coverage. Cold timeline shape draws can stream once before the stage admits their resident handles. A Loader streams newly attached content for its first stage submission. Once the renderer has prepared the geometry, the next render may retain its commands. Offscreen draws and mask geometry use their own preparation paths.

## Measurement conditions

Baseline `7aa47a3c7d220793023ac29fdbb2fbdc4e68feba` freezes the concurrent native renderer work common to both builds. The baseline has a measurement-only patch. Candidate source is `bdd30fe1f71b527407f5c14d6fd1f3c761ef3852`. The two executables use the same dist build profile, runner, fixtures and settings.

Executable SHA-256 values are `bd32506657b3930e0b7912e50644f0ddc5ccf1aea4c67576c845f7f54802cf3a` for the baseline and `a4520496f43dcb51c4ff2cf42f35a16d5d3cb25881eca6d9fe58c4ea7ae3822c` for the candidate. The workload manifest has SHA-256 `6b84027dec331eaaa873808848112f75caeebb670c2408e3e4b950941b7a4d35`.

Linux 7.3.4-1-cachyos and a Ryzen 7 H 260 ran both Vulkan series in the performance power profile. Radeon used CPUs 0 and 8 on one physical core. NVIDIA used CPUs 1–7. These affinities allow one and four native preparation workers respectively. The runs used a 1280 × 720 viewport, high quality, 4× MSAA, two frames in flight and GPU timestamps. Readback, debug assertions and AVM debug were disabled. During the Radeon comparison, another paced NVIDIA workload used CPUs 1–7. The devices shared the laptop’s power and thermal limits. The full Radeon comparison finished before the Radeon resource test. The NVIDIA launcher checked for competing benchmark and compiler processes before each run and allowed the paced Radeon test on the other adapter.

An initial NVIDIA baseline trial overlapped another native benchmark for 1.20 seconds and was archived before completing a pair. During the fresh series an observer on CPUs 0 and 8 checked for competing benchmark processes throughout each timed run. It recorded 80 uncontended processes and 0 additional excluded trials.

Each case used five fresh-process baseline/candidate pairs in alternating order, with independent cold shader caches, 120 warmup frames and 1,200 timed frames. All 160 accepted processes supplied a GPU timestamp for every measured frame. Absolute values in the tables are medians across each build’s five runs. Percentage changes are medians of within-pair ratios. RSS is the largest increase in peak process RSS across the five pairs.

The static target is native `commandPreparation`. Sparse targets use core `renderTree`. The four target cases require at least 10% phase P95 improvement and 5% higher completed-frame throughput. The four controls allow completed-frame time to increase by at most 3%. Every case limits each CPU/GPU P95/P99 increase to 3% and each pair’s peak RSS increase to 5%.

## AMD Radeon 780M

Vulkan driver `Mesa 26.2.6-arch3.1`. **5/8 cases passed every applicable gate.**

| Case | Phase P95, µs | Phase gain | Completed ms/frame | Throughput gain | CPU P95 / P99 change | GPU P95 / P99 change | Worst RSS change | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `static_tree` | 1091.24 → 5.39 | +99.51% | 1.885 → 0.344 | +442.72% | -84.34% / -81.22% | -27.54% / -20.18% | +1.41% | Pass |
| `dirty_1_percent` | 35.94 → 27.18 | +24.59% | 1.929 → 0.401 | +367.04% | -79.64% / -76.84% | -24.75% / -15.31% | +2.54% | Pass |
| `dirty_scattered` | 163.93 → 72.14 | +64.44% | 1.949 → 0.481 | +321.19% | -74.74% / -73.24% | -20.87% / -15.64% | +4.04% | Pass |
| `wide_single_edit` | 256.81 → 9.56 | +96.09% | 1.877 → 0.361 | +420.02% | -82.28% / -82.51% | -27.87% / -23.69% | +2.82% | Pass |
| `dirty_all` | 335.02 → 257.79 | +16.45% | 2.578 → 1.444 | +79.80% | -29.99% / -30.67% | -1.60% / -0.82% | +0.39% | Pass |
| `filter_cache` | 10.22 → 9.40 | +5.81% | 0.135 → 0.140 | -4.42% | +8.80% / +18.11% | +0.34% / -0.08% | +3.18% | Miss |
| `alpha_mask` | 18.47 → 22.51 | -24.57% | 0.619 → 0.662 | -6.96% | +8.52% / +0.40% | +4.94% / +4.06% | +2.86% | Miss |
| `load_unload` | 35.32 → 39.07 | -11.50% | 0.295 → 0.290 | +2.09% | +2.27% / +4.03% | -1.64% / -0.98% | +3.12% | Miss |

Thresholds exceeded in this series

- `filter_cache` exceeded its gate on completed frame time +4.63%, CPU P95 +8.80%, CPU P99 +18.11%.
- `alpha_mask` exceeded its gate on completed frame time +7.48%, CPU P95 +8.52%, GPU P95 +4.94%, GPU P99 +4.06%.
- `load_unload` exceeded its gate on CPU P99 +4.03%.

## NVIDIA GeForce RTX 5060 Laptop

Vulkan driver `620.70.06`. **5/8 cases passed every applicable gate.**

| Case | Phase P95, µs | Phase gain | Completed ms/frame | Throughput gain | CPU P95 / P99 change | GPU P95 / P99 change | Worst RSS change | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `static_tree` | 709.27 → 4.06 | +99.41% | 1.162 → 0.306 | +283.99% | -77.43% / -79.06% | -15.20% / -51.88% | +0.90% | Pass |
| `dirty_1_percent` | 24.88 → 13.48 | +45.83% | 1.141 → 0.336 | +232.94% | -72.56% / -75.14% | -15.87% / -52.29% | +0.91% | Pass |
| `dirty_scattered` | 130.39 → 41.48 | +66.82% | 1.190 → 0.371 | +217.10% | -72.68% / -72.28% | -15.83% / -51.25% | +1.63% | Pass |
| `wide_single_edit` | 184.96 → 6.51 | +96.48% | 1.136 → 0.308 | +266.54% | -73.55% / -69.83% | -14.95% / -51.66% | +0.91% | Pass |
| `dirty_all` | 258.34 → 271.04 | -4.90% | 1.878 → 1.315 | +42.09% | -22.58% / -20.20% | -1.07% / +0.00% | +0.58% | Pass |
| `filter_cache` | 10.52 → 13.87 | -35.56% | 0.109 → 0.116 | -6.60% | +17.93% / +52.20% | +1.11% / -5.31% | +0.91% | Miss |
| `alpha_mask` | 17.64 → 17.93 | -5.81% | 0.486 → 0.494 | -1.73% | +7.18% / +6.32% | +0.44% / +0.37% | +1.70% | Miss |
| `load_unload` | 31.22 → 35.23 | -12.84% | 0.214 → 0.208 | +3.62% | +4.24% / +3.31% | +0.00% / -31.89% | +1.37% | Miss |

Thresholds exceeded in this series

- `filter_cache` exceeded its gate on completed frame time +7.06%, CPU P95 +17.93%, CPU P99 +52.20%.
- `alpha_mask` exceeded its gate on CPU P95 +7.18%, CPU P99 +6.32%.
- `load_unload` exceeded its gate on CPU P95 +4.24%, CPU P99 +3.31%.

## Work per measured frame

The per-frame medians shown here were the same for both adapters. Each entry gives the baseline and candidate values.

| Case | Tree visits | Slots examined | Instances updated | Leaves captured | Prepared packet replays | Render passes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `static_tree` | 1.0 → 1.0 | 1.0 → 1.0 | 0.0 → 0.0 | 2000.0 → 0.0 | 0.0 → 1.0 | 4.0 → 4.0 |
| `dirty_1_percent` | 63.5 → 3.0 | 111.0 → 23.0 | 0.0 → 20.0 | 2000.0 → 0.0 | 0.0 → 18.5 | 4.0 → 4.0 |
| `dirty_scattered` | 842.5 → 22.0 | 871.0 → 61.0 | 0.0 → 20.0 | 2000.0 → 0.0 | 0.0 → 69.8 | 4.0 → 4.0 |
| `wide_single_edit` | 2002.0 → 2.0 | 2001.0 → 2.0 | 0.0 → 1.0 | 2000.0 → 0.0 | 0.0 → 22.0 | 4.0 → 4.0 |
| `dirty_all` | 2002.0 → 2002.0 | 2001.0 → 2001.0 | 0.0 → 0.0 | 2000.0 → 0.0 | 0.0 → 0.0 | 4.0 → 4.0 |
| `filter_cache` | 34.0 → 34.0 | 33.0 → 33.0 | 0.0 → 0.0 | 32.0 → 32.0 | 0.0 → 0.0 | 1.0 → 1.0 |
| `alpha_mask` | 74.0 → 74.0 | 49.0 → 49.0 | 0.0 → 0.0 | 48.0 → 48.0 | 0.0 → 0.0 | 25.0 → 25.0 |
| `load_unload` | 47.1 → 14.8 | 46.1 → 19.4 | 0.0 → 0.0 | 38.5 → 3.8 | 0.0 → 0.8 | 2.8 → 2.8 |

The baseline already visited one display-tree node per measured frame in the static case. It then captured 2,000 command leaves for native preparation. The candidate replayed the retained block with zero captured leaves. In the wide case the script moves one of 2,000 direct children each frame. The candidate visits two nodes and examines two child slots. Separate structural tests swap distant children of a 512-child container while holding an old iterator and limit copied child entries to 128. A 1,400-child stress case compares successive logical orders with a reference vector.

In the Radeon series, the filter and cached alpha-mask workloads still prepared 32 and 48 bitmap command leaves per frame respectively. Their pass counts were one and 25 for both builds. Bitmap residency and the mask operands are checked on the live path.

The load/unload workload replaces its child SWF every 12 frames. The candidate can retain geometry after its first timeline submission. The work tables include cold submissions, warm draws and the empty frame during replacement.

## Display-list and pixel regressions

The core, render and WGPU suites passed 276, 68 and 42 tests respectively. All 86 adapter-backed tests passed on Radeon Vulkan with `wgpu/counters` and four logical CPUs. An earlier two-CPU run gave the preparation pool one worker and the parallel-work test failed its assertion. The paired-metrics analyzer passed its 13 tests.

The candidate passed 333 distinct unmarked SWF regressions covering display lists, bitmaps, loaders, blending, scroll rectangles, masks and filters. Two additional stage-quality variants passed and 24 cases matched the suite’s expectations for known failures. The suite ignored its two JPEG XR loader cases. The frozen baseline passed 175 overlapping unmarked regressions. The mixed SWF checked immediate indices and pixel reads during reentrant removal, bounds during addition, and reparenting while a listener shrank the destination. Its two frames include an external mask, scroll rectangle, mutable bitmap and filter removal. Both candidate frames matched the checked-in references with a maximum per-channel difference of zero on Radeon and NVIDIA. The baseline matched on Radeon.

## Resource lifetime

The soak moves an 80-shape static group next to its Loader at load completion and detaches it before replacement. The blend and alpha-mask objects stay inside the loaded content.

The same binary ran `cache_soak` on AMD Radeon 780M Graphics (RADV PHOENIX) with Mesa 26.2.6-arch3.1. It completed 108,060 measured frames in 1801.00 seconds at a target of 60 FPS. All 108,060 frame IDs matched GPU timestamps. The movie crossed 900 reload boundaries. GC, drain and reclamation ran every 180 frames for 600 checkpoints.

| Frame in 120-frame load cycle | Samples | Resources | Active bytes | Reserved / idle / retiring bytes | First / last quarter RSS medians, MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 300 | 311 | 3,753,316 | 0 / 0 / 0 | 81.72 / 82.14 |
| 60 | 300 | 309 | 3,753,252 | 0 / 0 / 0 | 81.72 / 82.14 |

The 180-frame checks alternate between positions 0 and 60 in the 120-frame load cycle. The cross-phase `stableResourcePlateau` flag is false because these positions have different live resource sets. Within each phase all 300 samples shared one resource signature. Checkpoint `VmRSS` ranged from 81.31 to 82.18 MiB. The final kernel `VmHWM` reading was 82.05 MiB. After final unload, 0 non-descriptor resources remained. The runner verified that all 297 remaining descriptor resource identities matched its warmed baseline.

## Scope and reproduction

These measurements apply to the native SWF fixtures. Official online Aola scenes and Android devices have no frame-time measurements in this series. Wide structural edits preserve the logical order and share unaffected pages. A render after an order change can still scan the container to reconcile child identities and painter order. Searching for a child by identity can also scan the logical order.

Raw JSONL, per-case comparisons, executable hashes, pixel logs and resource checkpoints are under `/home/starknt/.local/state/aola/verification/incremental-display-tree-20261119/component-update`. The frozen baseline instrumentation is in `/home/starknt/.local/state/aola/verification/incremental-display-tree-20261119/baseline-instrumentation.patch`. The earlier complete Radeon matrix and paced test at `1226caa7b` are in the parent directory.

In a checkout of the frozen baseline, apply the measurement patch from the evidence directory with `git apply "$EVIDENCE/baseline-instrumentation.patch"`. Its SHA-256 is `4e8193652c3158f9ed4e3630d0a7afb772cbc031c3d6d52dd6ef6a5e92703c5d`. Build `native-perf` in the baseline and candidate checkouts using the same Rust toolchain. Set `BASELINE` and `CANDIDATE` to their respective executable paths. Run the paired driver from the candidate checkout to use identical fixtures for each pair.

```sh
cargo build --locked -p tests --no-default-features --features imgtests --bin native-perf --profile dist

python3 tools/native_perf.py --baseline "$BASELINE" --candidate "$CANDIDATE" \
  --manifest tests/perf/native.toml --output "$OUT/static_tree" \
  --case static_tree --phase commandPreparation
```

Set `VK_DRIVER_FILES` to the selected adapter’s Vulkan ICD and use the CPU affinity listed for that adapter. Use `--phase renderTree` for each other row, with `--control` for the final four rows. For the paced resource test use the candidate binary with the same workload and GC/drain/reclaim cadence.

```sh
"$CANDIDATE" --manifest tests/perf/soak.toml --case cache_soak --backend vulkan \
  --warmup 120 --frames 108060 --pace-fps 60 --reclaim-every 180 --output "$OUT/soak.jsonl"
```
