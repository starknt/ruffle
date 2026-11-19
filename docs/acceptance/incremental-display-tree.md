# Incremental stage display-list rendering

At `3e20256be`, NVIDIA GeForce RTX 5060 Laptop passed 8/8 performance cases and AMD Radeon 780M passed 8/8 against frozen baseline `7aa47a3c7`. Both devices matched both checked-in frames of the mixed display-list scene exactly. The Radeon resource test completed 108,060 measured frames.

## Display-list behavior

The stage publishes a render snapshot after the Flash render event. ActionScript child edits take effect immediately. Event handlers observe the current logical order through `numChildren`, `getChildAt()` and `getChildIndex()`, including during reentrant `ADDED` and `REMOVED` callbacks. An iterator holds the order it started with. Lists up to 64 children fit in one leaf. Larger lists copy the affected path through a ranked tree with up to 16 children per internal page.

The render state records local transform, color, visual content, visibility, mask, filter and child-order changes. An epoch identifies the child order used by each container index. A separate subtree generation invalidates enclosing snapshots. Exact world matrices retain Flash's twip rounding. Sparse changes patch painter-ordered pages, including the complete affected timeline-mask scope. Retained leaves keep their device-space coverage, so rebuilding a page can reuse its unchanged siblings' bounds.

Native WGPU replays validated resident geometry and material draws with per-submission uniform ranges. A draw spanning a uniform-window boundary keeps its painter order. Raster cache wrappers check residency and mutable bitmap inputs are prepared for the current frame. Dynamic text, video, Stage3D, custom shaders and independent offscreen rendering use the live path as needed. The stage checks a bounded number of weak cache records on warm frames and can sweep the complete registry under budget pressure. Retained command storage, spatial and directory metadata and native prepared packets share the 24 MiB stage allowance.

Local and inherited color edits update the leaf color while preserving its world matrix and exact coverage. Cold timeline shape draws can stream once before the stage admits their resident handles. A Loader streams newly attached content for its first stage submission. Once the renderer has prepared the geometry, the next render may retain its commands. Offscreen draws and mask geometry use their own preparation paths.

## RTX 5060 filter-cache tail diagnosis

The original comparison at `bdd30fe1f` reported a 52.20% increase in CPU P99. The runner takes the nearest-rank CPU P99 of each 1,200-frame process, then the median of the five within-pair changes. The original traces give the following individual P99s.

| Original pair | Baseline, µs | Previous candidate, µs | Paired change |
| --- | ---: | ---: | ---: |
| 1 | 181.045 | 176.307 | -2.62% |
| 2 | 178.921 | 159.866 | -10.65% |
| 3 | 166.118 | 316.395 | +90.46% |
| 4 | 147.503 | 225.387 | +52.80% |
| 5 | 148.695 | 226.309 | +52.20% |

Pair 5 determines the reported median. Baseline frame 785 and previous-candidate frame 568 performed identical visible work. All 12,000 measured frames had 32 raster-cache hits, zero redraws, 32 captured bitmap leaves, two draws and one pass. In the two P99 frames, encoder finalization grew by 18.93 µs, queue submission by 5.86 µs, the remaining backend work by 22.94 µs, and command preparation by 2.61 µs. AVM2 added 10.93 µs, GC added 5.75 µs, display-tree rendering added 1.30 µs and other frame logic added 9.29 µs. These are disjoint portions of the measured CPU duration. The trace has no scheduler or clock samples that identify a single cause for the 77.6 µs difference.

A separate five-pair run of those same two executables in the performance power profile produced CPU P99 −1.80% and CPU P95 +5.41%. The exact historical tail ratio varied across runs. The repeated bitmap capture, validation and multisample stage composition were present in every old process and could be removed and checked independently.

## Changes exercised by the control scenes

- **Cached filters.** When a fresh stage contains only resident, pixel-aligned 1:1 cache images on the same device, WGPU preserves painter order and viewport culling and draws the resolved pixels directly into a single-sample RGBA8 or BGRA8 attachment. The cached source is still rendered at the requested quality. Overlapping images, pixel snapping, and redrawing an existing cache handle are compared with the general graph on real hardware.
- **Alpha masks.** A repeated single-image mask can reuse its original resolved RGBA8 pixels. Two identical placements admit the raster under the renderer's shared byte budget. Its key includes content version, output rectangle, quality, full transform and color, source rectangle, smoothing, and snapping. Writes and memory reclamation invalidate residency. An experimental shader shortcut differed by one channel level on Radeon at a translucent edge. The measured implementation preserves the original rasterization and rounding.
- **Loader churn.** On the first optimized RTX run, native command preparation was still about 210 µs on reload frames and about 3 µs on warm frames in the first candidate process. Between collections, the renderer rebuilt roughly 13 GPU buffers per reload while previous shapes were still live. The renderer now indexes 64 weak slots by character ID, checks complete immutable SWF definitions and exact scale before reuse, and excludes movie-local bitmap fills and strokes. Collection releases both the original definitions and the handles. A genuinely cold batch of at most 32 meshes shares the tessellator and scratch permit, with owner-thread validation and the existing exclusive path for oversized input.

## Measurement conditions

Baseline source `7aa47a3c7d220793023ac29fdbb2fbdc4e68feba` carries the original measurement-only patch. Its preserved executable has SHA-256 `bd32506657b3930e0b7912e50644f0ddc5ccf1aea4c67576c845f7f54802cf3a`. Candidate source `3e20256be5df9e6827cecdb6e2a35f7633f73f6f` was built in the same dist profile and its executable has SHA-256 `8c2dba47870f5c0141a527fe023b8ef79fd24804fabcd1acedd97db35cc3d2aa`. The original workload manifest has SHA-256 `6b84027dec331eaaa873808848112f75caeebb670c2408e3e4b950941b7a4d35`.

The Ryzen 7 H 260 used Linux 7.3.5-1-cachyos for both sides of each new pair. The historical 52.20% result was on 7.3.4-1-cachyos. All 16 rows matched their historical baseline binary, workload digest, renderer settings and recording limits. Drivers remained NVIDIA 620.70.06 and RADV Mesa 26.2.6-arch3.1. The process held the performance power profile. Radeon used CPUs 0 and 8, and NVIDIA used CPUs 1–7. Each adapter ran alone with Vulkan, 1280 × 720, high quality, 4× stage MSAA and two frames in flight. Both builds collected GPU timestamps with readback, AVM debug and debug assertions disabled.

Each row used five alternating fresh-process baseline/candidate pairs with a fresh driver shader cache per process, 120 warmup frames and 1,200 measured frames. All 160 accepted processes provided a frame-linked GPU timestamp for every one of the 192,000 timed frames. The observer checked for competing benchmark and compiler processes every 100 ms. It archived and retried any overlapping trial. Accepted measurement intervals were disjoint across both GPUs.

The static target uses native `commandPreparation` and the three sparse targets use `renderTree`. These four cases require at least 10% lower phase P95 and 5% higher completed-frame throughput. The four controls allow at most 3% additional completed-frame time. Every row limits each CPU/GPU P95/P99 regression to 3% and each pair’s peak RSS growth to 5%. Absolute figures below are medians of the five runs per build. Percentages are medians of the five paired changes, except RSS, which reports the worst pair.

## GPU matrix

### NVIDIA GeForce RTX 5060 Laptop

8/8 passed. The isolation log contains 80 accepted processes and 0 excluded attempts.

| Case | Phase P95, µs | Phase gain | Completed ms/frame | Throughput gain | CPU P95 / P99 change | GPU P95 / P99 change | Worst RSS change | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `static_tree` | 620.63 → 3.65 | +99.43% | 1.085 → 0.291 | +269.22% | -75.95% / -78.25% | -15.72% / -52.45% | +1.07% | Pass |
| `dirty_1_percent` | 21.69 → 13.03 | +40.09% | 1.085 → 0.332 | +228.90% | -72.34% / -72.09% | -15.47% / -51.49% | +1.13% | Pass |
| `dirty_scattered` | 112.21 → 44.71 | +58.36% | 1.146 → 0.373 | +202.40% | -68.14% / -69.23% | -15.22% / -51.43% | +1.60% | Pass |
| `wide_single_edit` | 175.93 → 6.06 | +96.59% | 1.139 → 0.305 | +272.11% | -72.91% / -74.28% | -15.94% / -52.16% | +0.97% | Pass |
| `dirty_all` | 315.98 → 284.08 | +21.32% | 1.983 → 1.325 | +55.41% | -29.60% / -25.40% | -1.29% / +0.09% | +0.53% | Pass |
| `filter_cache` | 10.81 → 9.52 | +11.37% | 0.112 → 0.093 | +20.77% | -8.67% / -18.12% | -52.81% / -55.65% | +1.27% | Pass |
| `alpha_mask` | 18.39 → 16.52 | +14.11% | 0.490 → 0.176 | +182.30% | -69.21% / -70.52% | -92.03% / -89.50% | +0.66% | Pass |
| `load_unload` | 31.84 → 17.81 | +44.61% | 0.217 → 0.178 | +22.09% | -50.08% / -47.46% | -28.68% / -25.67% | +0.79% | Pass |

### AMD Radeon 780M

8/8 passed. The isolation log contains 80 accepted processes and 0 excluded attempts.

| Case | Phase P95, µs | Phase gain | Completed ms/frame | Throughput gain | CPU P95 / P99 change | GPU P95 / P99 change | Worst RSS change | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `static_tree` | 1060.69 → 4.95 | +99.52% | 1.830 → 0.339 | +439.62% | -83.21% / -81.51% | -28.06% / -22.80% | +1.17% | Pass |
| `dirty_1_percent` | 34.49 → 27.43 | +20.10% | 1.830 → 0.416 | +335.85% | -77.42% / -75.21% | -23.69% / -19.34% | +2.81% | Pass |
| `dirty_scattered` | 171.89 → 71.21 | +57.32% | 1.968 → 0.466 | +322.19% | -76.42% / -75.03% | -21.15% / -15.72% | +4.34% | Pass |
| `wide_single_edit` | 243.40 → 8.42 | +96.65% | 1.878 → 0.339 | +454.51% | -85.09% / -84.88% | -29.20% / -24.93% | +3.34% | Pass |
| `dirty_all` | 300.37 → 285.15 | +14.11% | 2.527 → 1.439 | +78.62% | -32.28% / -30.03% | -0.51% / -0.63% | +0.89% | Pass |
| `filter_cache` | 9.50 → 7.71 | +16.70% | 0.134 → 0.105 | +27.82% | -19.47% / -17.08% | -63.64% / -64.08% | +2.85% | Pass |
| `alpha_mask` | 18.55 → 14.69 | +21.64% | 0.632 → 0.192 | +228.80% | -76.74% / -79.75% | -72.83% / -73.51% | +0.98% | Pass |
| `load_unload` | 34.35 → 19.90 | +40.45% | 0.295 → 0.236 | +25.15% | -44.06% / -44.61% | -36.51% / -42.65% | +0.38% | Pass |

### Control CPU P99

| Case | RTX 5060, µs | Paired change | Radeon 780M, µs | Paired change |
| --- | ---: | ---: | ---: | ---: |
| `filter_cache` | 181.61 → 152.85 | -18.12% | 195.18 → 163.09 | -17.08% |
| `alpha_mask` | 930.24 → 272.57 | -70.52% | 1184.43 → 241.80 | -79.75% |
| `load_unload` | 627.63 → 329.78 | -47.46% | 676.13 → 393.20 | -44.61% |

## Work per measured frame

The table reports per-process work totals divided by measured frames, followed by the median across runs on NVIDIA. Tree and rendering counters are available for each process on both devices.

| Case | Tree visits | Slots examined | Instances updated | Leaves captured | Prepared packet replays | Draws | Passes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `static_tree` | 1.0 → 1.0 | 1.0 → 1.0 | 0.0 → 0.0 | 2000.0 → 0.0 | 0.0 → 1.0 | 4.0 → 4.0 | 4.0 → 4.0 |
| `dirty_1_percent` | 63.5 → 3.0 | 111.0 → 23.0 | 0.0 → 20.0 | 2000.0 → 0.0 | 0.0 → 18.5 | 4.0 → 4.0 | 4.0 → 4.0 |
| `dirty_scattered` | 842.5 → 22.0 | 871.0 → 61.0 | 0.0 → 20.0 | 2000.0 → 0.0 | 0.0 → 69.8 | 4.0 → 4.0 | 4.0 → 4.0 |
| `wide_single_edit` | 2002.0 → 2.0 | 2001.0 → 2.0 | 0.0 → 1.0 | 2000.0 → 0.0 | 0.0 → 22.0 | 4.0 → 4.0 | 4.0 → 4.0 |
| `dirty_all` | 2002.0 → 2002.0 | 2001.0 → 2001.0 | 0.0 → 0.0 | 2000.0 → 0.0 | 0.0 → 0.0 | 4.0 → 4.0 | 4.0 → 4.0 |
| `filter_cache` | 34.0 → 34.0 | 33.0 → 33.0 | 0.0 → 0.0 | 32.0 → 0.0 | 0.0 → 0.0 | 2.0 → 2.0 | 1.0 → 1.0 |
| `alpha_mask` | 74.0 → 74.0 | 49.0 → 49.0 | 0.0 → 0.0 | 48.0 → 48.0 | 0.0 → 0.0 | 48.0 → 24.0 | 25.0 → 1.0 |
| `load_unload` | 47.1 → 13.8 | 46.1 → 18.9 | 0.0 → 0.0 | 38.5 → 0.9 | 0.0 → 0.8 | 34.8 → 34.8 | 2.8 → 2.8 |

The filter workload still hits all 32 cached images every frame. The stage recaptures zero bitmap leaves. In the alpha-mask workload, saving the resolved mask removes 24 intermediate passes and 24 draws per frame. The loader replaces its child every 12 frames. Across the five runs, the median per-process median for the first nonempty frame after each replacement is shown below.

| Adapter | Baseline command preparation, µs | Candidate command preparation, µs | Baseline total CPU, µs/frame | Candidate total CPU, µs/frame |
| --- | ---: | ---: | ---: | ---: |
| NVIDIA | 225.65 | 4.63 | 457.95 | 210.73 |
| Radeon | 224.57 | 5.38 | 555.54 | 256.47 |

The earlier candidate at `bdd30fe1f` passed 5/8 on each GPU and missed all three controls. The intermediate candidate with cached bitmap and mask draws and serial mesh batching passed 7/8 on NVIDIA and 6/8 on Radeon. Both still missed load/unload CPU P99 (+6.00% and +8.32%). Radeon also missed the clustered 1% phase target (+9.14% measured versus +10% required). Its complete measurements are preserved alongside the final series.

## Correctness and resource lifetime

The core, render and regular WGPU suites passed 276, 68 and 42 tests. All 89 adapter-backed WGPU tests passed on Radeon. The four targeted tests passed on NVIDIA. The paired-metrics analyzer passed its 13 tests. The 14 SWF selectors passed 333 distinct ordinary cases plus two stage-quality variants. Another 24 cases matched the suite’s known-failure expectations and two JPEG XR loader cases were ignored.

The mixed SWF exercises synchronous child indices and pixel reads inside reentrant removal, bounds during addition, reparenting while a listener shrinks the destination, external masks, a scroll rectangle, bitmap mutation and filter removal. Its two reference frames had zero differing channel levels on both GPUs. The renderer tests compare cached filter and translucent mask draws against their general paths. On both adapters, rewriting a mask at the same placement or explicitly reclaiming its texture required a new raster pass and preserved every reference pixel. The Loader test covers definition equality, scale, movie-local bitmaps and weak ownership.

The final executable ran `cache_soak` at 60 FPS on Radeon for 1801.00 seconds. All 108,060 measured frame IDs had GPU timestamps. The scene advances through 900 reload boundaries and triggers GC, submission drain and reclamation every 180 frames for 600 checkpoints. These alternate between two positions in the 120-frame load cycle.

| Load cycle position | Checkpoints | Resources | Active bytes | Reserved / idle / retiring bytes | First / last quarter RSS median, MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 300 | 311 | 3,753,316 | 0 / 0 / 0 | 81.81 / 82.15 |
| 60 | 300 | 307 | 3,753,188 | 0 / 0 / 0 | 81.81 / 82.15 |

Each phase kept one resource signature across all 300 samples. The aggregate cross-phase plateau flag is false. After unload the runner found 0 object-owned resources and verified the identities of all 297 remaining baseline descriptors. Checkpoint RSS ranged from 81.32 to 82.16 MiB.

## Reproduction and scope

Local evidence is under `/home/starknt/.local/state/aola/verification/incremental-display-tree-20261119`. The complete original matrix is in `component-update/`, the tail attribution and the intermediate matrix are in `control-tails/`, and the final executables, raw paired traces, isolation logs, image checks and resource checkpoints are under `loader-geometry/`. The original frozen executable is `builds/baseline/native-perf`.

The results apply to these native SWF fixtures. Live Aola scenes and Android hardware were outside the measurement scope. Wide edits share unaffected logical pages. Reconciliation after a structural reorder and lookup by child identity can still scan the container.

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
