import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from native_perf import summarize, compare, main


class NativePerfTest(unittest.TestCase):
    def test_binding_array_fallback_cannot_qualify_as_a_measured_candidate(self):
        from native_perf import compare_bitmap_paths
        before, after = NativeWgpuAcceptanceTest().paired()
        for run in before:
            run['bitmapArrayFrames'] = 0
        for run in after:
            run.update(completedFrameMs=9., gpuP95Ms=1.6, bitmapArrayFrames=run['frames'])
        self.assertTrue(compare_bitmap_paths(before, after, 'gpu')['accepted'])
        after[2]['bitmapArrayFrames'] -= 1
        with self.assertRaises(ValueError):
            compare_bitmap_paths(before, after, 'gpu')

    def test_layers_must_run_on_every_frame_against_the_adopted_bitmap_array(self):
        from native_perf import compare_bitmap_paths
        before, after = NativeWgpuAcceptanceTest().paired()
        for run in before:
            run.update(bitmapArrayFrames=run['frames'], bitmapLayerFrames=0)
        for run in after:
            run.update(completedFrameMs=9., gpuP95Ms=1.6, bitmapArrayFrames=0, bitmapLayerFrames=run['frames'])
        self.assertTrue(compare_bitmap_paths(before, after, 'gpu', 'layers')['accepted'])
        before[2]['bitmapArrayFrames'] -= 1
        with self.assertRaises(ValueError):
            compare_bitmap_paths(before, after, 'gpu', 'layers')

    def test_indirect_comparison_requires_the_selected_dispatch_each_frame(self):
        from native_perf import compare_bitmap_draws
        for mode in ['indirect', 'cull', 'count']:
            with self.subTest(mode=mode):
                before, after = NativeWgpuAcceptanceTest().paired()
                for run in before + after:
                    run.update(bitmapIndirectFrames=0, bitmapCullFrames=0, bitmapCountFrames=0)
                for run in after:
                    run.update(completedFrameMs=9., gpuP95Ms=1.6, bitmapIndirectFrames=run['frames'])
                    if mode in ('cull', 'count'):
                        run['bitmapCullFrames'] = run['frames']
                    if mode == 'count':
                        run['bitmapCountFrames'] = run['frames']
                self.assertTrue(compare_bitmap_draws(before, after, 'gpu', mode)['accepted'])
                after[-1]['bitmapIndirectFrames'] -= 1
                with self.assertRaises(ValueError):
                    compare_bitmap_draws(before, after, 'gpu', mode)

    def test_cache_transfer_comparison_requires_both_copy_paths_on_the_measured_frames(self):
        from native_perf import compare_filter_paths
        before, after = NativeWgpuAcceptanceTest().paired()
        for row in before:
            row.update(filterDrawCopyFrames=1200, filterTransferFrames=0)
        for row in after:
            row.update(completedFrameMs=9., filterDrawCopyFrames=0, filterTransferFrames=1200)
        self.assertTrue(compare_filter_paths(before, after, 'gpu', transfer=True)['accepted'])
        before[0]['filterDrawCopyFrames'] = 1199
        with self.assertRaises(ValueError):
            compare_filter_paths(before, after, 'gpu', transfer=True)

    def test_direct_cache_comparison_requires_the_final_pass_without_a_copy(self):
        from native_perf import compare_filter_cache
        before, after = NativeWgpuAcceptanceTest().paired()
        for row in before:
            row.update(filterDrawCopyFrames=1200, filterTransferFrames=0, filterDirectFrames=0)
        for row in after:
            row.update(completedFrameMs=9., filterDrawCopyFrames=0, filterTransferFrames=0, filterDirectFrames=1200)
        self.assertTrue(compare_filter_cache(before, after, 'gpu', 'direct')['accepted'])
        for key, value in [('filterDrawCopyFrames', 1), ('filterTransferFrames', 1), ('filterDirectFrames', 1199)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                compare_filter_cache(before, [*after[:-1], dict(after[-1], **{key: value})], 'gpu', 'direct')

    def test_compute_comparison_requires_the_opt_in_dispatch_on_every_measured_frame(self):
        from native_perf import compare_filter_paths
        before, after = NativeWgpuAcceptanceTest().paired()
        for row in before + after:
            row['gpuPhaseP95Ms'] = {'ColorMatrixFilter': 2.}
            row['computeFilterFrames'] = 0
        for row in after:
            row.update(completedFrameMs=9., gpuPhaseP95Ms={'ColorMatrixFilter': 1.6}, computeFilterFrames=1200)
        self.assertTrue(compare_filter_paths(before, after, 'gpu:ColorMatrixFilter')['accepted'])
        after[0]['computeFilterFrames'] = 1199
        with self.assertRaises(ValueError):
            compare_filter_paths(before, after, 'gpu:ColorMatrixFilter')
    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux driver disk caches')
    def test_each_measured_process_starts_with_private_empty_shader_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = root / 'user-cache'
            shared.mkdir()
            (shared / 'existing').touch()
            binary = root / 'probe'
            binary.write_text(
                '#!' + sys.executable + '\n'
                'import json, os, sys\n'
                'from pathlib import Path\n'
                'keys = ("__GL_SHADER_DISK_CACHE_PATH", "MESA_SHADER_CACHE_DIR")\n'
                'caches = [Path(os.environ[k]) for k in keys]\n'
                'cold = all(not any(path.iterdir()) for path in caches)\n'
                'for path in caches: (path / "compiled").touch()\n'
                'out = Path(sys.argv[sys.argv.index("--output") + 1])\n'
                'out.write_text(json.dumps([cold, os.environ["XDG_CACHE_HOME"], str(caches[0])]))\n'
            )
            binary.chmod(0o700)
            manifest = root / 'manifest.toml'
            manifest.write_text('[[case]]\nname = "probe"\n')
            output = root / 'runs'
            env = dict.fromkeys(
                ('XDG_CACHE_HOME', '__GL_SHADER_DISK_CACHE_PATH', 'MESA_SHADER_CACHE_DIR'),
                str(shared),
            )
            args = ['native_perf', '--baseline', str(binary), '--candidate', str(binary),
                    '--manifest', str(manifest), '--output', str(output)]
            with (
                patch.dict(os.environ, env),
                patch.object(sys, 'argv', args),
                patch('native_perf.summarize', side_effect=lambda p: json.loads(p.read_text())),
                patch('native_perf.compare', return_value={}),
            ):
                main()
            runs = json.loads((output / 'summary.json').read_text())['probe']['runs']
            samples = runs['baseline'] + runs['candidate']
            self.assertEqual([sample[:2] for sample in samples], [[True, str(shared)]] * 10)
            self.assertEqual(len({sample[2] for sample in samples}), 10)
            self.assertEqual(sorted(p.name for p in shared.iterdir()), ['existing'])

    def test_instrumentation_modes_must_match(self):
        run = {'completedFrameMs': 10, 'cpuP95Ms': 10, 'cpuP99Ms': 10,
               'peakRssBytes': 100, 'phaseP95Ms': {'logic': 8}, 'profilingEnabled': True}
        with self.assertRaises(ValueError):
            compare([run]*5, [dict(run, profilingEnabled=False)]*5, 'logic', control=True)

    def test_paced_soak_cannot_be_counted_as_throughput_acceptance(self):
        run = {'completedFrameMs': 10, 'cpuP95Ms': 10, 'cpuP99Ms': 10,
               'peakRssBytes': 100, 'phaseP95Ms': {'logic': 8}, 'paceFps': 60}
        with self.assertRaises(ValueError):
            compare([run]*5, [run]*5, 'logic', control=True)

    def trace(self, directory, frames=2, complete=True):
        path = Path(directory) / 'run.jsonl'
        events = [{'event': 'session', 'frames': frames, 'case': 'small', 'backend': 'Vulkan', 'device': 'test'}]
        events += [{'event': 'frame', 'frameId': i + 1, 'cpuNs': (i+1)*1000000,
                    'phaseNs': {'renderBackend': (i+1)*500000}, 'work': {}} for i in range(frames)]
        if complete:
            events += [{'event': 'summary', 'frames': frames, 'completedNs': 8000000,
                        'peakRssBytes': 100000000, 'droppedGpuSamples': 0}]
        path.write_text('\n'.join(map(json.dumps, events)))
        return path

    def test_incomplete_run_cannot_be_used_as_performance_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                summarize(self.trace(d, complete=False))

    def test_throughput_includes_gpu_drain_not_sum_of_cpu_times(self):
        with tempfile.TemporaryDirectory() as d:
            result = summarize(self.trace(d))
            self.assertEqual(result['completedFrameMs'], 4)
            self.assertEqual(result['cpuP95Ms'], 2)
            self.assertIsNone(result['gpuP95Ms'])

    def test_peak_rss_includes_observed_samples_and_the_process_high_water_mark(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.trace(directory)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[1]['rssBytes'] = 300
            rows[2]['rssBytes'] = None
            for high_water, last_rss, expected in [
                (250, 150, 300), (400, 150, 400), (250, 350, 350), (None, 150, None),
            ]:
                with self.subTest(high_water=high_water, last_rss=last_rss):
                    rows[-1].update(peakRssBytes=high_water, rssBytes=last_rss)
                    path.write_text('\n'.join(map(json.dumps, rows)))
                    self.assertEqual(summarize(path)['peakRssBytes'], expected)

    def test_memory_soak_reports_gc_and_keeps_sampled_rss_and_gpu_memory_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.trace(directory, frames=100)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]['case'] = 'memory_soak'
            for i, frame in enumerate(rows[1:-1]):
                frame['phaseNs']['gc'] = (i + 1) * 1000
                frame['nativeResources'] = {'logical_bytes': i + 1}
            rows[1].update(rssBytes=100, allocator={'allocatedBytes': 200, 'reservedBytes': 800})
            rows[2].update(rssBytes=300, allocator={'allocatedBytes': 400, 'reservedBytes': 800})
            path.write_text('\n'.join(map(json.dumps, rows)))
            report = summarize(path)
            self.assertEqual(report.get('gcP99Ms'), 0.099)
            self.assertEqual(report['rssMedianBytes'], 200)
            self.assertEqual(report['gpuLogicalMedianBytes'], 50.5)
            self.assertEqual(report['gpuAllocatedMedianBytes'], 300)
            self.assertEqual(report['gpuReservedMedianBytes'], 800)
            self.assertEqual(report['completedFps'], 12500)
            del rows[2]['phaseNs']['gc']
            path.write_text('\n'.join(map(json.dumps, rows)))
            self.assertIsNone(summarize(path)['gcP99Ms'])

    def test_memory_comparison_labels_lower_throughput_and_higher_rss_as_regressions(self):
        before, after = NativeWgpuAcceptanceTest().paired()
        for run in before + after:
            run.update(case='memory_soak', completedFps=100., rssMedianBytes=100, gcP99Ms=2.)
        for run in after:
            run.update(completedFrameMs=12.5, completedFps=80., rssMedianBytes=130, gcP99Ms=1.)
        comparison = compare(before, after, 'gpu')
        self.assertIn('memoryMetrics', comparison)
        metrics = comparison['memoryMetrics']
        self.assertEqual(metrics['completedFps']['status'], 'pendingOptimization')
        self.assertEqual(metrics['rssMedianBytes']['status'], 'pendingOptimization')
        self.assertEqual(metrics['gcP99Ms']['status'], 'improved')
        self.assertEqual(metrics['gcP99Ms']['baselineMedian'], 2.)
        self.assertEqual(metrics['gcP99Ms']['candidateMedian'], 1.)
        self.assertEqual(metrics['gcP99Ms']['pairedChange'], -.5)
        self.assertEqual(metrics['gpuReservedMedianBytes']['status'], 'unavailable')

    def test_memory_comparison_retains_regressing_pairs_even_when_the_typical_pair_improves(self):
        before, after = NativeWgpuAcceptanceTest().paired()
        for i, (b, c) in enumerate(zip(before, after)):
            b.update(case='memory_soak', onePercentLowFps=[100, 200, 300, 400, 500][i])
            c.update(case='memory_soak', onePercentLowFps=[110, 210, 310, 120, 180][i])
        row = compare(before, after, 'gpu')['memoryMetrics']['onePercentLowFps']
        self.assertGreater(row['pairedChange'], 0)
        self.assertLess(row['candidateMedian'], row['baselineMedian'])
        self.assertEqual(row['status'], 'pendingOptimization')
        self.assertEqual(row['pendingOptimizationPairs'], [4, 5])

    def test_memory_comparison_flags_cpu_and_gpu_frame_time_regressions(self):
        before, after = NativeWgpuAcceptanceTest().paired()
        keys = ('cpuP95Ms', 'cpuP99Ms', 'gpuP95Ms', 'gpuP99Ms')
        for index, (baseline, candidate) in enumerate(zip(before, after)):
            baseline.update(case='memory_soak', **dict.fromkeys(keys, 10))
            candidate.update(case='memory_soak', **dict.fromkeys(keys, 12 if index < 3 else 8))
        metrics = compare(before, after, 'gpu')['memoryMetrics']
        for key in keys:
            with self.subTest(key=key):
                self.assertEqual(metrics.get(key, {}).get('status'), 'pendingOptimization')
                self.assertEqual(metrics[key]['pendingOptimizationPairs'], [1, 2, 3])

    def test_named_gpu_phases_remain_inclusive_and_require_every_frame(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.trace(d)
            with path.open('a') as stream:
                for frame, phases in [(1, [['Filter chain', 8000], ['BlurFilter', 6000]]),
                                      (2, [['Filter chain', 9000]])]:
                    stream.write('\n' + json.dumps(dict(event='gpu', frameId=frame, ns=11000, phaseNs=phases)))
            phases = summarize(path)['gpuPhaseP95Ms']
            self.assertEqual(phases['Filter chain'], .009)
            self.assertIsNone(phases['BlurFilter'])

    def test_gpu_completion_pacing_includes_idle_time_and_orders_samples_by_source_frame(self):
        with tempfile.TemporaryDirectory() as d:
            path=self.trace(d, frames=201)
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            end=1_000_000
            gpu=[]
            for index in range(201):
                end += (30_000_000 if index==199 else 50_000_000 if index==200 else 10_000_000)
                gpu.append(dict(event='gpu',frameId=index+1,ns=1_000_000,stageEndNs=end))
            rows.extend(reversed(gpu))
            path.write_text('\n'.join(map(json.dumps, rows)))
            result=summarize(path)
            self.assertEqual((result['frameIntervalP95Ms'],result['onePercentLowFps']), (10,25))
            self.assertEqual((result['intervalCount'],result['lowIntervalCount']), (200,2))
            del gpu[0]['stageEndNs']
            path.write_text('\n'.join(map(json.dumps, rows)))
            self.assertIsNone(summarize(path)['onePercentLowFps'])
            gpu[0]['stageEndNs']=gpu[1]['stageEndNs']
            path.write_text('\n'.join(map(json.dumps, rows)))
            with self.assertRaises(ValueError): summarize(path)

    def test_duplicate_or_invalid_named_timestamp_is_rejected(self):
        for phases in [[['BlurFilter', 1], ['BlurFilter', 2]], [['BlurFilter', -1]],
                       [['BlurFilter', True]], [['BlurFilter', float('nan')]]]:
            with self.subTest(phases=phases), tempfile.TemporaryDirectory() as d:
                path = self.trace(d)
                with path.open('a') as stream:
                    stream.write('\n' + json.dumps(dict(event='gpu', frameId=1, ns=11000, phaseNs=phases)))
                with self.assertRaises(ValueError):
                    summarize(path)

    def test_qualified_frame_improvements_report_their_memory_cost(self):
        baseline, candidate = NativeWgpuAcceptanceTest().paired()
        for run in candidate:
            run.update(completedFrameMs=8, cpuP95Ms=6, cpuP99Ms=7, peakRssBytes=106,
                       phaseP95Ms={'preparation': 3})
        result = compare(baseline, candidate, 'preparation')
        self.assertTrue(result['accepted'])
        self.assertAlmostEqual(result['peakRssGrowth'], .06)

class NativeWgpuAcceptanceTest(unittest.TestCase):
    def paired(self, **changes):
        base = dict(case='scene', backend='Vulkan', device='GPU', driver='driver',
                    frames=1200, warmup=120, profilingEnabled=True, paceFps=None,
                    settings={'quality': 'high', 'samples': 4, 'viewport': [800, 600],
                              'readback': False, 'inFlightLimit': 2, 'debugAssertions': False, 'avmDebug': False},
                    workloadSha256='0' * 64, gpuCoverage=1.0,
                    completedFrameMs=10., cpuP95Ms=8., cpuP99Ms=9.,
                    gpuP95Ms=8., gpuP99Ms=9., peakRssBytes=100,
                    frameIntervalP95Ms=10., onePercentLowFps=80.,
                    intervalCount=1199, lowIntervalCount=12,
                    phaseP95Ms={'preparation': 4.})
        before = [dict(base, runId=f'b{i}') for i in range(5)]
        after = [dict(base, runId=f'c{i}') | {'frameIntervalP95Ms': 9.99, 'onePercentLowFps': 80.01} | changes for i in range(5)]
        return before, after

    def test_small_positive_paired_gains_qualify(self):
        before, after = self.paired()
        for run in after:
            run.update(completedFrameMs=9.99, gpuP95Ms=7.99)
        self.assertTrue(compare(before, after, 'gpu')['accepted'])
        for frame_ms, gpu_ms, accepted in [(10., 7.99, False), (9.99, 8., True),
                                            (10., 8., False), (10.01, 7.99, False)]:
            with self.subTest(frame_ms=frame_ms, gpu_ms=gpu_ms):
                for run in after:
                    run.update(completedFrameMs=frame_ms, gpuP95Ms=gpu_ms)
                self.assertEqual(compare(before, after, 'gpu')['accepted'], accepted)

    def test_promotion_requires_faster_completion_smoother_frames_and_higher_one_percent_low(self):
        before, after = self.paired()
        for run in after:
            run.update(completedFrameMs=9.99, gpuP95Ms=12., peakRssBytes=120)
        result=compare(before, after, 'gpu')
        self.assertTrue(result['accepted'])
        self.assertGreater(result['onePercentLowGain'], 0)
        for key in ('frameIntervalP95Ms','onePercentLowFps'):
            for value in (None,before[0][key]):
                with self.subTest(key=key,value=value):
                    rows=[dict(run) for run in after]
                    for run in rows: run[key]=value
                    self.assertFalse(compare(before, rows, 'gpu')['accepted'])

    def test_tail_must_cover_every_pair_of_source_frames(self):
        before, after = self.paired(completedFrameMs=9.9)
        for field in ('intervalCount', 'lowIntervalCount'):
            with self.subTest(field=field):
                incomplete = [dict(run) for run in after]
                incomplete[0][field] -= 1
                self.assertFalse(compare(before, incomplete, 'gpu')['accepted'])

    def test_missing_gpu_samples_cannot_pass_a_fast_cpu_result(self):
        before, after = self.paired()
        for run in after:
            run.update(gpuCoverage=0., gpuP95Ms=None, gpuP99Ms=None,
                       completedFrameMs=7., phaseP95Ms={'preparation': 2.})
        with self.assertRaises(ValueError):
            compare(before, after, 'preparation')

    def test_controls_require_gpu_tail_and_completed_throughput_stability(self):
        before, after = self.paired()
        for run in after:
            run['gpuP99Ms'] = 10.
        self.assertFalse(compare(before, after, 'gpu', control=True)['accepted'])
        for run in after:
            run.update(gpuP99Ms=9., completedFrameMs=11.)
        self.assertFalse(compare(before, after, 'gpu', control=True)['accepted'])

    def test_control_rejects_pacing_regressions_even_with_unchanged_gpu_time(self):
        for key, factor in [('onePercentLowFps', .9), ('frameIntervalP95Ms', 1.1)]:
            with self.subTest(key=key):
                before, after = self.paired()
                for run in after:
                    run[key] = before[0][key] * factor
                self.assertFalse(compare(before, after, 'gpu', control=True)['accepted'])

    def test_gpu_targets_use_measured_gpu_time(self):
        before, after = self.paired()
        for run in after:
            run.update(completedFrameMs=9., gpuP95Ms=7., gpuP99Ms=8.)
        self.assertTrue(compare(before, after, 'gpu')['accepted'])
        for run in after:
            run.update(gpuP95Ms=8.1, gpuP99Ms=9., phaseP95Ms={'renderBackend': 1.})
        result = compare(before, after, 'gpu')
        self.assertTrue(result['accepted'])
        self.assertLess(result['phaseP95Gain'], 0)

    def test_named_gpu_gate_uses_shader_time_and_rejects_missing_queries(self):
        before, after = self.paired()
        for run in before:
            run['gpuPhaseP95Ms'] = {'ColorMatrixFilter': 2.}
        for run in after:
            run.update(completedFrameMs=9., gpuPhaseP95Ms={'ColorMatrixFilter': 1.6})
        self.assertTrue(compare(before, after, 'gpu:ColorMatrixFilter')['accepted'])
        after[2]['gpuPhaseP95Ms'] = {}
        self.assertFalse(compare(before, after, 'gpu:ColorMatrixFilter')['accepted'])

    def test_worker_targets_use_preparation_time(self):
        before, after = self.paired()
        for run in after:
            run.update(completedFrameMs=9., phaseP95Ms={'preparation': 3.})
        self.assertTrue(compare(before, after, 'preparation')['accepted'])

    def test_settings_workload_identity_and_fresh_processes_are_required(self):
        for key, value in [('settings', {'quality': 'low'}), ('workloadSha256', '1' * 64),
                           ('driver', 'another'), ('warmup', 119), ('frames', 1199)]:
            before, after = self.paired()
            after[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                compare(before, after, 'gpu', control=True)
        before, after = self.paired()
        after[1]['runId'] = after[0]['runId']
        with self.assertRaises(ValueError):
            compare(before, after, 'gpu', control=True)

    def test_one_pairs_rss_regression_cannot_hide_in_the_median(self):
        before, after = self.paired()
        after[0]['peakRssBytes'] = 106
        self.assertFalse(compare(before, after, 'gpu', control=True)['accepted'])

    def test_orphan_gpu_ids_are_rejected_instead_of_silently_filtered(self):
        with tempfile.TemporaryDirectory() as directory:
            path = NativePerfTest().trace(directory)
            with path.open('a') as stream:
                stream.write('\n' + json.dumps({'event': 'gpu', 'frameId': None, 'ns': 1000}))
            with self.assertRaises(ValueError):
                summarize(path)


if __name__ == '__main__':
    unittest.main()
