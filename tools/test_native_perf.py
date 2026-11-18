import json
from pathlib import Path
import tempfile
import unittest

from native_perf import summarize, compare


class NativePerfTest(unittest.TestCase):
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

    def test_rss_gate_is_not_relaxed_for_a_faster_candidate(self):
        baseline, candidate = NativeWgpuAcceptanceTest().paired()
        for run in candidate:
            run.update(completedFrameMs=8, cpuP95Ms=6, cpuP99Ms=7, peakRssBytes=106,
                       phaseP95Ms={'preparation': 3})
        self.assertFalse(compare(baseline, candidate, 'preparation')['accepted'])

class NativeWgpuAcceptanceTest(unittest.TestCase):
    def paired(self, **changes):
        base = dict(case='scene', backend='Vulkan', device='GPU', driver='driver',
                    frames=1200, warmup=120, profilingEnabled=True, paceFps=None,
                    settings={'quality': 'high', 'samples': 4, 'viewport': [800, 600],
                              'readback': False, 'inFlightLimit': 2, 'debugAssertions': False, 'avmDebug': False},
                    workloadSha256='0' * 64, gpuCoverage=1.0,
                    completedFrameMs=10., cpuP95Ms=8., cpuP99Ms=9.,
                    gpuP95Ms=8., gpuP99Ms=9., peakRssBytes=100,
                    phaseP95Ms={'preparation': 4.})
        before = [dict(base, runId=f'b{i}') for i in range(5)]
        after = [dict(base, runId=f'c{i}', **changes) for i in range(5)]
        return before, after

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

    def test_gpu_targets_use_measured_gpu_time(self):
        before, after = self.paired()
        for run in after:
            run.update(completedFrameMs=9., gpuP95Ms=7., gpuP99Ms=8.)
        self.assertTrue(compare(before, after, 'gpu')['accepted'])
        for run in after:
            run.update(gpuP95Ms=8., gpuP99Ms=9., phaseP95Ms={'renderBackend': 1.})
        self.assertFalse(compare(before, after, 'gpu')['accepted'])

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
