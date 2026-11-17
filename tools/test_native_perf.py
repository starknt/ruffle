import json
from pathlib import Path
import tempfile
import unittest

from native_perf import summarize, compare


class NativePerfTest(unittest.TestCase):
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
        baseline = {'completedFrameMs': 10, 'cpuP95Ms': 10, 'cpuP99Ms': 10,
                    'peakRssBytes': 100, 'phaseP95Ms': {'renderBackend': 8}}
        candidate = dict(baseline, completedFrameMs=8, cpuP95Ms=8, cpuP99Ms=8, peakRssBytes=106,
                         phaseP95Ms={'renderBackend': 6})
        self.assertFalse(compare([baseline]*5, [candidate]*5, 'renderBackend')['accepted'])


if __name__ == '__main__':
    unittest.main()
