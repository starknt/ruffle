#!/usr/bin/env python3
"""Paired native-perf runs. Unfinished traces never count as acceptance."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import tomllib


def percentile(values, fraction):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * fraction)-1)] if values else None


def summarize(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    sessions = [r for r in records if r.get('event') == 'session']
    summaries = [r for r in records if r.get('event') == 'summary']
    frames = [r for r in records if r.get('event') == 'frame']
    if len(sessions) != 1 or len(summaries) != 1:
        raise ValueError('trace must have one session and one completed summary')
    session, summary = sessions[0], summaries[0]
    ids = {r['frameId'] for r in frames}
    if not frames or len(ids) != len(frames) or len(frames) != session['frames'] or len(frames) != summary['frames']:
        raise ValueError('missing, duplicate, or incomplete frame samples')
    if (summary.get('resourcesAfterDrain') or {}).get('terminal_failure'):
        raise ValueError('trace ended with a terminal renderer failure')
    if summary['completedNs'] <= 0 or summary.get('droppedGpuSamples', 0):
        raise ValueError('invalid completion time or dropped GPU samples')
    gpu = [r for r in records if r.get('event') == 'gpu']
    if any(r.get('frameId') not in ids for r in gpu):
        raise ValueError('orphan or missing GPU frame identity')
    if any(not isinstance(r.get('ns'), (float, int)) or not math.isfinite(r['ns']) or r['ns'] < 0 for r in gpu):
        raise ValueError('invalid GPU timestamp duration')
    if len({r['frameId'] for r in gpu}) != len(gpu):
        raise ValueError('duplicate GPU frame identity')
    gpu_phases = {}
    for record in gpu:
        seen = set()
        intervals = record.get('phaseNs', [])
        if not isinstance(intervals, list):
            raise ValueError('invalid named GPU intervals')
        for interval in intervals:
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError('invalid named GPU interval')
            name, ns = interval
            if (not isinstance(name, str) or not name or name in seen or type(ns) not in (float, int)
                    or not math.isfinite(ns) or ns < 0):
                raise ValueError('duplicate or invalid named GPU interval')
            seen.add(name)
            gpu_phases.setdefault(name, []).append(ns / 1e6)
    phase_values = [dict(r.get('phaseNs', {}), **r.get('nativeNs', {})) for r in frames]
    phase_names = {key for r in phase_values for key in r}
    if any(not isinstance(r.get('cpuNs'), (float, int)) or not math.isfinite(r['cpuNs']) or r['cpuNs'] < 0 for r in frames):
        raise ValueError('invalid CPU frame duration')
    return {
        'warmup': session.get('warmup'), 'driver': session.get('driver'),
        'runId': session.get('runId'), 'settings': session.get('settings'),
        'workloadSha256': session.get('workloadSha256'),
        'case': session['case'], 'backend': session['backend'], 'device': session['device'],
        'paceFps': session.get('paceFps'),
        'profilingEnabled': session.get('profilingEnabled', True),
        'frames': len(frames), 'completedFrameMs': summary['completedNs']/len(frames)/1e6,
        'cpuP95Ms': percentile([r['cpuNs']/1e6 for r in frames], .95),
        'cpuP99Ms': percentile([r['cpuNs']/1e6 for r in frames], .99),
        'gpuP95Ms': percentile([r['ns']/1e6 for r in gpu], .95),
        'gpuP99Ms': percentile([r['ns']/1e6 for r in gpu], .99),
        'gpuCoverage': len(gpu)/len(frames), 'peakRssBytes': summary.get('peakRssBytes'),
        'gpuPhaseP95Ms': {name: percentile(times, .95) if len(times) == len(frames) else None
                          for name, times in gpu_phases.items()},
        'phaseP95Ms': {key: percentile([r[key]/1e6 for r in phase_values], .95)
                       if all(key in r for r in phase_values) else None for key in phase_names},
        'resourcesAfterDrain': summary.get('resourcesAfterDrain'),
        'allocatorAfterDrain': summary.get('allocatorAfterDrain'),
        'workTotals': {key: sum(r.get('work', {}).get(key, 0) for r in frames)
                       for key in {key for r in frames for key in r.get('work', {})}},
        'computeFilterFrames': sum(r.get('work', {}).get('compute_filter_passes', 0) > 0 for r in frames),
        'bitmapArrayFrames': sum(r.get('work', {}).get('bitmap_binding_array_batches', 0) > 0 for r in frames),
    }


def compare(baseline, candidate, phase, control=False):
    runs = baseline + candidate
    if len(baseline) != len(candidate) or len(baseline) < 5:
        raise ValueError('acceptance requires at least five complete pairs')
    if any(r.get('paceFps') is not None for r in runs):
        raise ValueError('paced soaks cannot be used for throughput acceptance')
    run_ids = [r.get('runId') for r in runs]
    if any(not run_id for run_id in run_ids) or len(set(run_ids)) != len(run_ids):
        raise ValueError('acceptance requires fresh process identities')
    for run in runs:
        if not run.get('profilingEnabled') or run.get('gpuCoverage') != 1.:
            raise ValueError('acceptance requires complete frame-linked GPU samples')
        if run.get('frames', 0) < 1200 or run.get('warmup', 0) < 120:
            raise ValueError('acceptance requires 120 warmup and 1200 measured frames')
        for key in ('completedFrameMs', 'cpuP95Ms', 'cpuP99Ms', 'gpuP95Ms', 'gpuP99Ms', 'peakRssBytes'):
            value = run.get(key)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'missing or invalid metric {key}')
        if not run.get('workloadSha256') or not run.get('driver'):
            raise ValueError('missing workload or driver identity')
        settings = run.get('settings') or {}
        required = ('quality', 'samples', 'viewport', 'readback', 'inFlightLimit', 'debugAssertions', 'avmDebug')
        if any(key not in settings for key in required):
            raise ValueError('incomplete rendering settings')
        if settings['readback'] or settings['debugAssertions'] or settings['avmDebug']:
            raise ValueError('acceptance requires a release build without image readback or AVM debug')
    for before, after in [(baseline[0], r) for r in runs]:
        for key in ('case', 'backend', 'device', 'driver', 'frames', 'warmup',
                    'profilingEnabled', 'settings', 'workloadSha256'):
            if before.get(key) != after.get(key):
                raise ValueError(f'mismatched pair {key}')
    median = statistics.median
    ratios = lambda key: [c[key]/b[key]-1 for b,c in zip(baseline,candidate)]
    throughput = median(b['completedFrameMs']/c['completedFrameMs']-1 for b,c in zip(baseline,candidate))
    p95, p99 = median(ratios('cpuP95Ms')), median(ratios('cpuP99Ms'))
    gpu95, gpu99 = median(ratios('gpuP95Ms')), median(ratios('gpuP99Ms'))
    # Total process high-water marks include setup, warmup and every optimization
    # phase. A bad pair must not disappear behind a median of quieter runs.
    rss = max(ratios('peakRssBytes'))
    if phase == 'gpu':
        improvement = -gpu95
    else:
        field, name = ('gpuPhaseP95Ms', phase[4:]) if phase.startswith('gpu:') else ('phaseP95Ms', phase)
        phases = [(b.get(field, {}).get(name), c.get(field, {}).get(name)) for b,c in zip(baseline,candidate)]
        improvement = None if any(not b or c is None or not math.isfinite(c) or c < 0 for b,c in phases) else median(1-c/b for b,c in phases)
    stable = rss <= .05 and max(p95, p99, gpu95, gpu99) <= .03
    if control:
        accepted = stable and median(ratios('completedFrameMs')) <= .03
    else:
        accepted = (stable and improvement is not None and min(throughput, improvement) >= 0
                    and max(throughput, improvement) > 0)
    return {'accepted': accepted, 'throughputGain':throughput, 'cpuP95Regression':p95,
            'cpuP99Regression':p99, 'gpuP95Regression':gpu95, 'gpuP99Regression':gpu99,
            'peakRssGrowth':rss, 'phase':phase, 'phaseP95Gain':improvement}


def compare_filter_paths(baseline, candidate, phase):
    if (any(run.get('computeFilterFrames') != 0 for run in baseline)
            or any(run.get('computeFilterFrames') != run['frames'] for run in candidate)):
        raise ValueError('filter comparison requires fragment and compute execution on every measured frame')
    return compare(baseline, candidate, phase)


def compare_bitmap_paths(baseline, candidate, phase):
    if (any(run.get('bitmapArrayFrames') != 0 for run in baseline)
            or any(run.get('bitmapArrayFrames') != run['frames'] for run in candidate)):
        raise ValueError('binding array comparison requires the selected path on every measured frame')
    return compare(baseline, candidate, phase)


@contextmanager
def driver_environment(isolate_shader_cache):
    if not isolate_shader_cache:
        yield None
        return
    # Include cold shader compilation in both process high-water marks. A
    # shared disk cache lets one executable warm the other between samples.
    with tempfile.TemporaryDirectory(prefix='ruffle-native-perf-') as directory:
        yield {**os.environ, '__GL_SHADER_DISK_CACHE': '1',
               '__GL_SHADER_DISK_CACHE_PATH': directory, 'MESA_SHADER_CACHE_DIR': directory}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', action='append')
    parser.add_argument('--backend', default='vulkan')
    parser.add_argument('--warmup', type=int, default=120)
    parser.add_argument('--frames', type=int, default=1200)
    parser.add_argument('--pairs', type=int, default=5)
    parser.add_argument('--phase', default='gpu', help='gpu, gpu:ColorMatrixFilter, or a CPU phase such as preparation')
    parser.add_argument('--control', action='store_true')
    experiment = parser.add_mutually_exclusive_group()
    experiment.add_argument('--compare-compute-filters', action='store_true',
                            help='compare fragment with opt-in compute in the same binary')
    experiment.add_argument('--compare-bitmap-bindings', action='store_true',
                            help='compare sixteen explicit textures with the optional binding array')
    parser.add_argument('--profile-off', action='store_true',
                        help='disable measurement instrumentation in both binaries')
    args = parser.parse_args()
    if ((args.compare_compute_filters or args.compare_bitmap_bindings)
            and (not args.baseline.samefile(args.candidate) or args.control or args.profile_off)):
        parser.error('path comparisons need the same executable and enabled target measurements')
    args.output.mkdir(parents=True, exist_ok=False)
    cases = args.case or [r['name'] for r in tomllib.loads(args.manifest.read_text())['case']]
    binaries = {'baseline': args.baseline.resolve(), 'candidate': args.candidate.resolve()}
    metadata = {label: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                for label,path in binaries.items()}
    metadata['manifestSha256'] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if args.compare_compute_filters:
        metadata['filterPaths'] = {'baseline': 'fragment', 'candidate': 'compute'}
    if args.compare_bitmap_bindings:
        metadata['bitmapBindings'] = {'baseline': 'explicit-on-array-device', 'candidate': 'array'}
    isolate_shader_cache = sys.platform.startswith('linux') and args.backend in ('vulkan', 'gl')
    metadata['shaderDiskCache'] = 'empty-per-process' if isolate_shader_cache else 'system'
    (args.output/'builds.json').write_text(json.dumps(metadata, indent=2)+'\n')
    report = {}
    for case in cases:
        runs = {'baseline': [], 'candidate': []}
        for pair in range(args.pairs):
            order = ('baseline','candidate') if pair%2 == 0 else ('candidate','baseline')
            for label in order:
                path = args.output/f'{case}-{pair+1}-{label}.jsonl'
                command = [str(binaries[label]), '--manifest', str(args.manifest.resolve()), '--case',case,
                           '--output',str(path.resolve()),'--backend',args.backend,'--warmup',str(args.warmup),'--frames',str(args.frames)]
                if args.profile_off:
                    command.append('--profile-off')
                with path.with_suffix('.log').open('w') as log, driver_environment(isolate_shader_cache) as env:
                    if args.compare_compute_filters:
                        env = dict(os.environ if env is None else env,
                                   RUFFLE_COMPUTE_FILTERS='1' if label == 'candidate' else '0')
                    elif args.compare_bitmap_bindings:
                        env = dict(os.environ if env is None else env,
                                   RUFFLE_BITMAP_BINDINGS='array' if label == 'candidate' else 'explicit-on-array-device')
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600, env=env)
                runs[label].append(summarize(path))
            print(f'{case}: pair {pair+1}/{args.pairs} complete', flush=True)
        if args.compare_compute_filters:
            comparison = compare_filter_paths(runs['baseline'], runs['candidate'], args.phase)
        elif args.compare_bitmap_bindings:
            comparison = compare_bitmap_paths(runs['baseline'], runs['candidate'], args.phase)
        else:
            comparison = compare(runs['baseline'], runs['candidate'], args.phase, args.control)
        report[case] = {'runs':runs, 'comparison': comparison}
        (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
