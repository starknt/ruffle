#!/usr/bin/env python3
"""Paired native-perf runs. Unfinished traces never count as acceptance."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
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
    if summary['completedNs'] <= 0 or summary.get('droppedGpuSamples', 0):
        raise ValueError('invalid completion time or dropped GPU samples')
    gpu = [r for r in records if r.get('event') == 'gpu' and r.get('frameId') in ids]
    if len({r['frameId'] for r in gpu}) != len(gpu):
        raise ValueError('duplicate GPU frame identity')
    phase_names = {key for r in frames for key in r.get('phaseNs', {})}
    return {
        'case': session['case'], 'backend': session['backend'], 'device': session['device'],
        'paceFps': session.get('paceFps'),
        'profilingEnabled': session.get('profilingEnabled', True),
        'frames': len(frames), 'completedFrameMs': summary['completedNs']/len(frames)/1e6,
        'cpuP95Ms': percentile([r['cpuNs']/1e6 for r in frames], .95),
        'cpuP99Ms': percentile([r['cpuNs']/1e6 for r in frames], .99),
        'gpuP95Ms': percentile([r['ns']/1e6 for r in gpu], .95),
        'gpuCoverage': len(gpu)/len(frames), 'peakRssBytes': summary.get('peakRssBytes'),
        'phaseP95Ms': {key: percentile([r.get('phaseNs', {}).get(key, 0)/1e6 for r in frames], .95) for key in phase_names},
        'workTotals': {key: sum(r.get('work', {}).get(key, 0) for r in frames)
                       for key in {key for r in frames for key in r.get('work', {})}},
    }


def compare(baseline, candidate, phase, control=False):
    if any(r.get('paceFps') is not None for r in baseline + candidate):
        raise ValueError('paced soaks cannot be used for throughput acceptance')
    if len(baseline) != len(candidate) or len(baseline) < 5:
        raise ValueError('acceptance requires at least five complete pairs')
    for before, after in zip(baseline, candidate):
        for key in ('case', 'backend', 'device', 'frames', 'profilingEnabled'):
            if before.get(key) != after.get(key):
                raise ValueError(f'mismatched pair {key}')
    median = statistics.median
    throughput = median(b['completedFrameMs']/c['completedFrameMs']-1 for b,c in zip(baseline,candidate))
    p95 = median(c['cpuP95Ms']/b['cpuP95Ms']-1 for b,c in zip(baseline,candidate))
    p99 = median(c['cpuP99Ms']/b['cpuP99Ms']-1 for b,c in zip(baseline,candidate))
    rss = None if any(not r.get('peakRssBytes') for r in baseline+candidate) else median(
        c['peakRssBytes']/b['peakRssBytes']-1 for b,c in zip(baseline,candidate))
    phases = [(b['phaseP95Ms'].get(phase), c['phaseP95Ms'].get(phase)) for b,c in zip(baseline,candidate)]
    improvement = None if any(not b or c is None for b,c in phases) else median(1-c/b for b,c in phases)
    accepted = (rss is not None and rss <= .05 and p95 <= .03 and p99 <= .03
                and (control or (throughput >= .05 and improvement is not None and improvement >= .10)))
    return {'accepted': accepted, 'throughputGain':throughput, 'cpuP95Regression':p95,
            'cpuP99Regression':p99, 'peakRssGrowth':rss, 'phase':phase, 'phaseP95Gain':improvement}


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
    parser.add_argument('--phase', default='renderBackend')
    parser.add_argument('--control', action='store_true')
    parser.add_argument('--profile-off', action='store_true',
                        help='disable measurement instrumentation in both binaries')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = args.case or [r['name'] for r in tomllib.loads(args.manifest.read_text())['case']]
    binaries = {'baseline': args.baseline.resolve(), 'candidate': args.candidate.resolve()}
    metadata = {label: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                for label,path in binaries.items()}
    metadata['manifestSha256'] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
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
                with path.with_suffix('.log').open('w') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
                runs[label].append(summarize(path))
            print(f'{case}: pair {pair+1}/{args.pairs} complete', flush=True)
        report[case] = {'runs':runs, 'comparison': compare(runs['baseline'], runs['candidate'], args.phase, args.control)}
        (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
