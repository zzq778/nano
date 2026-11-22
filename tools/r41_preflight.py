#!/usr/bin/env python3
"""Offline, read-mostly r41 release gate; never opens a camera/UART or contacts Jetson.

Examples:
  python3 tools/r41_preflight.py --tier quick --baseline-root /path/to/r40 --output /tmp/preflight.json
  python3 tools/r41_preflight.py --tier full --baseline-root /path/to/r40 --output /tmp/preflight.json

Performance checks are correctness gates, not minimum-speed assertions. A slower
sample triggers investigation, not fabricated 'failures' or claimed Jetson FPS.
"""
from __future__ import print_function
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
PROTECTED_SOURCES = (
    'python/rescuevision/uart.py',
    'python/rescuevision/continuous_uart.py',
    'src/astra_camera.cpp',
    'VISION_WIRE_CONTRACT.md',
    'src/detector_bridge.cpp',
)
NATIVE_LIBRARIES = ('libvision_metric.so', 'libvision_green.so',
                    'libvision_orange.so', 'libvision_detector.so')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(65536), b''):
            digest.update(block)
    return digest.hexdigest()


def source_syntax(root):
    inspected = 0
    for sub in ('python', 'tools', 'tests'):
        directory = root / sub
        if not directory.is_dir():
            raise RuntimeError('Missing source directory: ' + sub)
        for path in directory.rglob('*.py'):
            if '__pycache__' in path.parts:
                continue
            # compile() checks syntax without writing __pycache__ into source.
            compile(path.read_bytes(), str(path), 'exec')
            inspected += 1
    if inspected < 20:
        raise RuntimeError('Unexpectedly small Python source set')
    return {'python_files_compiled_without_writes': inspected}


def has_replay_fixtures(root):
    # The public GitHub source intentionally excludes private camera samples.
    # Never turn missing fixtures into fabricated synthetic 'real' images.
    directory = root / 'data' / 'approved_six' / 'raw'
    if len(list(directory.glob('*.png'))) != 6:
        return False
    return (root / 'data' / 'zone_contract' / 'astra_frame_000000.png').is_file()


def source_immutability(root, baseline):
    if baseline is None:
        return None
    checked = {}
    changed = []
    for path in PROTECTED_SOURCES:
        a, b = baseline / path, root / path
        if not a.is_file() or not b.is_file():
            raise RuntimeError('Protected source missing: ' + path)
        value = sha256(a) == sha256(b)
        checked[path] = value
        if not value:
            changed.append(path)
    if changed:
        raise AssertionError('Protected source differs from r40: ' + ', '.join(changed))
    libs = {}
    for lib in NATIVE_LIBRARIES:
        a, b = baseline / 'build' / lib, root / 'build' / lib
        if a.is_file() and b.is_file():
            libs[lib] = ('unchanged' if sha256(a) == sha256(b) else 'changed')
        else:
            libs[lib] = 'not_both_present'
    return {'protected_sources_unchanged': checked,
            'packaged_native_libraries': libs,
            'note': 'C++ source optimization requires a new ARM64 build to take effect; preserving an old .so does not apply it.'}


def execute(argv, cwd, timeout, extra_env=None):
    env = os.environ.copy()
    env.update({'PYTHONDONTWRITEBYTECODE': '1', 'OPENBLAS_NUM_THREADS': '1',
                'OMP_NUM_THREADS': '1'})
    if extra_env:
        env.update(extra_env)
    started = time.monotonic()
    process = subprocess.run(argv, cwd=str(cwd), env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             universal_newlines=True, timeout=timeout)
    return {'command': argv, 'returncode': process.returncode,
            'elapsed_s': round(time.monotonic()-started, 3),
            'output_tail': process.stdout[-10000:]}


def run_gate(tier, root, baseline, report_directory=None):
    result = {'schema_version': 1, 'tier': tier,
              'environment': {'machine': platform.machine(), 'python': platform.python_version(),
                              'system': platform.platform(), 'git': bool(shutil.which('git')),
                              'gpp': bool(shutil.which('g++'))},
              'conditions': ['Host-only: no camera, UART, CUDA or TensorRT inference is exercised.',
                             'All speedups are local CPU estimates; no Jetson FPS is inferred.'],
              'checks': []}

    def check(name, function, required=True):
        begin = time.monotonic()
        try:
            output = function()
            if isinstance(output, dict) and output.get('returncode', 0) != 0:
                raise RuntimeError('exit code {}: {}'.format(output['returncode'], output['output_tail'][-1500:]))
            record = {'name': name, 'status': 'PASS', 'elapsed_s': round(time.monotonic()-begin, 3)}
            if output is not None:
                record['detail'] = output
        except (Exception, KeyboardInterrupt) as err:
            record = {'name': name, 'status': 'FAIL',
                      'elapsed_s': round(time.monotonic()-begin, 3),
                      'error': '{}: {}'.format(type(err).__name__, str(err)[:1500])}
        result['checks'].append(record)
        print('{} {}'.format(record['status'], name), flush=True)
        return record['status'] == 'PASS'

    check('syntax_no_source_writes', lambda: source_syntax(root))
    if baseline:
        check('r40_protected_source_identity', lambda: source_immutability(root, baseline))
    else:
        result['checks'].append({'name': 'r40_protected_source_identity', 'status': 'SKIP',
                                 'reason': 'No independent r40 root provided'})
    check('python_regressions', lambda: execute(
        [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests'],
        root, 180 if tier == 'quick' else 360))
    if shutil.which('g++'):
        with tempfile.TemporaryDirectory(prefix='r41_nms_') as temp:
            check('nms_cpu_parity', lambda: execute(
                ['bash', 'scripts/phone_nms_smoke.sh'], root, 120,
                {'TMPDIR': temp, 'USER': 'r41_preflight'}))
        if baseline:
            iterations = 4 if tier == 'quick' else 25
            benchmark = [sys.executable, 'tools/bench_r41_native.py',
                         '--baseline-root', str(baseline), '--iterations', str(iterations)]
            if not has_replay_fixtures(root):
                benchmark.append('--synthetic-only')
            if report_directory:
                benchmark += ['--output', str(report_directory / 'native_ab.json')]
            check('native_geometry_ab_exact', lambda: execute(benchmark, root, 180))
    else:
        result['checks'].append({'name': 'native_geometry_ab_exact', 'status': 'SKIP',
                                 'reason': 'g++ not available on host'})
    if baseline and has_replay_fixtures(root):
        iterations = 10 if tier == 'quick' else 70
        # Synthetic detector boxes; no native metric or TensorRT.
        check('python_pipeline_ab_exact', lambda: execute(
            [sys.executable, 'tools/bench_r38_pipeline.py',
             '--baseline-root', str(baseline), '--candidate-root', str(root),
             '--iterations', str(iterations)], root, 240))
    else:
        result['checks'].append({'name': 'python_pipeline_ab_exact', 'status': 'SKIP',
                                 'reason': 'Independent r40 root or seven private Astra frames unavailable'})
    result['summary'] = {
        status: sum(x['status'] == status for x in result['checks'])
        for status in ('PASS', 'FAIL', 'SKIP')}
    result['release_gate'] = ('FAIL' if result['summary']['FAIL'] else
                              'HOST_PASS_WITH_SKIPS_AND_DEVICE_UNMEASURED')
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tier', choices=('quick', 'full'), default='quick')
    p.add_argument('--baseline-root', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args(argv)
    baseline = a.baseline_root.resolve() if a.baseline_root else None
    if baseline is not None and not (baseline / 'configs/runtime.json').is_file():
        p.error('--baseline-root must point to an independent extracted r40 project')
    if baseline is not None:
        identity = json.loads((baseline / 'configs/runtime.json').read_text()).get('baseline_id','')
        if '-r40-' not in identity:
            p.error('--baseline-root must be genuine r40 source, got: ' + identity)
    output = a.output.resolve() if a.output else None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
    report = run_gate(a.tier, ROOT, baseline, output.parent if output else None)
    if output:
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'release_gate': report['release_gate'], 'summary': report['summary']},
                     indent=2, sort_keys=True))
    return 1 if report['summary']['FAIL'] else 0


if __name__ == '__main__':
    sys.exit(main())
