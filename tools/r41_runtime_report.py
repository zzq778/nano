#!/usr/bin/env python3
"""Analyze existing vision runtime logs; does NOT launch Astra, TensorRT or UART.

Use real camera/UART logs for end-to-end latency and throughput. A recorded
replay does not imply a live camera result; output states the evidence type.
"""
from __future__ import print_function
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

STAGES = ('compute', 'source_read', 'read_completion_age_before_write',
          'runtime_before_log', 'native_metric_prepare', 'native_metric_orange_fit',
          'native_metric_cube_fit', 'detection', 'center_only',
          'scheduled_refinement_and_veto', 'metric_validation', 'uart_encode')


def pct(values, q):
    ordered = sorted(values)
    return ordered[max(0, int(math.ceil(q * len(ordered))) - 1)]


def distribution(values):
    if not values:
        return None
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Negative/nonfinite timing value')
    return {'n': len(values), 'mean_ms': round(statistics.mean(values), 4),
            'p50_ms': round(statistics.median(values), 4),
            'p90_ms': round(pct(values, .90), 4),
            'p95_ms': round(pct(values, .95), 4),
            'p99_ms': round(pct(values, .99), 4),
            'max_ms': round(max(values), 4)}


def count_group(count):
    if count == 0: return '0'
    if count == 1: return '1'
    if count < 15: return '2-14'
    if count == 15: return '15'
    if count < 32: return '16-31'
    if count == 32: return '32'
    return '>32'


def analyze(lines, summary=None, warmup=0, limit=200000):
    if warmup < 0 or limit < 1:
        raise ValueError('Invalid warmup / limit')
    result_rows = []
    wall_rows = []
    source_drops = 0
    skipped_sequences = 0
    error_rows = 0
    camera_reopen_rows = 0
    for line_number, line in enumerate(lines, 1):
        if line_number > limit:
            raise ValueError('Log exceeds limit: {} lines'.format(limit))
        if not line.strip(): continue
        try:
            row = json.loads(line)
        except ValueError as error:
            raise ValueError('Bad JSON on line {}: {}'.format(line_number, error))
        if not isinstance(row, dict):
            raise ValueError('Log row {} is not an object'.format(line_number))
        kind = row.get('record_kind')
        if kind == 'source_frame_drop':
            source_drops += 1
        if kind == 'runtime_error':
            error_rows += 1
        if kind == 'camera_reopened':
            camera_reopen_rows += 1
        if kind == 'wall_timing':
            wall_rows.append(row)
        if isinstance(row.get('timings_ms'), dict) and 'objects' in row:
            result_rows.append(row)
    records = result_rows[warmup:]
    frame_errors = [r for r in records if r.get('frame_error') is not None]
    successful_records = [r for r in records if r.get('frame_error') is None]
    stages = {}
    for key in STAGES:
        values = []
        for r in successful_records:
            number = r['timings_ms'].get(key)
            if number is None: continue
            if type(number) not in (int, float):
                raise ValueError('Invalid timing for stage ' + key)
            values.append(float(number))
        if values: stages[key] = distribution(values)
    native_sources = {
        'native_detector': ('timings_ms', 'native_detector'),
        'hsv_and_blue': ('timings_ms', 'hsv_and_blue'),
        'orange_recovery': ('timings_ms', 'orange_recovery'),
        'native_preprocess': ('native_stages_ms', 'preprocess'),
        'native_tensorrt_and_transfers': ('native_stages_ms', 'tensorrt_and_transfers'),
        'native_postprocess': ('native_stages_ms', 'postprocess'),
    }
    for name, keys in native_sources.items():
        values = []
        for r in successful_records:
            diag = r.get('frontend_diagnostics', {})
            if not isinstance(diag, dict):
                continue
            yolo = diag.get('yolo_diagnostics', diag)
            if not isinstance(yolo, dict):
                continue
            source = yolo.get(keys[0], {})
            value = source.get(keys[1]) if isinstance(source, dict) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
        if values:
            stages[name] = distribution(values)
    categories = {}
    for r in records:
        skipped_sequences += max(0, int(r.get('dropped_camera_frames') or 0))
    for r in successful_records:
        counts = r.get('counts', {})
        if not isinstance(counts, dict):continue
        n = sum(v for v in counts.values() if type(v) is int and v >= 0)
        group = count_group(n)
        value = r['timings_ms'].get('compute')
        if type(value) in (int,float):
            categories.setdefault(group, []).append(float(value))
    durations = {key: distribution(values) for key,values in sorted(categories.items())}
    wall = [float(r['wall_with_json_ms']) for r in wall_rows[warmup:]
            if type(r.get('wall_with_json_ms')) in (int, float)]
    summary = summary or {}
    real_camera = any(r.get('source_kind') == 'camera_read_completion_not_exposure' for r in records)
    physical_uart = any(r.get('transport', {}).get('kind') == 'physical_uart'
                        for r in records if isinstance(r.get('transport'), dict))
    pacing = summary.get('includes_rate_pacing')
    report = {'schema_version': 1,
              'frames_seen': len(result_rows), 'frames_analyzed': len(records),
              'successful_frames_analyzed': len(successful_records),
              'frame_processing_error_records': len(frame_errors),
              'camera_reopen_records': camera_reopen_rows,
              'warmup_excluded': min(warmup, len(result_rows)),
              'wall_timing_rows': len(wall_rows),
              'source_drop_records': source_drops,
              'camera_sequences_skipped_in_analyzed_frames': skipped_sequences,
              'runtime_error_records': error_rows,
              'real_camera_evidence': real_camera,
              'physical_uart_evidence': physical_uart,
              'stage_latency_ms': stages, 'compute_by_object_count_ms': durations,
              'wall_with_json_ms': distribution(wall),
              'summary_observed_loop_hz': summary.get('observed_loop_hz'),
              'includes_requested_rate_pacing': pacing,
              'conditions': [
                  'Application timing is not physical exposure-to-actuation latency.',
                  'native_tensorrt_and_transfers combines TRT inference and transfers, not isolated GPU time.',
                  'A logged request rate or observed loop Hz with pacing is not unconstrained max FPS.',
                  'Do not compare two sessions without confirming same camera, model, scene and power mode.'
              ],
              'warnings': []}
    if not records:
        report['warnings'].append('No frame-level results; FPS/latency cannot be established from this input.')
    if len(records) < 100:
        report['warnings'].append('Fewer than 100 data frames; P99 is statistically fragile.')
    if not real_camera:
        report['warnings'].append('No live Astra source evidence in selected records.')
    if not physical_uart:
        report['warnings'].append('No physical UART write evidence in selected records.')
    if 'native_tensorrt_and_transfers' not in stages:
        report['warnings'].append('TensorRT+transfer timings absent in this log.')
    if error_rows or source_drops or frame_errors or camera_reopen_rows:
        report['warnings'].append('Frame processing errors, source drops, camera reopens or runtime errors occurred; investigate before performance claims.')
    if frame_errors:
        report['warnings'].append('Per-stage compute distributions exclude frames with frame_error; wall timings and loop Hz may still include them.')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=Path, required=True, help='Existing frames.jsonl; never opens devices')
    parser.add_argument('--summary', type=Path, help='Optional run_summary.json from same session')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--max-lines', type=int, default=200000)
    parser.add_argument('--output', type=Path)
    a=parser.parse_args(argv)
    if not a.frames.is_file():parser.error('--frames does not exist')
    summary=json.loads(a.summary.read_text()) if a.summary else {}
    with a.frames.open('r', encoding='utf-8') as handle:
        report=analyze(handle, summary=summary, warmup=a.warmup, limit=a.max_lines)
    text=json.dumps(report, sort_keys=True, indent=2)+'\n'
    if a.output:a.output.write_text(text)
    print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
