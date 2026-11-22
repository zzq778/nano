#!/usr/bin/env python3
"""Summarize previously recorded tegrastats lines; this does not run tegrastats.

For an authorized later stress test, collect tegrastats separately on Nano and
pass that text file here. Temperatures/loads differ between JetPack versions.
"""
from __future__ import print_function
import argparse
import json
import re
from pathlib import Path
import statistics
import sys

try:
    from r41_runtime_report import distribution as ms_distribution
except ImportError:
    from tools.r41_runtime_report import distribution as ms_distribution


def values_distribution(values, unit):
    # Shared quantiles use *_ms field names; relabel for physical units.
    raw = ms_distribution(values)
    if raw is None:return None
    result = {'unit': unit}
    for name, value in raw.items():
        result[name[:-3] if name.endswith('_ms') else name] = value
    return result

GPU=re.compile(r'\bGR3D_FREQ\s+(\d+)%')
RAM=re.compile(r'\bRAM\s+(\d+)/(\d+)MB\b')
TEMP=re.compile(r'\b([A-Za-z][A-Za-z0-9_]*)@(-?\d+(?:\.\d+)?)C\b')
POWER=re.compile(r'\bVDD_IN\s+(\d+)/(\d+)\b')


def analyze(lines, expected_interval_s=1.0, max_lines=200000):
    rows=0
    gpu=[];ram=[];temp={};power=[];invalid_lines=0
    for line in lines:
        rows+=1
        if rows>max_lines:raise ValueError('Too many tegrastats lines')
        has_data=False
        match=GPU.search(line)
        if match:
            gpu.append(float(match.group(1)));has_data=True
        match=RAM.search(line)
        if match:
            used,total=map(int,match.groups())
            if used<=total and total>0:ram.append(float(used));has_data=True
        for match in TEMP.finditer(line):
            temp.setdefault(match.group(1),[]).append(float(match.group(2)));has_data=True
        match=POWER.search(line)
        if match:
            power.append(float(match.group(1)));has_data=True
        if not has_data:invalid_lines+=1
    valid_time=rows*expected_interval_s
    report={'schema_version':1,'lines_seen':rows,
            'configured_sample_interval_s':expected_interval_s,
            'approx_duration_s':round(valid_time,1),
            'gpu_utilization_pct':values_distribution(gpu,'percent'),
            'ram_used_mb':values_distribution(ram,'MB'),
            'temperatures_c':{k:values_distribution(v,'C') for k,v in sorted(temp.items())},
            'vdd_in_instant_mw':values_distribution(power,'mW'),
            'unrecognized_lines':invalid_lines,
            'warnings':[]}
    if rows < 600/expected_interval_s:
        report['warnings'].append('Less than 10 minutes of samples; not sufficient for sustained thermal testing.')
    if len(gpu)<rows//2:
        report['warnings'].append('GPU utilization not reported on many rows; no GPU load conclusion.')
    report['warnings'].append('Frequency changes alone do not prove thermal throttling; correlate clocks, loads, temperatures and power mode.')
    report['warnings'].append('tegrastats VDD_IN measures board power, not isolated GPU consumption.')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log',type=Path,required=True)
    p.add_argument('--interval-ms',type=int,default=1000)
    p.add_argument('--output',type=Path)
    a=p.parse_args(argv)
    if a.interval_ms<100 or a.interval_ms>10000:p.error('interval must be 100..10000 ms')
    with a.log.open('r',encoding='utf-8',errors='replace') as stream:
        report=analyze(stream,a.interval_ms/1000.)
    text=json.dumps(report,indent=2,sort_keys=True)+'\n'
    if a.output:a.output.write_text(text)
    print(text)
    return 0

if __name__=='__main__':sys.exit(main())
