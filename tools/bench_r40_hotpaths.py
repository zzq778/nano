"""r39/r40 CPU A/B: seven Astra images, exact output, no TensorRT.

Run with: python3 tools/bench_r40_hotpaths.py --baseline-root /path/to/r39
"""
import argparse
import importlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from bench_r38_pipeline import load_modules, detector_boxes, run as pipeline_ab, stats


def paired(before, after, iterations):
    # Run the same input, alternating execution order to reduce clock/cache bias.
    if before() != after():
        raise AssertionError('Color/zone outputs differ from r39')
    a, b = [], []
    for i in range(iterations+6):
        order = ((before, a), (after, b)) if i%2 else ((after, b), (before, a))
        for function, samples in order:
            start = time.perf_counter()
            function()
            if i >= 6:
                samples.append((time.perf_counter()-start)*1000)
    return dict(before=stats(a), after=stats(b),
                saved_pct=round(100*(statistics.median(a)-statistics.median(b)) /
                                statistics.median(a), 2))


def benchmark(baseline, iterations):
    old = load_modules(baseline, 'r40_bench_old')
    new = load_modules(ROOT, 'r40_bench_new')
    old_z = importlib.import_module('r40_bench_old.zones')
    new_z = importlib.import_module('r40_bench_new.zones')
    config = json.loads((ROOT/'configs/runtime.json').read_text())
    geometry = new['geometry'].Geometry(json.loads((ROOT/config['calibration']).read_text()))
    images = [cv2.imread(str(p)) for p in sorted((ROOT/'data/approved_six/raw').glob('*.png'))]
    images.append(cv2.imread(str(ROOT/'data/zone_contract/astra_frame_000000.png')))
    if len(images) != 7 or any(frame is None for frame in images):
        raise RuntimeError('This local benchmark requires the seven private Astra input images')
    boxes = detector_boxes(15)
    result = dict(environment=dict(machine=platform.machine(),python=platform.python_version(),
                   opencv=cv2.__version__,opencv_threads=cv2.getNumThreads(),
                   detector='fixed synthetic boxes; no TensorRT',image_count=7),
                  color_stage={}, zone_stage={})
    for index, image in enumerate(images):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        result['color_stage']['orange_%d'%index] = paired(
            lambda:old['frontends'].orange_visible_proposals(image, boxes, hsv),
            lambda:new['frontends'].orange_visible_proposals(image, boxes, hsv), iterations)
        result['color_stage']['blue_%d'%index] = paired(
            lambda:old['frontends'].blue_context_proposals(image, boxes,
                config['blue_context_supplement'], hsv),
            lambda:new['frontends'].blue_context_proposals(image, boxes,
                config['blue_context_supplement'], hsv), iterations)
    fixture = json.loads((ROOT/'data/zone_contract/fixture.json').read_text())
    image = cv2.imread(str(ROOT/'data/zone_contract'/fixture['image']))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for selected in ('blue','purple'):
        detections=[fixture['detection']]
        result['zone_stage']['real_%s'%selected] = paired(
            lambda:old_z.estimate_zone_regions(image, detections, geometry, selected, hsv),
            lambda:new_z.estimate_zone_regions(image, detections, geometry, selected, hsv), iterations)
    result['pipeline'] = pipeline_ab(baseline, ROOT, iterations)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=50)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.iterations<1:p.error('iterations must be positive')
    cv2.setNumThreads(1)
    report=benchmark(args.baseline_root,args.iterations)
    text=json.dumps(report,indent=2,sort_keys=True)
    if args.output:args.output.write_text(text+'\n')
    print(text)
