"""Reproducible offline CPU A/B: fake model boxes, real Astra pixels and pipeline.

This measures Python color processing/geometry/UART only. It does NOT measure
TensorRT, native metric fit, actual camera acquisition or physical serial I/O.

Usage: python tools/bench_r38_pipeline.py --baseline-root /path/to/jetson_vision_r37
"""
import argparse
import copy
import importlib
import json
import statistics
import sys
import time
import types
from pathlib import Path

import cv2

cv2.setNumThreads(1)
ROOT=Path(__file__).resolve().parents[1]


def load_modules(root,namespace):
    package=types.ModuleType(namespace)
    package.__path__=[str(root/'python/rescuevision')]
    sys.modules[namespace]=package
    return {name:importlib.import_module(namespace+'.'+name)
            for name in ('pipeline','frontends','geometry')}


def detector_boxes(count):
    # Fixed geometry-only stand-ins, not manually annotated real model detections.
    # Diverse sizes/locations cover 0, 1, 15, 32 input count pressure.
    return [dict(type=i%4,bbox=[float(40+(i%8)*69),float(100+(i//8)*60),
                                float(72+(i%8)*69),float(143+(i//8)*60)],confidence=.83)
            for i in range(count)]


def make_pipeline(modules,root,config,calibration,count):
    class Detector:
        def run(self,image):return copy.deepcopy(detector_boxes(count))
    p=modules['pipeline'].Pipeline.__new__(modules['pipeline'].Pipeline)
    p.root=root;p.config=config;p.mode='yolo';p.detector=None;p.green=None
    p.orange=None;p.orange_native=None;p.metric_solver=None;p.known_size_requested=True
    p.geometry=modules['geometry'].Geometry(calibration,{int(k):v for k,v in config['dimensions_m'].items()},
                                            config['geometry_dimension_relative_tolerance'])
    p.frontend=modules['frontends'].YoloFrontend(Detector(),config['blue_context_supplement'])
    p.zone_detector=p.frontend;p.default_yaw={int(k):v for k,v in config.get('default_yaw_deg',{}).items()}
    return p


def stable_result(result):
    result=copy.deepcopy(result)
    result.pop('timings_ms',None)
    result.pop('frontend_diagnostics',None)
    return result


def stats(timings):
    samples=sorted(timings)
    return dict(p50_ms=round(statistics.median(samples),3),
                p95_ms=round(samples[(95*len(samples)+99)//100-1],3),
                samples=len(samples))


def run(baseline,candidate,iterations):
    old=load_modules(baseline,'rvbench_old')
    new=load_modules(candidate,'rvbench_new')
    config=json.loads((candidate/'configs/runtime.json').read_text())
    calibration=json.loads((candidate/config['calibration']).read_text())
    images=[cv2.imread(str(p)) for p in sorted((candidate/'data/approved_six/raw').glob('*.png'))]
    images.append(cv2.imread(str(candidate/'data/zone_contract/astra_frame_000000.png')))
    if not all(im is not None for im in images) or len(images)!=7:
        raise RuntimeError('expected seven shipped Astra fixture images')
    output={}
    for count in (0,1,15,32):
        p_old=make_pipeline(old,baseline,config,calibration,count)
        p_new=make_pipeline(new,candidate,config,calibration,count)
        cases=list(enumerate(images))
        for index,image in cases:
            expected=stable_result(p_old.process(image))
            actual=stable_result(p_new.process(image))
            if expected!=actual:
                mismatches=[key for key in expected if expected[key]!=actual.get(key)]
                raise AssertionError('count=%d image=%d changed %s'%(count,index,mismatches))
        a,b=[],[]
        for i in range(iterations+10):
            image=images[i%len(images)]
            pairs=((p_old,a),(p_new,b)) if i%2 else ((p_new,b),(p_old,a))
            for pipeline,times in pairs:
                start=time.perf_counter()
                pipeline.process(image)
                elapsed=(time.perf_counter()-start)*1000
                if i>=10:times.append(elapsed)
        output[count]=dict(before=stats(a),after=stats(b),
                           p50_saved_pct=round(100*(statistics.median(a)-statistics.median(b))/statistics.median(a),2),
                           parity_images=len(images))
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline-root',required=True,type=Path)
    parser.add_argument('--candidate-root',type=Path,default=ROOT)
    parser.add_argument('--iterations',type=int,default=70)
    args=parser.parse_args()
    print(json.dumps(run(args.baseline_root,args.candidate_root,args.iterations),indent=2))


if __name__=='__main__':main()
