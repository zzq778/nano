"""Independent r38/r39 local A/B: identical fake boxes, real images, no TensorRT.

Exact wire bytes and every stable result field are checked. The only tolerated
change is <=1e-9 pixel floating-point roundoff in the *diagnostic* reprojection.
This is an x86 CPU regression benchmark, NOT the Jetson production FPS.
"""
import argparse
import copy
import json
import math
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from bench_r38_pipeline import load_modules, make_pipeline, stats


def same_stable(before, after, path='result'):
    if isinstance(before, dict) and isinstance(after, dict):
        if before.keys() != after.keys():
            raise AssertionError('%s changed keys: %s / %s' % (path, before.keys(), after.keys()))
        for name in before:
            if name in ('timings_ms', 'frontend_diagnostics'):
                continue
            same_stable(before[name], after[name], path+'.'+name)
        return
    if isinstance(before, (list,tuple)) and isinstance(after, (list,tuple)):
        if len(before) != len(after):
            raise AssertionError('%s length mismatch' % path)
        for index,(x,y) in enumerate(zip(before,after)):
            same_stable(x,y,'%s[%d]'%(path,index))
        return
    if before != after:
        if 'ground_center_projection_px' in path and isinstance(before,(int,float)) and isinstance(after,(int,float)) and math.isclose(before,after,abs_tol=1e-9,rel_tol=0):
            return
        raise AssertionError('%s changed %r -> %r' % (path,before,after))


def run(base, target, iterations=8):
    old = load_modules(base,'r39_audit_before')
    new = load_modules(target,'r39_audit_after')
    config=json.loads((target/'configs/runtime.json').read_text())
    calibration=json.loads((target/config['calibration']).read_text())
    images=[cv2.imread(str(p)) for p in sorted((target/'data/approved_six/raw').glob('*.png'))]
    images.append(cv2.imread(str(target/'data/zone_contract/astra_frame_000000.png')))
    if len(images)!=7 or any(im is None for im in images):raise RuntimeError('seven Astra fixture images required')
    output={}
    for count in (0,1,15,32):
        before=make_pipeline(old,base,config,calibration,count)
        after=make_pipeline(new,target,config,calibration,count)
        for image_index,image in enumerate(images):
            result_before=before.process(image)
            result_after=after.process(image)
            try:same_stable(result_before,result_after)
            except AssertionError as ex:raise AssertionError('count=%d image=%d: %s'%(count,image_index,ex))
            assert result_before['uart_hex']==result_after['uart_hex']
        times_before=[];times_after=[]
        for i in range(iterations+5):
            image=images[i%len(images)]
            pairs=((before,times_before),(after,times_after)) if i%2 else ((after,times_after),(before,times_before))
            for pipeline,times in pairs:
                started=time.perf_counter_ns();pipeline.process(image)
                if i>=5:times.append((time.perf_counter_ns()-started)/1e6)
        output[count]=dict(before=stats(times_before),after=stats(times_after),parity_images=7,
            median_change_pct=round(100*(statistics.median(times_before)-statistics.median(times_after))/statistics.median(times_before),2))
    return output


def geometry_ab(base,target,iterations=40):
    old=load_modules(base,'r39_geo_before')['geometry']
    new=load_modules(target,'r39_geo_after')['geometry']
    config=json.loads((target/'configs/astra_current.json').read_text())
    old_g=old.Geometry(config);new_g=new.Geometry(config)
    rng=np.random.RandomState(20261121)
    coords=[tuple(map(float,p)) for p in rng.uniform([.2,-.4],[1.5,.4],size=(200,2))]
    before=[];after=[]
    for i in range(iterations+5):
        ordered=((old_g,before),(new_g,after)) if i%2 else ((new_g,after),(old_g,before))
        for geometry,measurements in ordered:
            start=time.perf_counter_ns()
            for xy in coords:geometry.set_position({},xy,'bench')
            if i>=5:measurements.append((time.perf_counter_ns()-start)/1e6)
    return dict(before=stats(before),after=stats(after),positions_per_batch=len(coords),
                median_change_pct=round(100*(statistics.median(before)-statistics.median(after))/statistics.median(before),2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=12)
    args=parser.parse_args()
    cv2.setNumThreads(1)
    output=dict(environment=dict(python=sys.version.split()[0],opencv=cv2.__version__,model_execution='fake deterministic boxes; no TensorRT or native metric',camera='7 recorded Astra images, not a live camera',iterations=args.iterations),
        geometry_set_position=geometry_ab(args.baseline_root,ROOT),pipeline=run(args.baseline_root,ROOT,args.iterations))
    print(json.dumps(output,indent=2))
