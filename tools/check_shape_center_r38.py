"""Equivalence and CPU A/B for optional Python cube-box center fallback.

The real production metric_cube native path does not use this solver.
"""
import argparse
import importlib
import json
import math
import statistics
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def load(root, name):
    package=types.ModuleType(name)
    package.__path__=[str(root/'python/rescuevision')]
    sys.modules[name]=package
    g=importlib.import_module(name+'.geometry')
    sc=importlib.import_module(name+'.shape_center')
    calibration=json.loads((root/'configs/astra_current.json').read_text())
    return g.Geometry(calibration),sc.cube_bbox_center


def samples(geometry):
    rng=np.random.default_rng(38)
    boxes=[]
    edge=.04
    for _ in range(260):
        x,y,z=rng.uniform(.12,1.25),rng.uniform(-.25,.25),geometry.ground
        yaw=rng.uniform(0,math.pi/2)
        co,si=math.cos(yaw),math.sin(yaw)
        corners=np.array([[x+dx*co-dy*si,y+dx*si+dy*co,z+dz]
                          for dx in (-edge/2,edge/2) for dy in (-edge/2,edge/2) for dz in (0.,edge)])
        optical=(corners-geometry.t).dot(geometry.R)
        if np.any(optical[:,2]<=.01):continue
        uv=np.column_stack((geometry.K[0,0]*optical[:,0]/optical[:,2]+geometry.K[0,2],
                            geometry.K[1,1]*optical[:,1]/optical[:,2]+geometry.K[1,2]))
        box=[float(uv[:,0].min()),float(uv[:,1].min()),float(uv[:,0].max()),float(uv[:,1].max())]
        if box[0]>=2 and box[1]>=2 and box[2]<637 and box[3]<477 and min(box[2]-box[0],box[3]-box[1])>=8:
            boxes.append(box)
    return boxes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline-root',required=True,type=Path)
    args=parser.parse_args()
    old_geom,old=load(args.baseline_root,'sc_old')
    new_geom,new=load(ROOT,'sc_new')
    boxes=samples(new_geom)
    if len(boxes)<50:raise AssertionError('insufficient candidate geometries')
    a,b=[],[]
    for i,box in enumerate(boxes):
        pair=((old,old_geom,a),(new,new_geom,b)) if i%2 else ((new,new_geom,b),(old,old_geom,a))
        out=[]
        for fn,g,timings in pair:
            start=time.perf_counter()
            result=fn(g,box,(480,640))
            timings.append((time.perf_counter()-start)*1000)
            out.append(result)
        x,y=out
        if x is None or y is None:
            if x is not None or y is not None:raise AssertionError('validity changed for box %d: %s vs %s'%(i,x,y))
        else:
            for key in ('center_m','candidate_radius_mm','bbox_rms_px','latent_yaw_samples'):
                if not np.allclose(x[key],y[key],rtol=1e-9,atol=1e-8):
                    raise AssertionError('different %s box=%d old=%s new=%s'%(key,i,x[key],y[key]))
    print(json.dumps(dict(boxes=len(boxes),non_null=sum(old(old_geom,box,(480,640)) is not None for box in boxes),
                          before_median_ms=round(statistics.median(a),3),after_median_ms=round(statistics.median(b),3),
                          saved_pct=round(100*(statistics.median(a)-statistics.median(b))/statistics.median(a),2)),indent=2))


if __name__=='__main__':main()
