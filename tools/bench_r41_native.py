"""Host-only A/B benchmark for production model_geometry.cpp; no Jetson required.

Compile both versions from source and exercise them with identical real Astra RGB
pixels and synthetic model boxes. Reports detailed output parity and alternating
run P50/P95. Does not benchmark metric_prepare (OpenCV) or TensorRT.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PROBE=ROOT/'tests/model_fit_probe_r41.cpp'
BUNDLED_R40=ROOT/'tests/reference/model_geometry_r40.cpp'
BUNDLED_R40_SHA256='de74ef99b87f4b2e081d7c3bc9fff809a48ce81dd840a8f544ec57eb94c8f430'


def prepare(root, temp, synthetic_only=False):
    config=json.loads((root/'configs/astra_current.json').read_text())
    k=config['intrinsics'];t=config['optical_center_in_axle_frame_m'];a=config['camera_orientation_deg']
    r,p,y=[math.radians(a[x]) for x in ('roll','pitch_down','yaw')]
    rx=np.array([[1,0,0],[0,math.cos(r),-math.sin(r)],[0,math.sin(r),math.cos(r)]])
    ry=np.array([[math.cos(p),0,math.sin(p)],[0,1,0],[-math.sin(p),0,math.cos(p)]])
    rz=np.array([[math.cos(y),-math.sin(y),0],[math.sin(y),math.cos(y),0],[0,0,1]])
    optical=np.array([[0,0,1],[-1,0,0],[0,-1,0]],dtype=float)
    rotation=rz.dot(ry).dot(rx).dot(optical)
    # Fit uses a ground-referenced translation, as NativeMetric.prepare does.
    translation=[t['x'],t['y'],t['z']-config['ground_plane_z_m']]
    images=[]
    if not synthetic_only:
        images=[cv2.imread(str(x)) for x in sorted((root/'data/approved_six/raw').glob('*.png'))]
        images.append(cv2.imread(str(root/'data/zone_contract/astra_frame_000000.png')))
        if len(images)!=7 or any(x is None for x in images):
            raise RuntimeError('Seven Astra frames required')
    manifests=[]
    # Fixed bbox positions test a range of exposures and far/near geometry;
    # these are not YOLO-derived and therefore not accuracy ground truth.
    boxset=((0,205,215,258,291),(1,280,245,330,321),
            (2,365,260,429,346),(3,115,270,175,355),
            (0,300,302,382,432),(2,435,205,519,350))
    for i,img in enumerate(images):
        path=temp/('image_%02d.rgb'%i)
        cv2.cvtColor(img,cv2.COLOR_BGR2RGB).tofile(str(path))
        for cls,x0,y0,x1,y1 in boxset:
            manifests.append('%s %s %s %s %s %s' % (path,cls,x0,y0,x1,y1))
    # Additional model-generated RGB scenes exercise accepted fits instead of
    # benchmarking exclusively rejected, arbitrary boxes on real pixels.
    swatches={0:(50,205,40),1:(30,30,30),2:(245,110,25),3:(50,85,205)}
    xs=[.34,.42,.55,.65,.80]
    scenes=0
    for cls in (0,1,2,3):
        for idx,x in enumerate(xs):
            y=([-0.10,0.0,.12,-.08,.04][idx])
            yaw=[0.,.24,.62,.85,1.14][idx]
            if cls==1:
                rad=.04/math.sqrt(3.)
                verts=np.array([(rad,0,0),(-rad*.5,.02,0),(-rad*.5,-.02,0),
                                (0,0,.04*math.sqrt(2./3.))],float)
            else:
                verts=np.array([(a,b,z) for z in (0.,.04) for a in (-.04 if cls==2 else -.02,.04 if cls==2 else .02)
                                for b in (-.02,.02)],float)
            co,si=math.cos(yaw),math.sin(yaw)
            world=verts.copy()
            world[:,0]=x+verts[:,0]*co-verts[:,1]*si
            world[:,1]=y+verts[:,0]*si+verts[:,1]*co
            optical=(world-np.array(translation)).dot(rotation)
            if np.any(optical[:,2]<=.01):continue
            projection=np.column_stack((k['fx']*optical[:,0]/optical[:,2]+k['cx'],
                                       k['fy']*optical[:,1]/optical[:,2]+k['cy']))
            x0,y0=projection.min(axis=0);x1,y1=projection.max(axis=0)
            if x0<6 or y0<6 or x1>634 or y1>474:continue
            frame=np.full((480,640,3),190,np.uint8)
            hull=cv2.convexHull(np.rint(projection).astype(np.int32))
            cv2.fillConvexPoly(frame,hull,swatches[cls])
            path=temp/('synthetic_%02d.rgb'%scenes)
            frame.tofile(str(path))
            manifests.append('%s %s %.6f %.6f %.6f %.6f' % (path,cls,x0,y0,x1,y1))
            # True model boxes vs 1--3 px drift in the detector, the common
            # visible-fragment case with color occlusion, and low-light noise.
            manifests.append('%s %s %.6f %.6f %.6f %.6f' %
                             (path,cls,x0+1.3,y0-1.1,x1+2.7,y1-2.2))
            damaged=frame.copy()
            center=np.mean(projection,axis=0).astype(int)
            cv2.rectangle(damaged,(center[0]-8,center[1]-5),
                          (center[0]+7,center[1]+6),(190,190,190),-1)
            damaged_path=temp/('occluded_%02d.rgb'%scenes)
            damaged.tofile(str(damaged_path))
            manifests.append('%s %s %.6f %.6f %.6f %.6f' %
                             (damaged_path,cls,x0,y0,x1,y1))
            rng=np.random.RandomState(4141+scenes)
            dark=np.clip(frame.astype(np.int16)+rng.randint(-12,13,frame.shape),0,255).astype(np.uint8)
            dark_path=temp/('noise_%02d.rgb'%scenes)
            dark.tofile(str(dark_path))
            manifests.append('%s %s %.6f %.6f %.6f %.6f' %
                             (dark_path,cls,x0,y0,x1,y1))
            scenes+=1
    lines=[('%d %d %.12g %.12g %.12g %.12g' % (640,480,k['fx'],k['fy'],k['cx'],k['cy'])),
           ' '.join('%.12g'%x for x in rotation.reshape(-1)),
           ' '.join('%.12g'%x for x in translation),str(len(manifests))]+manifests
    manifest=temp/'manifest.txt';manifest.write_text('\n'.join(lines)+'\n')
    return manifest


def build(root,probe,target,model_source=None):
    subprocess.run(['g++','-std=c++14','-O3','-DNDEBUG','-I'+str(root/'vendor_model/include'),
                    '-I'+str(root/'include'),str(model_source or root/'vendor_model/src/model_geometry.cpp'),
                    str(root/'vendor_model/src/ground_geometry.cpp'),str(probe),'-o',str(target)],
                   check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)


def run(binary,manifest):
    result=subprocess.run([str(binary),str(manifest)],stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,universal_newlines=True,check=True)
    rows=[]
    for line in result.stdout.splitlines():
        columns=line.split()
        if len(columns)!=18:raise AssertionError('unexpected result '+line)
        rows.append([float(x) for x in columns])
    return rows


def compare(original,candidate):
    if len(original)!=len(candidate):raise AssertionError('case count changed')
    for case,(before,after) in enumerate(zip(original,candidate)):
        # Distinct time measurement is the only intentionally unequal field.
        for idx,(left,right) in enumerate(zip(before[:-1],after[:-1])):
            if left != right:
                raise AssertionError('native mismatch case=%d field=%d: %r != %r' %
                                     (case,idx,left,right))


def distribution(samples):
    samples=sorted(samples)
    return dict(p50_ms=round(statistics.median(samples),4),
                p95_ms=round(samples[int(math.ceil(.95*len(samples)))-1],4))


def main():
    parser=argparse.ArgumentParser()
    baseline_group=parser.add_mutually_exclusive_group(required=True)
    baseline_group.add_argument('--baseline-root',type=Path)
    baseline_group.add_argument('--bundled-r40',action='store_true',
                                help='Use the hash-locked r40 model_geometry.cpp fixture')
    parser.add_argument('--iterations',type=int,default=5)
    parser.add_argument('--synthetic-only',action='store_true',
                        help='Run without private Astra images (public-source smoke test)')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not (1<=args.iterations<=200):parser.error('iterations must be in 1..200')
    model_source=None
    baseline_root=args.baseline_root
    if args.bundled_r40:
        digest=hashlib.sha256(BUNDLED_R40.read_bytes()).hexdigest()
        if digest!=BUNDLED_R40_SHA256:
            parser.error('Bundled r40 geometry source hash mismatch')
        baseline_root=ROOT
        model_source=BUNDLED_R40
    else:
        configuration=baseline_root/'configs/runtime.json'
        if not configuration.is_file() or '-r40-' not in json.loads(configuration.read_text()).get('baseline_id',''):
            parser.error('--baseline-root must be independent original r40 source')
    cv2.setNumThreads(1)
    with tempfile.TemporaryDirectory(prefix='r41_native_model_') as directory:
        temp=Path(directory);manifest=prepare(ROOT,temp,args.synthetic_only)
        old=temp/'original';new=temp/'candidate'
        build(baseline_root,PROBE,old,model_source);build(ROOT,PROBE,new)
        reference=run(old,manifest);candidate=run(new,manifest)
        compare(reference,candidate)
        old_samples=[];new_samples=[]
        old_by_case=[[] for _ in reference];new_by_case=[[] for _ in reference]
        for i in range(args.iterations):
            sequence=((old,old_samples),(new,new_samples)) if i%2 else ((new,new_samples),(old,old_samples))
            for executable,samples in sequence:
                rows=run(executable,manifest)
                compare(reference,rows)
                by_case=old_by_case if executable == old else new_by_case
                for j,row in enumerate(rows):
                    if row[1]==1:
                        samples.append(row[-1]);by_case[j].append(row[-1])
        count=len(reference)
        report=dict(baseline_kind=('bundled_r40_cpp_exact' if args.bundled_r40 else 'independent_r40_tree'),
                    baseline_cpp_sha256=hashlib.sha256((model_source or baseline_root/'vendor_model/src/model_geometry.cpp').read_bytes()).hexdigest(),
                    candidate_cpp_sha256=hashlib.sha256((ROOT/'vendor_model/src/model_geometry.cpp').read_bytes()).hexdigest(),
                    cases=count,valid_ground_rays=sum(row[1]==1 for row in reference),
                    statuses={str(v):sum(row[2]==v for row in reference) for v in sorted(set(row[2] for row in reference))},
                    baseline=distribution(old_samples),candidate=distribution(new_samples),
                    equal_output=True,environment='host standalone C++14 -O3, %s' %
                   ('80 rendered synthetic cases' if args.synthetic_only else
                    '42 real Astra pixels/fixed synthetic boxes + 80 rendered synthetic cases'),
                    warning='Raw model fit only; no TensorRT, CUDA, real model detections, or Jetson timing')
        before=report['baseline']['p50_ms'];after=report['candidate']['p50_ms']
        report['p50_saved_pct']=round(100*(before-after)/before,2) if before else None
        report['fit_status_group']={}
        for status_name, statuses in [('accepted',(1.,2.)),('rejected',(3.,))]:
            indices=[i for i,row in enumerate(reference) if row[2] in statuses]
            a=[x for i in indices for x in old_by_case[i]]
            b=[x for i in indices for x in new_by_case[i]]
            report['fit_status_group'][status_name]=dict(cases=len(indices),
                  baseline=distribution(a) if a else None,
                  candidate=distribution(b) if b else None,
                  matched_case_median_speedup_pct=round(statistics.median(
                      [100*(statistics.median(old_by_case[i])-statistics.median(new_by_case[i])) /
                       statistics.median(old_by_case[i]) for i in indices]),2) if indices else None)
        result=json.dumps(report,indent=2)
        if args.output:args.output.write_text(result+'\n')
        print(result)

if __name__=='__main__':main()
