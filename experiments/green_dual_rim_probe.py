"""Fresh raw-image V12 verification of YOLO and classic proposals; pixel-only, no count truth."""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from rescuevision.native import NativeGreen,ReferenceGreen
import numpy as np
from rescuevision.frontends import green_seed
from rescuevision.scheduling import group_veto

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);p.add_argument('--backend',choices=['native','reference'],default='native');p.add_argument('--face-back-expand',type=float);args=p.parse_args()
    inputs=json.loads(Path(args.inputs).read_text());assert inputs['contains_ground_truth'] is False
    config=json.loads((ROOT/'vendor/config.json').read_text())['green_thresholds'];solver=NativeGreen(ROOT/'build/libvision_green.so',config) if args.backend=='native' else ReferenceGreen(ROOT)
    face_config=dict(config)
    if args.face_back_expand is not None:face_config['back_expand_width']=args.face_back_expand
    face_solver=NativeGreen(ROOT/'build/libvision_green.so',face_config) if args.backend=='native' else ReferenceGreen(ROOT)
    if args.backend=='reference':face_solver.reference.CONFIG=face_config
    rows=[]
    for row in inputs['images']:
        image=cv2.imdecode(np.frombuffer((ROOT/row['path']).read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        if image is None or hashlib.sha256(cv2.cvtColor(image,cv2.COLOR_BGR2RGB).tobytes()).hexdigest()!=row['rgb_sha256']:raise ValueError('input identity mismatch')
        started=time.perf_counter();context=solver.prepare(image,'raw');face_context=face_solver.prepare(image,'raw');sets={}
        try:
            proposals=dict(row['classic_sets']);proposals['yolo']=[green_seed(o['bbox'],image.shape[1],image.shape[0]) for o in row['yolo_objects']]
            for name,seeds in proposals.items():
                active,ctx=(solver,context) if name=='yolo' else (face_solver,face_context)
                before=[active.local(ctx,s) for s in seeds];after=copy.deepcopy(before);group_veto(after,seeds)
                sets[name]=[dict(seed=s,result=r,before_veto=q) for s,r,q in zip(seeds,after,before)]
        finally:solver.release(context);face_solver.release(face_context)
        record=dict(key=row['key'],image_size=[image.shape[1],image.shape[0]],yolo_objects=row['yolo_objects'],sets=sets,
                    raw_V12_ms=(time.perf_counter()-started)*1000,calibration_used=False)
        rows.append(record);print(row['key'],{k:sum(r['result']['status']=='edge_candidate' for r in v) for k,v in sets.items()},flush=True)
    Path(args.output).write_text(json.dumps(dict(images=rows,counts_not_provided=True,measurement='pixel_only',backend=args.backend,face_thresholds=face_config,timing_scope='native_or_host_reference_probe_only'),indent=2))

if __name__=='__main__':main()
