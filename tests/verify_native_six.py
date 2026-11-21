"""Native session parity against the original Python V12, on the approved six only."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
import cv2
import numpy as np
from rescuevision.native import NativeGreen
from rescuevision.frontends import load_vendor
from rescuevision.scheduling import group_veto

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    if not hasattr(np,'quantile'):np.quantile=lambda values,q:np.percentile(values,np.asarray(q)*100)
    front=load_vendor(ROOT,'topface_lightweight');reference=load_vendor(ROOT,'refine_reference')
    cfg=json.loads((ROOT/'vendor/config.json').read_text())['green_thresholds']
    native=NativeGreen(ROOT/'build/libvision_green.so',cfg)
    path=Path(args.manifest);manifest=json.loads(path.read_text());records=[];errors=[]
    for row in manifest['images']:
        image=cv2.imread(str(path.parent/row['path']))
        assert hashlib.sha256(cv2.cvtColor(image,cv2.COLOR_BGR2RGB).tobytes()).hexdigest()==row['rgb_sha256']
        seeds=front.detect_with_isolated_fallback(image)[0]
        for preprocess in ('raw','gaussian3'):
            before=image.copy();expected=reference.refine_candidates(image,'green',seeds,preprocess)
            context=native.prepare(image,preprocess)
            try:actual=[native.local(context,s) for s in seeds]
            finally:native.release(context)
            group_veto(actual,seeds)
            assert np.array_equal(image,before),'native preprocessing changed the input'
            differences=[]
            for i,(a,b) in enumerate(zip(expected,actual)):
                for key in ('status','reason','n_profiles','conflicting_candidate_index'):
                    if a.get(key)!=b.get(key):differences.append([i,key])
                for key in ('center','initial_center','proposed_center','rear_y','front_y','survival','back_line','front_line','back_points','front_points','debug_back','debug_front','median_contrast','x_span'):
                    if key not in a and key not in b:continue
                    if key not in a or key not in b or np.shape(a[key])!=np.shape(b[key]) or not np.allclose(a[key],b[key],rtol=0,atol=1e-6):differences.append([i,key])
            if len(expected)!=len(actual):differences.append(['count'])
            if differences:errors.append(dict(image=row['id'],preprocess=preprocess,differences=differences))
            records.append(dict(image=row['id'],preprocess=preprocess,candidates=len(seeds),differences=differences))
            print('parity',row['id'],preprocess,'differences',len(differences),flush=True)
    Path(args.output).write_text(json.dumps(dict(records=records,errors=errors,tolerance=1e-6),indent=2))
    return 1 if errors else 0
if __name__=='__main__':sys.exit(main())
