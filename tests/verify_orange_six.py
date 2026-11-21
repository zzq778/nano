"""36 current-image ROI comparisons: six approved frames, three detected oranges, two preprocess modes."""
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
import cv2
import numpy as np
from rescuevision.native import NativeDetector,NativeOrange
from rescuevision.frontends import load_vendor
cfg=json.loads((ROOT/'configs/instance_candidate.json').read_text())
manifest_path=ROOT/'data/approved_six/manifest.json';manifest=json.loads(manifest_path.read_text())
native=NativeOrange(ROOT/'build/libvision_orange.so');reference=load_vendor(ROOT,'color_core')
detector=NativeDetector(ROOT/'build/libvision_detector.so',cfg['model'],cfg['semantic_thresholds'],cfg['nms_iou'])
records=[];errors=[]
try:
 for row in manifest['images']:
    image=cv2.imread(str(manifest_path.parent/row['path']));h,w=image.shape[:2]
    assert hashlib.sha256(cv2.cvtColor(image,cv2.COLOR_BGR2RGB).tobytes()).hexdigest()==row['rgb_sha256']
    boxes=[d for d in detector.run(image) if d['type']==2]
    for preprocess in ('raw','gaussian3'):
     work=image if preprocess=='raw' else cv2.GaussianBlur(image,(3,3),.8)
     for index,box in enumerate(boxes):
        x0,y0,x1,y1=box['bbox'];px=max(3,int(.08*(x1-x0)));py=max(3,int(.08*(y1-y0)))
        a,b,c,d=max(0,int(x0)-px),max(0,int(y0)-py),min(w,int(np.ceil(x1))+px),min(h,int(np.ceil(y1))+py)
        roi=work[b:d,a:c];before=roi.copy()
        expected={k:v for k,v in reference.estimate(roi,'orange').items() if k!='mask'};actual=native.estimate(roi)
        assert np.array_equal(before,roi),'orange native mutated input'
        differences=[]
        for key in set(expected)|set(actual):
            if key not in expected or key not in actual:differences.append([key,'missing']);continue
            x,y=expected[key],actual[key]
            if isinstance(x,str):
                if x!=y:differences.append([key,x,y])
            elif np.shape(x)!=np.shape(y) or not np.allclose(x,y,atol=1e-5,rtol=0):differences.append([key,x,y])
        record=dict(image=row['id'],orange_index=index,preprocess=preprocess,differences=differences,python=expected,native=actual)
        records.append(record)
        if differences:errors.append(record)
        print('orange_parity',row['id'],index,preprocess,'differences',len(differences),flush=True)
finally:detector.close()
(ROOT/'build/orange_parity.json').write_text(json.dumps(dict(records=records,errors=errors,tolerance=1e-5),indent=2))
sys.exit(1 if errors else 0)
