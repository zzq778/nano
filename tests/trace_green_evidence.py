"""Locate candidate loss in the current green chain; no truth enters inference."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from rescuevision.topface_native import NativeTopFace
from rescuevision.native import NativeGreen
from rescuevision.green_proposals import GreenFaceProposer
from rescuevision.green_regions import propose_observed_region,refine_local
from rescuevision.green_association import fuse
from rescuevision.scheduling import group_veto


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--kernel-scale',type=float,default=1.)
    args=p.parse_args()
    cfg=json.loads((ROOT/'vendor/config.json').read_text())['green_thresholds']
    cfg['back_expand_width']=.08
    solver=NativeGreen(ROOT/'build/libvision_green.so',cfg)
    topface=NativeTopFace(ROOT/'build/libvision_green.so')
    if not .5<=args.kernel_scale<=2.:
        raise ValueError('bounded offline scale probe requires .5..2')
    def detect(crop):
        factor=args.kernel_scale
        if factor==1.:
            return topface.detect(crop)
        resized=cv2.resize(crop,None,fx=factor,fy=factor,interpolation=cv2.INTER_LINEAR)
        results=topface.detect(resized)
        for q in results:
            q['x']/=factor;q['y']/=factor
            q['x_span']=[v/factor for v in q['x_span']]
            q['y_span']=[v/factor for v in q['y_span']]
        return results
    proposer=GreenFaceProposer(detect)
    rows=[]
    for row in json.loads(Path(args.inputs).read_text())['images']:
        image=cv2.imread(str(ROOT/row['path']))
        if image is None or hashlib.sha256(cv2.cvtColor(image,cv2.COLOR_BGR2RGB).tobytes()).hexdigest()!=row['rgb_sha256']:
            raise ValueError('input RGB mismatch')
        objects=[o for o in row['yolo_objects'] if o['type']==0]
        seeds=propose_observed_region(proposer,image,[o['bbox'] for o in objects])
        results=refine_local(solver,image,seeds)
        before=[dict(r) for r in results]
        group_veto(results,seeds)
        evidence=[dict(seed=s,result=r) for s,r in zip(seeds,results)]
        association=fuse(objects,evidence,True)
        rows.append(dict(key=row['key'],path=row['path'],yolo=objects,seeds=seeds,
            rims_before_veto=before,evidence=evidence,association=association,
            summary=dict(yolo=len(objects),proposals=len(seeds),
                edge_candidates_before_veto=sum(r['status']=='edge_candidate' for r in before),
                trusted=len(association['trusted_faces']),groups=len(association['rejected_yolo_groups']),
                final_count=association['experimental_count'],
                rejected_reasons=dict(Counter(r['reason'] for r in association['rejected_faces'])))))
    Path(args.output).write_text(json.dumps(dict(known_counts_used=False,kernel_scale=args.kernel_scale,images=rows),indent=2))
    print(json.dumps([{r['key']:r['summary']} for r in rows]))


if __name__=='__main__':
    main()
