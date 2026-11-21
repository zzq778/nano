"""Compare native cube-center math with the reference on actual approved detections."""
import argparse,json,sys
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from rescuevision.geometry import Geometry
from rescuevision.shape_center import cube_bbox_center
from rescuevision.native import NativeMetric

def main():
    p=argparse.ArgumentParser();p.add_argument('--frames',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    g=Geometry(json.loads((ROOT/'configs/astra_six_estimate.json').read_text()))
    solver=NativeMetric(ROOT/'build/libvision_metric.so',g);results=[]
    for line in Path(args.frames).read_text().splitlines():
        row=json.loads(line)
        if 'objects' not in row or row['repeat']!=0:continue
        image=cv2.imread(str(ROOT/'data/approved_six/raw'/('%02d.png'%row['image_id'])))
        context=solver.prepare(image,row['objects'])
        try:
            for index,obj in enumerate(row['objects']):
                if obj['type']!=3 or obj.get('visible_only'):continue
                reference=cube_bbox_center(g,obj['bbox'],image.shape[:2]);candidate=solver.cube(context,index)
                assert (reference is None)==(candidate is None),'acceptance differs'
                if reference is None:continue
                delta=float(np.linalg.norm(np.array(reference['center_m'])-candidate['center_m']))*1000
                radius=abs(reference['candidate_radius_mm']-candidate['candidate_radius_mm'])
                assert delta<.1 and radius<.1,(delta,radius)
                assert abs(reference['bbox_rms_px']-candidate['bbox_rms_px'])<.01
                results.append(dict(image_id=row['image_id'],candidate_index=index,center_difference_mm=delta,radius_difference_mm=radius))
        finally:solver.release(context)
    assert len(results)>=6,'missing actual six-image comparisons'
    report=dict(comparisons=len(results),maximum_center_difference_mm=max(r['center_difference_mm'] for r in results),results=results,physical_accuracy_verified=False)
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
