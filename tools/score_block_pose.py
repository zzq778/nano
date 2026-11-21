#!/usr/bin/env python3
"""Independent measured block-position/angle scoring; truth is never read by inference."""
import argparse
import json
import math
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--frames',required=True);parser.add_argument('--measurements',required=True);parser.add_argument('--output',required=True)
args=parser.parse_args()
frames=[r for r in map(json.loads,Path(args.frames).read_text().splitlines()) if 'objects' in r]
truth=json.loads(Path(args.measurements).read_text())
rows=[]
for reference in truth['measurements']:
    matches=[r for r in frames if r['image_id']==reference['image_id']]
    if not matches:rows.append(dict(reference=reference,status='no_frame'));continue
    frame=matches[-1];objects=[o for o in frame['objects'] if o['candidate_index']==reference['candidate_index'] and o['type']==reference['type']]
    if len(objects)!=1:rows.append(dict(reference=reference,status='object_id_or_class_mismatch'));continue
    metric=objects[0]['metric']
    if not metric['position_valid']:
        rows.append(dict(reference=reference,status='not_measured',reason=metric['reason']));continue
    x,y=reference['x_mm'],reference['y_mm']
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (x,y)):raise ValueError('measured references must be finite physical values, not placeholders')
    row=dict(reference=reference,status='measured',estimated=metric,
             x_error_mm=metric['x_mm']-x,y_error_mm=metric['y_mm']-y,
             center_error_mm=math.hypot(metric['x_mm']-x,metric['y_mm']-y),
             horizontal_range_error_mm=metric['range_horizontal_mm']-math.hypot(x,y))
    if reference['type'] in (0,2) and reference.get('yaw_deg') is not None:
        if not metric.get('pose_valid'):row['angle_status']='not_measured'
        else:
            period=metric['yaw_period_deg'];row['yaw_error_deg']=(metric['yaw_world_deg']-reference['yaw_deg']+period/2)%period-period/2
    rows.append(row)
measured=[r for r in rows if r['status']=='measured'];angle=[r['yaw_error_deg'] for r in measured if 'yaw_error_deg' in r]
report=dict(reference_source=truth.get('source'),measured_positions=len(measured),measured_angles=len(angle),rows=rows,
            center_rmse_mm=math.sqrt(sum(r['center_error_mm']**2 for r in measured)/len(measured)) if measured else None,
            max_center_error_mm=max(r['center_error_mm'] for r in measured) if measured else None,
            yaw_rmse_deg=math.sqrt(sum(v*v for v in angle)/len(angle)) if angle else None,
            note='No valid physical observations means no numeric accuracy claim; black/blue require physical center positions but no orientation.')
Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
