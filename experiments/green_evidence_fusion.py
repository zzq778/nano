"""Count-free experimental association of current-image top faces and YOLO.

Does not replace the runtime baseline. Added entries are pixel face observations,
not complete cube boxes, ground centers, or validated metric poses.
"""
import argparse,json,math
from pathlib import Path

FACE_OPTIONS=dict(component_roi=True,target_width=96,minimum_area=225,
                  minimum_width=16,minimum_height=8,band_minimum_pixels=180,
                  face_minimum_pixels=140,canny_low=10,canny_high=30,
                  canny_blur=3,canny_thick=1)


def same_face(a,b):
    x,y=a['result']['center'];u,v=b['result']['center']
    wa=a['seed']['x_span'][1]-a['seed']['x_span'][0]
    wb=b['seed']['x_span'][1]-b['seed']['x_span'][0]
    if abs(x-u)>.30*min(wa,wb):return False
    mid=(x+u)*.5
    def interval(q):
        r=q['result'];return [r[k][0]*mid+r[k][1] for k in ('back_line','front_line')]
    a0,a1=interval(a);b0,b1=interval(b)
    return max(0,min(a1,b1)-max(a0,b0))>.45*min(a1-a0,b1-b0)


def fuse(yolo,faces,reject_groups=False):
    trusted=[];rejected=[]
    for i,q in enumerate(faces):
        r=q['result'];s=q['seed'];width=s['x_span'][1]-s['x_span'][0]
        why=None
        if r['status']!='edge_candidate':why=r['reason']
        elif s['y_span'][1]-s['y_span'][0]>.85*width:why='tall_surface_not_separated_top_face'
        elif abs(r['center'][1]-s['y'])>.25*width:why='refinement_left_face_support'
        elif any(same_face(q,t['evidence']) for t in trusted):why='same_measured_rims'
        if why:rejected.append(dict(face=i,reason=why))
        else:trusted.append(dict(face=i,evidence=q))
    groups=[]
    if reject_groups:
        for j,obj in enumerate(yolo):
            a,b,c,d=obj['bbox'];width,height=c-a,d-b
            # Whole top-only observations can be short. For normal full-body
            # boxes, lower overlapping faces may belong to a foreground cube;
            # use the upper 60% for conservative positive group evidence.
            top_limit=d if height<.8*width else b+.6*height
            contained=[t['face'] for t in trusted if a<=t['evidence']['result']['center'][0]<=c
                and b<=t['evidence']['result']['center'][1]<=top_limit]
            if len(contained)>=2:groups.append(dict(yolo=j,independent_top_faces=contained,reason='multiple_distinct_measured_top_faces'))
    rejected_indices=set(g['yolo'] for g in groups)
    # Maximum one-to-one association; positively identified group parents never
    # occupy a single-instance slot. Their children remain independent evidence.
    edges=[]
    for face in trusted:
        x,y=face['evidence']['result']['center'];options=[]
        for j,obj in enumerate(yolo):
            if j in rejected_indices:continue
            a,b,c,d=obj['bbox'];w,h=c-a,d-b
            if a-.08*w<=x<=c+.08*w and b-.08*h<=y<=d+.08*h:
                cost=((x-(a+c)*.5)/max(w,1))**2+((y-(b+.25*h))/max(h,1))**2
                options.append((cost,j))
        edges.append([j for _,j in sorted(options)])
    assigned={}
    def match(i,visited):
        for j in edges[i]:
            if j in visited:continue
            visited.add(j)
            if j not in assigned or match(assigned[j],visited):assigned[j]=i;return True
        return False
    for i in sorted(range(len(trusted)),key=lambda k:len(edges[k])):match(i,set())
    matched=set(assigned.values());additions=[t for i,t in enumerate(trusted) if i not in matched]
    return dict(baseline_count=len(yolo),experimental_count=len(yolo)-len(groups)+len(additions),
                trusted_faces=trusted,associations=[dict(yolo=j,face=trusted[i]['face']) for j,i in sorted(assigned.items())],
                added_faces=additions,rejected_faces=rejected,rejected_yolo_groups=groups,
                interpretation='YOLO instances plus unmatched current-image top-face observations; no metric claim')


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);p.add_argument('--variant',default='local_fine_edges');args=p.parse_args()
    data=json.loads(Path(args.inputs).read_text());rows=[]
    for r in data['images']:
        out=fuse(r['yolo_objects'],r['sets'][args.variant]);out.update(key=r['key'],image_size=r['image_size']);rows.append(out)
        print(r['key'],out['baseline_count'],'->',out['experimental_count'],'added',[t['face'] for t in out['added_faces']])
    Path(args.output).write_text(json.dumps(dict(variant=args.variant,ground_truth_used=False,production_baseline_changed=False,images=rows),indent=2))

if __name__=='__main__':main()
