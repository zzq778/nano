"""Bounded current-image work; coordinate transforms preserve the V12 algorithm."""
import copy,math
import numpy as np
from rescuevision.frontends import green_color_support


def shift_seed(s,dx,dy):
    q=dict(s,x=s['x']+dx,y=s['y']+dy)
    q['x_span']=[v+dx for v in s['x_span']];q['y_span']=[v+dy for v in s['y_span']]
    return q


def propose_observed_region(proposer,image,boxes):
    if not boxes:return []
    pad=max(8,int(.3*max(max(q[2]-q[0],q[3]-q[1]) for q in boxes)))
    a=max(0,int(min(q[0] for q in boxes))-pad);b=max(0,int(min(q[1] for q in boxes))-pad)
    c=min(image.shape[1],int(math.ceil(max(q[2] for q in boxes)))+pad)
    d=min(image.shape[0],int(math.ceil(max(q[3] for q in boxes)))+pad)
    seeds=proposer.propose(np.ascontiguousarray(image[b:d,a:c]),[[x0-a,y0-b,x1-a,y1-b] for x0,y0,x1,y1 in boxes])
    result=[]
    for s in seeds:
        q=shift_seed(s,a,b);x,y,w,h=s['source_comp'];q['source_comp']=[x+a,y+b,w,h];result.append(q)
    return result


def shift_result(r,dx,dy):
    q=dict(r)
    for k in ('center','initial_center','proposed_center'):
        if q.get(k) is not None:q[k]=[q[k][0]+dx,q[k][1]+dy]
    for k in ('back_line','front_line'):
        if k in q:q[k]=[q[k][0],q[k][1]+dy-q[k][0]*dx]
    for k in ('back_points','front_points','debug_back','debug_front'):
        if k in q:q[k]=[[v[0]+dx,v[1]+dy]+v[2:] for v in q[k]]
    for k in ('rear_y','front_y'):
        if k in q:q[k]+=dy
    if 'x_span' in q:q['x_span']=[v+dx for v in q['x_span']]
    return q


def refine_local(solver,image,seeds):
    groups={};results=[None]*len(seeds);h,w=image.shape[:2]
    for i,s in enumerate(seeds):groups.setdefault(tuple(s['source_comp']),[]).append(i)
    for indices in groups.values():
        width=max(seeds[i]['x_span'][1]-seeds[i]['x_span'][0] for i in indices);pad=int(math.ceil(.8*width))+8
        a=max(0,int(min(seeds[i]['x_span'][0] for i in indices))-8);c=min(w,int(math.ceil(max(seeds[i]['x_span'][1] for i in indices)))+9)
        b=max(0,int(min(seeds[i]['y_span'][0] for i in indices))-pad);d=min(h,int(math.ceil(max(seeds[i]['y_span'][1] for i in indices)))+pad)
        crop=np.ascontiguousarray(image[b:d,a:c]);ctx=solver.prepare(crop,'raw')
        try:
            for i in indices:results[i]=shift_result(solver.local(ctx,shift_seed(seeds[i],-a,-b)),a,b)
        finally:solver.release(ctx)
    return results


def added_support_local(image,green,additions):
    if not additions:return []
    h,w=image.shape[:2];output=[]
    # One crop per source component, rather than one full-frame watershed.
    groups={}
    for obj in additions:groups.setdefault(tuple(obj['seed']['source_comp']),[]).append(obj)
    for comp,added in groups.items():
        x,y,bw,bh=comp;a,b=max(0,int(x)-6),max(0,int(y)-6);c,d=min(w,int(x+bw)+6),min(h,int(y+bh)+6)
        local=[]
        for o in green+added:
            x0,y0,x1,y1=o['bbox']
            if o not in added and not (x0<c and x1>a and y0<d and y1>b):continue
            q=copy.deepcopy(o);q['bbox']=[x0-a,y0-b,x1-a,y1-b];q['seed']=shift_seed(o['seed'],-a,-b);local.append(q)
        supports=green_color_support(np.ascontiguousarray(image[b:d,a:c]),local)[-len(added):]
        for obj,original in zip(supports,added):
            out=dict(original);x0,y0,x1,y1=obj['bbox'];out.update(bbox=[x0+a,y0+b,x1+a,y1+b],bbox_kind=obj.get('bbox_kind'),support_area_px=obj.get('support_area_px'))
            output.append(out)
    return output
