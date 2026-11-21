"""Pose task order and the original frozen-anchor V12 group veto."""
import copy

def pose_order(objects):
    return sorted((i for i,o in enumerate(objects) if o['type']==2),
                  key=lambda i:(not objects[i]['near'],0 if objects[i]['type']==2 else 1,
                                -objects[i]['bbox'][3],i))

def group_veto(results,seeds):
    anchors=copy.deepcopy([r.get('center') for r in results])
    for i,r in enumerate(results):
        if r.get('status')!='edge_candidate':continue
        xa,xb=seeds[i]['x_span'];margin=.08*(xb-xa)
        for j,q in enumerate(anchors):
            if i==j or q is None:continue
            x,y=q
            if not xa+margin<x<xb-margin:continue
            back=r['back_line'][0]*x+r['back_line'][1]
            front=r['front_line'][0]*x+r['front_line'][1]
            if back+.10*(front-back)<y<front-.10*(front-back):
                r.update(proposed_center=r['center'],center=r['initial_center'],status='fallback',
                         reason='rims_span_another_candidate',conflicting_candidate_index=j)
                break
