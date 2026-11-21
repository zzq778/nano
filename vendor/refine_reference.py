"""V12 exploratory image-space refinement. Requires supplied candidate spans, not a detector."""
import cv2,numpy as np
import json
from pathlib import Path
cv2.setNumThreads(1)
CONFIG=json.loads((Path(__file__).resolve().parent/'config.json').read_text())['green_thresholds']

def refine_green(image,seed):
    old=np.array([seed['x'],seed['y']],float)
    h,w=image.shape[:2]; xa,xb=seed['x_span'];ya,yb=seed['y_span'];width=xb-xa
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,(35,42,25),(88,255,255))>0
    L=cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2LAB)[:,:,0],(7,7),0).astype(float)
    d=max(2,int(round(width*.018))); step=max(2,int(width/45))
    xx=np.arange(int(xa+.30*width),int(xb-.30*width),step)
    def coherent_line(points):
        if len(points)<7:return None
        pts=np.asarray(points,float);tol=max(3,width*.025);best=None
        # A small deterministic slope grid keeps work bounded and reproducible.
        for slope in np.linspace(-.65,.65,53):
            intercepts=pts[:,1]-slope*pts[:,0]
            hypotheses=intercepts[::max(1,len(intercepts)//70)]
            good=abs(intercepts[:,None]-hypotheses[None,:])<tol
            _,column_index=np.unique(pts[:,0],return_inverse=True)
            votes=np.zeros((column_index.max()+1,len(hypotheses)),bool)
            np.maximum.at(votes,column_index,good)
            scores=votes.sum(axis=0)+.002*(np.minimum(pts[:,2],40)[:,None]*good).sum(axis=0)
            j=int(np.argmax(scores));score=float(scores[j])
            if best is None or score>best[0]:best=(score,slope,float(hypotheses[j]),good[:,j])
        good=best[3];chosen=[]
        for x in np.unique(pts[good,0]):
            pp=pts[good&(pts[:,0]==x)];j=np.argmin(abs(pp[:,1]-(best[1]*x+best[2])));chosen.append(pp[j])
        p=np.array(chosen);line=np.polyfit(p[:,0],p[:,1],1)
        return line,p
    back_candidates=[]
    for x in xx:
        lo=max(d,int(ya-CONFIG['back_expand_width']*width));hi=min(h-d-1,int(ya+CONFIG['back_upper_width']*width),int(old[1]))
        yy=np.arange(lo,hi)
        if len(yy)<3:continue
        rise=L[yy+d,x]-L[yy-d,x]
        entry=mask[yy+d,x]&~mask[yy-d,x]
        near=np.abs(yy-ya)<.22*width
        score=np.where(mask[yy+d,x],rise,-100.)
        score=np.maximum(score,np.where(entry&near,abs(rise),-100.))
        # Keep alternatives for consensus; the strongest edge may belong to a neighbour.
        for _ in range(4):
            bi=int(np.argmax(score));bs=float(score[bi]);by=int(yy[bi])
            if bs<CONFIG['minimum_contrast']:break
            back_candidates.append([x,by,bs]);score[np.abs(yy-by)<max(4,int(width*.025))]=-100.
    back_model=coherent_line(back_candidates)
    backs=[];fronts=[];strengths=[]
    if back_model is not None:
        initial_line,bpoints=back_model
        for x,_,bs in bpoints:
            x=int(x);by=int(round(np.polyval(initial_line,x)))
            low=max(d,by+int(CONFIG['min_depth_width']*width));high=min(h-d-1,by+int(CONFIG['max_depth_width']*width),int(yb+.20*width))
            ys=np.arange(low,high)
            if len(ys)<3:continue
            fall=L[ys-d,x]-L[ys+d,x];fall=np.where(mask[ys-d,x],fall,-100.)
            fi=int(np.argmax(fall));fy=int(ys[fi]);fs=float(fall[fi])
            if fs>=CONFIG['minimum_contrast']:
                backs.append([x,by]);fronts.append([x,fy]);strengths.append([bs,fs])
    out=dict(initial_center=old.tolist(),center=old.tolist(),status='fallback',reason='insufficient_two_rim_evidence',n_profiles=len(backs))
    if len(backs)<CONFIG['minimum_profiles']:return out
    bp=np.array(backs,float);fp=np.array(fronts,float)
    def fit(p):
        line=np.polyfit(p[:,0],p[:,1],1)
        for _ in range(3):
            err=abs(np.polyval(line,p[:,0])-p[:,1]);keep=err<max(4,width*CONFIG['fit_residual_fraction'])
            if keep.sum()<4:break
            line=np.polyfit(p[keep,0],p[keep,1],1)
        return line,abs(np.polyval(line,p[:,0])-p[:,1])
    bl,be=fit(bp);fl,fe=fit(fp);tol=max(4,width*CONFIG['fit_residual_fraction']);keep=(be<tol)&(fe<tol)
    if keep.sum()<CONFIG['minimum_profiles'] or keep.mean()<CONFIG['minimum_survival']:
        out['reason']='inconsistent_rims';out['debug_back']=bp.tolist();out['debug_front']=fp.tolist();return out
    bl=np.polyfit(bp[keep,0],bp[keep,1],1);fl=np.polyfit(fp[keep,0],fp[keep,1],1)
    # The supported horizontal scan interval is an image-space anchor, not a
    # calibrated physical square centre. Do not use brightness-weighted mask mass.
    cx=float(np.median(bp[keep,0]))
    by=float(np.polyval(bl,cx));fy=float(np.polyval(fl,cx));cy=(by+fy)/2
    if not .12*width<fy-by<.75*width:
        out['reason']='implausible_rim_separation';return out
    out.update(center=[cx,cy],status='edge_candidate',reason='two_coherent_signed_rims',back_line=bl.tolist(),front_line=fl.tolist(),
               rear_y=by,front_y=fy,n_profiles=int(keep.sum()),survival=float(keep.mean()),median_contrast=np.median(np.array(strengths)[keep],axis=0).tolist(),
               back_points=bp[keep].tolist(),front_points=fp[keep].tolist(),x_span=[xa,xb])
    return out

def refine_candidate(image,color,candidate,preprocess='raw'):
    """image: BGR frame/crop; candidate: green spans or ROI + initial_center.
    Coordinates in return values always belong to `image`. No object IDs/counts
    are consumed by the estimator. Explicit fallback never silently becomes accepted.
    """
    if preprocess not in ('raw','gaussian3'):
        raise ValueError('only raw or optional gaussian3; no automatic enhancement')
    if image is None or image.ndim!=3:raise ValueError('BGR image required')
    work=image if preprocess=='raw' else cv2.GaussianBlur(image,(3,3),.8)
    if color=='green':
        r=refine_green(work,candidate)
    else:
        from color_core import estimate
        x0,y0,x1,y1=map(int,candidate['roi'])
        if not (0<=x0<x1<=image.shape[1] and 0<=y0<y1<=image.shape[0]):raise ValueError('ROI out of frame')
        z=estimate(work[y0:y1,x0:x1],color)
        initial=candidate.get('initial_center')
        r=dict(initial_center=initial,center=initial,status='fallback',reason=z.get('reason','missing_evidence'),evidence={k:v for k,v in z.items() if k!='mask'})
        if z.get('center_local') is not None:
            proposal=(np.array(z['center_local'])+[x0,y0]).tolist();r['proposed_center']=proposal
            if z['status']=='two_rim_candidate':r.update(center=proposal,status='edge_candidate')
            elif z['status']=='silhouette_only':r.update(center=proposal,status='silhouette_only')
            else:r['status']='weak_fallback'
    r.update(color=color,preprocess=preprocess,coordinate_frame='input_image_px',stage='refined',metric_pose_valid=False)
    return r

def refine_candidates(image,color,candidates,preprocess='raw'):
    """Required group entrypoint: local refinement followed by cross-object check.
    Candidate count and IDs never enter the local optimization. This is a
    consistency veto, not an accuracy score or an object-count constraint.
    """
    results=[refine_candidate(image,color,c,preprocess) for c in candidates]
    anchors=[r['center'] for r in results]
    if color=='green':
        for i,r in enumerate(results):
            if r['status']!='edge_candidate':continue
            xa,xb=r['x_span'];margin=.08*(xb-xa)
            for j,q in enumerate(anchors):
                if i==j or q is None:continue
                x,y=q
                if not xa+margin<x<xb-margin:continue
                back=float(np.polyval(r['back_line'],x));front=float(np.polyval(r['front_line'],x))
                if back+.10*(front-back)<y<front-.10*(front-back):
                    r.update(proposed_center=r['center'],center=r['initial_center'],status='fallback',reason='rims_span_another_candidate',conflicting_candidate_index=j)
                    break
    return results
