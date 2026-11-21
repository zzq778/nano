"""Preserved V10 orange/blue/black local estimator; manual ROIs, no detector."""

import cv2,numpy as np

HSV_RANGE={'orange':((0,25,20),(23,255,255)),'blue':((95,35,18),(140,255,255))}

def comp_roi(roi,color):
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    if color=='black':
        raw=cv2.inRange(hsv,(0,0,0),(179,255,80))
    else:
        raw=cv2.inRange(hsv,*HSV_RANGE[color])
        if color=='orange':raw|=cv2.inRange(hsv,(173,35,20),(179,255,255))
    raw=cv2.morphologyEx(raw,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    n,lab,s,ct=cv2.connectedComponentsWithStats(raw,8)
    if n<2: return None,None
    i=1+np.argmax(s[1:,4]);
    return (np.uint8(lab==i)*255),tuple(map(int,s[i,:4]))

def estimate(roi,color):
    mask,bb=comp_roi(roi,color)
    if mask is None:return {'status':'reject','reason':'no color component'}
    x,y,w,h=bb;H,W=mask.shape
    basic={'bbox_local':[x,y,w,h], 'mask_area_px':int(cv2.countNonZero(mask))}
    if color=='black':
        ys,xs=np.where(mask>0)
        cx=float(xs.mean());cy=float(ys.mean())
        return {**basic,'status':'silhouette_only','reason':'black tetrahedron has no horizontal square top',
                'center_local':[round(cx,1),round(cy,1)],'mask':mask}
    if w<25 or h<25: return {**basic,'status':'reject','reason':'too small component','mask':mask}
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    lab=cv2.cvtColor(roi,cv2.COLOR_BGR2LAB)
    sm=cv2.GaussianBlur(hsv[:,:,1],(5,7),1.4).astype(np.float32)
    lm=cv2.GaussianBlur(lab[:,:,0],(5,7),1.4).astype(np.float32)
    # shared method: use V10 concept of two contrasting rims per vertical profile.
    # Orange upper face can be saturated in V; saturation rises at top/front boundary.
    # Blue top/front usually differentiated more weakly by falling Lab lightness.
    grad=(sm if color=='orange' else -lm)
    d=max(3,min(8,round(h*.02)))
    bpts=[];fpts=[];strength=[];step=max(3,round(w/40))
    for xx in range(x+int(.25*w),x+int(.75*w),step):
        row=np.where(mask[:,xx]>0)[0];row=row[(row>=y)&(row<=y+int(.7*h))]
        if len(row)==0:continue
        back=int(row[0]);lo=max(d,y+int(.1*h),back+int(.09*h));hi=min(H-d-1,y+int(.6*h),back+int(.62*h))
        if hi<=lo:continue
        cand=np.arange(lo,hi)
        measure=grad[cand+d,xx]-grad[cand-d,xx]
        j=int(np.argmax(measure));front=int(cand[j]);val=float(measure[j])
        if val>8 and .07*h<front-back<.58*h:
            bpts.append((xx,back));fpts.append((xx,front));strength.append(val)
    n_before=len(fpts)
    if n_before<6:
        return {**basic,'status':'reject','reason':'insufficient signed two-rim profiles', 'n_profiles':n_before,'mask':mask}
    bp=np.asarray(bpts,dtype=np.float32);fp=np.asarray(fpts,dtype=np.float32);st=np.asarray(strength)
    fit=np.polyfit(fp[:,0],fp[:,1],1)
    resid=np.abs(fp[:,1]-np.polyval(fit,fp[:,0]))
    keep=resid<max(8,.06*h)
    if keep.sum()<6:
        return {**basic,'status':'reject','reason':'inconsistent front rim', 'n_profiles':n_before,'mask':mask}
    fl=np.polyfit(fp[keep,0],fp[keep,1],1);bl=np.polyfit(bp[keep,0],bp[keep,1],1)
    cx=x+.5*w;re=float(np.polyval(bl,cx));fr=float(np.polyval(fl,cx));depth=fr-re
    scatter=float(np.median(resid[keep]));contrast=float(np.median(st[keep]));survival=float(keep.mean());n=int(keep.sum())
    reasons=[]
    if y<=2:reasons.append('ROI clips upper face')
    if depth/h<.10 or depth/h>.58:reasons.append('implausible depth fraction')
    if contrast<18:reasons.append('weak front-edge contrast')
    if scatter>max(8,.05*h):reasons.append('incoherent front-edge line')
    if survival<.7:reasons.append('unstable across horizontal samples')
    status='two_rim_candidate' if not reasons else 'weak_or_reject'
    return {**basic,'status':status,'reason':'; '.join(reasons) if reasons else 'two signed rims accepted',
      'center_local':[round(cx,1),round(.5*(re+fr),1)],'rims_at_mid':[round(re,1),round(fr,1)],
      'top_depth_fraction':round(depth/h,3),'edge_contrast':round(contrast,1),'scatter_px':round(scatter,1),
      'survival':round(survival,3),'n_profiles':n,'back_line':bl.tolist(),'front_line':fl.tolist(),
      'back_points':bp[keep].astype(int).tolist(),'front_points':fp[keep].astype(int).tolist(),'mask':mask}
