"""Proof of concept: no YOLO, no ground-truth count fed into algorithm.
Faces from GREEN HSV + LAB L + Canny; gross merged faces split with perspective-local estimated
size and dark-gradient separator peaks. Pixel centers only, not metric ground positions.
"""
from pathlib import Path
import cv2,numpy as np,time,json
cv2.setNumThreads(1)
OUT=Path(__file__).resolve().parent / 'outputs'
SRC={'photo2':'/mnt/data/classic_cv_v3/photo2_real_roi.png', 'photo3':'/mnt/data/classic_cv_v3/photo3_real_roi.png'}

def find_peaks_light(values, min_prominence=1.7, min_distance=24):
    """1D gradient ridge peaks using only OpenCV/NumPy."""
    a=np.asarray(values,dtype=np.float32).reshape(1,-1)
    zz=cv2.GaussianBlur(a,(0,0),4.0).ravel()
    m=len(zz)
    if m<11:return [], []
    possible=[]
    rad=max(32,int(min_distance*1.5))
    for k in range(5,m-5):
        if not (zz[k]>zz[k-1] and zz[k]>=zz[k+1]):continue
        l=float(np.min(zz[max(0,k-rad):k]))
        r=float(np.min(zz[k+1:min(m,k+rad)]))
        prominence=float(zz[k]-max(l,r))
        if prominence>=min_prominence:possible.append((k,prominence))
    kept=[]
    for k,pr in sorted(possible,key=lambda v:-v[1]):
        if all(abs(k-j)>=min_distance for j,_ in kept):kept.append((k,pr))
    kept.sort()
    return [i for i,p in kept],[p for i,p in kept]

def split_run(im,level=156,canny_thick=2):
    hh,ww=im.shape[:2]
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    L=cv2.cvtColor(im,cv2.COLOR_BGR2LAB)[:,:,0].astype('float32')
    green=cv2.inRange(hsv,(35,60,25),(83,255,255))>0
    E=cv2.Canny(cv2.GaussianBlur(np.uint8(L),(5,5),0),25,65)
    m=np.uint8(green & (L>level))*255
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    if canny_thick:
        edges=cv2.dilate(E,np.ones((canny_thick+1,canny_thick+1),np.uint8))>0
        m[edges]=0
    n,labels,stats,cent=cv2.connectedComponentsWithStats(m,8)
    comps=[]
    for i in range(1,n):
        x,y,w,h,area=map(int,stats[i]);
        if area<900 or w<32 or h<19 or area/(w*h)<.14:continue
        # look at top faces only: suppress possible very small dark or bottom side fragments
        comps.append({'idx':i,'x':x,'y':y,'w':w,'h':h,'area':area,'cx':float(cent[i,0]),'cy':float(cent[i,1])})
    # Distinguish extra bright SIDE fragments from top-face evidence.
    # A normal top often has colored cube body immediately below it. When no colored
    # body is visible, allow only a large merged region, which may be front-occluded.
    median_area=float(np.median([r['area'] for r in comps])) if comps else 0.
    kept=[]
    for r in comps:
        x,y,w,h=[r[k] for k in ('x','y','w','h')]
        cx=x+w//2
        yy0=min(hh,y+h+max(7,int(h*.06)))
        yy1=min(hh,yy0+max(16,int(h*.23)))
        xa=max(0,cx-max(8,int(w*.18)))
        xb=min(ww,cx+max(8,int(w*.18)))
        below=green[yy0:yy1,xa:xb]
        frac=float(below.mean()) if below.size else 0.
        r['green_below']=frac
        if frac>0.20 or r['area']>median_area*1.7:
            kept.append(r)
    comps=kept
    if not comps:
        return [],[],m,E,[],{'seed_size':None,'at_y':None}
    # calibrate growth of apparent face size with y among smaller visible top fragments
    # avoid using ground truth counts
    sizes=sorted(comps,key=lambda p:p['area'])
    seed=[r for r in comps if r['h']<115 and r['w']<320 and r['area']>1600]
    if not seed: seed=sizes[:max(1,len(sizes)//2)]
    ys=np.array([r['cy'] for r in seed],float)
    widths=np.array([r['w'] for r in seed],float)
    hs=np.array([r['h'] for r in seed],float)
    # compare face size at different depths using modest perspective scaling
    expected_w0=float(np.median(widths)); expected_h0=float(np.median(hs)); y0=float(np.median(ys))
    def size_at(y):
        fac=np.clip((y+200)/(y0+200),.65,1.7)
        return expected_w0*fac,expected_h0*fac
    # distinguish merged faces: length greatly exceeds a single visible top face
    result=[];analysis=[]
    LS=cv2.GaussianBlur(L,(5,5),0)
    gx=cv2.Sobel(LS,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(LS,cv2.CV_32F,0,1,ksize=3)
    for r in comps:
        x,y,w,h,area=[r[k] for k in ('x','y','w','h','area')]
        ew,eh=size_at(r['cy'])
        # Unlike a raw connected component, one side cannot become many objects without structural ridges
        row_n=int(np.clip(np.floor(h/max(40,eh)*1.22+.35),1,3))
        if area<2500:row_n=1
        row_splits=[]
        if row_n>1:
            gsub=green[y:y+h,x:x+w]; grads=np.maximum(0,-gy[y:y+h,x:x+w])
            v=(grads*gsub).sum(axis=1)/np.maximum(1,gsub.sum(axis=1))
            v=cv2.GaussianBlur(np.float32(v).reshape(1,-1),(0,0),4.0).ravel()
            idx,pr=find_peaks_light(v, min_prominence=1.7, min_distance=max(22,int(.36*eh)))
            idx=[int(k) for k in idx if k>h*.15 and k<h*.87]
            # Prefer structural edges at the predicted face boundary depths.
            # For a 2-face vertical merge, the front face gets larger with depth:
            # the cut is usually beyond one projected upper-face height.
            if row_n==2:
                targets=[float(min(h*.8,1.25*eh))]
            else:
                targets=[h*j/row_n for j in range(1,row_n)]
            for target in targets:
                candidates=[]
                for k in idx:
                    sigma=max(17,eh*.34)
                    nearness=np.exp(-.5*((k-target)/sigma)**2)
                    candidates.append((float(v[k])*(.18+.82*nearness),int(k)))
                if candidates:
                    best=max(candidates)[1]
                    if all(abs(best-t)>max(24,int(.32*eh)) for t in row_splits):
                        row_splits.append(best)
            # If a separation line is not detected, retain one merged unknown
            # instead of inserting an arbitrary split line.
            row_splits=sorted(row_splits)
        cuts=[0]+row_splits+[h]
        info={'rect':[x,y,w,h],'model_expected_wh':[round(ew),round(eh)],'requested_rows':row_n,'found_cuts':[y+z for z in row_splits]}
        for i in range(len(cuts)-1):
            ra,rb=cuts[i],cuts[i+1]
            if rb-ra<18:continue
            ccm=(labels[y+ra:y+rb,x:x+w]==r['idx'])
            coords=np.where(ccm)
            if len(coords[0])<800:continue
            xx=coords[1]+x;yy=coords[0]+y+ra
            # subdivide wide top bands only if strong internal x gradient and width excessive for this depth
            xmin,xmax=int(np.percentile(xx,2)),int(np.percentile(xx,98))
            xwidth=xmax-xmin
            eww,ehh=size_at(y+(ra+rb)/2)
            xs=[]
            if xwidth/max(eww,1)>1.7 and (rb-ra)<1.3*ehh:
                cropmask=green[y+ra:y+rb,xmin:xmax+1]
                gr=np.abs(gx[y+ra:y+rb,xmin:xmax+1])
                prof=(gr*cropmask).sum(axis=0)/np.maximum(1,cropmask.sum(axis=0))
                prof=cv2.GaussianBlur(np.float32(prof).reshape(1,-1),(0,0),4.0).ravel()
                peaks,prominences=find_peaks_light(prof,min_prominence=1.7,min_distance=20)
                possible=[(float(prom),int(pos+xmin)) for pos,prom in zip(peaks,prominences) if .2*xwidth<pos<.8*xwidth]
                if possible:
                    xxcut=max(possible)[1]
                    xs=[xxcut]
            for xa,xb in zip([xmin-1]+xs,xs+[xmax+1]):
                ind=(xx>=xa)&(xx<xb)
                if ind.sum()<550:continue
                pp={'x':round(float(xx[ind].mean()),1),'y':round(float(yy[ind].mean()),1),
                    'n_pixels':int(ind.sum()),'source_comp':[x,y,w,h],
                    'split_from_large':bool(row_splits or xs),
                    'x_span':[int(xx[ind].min()),int(xx[ind].max())],
                    'y_span':[int(yy[ind].min()),int(yy[ind].max())]}
                result.append(pp)
        analysis.append(info)
    return result,analysis,m,E,comps,{'seed_size':[round(expected_w0),round(expected_h0)],'at_y':round(y0)}

def detect_with_isolated_fallback(im, level=None):
    """Brightness + edge face proposals. Only if NO face is found in a compact
    isolated green HSV connected component, use its visible-color centroid.
    Does not estimate hidden objects, scene metric coordinates, or yaw.
    """
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    L=cv2.cvtColor(im,cv2.COLOR_BGR2LAB)[:,:,0]
    green=cv2.inRange(hsv,(35,60,25),(83,255,255))
    if level is None:
        if np.count_nonzero(green)<100:
            return [],[],np.zeros(green.shape,np.uint8),np.zeros(green.shape,np.uint8),[],{}
        q20,q80=np.quantile(L[green>0],[.2,.8])
        level=round(q20+.69*(q80-q20))
    proposals,analysis,m,edges,comps,diag=split_run(im,level=level)
    gm=cv2.morphologyEx(green,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    n,lab,stats,cent=cv2.connectedComponentsWithStats(gm,8)
    for i in range(1,n):
        x,y,w,h,area=map(int,stats[i]);
        if area<300 or w<12 or h<12 or area/(w*h)<.35:continue
        if max(w/max(h,1),h/max(w,1))>2.3:continue
        # A small isolated blob (or one that no top-face seeds explain).
        contains=any(0<=round(p['y'])<lab.shape[0] and 0<=round(p['x'])<lab.shape[1]
                     and lab[round(p['y']),round(p['x'])]==i for p in proposals)
        if not contains:
            cx,cy=cent[i]
            proposals.append({'x':round(float(cx),1),'y':round(float(cy),1),
                'n_pixels':int(area),'source_comp':[x,y,w,h],
                'split_from_large':False,'x_span':[x,x+w-1],'y_span':[y,y+h-1],
                'method':'HSV isolated fallback (not top face)'} )
    return proposals,analysis,m,edges,comps,diag

def overlay(im,props,analysis,m,E,outpath):
    show=im.copy()
    for n,p in enumerate(sorted(props,key=lambda z:(z['y'],z['x'])),1):
        cx,cy=round(p['x']),round(p['y']);col=(0,190,0) if not p['split_from_large'] else (0,160,255)
        cv2.circle(show,(cx,cy),6,col,-1)
        cv2.circle(show,(cx,cy),9,(255,255,255),1)
        cv2.putText(show,str(n),(cx+8,cy-5),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),3)
        cv2.putText(show,str(n),(cx+8,cy-5),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1)
    for a in analysis:
      x,y,w,h=a['rect']
      if a['found_cuts']:
       cv2.rectangle(show,(x,y),(x+w,y+h),(190,0,255),1)
       for yy in a['found_cuts']:cv2.line(show,(x,yy),(x+w,yy),(0,175,255),1)
    cv2.imwrite(str(outpath),show)
    return show

