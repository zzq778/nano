"""Automatic frontends. No counts, historical seeds or image-specific ROIs are inputs."""
import importlib.util
import time
from pathlib import Path
import cv2
import numpy as np

def load_vendor(root,name):
    path=Path(root)/'vendor'/(name+'.py')
    spec=importlib.util.spec_from_file_location('rescue_vendor_'+name,str(path));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def green_seed(box,width,height):
    x0,y0,x1,y1=box
    return dict(x=(x0+x1)*.5,y=(y0+y1)*.5,x_span=[max(0.,x0),min(width-1.,x1)],y_span=[max(0.,y0),min(height-1.,y1)])

def green_observation(seed):
    x,y,w,h=seed['source_comp'];xa,xb=seed['x_span'];ya,yb=seed['y_span']
    return dict(type=0,bbox=[xa,ya,xb+1,yb+1],component_bbox=[x,y,x+w,y+h],
                center_px=[seed['x'],seed['y']],seed=seed,confidence=None,source='original_topface_frontend',
                bbox_kind='individual_visible_support_not_full_cuboid',center_kind=seed.get('method','brightness_face_proposal'))

def color_mask(hsv,cls):
    if cls==1:return cv2.inRange(hsv,(0,0,0),(179,255,80))
    if cls==2:return cv2.inRange(hsv,(0,25,20),(23,255,255))|cv2.inRange(hsv,(173,35,20),(179,255,255))
    if cls==3:return cv2.inRange(hsv,(95,35,18),(140,255,255))
    return cv2.inRange(hsv,(35,42,25),(88,255,255))

def green_color_support(image,objects,hsv=None,output_start=0):
    """One marker per old top-face proposal; partition current visible green, no invented count.

    The same seeded watershed applies to isolated and touching instances. Its box is
    visible color support, never a claimed complete/amodal cube or metric center.
    """
    if not objects:return []
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV) if hsv is None else hsv
    green=cv2.inRange(hsv,(35,60,25),(83,255,255))
    h,w=green.shape;markers=np.where(green>0,0,1).astype(np.int32);seeded=[]
    for index,obj in enumerate(objects):
        seed=obj['seed'];xa,xb=seed['x_span'];ya,yb=seed['y_span']
        a,b=max(0,int(xa)),max(0,int(ya));c,d=min(w,int(xb)+1),min(h,int(yb)+1)
        yy,xx=np.nonzero(green[b:d,a:c])
        if not len(xx):seeded.append(False);continue
        distances=(xx+a-seed['x'])**2+(yy+b-seed['y'])**2
        free=markers[yy+b,xx+a]==0
        if not free.any():seeded.append(False);continue
        distances=np.where(free,distances,np.inf);point=int(np.argmin(distances));x,y=int(xx[point]+a),int(yy[point]+b)
        radius=max(1,int(.12*min(c-a,d-b)));core=np.zeros((d-b,c-a),np.uint8)
        cv2.circle(core,(x-a,y-b),radius,1,-1)
        local=markers[b:d,a:c];use=(core>0)&(green[b:d,a:c]>0)&(local==0);local[use]=index+2;seeded.append(True)
    labels=cv2.watershed(image.copy(),markers);result=[]
    for index,obj in enumerate(objects[output_start:],start=output_start):
        out=dict(obj);out['top_face_bbox']=obj['bbox'][:]
        points=cv2.findNonZero(np.uint8((labels==index+2)&(green>0))) if seeded[index] else None
        if points is not None:
            x,y,bw,bh=cv2.boundingRect(points)
            if bw>=3 and bh>=3:
                out.update(bbox=[x,y,x+bw,y+bh],bbox_kind='green_visible_color_support_not_amodal_cube',source='classic_topface_current_color_support_partition',support_area_px=len(points))
        result.append(out)
    return result

def color_center(image,obj,hsv=None):
    x0,y0,x1,y1=obj['bbox'];h,w=image.shape[:2]
    a,b=max(0,int(x0)),max(0,int(y0));c,d=min(w,int(np.ceil(x1))),min(h,int(np.ceil(y1)))
    if c<=a or d<=b:return obj['center_px'],'bbox_center_fallback'
    region_hsv=hsv[b:d,a:c] if hsv is not None else cv2.cvtColor(image[b:d,a:c],cv2.COLOR_BGR2HSV)
    mask=color_mask(region_hsv,obj['type'])
    n,lab,stats,cent=cv2.connectedComponentsWithStats(mask,8)
    if n<2:return obj['center_px'],'bbox_center_fallback'
    i=1+int(np.argmax(stats[1:,4]))
    if stats[i,4]<max(20,.03*mask.size):return obj['center_px'],'bbox_center_fallback'
    return [float(cent[i,0]+a),float(cent[i,1]+b)],'visible_silhouette_center'

def merge_blue_proposals(objects,proposals):
    result=list(objects)
    for proposal in proposals:
        x,y=proposal['center_px']
        explained=False
        for existing in result:
            if existing['type']!=3:continue
            x0,y0,x1,y1=existing['bbox'];margin=max(2.,.05*max(x1-x0,y1-y0))
            if x0-margin<=x<=x1+margin and y0-margin<=y<=y1+margin:explained=True;break
        if not explained:result.append(proposal)
    return result

def orange_visible_proposals(image,objects,hsv=None):
    """Recover visible orange fragments without full-frame float color arrays."""
    if hsv is None:hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,(4,145,55),(20,255,255))
    candidates=cv2.findNonZero(mask)
    if candidates is not None:
        xy=candidates.reshape(-1,2)
        bgr=image[xy[:,1],xy[:,0]]
        # Identical RGB ratio gates to the original, evaluated only for HSV candidates.
        invalid=(bgr[:,2]<bgr[:,1]*1.38)|(bgr[:,2]<bgr[:,0]*1.55)
        mask[xy[invalid,1],xy[invalid,0]]=0
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8)
    out=[]
    for i in range(1,n):
        x,y,w,h,area=map(int,stats[i])
        if area<700 or w<18 or h<12 or area/float(w*h)<.55:continue
        if max(w/float(h),h/float(w))>7:continue
        cx,cy=centers[i]
        if any(o['type']==2 and o['bbox'][0]-5<=cx<=o['bbox'][2]+5 and o['bbox'][1]-5<=cy<=o['bbox'][3]+5 for o in objects):continue
        out.append(dict(type=2,bbox=[x,y,x+w,y+h],center_px=[float(cx),float(cy)],confidence=None,
                        source='orange_visible_color_recovery',center_kind='visible_fragment_centroid_not_full_object_center',visible_only=True))
    return out

def black_apex_evidence(image,obj,hsv=None):
    h,w=image.shape[:2];x0,y0,x1,y1=obj['bbox'];pad=max(4,int(.08*max(x1-x0,y1-y0)))
    a,b,c,d=max(0,int(x0)-pad),max(0,int(y0)-pad),min(w,int(np.ceil(x1))+pad),min(h,int(np.ceil(y1))+pad)
    region_hsv=hsv[b:d,a:c] if hsv is not None else cv2.cvtColor(image[b:d,a:c],cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(region_hsv,(0,0,0),(179,100,80));mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    if n<2:return None
    i=1+int(np.argmax(stats[1:,4]));top=int(stats[i,1]);height=int(stats[i,3])
    if top<=1 or height<12:return None
    ys=[];left=[];right=[]
    for y in range(top,min(mask.shape[0],top+min(30,max(8,int(height*.35))))):
        xx=np.where(labels[y]==i)[0]
        if len(xx)>=2:ys.append(y);left.append(xx[0]);right.append(xx[-1])
    if len(ys)<6:return None
    l=np.polyfit(ys,left,1);r=np.polyfit(ys,right,1)
    if abs(l[0]-r[0])<.1:return None
    y=(r[1]-l[1])/(l[0]-r[0]);x=l[0]*y+l[1]
    if abs(y-top)>5 or not 0<=x<mask.shape[1]:return None
    return dict(observed_apex_px=[float(x+a),float(y+b)],apex_evidence_valid=True,apex_evidence='two_converging_visible_silhouette_edges')

def blue_context_proposals(image,objects,cfg,hsv=None):
    """Positive visible-color evidence near detected objects; never removes model detections."""
    if not objects:return []
    h,w=image.shape[:2]
    pads=[max(6.,cfg['context_padding_fraction']*max(o['bbox'][2]-o['bbox'][0],o['bbox'][3]-o['bbox'][1])) for o in objects]
    # A component is only kept when it overlaps some object bbox expanded by that
    # object's pad, so every kept component overlaps this union. Half a maximal
    # object dimension of extra margin keeps an occluded fragment whose visible part
    # starts just outside an object's padded box whole; anything still clipped at an
    # interior edge triggers the exact full-frame path below.
    margin=2
    a=max(0,int(min(o['bbox'][0]-p for o,p in zip(objects,pads)))-margin)
    b=max(0,int(min(o['bbox'][1]-p for o,p in zip(objects,pads)))-margin)
    c=min(w,int(max(o['bbox'][2]+p for o,p in zip(objects,pads)))+margin+1)
    d=min(h,int(max(o['bbox'][3]+p for o,p in zip(objects,pads)))+margin+1)
    if (a,b,c,d)==(0,0,w,h):
        return _blue_context_region(image,objects,cfg,0,0,w,h,False,hsv)
    proposals=_blue_context_region(image,objects,cfg,a,b,c,d,True,hsv)
    if proposals is not None:return proposals
    # A kept component touches an interior crop edge and could be truncated.
    return _blue_context_region(image,objects,cfg,0,0,w,h,False,hsv)

def _blue_context_region(image,objects,cfg,ox,oy,xmax,ymax,check_border,hsv=None):
    full_h,full_w=image.shape[:2]
    crop=image[oy:ymax,ox:xmax]
    ch,cw=crop.shape[:2]
    if cw<1 or ch<1:return []
    region_hsv=hsv[oy:ymax,ox:xmax] if hsv is not None else cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(region_hsv,tuple(cfg['lower_hsv']),tuple(cfg['upper_hsv']))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8);proposals=[];touches=False
    for i in range(1,n):
        x,y,bw,bh,area=map(int,stats[i])
        # Clip detection runs before the context test: clipping shrinks the component,
        # which can make it lose the very overlap that would have kept it, so a
        # context check first would silently drop a real fragment. The size gate keeps
        # unrelated specks from forcing a full-frame pass.
        if check_border and area>=cfg['minimum_area']*0.25 and ((x<=0 and ox>0) or (y<=0 and oy>0) or (x+bw>=cw and ox+cw<full_w) or (y+bh>=ch and oy+ch<full_h)):
            touches=True;break
        # Object bboxes live in full-image coordinates; the component is local to the
        # crop, so the context test must use the shifted component rectangle.
        gx,gy=x+ox,y+oy;context=[]
        for index,obj in enumerate(objects):
            a,b,c,d=obj['bbox'];pad=max(6.,cfg['context_padding_fraction']*max(c-a,d-b))
            if gx+bw>=a-pad and gx<=c+pad and gy+bh>=b-pad and gy<=d+pad:context.append(index)
        if not context:continue
        if area<cfg['minimum_area'] or bw<cfg['minimum_width'] or bh<cfg['minimum_height']:continue
        if max(bw/float(bh),bh/float(bw))>cfg['maximum_aspect']:continue
        center=centers[i].tolist()
        proposals.append(dict(type=3,bbox=[x+ox,y+oy,x+bw+ox,y+bh+oy],center_px=[center[0]+ox,center[1]+oy],source='positive_blue_color_with_object_context',confidence=None,
                              center_kind='visible_fragment_centroid_not_full_object_center',visible_only=True,context_indices=context,mask_area_px=area))
    if touches:return None
    return proposals

class ColorFrontend:
    coverage=[0,1,2,3]
    def __init__(self,root,config):
        self.module=load_vendor(root,'topface_lightweight');self.config=config['color_components']
        self.full_support=config.get('green_full_color_support',False)
        if not hasattr(np,'quantile'):np.quantile=lambda values,q:np.percentile(values,np.asarray(q)*100)
    def detect(self,image):
        proposals=self.module.detect_with_isolated_fallback(image)[0]
        objects=[]
        for seed in proposals:
            objects.append(green_observation(seed))
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);self.last_hsv=hsv;cfg=self.config
        if self.full_support:objects=green_color_support(image,objects,hsv)
        for cls in (1,2,3):
            if cls in (1,3):
                lo,hi=cfg['proposal_black_hsv' if cls==1 else 'proposal_blue_hsv']
                raw=cv2.inRange(hsv,tuple(lo),tuple(hi))
            else:raw=color_mask(hsv,cls)
            mask=cv2.morphologyEx(raw,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
            n,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8)
            for i in range(1,n):
                x,y,w,h,area=map(int,stats[i])
                if area<cfg['minimum_area'] or w<cfg['minimum_width'] or h<cfg['minimum_height']:continue
                if cls==1:
                    aspect=w/float(h)
                    if not cfg['black_aspect_range'][0]<=aspect<=cfg['black_aspect_range'][1] or area/float(w*h)<cfg['black_minimum_fill']:continue
                    local=np.uint8(labels[y:y+h,x:x+w]==i)*255
                    contours=cv2.findContours(local,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
                    contour=max(contours,key=cv2.contourArea)
                    hull_area=cv2.contourArea(cv2.convexHull(contour))
                    if hull_area<=0 or cv2.contourArea(contour)/hull_area<cfg['black_minimum_solidity']:continue
                objects.append(dict(type=cls,bbox=[x,y,x+w,y+h],center_px=centers[i].tolist(),confidence=None,source='automatic_hsv_component',center_kind='visible_silhouette_center'))
        return objects

class YoloFrontend:
    coverage=[0,1,2,3]
    def __init__(self,native,blue_supplement=None):self.native=native;self.blue_supplement=blue_supplement
    def detect(self,image):
        started=time.perf_counter()
        raw=self.native.run(image)
        native_done=time.perf_counter()
        self.zones=[dict(color='purple' if o['type']==4 else 'blue',bbox=o['bbox'],confidence=o['confidence']) for o in raw if o['type'] in (4,5)]
        objects=[o for o in raw if o['type']<4]
        for obj in objects:
            x0,y0,x1,y1=obj['bbox'];obj.update(center_px=[(x0+x1)*.5,(y0+y1)*.5],source='current_sixclass_yolo',center_kind='bbox_center')
            if obj['type']==0:obj['seed']=green_seed(obj['bbox'],image.shape[1],image.shape[0])
        # Both color supplements use the same image-to-HSV conversion.
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        self.last_hsv=hsv
        if self.blue_supplement:
            objects=merge_blue_proposals(objects,blue_context_proposals(image,objects,self.blue_supplement,hsv))
        blue_done=time.perf_counter()
        objects+=orange_visible_proposals(image,objects,hsv)
        done=time.perf_counter()
        self.last_diagnostics=dict(timings_ms=dict(native_detector=(native_done-started)*1000,
            hsv_and_blue=(blue_done-native_done)*1000,orange_recovery=(done-blue_done)*1000))
        native_timings=getattr(self.native,'last_timings_ms',None)
        if native_timings is not None:self.last_diagnostics['native_stages_ms']=native_timings
        return objects
