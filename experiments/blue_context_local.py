"""Same positive-blue context algorithm on a bounded crop, with boundary fallback."""
import math
import cv2,numpy as np


def local_blue_context(image,objects,cfg,reference):
    if not objects:return []
    h,w=image.shape[:2];regions=[]
    for obj in objects:
        a,b,c,d=obj['bbox'];pad=max(6.,cfg['context_padding_fraction']*max(c-a,d-b))
        regions.append((a-pad,b-pad,c+pad,d+pad))
    a=max(0,int(math.floor(min(r[0] for r in regions)))-4);b=max(0,int(math.floor(min(r[1] for r in regions)))-4)
    c=min(w,int(math.ceil(max(r[2] for r in regions)))+4);d=min(h,int(math.ceil(max(r[3] for r in regions)))+4)
    if c<=a or d<=b:return reference(image,objects,cfg)
    if (c-a)*(d-b)>=.8*w*h:return reference(image,objects,cfg)
    crop=np.ascontiguousarray(image[b:d,a:c]);hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,tuple(cfg['lower_hsv']),tuple(cfg['upper_hsv']))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    # A connected component crossing an artificial boundary may have a different
    # centroid, area, or connectivity outside. Keep exact full-frame semantics.
    if (a>0 and mask[:,0].any()) or (b>0 and mask[0,:].any()) or (c<w and mask[:,-1].any()) or (d<h and mask[-1,:].any()):
        return reference(image,objects,cfg)
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8);proposals=[]
    for i in range(1,n):
        x,y,bw,bh,area=map(int,stats[i]);x+=a;y+=b
        if area<cfg['minimum_area'] or bw<cfg['minimum_width'] or bh<cfg['minimum_height']:continue
        if max(bw/float(bh),bh/float(bw))>cfg['maximum_aspect']:continue
        context=[]
        for index,region in enumerate(regions):
            x0,y0,x1,y1=region
            if x+bw>=x0 and x<=x1 and y+bh>=y0 and y<=y1:context.append(index)
        if not context:continue
        center=[float(centers[i,0]+a),float(centers[i,1]+b)]
        proposals.append(dict(type=3,bbox=[x,y,x+bw,y+bh],center_px=center,source='positive_blue_color_with_object_context',confidence=None,
            center_kind='visible_fragment_centroid_not_full_object_center',visible_only=True,context_indices=context,mask_area_px=area))
    return proposals
