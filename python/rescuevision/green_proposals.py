"""Automatic component-local proposals with an explicitly supplied face kernel."""
import cv2,numpy as np

class GreenFaceProposer:
    def __init__(self, detect_faces):
        self._detect_faces = detect_faces
    def propose(self,image,boxes,hsv=None):
        h,w=image.shape[:2]
        if hsv is None:hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        green=cv2.inRange(hsv,(35,60,25),(83,255,255))
        green=cv2.morphologyEx(green,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        n,labels,stats,_=cv2.connectedComponentsWithStats(green,8);result=[]
        for label in range(1,n):
            x,y,bw,bh,area=[int(v) for v in stats[label]]
            if area<150 or bw<8 or bh<8:continue
            local=[];matched=[]
            for box in boxes:
                a,b,c,d=box;intersection=max(0,min(x+bw,c)-max(x,a))*max(0,min(y+bh,d)-max(y,b))
                if intersection/max(1,min(bw*bh,(c-a)*(d-b)))>.15:local.append(c-a);matched.append(box)
            if not local:continue
            pad=max(6,int(.12*max(bw,bh)))
            if len(local)==1:
                # Additional support beyond the only detector box justifies work.
                # This schedules inspection; it does not prove a group or count.
                a,b,c,d=matched[0]
                if a-pad<=x and b-pad<=y and c+pad>=x+bw and d+pad>=y+bh:continue
            expected=float(np.median(local))
            scale=float(np.clip(96/max(8,expected),.5,4))
            pad=max(6,int(.12*max(bw,bh)));a,b=max(0,x-pad),max(0,y-pad);c,d=min(w,x+bw+pad),min(h,y+bh+pad)
            crop=cv2.resize(image[b:d,a:c],None,fx=scale,fy=scale,interpolation=cv2.INTER_LINEAR)
            for seed in self._detect_faces(crop):
                q=dict(seed);q['x']=float(seed['x']/scale+a);q['y']=float(seed['y']/scale+b)
                q['x_span']=[float(z/scale+a) for z in seed['x_span']];q['y_span']=[float(z/scale+b) for z in seed['y_span']]
                cx,cy=int(round(q['x'])),int(round(q['y']))
                # Retain proposals belonging to this automatically detected component only.
                xa,xb=[int(round(v)) for v in q['x_span']];ya,yb=[int(round(v)) for v in q['y_span']]
                yy,xx=np.nonzero(labels[max(0,ya):min(h,yb+1),max(0,xa):min(w,xb+1)]==label)
                if not len(xx):continue
                xx=xx+max(0,xa);yy=yy+max(0,ya);distance=(xx-q['x'])**2+(yy-q['y'])**2;nearest=int(np.argmin(distance))
                if distance[nearest]>(.35*max(3,xb-xa,yb-ya))**2:continue
                if not(0<=cx<w and 0<=cy<h and labels[cy,cx]==label):q['x'],q['y']=float(xx[nearest]),float(yy[nearest])
                q.update(source_comp=[x,y,bw,bh],component_label=label,proposal_scale=scale,method=seed.get('method','original_topface_on_auto_component_roi'))
                result.append(q)
        return result
