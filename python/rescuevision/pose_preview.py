"""Draw only returned geometric candidates; defaults never manufacture a 3-D box."""
import math
import cv2
import numpy as np

COLORS=((35,230,35),(240,240,240),(20,150,255),(255,170,40))
EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))

def dashed(image,a,b,color):
    a=np.asarray(a,float);b=np.asarray(b,float);steps=max(1,int(np.linalg.norm(b-a)/6))
    for i in range(0,steps,2):cv2.line(image,tuple(np.int32(a+(b-a)*i/steps)),tuple(np.int32(a+(b-a)*min(i+1,steps)/steps)),color,1,cv2.LINE_AA)

def annotate_pose(image,result,geometry):
    out=image.copy();h,w=out.shape[:2]
    for x in range(0,w,20):cv2.line(out,(x,h//2),(min(w-1,x+10),h//2),(220,220,220),1)
    for group in result.get('frontend_diagnostics',{}).get('rejected_yolo_groups',[]):
        a,b,c,d=group['bbox'];color=(150,150,150)
        for p,q in (((a,b),(c,b)),((c,b),(c,d)),((c,d),(a,d)),((a,d),(a,b))):dashed(out,p,q,color)
        cv2.putText(out,'group rejected',(max(0,int(a)),max(12,int(b)-4)),0,.33,color,1,cv2.LINE_AA)
    for obj in result['objects']:
        cls=obj['type'];color=COLORS[cls];box=np.int32(np.round(obj['bbox']));cv2.rectangle(out,tuple(box[:2]),tuple(box[2:]),color,1)
        m=obj['metric'];fit=m.get('model_fit',{});status=fit.get('status')
        if obj.get('top_face_anchor_px'):
            cv2.drawMarker(out,tuple(np.int32(np.round(obj['top_face_anchor_px']))),(220,40,255),cv2.MARKER_CROSS,10,1)
        if geometry.ready and cls in (0,2) and status in (1,2):
            sx,sy,height=geometry.dimensions[cls];co,si=math.cos(fit['yaw_rad']),math.sin(fit['yaw_rad'])
            xyz=np.array([[fit['x']+co*x-si*y,fit['y']+si*x+co*y,z] for z in (geometry.ground,geometry.ground+height) for x,y in ((-sx/2,-sy/2),(sx/2,-sy/2),(sx/2,sy/2),(-sx/2,sy/2))])
            optical=(xyz-geometry.t).dot(geometry.R)
            if optical[:,2].min()>.01:
                uv=cv2.projectPoints(optical,np.zeros(3),np.zeros(3),geometry.K,None if geometry.use_pinhole else geometry.D)[0].reshape(-1,2)
                for a,b in EDGES:
                    if status==1:cv2.line(out,tuple(np.int32(uv[a])),tuple(np.int32(uv[b])),(220,40,255),1,cv2.LINE_AA)
                    else:dashed(out,uv[a],uv[b],(20,200,255))
        point=m.get('ground_center_projection_px')
        if point:
            p=tuple(np.int32(np.round(point)));cv2.drawMarker(out,p,(255,240,30),cv2.MARKER_CROSS,12,1)
        else:
            p=tuple(np.int32(np.round(obj['center_px'])));cv2.drawMarker(out,p,(40,40,255),cv2.MARKER_TILTED_CROSS,10,1)
        label='GKOB'[cls]+str(obj['candidate_index'])
        if cls==2:
            quality=dict(fitted='F',ambiguous_best='M',best_effort='B',default='D').get(m.get('pose_quality'),'D')
            label+=' %.0f%s'%(m.get('yaw_world_deg') or 0.,quality)
        if m.get('position_valid'):label+=' X%d Y%d'%(m['x_mm'],m['y_mm'])
        else:label+=' no XY'
        pos=(max(0,int(box[0])),max(12,int(box[1])-4));cv2.putText(out,label,pos,0,.38,(0,0,0),3,cv2.LINE_AA);cv2.putText(out,label,pos,0,.38,color,1,cv2.LINE_AA)
    return out
