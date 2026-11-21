"""Full-frame evidence overlay; no invented 3-D pose boxes."""
import cv2

COLORS=((30,230,30),(230,230,230),(0,145,255),(255,160,30),(180,100,255),(255,100,180))
def annotate(image,result):
    h,w=image.shape[:2]
    show=cv2.copyMakeBorder(image,0,28,0,0,cv2.BORDER_CONSTANT,value=(15,15,15))
    for x in range(0,w,20):cv2.line(show,(x,h//2),(min(w-1,x+10),h//2),(210,210,210),1)
    for obj in result.get('objects',[]):
        color=COLORS[obj['type']];x0,y0,x1,y1=map(lambda x:int(round(x)),obj['bbox'])
        cv2.rectangle(show,(x0,y0),(x1,y1),color,1)
        x,y=map(lambda x:int(round(x)),obj['center_px']);status=obj['refinement']['status']
        cross=(255,0,255) if status=='edge_candidate' else color if status=='center_only' else (50,50,255)
        cv2.drawMarker(show,(x,y),cross,cv2.MARKER_CROSS,14,2)
        label='%s%d %s'%('GKOBRC'[obj['type']],obj['candidate_index'],'N' if obj['near'] else 'F')
        cv2.putText(show,label,(max(0,x0),max(15,y0-5)),cv2.FONT_HERSHEY_SIMPLEX,.44,(0,0,0),3)
        cv2.putText(show,label,(max(0,x0),max(15,y0-5)),cv2.FONT_HERSHEY_SIMPLEX,.44,color,1)
        if obj.get('top_corners_px'):
            import numpy as np
            cv2.polylines(show,[np.array(obj['top_corners_px'],np.int32)],True,(255,0,255),1)
    counts=result.get('counts',{})
    text='%s G:%d O:%d B:%d K:%d | %s'%(result.get('mode','ERROR'),counts.get('green',0),counts.get('orange',0),counts.get('blue',0),counts.get('black',0),'UART TARGETS')
    cv2.putText(show,text,(5,h+20),cv2.FONT_HERSHEY_SIMPLEX,.43,(245,245,245),1)
    return show
