"""Known upright shape constraints, not visible-pixel centroids. Calibration is supplied explicitly."""
import math
import cv2
import numpy as np

def cube_bbox_center(geometry,bbox,image_shape,edge=.04):
    h,w=image_shape[:2];box=np.asarray(bbox,float)
    if box.shape!=(4,) or not np.isfinite(box).all() or box[0]<2 or box[1]<2 or box[2]>w-3 or box[3]>h-3:return None
    if min(box[2]-box[0],box[3]-box[1])<8:return None
    observed=np.array([box[0],box[2],box[1],box[3]])
    initial=geometry.project([(box[0]+box[2])*.5,(box[1]+box[3])*.5],image_shape,geometry.ground+edge*.5)
    if initial is None:return None
    local=np.array([[x,y,z] for z in (geometry.ground,geometry.ground+edge) for y in (-edge/2,edge/2) for x in (-edge/2,edge/2)],float)
    # Native metric_cube is the production path; this reference fallback is
    # exercised by desktop/color experiments. For pinhole cameras, direct
    # projection avoids a costly cv2.projectPoints call in every iteration.
    if geometry.use_pinhole:
        fx,fy=geometry.K[0,0],geometry.K[1,1]
        cx,cy=geometry.K[0,2],geometry.K[1,2]
        def projected_points(optical):
            z=optical[:,2]
            return np.column_stack((fx*optical[:,0]/z+cx,fy*optical[:,1]/z+cy))
    else:
        def projected_points(optical):
            return cv2.projectPoints(optical,np.zeros(3),np.zeros(3),geometry.K,geometry.D)[0].reshape(-1,2)
    results=[]
    for yaw in np.linspace(0,math.pi/2,19):
        co,si=math.cos(yaw),math.sin(yaw);rotation=np.array([[co,-si],[si,co]])
        offsets=local.copy();offsets[:,:2]=local[:,:2].dot(rotation.T)
        optical_offset=(offsets-geometry.t).dot(geometry.R)
        xy=np.array(initial[:2])
        for _ in range(5):
            optical=(offsets+np.array([xy[0],xy[1],0])-geometry.t).dot(geometry.R)
            if np.min(optical[:,2])<=.01:break
            pixels=projected_points(optical)
            indices=[np.argmin(pixels[:,0]),np.argmax(pixels[:,0]),np.argmin(pixels[:,1]),np.argmax(pixels[:,1])]
            A=[];values=[]
            for side,index in enumerate(indices):
                axis=0 if side<2 else 1;uv=pixels[index].copy();uv[axis]=observed[side]
                if geometry.use_pinhole:q=(uv[axis]-geometry.K[axis,2])/geometry.K[axis,axis]
                else:q=cv2.undistortPoints(uv.reshape(1,1,2),geometry.K,geometry.D).reshape(2)[axis]
                A.append([geometry.R[0,axis]-q*geometry.R[0,2],geometry.R[1,axis]-q*geometry.R[1,2]])
                values.append(q*optical_offset[index,2]-optical_offset[index,axis])
            # This is a four-equation, two-unknown least-squares system.
            # Its 2x2 normal matrix avoids two SVDs per inner iteration.
            matrix=np.array(A);rhs=np.array(values)
            if geometry.use_pinhole:
                gram=matrix.T@matrix
                trace=gram[0,0]+gram[1,1]
                spread=math.hypot(gram[0,0]-gram[1,1],2*gram[0,1])
                min_eigenvalue=(trace-spread)*.5
                # cond(matrix) > 1e6 is equivalent to this eigenvalue ratio.
                if min_eigenvalue<=0 or (trace+spread)>2e12*min_eigenvalue:break
                estimate=np.linalg.solve(gram,matrix.T@rhs)
            else:
                # Preserve the established distorted-camera reference path.
                if np.linalg.cond(matrix)>1e6:break
                estimate=np.linalg.lstsq(matrix,rhs,rcond=-1)[0]
            if not np.isfinite(estimate).all():break
            delta=np.linalg.norm(estimate-xy);xy=estimate
            if delta<1e-6:break
        optical=(offsets+np.array([xy[0],xy[1],0])-geometry.t).dot(geometry.R)
        if np.min(optical[:,2])<=.01:continue
        pixels=projected_points(optical)
        projected=np.array([np.min(pixels[:,0]),np.max(pixels[:,0]),np.min(pixels[:,1]),np.max(pixels[:,1])])
        error=float(np.sqrt(np.mean((projected-observed)**2)))
        if np.linalg.norm(xy-geometry.t[:2])<4:results.append((error,xy.copy()))
    if not results:return None
    best=min(r[0] for r in results)
    if best>max(3.,.06*max(box[2]-box[0],box[3]-box[1])):return None
    candidates=np.array([r[1] for r in results if r[0]<=best+1.])
    center=np.mean(candidates,axis=0);uncertainty=float(np.max(np.linalg.norm(candidates-center,axis=1)))
    return dict(center_m=center.tolist(),candidate_radius_mm=uncertainty*1000,bbox_rms_px=best,latent_yaw_samples=len(results))
