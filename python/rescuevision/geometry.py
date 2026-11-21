"""Explicit optical-to-axle geometry. A 2-D anchor alone is never a ground center."""
import math
import cv2
import numpy as np
from .shape_center import cube_bbox_center

class Geometry:
    def __init__(self,config,dimensions=None,tolerance=.35):
        self.config=config
        self.dimensions=dimensions or {0:[.04,.04,.04],1:[.04,.04*math.sqrt(3.)/2.,.04*math.sqrt(2./3.)],2:[.08,.04,.04],3:[.04,.04,.04]}
        self.tolerance=float(tolerance)
        self.ready=False;self.K=None;self.D=None;self.t=None;self.R=None;self.ground=None
        if config.get('matches_input_camera') is not True:
            self.reason='calibration_not_bound_to_input_camera';return
        required=('intrinsics','optical_center_in_axle_frame_m','camera_orientation_deg','ground_plane_z_m')
        if any(config.get(k) is None for k in required):
            self.reason='intrinsics_or_extrinsics_missing';return
        self.rectified=config.get('rectification_status')=='rectified'
        self.pinhole_estimate=config.get('projection_policy')=='pinhole_estimate_unverified_distortion'
        self.use_pinhole=self.rectified or self.pinhole_estimate
        distortion=config.get('distortion_coefficients')
        self.ready=self.use_pinhole or (config.get('rectification_status')=='unrectified' and isinstance(distortion,list) and len(distortion) in (4,5,8,12,14))
        self.reason='calibration_ready' if self.ready else 'distortion_or_rectification_unknown'
        if self.pinhole_estimate:self.reason='measured_K_pinhole_estimate_distortion_unverified'
        if not self.ready:return
        k=config['intrinsics'];self.K=np.array([[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1]],float)
        if not np.isfinite(self.K).all() or self.K[0,0]<=0 or self.K[1,1]<=0:raise ValueError('invalid intrinsics')
        self.D=np.asarray(distortion,float) if distortion is not None else None
        t=config['optical_center_in_axle_frame_m'];self.t=np.array([t['x'],t['y'],t['z']],float)
        angles=config['camera_orientation_deg'];r,p,y=[math.radians(angles[k]) for k in ('roll','pitch_down','yaw')]
        rx=np.array([[1,0,0],[0,math.cos(r),-math.sin(r)],[0,math.sin(r),math.cos(r)]])
        ry=np.array([[math.cos(p),0,math.sin(p)],[0,1,0],[-math.sin(p),0,math.cos(p)]])
        rz=np.array([[math.cos(y),-math.sin(y),0],[math.sin(y),math.cos(y),0],[0,0,1]])
        optical=np.array([[0,0,1],[-1,0,0],[0,-1,0]],float)
        self.R=rz.dot(ry).dot(rx).dot(optical)
        self.ground=float(config['ground_plane_z_m'])

    def project(self,pixel,image_shape,plane_z):
        if tuple(image_shape[:2][::-1])!=tuple(self.config['image_size']):
            raise ValueError('image size differs from calibration; explicit transformed calibration required')
        if not self.ready:return None
        uv=np.asarray(pixel,float)
        if uv.shape!=(2,) or not np.isfinite(uv).all():return None
        if self.use_pinhole:
            ray=np.array([(uv[0]-self.K[0,2])/self.K[0,0],(uv[1]-self.K[1,2])/self.K[1,1],1.])
        else:
            xy=cv2.undistortPoints(uv.reshape(1,1,2),self.K,self.D).reshape(2)
            ray=np.array([xy[0],xy[1],1.])
        direction=self.R.dot(ray)
        if abs(direction[2])<1e-10:return None
        distance=(float(plane_z)-self.t[2])/direction[2]
        if distance<=0:return None
        return (self.t+distance*direction).tolist()

    def project_many(self,pixels,image_shape,plane_z):
        """Project a zone contour in one batch; reject the whole contour on invalid rays."""
        if tuple(image_shape[:2][::-1])!=tuple(self.config['image_size']):
            raise ValueError('image size differs from calibration; explicit transformed calibration required')
        if not self.ready:return None
        uv=np.asarray(pixels,dtype=float)
        if uv.ndim!=2 or uv.shape[1]!=2 or not np.isfinite(uv).all():return None
        if self.use_pinhole:
            xy=(uv-self.K[:2,2])/np.array([self.K[0,0],self.K[1,1]])
        else:
            xy=cv2.undistortPoints(uv.reshape(-1,1,2),self.K,self.D).reshape(-1,2)
        rays=np.column_stack((xy,np.ones(len(xy))))
        directions=rays.dot(self.R.T)
        vertical=directions[:,2]
        if np.any(np.abs(vertical)<1e-10):return None
        distance=(float(plane_z)-self.t[2])/vertical
        if np.any(distance<=0):return None
        return (self.t+distance[:,None]*directions).tolist()

    def estimate(self,observation,image_shape):
        result=dict(position_valid=False,pose_valid=False,accuracy_verified=False,x_mm=None,y_mm=None,
                    yaw_world_deg=None,range_horizontal_mm=None,ground_point_to_axle_distance_mm=None,
                    target_point='vertical_ground_projection_of_3d_geometric_center',reason=self.reason,coordinate_frame='axle_midpoint_XY_mm')
        result['distortion_verified']=not getattr(self,'pinhole_estimate',False) and self.ready
        if not self.ready:return result
        cls=observation['type']
        if cls==1:
            apex=observation.get('observed_apex_px')
            if apex is not None and observation.get('apex_evidence_valid'):
                edge=.04;height=edge*math.sqrt(2./3.)
                p=self.project(apex,image_shape,self.ground+height)
                # The tetrahedron apex projects vertically onto the equilateral base centroid.
                if p is not None and math.hypot(p[0]-self.t[0],p[1]-self.t[1])>1.1*(edge/math.sqrt(3.))*(self.t[2]-self.ground)/height:
                    self.set_position(result,p,'tetrahedron_apex_height_plane_estimate')
                    return result
            pixel=observation.get('observed_ground_center_px')
            if pixel is None:result['reason']='silhouette_center_is_not_ground_center';return result
            p=self.project(pixel,image_shape,self.ground)
            if p is not None:self.set_position(result,p,'observed_ground_center_projected')
            return result
        quad=observation.get('top_corners_px');dims=self.dimensions.get(cls)
        if cls in (0,3) and quad is None and 'bbox' in observation:
            if not observation.get('visible_only') and observation.get('geometry_body_supported') is not False:
                fit=observation['native_cube_fit'] if 'native_cube_fit' in observation else cube_bbox_center(self,observation['bbox'],image_shape)
                if fit is not None:
                    self.set_position(result,fit['center_m'],'known_cube_full_box_center_estimate')
                    result.update(position_quality='shape_center',position_uncertainty_mm=fit['candidate_radius_mm'],bbox_rms_px=fit['bbox_rms_px']);return result
            if not observation.get('near',True):
                bottom=self.project(observation['center_px'],image_shape,self.ground)
                top=self.project(observation['center_px'],image_shape,self.ground+.04)
                if bottom is not None and top is not None:
                    center=(np.array(bottom)+np.array(top))*.5
                    self.set_position(result,center,'far_cube_center_bounded_by_visible_fragment_and_known_height')
                    result['position_uncertainty_mm']=float(np.linalg.norm(np.array(top[:2])-np.array(bottom[:2]))*.5*1000+.04/math.sqrt(2.)*1000)
                    result['position_quality']='coarse_bounded';return result
        if quad is None or dims is None:
            result['reason']='no_observed_top_quadrilateral';return result
        q=np.asarray(quad,float)
        if q.shape!=(4,2) or not np.isfinite(q).all() or not cv2.isContourConvex(q.astype(np.float32)):
            result['reason']='invalid_top_quadrilateral';return result
        points=[self.project(p,image_shape,self.ground+dims[2]) for p in q]
        if any(p is None for p in points):result['reason']='top_rays_do_not_intersect_forward_plane';return result
        points=np.array(points)[:,:2];edges=np.roll(points,-1,axis=0)-points
        lengths=np.linalg.norm(edges,axis=1)
        if min(lengths)<1e-6:result['reason']='degenerate_top_edges';return result
        measured=sorted([(lengths[0]+lengths[2])*.5,(lengths[1]+lengths[3])*.5]);expected=sorted(dims[:2])
        dimensions_ok=all(abs(a-b)/b<=self.tolerance for a,b in zip(measured,expected))
        adjacent_cos=abs(float(np.dot(edges[0],edges[1])/(lengths[0]*lengths[1])))
        opposite_error=max(abs(lengths[0]-lengths[2])/max(lengths[0],lengths[2]),abs(lengths[1]-lengths[3])/max(lengths[1],lengths[3]))
        if not dimensions_ok or adjacent_cos>.35 or opposite_error>.35:
            result.update(reason='projected_top_fails_known_dimensions_or_rectangularity',measured_sides_m=measured);return result
        center=np.mean(points,axis=0)
        self.set_position(result,center,'calibrated_top_quad_candidate');result['footprint_corners_m']=points.tolist()
        if cls==2:
            axis=edges[int(np.argmax(lengths[:2]))] if cls==2 else edges[0]
            period=180 if cls==2 else 90
            result.update(pose_valid=True,yaw_world_deg=math.degrees(math.atan2(axis[1],axis[0]))%period,yaw_period_deg=period)
        return result

    def from_model(self,observation,fit):
        result=dict(position_valid=False,pose_valid=False,accuracy_verified=False,distortion_verified=not self.pinhole_estimate,
                    x_mm=None,y_mm=None,yaw_world_deg=None,range_horizontal_mm=None,ground_point_to_axle_distance_mm=None,
                    target_point='vertical_ground_projection_of_3d_geometric_center',coordinate_frame='axle_midpoint_XY_mm',
                    reason='known_size_rgb_model_status_'+str(fit['status']),model_fit=fit)
        if not self.ready or fit['status'] not in (1,2):return result
        candidates=fit['candidates']
        self.set_position(result,[fit['x'],fit['y']],'known_size_rgb_model_estimate')
        result['position_uncertainty_mm']=max([math.hypot(c['x']-fit['x'],c['y']-fit['y'])*1000 for c in candidates] or [0.])
        result['footprint_corners_m']=fit.get('footprint',[])
        period=180 if observation['type']==2 else 90
        result['yaw_period_deg']=period
        result['yaw_candidates_deg']=[math.degrees(c['yaw_rad'])%period for c in candidates]
        if fit['status']==1:
            result.update(pose_valid=True,yaw_world_deg=math.degrees(fit['yaw_rad'])%period)
        else:result['reason']='known_size_rgb_model_multiple_solutions'
        return result

    def best_effort_position(self,observation,image_shape,result):
        """Retain the best available center; quality is separate from availability."""
        if result['position_valid'] or not self.ready:return result
        cls=observation['type'];dims=self.dimensions.get(cls)
        if cls not in (0,1,2,3) or dims is None:return result
        previous=result.get('reason')
        result.update(accuracy_verified=False,distortion_verified=not self.pinhole_estimate,
                      prior_position_reason=previous)
        best=result.get('model_fit',{}).get('best_effort',{})
        values=[best.get(k) for k in ('x','y','error_px')]
        if all(isinstance(v,(int,float)) and math.isfinite(v) for v in values) and values[2]>=0:
            point=[values[0],values[1],self.ground+dims[2]*.5]
            if self.R.T.dot(np.array(point)-self.t)[2]>.01:
                self.set_position(result,point,'model_best_effort_center')
                result.update(position_quality='best_effort',position_uncertainty_mm=None,
                              position_reprojection_error_px=values[2])
                return result
        anchor=observation.get('top_face_anchor_px') if cls==0 else None
        if anchor is not None:
            point=self.project(anchor,image_shape,self.ground+dims[2])
            if point is not None:
                self.set_position(result,point,'observed_topface_center_known_height_estimate')
                result.update(position_quality='topface_estimate',position_uncertainty_mm=None)
                return result
        pixel=observation.get('center_px')
        if pixel is None:return result
        low=self.project(pixel,image_shape,self.ground)
        high=self.project(pixel,image_shape,self.ground+dims[2])
        if low is None or high is None:return result
        point=(np.array(low)+np.array(high))*.5
        footprint_radius=dims[0]/math.sqrt(3.) if cls==1 else math.hypot(*dims[:2])*.5
        self.set_position(result,point,'visible_center_known_height_and_footprint_bound')
        result.update(position_quality='coarse_bounded',
                      position_uncertainty_mm=(np.linalg.norm(np.array(low[:2])-np.array(high[:2]))*.5+footprint_radius)*1000)
        return result

    def set_position(self,result,point,reason):
        x,y=float(point[0])*1000,float(point[1])*1000
        result.update(position_valid=True,x_mm=int(round(x)),y_mm=int(round(y)),position_xy_m=[float(point[0]),float(point[1])],
                      range_horizontal_mm=math.hypot(x,y),ground_point_to_axle_distance_mm=math.sqrt(x*x+y*y+(self.ground*1000)**2),
                      bearing_deg=math.degrees(math.atan2(y,x)),reason=reason)
        optical=self.R.T.dot(np.array([float(point[0]),float(point[1]),self.ground])-self.t)
        if optical[2]>0:
            if self.use_pinhole:
                # Native metric uses this same pinhole model. Avoid building
                # rotation vectors and calling projectPoints for every target.
                uv=[float(self.K[0,0]*optical[0]/optical[2]+self.K[0,2]),
                    float(self.K[1,1]*optical[1]/optical[2]+self.K[1,2])]
            else:
                uv=cv2.projectPoints(optical.reshape(1,3),np.zeros(3),np.zeros(3),self.K,self.D)[0].reshape(2).tolist()
            result['ground_center_projection_px']=uv
