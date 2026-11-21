"""Full-frame detections first, then explicit near/color pose scheduling."""
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
from .frontends import ColorFrontend,YoloFrontend,color_center,color_mask,load_vendor,black_apex_evidence
from .native import NativeGreen,ReferenceGreen,NativeDetector,NativeOrange,NativeMetric
from .geometry import Geometry
from .orientation import complete_orientation, orange_yaw_cdeg
from .scheduling import pose_order,group_veto
from . import uart
from .zones import estimate_zone_regions, own_subregion_targets

NAMES=('green','black','orange','blue','normal_refuge','casualty_refuge')

def top_quad(image,obj,result,hsv=None):
    """Image evidence candidate only; geometry must still validate dimensions and calibration."""
    if result.get('status')!='edge_candidate' or 'back_line' not in result:return None
    h,w=image.shape[:2];x0,y0,x1,y1=obj['bbox']
    x0,y0=max(0,int(x0)),max(0,int(y0));x1,y1=min(w,int(np.ceil(x1))),min(h,int(np.ceil(y1)))
    if x1<=x0 or y1<=y0:return None
    # Broadcasting two 1-D axes avoids allocating two full ROI coordinate grids.
    ys=np.arange(y0,y1)[:,None];xs=np.arange(x0,x1)[None,:]
    back=result['back_line'];front=result['front_line']
    band=(ys>=back[0]*xs+back[1])&(ys<=front[0]*xs+front[1])
    roi_hsv=hsv[y0:y1,x0:x1] if hsv is not None else cv2.cvtColor(image[y0:y1,x0:x1],cv2.COLOR_BGR2HSV)
    mask=color_mask(roi_hsv,obj['type'])
    mask=np.uint8(band&(mask>0))*255
    contours=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    if not contours:return None
    hull=cv2.convexHull(max(contours,key=cv2.contourArea))
    poly=cv2.approxPolyDP(hull,.025*cv2.arcLength(hull,True),True).reshape(-1,2)
    if len(poly)!=4:return None
    poly=poly+np.array([x0,y0])
    if np.any(poly[:,0]<=1) or np.any(poly[:,0]>=w-2) or np.any(poly[:,1]<=1) or np.any(poly[:,1]>=h-2):return None
    return poly.tolist()

class Pipeline:
    def __init__(self,root,config,mode):
        self.root=Path(root);self.config=config;self.mode=mode;self.detector=None
        path=lambda key:self.root/config[key]
        calibration=json.loads(path('calibration').read_text())
        self.geometry=Geometry(calibration,{int(k):v for k,v in config['dimensions_m'].items()},config['geometry_dimension_relative_tolerance'])
        self.known_size_requested=config.get('geometry_backend')=='native_known_size_rgb'
        self.metric_solver=NativeMetric(path('native_metric'),self.geometry) if self.known_size_requested and self.geometry.ready else None
        self.default_yaw={int(k):v for k,v in self.config.get('default_yaw_deg',{}).items()}
        self.green=None;self.orange=None;self.orange_native=None
        if not self.known_size_requested:
            green_cfg=json.loads((self.root/'vendor/config.json').read_text())['green_thresholds']
            if config.get('green_backend','native')=='native':self.green=NativeGreen(path('native_green'),green_cfg)
            elif config['green_backend']=='python':self.green=ReferenceGreen(root)
            else:raise ValueError('unsupported green backend')
            self.orange_native=NativeOrange(path('native_orange')) if config.get('orange_backend')=='native' else None
            if not self.orange_native:self.orange=load_vendor(root,'color_core')
        if config['preprocess'] not in ('raw','gaussian3'):raise ValueError('unsupported preprocess')
        if mode=='color':self.frontend=ColorFrontend(root,config)
        elif mode=='yolo':
            if config['engine_semantic_class_count']!=6 or config['model_channel_to_type']!=[2,3,1,0,4,5]:raise ValueError('this engine adapter requires the audited six-class detection mapping')
            model=Path(config['model'])
            if not model.is_absolute():model=self.root/model
            if hashlib.sha256(model.read_bytes()).hexdigest()!=config['model_sha256']:raise ValueError('model identity mismatch')
            self.detector=NativeDetector(path('native_detector'),model,config['semantic_thresholds'],config['nms_iou'])
            self.frontend=YoloFrontend(self.detector,config.get('blue_context_supplement'))
            self.zone_detector=self.frontend
            if config.get('green_crowd_evidence',False):
                from .green_detection import GreenCrowdFrontend
                self.frontend=GreenCrowdFrontend(self.frontend,self.root,path('native_green'))
        else:raise ValueError('mode must be color or yolo')

    def close(self):
        if self.detector:self.detector.close()

    def _orange(self,image,obj):
        h,w=image.shape[:2];x0,y0,x1,y1=obj['bbox']
        px=max(3,int(.08*(x1-x0)));py=max(3,int(.08*(y1-y0)))
        a,b,c,d=max(0,int(x0)-px),max(0,int(y0)-py),min(w,int(np.ceil(x1))+px),min(h,int(np.ceil(y1))+py)
        z=self.orange_native.estimate(image[b:d,a:c]) if self.orange_native else self.orange.estimate(image[b:d,a:c],'orange')
        r=dict(initial_center=obj['center_px'],center=obj['center_px'],status='fallback',reason=z.get('reason','no_orange_evidence'),n_profiles=z.get('n_profiles',0),evidence={k:v for k,v in z.items() if k!='mask'})
        if z.get('center_local') is not None:r['proposed_center']=[z['center_local'][0]+a,z['center_local'][1]+b]
        if z.get('status')=='two_rim_candidate':
            r.update(center=r['proposed_center'],status='edge_candidate')
            for key in ('back_line','front_line'):
                slope,intercept=z[key];r[key]=[slope,intercept+b-slope*a]
        return r

    def process(self,image):
        start=time.perf_counter()
        if image is None or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:raise ValueError('uint8 BGR frame required')
        if [image.shape[1],image.shape[0]]!=self.geometry.config['image_size']:raise ValueError('input size differs from configured calibration; no implicit resizing')
        image=np.ascontiguousarray(image)
        objects=self.frontend.detect(image);detected=time.perf_counter()
        hsv=getattr(self.frontend,'last_hsv',None)
        for i,o in enumerate(objects):
            bottom=max(o['bbox'][3],o.get('support_bbox',o['bbox'])[3])
            o.update(candidate_index=i,color=NAMES[o['type']],near=bottom>=image.shape[0]*self.config['near_line_fraction'],position_valid=False,pose_valid=False)
            o['refinement']=dict(center=o['center_px'][:],initial_center=o['center_px'][:],status='not_refined',reason='not_scheduled')
            if o['type'] in (1,3):
                o['center_px'],o['center_kind']=color_center(image,o,hsv)
                o['refinement']=dict(center=o['center_px'][:],status='center_only',reason='no_orientation_requested')
                if o['type']==1:
                    evidence=black_apex_evidence(image,o,hsv)
                    if evidence:o.update(evidence)
        centered=time.perf_counter();order=pose_order(objects);tasks=[];context=None;metric_context=None;work=image
        if self.green and self.config['preprocess']=='gaussian3':work=cv2.GaussianBlur(image,(3,3),.8)
        try:
            for index in order:
                obj=objects[index];begin=time.perf_counter();budget=self.config.get('pose_budget_ms')
                if obj.get('geometry_body_supported') is False:
                    obj['metric']=self.geometry.estimate(obj,image.shape[:2]);obj['metric']['reason']='individual_topface_only_body_partition_unresolved'
                    obj['refinement']['reason']=obj['metric']['reason']
                    tasks.append(dict(index=index,color=obj['color'],near=obj['near'],skipped=True,ms=0));continue
                if self.known_size_requested and not self.metric_solver:
                    obj['refinement']['reason']='metric_disabled:'+self.geometry.reason
                    tasks.append(dict(index=index,color=obj['color'],near=obj['near'],skipped=True,ms=0));continue
                if not obj['near'] and (not self.config['refine_far'] or (budget is not None and (begin-start)*1000>=budget)):
                    obj['refinement']['reason']='far_coarse_budget_policy';tasks.append(dict(index=index,color=obj['color'],near=False,skipped=True,ms=0));continue
                if self.metric_solver:
                    if metric_context is None:metric_context=self.metric_solver.prepare(image,objects,self.config.get('metric_white_reference',True))
                    task_budget=self.config.get('metric_per_object_budget_ms',0)
                    if not obj['near'] and budget is not None:
                        remaining=max(.1,budget-(time.perf_counter()-start)*1000)
                        task_budget=min(task_budget,remaining) if task_budget>0 else remaining
                    fit=self.metric_solver.fit(metric_context,index,task_budget)
                    obj['metric']=self.geometry.from_model(obj,fit)
                    obj['refinement'].update(status='model_estimate' if fit['status'] in (1,2) else 'model_rejected',reason=obj['metric']['reason'])
                elif obj['type']==2:obj['refinement']=self._orange(work,obj)
                else:
                    seed=obj['seed']
                    if seed['x_span'][1]-seed['x_span'][0]<2 or seed['y_span'][1]-seed['y_span'][0]<2:
                        obj['refinement']['reason']='degenerate_seed_span'
                    else:
                        if context is None:context=self.green.prepare(image,self.config['preprocess'])
                        obj['refinement']=self.green.local(context,seed)
                tasks.append(dict(index=index,color=obj['color'],near=obj['near'],skipped=False,ms=(time.perf_counter()-begin)*1000))
            # A frame without a cube never needs an additional RGB/white-balance
            # context. In particular, an empty detection frame must not process
            # another full 640x480 image after YOLO has already finished.
            cube_indices=[index for index,obj in enumerate(objects)
                          if obj['type'] in (0,3) and not obj.get('visible_only')
                          and obj.get('geometry_body_supported') is not False]
            if self.metric_solver and cube_indices:
                if metric_context is None:
                    metric_context=self.metric_solver.prepare(image,objects,self.config.get('metric_white_reference',True))
                for index in cube_indices:
                    obj=objects[index]
                    obj['native_cube_fit']=self.metric_solver.cube(metric_context,index)
                    obj['refinement'].update(status='center_only',reason='known_cube_center_without_orientation')
        finally:
            if self.green and context is not None:self.green.release(context)
            if self.metric_solver and metric_context is not None:self.metric_solver.release(metric_context)
        if self.green:
            green_objects=[o for o in objects if o['type']==0]
            group_veto([o['refinement'] for o in green_objects],[o['seed'] for o in green_objects])
        refined=time.perf_counter()
        for obj in objects:
            obj['center_px']=obj['refinement']['center']
            if obj['refinement']['status']=='edge_candidate':
                obj['center_kind']='two_rim_image_anchor_not_verified_metric_center'
                quad=top_quad(work,obj,obj['refinement'],hsv if work is image else None)
                if quad is not None:obj['top_corners_px']=quad
            metric=obj.get('metric') or self.geometry.estimate(obj,image.shape[:2])
            metric=self.geometry.best_effort_position(obj,image.shape[:2],metric)
            metric=complete_orientation(obj['type'],metric,self.default_yaw)
            obj['metric']=metric
            obj['position_valid']=metric['position_valid'];obj['pose_valid']=metric['pose_valid']
        zones=estimate_zone_regions(image,getattr(getattr(self,'zone_detector',None),'zones',[]),self.geometry,self.config['placement_zone_color'],hsv)
        projected=time.perf_counter()
        reasons=[]
        missing=sorted(set(self.config['uart_required_types'])-set(self.frontend.coverage))
        if missing:reasons.append('required_model_classes_unavailable:'+','.join(map(str,missing)))
        if self.geometry.ready and self.geometry.pinhole_estimate:reasons.append('distortion_unverified')
        if self.geometry.config.get('simulation_only'):reasons.append('assumed_stress_calibration_not_measured')
        targets=[dict(type=o['type'],x_mm=o['metric']['x_mm'],y_mm=o['metric']['y_mm']) for o in objects if o['position_valid']]
        # Keep the two positioning landmarks together even when blocks fill
        # the 32-record packet. Remaining capacity retains block priority.
        # Orange attitude has the same axle-frame angle definition as its model.
        located=[o for o in objects if o['position_valid']]
        for target,obj in zip(targets,located):
            if target['type']==2:target['yaw_cdeg']=orange_yaw_cdeg(obj['metric'])
        targets=own_subregion_targets(zones)+targets
        wire_targets=[]
        if not missing and not self.geometry.config.get('simulation_only'):
            for target in targets:
                # Validate a record without repeatedly building a full frame and CRC.
                try:uart._pack_target(target,0)
                except ValueError as error:
                    reasons.append('wire_target_omitted:'+str(error));continue
                wire_targets.append(target)
        if len(wire_targets)>uart.MAX_TARGETS:
            reasons.append('target_capacity_exceeded')
            wire_targets=wire_targets[:uart.MAX_TARGETS]
        packet=uart.encode(wire_targets)
        encoded=time.perf_counter()
        if len(targets)==len(wire_targets):
            # All targets passed validation: packet already contains exactly these bytes.
            preview=packet.hex()
        else:
            try:preview=uart.encode(targets).hex() if len(targets)<=uart.MAX_TARGETS else None
            except ValueError:preview=None
        return dict(mode=self.mode,image_size=[image.shape[1],image.shape[0]],objects=objects,placement_zones=zones,
                    frontend_diagnostics=getattr(self.frontend,'last_diagnostics',{}),
                    counts={name:sum(o['color']==name for o in objects) for name in NAMES},pose_task_order=tasks,
                    class_coverage=self.frontend.coverage,calibration_status=self.geometry.reason,metric_calculation_enabled=bool(self.metric_solver or (self.geometry.ready and self.green)),measurement_context='assumed_camera_stress_only' if self.geometry.config.get('simulation_only') else 'input_bound_calibration',uart_hex=packet.hex(),diagnostics=reasons,uart_target_count=len(wire_targets),unlocated_object_count=sum(not o['position_valid'] for o in objects),
                    theoretical_wire_ms=len(packet)*10*1000./self.config['uart_baud'],
                    estimate_packet_preview=dict(hex=preview,count=len(targets),partial=len(targets)!=len(objects),transmitted=False,wire_yaw_zero=all(t.get('yaw_cdeg',0)==0 for t in targets)),
                    timings_ms=dict(detection=(detected-start)*1000,center_only=(centered-detected)*1000,scheduled_refinement_and_veto=(refined-centered)*1000,
                                    metric_validation=(projected-refined)*1000,uart_encode=(encoded-projected)*1000,compute=(encoded-start)*1000))
