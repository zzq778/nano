"""Visible-floor ground rectangle estimates, separate from block and wire types."""
import math
import cv2
import numpy as np

# 宽松几何门槛：允许RGB投影尺寸偏差，排除退化和明显不成比例轮廓。
ZONE_MIN_SIZE_SCALE = .45
ZONE_MAX_SIZE_SCALE = 1.80

def plausible_platform_sides(sides):
    short,long=sorted(sides)
    return (all(math.isfinite(v) for v in (short,long)) and
            .36*ZONE_MIN_SIZE_SCALE <= short <= .36*ZONE_MAX_SIZE_SCALE and
            .66*ZONE_MIN_SIZE_SCALE <= long <= .66*ZONE_MAX_SIZE_SCALE and
            1.2 <= long/short <= 3.5)

def estimate_zone_regions(image,detections,geometry,selected_color,hsv=None):
    if selected_color not in ('purple','blue'):raise ValueError('placement_zone_color must be purple or blue')
    height,width=image.shape[:2];results=[]
    for detection in detections:
        result=dict(detection,selected_for_uart=detection['color']==selected_color,
                    position_valid=False,pose_valid=False,transmitted=False)
        results.append(result)
        a,b,c,d=detection['bbox'];x0,y0=max(0,int(a)),max(0,int(b));x1,y1=min(width,int(math.ceil(c))),min(height,int(math.ceil(d)))
        if x1-x0<12 or y1-y0<12:result['reason']='region_too_small';continue
        if detection['color']=='purple':
            y0=max(0,y0-6);y1=min(height,y1+6)
            if a > width-c:
                x0=max(0,int(a)-15);x1=width
            else:
                x0=0;x1=min(width,int(math.ceil(c))+15)
        region_hsv=hsv[y0:y1,x0:x1] if hsv is not None else cv2.cvtColor(image[y0:y1,x0:x1],cv2.COLOR_BGR2HSV)
        inner_floor_mask=None
        if detection['color']=='blue':mask=cv2.inRange(region_hsv,(90,85,30),(145,255,255))
        else:
            bgr_b,bgr_g,bgr_r=cv2.split(image[y0:y1,x0:x1])
            # Saturating subtraction preserves these positive-difference tests for
            # uint8 pixels while avoiding a full int16 copy of the region.
            pink=cv2.inRange(bgr_r,65,255)
            cv2.bitwise_and(pink,cv2.inRange(cv2.subtract(bgr_r,bgr_g),7,255),dst=pink)
            cv2.bitwise_and(pink,cv2.inRange(cv2.subtract(bgr_r,bgr_b),5,255),dst=pink)
            inner_floor_mask=pink
            rail=cv2.inRange(region_hsv,(100,42,35),(179,255,205))
            mask=cv2.bitwise_or(pink,rail)
        # Use one coherent saturated platform component; wall/floor noise must
        # not expand a global hull to the YOLO rectangle corners.
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        connected=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
        contours,_=cv2.findContours(connected,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        component=np.zeros_like(mask)
        if contours:
            if detection['color']=='purple':
                accepted=[contour for contour in contours if cv2.contourArea(contour)>=max(40,.002*(c-a)*(d-b))]
                if accepted:cv2.drawContours(component,accepted,-1,255,-1)
            else:cv2.drawContours(component,[max(contours,key=cv2.contourArea)],-1,255,-1)
        mask=cv2.bitwise_and(mask,component)
        pixels=cv2.findNonZero(mask)
        if pixels is None or len(pixels)<100:
            if detection['color']!='purple':result['reason']='floor_color_evidence_missing';continue
            polygon=np.array([[a,b],[c,b],[c,d],[a,d]],dtype=float)
            result['outline_evidence']='purple_yolo_outer_bbox_fallback'
        else:
            hull=cv2.convexHull(pixels)
            polygon=hull.reshape(-1,2)+np.array([x0,y0])
            result['outline_evidence']='visible_platform_color_hull' if detection['color']=='purple' else 'visible_floor_color_hull'
        # A complete red inner floor is a more stable center cue than the raised
        # 20 mm perimeter.  Keep the perimeter for yaw: the floor may be
        # partially occluded by its rails and tilt the fitted long axis.
        floor_center_override=None
        if inner_floor_mask is not None and geometry.ready:
            floor=cv2.morphologyEx(inner_floor_mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
            floor=cv2.morphologyEx(floor,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
            floor_contours,_=cv2.findContours(floor,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            floor_components=[contour for contour in floor_contours if cv2.contourArea(contour)>=max(40,.002*(c-a)*(d-b))]
            if floor_components:
                floor[:]=0
                cv2.drawContours(floor,floor_components,-1,255,-1)
                floor_pixels=cv2.findNonZero(floor)
                if floor_pixels is not None and len(floor_pixels)>=100:
                    floor_hull=cv2.convexHull(floor_pixels).reshape(-1,2)+np.array([x0,y0])
                    floor_points=geometry.project_many(floor_hull,image.shape[:2],geometry.ground)
                    if floor_points is not None:
                        floor_rect=cv2.minAreaRect(np.ascontiguousarray(np.asarray(floor_points,np.float32)[:,:2]))
                        floor_box=cv2.boxPoints(floor_rect)
                        floor_edges=np.roll(floor_box,-1,axis=0)-floor_box
                        floor_lengths=np.linalg.norm(floor_edges,axis=1)
                        floor_sides=sorted([(floor_lengths[0]+floor_lengths[2])*.5,(floor_lengths[1]+floor_lengths[3])*.5])
                        if all(.75*expected<=actual<=1.25*expected for actual,expected in zip(floor_sides,(.30,.60))):
                            floor_center_override=floor_box.mean(axis=0).astype(float)
                            result['visible_inner_floor_hull_px']=floor_hull.tolist()
                            result['measured_inner_floor_sides_m']=[float(v) for v in floor_sides]
        result['visible_floor_hull_px']=polygon.tolist()
        if not geometry.ready:result['reason']='calibration_not_ready';continue
        projected=geometry.project_many(polygon,image.shape[:2],geometry.ground)
        if projected is None:result['reason']='ground_intersection_failed';continue
        visible=np.ascontiguousarray(np.array(projected,dtype=np.float32)[:,:2])
        rectangle=cv2.minAreaRect(visible)
        points=cv2.boxPoints(rectangle).astype(float)
        edges=np.roll(points,-1,axis=0)-points;lengths=np.linalg.norm(edges,axis=1)
        result['center_source']='visible_floor_ground_rectangle_estimate'
        result['outline_semantics']='outer_platform_660x360mm'
        result['expected_outer_sides_m']=[.36,.66]
        sides=sorted([(lengths[0]+lengths[2])*.5,(lengths[1]+lengths[3])*.5])
        result['measured_floor_sides_m']=sides
        result['size_quality']='nominal' if all(abs(a-b)<=b*.25 for a,b in zip(sides,(.36,.66))) else 'relaxed_size_estimate'
        if (min(lengths)<1e-6 or not plausible_platform_sides(sides)) and detection['color']=='purple' and result['outline_evidence']!='purple_yolo_outer_bbox_fallback':
            polygon=np.array([[a,b],[c,b],[c,d],[a,d]],dtype=float)
            projected=geometry.project_many(polygon,image.shape[:2],geometry.ground)
            if projected is not None:
                visible=np.ascontiguousarray(np.array(projected,dtype=np.float32)[:,:2])
                rectangle=cv2.minAreaRect(visible)
                points=cv2.boxPoints(rectangle).astype(float)
                edges=np.roll(points,-1,axis=0)-points;lengths=np.linalg.norm(edges,axis=1)
                sides=sorted([(lengths[0]+lengths[2])*.5,(lengths[1]+lengths[3])*.5])
                result['visible_floor_hull_px']=polygon.tolist()
                result['outline_evidence']='purple_yolo_outer_bbox_fallback'
                result['measured_floor_sides_m']=sides
                result['size_quality']='nominal' if all(abs(a-b)<=b*.25 for a,b in zip(sides,(.36,.66))) else 'relaxed_size_estimate'
        if min(lengths)<1e-6 or not plausible_platform_sides(sides):
            result['reason']='known_floor_dimensions_disagree';continue
        center=points.mean(axis=0);axis=edges[int(np.argmax(lengths))]
        result.update(position_valid=True,pose_valid=False,center_xy_m=center.tolist(),
                      long_axis_deg=math.degrees(math.atan2(axis[1],axis[0]))%180,orientation_period_deg=180,
                      reason='floor_pose_estimate_entrance_unconfirmed',metric_accuracy_verified=False,
                      entrance_valid=False)
        long_index=int(np.argmax(lengths));opposite=(long_index+2)%4
        midpoints=[(points[i]+points[(i+1)%4])*.5 for i in (long_index,opposite)]
        distances=[np.linalg.norm(point-geometry.t[:2]) for point in midpoints]
        if abs(distances[0]-distances[1])<.05:result['reason']='field_side_direction_ambiguous';continue
        entrance=midpoints[int(np.argmin(distances))]
        inward=center-entrance;inward/=np.linalg.norm(inward)
        result.update(entrance_valid=True,pose_valid=True,
                      inward_yaw_deg=math.degrees(math.atan2(inward[1],inward[0])),
                      entrance_method='nearest_long_edge_assuming_camera_inside_competition_field',
                      reason='whole_refuge_center_with_field_side_normal')
        if floor_center_override is not None:
            result['center_xy_m']=floor_center_override.tolist()
            result['center_source']='visible_inner_floor_ground_rectangle_estimate'
    return results


def own_subregion_targets(zones):
    """One own 660x360mm outline -> two 300x300mm inner subzone centers.

    Facing inward: material on the left, casualty on the right. Both yaw
    values are the inward normal, in axle-frame CCW centidegrees.
    """
    eligible=[z for z in zones if z.get('selected_for_uart') and
              z.get('position_valid') and z.get('pose_valid') and z.get('entrance_valid')]
    for zone in zones: zone['included_in_target_list']=False
    if not eligible:return []
    # Multiple overlapping detections must not publish duplicate landmarks.
    zone=max(eligible,key=lambda z: float(z.get('confidence') or 0.))
    x,y=zone['center_xy_m']
    if not all(math.isfinite(v) for v in (x,y,zone['inward_yaw_deg'])):
        zone['uart_omission_reason']='nonfinite_subregion_pose'
        return []
    angle=math.radians(zone['inward_yaw_deg'])
    left=(-math.sin(angle),math.cos(angle))
    yaw=int(round(((zone['inward_yaw_deg']+180.)%360.-180.)*100))
    if yaw==18000:yaw=-18000
    targets=[]
    for cls,sign in ((4,1),(5,-1)):
        targets.append(dict(type=cls,x_mm=int(round((x+sign*.15*left[0])*1000)),
                            y_mm=int(round((y+sign*.15*left[1])*1000)),yaw_cdeg=yaw))
    # Do not let the codec omit only one member of an out-of-range pair.
    if any(not -32768 <= t[key] <= 32767 for t in targets for key in ('x_mm','y_mm')):
        zone['uart_omission_reason']='subregion_pair_out_of_int16_range'
        return []
    zone['subregion_targets']=targets
    zone['included_in_target_list']=True
    return targets
