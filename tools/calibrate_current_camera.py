#!/usr/bin/env python3
"""Explicit current-camera calibration and independent ground distance/angle validation."""
import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
import cv2
import numpy as np
from rescuevision.geometry import Geometry
from rescuevision.runtime import LatestCamera

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def corners(image,cols,rows):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    found,points=cv2.findChessboardCorners(gray,(cols,rows),cv2.CALIB_CB_ADAPTIVE_THRESH|cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:return None
    return cv2.cornerSubPix(gray,points,(5,5),(-1,-1),(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,40,.001))
def board_points(cols,rows,square_mm):
    points=np.zeros((cols*rows,3),np.float32);points[:,:2]=np.mgrid[0:cols,0:rows].T.reshape(-1,2)*float(square_mm)/1000.
    return points
def usb_identity(device):
    if str(device)=='astra' or str(device).startswith('2bc5/0402@'):return '2bc5:0402'
    raise ValueError('capture source must be astra or an explicit Astra URI')

def capture(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    identity=usb_identity(args.device)
    config=json.loads((ROOT/'configs/runtime.json').read_text())
    camera=LatestCamera(args.device,args.width,args.height,ROOT/config['native_camera']);sequence=0;count=0;start=time.perf_counter();last=-1e9
    try:
        while count<args.count and time.perf_counter()-start<args.timeout:
            sequence,image,stamp=camera.get(sequence)
            if time.perf_counter()-last<args.interval:continue
            points=corners(image,args.cols,args.rows)
            if points is None:continue
            path=out/('board_%03d.png'%count)
            if not cv2.imwrite(str(path),image):raise OSError('calibration frame write failed')
            print('saved',path,flush=True);count+=1;last=time.perf_counter()
    finally:camera.close()
    (out/'capture.json').write_text(json.dumps(dict(camera_usb_id=identity,device=args.device,image_size=[args.width,args.height],
              recognized_views=count,cols=args.cols,rows=args.rows,source_kind='explicit_checkerboard_calibration_capture_not_block_test'),indent=2))
    if count<args.count:raise RuntimeError('not enough recognizable board views before deadline')

def intrinsics(args):
    directory=Path(args.images);files=sorted(p for p in directory.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg'))
    object_points=[];image_points=[];sources=[];size=None
    model=board_points(args.cols,args.rows,args.square_mm)
    for path in files:
        image=cv2.imread(str(path))
        if image is None:continue
        current=(image.shape[1],image.shape[0])
        if size is None:size=current
        if current!=size:raise ValueError('mixed calibration resolutions')
        points=corners(image,args.cols,args.rows)
        if points is None:continue
        object_points.append(model.copy());image_points.append(points)
        sources.append(dict(path=str(path.resolve()),sha256=sha(path)))
    if len(sources)<args.min_views:raise ValueError('only %d recognizable views; need %d, no guessed calibration written'%(len(sources),args.min_views))
    rms,K,D,rvecs,tvecs=cv2.calibrateCamera(object_points,image_points,size,None,None)
    if not np.isfinite(K).all() or not np.isfinite(D).all() or rms>args.max_rms:raise ValueError('calibration numeric/RMS gate failed')
    tilts=[];per_view=[]
    for model,observed,rvec,tvec in zip(object_points,image_points,rvecs,tvecs):
        projected=cv2.projectPoints(model,rvec,tvec,K,D)[0]
        per_view.append(float(np.sqrt(np.mean((observed-projected)**2))))
        rotation=cv2.Rodrigues(rvec)[0];tilts.append(math.degrees(math.acos(float(np.clip(abs(rotation[2,2]),0,1)))))
    if max(tilts)-min(tilts)<10 or np.median(tilts)<8:raise ValueError('insufficient board tilt diversity; do not accept a weakly constrained focal length')
    capture_path=directory/'capture.json';capture_info=json.loads(capture_path.read_text()) if capture_path.is_file() else {}
    base=json.loads(Path(args.base_config).read_text());captured_id=capture_info.get('camera_usb_id') or args.camera_id
    camera_matches=bool(captured_id and captured_id==base.get('camera_usb_id'))
    output=copy.deepcopy(base)
    output.pop('projection_policy',None)  # Measured unrectified D must reach Geometry.project.
    output.update(image_size=list(size),camera_usb_id=captured_id,intrinsics=dict(fx=float(K[0,0]),fy=float(K[1,1]),cx=float(K[0,2]),cy=float(K[1,2])),
                  distortion_coefficients=D.ravel().tolist(),rectification_status='unrectified',matches_input_camera=camera_matches,
                  status='measured_intrinsics_pending_independent_metric_validation',metric_accuracy_verified=False,
                  calibration_measurement=dict(rms_px=float(rms),per_view_rms_px=per_view,tilt_deg=tilts,cols=args.cols,rows=args.rows,
                       square_mm=args.square_mm,source_images=sources,source='OpenCV calibrateCamera, no imported Astra K/D'))
    Path(args.output).write_text(json.dumps(output,indent=2))
    print(json.dumps(dict(views=len(sources),rms_px=rms,camera_identity_matches=camera_matches,output=args.output),indent=2))

def validate_ground(args):
    if not args.corner_order_confirmed:raise ValueError('confirm A0 corner order against the marked board; use --reverse if needed')
    path=Path(args.image);config=json.loads(Path(args.calibration).read_text())
    if sha(path) in {s['sha256'] for s in config.get('calibration_measurement',{}).get('source_images',[])}:
        raise ValueError('validation image was used for calibration; use an independent measured placement')
    geometry=Geometry(config)
    if not geometry.ready:raise ValueError('calibration is not ready: '+geometry.reason)
    image=cv2.imread(str(path));points=corners(image,args.cols,args.rows)
    if points is None:raise ValueError('board not found')
    points=points.reshape(-1,2)
    if args.reverse:points=points[::-1]
    # Printed board axes: +X increases columns; +Y decreases rows, +Z is upward.
    local=board_points(args.cols,args.rows,args.square_mm)[:,:2];local[:,1]*=-1
    angle=math.radians(args.board_yaw_deg);co,si=math.cos(angle),math.sin(angle)
    expected=local.dot(np.array([[co,si],[-si,co]]))+np.array([args.origin_x_mm,args.origin_y_mm])/1000.
    projected=[geometry.project(p,image.shape[:2],geometry.ground) for p in points]
    if any(p is None for p in projected):raise ValueError('ground projection failed')
    measured=np.array(projected)[:,:2];errors=np.linalg.norm(measured-expected,axis=1)*1000
    vectors=[measured[r*args.cols+args.cols-1]-measured[r*args.cols] for r in range(args.rows)]
    axis=np.mean(vectors,axis=0);yaw=math.degrees(math.atan2(axis[1],axis[0]));yaw_error=(yaw-args.board_yaw_deg+180)%360-180
    measured_center=np.mean(measured,axis=0)*1000;expected_center=np.mean(expected,axis=0)*1000
    report=dict(test='independent_camera_ground_projection; does not certify block-detection corner accuracy',image=str(path.resolve()),image_sha256=sha(path),
        rmse_xy_mm=float(np.sqrt(np.mean(errors**2))),max_xy_error_mm=float(np.max(errors)),center_error_mm=float(np.linalg.norm(measured_center-expected_center)),
        estimated_center_xy_mm=measured_center.tolist(),measured_reference_center_xy_mm=expected_center.tolist(),
        horizontal_range_mm=float(np.linalg.norm(measured_center)),reference_horizontal_range_mm=float(np.linalg.norm(expected_center)),
        ground_point_to_axle_distance_mm=float(np.sqrt(np.dot(measured_center,measured_center)+(geometry.ground*1000)**2)),
        yaw_deg=yaw,reference_yaw_deg=args.board_yaw_deg,yaw_error_deg=yaw_error,
        reference_placement=dict(origin_x_mm=args.origin_x_mm,origin_y_mm=args.origin_y_mm,board_yaw_deg=args.board_yaw_deg,square_mm=args.square_mm),
        camera_calibration_sha256=sha(args.calibration))
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode')
    c=sub.add_parser('capture');c.add_argument('--device',required=True);c.add_argument('--output',required=True)
    c.add_argument('--width',type=int,default=640);c.add_argument('--height',type=int,default=480)
    c.add_argument('--count',type=int,default=20);c.add_argument('--interval',type=float,default=2.);c.add_argument('--timeout',type=float,default=180.)
    i=sub.add_parser('intrinsics');i.add_argument('--images',required=True);i.add_argument('--output',required=True)
    i.add_argument('--base-config',default=str(ROOT/'configs/astra_current.json'));i.add_argument('--camera-id')
    i.add_argument('--min-views',type=int,default=12);i.add_argument('--max-rms',type=float,default=1.)
    v=sub.add_parser('validate-ground');v.add_argument('--image',required=True);v.add_argument('--calibration',required=True);v.add_argument('--output',required=True)
    for flag in ('origin-x-mm','origin-y-mm','board-yaw-deg'):v.add_argument('--'+flag,type=float,required=True)
    v.add_argument('--corner-order-confirmed',action='store_true');v.add_argument('--reverse',action='store_true')
    for command in (c,i,v):command.add_argument('--cols',type=int,default=9);command.add_argument('--rows',type=int,default=6)
    for command in (i,v):command.add_argument('--square-mm',type=float,required=True)
    args=parser.parse_args()
    if args.mode is None:parser.error('select capture, intrinsics or validate-ground')
    if args.cols<3 or args.rows<3:parser.error('invalid checkerboard geometry')
    if hasattr(args,'square_mm') and args.square_mm<=0:parser.error('square size must be measured and positive')
    try:{'capture':capture,'intrinsics':intrinsics,'validate-ground':validate_ground}[args.mode](args)
    except Exception as error:print(json.dumps(dict(valid=False,error=str(error)),ensure_ascii=False),file=sys.stderr);return 2
    return 0
if __name__=='__main__':sys.exit(main())
