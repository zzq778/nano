#!/usr/bin/env python3
"""Measure RAW BGR frame channel balance; no correction and no camera access.

Compare a few normal/bad exact PNGs from the same lighting and ROI. Color
imbalance alone cannot prove camera white-balance malfunction.
"""
from __future__ import print_function
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np


def image_profile(image, roi=None):
    if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('Expected 8-bit 3-channel BGR source image')
    height,width=image.shape[:2]
    if roi:
        x,y,w,h = roi
        if min(x,y)<0 or min(w,h)<1 or x+w>width or y+h>height:
            raise ValueError('ROI outside captured image')
        image=image[y:y+h,x:x+w]
    pixels=image.reshape(-1,3).astype(np.float32)
    b,g,r=np.median(pixels,axis=0)
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    return {'pixels':int(pixels.shape[0]),'bgr_median':[float(b),float(g),float(r)],
            'red_over_green':round(float(r/max(g,1)),4),
            'blue_over_green':round(float(b/max(g,1)),4),
            'hsv_saturation_p50':float(np.median(hsv[:,:,1])),
            'hsv_value_p50':float(np.median(hsv[:,:,2])),
            'overexposed_fraction':round(float(np.mean(np.max(image,axis=2)>=250)),4),
            'underexposed_fraction':round(float(np.mean(np.max(image,axis=2)<=15)),4)}


def profile_paths(paths,roi=None):
    result=[]
    for path in paths:
        path=Path(path)
        img=cv2.imread(str(path),cv2.IMREAD_COLOR)
        if img is None:raise ValueError('Cannot decode frame: '+str(path))
        result.append({'source':path.name,'measurements':image_profile(img,roi)})
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--normal',type=Path,nargs='+',required=True)
    p.add_argument('--suspect',type=Path,nargs='+',required=True)
    p.add_argument('--roi',type=int,nargs=4,metavar=('X','Y','W','H'))
    p.add_argument('--output',type=Path)
    a=p.parse_args(argv)
    data={'schema_version':1,'normal':profile_paths(a.normal,a.roi),
          'suspect':profile_paths(a.suspect,a.roi),
          'caution':('Equal illumination and ROI required. This compares source PNGs, '
                     'not a proof of AWB/exposure cause; no pixel correction is applied.')}
    text=json.dumps(data,indent=2,sort_keys=True)+'\n'
    if a.output:a.output.write_text(text)
    print(text)
    return 0


if __name__=='__main__':sys.exit(main())
