import time
"""Narrow ctypes boundary; owned native contexts are used on one processing thread."""
import ctypes as C
import json
from pathlib import Path
import numpy as np

U8P=C.POINTER(C.c_ubyte);DP=C.POINTER(C.c_double)
GREEN_KEYS=['back_expand_width','back_upper_width','min_depth_width','max_depth_width','minimum_contrast','fit_residual_fraction','minimum_profiles','minimum_survival']

class NativeGreen:
    def __init__(self,library,config):
        self.lib=C.CDLL(str(library));self.config=np.array([config[k] for k in GREEN_KEYS],np.float64)
        self.buffer=C.create_string_buffer(1024*1024);self.err=C.create_string_buffer(4096)
        self.lib.green_prepare.argtypes=[U8P,C.c_int,C.c_int,C.c_int,DP,C.c_int,C.c_char_p,C.c_int];self.lib.green_prepare.restype=C.c_void_p
        self.lib.green_local_json.argtypes=[C.c_void_p,DP,C.c_char_p,C.c_int]
        self.lib.green_release.argtypes=[C.c_void_p]
        self.lib.v12_refine_json.argtypes=[U8P,C.c_int,C.c_int,C.c_int,DP,C.c_int,DP,C.c_int,C.c_char_p,C.c_int]
    def prepare(self,image,preprocess):
        context=self.lib.green_prepare(image.ctypes.data_as(U8P),image.shape[1],image.shape[0],int(image.strides[0]),self.config.ctypes.data_as(DP),int(preprocess=='gaussian3'),self.err,len(self.err))
        if not context:raise RuntimeError(self.err.value.decode())
        return context
    def local(self,context,seed):
        values=np.array([seed['x'],seed['y']]+list(seed['x_span'])+list(seed['y_span']),np.float64)
        if self.lib.green_local_json(context,values.ctypes.data_as(DP),self.buffer,len(self.buffer))<0:raise RuntimeError(self.buffer.value.decode())
        return json.loads(self.buffer.value.decode())
    def release(self,context):
        if context:self.lib.green_release(context)
    def batch(self,image,seeds,preprocess='raw'):
        values=np.array([[s['x'],s['y']]+list(s['x_span'])+list(s['y_span']) for s in seeds],np.float64)
        code=self.lib.v12_refine_json(image.ctypes.data_as(U8P),image.shape[1],image.shape[0],int(image.strides[0]),values.ctypes.data_as(DP),len(seeds),self.config.ctypes.data_as(DP),int(preprocess=='gaussian3'),self.buffer,len(self.buffer))
        if code<0:raise RuntimeError(self.buffer.value.decode())
        return json.loads(self.buffer.value.decode())['results']

class ReferenceGreen:
    """Explicit desktop/debug backend; never silently replaces a failed native library."""
    def __init__(self,root):
        from .frontends import load_vendor
        self.reference=load_vendor(root,'refine_reference')
    def prepare(self,image,preprocess):
        if preprocess=='gaussian3':
            import cv2
            return cv2.GaussianBlur(image,(3,3),.8)
        return image
    def local(self,context,seed):return self.reference.refine_green(context,seed)
    def release(self,context):pass

class Box(C.Structure):
    _fields_=[('cls',C.c_int),('confidence',C.c_float),('x0',C.c_float),('y0',C.c_float),('x1',C.c_float),('y1',C.c_float)]

class NativeOrange:
    def __init__(self,library):
        self.lib=C.CDLL(str(library));self.buffer=C.create_string_buffer(256*1024)
        self.lib.orange_estimate_json.argtypes=[U8P,C.c_int,C.c_int,C.c_int,C.c_char_p,C.c_int]
    def estimate(self,roi):
        if roi.strides[1]!=3:roi=np.ascontiguousarray(roi)
        code=self.lib.orange_estimate_json(roi.ctypes.data_as(U8P),roi.shape[1],roi.shape[0],int(roi.strides[0]),self.buffer,len(self.buffer))
        if code<0:raise RuntimeError(self.buffer.value.decode())
        return json.loads(self.buffer.value.decode())

class NativeDetector:
    def __init__(self,library,model,thresholds,nms):
        start=time.perf_counter()
        self.lib=C.CDLL(str(library));self.err=C.create_string_buffer(4096);self.boxes=(Box*512)();self.context=None
        self.startup_timings_ms=dict(library_and_buffers_ms=(time.perf_counter()-start)*1000)
        self.lib.detector_create.argtypes=[C.c_char_p,C.c_char_p,C.c_int];self.lib.detector_create.restype=C.c_void_p
        self.lib.detector_destroy.argtypes=[C.c_void_p]
        self.lib.detector_run.argtypes=[C.c_void_p,U8P,C.c_int,C.c_int,C.c_int,C.POINTER(Box),C.c_int,C.c_char_p,C.c_int]
        self.lib.detector_configure.argtypes=[C.c_void_p,C.POINTER(C.c_float),C.c_float]
        # Optional on freshly rebuilt binaries; shipped older Nano binaries keep working.
        self._native_timings=getattr(self.lib,'detector_last_timings',None)
        if self._native_timings is not None:
            self._native_timings.argtypes=[C.c_void_p,C.POINTER(C.c_double)]
            self._native_timings.restype=C.c_int
        self.last_timings_ms=None
        engine_start=time.perf_counter()
        self.context=self.lib.detector_create(str(model).encode(),self.err,len(self.err))
        self.startup_timings_ms['engine_cuda_context_and_two_warmups_ms']=(time.perf_counter()-engine_start)*1000
        if not self.context:raise RuntimeError(self.err.value.decode())
        values=(C.c_float*6)(*thresholds)
        if self.lib.detector_configure(self.context,values,nms)!=0:self.close();raise ValueError('invalid detector thresholds')
    def run(self,image):
        n=self.lib.detector_run(self.context,image.ctypes.data_as(U8P),image.shape[1],image.shape[0],int(image.strides[0]),self.boxes,len(self.boxes),self.err,len(self.err))
        if n<0:raise RuntimeError('detector error %d: %s'%(n,self.err.value.decode()))
        if self._native_timings is not None:
            values=(C.c_double*3)()
            if self._native_timings(self.context,values)==0:
                self.last_timings_ms=dict(preprocess=values[0],tensorrt_and_transfers=values[1],postprocess=values[2])
        return [dict(type=b.cls,confidence=float(b.confidence),bbox=[float(b.x0),float(b.y0),float(b.x1),float(b.y1)]) for b in self.boxes[:n]]
    def close(self):
        if self.context:self.lib.detector_destroy(self.context);self.context=None

class NativeMetric:
    def __init__(self,library,geometry):
        if not geometry.ready or not geometry.use_pinhole:raise ValueError('native pinhole solver requires explicit pinhole or rectified calibration')
        self.lib=C.CDLL(str(library));self.buffer=C.create_string_buffer(1024*1024)
        self.k=np.array([geometry.K[0,0],geometry.K[1,1],geometry.K[0,2],geometry.K[1,2]],np.float64)
        t=geometry.t.copy();t[2]-=geometry.ground
        self.rt=np.concatenate([geometry.R.reshape(9),t]).astype(np.float64)
        self.lib.metric_prepare.argtypes=[U8P,C.c_int,C.c_int,C.c_int,DP,DP,C.POINTER(Box),C.c_int,C.c_int,C.c_char_p,C.c_int]
        self.lib.metric_prepare.restype=C.c_void_p
        self.lib.metric_fit_json.argtypes=[C.c_void_p,C.c_int,C.c_double,C.c_char_p,C.c_int]
        self.lib.metric_release.argtypes=[C.c_void_p]
        self.lib.metric_cube_json.argtypes=[C.c_void_p,C.c_int,C.c_char_p,C.c_int]
    def prepare(self,image,objects,balance=True):
        boxes=(Box*len(objects))(*[Box(o['type'],float(o.get('confidence') or 0.),*o['bbox']) for o in objects])
        context=self.lib.metric_prepare(image.ctypes.data_as(U8P),image.shape[1],image.shape[0],int(image.strides[0]),self.k.ctypes.data_as(DP),self.rt.ctypes.data_as(DP),boxes,len(objects),int(balance),self.buffer,len(self.buffer))
        if not context:raise RuntimeError(self.buffer.value.decode())
        return context
    def fit(self,context,index,budget_ms=0):
        if self.lib.metric_fit_json(context,index,budget_ms,self.buffer,len(self.buffer))<0:raise RuntimeError(self.buffer.value.decode())
        return json.loads(self.buffer.value.decode())
    def release(self,context):
        if context:self.lib.metric_release(context)
    def cube(self,context,index):
        if self.lib.metric_cube_json(context,index,self.buffer,len(self.buffer))<0:raise RuntimeError(self.buffer.value.decode())
        result=json.loads(self.buffer.value.decode())
        return result if result['valid'] else None
