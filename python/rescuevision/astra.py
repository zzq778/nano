"""Astra owns the device in the capture process; returned BGR storage is reused."""
import ctypes as C
import numpy as np


class NativeAstra:
    def __init__(self,library,uri,width=640,height=480):
        self.context=None
        self.lib=C.CDLL(str(library));self.error=C.create_string_buffer(1024)
        self.lib.astra_open.argtypes=[C.c_char_p,C.c_int,C.c_int,C.c_char_p,C.c_int]
        self.lib.astra_open.restype=C.c_void_p
        self.lib.astra_read.argtypes=[C.c_void_p,C.POINTER(C.c_ubyte),C.c_int,C.c_int,
                                     C.POINTER(C.c_ulonglong),C.POINTER(C.c_int),C.c_char_p,C.c_int]
        self.lib.astra_read.restype=C.c_int
        self.lib.astra_close.argtypes=[C.c_void_p];self.lib.astra_close.restype=None
        self.image=np.empty((height,width,3),np.uint8)
        self.device_timestamp=C.c_ulonglong();self.device_frame_index=C.c_int()
        self.context=self.lib.astra_open(str(uri).encode('utf-8'),width,height,self.error,len(self.error))
        if not self.context:raise RuntimeError(self.error.value.decode('utf-8','replace'))

    def read(self):
        if self.context is None:raise RuntimeError('Astra session is closed')
        result=self.lib.astra_read(self.context,self.image.ctypes.data_as(C.POINTER(C.c_ubyte)),
            self.image.nbytes,200,C.byref(self.device_timestamp),C.byref(self.device_frame_index),self.error,len(self.error))
        if result==0:raise TimeoutError('Astra frame wait timed out')
        if result<0:raise RuntimeError(self.error.value.decode('utf-8','replace'))
        return self.image

    def close(self):
        if self.context is not None:
            self.lib.astra_close(self.context);self.context=None
