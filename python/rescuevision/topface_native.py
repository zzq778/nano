"""Typed boundary for the selected native top-face kernel; no automatic fallback."""
import ctypes as C
import numpy as np


class _Face(C.Structure):
    _fields_ = [('x', C.c_double), ('y', C.c_double)] + [
        (name, C.c_int) for name in (
            'pixels', 'component_x', 'component_y', 'component_width',
            'component_height', 'xmin', 'xmax', 'ymin', 'ymax',
            'split', 'isolated_fallback')]


class NativeTopFace:
    def __init__(self, library):
        self._library = C.CDLL(str(library))
        self._detect = self._library.topface_detect
        self._detect.argtypes = [C.POINTER(C.c_ubyte)] + [C.c_int] * 3 + [
            C.POINTER(_Face), C.c_int, C.c_char_p, C.c_int]
        self._detect.restype = C.c_int
        self._capacity = 0
        self._output = None
        self._error = C.create_string_buffer(512)

    def detect(self, image):
        if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('topface requires uint8 BGR image')
        height, width = image.shape[:2]
        if not height or not width:
            raise ValueError('topface requires nonempty image')
        if height > 10000 or width > 10000:
            raise ValueError('topface image dimensions exceed the native limit')
        image = np.ascontiguousarray(image)
        # Measured surfaces have >=140 distinct pixels, isolated fallbacks >=300.
        # Their sum is a conservative bound; reuse the buffer across frames.
        required = width * height // 140 + width * height // 300 + 1
        if required > self._capacity:
            self._capacity = required
            self._output = (_Face * required)()
        count = self._detect(image.ctypes.data_as(C.POINTER(C.c_ubyte)), width,
                             height, int(image.strides[0]), self._output,
                             self._capacity, self._error, len(self._error))
        if count < 0:
            raise RuntimeError(self._error.value.decode('utf-8', 'replace'))
        result = []
        for p in self._output[:count]:
            face = dict(x=p.x, y=p.y, n_pixels=p.pixels,
                        source_comp=[p.component_x, p.component_y,
                                     p.component_width, p.component_height],
                        x_span=[p.xmin, p.xmax], y_span=[p.ymin, p.ymax],
                        split_from_large=bool(p.split))
            if p.isolated_fallback:
                face['method'] = 'HSV isolated fallback (not top face)'
            result.append(face)
        return result
