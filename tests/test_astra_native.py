import ctypes as C
import os
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from rescuevision.astra import NativeAstra
from native_support import native_library_matches_host


@unittest.skipUnless(os.name=='posix' and native_library_matches_host(ROOT/'build/libvision_astra.so'),'Astra native ABI checked on Nano')
class NativeAstraContracts(unittest.TestCase):
    def test_invalid_mode_and_discarded_uri_are_rejected_without_opening_hardware(self):
        library=ROOT/'build/libvision_astra.so'
        with self.assertRaisesRegex(RuntimeError,'640x480'):
            NativeAstra(library,'astra',320,240)
        with self.assertRaisesRegex(RuntimeError,'expected astra'):
            NativeAstra(library,'0')

    def test_invalid_read_does_not_write_the_output_buffer(self):
        library=C.CDLL(str(ROOT/'build/libvision_astra.so'))
        library.astra_read.argtypes=[C.c_void_p,C.POINTER(C.c_ubyte),C.c_int,C.c_int,
                                    C.POINTER(C.c_ulonglong),C.POINTER(C.c_int),C.c_char_p,C.c_int]
        storage=np.full(8,167,np.uint8);stamp=C.c_ulonglong(123);index=C.c_int(456);error=C.create_string_buffer(1024)
        code=library.astra_read(None,storage.ctypes.data_as(C.POINTER(C.c_ubyte)),len(storage),200,
                               C.byref(stamp),C.byref(index),error,len(error))
        self.assertEqual(code,-1)
        np.testing.assert_array_equal(storage,np.full(8,167,np.uint8))
        self.assertEqual((stamp.value,index.value),(123,456))


if __name__=='__main__':unittest.main()
