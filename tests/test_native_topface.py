import ctypes as C
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from rescuevision.topface_native import NativeTopFace, _Face
from native_support import native_library_matches_host


@unittest.skipUnless(native_library_matches_host(ROOT / 'build/libvision_green.so'),
                     'Native top-face contracts run after the Nano build')
class NativeTopFaceContracts(unittest.TestCase):
    def setUp(self):
        self.kernel = NativeTopFace(ROOT / 'build/libvision_green.so')

    def test_rejects_invalid_images_before_native_allocation(self):
        for image in (None, np.zeros((0, 2, 3), np.uint8),
                      np.zeros((3, 3), np.uint8), np.zeros((3, 3, 3), np.float32),
                      np.zeros((1, 10001, 3), np.uint8)):
            with self.assertRaises(ValueError):
                self.kernel.detect(image)

    def test_empty_scene_is_not_a_fault_or_invented_target(self):
        self.assertEqual(self.kernel.detect(np.zeros((48, 64, 3), np.uint8)), [])

    def test_strided_input_and_owned_result_do_not_alias_reused_storage(self):
        sample = ROOT / 'data/approved_six/raw/01.png'
        if not sample.is_file():
            self.skipTest('requires private approved_six frame 01.png')
        image = cv2.imread(str(sample))
        self.assertIsNotNone(image)
        view = image[:, ::2]
        before = view.copy()
        result = self.kernel.detect(view)
        expected = self.kernel.detect(np.ascontiguousarray(view))
        self.assertEqual(result, expected)
        self.kernel.detect(np.zeros_like(image))
        self.assertEqual(result, expected)
        np.testing.assert_array_equal(view, before)

    def test_capacity_failure_does_not_publish_a_partial_candidate_list(self):
        sample = ROOT / 'data/approved_six/raw/01.png'
        if not sample.is_file():
            self.skipTest('requires private approved_six frame 01.png')
        image = cv2.imread(str(sample))
        self.assertIsNotNone(image)
        self.assertGreater(len(self.kernel.detect(image)), 1)
        output = (_Face * 1)()
        output[0].x = 12345
        error = C.create_string_buffer(512)
        count = self.kernel._detect(image.ctypes.data_as(C.POINTER(C.c_ubyte)),
                                     image.shape[1], image.shape[0], image.strides[0],
                                     output, 1, error, len(error))
        self.assertLess(count, 0)
        self.assertIn(b'capacity', error.value)
        self.assertEqual(output[0].x, 12345)
        self.assertGreater(len(self.kernel.detect(image)), 1)


if __name__ == '__main__':
    unittest.main()