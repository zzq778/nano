"""Independent numeric parity tests for the r39 frame-local hot paths."""
import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from rescuevision.geometry import Geometry
from rescuevision.pipeline import top_quad


class PinholeAndTopQuadContracts(unittest.TestCase):
    def test_pinhole_projection_matches_cv2_over_field_points(self):
        calibration = json.loads((ROOT / 'configs/astra_current.json').read_text())
        geometry = Geometry(calibration)
        self.assertTrue(geometry.use_pinhole)
        rng = np.random.RandomState(39)
        for x, y in rng.uniform([.08, -.6], [2., .6], size=(500, 2)):
            observed = {}
            geometry.set_position(observed, [x, y], 'test')
            optical = geometry.R.T.dot(np.array([x, y, geometry.ground]) - geometry.t)
            if optical[2] <= 0:
                self.assertNotIn('ground_center_projection_px', observed)
                continue
            expected = cv2.projectPoints(optical.reshape(1, 3), np.zeros(3), np.zeros(3),
                                         geometry.K, None)[0].reshape(2)
            np.testing.assert_allclose(observed['ground_center_projection_px'], expected,
                                       atol=1e-10, rtol=0)
            self.assertEqual(observed['x_mm'], round(x * 1000))
            self.assertEqual(observed['y_mm'], round(y * 1000))

    def test_pinhole_and_distorted_camera_keep_distinct_projection_paths(self):
        cfg = json.loads((ROOT / 'configs/astra_current.json').read_text())
        cfg['projection_policy'] = 'distortion_calibrated'
        cfg['rectification_status'] = 'unrectified'
        cfg['distortion_coefficients'] = [.1, -.03, .001, -.002, .01]
        geometry = Geometry(cfg)
        self.assertTrue(geometry.ready)
        self.assertFalse(geometry.use_pinhole)
        point = [.6, .1]
        result = {}
        geometry.set_position(result, point, 'distorted')
        optical = geometry.R.T.dot(np.array([point[0], point[1], geometry.ground]) - geometry.t)
        expected = cv2.projectPoints(optical.reshape(1,3), np.zeros(3), np.zeros(3),
                                     geometry.K, geometry.D)[0].reshape(2)
        np.testing.assert_allclose(result['ground_center_projection_px'], expected, atol=1e-10, rtol=0)

    def test_top_quad_shared_hsv_preserves_corners_and_edge_cases(self):
        image = np.full((480, 640, 3), 235, dtype=np.uint8)
        cv2.fillConvexPoly(image, np.array([[180,120],[305,130],[300,225],[185,219]]), (0, 135, 255))
        examples = [
            (dict(type=2, bbox=[170, 115, 315, 235]),
             dict(status='edge_candidate', back_line=[0, 110], front_line=[0, 235])),
            (dict(type=2, bbox=[0, 0, 40, 45]),
             dict(status='edge_candidate', back_line=[0, 0], front_line=[0, 40])),
            (dict(type=2, bbox=[-10, 460, 20, 500]),
             dict(status='edge_candidate', back_line=[0, 460], front_line=[0, 480]))]
        for path in sorted((ROOT/'data/approved_six/raw').glob('*.png')):
            sample = cv2.imread(str(path))
            examples.append((dict(type=2,bbox=[150,130,420,390]),
                             dict(status='edge_candidate',back_line=[0,155],front_line=[0,385])))
            self.assertEqual(top_quad(sample,examples[-1][0],examples[-1][1]),
                             top_quad(sample,examples[-1][0],examples[-1][1],cv2.cvtColor(sample,cv2.COLOR_BGR2HSV)))
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        for observation,refinement in examples[:3]:
            self.assertEqual(top_quad(image,observation,refinement),
                             top_quad(image,observation,refinement,hsv))
        self.assertIsNotNone(top_quad(image,*examples[0]))


if __name__=='__main__':
    unittest.main()
