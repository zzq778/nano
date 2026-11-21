"""r37 golden geometry projections, including invalid ROI and near/far samples."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from rescuevision.geometry import Geometry
from rescuevision.shape_center import cube_bbox_center


class ShapeCenterParity(unittest.TestCase):
    def test_pinhole_fast_math_matches_r37_golden_positions_and_quantization(self):
        fixture=json.loads((ROOT/'tests/fixtures/shape_center_r37_goldens.json').read_text())
        geometry=Geometry(json.loads((ROOT/'configs/astra_current.json').read_text()))
        for index,row in enumerate(fixture['cases']):
            with self.subTest(index=index):
                actual=cube_bbox_center(geometry,row['bbox'],(480,640))
                expected=row['expected']
                if expected is None:
                    self.assertIsNone(actual)
                else:
                    self.assertIsNotNone(actual)
                    self.assertEqual(actual['latent_yaw_samples'],expected['latent_yaw_samples'])
                    np.testing.assert_allclose(actual['center_m'],expected['center_m'],rtol=1e-9,atol=1e-8)
                    self.assertAlmostEqual(actual['bbox_rms_px'],expected['bbox_rms_px'],places=6)
                    self.assertAlmostEqual(actual['candidate_radius_mm'],expected['candidate_radius_mm'],places=4)
                    self.assertEqual([round(x*1000) for x in actual['center_m']],
                                     [round(x*1000) for x in expected['center_m']])

    def test_orange_top_quad_broadcast_preserves_corners(self):
        from rescuevision.pipeline import top_quad
        image=np.zeros((480,640,3),np.uint8)
        image[100:200,200:320]=[0,120,255]
        obj=dict(type=2,bbox=[180,80,340,220])
        edges=dict(status='edge_candidate',back_line=[0.,100.],front_line=[0.,200.])
        actual=top_quad(image,obj,edges)
        expected=[[319,100],[319,199],[200,199],[200,100]]
        # OpenCV may choose any starting contour vertex across versions.
        # Preserve all corners and their winding; do not accept reversed order.
        self.assertIsNotNone(actual)
        self.assertTrue(any(actual == expected[i:]+expected[:i] for i in range(4)), actual)
        self.assertIsNone(top_quad(image,obj,dict(status='fallback')))


if __name__=='__main__':unittest.main()