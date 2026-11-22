"""Equivalent full-frame vs occupied-ROI color components and floor hulls."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from rescuevision import zones
from rescuevision.pipeline import Pipeline
from rescuevision.geometry import Geometry
from rescuevision import uart


class OccupiedRoiContracts(unittest.TestCase):
    def test_mask_hull_is_exact_for_real_and_fragmented_masks(self):
        rng = np.random.RandomState(4040)
        masks = []
        for filename in sorted((ROOT / 'data/approved_six/raw').glob('*.png')):
            image = cv2.imread(str(filename))
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            masks.extend((cv2.inRange(hsv, (90, 85, 30), (145, 255, 255)),
                          cv2.inRange(hsv, (4, 145, 55), (20, 255, 255))))
        for _ in range(160):
            mask = np.zeros((64, 96), np.uint8)
            for j in range(int(rng.randint(1, 18))):
                x = int(rng.randint(0, 96))
                y = int(rng.randint(0, 64))
                w, h = map(int, rng.randint(1, 25, 2))
                cv2.rectangle(mask, (x, y), (min(x+w, 95), min(y+h, 63)), 255, -1)
            # Include islands, thin diagonal connections, and open holes.
            masks.append(mask)
        for mask in masks:
            points = cv2.findNonZero(mask)
            expected = cv2.convexHull(points).reshape(-1, 2) if points is not None else None
            actual = zones._color_hull(mask)
            if expected is None:
                self.assertIsNone(actual)
            else:
                np.testing.assert_array_equal(actual, expected)

    def test_native_metric_substage_timings_are_separately_reported(self):
        class FakeMetric:
            def __init__(self):
                self.calls = []
            def prepare(self, image, objects, balanced):
                self.calls.append('prepare')
                return 1
            def fit(self, context, index, budget):
                self.calls.append('fit')
                return dict(status=3, yaw_rad=0., candidates=[])
            def cube(self, context, index):
                self.calls.append('cube')
                return dict(center_m=[.65, .02, .02], candidate_radius_mm=3., bbox_rms_px=2.)
            def release(self, context):
                self.calls.append('release')
        class FakeFrontend:
            coverage = [0, 1, 2, 3]
            last_diagnostics = {}
            def detect(self, image):
                return [dict(type=2, bbox=[200., 230., 260., 300.], center_px=[230., 265.], confidence=.8),
                        dict(type=0, bbox=[280., 270., 340., 340.], center_px=[310., 305.], confidence=.8),
                        dict(type=3, bbox=[370., 265., 430., 335.], center_px=[400., 300.], confidence=.8)]
        metric = FakeMetric()
        p = Pipeline.__new__(Pipeline)
        p.geometry = Geometry(json.loads((ROOT / 'configs/astra_current.json').read_text()),
                              {0: [.04, .04, .04], 2: [.08, .04, .04], 3: [.04, .04, .04]})
        p.frontend = FakeFrontend();p.zone_detector = SimpleNamespace(zones=[])
        p.metric_solver = metric;p.known_size_requested = True
        p.green = None;p.orange = None;p.orange_native = None
        p.config = dict(near_line_fraction=.5,refine_far=True,pose_budget_ms=105,
                        metric_per_object_budget_ms=0,metric_white_reference=True,
                        placement_zone_color='blue',uart_required_types=[0,1,2,3],uart_baud=230400)
        p.mode = 'yolo';p.default_yaw = {}
        result = p.process(np.full((480, 640, 3), 100, np.uint8))
        self.assertEqual(metric.calls, ['prepare', 'fit', 'cube', 'cube', 'release'])
        times = result['timings_ms']
        for stage in ('native_metric_prepare', 'native_metric_orange_fit', 'native_metric_cube_fit'):
            self.assertIn(stage, times)
            self.assertGreaterEqual(times[stage], 0)
        # All three stage timers are parts of the already-reported total.
        self.assertLessEqual(sum(times[stage] for stage in
                             ('native_metric_prepare', 'native_metric_orange_fit', 'native_metric_cube_fit')),
                             times['scheduled_refinement_and_veto'] + 0.5)
        self.assertEqual(uart.decode(bytes.fromhex(result['uart_hex'])),
                         [dict(type=o['type'],x_mm=o['metric']['x_mm'],y_mm=o['metric']['y_mm'],
                               **({'yaw_cdeg':0} if o['type']==2 else {}))
                          for o in result['objects'] if o['position_valid']])


if __name__ == '__main__':
    unittest.main()
