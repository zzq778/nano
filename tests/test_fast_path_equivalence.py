"""Focused regressions for the color and contour fast paths."""
import copy
import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from rescuevision.frontends import (
    YoloFrontend, blue_context_proposals, merge_blue_proposals,
    orange_visible_proposals, green_seed,
)
from rescuevision.geometry import Geometry


class FastPathContracts(unittest.TestCase):
    def test_reused_hsv_preserves_color_supplement_detections(self):
        config = json.loads((ROOT / 'configs/runtime.json').read_text())
        image = cv2.imread(str(ROOT / 'data/approved_six/raw/01.png'))
        rows = [
            dict(type=0, bbox=[110., 200., 175., 280.], confidence=.9),
            dict(type=2, bbox=[290., 250., 370., 310.], confidence=.9),
            dict(type=3, bbox=[440., 220., 520., 310.], confidence=.9),
            dict(type=4, bbox=[20., 140., 200., 260.], confidence=.9),
        ]

        class Detector:
            def run(self, frame):
                return copy.deepcopy(rows)

        frontend = YoloFrontend(Detector(), config['blue_context_supplement'])
        actual = frontend.detect(image)
        expected = copy.deepcopy(rows[:3])
        for row in expected:
            x0, y0, x1, y1 = row['bbox']
            row.update(center_px=[(x0+x1)*.5, (y0+y1)*.5],
                       source='current_sixclass_yolo', center_kind='bbox_center')
            if row['type'] == 0:
                row['seed'] = green_seed(row['bbox'], image.shape[1], image.shape[0])
        expected = merge_blue_proposals(expected, blue_context_proposals(image, expected, config['blue_context_supplement']))
        expected += orange_visible_proposals(image, expected)
        self.assertEqual(actual, expected)
        self.assertEqual(frontend.zones[0]['color'], 'purple')
        self.assertIn('native_detector', frontend.last_diagnostics['timings_ms'])

    def test_orange_color_gate_matches_original_float_comparison(self):
        images = [cv2.imread(str(path)) for path in sorted((ROOT / 'data/approved_six/raw').glob('*.png'))]
        images.append(np.random.RandomState(7).randint(0, 256, (480, 640, 3)).astype(np.uint8))
        for image in images:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            old = cv2.inRange(hsv, (4, 145, 55), (20, 255, 255))
            b, g, r = cv2.split(image)
            old[(r.astype(np.int16) < g.astype(np.int16)*1.38) |
                (r.astype(np.int16) < b.astype(np.int16)*1.55)] = 0

            new = cv2.inRange(hsv, (4, 145, 55), (20, 255, 255))
            coordinates = cv2.findNonZero(new)
            if coordinates is not None:
                xy = coordinates.reshape(-1, 2)
                bgr = image[xy[:, 1], xy[:, 0]]
                invalid = (bgr[:, 2] < bgr[:, 1]*1.38) | (bgr[:, 2] < bgr[:, 0]*1.55)
                new[xy[invalid, 1], xy[invalid, 0]] = 0
            self.assertTrue(np.array_equal(old, new))

    def test_contour_batch_projection_matches_existing_scalar_api(self):
        geometry = Geometry(json.loads((ROOT / 'configs/astra_current.json').read_text()))
        pixels = np.array([[250., 350.], [300., 370.], [320., 420.], [450., 300.]])
        expected = [geometry.project(pixel, (480, 640), geometry.ground) for pixel in pixels]
        actual = geometry.project_many(pixels, (480, 640), geometry.ground)
        self.assertTrue(np.allclose(actual, expected, rtol=1e-12, atol=1e-12))
        self.assertIsNone(geometry.project_many([[float('nan'), 220]], (480, 640), geometry.ground))
        with self.assertRaises(ValueError):
            geometry.project_many(pixels, (240, 320), geometry.ground)


if __name__ == '__main__':
    unittest.main()
