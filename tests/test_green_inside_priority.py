"""Regression for the competing associations observed in M284."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from rescuevision.green_association import fuse


def face(x, y):
    return dict(seed=dict(x=x, y=y, x_span=[x-10, x+10], y_span=[y-5, y+5]),
                result=dict(status='edge_candidate', center=[x, y],
                            back_line=[0, y-5], front_line=[0, y+5]))


class InsidePriority(unittest.TestCase):
    def test_inside_face_displaces_only_padding_match(self):
        result = fuse([dict(bbox=[0, 0, 100, 100])],
                                  [face(105, 28), face(95, 30)], True)
        self.assertEqual(result['associations'], [dict(yolo=0, face=1)])
        self.assertEqual([item['face'] for item in result['added_faces']], [0])
        self.assertEqual(result['experimental_count'], 2)
        self.assertEqual(result['rejected_yolo_groups'], [])

    def test_only_padded_face_can_still_be_matched(self):
        result = fuse([dict(bbox=[0, 0, 100, 100])], [face(105, 28)], True)
        self.assertEqual(result['associations'], [dict(yolo=0, face=0)])
        self.assertEqual(result['experimental_count'], 1)


if __name__ == '__main__':
    unittest.main()
