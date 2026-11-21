"""No legacy fault packet producer or decoder; partial targets remain useful."""
import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from rescuevision import uart
from rescuevision.pipeline import Pipeline

class TargetOnlyContracts(unittest.TestCase):
    def test_legacy_fault_packet_is_rejected_and_has_no_public_producer(self):
        self.assertFalse(hasattr(uart,'fault'))
        with self.assertRaises(ValueError):uart.decode(bytes.fromhex('aa55ff00ff'))
        self.assertEqual(uart.decode(uart.encode([])),[])

    def run_scene(self,positions,simulation=False):
        objects=[dict(type=3,bbox=[10+30*i,10,30+30*i,40],center_px=[20+30*i,25]) for i in range(len(positions))]
        p=Pipeline.__new__(Pipeline)
        p.config=json.loads((ROOT/'configs/runtime.json').read_text())
        p.frontend=SimpleNamespace(coverage=[0,1,2,3],detect=lambda image:copy.deepcopy(objects))
        def estimate(obj,shape):
            index=int((obj['bbox'][0]-10)/30);xy=positions[index]
            return dict(position_valid=xy is not None,pose_valid=False,x_mm=xy[0] if xy else None,y_mm=xy[1] if xy else None)
        p.geometry=SimpleNamespace(config={'image_size':[640,480],'simulation_only':simulation},ready=True,pinhole_estimate=True,reason='unverified_distortion',estimate=estimate,best_effort_position=lambda obj,shape,result:result)
        p.mode='yolo';p.green=None;p.metric_solver=None;p.known_size_requested=True;p.default_yaw={}
        return p.process(np.zeros((480,640,3),np.uint8))

    def test_one_unlocated_object_does_not_suppress_another(self):
        r=self.run_scene([(400,20),None])
        self.assertEqual(uart.decode(bytes.fromhex(r['uart_hex'])),[dict(type=3,x_mm=400,y_mm=20)])
        self.assertEqual(r['unlocated_object_count'],1)
        self.assertNotIn('uart_fault',r)
        self.assertIn('distortion_unverified',r['diagnostics'])

    def test_bad_wire_coordinate_does_not_suppress_another(self):
        r=self.run_scene([(400,20),(50000,0)])
        self.assertEqual(len(uart.decode(bytes.fromhex(r['uart_hex']))),1)
        self.assertTrue(any(d.startswith('wire_target_omitted:') for d in r['diagnostics']))

    def test_unlocated_and_simulated_scenes_send_empty_normal_packets(self):
        for r in (self.run_scene([None]),self.run_scene([(400,20)],True)):
            self.assertEqual(uart.decode(bytes.fromhex(r['uart_hex'])),[])

if __name__=='__main__':unittest.main()
