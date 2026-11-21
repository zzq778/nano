"""Actual zone pixels -> geometry -> Pipeline target list -> UART codec.

The rectified flag in these synthetic fixtures is not a real calibration result.
"""
import json,sys,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from rescuevision.pipeline import Pipeline
from rescuevision import uart
import test_zones as zone_fixtures
ROOT=zone_fixtures.ROOT

class ZonePipelineContracts(unittest.TestCase):
    def process(self,color='purple',selected='purple',verified_fixture=True,truncated=False,block_count=0):
        image,detection,geometry=zone_fixtures.ZoneContracts().scene(color,divider=False)
        if verified_fixture:
            geometry.pinhole_estimate=False
            geometry.reason='synthetic_rectified_fixture_not_a_measured_calibration'
        if truncated:detection['bbox'][0]=0
        pipeline=Pipeline.__new__(Pipeline)
        pipeline.config=json.loads((ROOT/'configs/runtime.json').read_text())
        pipeline.config['placement_zone_color']=selected
        blocks=[dict(type=2,bbox=[10,200,30,230],center_px=[20,215],
                     metric=dict(position_valid=True,pose_valid=False,x_mm=400,y_mm=i))
                for i in range(block_count)]
        pipeline.frontend=SimpleNamespace(coverage=[0,1,2,3],detect=lambda frame:blocks)
        pipeline.zone_detector=SimpleNamespace(zones=[detection])
        pipeline.geometry=geometry;pipeline.mode='yolo';pipeline.green=None
        pipeline.metric_solver=None;pipeline.known_size_requested=True;pipeline.default_yaw={}
        return pipeline.process(image)
    def test_selected_platform_sends_two_subregion_centers(self):
        result=self.process();targets=uart.decode(bytes.fromhex(result['uart_hex']))
        self.assertNotIn('uart_fault',result)
        self.assertEqual([t['type'] for t in targets],[4,5])
        self.assertAlmostEqual(targets[0]['x_mm'],800,delta=10)
        self.assertAlmostEqual(targets[0]['y_mm'],150,delta=10)
        self.assertAlmostEqual(targets[1]['y_mm'],-150,delta=10)
        self.assertAlmostEqual(targets[1]['x_mm'],800,delta=10)
        self.assertAlmostEqual(targets[0]['yaw_cdeg'],0,delta=100)
    def test_other_color_stays_diagnostic_and_does_not_leak_into_uart(self):
        result=self.process('blue','purple')
        self.assertEqual(len(result['placement_zones']),1)
        self.assertEqual(uart.decode(bytes.fromhex(result['uart_hex'])),[])
    def test_full_block_capacity_preserves_both_landmarks(self):
        result=self.process(block_count=32)
        targets=uart.decode(bytes.fromhex(result['uart_hex']))
        self.assertEqual(len(targets),32)
        self.assertEqual([t['type'] for t in targets[:2]],[4,5])
        self.assertEqual(sum(t['type']==2 for t in targets),30)
    def test_switch_to_blue_keeps_same_wire_type4(self):
        result=self.process('blue','blue')
        self.assertEqual(uart.decode(bytes.fromhex(result['uart_hex']))[0]['type'],4)
    def test_frame_boundary_is_not_a_wire_exclusion_rule(self):
        result=self.process(truncated=True)
        self.assertEqual(uart.decode(bytes.fromhex(result['uart_hex']))[0]['type'],4)
    def test_unverified_distortion_is_diagnostic_and_does_not_block_targets(self):
        result=self.process(verified_fixture=False)
        self.assertEqual(len(uart.decode(bytes.fromhex(result['uart_hex']))),2)
        self.assertIn('distortion_unverified',result['diagnostics'])
        preview=uart.decode(bytes.fromhex(result['estimate_packet_preview']['hex']))
        self.assertEqual(len(preview),2);self.assertEqual(preview[0]['type'],4)

if __name__=='__main__':unittest.main()
