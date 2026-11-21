"""Regressions specific to frame-local color reuse and nonblocking sampling."""
import copy
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from rescuevision.frontends import color_center,black_apex_evidence
from rescuevision.green_proposals import GreenFaceProposer
from rescuevision.geometry import Geometry
from rescuevision.pipeline import Pipeline
from rescuevision.sampling import AsyncSampleRecorder
from rescuevision import uart
from rescuevision.zones import estimate_zone_regions


class SharedFrameContracts(unittest.TestCase):
    def test_reusing_hsv_preserves_block_center_and_apex(self):
        for path in sorted((ROOT/'data/approved_six/raw').glob('*.png')):
            image=cv2.imread(str(path))
            hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
            for cls in (1,3):
                obj=dict(type=cls,bbox=[90,120,430,340],center_px=[260,230])
                self.assertEqual(color_center(image,obj),color_center(image,obj,hsv))
            obj=dict(type=1,bbox=[90,120,430,340])
            self.assertEqual(black_apex_evidence(image,obj),black_apex_evidence(image,obj,hsv))

    def test_reusing_hsv_preserves_zone_geometry(self):
        geometry=Geometry(json.loads((ROOT/'configs/astra_current.json').read_text()))
        fixture=json.loads((ROOT/'data/zone_contract/fixture.json').read_text())
        image=cv2.imread(str(ROOT/'data/zone_contract'/fixture['image']))
        for color in ('blue','purple'):
            detections=[dict(fixture['detection']),dict(color='purple',bbox=[0.,160.,170.,330.],confidence=.2)]
            expected=estimate_zone_regions(image,copy.deepcopy(detections),geometry,color)
            actual=estimate_zone_regions(image,copy.deepcopy(detections),geometry,color,cv2.cvtColor(image,cv2.COLOR_BGR2HSV))
            self.assertEqual(expected,actual)

    def test_green_proposals_same_when_reusing_hsv(self):
        image=cv2.imread(str(ROOT/'data/approved_six/raw/02.png'))
        boxes=[[150.,180.,380.,410.]]
        def kernel(crop):return []
        proposer=GreenFaceProposer(kernel)
        expected=proposer.propose(image,boxes)
        actual=proposer.propose(image,boxes,cv2.cvtColor(image,cv2.COLOR_BGR2HSV))
        self.assertEqual(expected,actual)

    def test_empty_detections_do_not_allocate_metric_context(self):
        class Metric:
            def prepare(self,*args):raise AssertionError('RGB metric context must be skipped')
        class Frontend:
            coverage=[0,1,2,3]
            last_diagnostics={}
            def detect(self,image):return []
        class GeometryStub:
            config=dict(image_size=[640,480])
            ready=False
            reason='not_calibrated'
            pinhole_estimate=False
        p=Pipeline.__new__(Pipeline)
        p.frontend=Frontend();p.metric_solver=Metric();p.geometry=GeometryStub()
        p.green=None;p.zone_detector=SimpleNamespace(zones=[])
        p.config=dict(near_line_fraction=.5,placement_zone_color='blue',uart_required_types=[0,1,2,3],uart_baud=230400)
        p.mode='yolo';p.default_yaw={}
        result=p.process(np.zeros((480,640,3),np.uint8))
        self.assertEqual(result['uart_target_count'],0)
        self.assertEqual(result['objects'],[])


class AsyncSamplingContracts(unittest.TestCase):
    def test_sample_pair_is_owned_and_io_is_off_main_thread(self):
        with tempfile.TemporaryDirectory() as root:
            sampler=AsyncSampleRecorder(root,5.,2,{'origin':'fixture'})
            image=np.zeros((480,640,3),np.uint8)
            image[0,0]=[11,22,33]
            packet=uart.encode([dict(type=0,x_mm=100,y_mm=200)])
            original_imwrite=cv2.imwrite
            def slow_write(path,frame):
                time.sleep(.12)
                return original_imwrite(path,frame)
            with patch('rescuevision.sampling.cv2.imwrite',side_effect=slow_write):
                start=time.perf_counter()
                event=sampler.maybe_save(image,dict(sequence_local=7,transport=dict(kind='physical_uart')),packet,now=1.)
                duration=time.perf_counter()-start
                image[0,0]=[77,88,99]  # producer is allowed to reuse/mutate after submission
                self.assertEqual(event['record_kind'],'sample_queued')
                self.assertLess(duration,.06)
                self.assertIsNone(sampler.maybe_save(image,dict(sequence_local=8),packet,now=2.))
                summary=sampler.close()
            self.assertEqual((summary['accepted'],summary['saved'],summary['error']),(1,1,None))
            rows=list(sampler.recorder.root.glob('*seq*.json'))
            self.assertEqual(len(rows),1)
            row=json.loads(rows[0].read_text())
            self.assertEqual(row['sent_uart_hex'],packet.hex())
            self.assertEqual(row['result']['sequence_local'],7)
            sample=cv2.imread(str(sampler.recorder.root/row['image']))
            self.assertEqual(sample[0,0].tolist(),[11,22,33])

    def test_error_is_reported_after_async_flush(self):
        with tempfile.TemporaryDirectory() as root:
            sampler=AsyncSampleRecorder(root,5.,2,{})
            with patch('rescuevision.sampling.cv2.imwrite',side_effect=OSError('full disk')):
                sampler.maybe_save(np.zeros((4,4,3),np.uint8),dict(sequence_local=3),uart.encode([]),now=1.)
                summary=sampler.close()
            self.assertEqual(summary['saved'],0)
            self.assertIn('full disk',summary['error'])


if __name__=='__main__':unittest.main()
