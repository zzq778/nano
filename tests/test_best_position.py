"""Position and pose quality are independent; lower quality must not drop detections."""
import json,math,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from rescuevision.geometry import Geometry
from rescuevision.orientation import complete_orientation
from rescuevision.pipeline import Pipeline
from rescuevision.scheduling import pose_order
from rescuevision import uart

class BestPositionContracts(unittest.TestCase):
    def geometry(self):
        c=json.loads((ROOT/'configs/astra_current.json').read_text())
        dimensions=json.loads((ROOT/'configs/runtime.json').read_text())['dimensions_m']
        return Geometry(c,{int(k):v for k,v in dimensions.items()})

    def test_green_is_not_a_pose_task_and_never_exports_an_angle(self):
        objects=[dict(type=t,near=True,bbox=[0,0,10,20]) for t in (0,2,3,1)]
        self.assertEqual(pose_order(objects),[1])
        r=complete_orientation(0,dict(position_valid=True,pose_valid=True,yaw_world_deg=35))
        self.assertTrue(r['position_valid']);self.assertFalse(r['pose_available'])
        self.assertIsNone(r['yaw_world_deg'])

    def test_rejected_orange_retains_finite_model_best_position(self):
        g=self.geometry();obj=dict(type=2,center_px=[320,300],bbox=[250,250,390,350])
        fit=dict(status=3,best_effort=dict(x=.3,y=.04,error_px=4.,yaw_rad=.4))
        r=g.best_effort_position(obj,(480,640),g.from_model(obj,fit))
        self.assertTrue(r['position_valid']);self.assertEqual((r['x_mm'],r['y_mm']),(300,40))
        self.assertFalse(r['accuracy_verified']);self.assertEqual(r['position_quality'],'best_effort')
        self.assertFalse(r['pose_valid'])

    def test_all_four_types_can_use_known_height_bound_when_precise_solution_fails(self):
        g=self.geometry()
        for cls in (0,1,2,3):
            obj=dict(type=cls,center_px=[320,330],bbox=[290,300,350,360],visible_only=True,near=True)
            r=g.best_effort_position(obj,(480,640),dict(position_valid=False,pose_valid=False,reason='fixture_rejected'))
            self.assertTrue(r['position_valid'],(cls,r))
            self.assertEqual(r['position_quality'],'coarse_bounded')
            self.assertGreater(r['position_uncertainty_mm'],0)
            self.assertEqual(r['prior_position_reason'],'fixture_rejected')

    def test_topface_anchor_uses_top_height_without_requiring_body_partition(self):
        g=self.geometry();p=g.project([320,280],(480,640),g.ground+.04)
        obj=dict(type=0,top_face_anchor_px=[320,280],geometry_body_supported=False)
        r=g.best_effort_position(obj,(480,640),dict(position_valid=False,pose_valid=False))
        self.assertEqual((r['x_mm'],r['y_mm']),(round(p[0]*1000),round(p[1]*1000)))
        self.assertEqual(r['position_quality'],'topface_estimate')

    def test_unavailable_calibration_does_not_invent_coordinates(self):
        g=self.geometry();g.ready=False
        r=g.best_effort_position(dict(type=0,center_px=[320,300]),(480,640),dict(position_valid=False))
        self.assertFalse(r['position_valid'])

    def test_green_pipeline_uses_cube_position_not_model_pose_fit(self):
        import rescuevision.pipeline as module
        g=self.geometry();p=Pipeline.__new__(Pipeline)
        p.config=json.loads((ROOT/'configs/runtime.json').read_text());p.geometry=g;p.mode='yolo';p.green=None;p.known_size_requested=True;p.default_yaw={}
        p.frontend=SimpleNamespace(coverage=[0,1,2,3],detect=lambda image:[dict(type=0,confidence=.9,bbox=[200,200,290,310],center_px=[245,255])])
        calls=[]
        p.metric_solver=SimpleNamespace(prepare=lambda *a:object(),cube=lambda *a:dict(center_m=[.25,.03],candidate_radius_mm=2.,bbox_rms_px=1.),fit=lambda *a:(_ for _ in ()).throw(AssertionError('green pose fit called')),release=lambda ctx:calls.append('released'))
        r=p.process(np.zeros((480,640,3),np.uint8));obj=r['objects'][0]
        self.assertEqual(uart.decode(bytes.fromhex(r['uart_hex'])),[dict(type=0,x_mm=250,y_mm=30)])
        self.assertFalse(obj['metric']['pose_available']);self.assertEqual(r['pose_task_order'],[])
        self.assertEqual(calls,['released'])

if __name__=='__main__':unittest.main()
