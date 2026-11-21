import json,sys,unittest
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from rescuevision.geometry import Geometry
from rescuevision.zones import estimate_zone_regions

class ZoneContracts(unittest.TestCase):
    def scene(self,color='blue',divider=True):
        geometry=Geometry(json.loads((ROOT/'configs/astra_current.json').read_text()))
        # Cyclic inner floor corners for a rectangle facing the camera in the field.
        world=np.array([[.65,.3,geometry.ground],[.65,-.3,geometry.ground],[.95,-.3,geometry.ground],[.95,.3,geometry.ground]])
        rays=(world-geometry.t).dot(geometry.R)
        uv=np.column_stack([geometry.K[0,0]*rays[:,0]/rays[:,2]+geometry.K[0,2],geometry.K[1,1]*rays[:,1]/rays[:,2]+geometry.K[1,2]])
        image=np.full((480,640,3),180,np.uint8)
        cv2.fillConvexPoly(image,np.int32(uv),(200,100,30) if color=='blue' else (40,20,210))
        if divider:cv2.line(image,tuple(np.int32((uv[0]+uv[1])/2)),tuple(np.int32((uv[2]+uv[3])/2)),(25,25,25),4)
        low=uv.min(axis=0)-10;high=uv.max(axis=0)+10
        return image,dict(color=color,bbox=[*low,*high],confidence=.9),geometry
    def test_complete_blue_floor_has_geometry_but_is_not_selected_by_default_purple(self):
        image,detection,geometry=self.scene();result=estimate_zone_regions(image,[detection],geometry,'purple')[0]
        self.assertTrue(result['position_valid'],result)
        self.assertFalse(result['selected_for_uart']);self.assertFalse(result['transmitted'])
    def test_purple_is_selected_and_configuration_can_choose_blue(self):
        image,detection,geometry=self.scene('purple');result=estimate_zone_regions(image,[detection],geometry,'purple')[0]
        self.assertTrue(result['selected_for_uart']);self.assertTrue(result['entrance_valid'],result)
        self.assertAlmostEqual(result['center_xy_m'][1],0.,delta=.01)
        image,detection,geometry=self.scene();self.assertTrue(estimate_zone_regions(image,[detection],geometry,'blue')[0]['selected_for_uart'])
    def test_whole_region_does_not_require_internal_divider(self):
        image,detection,geometry=self.scene(divider=False);result=estimate_zone_regions(image,[detection],geometry,'blue')[0]
        self.assertTrue(result['position_valid'],result);self.assertTrue(result['entrance_valid'])
    def test_frame_edge_does_not_automatically_reject_geometry(self):
        image,detection,geometry=self.scene();detection['bbox'][0]=0
        result=estimate_zone_regions(image,[detection],geometry,'blue')[0]
        self.assertTrue(result['position_valid'],result)
        self.assertNotIn('complete_visible',result)
    def test_missing_corner_has_no_four_corner_completeness_gate(self):
        image,detection,geometry=self.scene(divider=False)
        mask=cv2.inRange(cv2.cvtColor(image,cv2.COLOR_BGR2HSV),(80,35,40),(130,255,255))
        ys,xs=np.nonzero(mask)
        index=int(np.argmin(xs+ys));cv2.circle(image,(int(xs[index]),int(ys[index])),12,(180,180,180),-1)
        result=estimate_zone_regions(image,[detection],geometry,'blue')[0]
        self.assertTrue(result['position_valid'],result)
        self.assertEqual(result['center_source'],'visible_floor_ground_rectangle_estimate')

if __name__=='__main__':unittest.main()
