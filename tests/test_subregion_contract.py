import sys,math,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from rescuevision.zones import own_subregion_targets, plausible_platform_sides
from rescuevision import uart

class SubregionContract(unittest.TestCase):
    def test_relaxed_sizes_reject_extremes(self):
        for sides in ((.3,.6),(.5,.964),(.17,.32)):
            self.assertTrue(plausible_platform_sides(sides))
        for sides in ((.084,.362),(.7,1.5),(.3,.3),(float('nan'),.6),(.01,.8)):
            self.assertFalse(plausible_platform_sides(sides))
    def zone(self,angle,center=(.8,0),selected=True):
        return dict(selected_for_uart=selected,position_valid=True,pose_valid=True,
                    entrance_valid=True,center_xy_m=list(center),inward_yaw_deg=angle)
    def test_rotations_and_material_left(self):
        for angle in (0,90,-90,180,-179.99,45):
            targets=own_subregion_targets([self.zone(angle)])
            self.assertEqual([t['type'] for t in targets],[4,5])
            decoded=uart.decode(uart.encode(targets));self.assertEqual(decoded,targets)
            a=math.radians(angle)
            for target,sign in zip(targets,(1,-1)):
                self.assertAlmostEqual(target['x_mm'],800-sign*150*math.sin(a),delta=.51)
                self.assertAlmostEqual(target['y_mm'],sign*150*math.cos(a),delta=.51)
    def test_upper_lower_match_f407_field(self):
        for y,angle,material_x in ((1350,90,-150),(-1350,-90,150)):
            t=own_subregion_targets([self.zone(angle,(0,y/1000.))])
            self.assertEqual((t[0]['x_mm'],t[0]['y_mm']),(material_x,y))
            self.assertEqual((t[1]['x_mm'],t[1]['y_mm']),(-material_x,y))
    def test_no_enemy_invalid_or_duplicate(self):
        self.assertEqual(own_subregion_targets([self.zone(0,selected=False)]),[])
        z=self.zone(0);z['entrance_valid']=False
        self.assertEqual(own_subregion_targets([z]),[])
        self.assertEqual(len(own_subregion_targets([self.zone(0),self.zone(0)])),2)
    def test_type5_yaw_is_preserved(self):
        t=dict(type=5,x_mm=-32768,y_mm=32767,yaw_cdeg=-9000)
        self.assertEqual(uart.decode(uart.encode([t])),[t])
    def test_invalid_pair_is_not_partially_sent(self):
        for center in ((.8,32.7),(float('nan'),0),(float('inf'),0)):
            z=self.zone(0,center)
            self.assertEqual(own_subregion_targets([z]),[])
            self.assertFalse(z['included_in_target_list'])
            self.assertIn('uart_omission_reason',z)

if __name__=='__main__':unittest.main()
