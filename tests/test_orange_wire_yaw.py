import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from rescuevision import uart
from rescuevision.orientation import orange_yaw_cdeg

class OrangeWireYaw(unittest.TestCase):
    def test_orange_measured_and_best_effort_angles_reach_wire(self):
        for angle in (0,23.42,168.19,179.999,-10):
            for source in ('accepted_single_solution','rejected_model_best_effort'):
                yaw=orange_yaw_cdeg(dict(yaw_world_deg=angle,yaw_source=source))
                self.assertEqual(yaw,round(angle*100)%18000)
                target=dict(type=2,x_mm=230,y_mm=-10,yaw_cdeg=yaw)
                self.assertEqual(uart.decode(uart.encode([target])),[target])
    def test_unknown_angle_does_not_invent_a_measurement(self):
        for m in ({},{'yaw_world_deg':float('nan')},{'yaw_world_deg':30,'yaw_source':'configured_default'}):
            self.assertEqual(orange_yaw_cdeg(m),0)
    def test_other_blocks_stay_zero(self):
        for cls in (0,1,3):
            p=uart.encode([dict(type=cls,x_mm=10,y_mm=-20,yaw_cdeg=1234)])
            self.assertEqual(p[8:10],b'\0\0')

if __name__=='__main__':unittest.main()
