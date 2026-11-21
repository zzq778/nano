import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from rescuevision import uart

class WholeRefugeWire(unittest.TestCase):
    def test_body_frame_signed_normal_angles(self):
        for angle in (0,9000,-9000,-18000,17999):
            target=dict(type=4,x_mm=800,y_mm=-100,yaw_cdeg=angle)
            self.assertEqual(uart.decode(uart.encode([target])),[target])
    def test_positive_180_and_out_of_range_angles_are_rejected(self):
        for angle in (18000,-18001,32767):
            with self.assertRaises(ValueError):uart.encode([dict(type=4,x_mm=0,y_mm=0,yaw_cdeg=angle)])
    def test_blocks_keep_zero_wire_yaw(self):
        for cls in (0,1,3):
            packet=uart.encode([dict(type=cls,x_mm=450,y_mm=-100,yaw_cdeg=9000)])
            self.assertEqual(packet[8:10],b'\0\0')
    def test_empty_and_green_official_vectors_stay_unchanged(self):
        self.assertEqual(uart.encode([]).hex(),'aa5500f0e1')
        self.assertEqual(uart.encode([dict(type=0,x_mm=450,y_mm=-100)]).hex(),'aa550100c2019cff000002af')

if __name__=='__main__':unittest.main()
