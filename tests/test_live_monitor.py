import json,sys,time,unittest,urllib.request
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from rescuevision.live_monitor import LiveMonitor,sent_targets,project
from rescuevision import uart

class Contracts(unittest.TestCase):
    def test_projection_respects_calibrated_distortion(self):
        g=SimpleNamespace(R=np.diag([1.,-1.,-1.]),t=np.array([0.,0.,1.]),ground=0.,K=np.array([[100.,0,320],[0,100,240],[0,0,1.]]),use_pinhole=False,D=np.array([.1,0,0,0,0]))
        u,v=project(g,.5,-.1)
        self.assertAlmostEqual(u,371.3);self.assertAlmostEqual(v,250.26)
    def test_filters_unsent_and_uses_wire_coordinates(self):
        g=SimpleNamespace(ready=True,R=np.diag([1.,-1.,-1.]),t=np.array([0.,0.,1.]),ground=0.,K=np.array([[100.,0,320],[0,100,240],[0,0,1.]]))
        result=dict(objects=[dict(type=2,metric=dict(x_mm=500,y_mm=-100,yaw_world_deg=23.,pose_valid=False,yaw_source='best_effort')),dict(type=0,metric=dict(x_mm=100,y_mm=0))])
        packet=uart.encode([dict(type=2,x_mm=500,y_mm=-100)])
        targets=sent_targets(packet,result,g)
        self.assertEqual(len(targets),1);self.assertEqual(targets[0]['yaw_cdeg'],0)
        self.assertEqual(targets[0]['estimated_yaw_deg'],23.)
        self.assertEqual(targets[0]['ground_px'],[370.,250.])
        self.assertEqual(sent_targets(uart.encode([]),result,g),[])
    def test_only_successful_physical_write_has_a_frame(self):
        m=LiveMonitor(port=0,hz=20)
        try:
            packet=uart.encode([]);m.on_packet(packet)
            result=dict(sequence_local=17,objects=[],transport=dict(kind='disabled',bytes=0,write_ms=0))
            m.submit(np.zeros((480,640,3),np.uint8),result,packet,None)
            self.assertIsNone(m.pending)
            result['transport']=dict(kind='physical_uart',bytes=len(packet),write_ms=.1)
            m.submit(np.zeros((480,640,3),np.uint8),result,packet,None)
            end=time.monotonic()+2
            while m.frame is None and time.monotonic()<end:time.sleep(.01)
            self.assertIsNotNone(m.frame)
            state=json.load(urllib.request.urlopen('http://127.0.0.1:%d/state'%m.server.server_port))
            self.assertEqual(state['frame']['sequence'],17)
            self.assertEqual(state['frame']['packet_hex'],packet.hex())
            self.assertEqual(state['frame']['targets'],[])
            m.on_packet(packet);self.assertGreater(m.last_tx['id'],m.frame['tx_id'])
        finally:m.close()

if __name__=='__main__':unittest.main()
