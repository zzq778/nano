import sys,threading,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from rescuevision.ros_image import RosImagePublisher

class RosImageContracts(unittest.TestCase):
    def ros(self,connections=1):
        rows=[];ready=threading.Event();sent=threading.Event();released=[]
        class Publisher:
            def get_num_connections(self):ready.set();return connections
            def publish(self,message):rows.append(message);sent.set()
            def unregister(self):released.append(True)
        class Message:
            def __init__(self):self.header=SimpleNamespace()
        ros=SimpleNamespace(init_node=lambda *a,**k:None,Publisher=lambda *a,**k:Publisher(),
            is_shutdown=lambda:False,signal_shutdown=lambda reason:None,Time=SimpleNamespace(from_sec=lambda x:x))
        return ros,Message,rows,ready,sent,released

    def test_latest_frame_bytes_and_bgr_metadata_are_published(self):
        ros,msg,rows,ready,sent,released=self.ros()
        with patch('rescuevision.ros_image._load_ros',return_value=(ros,msg)):
            p=RosImagePublisher('/test/image',100.)
            try:
                self.assertTrue(ready.wait(1.))
                image=np.zeros((2,3,3),np.uint8);image[0,0]=[3,7,11]
                p.submit(image)
                self.assertTrue(sent.wait(1.))
                self.assertEqual(rows[0].data,image.tobytes())
                self.assertEqual((rows[0].height,rows[0].width,rows[0].step,rows[0].encoding),(2,3,9,'bgr8'))
                self.assertEqual(rows[0].header.frame_id,'astra_rgb_optical_frame')
            finally:p.close()
        self.assertEqual(released,[True]);self.assertFalse(p._worker.is_alive())

    def test_no_subscribers_does_not_serialize_image(self):
        ros,msg,rows,ready,sent,released=self.ros(0)
        class Image:
            def tobytes(self):raise AssertionError('must not serialize without viewers')
        with patch('rescuevision.ros_image._load_ros',return_value=(ros,msg)):
            p=RosImagePublisher('/test/image',100.)
            try:
                p.submit(Image());self.assertTrue(ready.wait(1.));time.sleep(.03)
                self.assertEqual(rows,[]);self.assertIsNone(p.error)
            finally:p.close()

    def test_missing_ros_does_not_raise_into_capture_or_uart(self):
        with patch('rescuevision.ros_image._load_ros',side_effect=ImportError('injected missing ROS')):
            p=RosImagePublisher('/test/image')
            p._worker.join(1.)
            p.submit(np.zeros((2,2,3),np.uint8));p.close()
            self.assertIn('injected missing ROS',p.diagnostics()['error'])

if __name__=='__main__':unittest.main()
