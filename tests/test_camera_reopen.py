import contextlib,io,json,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import test_astra_capture as fixture

class CameraReopen(unittest.TestCase):
    def exercise(self,startup_failures=0,stalled=False):
        app=fixture.application();cameras=[];models=[];ports=[];packets=[]
        class Camera:
            def __init__(self,*args):
                self.index=len(cameras);self.calls=0;self.failed=SimpleNamespace(value=self.index<startup_failures)
                self.closes=0;cameras.append(self)
            def get(self,after):
                self.calls+=1
                if self.failed.value:raise RuntimeError('injected camera absent at boot')
                if startup_failures==0 and self.index==0:
                    if self.calls==1:return 100,np.zeros((480,640,3),np.uint8),time.perf_counter()
                    if stalled:raise TimeoutError('injected live owner without new frames')
                    self.failed.value=True;raise RuntimeError('injected capture owner exit')
                if after!=0:raise AssertionError('new camera inherited previous sequence')
                return 1,np.zeros((480,640,3),np.uint8),time.perf_counter()
            def close(self):self.closes+=1
        class Model(fixture.GoodPipeline):
            def __init__(self,*args):super().__init__(*args);models.append(self)
        class Port:
            kind='memory_test'
            def __init__(self):self.closed=False;ports.append(self)
            def write(self,packet):packets.append(packet);return dict(kind=self.kind)
            def close(self):self.closed=True
        with tempfile.TemporaryDirectory() as directory:
            count=1 if startup_failures else 2
            args=['app.py','--camera','astra','--max-frames',str(count),'--quiet','--uart-loopback','--output',directory]
            with patch.object(sys,'argv',args),patch.object(app,'LatestCamera',Camera),patch.object(app,'Pipeline',Model),patch.object(app,'LoopbackSink',Port),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(app.main(),0)
            rows=[json.loads(line) for line in (Path(directory)/'frames.jsonl').read_text().splitlines()]
        self.assertEqual(len(models),1);self.assertEqual(len(ports),1);self.assertTrue(ports[0].closed)
        self.assertTrue(all(camera.closes==1 for camera in cameras))
        return rows,cameras
    def test_owner_death_reopens_without_reloading_model_or_reopening_uart(self):
        rows,cameras=self.exercise()
        self.assertEqual(len(cameras),2)
        self.assertEqual([row['sequence_local'] for row in rows if 'objects' in row],[100,1])
    def test_live_but_stalled_owner_reopens_without_reloading_model_or_uart(self):
        rows,cameras=self.exercise(stalled=True)
        self.assertEqual(len(cameras),2)
        self.assertEqual(sum(row.get('record_kind')=='source_frame_drop' for row in rows),3)
        self.assertEqual([row['sequence_local'] for row in rows if 'objects' in row],[100,1])

    def test_late_camera_is_retried_until_available(self):
        rows,cameras=self.exercise(startup_failures=3)
        self.assertEqual(len(cameras),4)
        self.assertEqual(sum(row.get('record_kind')=='camera_reopened' for row in rows),3)

if __name__=='__main__':unittest.main()
