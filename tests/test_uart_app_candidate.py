"""Application integration contracts usable without Linux PTY."""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from rescuevision import uart

class AppIntegration(unittest.TestCase):
    def test_model_initialization_has_status_and_recording_failure_is_visible(self):
        spec=importlib.util.spec_from_file_location('guard_app',ROOT/'app.py')
        app=importlib.util.module_from_spec(spec);spec.loader.exec_module(app)
        rows=[];heartbeat=threading.Event();closed=[];calls=[]
        class Sink:
            kind='memory_test_not_physical_uart'
            def write(self,packet):
                rows.append(packet)
                if len(rows)>=2:heartbeat.set()
                return dict(kind=self.kind)
            def close(self):closed.append(True)
        class Pipeline:
            def __init__(self,*args):
                assert rows==[uart.encode([])],'initial status must precede model construction'
                assert heartbeat.wait(2.),'status stopped during model initialization'
                self.geometry=SimpleNamespace(config={'image_size':[640,480]})
            def process(self,image):
                return dict(objects=[],counts={},timings_ms={'compute':0.},uart_hex=uart.encode([]).hex(),uart_fault=False,uart_fault_reasons=[])
            def close(self):pass
        class Camera:
            def __init__(self,*args):pass
            def get(self,*args):return 1,np.zeros((480,640,3),np.uint8),time.perf_counter()
            def close(self):pass
        class Recorder:
            def __init__(self,*args):pass
            def record(self,packet):
                calls.append(packet)
                if len(calls)>1:raise OSError('injected recording disk failure')
            def close(self):pass
        with tempfile.TemporaryDirectory() as directory:
            args=['app.py','--camera','astra','--output',directory,'--max-frames','1','--quiet','--uart-loopback']
            with patch.object(sys,'argv',args),patch.object(app,'LoopbackSink',Sink),patch.object(app,'Pipeline',Pipeline),patch.object(app,'LatestCamera',Camera),patch.object(app,'PacketRecorder',Recorder),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(app.main(),2)
            records=[json.loads(line) for line in (Path(directory)/'frames.jsonl').read_text().splitlines()]
        self.assertEqual(rows[-1],uart.encode([]),'healthy transport must still send the fresh result')
        self.assertEqual(len(calls),2)
        self.assertEqual(closed,[True])
        summary=[r for r in records if r.get('record_kind')=='uart_link_summary'][0]
        self.assertEqual(summary['recording_error'],'injected recording disk failure')
        self.assertIsNone(summary['transport_error'])
        self.assertFalse(any(t.name=='uart-progress-status' for t in threading.enumerate()))

if __name__=='__main__':unittest.main(verbosity=2)
