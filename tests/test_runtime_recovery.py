import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from rescuevision import uart


class FakePipeline:
    def __init__(self, *args):
        self.geometry = SimpleNamespace(config={'image_size':[640,480]})
        self.calls = 0

    def process(self, image):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError('injected isolated frame error')
        return dict(objects=[],counts={},timings_ms={'compute':0.},
                    uart_hex=uart.encode([]).hex(),uart_fault=False,uart_fault_reasons=[])

    def close(self):
        pass


class RecoveringCamera:
    calls = 0

    def __init__(self, *args):
        pass

    def get(self, *args):
        RecoveringCamera.calls += 1
        if RecoveringCamera.calls == 1:
            raise TimeoutError('injected camera gap')
        return RecoveringCamera.calls, np.zeros((480,640,3),np.uint8), time.perf_counter()

    def close(self):
        pass


class RuntimeRecoveryContracts(unittest.TestCase):
    def test_camera_gap_and_one_bad_frame_do_not_end_the_next_frame(self):
        spec=importlib.util.spec_from_file_location('runtime_under_test',ROOT/'app.py')
        app=importlib.util.module_from_spec(spec);spec.loader.exec_module(app)
        RecoveringCamera.calls=0
        with tempfile.TemporaryDirectory() as directory:
            args=['app.py','--camera','astra','--output',directory,'--max-frames','2','--quiet']
            with patch.object(sys,'argv',args),patch.object(app,'Pipeline',FakePipeline),patch.object(app,'LatestCamera',RecoveringCamera):
                self.assertEqual(app.main(),0)
            records=[json.loads(l) for l in (Path(directory)/'frames.jsonl').read_text().splitlines()]
            self.assertEqual(sum(r.get('record_kind')=='source_frame_drop' for r in records),1)
            frames=[r for r in records if 'objects' in r]
            self.assertEqual(len(frames),2)
            self.assertEqual(uart.decode(bytes.fromhex(frames[0]['uart_hex'])),[])
            self.assertIn('frame_processing_failed',frames[0]['diagnostics'])
            self.assertFalse(frames[1]['uart_fault'])
            self.assertEqual(json.loads((Path(directory)/'run_summary.json').read_text())['frames'],2)

    @unittest.skipUnless(os.name=='posix','Actual same-port PTY recovery runs on Nano')
    def test_frame_failures_keep_one_port_and_allow_next_valid_packet(self):
        spec=importlib.util.spec_from_file_location('uart_recovery_under_test',ROOT/'app.py')
        app=importlib.util.module_from_spec(spec);spec.loader.exec_module(app)
        actual_sink=app.LoopbackSink
        packets=[];opened=[]
        class TracedSink(actual_sink):
            def __init__(self):
                super().__init__();opened.append(self.fd)
            def write(self,packet):
                receipt=super().write(packet)
                packets.append((self.fd,packet,receipt))
                return receipt
        RecoveringCamera.calls=0
        with tempfile.TemporaryDirectory() as directory:
            args=['app.py','--camera','astra','--output',directory,'--max-frames','2','--quiet','--uart-loopback']
            with patch.object(sys,'argv',args),patch.object(app,'Pipeline',FakePipeline),patch.object(app,'LatestCamera',RecoveringCamera),patch.object(app,'LoopbackSink',TracedSink):
                self.assertEqual(app.main(),0)
        self.assertEqual(len(opened),1)
        self.assertEqual([p for _,p,_ in packets],[uart.encode([]),uart.encode([]),uart.encode([]),uart.encode([])])
        self.assertEqual({fd for fd,_,_ in packets},set(opened))
        self.assertTrue(all(receipt['loopback_verified'] for _,_,receipt in packets))

    @unittest.skipUnless(os.name=='posix','Over-budget target PTY integration runs on Nano')
    def test_continuously_old_valid_targets_use_one_open_uart_port(self):
        spec=importlib.util.spec_from_file_location('age_budget_under_test',ROOT/'app.py')
        app=importlib.util.module_from_spec(spec);spec.loader.exec_module(app)
        expected=uart.encode([dict(type=0,x_mm=450,y_mm=-100)])
        class Pipeline:
            def __init__(self,*args):self.geometry=SimpleNamespace(config={'image_size':[640,480]})
            def process(self,image):
                return dict(objects=[],counts={},timings_ms={'compute':0.},uart_hex=expected.hex(),uart_fault=False,uart_fault_reasons=[])
            def close(self):pass
        class Camera:
            def __init__(self,*args):self.sequence=0
            def get(self,*args):
                self.sequence+=1
                return self.sequence,np.zeros((480,640,3),np.uint8),time.perf_counter()-.35
            def close(self):pass
        actual_sink=app.LoopbackSink;opened=[];received=[]
        class Sink(actual_sink):
            def __init__(self):super().__init__();opened.append(self.fd)
            def write(self,packet):
                receipt=super().write(packet);received.append((self.fd,packet,receipt));return receipt
        with tempfile.TemporaryDirectory() as directory:
            args=['app.py','--camera','astra','--output',directory,'--max-frames','4','--quiet','--uart-loopback']
            with patch.object(sys,'argv',args),patch.object(app,'Pipeline',Pipeline),patch.object(app,'LatestCamera',Camera),patch.object(app,'LoopbackSink',Sink):
                self.assertEqual(app.main(),0)
            records=[json.loads(l) for l in (Path(directory)/'frames.jsonl').read_text().splitlines()]
            frames=[r for r in records if 'objects' in r]
            self.assertEqual(len(frames),4)
            self.assertTrue(all(r['result_age_over_budget'] and not r['uart_fault'] for r in frames))
        self.assertEqual(len(opened),1)
        self.assertEqual({fd for fd,_,_ in received},set(opened))
        self.assertEqual([p for _,p,_ in received],[uart.encode([])]+[expected]*4)
        self.assertTrue(all(r['loopback_verified'] for _,_,r in received))


if __name__ == '__main__':
    unittest.main()
