import importlib.util
import contextlib
import io
import json
import multiprocessing
import os
import queue
import signal
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
from rescuevision import runtime,uart


def application():
    spec=importlib.util.spec_from_file_location('astra_app_under_test',ROOT/'app.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def abandon_frame_lock(lock):
    lock.acquire()  # Simulate a capture process dying in its publication section.


class GoodPipeline:
    def __init__(self,*args):self.geometry=SimpleNamespace(config={'image_size':[640,480]})
    def process(self,image):
        return dict(objects=[],counts={},timings_ms={'compute':0.},uart_hex=uart.encode([]).hex(),uart_fault=False,uart_fault_reasons=[])
    def close(self):pass


class AstraCaptureContracts(unittest.TestCase):
    def test_termination_signal_uses_the_normal_resource_cleanup(self):
        app=application();closed=[];previous=signal.getsignal(signal.SIGTERM)
        class Camera:
            def __init__(self,*args):self.calls=0
            def get(self,*args):
                self.calls+=1
                if self.calls>1:raise KeyboardInterrupt('test stopped: handler missing')
                handler=signal.getsignal(signal.SIGTERM)
                if not callable(handler):raise RuntimeError('termination handler not installed')
                handler(signal.SIGTERM,None)
            def close(self):closed.append(True)
        with tempfile.TemporaryDirectory() as directory:
            args=['app.py','--camera','astra','--output',directory,'--max-frames','1','--quiet']
            with patch.object(sys,'argv',args),patch.object(app,'Pipeline',GoodPipeline),patch.object(app,'LatestCamera',Camera),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(app.main(),2)
            rows=[json.loads(line) for line in (Path(directory)/'frames.jsonl').read_text().splitlines()]
            self.assertIn('termination signal',rows[-1]['error'])
        self.assertEqual(closed,[True])
        self.assertEqual(signal.getsignal(signal.SIGTERM),previous)

    def test_capture_stops_when_its_parent_has_exited(self):
        stop=SimpleNamespace(value=False);ready=SimpleNamespace(value=False);failed=SimpleNamespace(value=False);closed=[]
        class Camera:
            def __init__(self,*args):pass
            def read(self):raise AssertionError('read continued after parent exit')
            def close(self):closed.append(True)
        with patch.object(runtime,'NativeAstra',Camera),patch.object(runtime.os,'getppid',side_effect=[123,456]):
            runtime._camera_reader(dict(uri='astra',library='unused'),640,480,bytearray(640*480*3),SimpleNamespace(value=0),
                                   SimpleNamespace(value=0.),threading.Lock(),stop,ready,failed,queue.Queue(1))
        self.assertFalse(failed.value)
        self.assertEqual(closed,[True])

    def test_capture_owner_death_cannot_hold_the_consumer_forever(self):
        context=multiprocessing.get_context('spawn');lock=context.Lock()
        publisher=context.Process(target=abandon_frame_lock,args=(lock,))
        publisher.start();publisher.join(5.)
        self.assertFalse(publisher.is_alive())
        camera=runtime.LatestCamera.__new__(runtime.LatestCamera)
        camera.frame_lock=lock
        camera.sequence=SimpleNamespace(value=0);camera._failure=None
        camera.failed=SimpleNamespace(value=True)
        camera.process=publisher;camera.errors=queue.Queue(1);camera.errors.put('publication owner exited')
        done=threading.Event();caught=[]
        def consume():
            try:camera.get(0,timeout=.05)
            except Exception as error:caught.append(error)
            finally:done.set()
        consumer=threading.Thread(target=consume,daemon=True);consumer.start()
        completed=done.wait(.3)
        if not completed:
            # Release only the test-owned semaphore so a red test leaves no thread behind.
            lock.release();consumer.join(1.)
        self.assertTrue(completed,'consumer exceeded its deadline after publication owner exited')
        self.assertEqual(len(caught),1)
        self.assertIsInstance(caught[0],RuntimeError)

    def test_explicit_camera_warmup_discards_its_target_and_uses_a_fresh_frame(self):
        app=application()
        class Camera:
            def __init__(self,*args):self.sequence=0
            def get(self,*args):
                self.sequence+=1
                return self.sequence,np.zeros((480,640,3),np.uint8),time.perf_counter()
            def close(self):pass
        with tempfile.TemporaryDirectory() as directory:
            args=['app.py','--camera','astra','--warmup','1','--output',directory,'--max-frames','1','--quiet']
            with patch.object(sys,'argv',args),patch.object(app,'Pipeline',GoodPipeline),patch.object(app,'LatestCamera',Camera):
                self.assertEqual(app.main(),0)
            records=[json.loads(line) for line in (Path(directory)/'frames.jsonl').read_text().splitlines()]
            warmup=[r for r in records if r.get('record_kind')=='camera_warmup']
            frames=[r for r in records if 'objects' in r]
            self.assertEqual(len(warmup),1)
            self.assertEqual(warmup[0]['uart_hex'],uart.encode([]).hex())
            self.assertEqual(frames[0]['sequence_local'],2)

    def test_usb_numeric_source_is_rejected_in_the_astra_engineering_entry(self):
        app=application()
        class Camera:
            def __init__(self,*args):pass
            def get(self,*args):return 1,np.zeros((480,640,3),np.uint8),time.perf_counter()
            def close(self):pass
        with tempfile.TemporaryDirectory() as directory:
            args=['app.py','--camera','0','--output',directory,'--max-frames','1','--quiet']
            with patch.object(sys,'argv',args),patch.object(app,'Pipeline',GoodPipeline),patch.object(app,'LatestCamera',Camera),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:app.main()
            self.assertEqual(raised.exception.code,2)

    def test_sdk_wait_timeout_does_not_prevent_the_next_rgb_frame(self):
        stop=SimpleNamespace(value=False);ready=SimpleNamespace(value=False);failed=SimpleNamespace(value=False);closed=[]
        expected=np.full((480,640,3),(7,91,203),np.uint8)
        class Camera:
            def __init__(self,*args):self.calls=0
            def read(self):
                self.calls+=1
                if self.calls==1:raise TimeoutError('no frame yet')
                stop.value=True;return expected
            def close(self):closed.append(True)
        buffer=bytearray(expected.nbytes);sequence=SimpleNamespace(value=0);stamp=SimpleNamespace(value=0.)
        with patch.object(runtime,'NativeAstra',Camera,create=True),patch('cv2.VideoCapture',side_effect=AssertionError('discarded USB camera path used')):
            runtime._camera_reader(dict(uri='astra',library='unused_test_library'),640,480,buffer,sequence,stamp,threading.Condition(),stop,ready,failed,queue.Queue(1))
        self.assertFalse(failed.value)
        self.assertTrue(ready.value)
        self.assertEqual(sequence.value,1)
        np.testing.assert_array_equal(np.frombuffer(buffer,np.uint8).reshape(expected.shape),expected)
        self.assertEqual(closed,[True])

    def test_terminal_capture_reports_are_bounded_instead_of_flooding_uart(self):
        camera=runtime.LatestCamera.__new__(runtime.LatestCamera)
        camera.sequence=SimpleNamespace(value=0);camera.frame_lock=threading.Lock()
        camera.failed=SimpleNamespace(value=True)
        camera.process=SimpleNamespace(is_alive=lambda:False)
        camera.errors=queue.Queue(1);camera.errors.put('injected source exit')
        camera._failure=None
        started=time.perf_counter()
        with self.assertRaises(RuntimeError):camera.get(0,timeout=.15)
        self.assertGreaterEqual(time.perf_counter()-started,.10)

    def test_source_runtime_error_preserves_the_following_frame(self):
        app=application()
        class Camera:
            def __init__(self,*args):self.calls=0
            def get(self,*args):
                self.calls+=1
                if self.calls==1:raise RuntimeError('injected SDK process exit')
                return self.calls,np.zeros((480,640,3),np.uint8),time.perf_counter()
            def close(self):pass
        with tempfile.TemporaryDirectory() as directory:
            args=['app.py','--camera','astra','--output',directory,'--max-frames','1','--quiet']
            with patch.object(sys,'argv',args),patch.object(app,'Pipeline',GoodPipeline),patch.object(app,'LatestCamera',Camera):
                self.assertEqual(app.main(),0)
            records=[json.loads(line) for line in (Path(directory)/'frames.jsonl').read_text().splitlines()]
            self.assertEqual(sum(r.get('record_kind')=='source_frame_drop' for r in records),1)
            self.assertEqual(sum('objects' in r for r in records),1)

    @unittest.skipUnless(os.name=='posix','PTY lifetime verified on Nano')
    def test_source_gaps_and_reopen_keep_one_uart_with_empty_packets(self):
        app=application();actual_sink=app.LoopbackSink;opened=[];packets=[]
        class Camera:
            calls=0
            def __init__(self,*args):self.failed=SimpleNamespace(value=False)
            def get(self,*args):
                Camera.calls+=1
                if Camera.calls<=3:raise RuntimeError('injected source failure')
                return Camera.calls,np.zeros((480,640,3),np.uint8),time.perf_counter()
            def close(self):pass
        class Sink(actual_sink):
            def __init__(self):super().__init__();opened.append(self.fd)
            def write(self,packet):
                receipt=super().write(packet);packets.append((self.fd,packet,receipt));return receipt
        with tempfile.TemporaryDirectory() as directory:
            args=['app.py','--camera','astra','--output',directory,'--max-frames','1','--quiet','--uart-loopback']
            with patch.object(sys,'argv',args),patch.object(app,'Pipeline',GoodPipeline),patch.object(app,'LatestCamera',Camera),patch.object(app,'LoopbackSink',Sink):
                self.assertEqual(app.main(),0)
        self.assertEqual(len(opened),1)
        self.assertEqual({fd for fd,_,_ in packets},set(opened))
        self.assertEqual([packet for _,packet,_ in packets],[uart.encode([])]*4+[uart.encode([])])
        self.assertTrue(all(receipt['loopback_verified'] for _,_,receipt in packets))


if __name__=='__main__':unittest.main()
