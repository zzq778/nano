"""Candidate-only tests; no physical transport claim."""
import importlib.util
import json
import sys
import threading
import time
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from rescuevision import uart,continuous_uart as module

class MemorySink:
    kind='memory_test'
    def __init__(self):self.rows=[];self.active=False;self.closed=False;self.fail=False
    def write(self,packet):
        assert not self.active,'overlapping physical writes'
        assert not self.closed
        self.active=True
        try:
            time.sleep(.002)
            if self.fail:raise OSError('injected transport failure')
            self.rows.append((time.monotonic(),packet))
            return dict(kind=self.kind)
        finally:self.active=False
    def close(self):self.closed=True

class Contracts(unittest.TestCase):
    def test_diagnostic_disk_failure_does_not_stop_working_uart(self):
        sink=MemorySink();calls=[]
        def recorder(packet):
            calls.append(packet)
            if len(calls)>1:raise OSError('snapshot disk failure')
        link=module.ContinuousSink(sink,.02,on_packet=recorder)
        try:
            deadline=time.monotonic()+.5
            while len(sink.rows)<4 and time.monotonic()<deadline:time.sleep(.005)
            self.assertTrue(link._worker.is_alive())
            self.assertEqual(link.recording_error,'snapshot disk failure')
            self.assertEqual(len(calls),2)
            self.assertGreaterEqual(len(sink.rows),4)
            link.write(uart.encode([]))
        finally:link.close()

    def test_recorder_matches_actual_order_including_idle_status(self):
        sink=MemorySink();recorded=[]
        link=module.ContinuousSink(sink,.02,recorded.append)
        try:
            link.write(uart.encode([]));time.sleep(.07)
        finally:link.close()
        self.assertEqual(recorded,[packet for _,packet in sink.rows])
        self.assertEqual(recorded[-1],uart.encode([]))

    def test_initialization_and_compute_stalls_keep_status_without_target_replay(self):
        sink=MemorySink();link=module.ContinuousSink(sink,.04)
        try:
            time.sleep(.14)  # model/camera initialization simulation
            target=uart.encode([dict(type=0,x_mm=100,y_mm=0,yaw_deg=0)])
            link.write(target)
            time.sleep(.14)  # computation stalls
        finally:link.close()
        self.assertGreaterEqual(len(sink.rows),6)
        self.assertEqual(sum(p==target for _,p in sink.rows),1)
        self.assertTrue(all(p in (target,uart.encode([])) for _,p in sink.rows))
        self.assertLess(max(b[0]-a[0] for a,b in zip(sink.rows,sink.rows[1:])),.13)
        count=len(sink.rows);time.sleep(.06);self.assertEqual(len(sink.rows),count)
        self.assertFalse(link._worker.is_alive())

    def test_result_and_status_writes_are_serialized(self):
        sink=MemorySink();link=module.ContinuousSink(sink,.005)
        try:
            for _ in range(25):link.write(uart.encode([]));time.sleep(.002)
        finally:link.close()
        self.assertTrue(sink.closed)

    def test_status_write_failure_is_exposed_to_main_loop(self):
        sink=MemorySink();link=module.ContinuousSink(sink,.02)
        try:
            sink.fail=True
            link._worker.join(.5)
            self.assertFalse(link._worker.is_alive())
            with self.assertRaisesRegex(OSError,'transport failure'):link.write(uart.encode([]))
        finally:
            with self.assertRaisesRegex(OSError,'transport failure'):link.close()
        self.assertTrue(sink.closed)

    def test_late_worker_error_is_reported_on_close_without_another_write(self):
        sink=MemorySink();link=module.ContinuousSink(sink,.02)
        sink.fail=True;link._worker.join(.5)
        with self.assertRaisesRegex(OSError,'transport failure'):link.close()
        self.assertTrue(sink.closed)
        self.assertEqual(link.diagnostics()['transport_error'],'injected transport failure')

if __name__=='__main__':unittest.main(verbosity=2)
