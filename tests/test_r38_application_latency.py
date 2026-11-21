"""Verify production no-frame-log preserves bounded delivery timing evidence."""
import contextlib
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import test_astra_capture as fixture


class DeliveryTimingContracts(unittest.TestCase):
    def test_summary_reports_valid_sample_counts_without_frame_json(self):
        app=fixture.application()

        class Camera:
            def __init__(self,*args):self.sequence=0
            def get(self,*args):
                self.sequence+=1
                return self.sequence,np.zeros((480,640,3),np.uint8),time.perf_counter()-.012
            def close(self):pass

        class Sink:
            kind='timing_test_not_real_uart'
            def write(self,packet):return dict(kind=self.kind,bytes=len(packet),write_ms=.125)
            def close(self):pass

        with tempfile.TemporaryDirectory() as directory:
            args=['app.py','--camera','astra','--output',directory,'--max-frames','5',
                  '--quiet','--no-frame-log','--uart-loopback']
            with patch.object(sys,'argv',args),patch.object(app,'LatestCamera',Camera),\
                 patch.object(app,'Pipeline',fixture.GoodPipeline),patch.object(app,'LoopbackSink',Sink),\
                 contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(app.main(),0)
            summary=json.loads((Path(directory)/'run_summary.json').read_text())
            self.assertEqual(summary['frames'],5)
            self.assertEqual(summary['application_results_written'],5)
            self.assertEqual(summary['latency_ms']['pipeline_compute']['samples'],5)
            self.assertEqual(summary['latency_ms']['app_serial_write']['p50_ms'],.125)
            self.assertGreater(summary['latency_ms']['app_write_result_age']['p95_ms'],10)
            self.assertEqual(summary['latency_ms']['app_write_result_age']['samples'],5)
            self.assertIsNone(summary['latency_stage_ms']['native_tensorrt_and_transfers'])
            self.assertEqual(summary['camera_sequences_skipped_during_processing'],0)
            self.assertFalse(any('objects' in json.loads(line) for line in
                                 (Path(directory)/'frames.jsonl').read_text().splitlines()))

    def test_disabled_serial_does_not_report_fake_wire_write_latency(self):
        app=fixture.application()

        class Camera:
            def __init__(self,*args):pass
            def get(self,*args):return 1,np.zeros((480,640,3),np.uint8),time.perf_counter()
            def close(self):pass

        with tempfile.TemporaryDirectory() as directory:
            args=['app.py','--camera','astra','--output',directory,'--max-frames','1','--quiet','--no-frame-log']
            with patch.object(sys,'argv',args),patch.object(app,'LatestCamera',Camera),\
                 patch.object(app,'Pipeline',fixture.GoodPipeline),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(app.main(),0)
            summary=json.loads((Path(directory)/'run_summary.json').read_text())
            self.assertEqual(summary['application_results_written'],0)
            self.assertIsNone(summary['latency_ms']['app_serial_write'])
            self.assertIsNone(summary['latency_ms']['app_write_result_age'])


if __name__=='__main__':unittest.main()
