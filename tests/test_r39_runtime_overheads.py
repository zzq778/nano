"""The optional observers must have separately measured, bounded costs."""
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
from rescuevision import live_monitor, uart


class RuntimeOverheadContracts(unittest.TestCase):
    def test_blocking_optional_observer_is_reported_without_faking_uart_delay(self):
        app = fixture.application()

        class Camera:
            def __init__(self, *args):
                self.sequence = 0
            def get(self, *args):
                self.sequence += 1
                return self.sequence, np.zeros((480, 640, 3), np.uint8), time.perf_counter()
            def close(self):
                pass

        class Sink:
            def __init__(self, *args):
                pass
            kind = 'physical_uart'
            recording_error = None
            def write(self, packet):
                self.last = packet
                return dict(kind=self.kind, bytes=len(packet), write_ms=0.05)
            def close(self):
                pass

        class Monitor:
            def __init__(self, *args, **kwargs):
                self.packets = []
            def on_packet(self, packet):
                self.packets.append(packet)
            def submit(self, image, result, packet, geometry):
                time.sleep(0.002)
                assert packet == uart.encode([])
            def close(self):
                pass

        class Sampler:
            def __init__(self, *args):
                pass
            def maybe_save(self, *args):
                time.sleep(0.003)
                return None
            def close(self):
                return {'record_kind': 'sample_writer_summary', 'accepted': 0, 'saved': 0}

        with tempfile.TemporaryDirectory() as directory:
            args = ['app.py', '--camera', 'astra', '--output', directory,
                    '--max-frames', '4', '--quiet', '--no-frame-log',
                    '--uart-device', '/dev/test-fake-uart', '--sample-interval', '1']
            with patch.object(sys, 'argv', args), patch.object(app, 'LatestCamera', Camera), \
                 patch.object(app, 'Pipeline', fixture.GoodPipeline), \
                 patch.object(app, 'SerialSink', Sink), \
                 patch.object(live_monitor, 'LiveMonitor', Monitor), \
                 patch.object(app, 'AsyncSampleRecorder', Sampler), \
                 contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(app.main(), 0)
            summary = json.loads((Path(directory) / 'run_summary.json').read_text())
            stages = summary['latency_stage_ms']
            self.assertEqual(stages['monitor_submit']['samples'], 4)
            self.assertGreaterEqual(stages['monitor_submit']['p50_ms'], 2)
            self.assertEqual(stages['sample_submit']['samples'], 4)
            self.assertGreaterEqual(stages['sample_submit']['p50_ms'], 3)
            self.assertIsNone(stages['frame_json_log'])
            self.assertEqual(summary['application_results_written'], 4)
            self.assertLess(summary['latency_ms']['app_write_result_age']['p95_ms'], 5)
            self.assertTrue(all('objects' not in json.loads(row) for row in
                                (Path(directory) / 'frames.jsonl').read_text().splitlines()))

    def test_disabled_observers_are_unmeasured_not_zero(self):
        app = fixture.application()
        class Camera:
            def __init__(self, *args):
                pass
            def get(self, *args):
                return 1, np.zeros((480,640,3), np.uint8), time.perf_counter()
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, 'argv', ['app.py', '--camera', 'astra', '--output', directory,
                                            '--max-frames', '1', '--quiet', '--no-frame-log']), \
                 patch.object(app, 'LatestCamera', Camera), \
                 patch.object(app, 'Pipeline', fixture.GoodPipeline), \
                 contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(app.main(), 0)
            summary = json.loads((Path(directory) / 'run_summary.json').read_text())
            for name in ('monitor_submit', 'sample_submit', 'frame_json_log'):
                self.assertIsNone(summary['latency_stage_ms'][name])


if __name__ == '__main__':
    unittest.main()
