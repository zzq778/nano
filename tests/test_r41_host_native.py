"""Host-compiled RGB fitter smoke test; no OpenCV C++ SDK or Jetson needed.

A full pre/post native A/B is in tools/bench_r41_native.py and requires an
independent copy of the r40 source tree. This self-contained test instead
checks physical pose recovery using 80 freshly rendered model scenes.
"""
import importlib.util
import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools' / 'bench_r41_native.py'


@unittest.skipUnless(shutil.which('g++'), 'host C++ compiler unavailable')
class NativeModelProbe(unittest.TestCase):
    def test_model_fitter_recovers_rendered_position_and_is_deterministic(self):
        spec = importlib.util.spec_from_file_location('r41_probe', str(SCRIPT))
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        with tempfile.TemporaryDirectory(prefix='r41_probe_test_') as directory:
            temporary = Path(directory)
            manifest = probe.prepare(ROOT, temporary, synthetic_only=True)
            executable = temporary / 'model_probe'
            probe.build(ROOT, probe.PROBE, executable)
            initial = probe.run(executable, manifest)
            repeated = probe.run(executable, manifest)
            probe.compare(initial, repeated)
            self.assertEqual(len(initial), 80)
            self.assertTrue(all(row[1] == 1 for row in initial))
            self.assertGreaterEqual(sum(row[2] in (1, 2) for row in initial), 70)
            xs = (.34, .42, .55, .65, .80)
            ys = (-.10, 0., .12, -.08, .04)
            for index, row in enumerate(initial):
                self.assertGreater(row[4], 0)
                self.assertTrue(all(math.isfinite(v) for v in row[2:-1]))
                if row[2] in (1, 2):
                    position_index = (index // 4) % 5
                    difference = math.hypot(row[5]-xs[position_index],
                                            row[6]-ys[position_index])
                    # Reconstructed 3D position, not merely bbox-center proximity.
                    self.assertLess(difference, .020)


if __name__ == '__main__':
    unittest.main()
