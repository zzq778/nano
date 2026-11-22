"""Fast, hardware-free checks of prepared performance and diagnostics harnesses."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def load(name):
    path=ROOT / 'tools' / (name+'.py')
    spec=importlib.util.spec_from_file_location(name,str(path))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeFreePreflightTests(unittest.TestCase):
    def test_compiles_sources_without_writing_bytecode(self):
        module=load('r41_preflight')
        result=module.source_syntax(ROOT)
        self.assertGreater(result['python_files_compiled_without_writes'],35)

    def test_missing_private_replay_data_is_not_misreported(self):
        module=load('r41_preflight')
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(module.has_replay_fixtures(Path(d)))

    def test_rejects_protected_uart_source_change(self):
        module=load('r41_preflight')
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a';b=Path(d)/'b'
            for root in (a,b):
                for entry in module.PROTECTED_SOURCES:
                    p=root/entry;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('original')
            self.assertTrue(all(module.source_immutability(a,b)['protected_sources_unchanged'].values()))
            (a/module.PROTECTED_SOURCES[0]).write_text('changed')
            with self.assertRaisesRegex(AssertionError,'Protected source differs'):
                module.source_immutability(a,b)

    def test_bundled_r40_cpp_reference_matches_locked_sha(self):
        import hashlib
        probe=load('r41_preflight')
        baseline=ROOT/'tests/reference/model_geometry_r40.cpp'
        self.assertTrue(baseline.is_file())
        digest=hashlib.sha256(baseline.read_bytes()).hexdigest()
        self.assertEqual(digest,'de74ef99b87f4b2e081d7c3bc9fff809a48ce81dd840a8f544ec57eb94c8f430')
        self.assertNotEqual(digest,hashlib.sha256((ROOT/'vendor_model/src/model_geometry.cpp').read_bytes()).hexdigest())

    def test_runtime_report_quantiles_grouping_and_incomplete_evidence(self):
        module=load('r41_runtime_report')
        rows=[]
        for i in range(120):
            n=15 if i%2 else 0
            rows.append(json.dumps(dict(record_kind='vision_result',objects=[],counts={'green':n},
                                        timings_ms={'compute':20+i*.01, 'native_metric_prepare':1.2},
                                        dropped_camera_frames=int(i==30),
                                        source_kind='camera_read_completion_not_exposure',
                                        transport={'kind':'physical_uart'})))
        report=module.analyze(rows,dict(observed_loop_hz=5.9,includes_rate_pacing=False),warmup=10)
        self.assertEqual(report['frames_analyzed'],110)
        self.assertEqual(report['stage_latency_ms']['compute']['n'],110)
        self.assertIn('0',report['compute_by_object_count_ms'])
        self.assertIn('15',report['compute_by_object_count_ms'])
        self.assertEqual(report['camera_sequences_skipped_in_analyzed_frames'],1)
        self.assertTrue(report['real_camera_evidence'])
        self.assertTrue(report['physical_uart_evidence'])
        self.assertTrue(any('TensorRT' in w for w in report['warnings']))

    def test_runtime_report_accepts_direct_and_wrapped_yolo_timings(self):
        module=load('r41_runtime_report')
        rows=[]
        for wrapper in (False,True):
            yolo={'timings_ms':{'native_detector':4.0,'hsv_and_blue':1.0},
                  'native_stages_ms':{'preprocess':.8,'tensorrt_and_transfers':7.0,
                                      'postprocess':.5}}
            diagnostics={'yolo_diagnostics':yolo} if wrapper else yolo
            rows.append(json.dumps({'objects':[],'counts':{},'timings_ms':{'compute':13.2},
                                    'frontend_diagnostics':diagnostics}))
        report=module.analyze(rows,warmup=0)
        self.assertEqual(report['stage_latency_ms']['native_detector']['n'],2)
        self.assertEqual(report['stage_latency_ms']['native_tensorrt_and_transfers']['n'],2)
        self.assertFalse(any('TensorRT+transfer timings absent' in w for w in report['warnings']))

    def test_failed_frames_do_not_inflate_successful_performance(self):
        module=load('r41_runtime_report')
        good={'objects':[], 'counts':{'green':15},
              'timings_ms':{'compute':12.0},
              'source_kind':'camera_read_completion_not_exposure'}
        bad={'objects':[], 'counts':None, 'timings_ms':{'compute':0.01},
             'frame_error':{'type':'RuntimeError','message':'synthetic failure'},
             'source_kind':'camera_read_completion_not_exposure'}
        reopen={'record_kind':'camera_reopened', 'source_sequence_reset':True}
        output=module.analyze([json.dumps(item) for item in [good,bad,reopen]],warmup=0)
        self.assertEqual(output['frames_analyzed'],2)
        self.assertEqual(output['successful_frames_analyzed'],1)
        self.assertEqual(output['frame_processing_error_records'],1)
        self.assertEqual(output['camera_reopen_records'],1)
        self.assertEqual(output['stage_latency_ms']['compute']['n'],1)
        self.assertEqual(output['stage_latency_ms']['compute']['p50_ms'],12.0)
        self.assertNotIn('0',output['compute_by_object_count_ms'])
        self.assertTrue(any('Frame processing errors' in item for item in output['warnings']))

    def test_runtime_report_rejects_nonfinite_and_bad_json(self):
        module=load('r41_runtime_report')
        with self.assertRaises(ValueError):
            module.distribution([1.,float('nan')])
        with self.assertRaises(ValueError):
            module.analyze(['not json'])
        with self.assertRaises(ValueError):
            module.analyze(['{}','{}'],limit=1)

    def test_runtime_report_does_not_claim_fps_from_missing_data(self):
        module=load('r41_runtime_report')
        report=module.analyze([],warmup=0)
        self.assertEqual(report['frames_analyzed'],0)
        self.assertIsNone(report['summary_observed_loop_hz'])
        self.assertIsNone(report['wall_with_json_ms'])
        self.assertTrue(report['warnings'])

    def test_color_stats_detects_channel_change_without_modifying_image(self):
        module=load('r41_color_diagnostic')
        original=np.full((32,32,3),[120,120,120],np.uint8)
        suspect=original.copy();suspect[:,:,1]=200
        saved=suspect.copy()
        profile=module.image_profile(original,(4,4,20,20))
        affected=module.image_profile(suspect,(4,4,20,20))
        self.assertAlmostEqual(profile['red_over_green'],1.0)
        self.assertLess(affected['red_over_green'],.75)
        np.testing.assert_array_equal(suspect,saved)
        with self.assertRaises(ValueError):
            module.image_profile(original,(20,20,20,20))

    def test_tegrastats_parser_handles_gpu_memory_temp_and_power(self):
        module=load('r41_tegrastats_report')
        lines=['RAM 1521/3956MB (lfb 121x4MB) CPU [10%@1020,off,20%@918] '
               'GR3D_FREQ 45%@921 GPU@44.0C CPU@43.5C VDD_IN 3400/3290\n']*20
        report=module.analyze(lines,expected_interval_s=1)
        self.assertEqual(report['gpu_utilization_pct']['p95'],45)
        self.assertEqual(report['ram_used_mb']['p50'],1521)
        self.assertEqual(report['temperatures_c']['GPU']['max'],44)
        self.assertEqual(report['vdd_in_instant_mw']['mean'],3400)
        self.assertTrue(report['warnings'])

    def test_arm64_build_script_cannot_redirect_into_production_build(self):
        script=ROOT/'scripts/build_native_on_jetson.sh'
        body=script.read_text()
        self.assertIn('build_dir="$repo_root/build_chat_verify"',body)
        self.assertNotIn('BUILD_DIR:-',body)
        self.assertIn('if [ -L "$build_dir" ]',body)
        self.assertNotIn('args=(-S',body)
        self.assertNotIn('cmake --build "$build_dir" --parallel',body)
        self.assertIn('cmake --build "$build_dir" -- -j"$jobs"',body)
        response=subprocess.run(['bash','-n',str(script)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(response.returncode,0,response.stderr.decode('utf-8','replace'))

    def test_readonly_device_probe_has_no_device_actions(self):
        module=load('r41_device_readiness')
        with tempfile.TemporaryDirectory() as d:
            response=module.probe(Path(d))
        self.assertIn('ready_for_separate_build',response)
        self.assertNotIn('run',response)


if __name__=='__main__':unittest.main()
