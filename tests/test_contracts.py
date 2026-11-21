import importlib
import importlib.util
import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))

class Contracts(unittest.TestCase):
    def test_support_output_selection_preserves_other_watershed_markers(self):
        import numpy as np
        f=self.module('frontends').green_color_support
        image=np.full((100,120,3),230,np.uint8);image[20:85,15:105]=[20,160,20]
        objects=[dict(bbox=[x-15,20,x+15,40],seed=dict(x=x,y=30,x_span=[x-15,x+15],y_span=[20,40])) for x in (35,85)]
        full=f(image,objects);selected=f(image,objects,output_start=1)
        self.assertEqual(selected,full[1:])
        self.assertNotEqual(selected,f(image,objects[1:]))

    def test_support_empty_output_selection_is_empty(self):
        import numpy as np
        self.assertEqual(self.module('frontends').green_color_support(np.full((30,30,3),230,np.uint8),[],output_start=0),[])

    def test_proven_group_parent_is_removed_and_children_are_independent(self):
        f=self.module('green_association').fuse
        def face(y):
            return dict(seed=dict(x=40,y=y,x_span=[25,55],y_span=[y-5,y+5]),
                result=dict(center=[40,y],status='edge_candidate',back_line=[0,y-5],front_line=[0,y+5]))
        parent=[dict(bbox=[0,0,100,100],confidence=.9)]
        one=f(parent,[face(30)],True)
        self.assertEqual(one['rejected_yolo_groups'],[])
        two=f(parent,[face(30),face(50)],True)
        self.assertEqual(two['experimental_count'],2)
        self.assertEqual(len(two['rejected_yolo_groups']),1)
        self.assertEqual(two['associations'],[])
        self.assertEqual(len(two['added_faces']),2)
        duplicate=f(parent,[face(30),face(30)],True)
        self.assertEqual(duplicate['rejected_yolo_groups'],[])
        self.assertEqual(duplicate['experimental_count'],1)

    def module(self,name):
        if importlib.util.find_spec('rescuevision.'+name) is None:
            self.fail('Missing required runtime module: '+name)
        return importlib.import_module('rescuevision.'+name)

    def test_official_uart_vectors(self):
        u=self.module('uart')
        self.assertEqual(u.crc16(b'123456789'),0x29b1)
        self.assertEqual(u.encode([]).hex(),'aa5500f0e1')
        self.assertEqual(u.encode([]).hex(),'aa5500f0e1')
        self.assertEqual(u.encode([{'type':0,'x_mm':450,'y_mm':-100}]).hex(),'aa550100c2019cff000002af')

    def test_uart_overflow_invalid_and_yaw_zero(self):
        u=self.module('uart')
        with self.assertRaises(ValueError):u.encode([{'type':0,'x_mm':0,'y_mm':0}]*33)
        for row in ({'type':8,'x_mm':0,'y_mm':0},{'type':0,'x_mm':32768,'y_mm':0},{'type':0,'x_mm':float('nan'),'y_mm':0}):
            with self.assertRaises(ValueError):u.encode([row])
        packet=u.encode([{'type':2,'x_mm':0,'y_mm':0,'yaw':123}])
        self.assertEqual(packet[8:10],b'\x00\x00')

    def test_near_color_priority_and_center_only(self):
        s=self.module('scheduling')
        objects=[dict(type=t,near=n,bbox=[0,0,20,b],index=i) for i,(t,n,b) in enumerate([(2,False,150),(0,True,400),(2,True,300),(0,False,190),(1,True,450),(3,True,440)])]
        self.assertEqual(s.pose_order(objects),[2,0])

    def test_cross_zone_group_veto_uses_frozen_anchors(self):
        s=self.module('scheduling')
        seeds=[{'x_span':[0,100]},{'x_span':[25,75]}]
        rows=[dict(status='edge_candidate',reason='two_coherent_signed_rims',center=[50,20],initial_center=[50,5],back_line=[0,0],front_line=[0,40]),dict(status='fallback',center=[50,25],initial_center=[50,25])]
        s.group_veto(rows,seeds)
        self.assertEqual(rows[0]['reason'],'rims_span_another_candidate')
        self.assertEqual(rows[0]['center'],[50,5])

    def calibration(self,rectification):
        return dict(image_size=[640,480],matches_input_camera=True,intrinsics=dict(fx=500,fy=500,cx=320,cy=240),optical_center_in_axle_frame_m=dict(x=0,y=.0179,z=.129),camera_orientation_deg=dict(roll=0,pitch_down=0,yaw=0),ground_plane_z_m=-.036,distortion_coefficients=None,rectification_status=rectification)

    def test_unknown_calibration_never_guesses_zero_distortion(self):
        g=self.module('geometry').Geometry(self.calibration('unknown'))
        self.assertFalse(g.ready)
        self.assertIsNone(g.project([320,740],(480,640),-.036))

    def test_explicit_astra_pinhole_estimate_remains_unverified(self):
        config=self.calibration('unknown');config['projection_policy']='pinhole_estimate_unverified_distortion'
        g=self.module('geometry').Geometry(config)
        self.assertTrue(g.ready);self.assertIsNone(g.D)
        r=g.estimate(dict(type=1,observed_apex_px=[320,450],apex_evidence_valid=True),(480,640))
        self.assertFalse(r['accuracy_verified']);self.assertFalse(r['distortion_verified'])
        self.assertEqual(g.config['rectification_status'],'unknown')

    def test_known_model_multiple_solutions_do_not_claim_unique_angle(self):
        g=self.module('geometry').Geometry(self.calibration('rectified'))
        fit=dict(status=2,x=.3,y=.05,yaw_rad=.2,error_px=1.,candidates=[dict(x=.3,y=.05,yaw_rad=.2),dict(x=.301,y=.05,yaw_rad=.5)])
        r=g.from_model(dict(type=2),fit)
        self.assertTrue(r['position_valid']);self.assertFalse(r['pose_valid'])
        self.assertIsNone(r['yaw_world_deg']);self.assertEqual(len(r['yaw_candidates_deg']),2)
        self.assertFalse(r['accuracy_verified'])

    def test_orientation_policy_retains_best_ambiguous_and_default(self):
        policy=self.module('orientation').complete_orientation
        for cls,period in ((2,180),):
            fit=dict(status=2,yaw_rad=math.radians(127))
            r=policy(cls,dict(pose_valid=False,yaw_world_deg=None,model_fit=fit))
            self.assertTrue(r['pose_available']);self.assertFalse(r['pose_valid'])
            self.assertAlmostEqual(r['yaw_world_deg'],127%period)
            self.assertEqual(r['yaw_source'],'accepted_model_best_of_multiple')
            best=policy(cls,dict(pose_valid=False,model_fit=dict(status=3,best_effort=dict(yaw_rad=.3))))
            self.assertEqual(best['yaw_source'],'rejected_model_best_effort');self.assertFalse(best['yaw_measured'])
            default=policy(cls,dict(pose_valid=False),{0:10,2:25})
            self.assertEqual(default['yaw_world_deg'],10 if cls==0 else 25)
            self.assertEqual(default['yaw_source'],'configured_default');self.assertFalse(default['pose_valid'])
        for cls in (0,1,3):
            r=policy(cls,dict(pose_valid=False,yaw_world_deg=None))
            self.assertFalse(r['pose_available']);self.assertIsNone(r['yaw_world_deg'])

    def test_best_effort_angle_does_not_make_failed_position_valid(self):
        policy=self.module('orientation').complete_orientation
        r=policy(2,dict(position_valid=False,pose_valid=False,x_mm=None,y_mm=None,model_fit=dict(status=5,best_effort=dict(yaw_rad=1.))))
        self.assertFalse(r['position_valid']);self.assertIsNone(r['x_mm'])
        self.assertTrue(r['pose_available']);self.assertFalse(r['pose_valid'])

    def test_assumed_stress_calibration_rejected_before_opening_camera_or_real_uart(self):
        import subprocess,tempfile
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as output:
            cases=[['--camera','astra'],['--manifest','not_read.json','--uart-device','DO_NOT_OPEN']]
            for options in cases:
                p=subprocess.run([sys.executable,str(root/'app.py'),'--mode','color','--calibration',str(root/'configs/multi_block_assumed_astra.json'),'--output',output]+options,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                self.assertEqual(p.returncode,2)
                self.assertIn(b'assumed-camera stress configuration',p.stderr)

    def test_missing_camera_calibration_keeps_detection_and_default_angle_without_loading_legacy_kernels(self):
        import json,numpy as np
        from unittest.mock import patch
        module=self.module('pipeline');root=Path(__file__).resolve().parents[1]
        config=json.loads((root/'configs/runtime.json').read_text());config.update(geometry_backend='native_known_size_rgb',native_metric='build/libvision_metric.so',calibration='configs/calibration.json')
        class Frontend:
            coverage=[0,1,2,3]
            def detect(self,image):return [dict(type=0,bbox=[100,100,160,180],center_px=[130,140],confidence=None)]
        with patch.object(module,'ColorFrontend',return_value=Frontend()),patch.object(module,'NativeMetric',side_effect=AssertionError('must not load without calibration')),patch.object(module,'NativeGreen',side_effect=AssertionError('must not select another geometry backend')):
            pipeline=module.Pipeline(root,config,'color');result=pipeline.process(np.zeros((480,640,3),np.uint8))
        self.assertEqual(result['counts']['green'],1)
        obj=result['objects'][0];self.assertFalse(obj['metric']['pose_available']);self.assertIsNone(obj['metric']['yaw_world_deg'])
        self.assertFalse(obj['position_valid']);self.assertEqual(self.module('uart').decode(bytes.fromhex(result['uart_hex'])),[])

    def test_ray_frame_convention_and_size_gate(self):
        g=self.module('geometry').Geometry(self.calibration('rectified'))
        self.assertTrue(g.ready)
        p=g.project([320,740],(480,640),-.036)
        self.assertAlmostEqual(p[0],.165,places=8);self.assertAlmostEqual(p[1],.0179,places=8)
        with self.assertRaises(ValueError):g.project([320,240],(240,320),-.036)
        self.assertIsNone(g.project([320,240],(480,640),-.036))

    def test_unobserved_shape_never_becomes_metric_pose(self):
        g=self.module('geometry').Geometry(self.calibration('rectified'))
        r=g.estimate({'type':0,'center_px':[320,400]},(480,640))
        self.assertFalse(r['position_valid'])

    def test_wrong_camera_never_reuses_valid_old_calibration(self):
        config=self.calibration('rectified');config['matches_input_camera']=False
        g=self.module('geometry').Geometry(config)
        self.assertFalse(g.ready)
        self.assertIsNone(g.project([320,740],(480,640),-.036))

    def test_uart_rejects_crc_length_and_type(self):
        u=self.module('uart')
        p=u.encode([{'type':0,'x_mm':450,'y_mm':-100}])
        for invalid in (p[:-1],p+b'\x00',p[:-1]+bytes([p[-1]^1]),b'\xaa\x55\x21\x00\x00'):
            with self.assertRaises(ValueError):u.decode(invalid)

    def test_continuous_over_budget_results_keep_valid_uart_targets(self):
        io=self.module('runtime');u=self.module('uart')
        for age in (201,350,900,199,200):
            p=u.encode([dict(type=0,x_mm=450,y_mm=-100)])
            r=dict(uart_hex=p.hex(),uart_fault=False,uart_fault_reasons=[])
            packet=io.packet_for_age(r,age,200)
            self.assertEqual(packet,p)
            self.assertFalse(r['uart_fault']);self.assertEqual(r['uart_fault_reasons'],[])
            self.assertEqual(r['result_age_ms'],age)
            self.assertEqual(r['result_age_over_budget'],age>200)

    def test_over_budget_does_not_restore_an_invalid_result(self):
        io=self.module('runtime');u=self.module('uart')
        r=dict(uart_hex=u.encode([]).hex(),uart_fault=True,uart_fault_reasons=['calibration_incomplete'])
        self.assertEqual(io.packet_for_age(r,350,200),u.encode([]))
        self.assertEqual(r['uart_fault_reasons'],['calibration_incomplete'])

    def test_pseudo_uart_actual_write_and_decode(self):
        import os
        if os.name!='posix':self.skipTest('POSIX pseudo UART checked on Nano')
        io=self.module('runtime');u=self.module('uart')
        with io.LoopbackSink() as sink:
            packet=u.encode([{'type':0,'x_mm':450,'y_mm':-100}])
            for age in (201,350,900):
                result=dict(uart_hex=packet.hex(),uart_fault=False,uart_fault_reasons=[])
                receipt=sink.write(io.packet_for_age(result,age,200))
                self.assertTrue(receipt['loopback_verified'])
                self.assertEqual(receipt['bytes'],12)

    def test_split_green_priority_uses_own_support_span(self):
        f=self.module('frontends');self.assertTrue(hasattr(f,'green_observation'),'missing per-proposal support adapter')
        seed=dict(x=50,y=130,x_span=[20,80],y_span=[100,160],source_comp=[0,80,100,260])
        obj=f.green_observation(seed)
        self.assertLess(obj['bbox'][3],240)
        self.assertEqual(obj['component_bbox'][3],340)

    def test_classic_top_support_is_extended_using_only_current_green_pixels(self):
        import numpy as np
        f=self.module('frontends');self.assertTrue(hasattr(f,'green_color_support'),'missing top-face to visible-body adapter')
        # Array fixture checks adapter semantics, not visual accuracy.
        image=np.zeros((100,100,3),np.uint8);image[:]=230;image[20:81,20:71]=[20,160,20]
        seeds=[dict(x=45,y=30,x_span=[23,67],y_span=[22,38],source_comp=[20,20,51,61])]
        objects=[f.green_observation(seeds[0])]
        result=f.green_color_support(image,objects)
        self.assertEqual(len(result),1)
        self.assertGreater(result[0]['bbox'][3],70)
        self.assertEqual(result[0]['top_face_bbox'],objects[0]['bbox'])
        self.assertEqual(result[0]['bbox_kind'],'green_visible_color_support_not_amodal_cube')

    def test_blocked_capture_process_has_bounded_shutdown(self):
        io=self.module('runtime');self.assertTrue(hasattr(io,'stop_capture_process'),'missing bounded capture shutdown')
        from types import SimpleNamespace
        class Process:
            def __init__(self):self.alive=True;self.terminated=False;self.joins=0
            def join(self,timeout):self.joins+=1
            def is_alive(self):return self.alive
            def terminate(self):self.terminated=True;self.alive=False
        process=Process();stop=SimpleNamespace(value=False)
        io.stop_capture_process(process,stop,.01)
        self.assertTrue(stop.value);self.assertTrue(process.terminated);self.assertEqual(process.joins,2)

    def test_blue_supplement_preserves_model_boxes_and_deduplicates(self):
        f=self.module('frontends');self.assertTrue(hasattr(f,'merge_blue_proposals'),'missing positive color supplement merge')
        model=[dict(type=3,bbox=[10,10,50,60],source='model'),dict(type=0,bbox=[80,10,130,60],source='model')]
        proposals=[dict(type=3,bbox=[12,12,48,58],center_px=[30,35]),dict(type=3,bbox=[150,10,190,50],center_px=[170,30])]
        result=f.merge_blue_proposals(model,proposals)
        self.assertEqual(result[:2],model)
        self.assertEqual(len(result),3)
        self.assertEqual(model[0]['source'],'model')

    def test_packet_snapshot_is_outside_frame_hot_path(self):
        io=self.module('runtime');self.assertTrue(hasattr(io,'PacketRecorder'),'missing deferred packet snapshot')
        class File:
            def __init__(self):self.writes=[]
            def write_bytes(self,p):self.writes.append(p)
        f=File();recorder=io.PacketRecorder(f)
        recorder.record(b'first');recorder.record(b'last')
        self.assertEqual(f.writes,[])
        recorder.close();self.assertEqual(f.writes,[b'last'])

    def test_black_apex_measures_projected_physical_center_and_ranges(self):
        import numpy as np
        g=self.module('geometry').Geometry(self.calibration('rectified'))
        center=np.array([.3,.05]);height=.04*math.sqrt(2./3.)
        optical=g.R.T.dot(np.array([center[0],center[1],g.ground+height])-g.t)
        pixel=[g.K[0,0]*optical[0]/optical[2]+g.K[0,2],g.K[1,1]*optical[1]/optical[2]+g.K[1,2]]
        r=g.estimate(dict(type=1,observed_apex_px=pixel,apex_evidence_valid=True),(480,640))
        self.assertTrue(r['position_valid'])
        self.assertEqual((r['x_mm'],r['y_mm']),(300,50))
        self.assertAlmostEqual(r['range_horizontal_mm'],math.hypot(300,50),places=6)
        self.assertAlmostEqual(r['ground_point_to_axle_distance_mm'],math.sqrt(300**2+50**2+36**2),places=6)
        self.assertFalse(r['pose_valid']);self.assertIsNone(r['yaw_world_deg'])

    def test_blue_shape_center_uses_physical_cube_without_exporting_yaw(self):
        import numpy as np
        g=self.module('geometry').Geometry(self.calibration('rectified'))
        angle=math.radians(30);co,si=math.cos(angle),math.sin(angle)
        points=np.array([[.45+co*x-si*y,.06+si*x+co*y,z] for z in (g.ground,g.ground+.04) for x in (-.02,.02) for y in (-.02,.02)])
        optical=(points-g.t).dot(g.R);uv=optical[:,:2]/optical[:,2:]*np.array([g.K[0,0],g.K[1,1]])+g.K[:2,2]
        bbox=[float(uv[:,0].min()),float(uv[:,1].min()),float(uv[:,0].max()),float(uv[:,1].max())]
        r=g.estimate(dict(type=3,bbox=bbox,near=True),(480,640))
        self.assertTrue(r['position_valid'])
        self.assertLess(abs(r['x_mm']-450),5);self.assertLess(abs(r['y_mm']-60),5)
        self.assertFalse(r['pose_valid']);self.assertIsNone(r['yaw_world_deg'])

if __name__=='__main__':unittest.main()
