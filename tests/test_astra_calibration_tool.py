import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from rescuevision.geometry import Geometry


def tool():
    spec=importlib.util.spec_from_file_location('astra_calibration_tool',ROOT/'tools/calibrate_current_camera.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class CalibrationToolContracts(unittest.TestCase):
    def test_checkerboard_capture_uses_the_sdk_library_and_source(self):
        module=tool();closed=[]
        class Camera:
            def __init__(self,source,width,height,library):
                self.assertions=(source,width,height,Path(library))
                if self.assertions!=('astra',640,480,ROOT/'build/libvision_astra.so'):raise ValueError('wrong SDK capture arguments')
            def get(self,*args):return 1,np.zeros((480,640,3),np.uint8),0.
            def close(self):closed.append(True)
        with tempfile.TemporaryDirectory() as directory:
            args=['tool.py','capture','--device','astra','--output',directory,'--count','1','--interval','0']
            with patch.object(sys,'argv',args),patch.object(module,'LatestCamera',Camera),patch.object(module,'corners',return_value=np.zeros((54,1,2),np.float32)),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(module.main(),0)
            capture=json.loads((Path(directory)/'capture.json').read_text())
            self.assertEqual(capture['camera_usb_id'],'2bc5:0402')
            self.assertEqual(capture['recognized_views'],1)
        self.assertEqual(closed,[True])

    def test_measured_distortion_is_not_ignored_by_an_inherited_pinhole_policy(self):
        module=tool();model=module.board_points(9,6,18)
        K=np.array([[493.3689,0,327.3957],[0,494.2357,234.8487],[0,0,1]],float)
        D=np.array([-.03,.005,0,0,0.],float);observations=[]
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            for i in range(12):
                rvec=np.array([.12+.028*i,-.2+.015*i,.03*i])
                tvec=np.array([-.07+.008*(i%4),-.04+.01*(i%3),.38+.02*(i%5)])
                observations.append(cv2.projectPoints(model,rvec,tvec,K,D)[0].astype(np.float32))
                image=np.full((480,640,3),i,np.uint8);cv2.imwrite(str(directory/('board_%02d.png'%i)),image)
            (directory/'capture.json').write_text(json.dumps(dict(camera_usb_id='2bc5:0402')))
            output=directory/'measured.json'
            args=['tool.py','intrinsics','--images',str(directory),'--square-mm','18','--base-config',str(ROOT/'configs/astra_current.json'),'--output',str(output)]
            def observed(image,*_):return observations[int(image[0,0,0])]
            with patch.object(sys,'argv',args),patch.object(module,'corners',side_effect=observed),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(module.main(),0)
            geometry=Geometry(json.loads(output.read_text()))
            self.assertTrue(geometry.ready)
            self.assertFalse(geometry.use_pinhole,'measured D must be used by ground projection')


if __name__=='__main__':unittest.main()
