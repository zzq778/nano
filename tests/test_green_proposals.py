import sys,unittest
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from rescuevision.green_proposals import GreenFaceProposer
from rescuevision.topface_native import NativeTopFace
from native_support import native_library_matches_host
class GreenCandidateContracts(unittest.TestCase):
 def test_extra_support_beyond_one_box_is_inspected(self):
  seen=[]
  def kernel(crop):seen.append(crop.shape);return []
  image=np.zeros((100,120,3),np.uint8);image[20:70,30:90]=[20,170,20]
  GreenFaceProposer(kernel).propose(image,[[30,20,60,70]])
  self.assertEqual(len(seen),1)
 def test_compact_component_inside_one_box_skips_redundant_work(self):
  seen=[]
  def kernel(crop):seen.append(crop.shape);return []
  image=np.zeros((100,120,3),np.uint8);image[20:70,30:90]=[20,170,20]
  GreenFaceProposer(kernel).propose(image,[[30,20,90,70]])
  self.assertEqual(seen,[])
 @unittest.skipUnless(native_library_matches_host(ROOT/'build/libvision_green.so'),'native short-top fixture runs on Nano')
 def test_short_bright_top_survives_as_top_evidence_not_full_body_fallback(self):
  image=np.full((100,120,3),230,np.uint8)
  image[20:33,25:90]=[20,210,20];image[33:85,25:90]=[15,100,15]
  faces=NativeTopFace(ROOT/'build/libvision_green.so').detect(image)
  measured=[f for f in faces if 'method' not in f]
  self.assertTrue(measured,faces)
  self.assertTrue(any(8<=f['y_span'][1]-f['y_span'][0]+1<18 for f in measured),faces)
if __name__=='__main__':unittest.main()
