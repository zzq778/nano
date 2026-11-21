import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np,cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from rescuevision.sampling import SampleRecorder
from rescuevision import uart

class SamplingContracts(unittest.TestCase):
    def test_pairs_exact_image_packet_and_frame_with_interval_and_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            s=SampleRecorder(directory,10,2,{'baseline':'fixture'})
            image=np.arange(36,dtype=np.uint8).reshape(3,4,3);packet=uart.encode([])
            self.assertIsNotNone(s.maybe_save(image,{'sequence_local':13},packet,now=0))
            self.assertIsNone(s.maybe_save(image,{'sequence_local':14},packet,now=9))
            self.assertIsNotNone(s.maybe_save(image,{'sequence_local':15},packet,now=10))
            self.assertIsNone(s.maybe_save(image,{'sequence_local':16},packet,now=20))
            rows=[json.loads(p.read_text()) for p in sorted(s.root.glob('*seq*.json'))]
            self.assertEqual([r['result']['sequence_local'] for r in rows],[13,15])
            for row in rows:
                self.assertEqual(row['sent_uart_hex'],packet.hex())
                np.testing.assert_array_equal(cv2.imread(str(s.root/row['image'])),image)

    def test_disk_error_disables_sampling_without_raising_into_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            s=SampleRecorder(directory,10,2,{})
            with patch('rescuevision.sampling.cv2.imwrite',side_effect=OSError('injected disk failure')):
                event=s.maybe_save(np.zeros((2,2,3),np.uint8),{'sequence_local':1},uart.encode([]),now=0)
            self.assertEqual(event['record_kind'],'sampling_disabled')
            self.assertIsNone(s.maybe_save(None,{'sequence_local':2},uart.encode([]),now=20))

if __name__=='__main__':unittest.main()
