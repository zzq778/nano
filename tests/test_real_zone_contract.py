"""Real Astra pixels catch the wall/floor hull contamination regression."""
import hashlib,json,sys,unittest
from pathlib import Path
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from rescuevision.geometry import Geometry
from rescuevision.zones import estimate_zone_regions,own_subregion_targets
ROOT=Path(__file__).resolve().parents[1]

class RealZoneContract(unittest.TestCase):
    def test_real_outline_is_not_the_yolo_rectangle(self):
        directory=ROOT/'data/zone_contract'
        fixture_path=directory/'fixture.json'
        if not fixture_path.is_file():
            self.skipTest('requires private zone_contract fixtures')
        fixture=json.loads(fixture_path.read_text())
        path=directory/fixture['image']
        if not path.is_file():
            self.skipTest('requires private zone_contract image')
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),fixture['file_sha256'])
        image=cv2.imread(str(path))
        geometry=Geometry(json.loads((ROOT/'configs/astra_current.json').read_text()))
        zone=estimate_zone_regions(image,[fixture['detection']],geometry,'blue')[0]
        self.assertTrue(zone['pose_valid'],zone)
        short,long=zone['measured_floor_sides_m']
        # The original broad color mask inflated this frame to ~500x964 mm.
        self.assertTrue(.28<short<.45,(short,long))
        self.assertTrue(.55<long<.85,(short,long))
        self.assertEqual([t['type'] for t in own_subregion_targets([zone])],[4,5])
        enemy=estimate_zone_regions(image,[fixture['detection']],geometry,'purple')[0]
        self.assertEqual(own_subregion_targets([enemy]),[])

if __name__=='__main__':unittest.main()