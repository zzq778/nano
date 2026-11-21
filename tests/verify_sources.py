"""Check delivered source/data identities before running any image experiment."""
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
identity=json.loads((root/'source_identity.json').read_text())
for name,expected in identity.items():
    path=root/name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
import cv2
manifest_path=root/'data/approved_six/manifest.json'
manifest=json.loads(manifest_path.read_text())
assert len(manifest['images'])==6
for row in manifest['images']:
    image=cv2.imread(str(manifest_path.parent/row['path']))
    assert image.shape==(480,640,3)
    assert hashlib.sha256(cv2.cvtColor(image,cv2.COLOR_BGR2RGB).tobytes()).hexdigest()==row['rgb_sha256']
print('PASS_PROJECT_SOURCE_AND_SIX_PIXEL_IDENTITIES')
