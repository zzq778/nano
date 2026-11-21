"""Actual implementation/config fingerprints shared by every frontend experiment."""
import hashlib,json
from pathlib import Path

def fingerprint(root):
    root=Path(root);files=[root/'app.py']
    for directory in ('python/rescuevision','src','include','vendor','vendor_model'):
        files.extend(p for p in (root/directory).rglob('*') if p.is_file() and p.suffix in ('.py','.cpp','.hpp','.json'))
    identities={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
    config_path=root/'configs/runtime.json';config=json.loads(config_path.read_text())
    return dict(baseline_id=config.get('baseline_id','unversioned'),entrypoint='app.py',configuration='configs/runtime.json',
                baseline_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                implementation_sha256=hashlib.sha256(json.dumps(identities,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                implementation_files=identities)
