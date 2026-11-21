"""Consolidate only standalone algorithm-config duplicates; never touch ROS/user data."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
target_dir=ROOT/'experiments/configs';target_dir.mkdir(parents=True,exist_ok=True)
for source in (ROOT/'configs').glob('*.json'):
    if source.name=='runtime.json' or 'model' not in json.loads(source.read_text()):continue
    target=target_dir/source.name
    assert ROOT.resolve() in source.resolve().parents and ROOT.resolve() in target.resolve().parents
    if target.exists():
        if target.read_bytes()!=source.read_bytes():raise ValueError('different experiment config already archived: '+source.name)
        source.unlink()  # Identical recoverable copy already checked above.
    else:source.rename(target)
    print('archived standalone experiment config',source.name)
