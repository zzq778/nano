"""Image-only native/reference comparison; never uses human counts as input."""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from rescuevision.topface_native import NativeTopFace
from rescuevision.green_proposals import GreenFaceProposer
from rescuevision.green_regions import propose_observed_region


def compare(reference, actual):
    if len(reference) != len(actual):
        return dict(reason='count', reference=len(reference), actual=len(actual))
    for i, (a, b) in enumerate(zip(reference, actual)):
        if set(a) != set(b):
            return dict(reason='fields', index=i, reference=sorted(a), actual=sorted(b))
        for key in a:
            if a[key] != b[key]:
                return dict(reason='value', index=i, field=key, reference=a[key], actual=b[key])
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location('frozen_topface', ROOT / 'tests/reference/green_topface_reference.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    native = NativeTopFace(ROOT / 'build/libvision_green.so')
    ref_proposer = GreenFaceProposer(lambda image: reference.detect_with_isolated_fallback(image)[0])
    native_proposer = GreenFaceProposer(native.detect)
    rows = []
    for number, row in enumerate(json.loads(Path(args.inputs).read_text())['images'], 1):
        image = cv2.imread(str(ROOT / row['path']))
        if image is None:
            raise ValueError('image decode failed: ' + row['path'])
        boxes = [o['bbox'] for o in row['objects'] if o['type'] == 0]
        outputs = {}
        timings = {}
        for mode in ('reference', 'native'):
            start = time.perf_counter()
            full = reference.detect_with_isolated_fallback(image)[0] if mode == 'reference' else native.detect(image)
            proposed = propose_observed_region(ref_proposer if mode == 'reference' else native_proposer, image, boxes)
            timings[mode] = (time.perf_counter() - start) * 1000
            outputs[mode] = dict(full=full, proposed=proposed)
        errors = {stage: compare(outputs['reference'][stage], outputs['native'][stage]) for stage in ('full', 'proposed')}
        rows.append(dict(key=row['key'], errors=errors, timings_ms=timings, outputs=outputs))
        if number % 25 == 0:
            print('images', number, 'mismatches', sum(any(r['errors'].values()) for r in rows), flush=True)
    summary = dict(images=len(rows), mismatches=sum(any(r['errors'].values()) for r in rows),
                   reference_is_test_only=True, known_counts_used=False)
    Path(args.output).write_text(json.dumps(dict(summary=summary, images=rows), indent=2))
    print(json.dumps(summary), flush=True)
    if summary['mismatches']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
