"""Same-session ABBA replay of the received r3 package and current native pipeline."""
import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', required=True)
    p.add_argument('--archive', required=True)
    p.add_argument('--multi-manifest', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    project = Path(args.project).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / 'received_r3'
    if checkpoint.exists():
        raise ValueError('use a new output directory; do not overwrite baseline evidence')
    with zipfile.ZipFile(args.archive) as archive:
        for member in archive.namelist():
            try:
                (checkpoint / member).resolve().relative_to(checkpoint.resolve())
            except ValueError:
                raise ValueError('unsafe archive path')
        archive.extractall(checkpoint)
    before = checkpoint / 'jetson_vision'
    (before / 'build').symlink_to(project / 'build', target_is_directory=True)
    # The original kernels in this library are unchanged; the new symbols are
    # never called by r3. CUDA/metric libraries are shared identically by both.
    rows = []
    for profile in ('multi', 'astra'):
        for block, mode in enumerate(('before', 'after', 'after', 'before')):
            root = before if mode == 'before' else project
            manifest = Path(args.multi_manifest).resolve() if profile == 'multi' else root / 'data/approved_six/manifest.json'
            calibration = 'configs/multi_block_assumed_astra.json' if profile == 'multi' else 'configs/astra_six_estimate.json'
            destination = output / ('%s_%d_%s' % (profile, block, mode))
            subprocess.run([sys.executable, str(root / 'app.py'), '--manifest', str(manifest),
                            '--calibration', calibration, '--output', str(destination),
                            '--warmup', '1', '--repeat', '2', '--preload-replay',
                            '--quiet', '--uart-loopback'], cwd=str(root), check=True)
            summary = json.loads((destination / 'run_summary.json').read_text())
            metadata = json.loads((destination / 'metadata.json').read_text())
            rows.append(dict(profile=profile, mode=mode, block=block,
                             summary=summary, baseline=metadata['baseline']))
            print(profile, block, mode, summary['mean_loop_ms'], flush=True)
    (output / 'runs.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
