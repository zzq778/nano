"""Stage-only ABBA timing and exact proposal equivalence on real RGB inputs."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from rescuevision.frontends import blue_context_proposals
from blue_context_local import local_blue_context


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    cfg=json.loads((ROOT/'configs/runtime.json').read_text())['blue_context_supplement'];inputs=json.loads(Path(args.inputs).read_text());rows=[]
    for row in inputs['images']:
        im=cv2.imread(str(ROOT/row['path']));assert hashlib.sha256(cv2.cvtColor(im,cv2.COLOR_BGR2RGB).tobytes()).hexdigest()==row['rgb_sha256']
        outputs={};timings={mode:[] for mode in ('full','local')}
        for mode in ('full','local','local','full'):
            start=time.perf_counter();out=blue_context_proposals(im,row['objects'],cfg) if mode=='full' else local_blue_context(im,row['objects'],cfg,blue_context_proposals)
            timings[mode].append((time.perf_counter()-start)*1000);outputs[mode]=out
        same=outputs['full']==outputs['local']
        rows.append(dict(key=row['key'],exact_proposal_parity=same,timings_ms=timings,full=outputs['full'],local=outputs['local']))
    summary=dict(images=len(rows),exact_match=sum(r['exact_proposal_parity'] for r in rows),physical_accuracy_measured=False,
        means_ms={mode:float(np.mean([t for r in rows for t in r['timings_ms'][mode]])) for mode in ('full','local')})
    Path(args.output).write_text(json.dumps(dict(summary=summary,images=rows),indent=2));print(json.dumps(summary),flush=True)
    if summary['exact_match']!=len(rows):raise SystemExit(2)

if __name__=='__main__':main()
