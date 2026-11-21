"""Current canonical pipeline vs same-source crowd supplement off; still/PTY only."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from rescuevision.pipeline import Pipeline
from rescuevision.runtime import LoopbackSink,packet_for_age
from rescuevision.baseline import fingerprint


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);p.add_argument('--repeat',type=int,default=2);args=p.parse_args()
    inputs=json.loads(Path(args.inputs).read_text());assert not inputs['contains_ground_truth']
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);times=[]
    cfg=json.loads((ROOT/'configs/runtime.json').read_text())
    with (out/'frames.jsonl').open('w',buffering=1) as log:
        for profile,calib,prefix in [('multi','configs/multi_block_assumed_astra.json','A'),('astra','configs/astra_six_estimate.json','S')]:
            images=[]
            for r in inputs['images']:
                if r['key'].startswith(prefix):
                    im=cv2.imread(str(ROOT/r['path']));assert hashlib.sha256(cv2.cvtColor(im,cv2.COLOR_BGR2RGB).tobytes()).hexdigest()==r['rgb_sha256'];images.append((r['key'],im))
            config=dict(cfg,calibration=calib);pipe=Pipeline(ROOT,config,'yolo');canonical=pipe.frontend;base=canonical.base;sink=LoopbackSink()
            try:
                for key,im in images:pipe.process(im)
                for block,mode in enumerate(('crowd_off','canonical','canonical','crowd_off')):
                    pipe.frontend=canonical if mode=='canonical' else base
                    for repeat in range(args.repeat):
                        for key,im in images:
                            start=time.perf_counter();result=pipe.process(im);packet=packet_for_age(result,(time.perf_counter()-start)*1000,config['result_age_budget_ms']);transport=sink.write(packet)
                            log.write(json.dumps(dict(key=key,profile=profile,mode=mode,block=block,repeat=repeat,result=result,transport=transport),separators=(',',':'))+'\n');log.flush()
                            timing=dict(record_kind='timing',key=key,profile=profile,mode=mode,wall_ms=(time.perf_counter()-start)*1000);times.append(timing);log.write(json.dumps(timing)+'\n')
                    print(profile,block,mode,flush=True)
            finally:pipe.close();sink.close()
    summaries=[]
    for profile in ('multi','astra'):
        for mode in ('crowd_off','canonical'):
            v=np.array([r['wall_ms'] for r in times if r['profile']==profile and r['mode']==mode]);summaries.append(dict(profile=profile,mode=mode,frames=len(v),mean_ms=float(v.mean()),p95_ms=float(np.percentile(v,95)),hz=float(1000/v.mean())))
    (out/'summary.json').write_text(json.dumps(dict(baseline=fingerprint(ROOT),summaries=summaries,physical_camera=False,physical_uart=False,includes='fresh YOLO + all current image computation + geometry + UART encode + PTY + JSON',metric_accuracy_verified=False),indent=2))

if __name__=='__main__':main()
