"""Explicit still-only experiment around the frozen Pipeline; no camera/physical UART.

Every measured frame runs fresh YOLO, classic image proposals, native rims,
association, the existing geometry/default-angle policy, encoding and PTY.
"""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from rescuevision.pipeline import Pipeline
from rescuevision.native import NativeGreen
from rescuevision.frontends import green_observation,green_color_support
from rescuevision.scheduling import group_veto
from rescuevision.runtime import LoopbackSink,packet_for_age
from rescuevision.baseline import fingerprint
from green_classic_sensitivity import ClassicProposer
from green_evidence_fusion import FACE_OPTIONS,fuse
from green_local_evidence import refine_local,added_support_local,propose_observed_region


class Supplement:
    def __init__(self,base,optimized=False,shared_features=False,reject_groups=False):
        self.base=base;self.coverage=base.coverage;self.optimized=optimized
        self.reject_groups=reject_groups
        self.proposer=ClassicProposer(dict(FACE_OPTIONS,crowd_only=optimized,shared_features=shared_features))
        cfg=json.loads((ROOT/'vendor/config.json').read_text())['green_thresholds'];cfg['back_expand_width']=.08
        self.solver=NativeGreen(ROOT/'build/libvision_green.so',cfg);self.last={}
    def detect(self,image):
        t=time.perf_counter();objects=self.base.detect(image)
        low_confidence=[o for o in objects if o['type']!=3 and o.get('confidence') is not None and o['confidence']<.5] if self.reject_groups else []
        objects=[o for o in objects if o not in low_confidence];a=time.perf_counter()
        green=[o for o in objects if o['type']==0];boxes=[o['bbox'] for o in green]
        seeds=propose_observed_region(self.proposer,image,boxes) if self.optimized else self.proposer.propose(image,boxes);b=time.perf_counter()
        if self.optimized:results=refine_local(self.solver,image,seeds)
        else:
            ctx=self.solver.prepare(image,'raw')
            try:results=[self.solver.local(ctx,s) for s in seeds]
            finally:self.solver.release(ctx)
        group_veto(results,seeds);evidence=[dict(seed=s,result=r) for s,r in zip(seeds,results)];c=time.perf_counter()
        fused=fuse(green,evidence,self.reject_groups);additions=[green_observation(t['evidence']['seed']) for t in fused['added_faces']]
        rejected_groups=[green[g['yolo']] for g in fused['rejected_yolo_groups']]
        support_green=[g for g in green if g not in rejected_groups]
        if additions:
            # Partition current visible color for new candidate boxes only. Existing
            # baseline observations are preserved; these are not amodal boxes.
            supports=added_support_local(image,support_green,additions) if self.optimized else green_color_support(image,support_green+additions)[len(support_green):]
            for out in supports:out['source']='experimental_classic_local_rims_visible_support'
            objects+=supports
        objects=[o for o in objects if o not in rejected_groups]
        d=time.perf_counter();self.last=dict(evidence=fused,rejected_group_objects=rejected_groups,rejected_low_confidence=low_confidence,timings_ms=dict(yolo_and_existing_color=(a-t)*1000,
            classic=(b-a)*1000,native_rims=(c-b)*1000,association_and_added_support=(d-c)*1000,supplement=(d-a)*1000))
        return objects


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);p.add_argument('--repeat',type=int,default=2);p.add_argument('--optimized',action='store_true');p.add_argument('--shared-features',action='store_true');p.add_argument('--blue-local',action='store_true');p.add_argument('--reject-groups',action='store_true');args=p.parse_args()
    if args.blue_local:
        import rescuevision.frontends as frontends
        from blue_context_local import local_blue_context
        reference=frontends.blue_context_proposals
        frontends.blue_context_proposals=lambda image,objects,cfg:local_blue_context(image,objects,cfg,reference)
    inp=json.loads(Path(args.inputs).read_text());assert inp['contains_ground_truth'] is False
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'configs/runtime.json').read_text());rows=[]
    with (out/'frames.jsonl').open('w',buffering=1) as log:
        for profile,calib,prefix in [('multi','configs/multi_block_assumed_astra.json','A'),('astra','configs/astra_six_estimate.json','S')]:
            images=[]
            for r in inp['images']:
                if not r['key'].startswith(prefix):continue
                im=cv2.imread(str(ROOT/r['path']))
                assert hashlib.sha256(cv2.cvtColor(im,cv2.COLOR_BGR2RGB).tobytes()).hexdigest()==r['rgb_sha256']
                images.append((r['key'],im))
            cfg=dict(config,calibration=calib);pipeline=Pipeline(ROOT,cfg,'yolo');original=pipeline.frontend;supplement=Supplement(original,args.optimized,args.shared_features,args.reject_groups);sink=LoopbackSink()
            try:
                for key,im in images:pipeline.process(im)
                # ABBA controls warmup/order effects. Image order remains fixed.
                for block,mode in enumerate(('baseline','fusion','fusion','baseline')):
                    pipeline.frontend=supplement if mode=='fusion' else original
                    for repeat in range(args.repeat):
                        for key,im in images:
                            start=time.perf_counter();result=pipeline.process(im);age=(time.perf_counter()-start)*1000
                            packet=packet_for_age(result,age,cfg['result_age_budget_ms']);transport=sink.write(packet)
                            record=dict(key=key,profile=profile,mode=mode,block=block,repeat=repeat,result=result,transport=transport,
                                supplement=supplement.last if mode=='fusion' else None)
                            log.write(json.dumps(record,separators=(',',':'))+'\n');log.flush();elapsed=(time.perf_counter()-start)*1000
                            timing=dict(record_kind='timing',key=key,profile=profile,mode=mode,block=block,repeat=repeat,wall_with_json_ms=elapsed)
                            rows.append(timing);log.write(json.dumps(timing)+'\n')
                    print(profile,block,mode,'finished',flush=True)
            finally:pipeline.close();sink.close()
    summaries=[]
    for profile in ('multi','astra'):
        for mode in ('baseline','fusion'):
            times=np.array([r['wall_with_json_ms'] for r in rows if r['profile']==profile and r['mode']==mode])
            summaries.append(dict(profile=profile,mode=mode,frames=len(times),mean_ms=float(times.mean()),p50_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),throughput_hz=float(1000/times.mean())))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('green_*.py')}
    (out/'summary.json').write_text(json.dumps(dict(summaries=summaries,baseline=fingerprint(ROOT),experiment_sha256=hashes,
        input='preloaded stills; fresh inference/proposals/geometry each frame',physical_camera=False,physical_uart=False,transport='PTY',
        production_baseline_changed=False,metric_accuracy_measured=False,assumed_camera_only_for_multi=True,
        local_work_optimized=args.optimized,shared_features=args.shared_features,blue_local=args.blue_local,reject_groups=args.reject_groups,
        crowd_policy='YOLO green union crop; classic supplement on automatic components overlapping at least two boxes' if args.optimized else 'all automatic green components'),indent=2))

if __name__=='__main__':main()
