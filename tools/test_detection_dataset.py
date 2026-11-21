"""Uncalibrated dataset evaluation. Existing labels are intentionally never loaded."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from rescuevision.native import NativeDetector
from rescuevision.frontends import YoloFrontend

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--no-blue-supplement',action='store_true');p.add_argument('--render-step',type=int,default=25);args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text());model=Path(cfg['model']);model=model if model.is_absolute() else ROOT/model
    assert hashlib.sha256(model.read_bytes()).hexdigest()==cfg['model_sha256'],'model identity'
    manifest_path=Path(args.manifest);manifest=json.loads(manifest_path.read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'frames').mkdir(exist_ok=True);times=[];records=[];start=time.perf_counter()
    detector=NativeDetector(ROOT/cfg['native_detector'],model,cfg['semantic_thresholds'],cfg['nms_iou'])
    frontend=YoloFrontend(detector,None if args.no_blue_supplement else cfg.get('blue_context_supplement'))
    try:
        first=cv2.imread(str(manifest_path.parent/manifest['images'][0]['path']))
        for _ in range(3):frontend.detect(first)
        with (out/'detections.jsonl').open('w',buffering=1) as log:
            for i,row in enumerate(manifest['images']):
                im=cv2.imread(str(manifest_path.parent/row['path']))
                if hashlib.sha256(cv2.cvtColor(im,cv2.COLOR_BGR2RGB).tobytes()).hexdigest()!=row['rgb_sha256']:raise ValueError('dataset identity')
                begin=time.perf_counter();objects=frontend.detect(im);elapsed=(time.perf_counter()-begin)*1000;times.append(elapsed)
                record=dict(image_id=row['id'],path=row['path'],source_video_group=row['source_video_group'],objects=objects,
                            counts={str(t):sum(o['type']==t for o in objects) for t in range(4)},detection_ms=elapsed,
                            metric_positions_computed=False,calibration_applied=False,ground_truth_available=False)
                log.write(json.dumps(record,separators=(',',':'))+'\n');records.append(record)
                if args.render_step>0 and i%args.render_step==0:
                    preview=im.copy();colors=[(30,235,30),(240,240,240),(20,150,255),(255,170,40)]
                    for n,o in enumerate(objects):
                        x0,y0,x1,y1=[int(v) for v in o['bbox']];cv2.rectangle(preview,(x0,y0),(x1,y1),colors[o['type']],2)
                        cv2.putText(preview,'GKOB'[o['type']]+str(n),(x0,max(20,y0-3)),cv2.FONT_HERSHEY_SIMPLEX,.65,colors[o['type']],2)
                    cv2.imwrite(str(out/'frames'/('%03d.jpg'%row['id'])),preview)
        report=dict(images=len(records),source=str(manifest_path),model_sha256=cfg['model_sha256'],blue_supplement=not args.no_blue_supplement,
                    detection_p50_ms=float(np.percentile(times,50)),detection_p95_ms=float(np.percentile(times,95)),
                    detector_only_hz=1000./float(np.mean(times)),wall_including_setup_decode_hash_render_log_ms=(time.perf_counter()-start)*1000,
                    total_predictions={str(t):sum(r['counts'][str(t)] for r in records) for t in range(4)},
                    accuracy_claim='Not scored: existing labels are machine preannotations; manual truth unavailable',metric_positions_computed=False)
        (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    finally:detector.close()

if __name__=='__main__':main()
