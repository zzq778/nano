"""Pixel-only experiments, isolated from the frozen production baseline.

The callable takes RGB evidence and baseline pixel boxes, never scene names/counts.
Parameter substitutions apply to an in-memory copy of the user original; vendor
files are not edited. No per-image parameter selection or count cap is performed.
"""
import importlib.util,json,sys,time,types
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))

class ClassicProposer:
    def __init__(self,options):
        source_path=ROOT/'vendor/topface_lightweight.py';source=source_path.read_text(encoding='utf-8')
        substitutions={
            'area<900':'area<'+str(options.get('minimum_area',900)),
            'w<32':'w<'+str(options.get('minimum_width',32)),
            'h<19':'h<'+str(options.get('minimum_height',19)),
            'q20+.69*(q80-q20)':'q20+'+str(options.get('light_fraction',.69))+'*(q80-q20)',
            '(35,60,25)':'(35,'+str(options.get('saturation',60))+','+str(options.get('value',25))+')',
            '),25,65)':'),'+str(options.get('canny_low',25))+','+str(options.get('canny_high',65))+')',
            'canny_thick=2':'canny_thick='+str(options.get('canny_thick',2)),
            'if len(coords[0])<800':'if len(coords[0])<'+str(options.get('band_minimum_pixels',800)),
            'if ind.sum()<550':'if ind.sum()<'+str(options.get('face_minimum_pixels',550)),
            'GaussianBlur(np.uint8(L),(5,5),0)':'GaussianBlur(np.uint8(L),('+str(options.get('canny_blur',5))+','+str(options.get('canny_blur',5))+'),0)',
            'np.quantile(L[green>0],[.2,.8])':'np.percentile(L[green>0],[20,80])',
        }
        for old,new in substitutions.items():
            if old not in source:raise ValueError('original source changed; re-audit substitution '+old)
            source=source.replace(old,new)
        if options.get('shared_features'):
            old='''    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    L=cv2.cvtColor(im,cv2.COLOR_BGR2LAB)[:,:,0].astype('float32')
    green=cv2.inRange(hsv,(35,60,25),(83,255,255))>0'''
            new='''    if prepared is None:
        hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        L=cv2.cvtColor(im,cv2.COLOR_BGR2LAB)[:,:,0].astype('float32')
        green=cv2.inRange(hsv,(35,60,25),(83,255,255))>0
    else:
        L=prepared[0].astype('float32');green=prepared[1]>0'''
            assert old in source
            source=source.replace(old,new).replace('def split_run(im,level=156,canny_thick=', 'def split_run(im,level=156,prepared=None,canny_thick=')
            source=source.replace('split_run(im,level=level)','split_run(im,level=level,prepared=(L,green))')
        self.source_text=source
        self.module=types.ModuleType('classic_green_experiment');self.module.__file__=str(source_path)
        exec(compile(source,str(source_path),'exec'),self.module.__dict__);self.options=options
    def raw(self,image):return self.module.detect_with_isolated_fallback(image)[0]
    def propose(self,image,boxes):
        if not self.options.get('component_roi'):return self.raw(image)
        h,w=image.shape[:2];green=cv2.inRange(cv2.cvtColor(image,cv2.COLOR_BGR2HSV),(35,60,25),(83,255,255))
        green=cv2.morphologyEx(green,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        n,labels,stats,_=cv2.connectedComponentsWithStats(green,8);result=[]
        for label in range(1,n):
            x,y,bw,bh,area=[int(v) for v in stats[label]]
            if area<150 or bw<8 or bh<8:continue
            local=[]
            for box in boxes:
                a,b,c,d=box;intersection=max(0,min(x+bw,c)-max(x,a))*max(0,min(y+bh,d)-max(y,b))
                if intersection/max(1,min(bw*bh,(c-a)*(d-b)))>.15:local.append(c-a)
            if self.options.get('crowd_only') and len(local)<2:continue
            expected=float(np.median(local)) if local else float(max(bw,bh))
            scale=float(np.clip(self.options.get('target_width',80)/max(8,expected),.5,4))
            pad=max(6,int(.12*max(bw,bh)));a,b=max(0,x-pad),max(0,y-pad);c,d=min(w,x+bw+pad),min(h,y+bh+pad)
            crop=cv2.resize(image[b:d,a:c],None,fx=scale,fy=scale,interpolation=cv2.INTER_LINEAR)
            for seed in self.raw(crop):
                q=dict(seed);q['x']=float(seed['x']/scale+a);q['y']=float(seed['y']/scale+b)
                q['x_span']=[float(z/scale+a) for z in seed['x_span']];q['y_span']=[float(z/scale+b) for z in seed['y_span']]
                cx,cy=int(round(q['x'])),int(round(q['y']))
                # Retain proposals belonging to this automatically detected component only.
                xa,xb=[int(round(v)) for v in q['x_span']];ya,yb=[int(round(v)) for v in q['y_span']]
                yy,xx=np.nonzero(labels[max(0,ya):min(h,yb+1),max(0,xa):min(w,xb+1)]==label)
                if not len(xx):continue
                xx=xx+max(0,xa);yy=yy+max(0,ya);distance=(xx-q['x'])**2+(yy-q['y'])**2;nearest=int(np.argmin(distance))
                if distance[nearest]>(.35*max(3,xb-xa,yb-ya))**2:continue
                if not(0<=cx<w and 0<=cy<h and labels[cy,cx]==label):q['x'],q['y']=float(xx[nearest]),float(yy[nearest])
                q.update(source_comp=[x,y,bw,bh],component_label=label,proposal_scale=scale,method=seed.get('method','original_topface_on_auto_component_roi'))
                result.append(q)
        return result

def main():
    project=ROOT.parent;dataset=project/'outputs/multi_block_20261028';out=dataset/'green_sensitivity';out.mkdir(exist_ok=True)
    manifest=json.loads((dataset/'manifest.json').read_text());review=json.loads((dataset/'human_green_review/pending_labels.json').read_text())
    yolo={r['image_id']:r for r in map(json.loads,(dataset/'nano/output/multi_instance_blue/detections.jsonl').read_text().splitlines())}
    inputs=[]
    for row in review['measurements']:
        r=manifest['images'][row['image_id']-1];p=Path(manifest['source'])/r['path']
        inputs.append((row['review_id'],cv2.imdecode(np.frombuffer(p.read_bytes(),np.uint8),cv2.IMREAD_COLOR),[o['bbox'] for o in yolo[row['image_id']]['objects'] if o['type']==0]))
    old=project/'outputs/jetson_project_20261027/single_baseline/output/single_baseline_yolo_render/frames.jsonl'
    original=[r for r in map(json.loads,old.read_text().splitlines()) if 'objects' in r]
    for row in original:
        inputs.append(('S%d'%row['image_id'],cv2.imread(str(project/'outputs/approved_six_20261027/raw'/('%02d.png'%row['image_id']))),[o['bbox'] for o in row['objects'] if o['type']==0]))
    variants={
      'original':{},'L45':dict(light_fraction=.45),'L55':dict(light_fraction=.55),'L80':dict(light_fraction=.80),
      'area450':dict(minimum_area=450),'area225':dict(minimum_area=225),'minWHhalf':dict(minimum_width=16,minimum_height=10),
      'S35':dict(saturation=35),'V15':dict(value=15),'canny15_40':dict(canny_low=15,canny_high=40),
      'roi80':dict(component_roi=True,target_width=80),'roi96':dict(component_roi=True,target_width=96),
      'roi80_area450':dict(component_roi=True,target_width=80,minimum_area=450),
      'roi80_L55':dict(component_roi=True,target_width=80,light_fraction=.55),
    }
    with (out/'predictions.jsonl').open('w',buffering=1) as log:
        for name,options in variants.items():
            proposer=ClassicProposer(options);counts=[]
            for key,image,boxes in inputs:
                begin=time.perf_counter();props=proposer.propose(image,boxes);elapsed=(time.perf_counter()-begin)*1000
                log.write(json.dumps(dict(variant=name,parameters=options,image_key=key,count=len(props),proposals=props,host_ms=elapsed,calibration_applied=False))+'\n');counts.append(len(props))
            print(name,counts,flush=True)
    (out/'experiment.json').write_text(json.dumps(dict(variants=variants,input_keys=[r[0] for r in inputs],baseline_commit='e2d120f',counts_never_passed_to_proposer=True,metric_coordinates_computed=False,host_timings_not_Nano=True),indent=2))

if __name__=='__main__':main()
