"""Current RGB top-face evidence supplements instances and rejects proven group boxes."""
import json,time
from .native import NativeGreen
from .topface_native import NativeTopFace
from .frontends import green_observation
from .green_proposals import GreenFaceProposer
from .green_regions import propose_observed_region,refine_local,added_support_local
from .green_association import fuse
from .scheduling import group_veto


class GreenCrowdFrontend:
    def __init__(self,base,root,library):
        self.base=base;self.coverage=base.coverage;self.last_diagnostics={}
        self._topface=NativeTopFace(library)
        self.proposer=GreenFaceProposer(self._topface.detect)
        cfg=json.loads((root/'vendor/config.json').read_text())['green_thresholds']
        cfg['back_expand_width']=.08
        self.solver=NativeGreen(library,cfg)

    def detect(self,image):
        start=time.perf_counter();objects=self.base.detect(image);detected=time.perf_counter()
        green=[o for o in objects if o['type']==0]
        self.last_hsv=getattr(self.base,'last_hsv',None)
        seeds=propose_observed_region(self.proposer,image,[o['bbox'] for o in green],self.last_hsv);proposed=time.perf_counter()
        results=refine_local(self.solver,image,seeds);group_veto(results,seeds)
        evidence=[dict(seed=s,result=r) for s,r in zip(seeds,results)];refined=time.perf_counter()
        association=fuse(green,evidence,reject_groups=True)
        parents=[green[g['yolo']] for g in association['rejected_yolo_groups']]
        retained=[o for o in objects if o not in parents]
        support_green=[o for o in green if o not in parents]
        additions=[]
        for face in association['added_faces']:
            q=face['evidence'];obj=green_observation(q['seed'])
            obj.update(top_face_anchor_px=q['result']['center'],center_px=q['result']['center'],
                       center_kind='two_rim_topface_anchor_not_ground_center')
            additions.append(obj)
        supports=added_support_local(image,support_green,additions)
        for obj in supports:obj['source']='classic_local_topface_two_rims_visible_support'
        retained+=supports
        self.last_diagnostics=dict(
            input_yolo_green=len(green),retained_green=sum(o['type']==0 for o in retained),
            rejected_yolo_groups=[dict(bbox=o['bbox'],confidence=o.get('confidence'),source=o['source'],yolo_green_index=g['yolo'],
                reason=g['reason'],independent_top_face_anchors=[evidence[i]['result']['center'] for i in g['independent_top_faces']])
                for o,g in zip(parents,association['rejected_yolo_groups'])],
            added_individual_top_face_anchors=[o['top_face_anchor_px'] for o in supports],
            group_boxes_counted_as_individuals=False,
            timings_ms=dict(base=(detected-start)*1000,classic=(proposed-detected)*1000,
                native_rims=(refined-proposed)*1000,association_and_support=(time.perf_counter()-refined)*1000),
            yolo_diagnostics=getattr(self.base,'last_diagnostics',{}))
        return retained
