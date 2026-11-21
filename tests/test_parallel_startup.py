"""Ordering, fresh-frame warmup and cleanup contracts for parallel startup."""
import contextlib,io,json,sys,tempfile,threading,unittest
from pathlib import Path
from unittest.mock import patch
import test_astra_capture as fixture

class ParallelStartup(unittest.TestCase):
    def run_case(self,model_failure=False,camera_failure=False):
        app=fixture.application();started=threading.Event();release=threading.Event();closed=[]
        class Camera:
            def __init__(self,*args):
                started.set()
                if camera_failure:raise RuntimeError('injected camera constructor failure')
                if model_failure:
                    if not release.wait(2.):raise RuntimeError('test model never released camera')
                self.seq=0
            def get(self,*args):
                self.seq+=1
                return self.seq,fixture.np.zeros((480,640,3),fixture.np.uint8),fixture.time.perf_counter()
            def close(self):closed.append('camera')
        class Pipeline(fixture.GoodPipeline):
            def __init__(self,*args):
                if not started.wait(2.):raise AssertionError('camera was not started alongside model')
                release.set()
                if model_failure:raise RuntimeError('injected model constructor failure')
                super().__init__(*args)
            def close(self):closed.append('model')
        with tempfile.TemporaryDirectory() as directory:
            args=['app.py','--camera','astra','--warmup','1','--max-frames','1','--quiet','--output',directory]
            with patch.object(sys,'argv',args),patch.object(app,'Pipeline',Pipeline),patch.object(app,'LatestCamera',Camera),contextlib.redirect_stderr(io.StringIO()):
                status=app.main()
            rows=[json.loads(line) for line in (Path(directory)/'frames.jsonl').read_text().splitlines()]
            metadata=Path(directory)/'metadata.json'
            meta=json.loads(metadata.read_text()) if metadata.exists() else None
        return status,closed,rows,meta
    def test_camera_starts_before_model_completion_and_warmup_is_not_reused(self):
        status,closed,rows,meta=self.run_case()
        self.assertEqual(status,0);self.assertCountEqual(closed,['camera','model'])
        warmup=[row for row in rows if row.get('record_kind')=='camera_warmup'][0]
        frame=[row for row in rows if 'objects' in row][0]
        self.assertGreater(frame['sequence_local'],warmup['sequence_local'])
        self.assertEqual(meta['startup_timings_ms']['mode'],'parallel_camera_model')
    def test_model_failure_still_closes_camera_created_in_background(self):
        status,closed,rows,_=self.run_case(model_failure=True)
        self.assertEqual(status,2);self.assertEqual(closed,['camera'])
        self.assertIn('injected model',[r for r in rows if r.get('record_kind')=='runtime_error'][0]['error'])
    def test_camera_failure_closes_successfully_created_model(self):
        status,closed,rows,_=self.run_case(camera_failure=True)
        self.assertEqual(status,2);self.assertEqual(closed,['model'])
        self.assertIn('injected camera',[r for r in rows if r.get('record_kind')=='runtime_error'][0]['error'])

if __name__=='__main__':unittest.main()
