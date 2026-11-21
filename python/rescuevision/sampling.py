"""Bounded, infrequent, exact input/output sampling after UART write."""
import datetime
import json
import os
import time
from pathlib import Path
import cv2

class SampleRecorder:
    def __init__(self,root,interval,limit,metadata):
        if interval<=0 or limit<1:raise ValueError('positive sample interval and limit required')
        self.root=Path(root)/(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))
        self.interval=interval;self.limit=limit;self.count=0;self.deadline=0.;self.error=None
        self.metadata=metadata;self.created=False

    def maybe_save(self,image,result,packet,now=None):
        now=time.monotonic() if now is None else now
        if self.error is not None or self.count>=self.limit or now<self.deadline:return None
        started=time.perf_counter();self.deadline=now+self.interval
        try:
            if not self.created:
                self.root.mkdir(parents=True,exist_ok=False)
                (self.root/'metadata.json').write_text(json.dumps(self.metadata,indent=2))
                self.created=True
            stem='%03d_seq_%s'%(self.count,result['sequence_local'])
            path=self.root/(stem+'.png')
            if not cv2.imwrite(str(path),image):raise OSError('sample PNG write failed')
            record=dict(record_kind='paired_sample',host_time=datetime.datetime.now().astimezone().isoformat(),
                        image=path.name,sent_uart_hex=packet.hex(),result=result)
            (self.root/(stem+'.json')).write_text(json.dumps(record,indent=2))
            self.count+=1
            return dict(record_kind='sample_saved',path=str(path),sequence_local=result['sequence_local'],
                        count=self.count,limit=self.limit,write_ms=(time.perf_counter()-started)*1000)
        except Exception as error:
            self.error=str(error)
            return dict(record_kind='sampling_disabled',error=self.error)


class AsyncSampleRecorder:
    """Write exact image/result/packet pairs without blocking the detection loop.

    At most one sample is waiting behind the active disk write. Images from the
    camera loop are owned and immutable; copy once at submission so downstream
    callers may safely reuse their arrays. The existing SampleRecorder retains
    responsibility for file names, interval/limit checks, and disk errors.
    """
    def __init__(self,root,interval,limit,metadata):
        import threading
        self.recorder=SampleRecorder(root,interval,limit,metadata)
        self.interval=interval
        self.limit=limit
        self.deadline=0.
        self.accepted=0
        self.dropped_busy=0
        self._error_reported=False
        self._pending=None
        self._closed=False
        self._condition=threading.Condition()
        self._thread=threading.Thread(target=self._write_loop,name='vision-sample-writer',daemon=True)
        self._thread.start()

    def maybe_save(self,image,result,packet,now=None):
        now=time.monotonic() if now is None else now
        with self._condition:
            if self._closed:return None
            if self.recorder.error is not None:
                if self._error_reported:return None
                self._error_reported=True
                return dict(record_kind='sampling_disabled',error=self.recorder.error)
            if self.accepted>=self.limit or now<self.deadline:return None
            self.deadline=now+self.interval
            if self._pending is not None:
                self.dropped_busy+=1
                return dict(record_kind='sample_busy_skip',count=self.accepted)
            # Copy before returning. Neither a slow disk nor a reused caller
            # array can change the frame associated with the transmitted packet.
            self._pending=(image.copy(),result.copy(),bytes(packet),now)
            self.accepted+=1
            self._condition.notify()
            return dict(record_kind='sample_queued',count=self.accepted,limit=self.limit,
                        sequence_local=result['sequence_local'])

    def _write_loop(self):
        while True:
            with self._condition:
                while self._pending is None and not self._closed:self._condition.wait()
                if self._pending is None:return
                sample=self._pending
                self._pending=None
            self.recorder.maybe_save(*sample[:3],now=sample[3])

    def close(self):
        with self._condition:
            self._closed=True
            self._condition.notify_all()
        self._thread.join()
        return dict(record_kind='sample_writer_summary',accepted=self.accepted,
                    saved=self.recorder.count,dropped_busy=self.dropped_busy,
                    error=self.recorder.error)
