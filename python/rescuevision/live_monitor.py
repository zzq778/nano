"""Read-only, paired camera/UART observer. Never opens camera or serial devices."""
import base64,json,math,threading,time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer
from socketserver import ThreadingMixIn
import cv2
import numpy as np
from . import uart

NAMES=['绿色物块','黑色物块','橙色物块','蓝色物块','物资区','伤员区','物资区','伤员区']

def project(geometry,x,y):
    optical=geometry.R.T.dot(np.array([x,y,geometry.ground])-geometry.t)
    if not np.isfinite(optical).all() or optical[2]<=.01:return None
    if getattr(geometry,'use_pinhole',True):
        uv=[geometry.K[0,0]*optical[0]/optical[2]+geometry.K[0,2],geometry.K[1,1]*optical[1]/optical[2]+geometry.K[1,2]]
    else:
        uv=cv2.projectPoints(optical.reshape(1,3),np.zeros(3),np.zeros(3),geometry.K,geometry.D)[0].reshape(2)
    return [float(v) for v in uv] if all(math.isfinite(v) for v in uv) else None

def sent_targets(packet,result,geometry):
    targets=[];used=set()
    for index,t in enumerate(uart.decode(packet)):
        row=dict(t,index=index+1,name=NAMES[t['type']],yaw_cdeg=t.get('yaw_cdeg',0))
        row['ground_px']=project(geometry,t['x_mm']/1000.,t['y_mm']/1000.) if geometry.ready else None
        if t['type']==2:
            for i,obj in enumerate(result.get('objects',[])):
                m=obj.get('metric',{})
                if i in used or obj.get('type')!=2 or (m.get('x_mm'),m.get('y_mm'))!=(t['x_mm'],t['y_mm']):continue
                used.add(i);angle=m.get('yaw_world_deg')
                if isinstance(angle,(int,float)) and math.isfinite(angle):
                    row.update(estimated_yaw_deg=angle,pose_valid=bool(m.get('pose_valid')),pose_quality=m.get('pose_quality'),yaw_source=m.get('yaw_source'))
                    a=math.radians(angle)
                    row['arrow_px']=project(geometry,t['x_mm']/1000.+.08*math.cos(a),t['y_mm']/1000.+.08*math.sin(a))
                break
        targets.append(row)
    return targets

class Server(ThreadingMixIn,HTTPServer):
    daemon_threads=True

class LiveMonitor:
    def __init__(self,port=8767,hz=5):
        self.lock=threading.Lock();self.event=threading.Event();self.stop=threading.Event()
        self.pending=None;self.frame=None;self.last_tx=None;self.serial=0;self.error=None;self.dropped=0
        self.interval=1./hz;owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                if self.path=='/':body=Path(__file__).with_suffix('.html').read_bytes();kind='text/html; charset=utf-8'
                elif self.path=='/state':
                    with owner.lock:
                        state=dict(frame=owner.frame,last_tx=owner.last_tx,error=owner.error,dropped=owner.dropped)
                        state['frame_age_ms']=(time.monotonic()-owner.frame['at'])*1000 if owner.frame else None
                        state['tx_age_ms']=(time.monotonic()-owner.last_tx['at'])*1000 if owner.last_tx else None
                    body=json.dumps(state,allow_nan=False).encode();kind='application/json'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
        self.server=Server(('127.0.0.1',port),Handler)
        self.http=threading.Thread(target=self.server.serve_forever,daemon=True);self.http.start()
        self.worker=threading.Thread(target=self._work,daemon=True);self.worker.start()

    def on_packet(self,packet):
        # Called by ContinuousSink only after successful write, including idle packets.
        with self.lock:
            self.serial+=1;self.last_tx=dict(id=self.serial,hex=packet.hex(),at=time.monotonic(),count=packet[2])

    def submit(self,image,result,packet,geometry):
        try:
            transport=result.get('transport',{})
            if transport.get('kind')!='physical_uart' or transport.get('bytes')!=len(packet):return
            with self.lock:
                if not self.last_tx or self.last_tx['hex']!=packet.hex():return
                tx=dict(self.last_tx)
                if self.pending is not None:self.dropped+=1
                # LatestCamera.get() returns an owned image. The processing
                # loop never mutates it, so keeping this reference avoids a
                # 640x480x3 copy on every monitored frame.
                self.pending=(image,result,packet,geometry,tx,time.monotonic())
            self.event.set()
        except Exception as ex:self.error=repr(ex)

    def _work(self):
        while not self.stop.is_set():
            self.event.wait(.5);self.event.clear()
            with self.lock:item=self.pending;self.pending=None
            if item is None:continue
            try:
                image,result,packet,geometry,tx,at=item
                ok,jpg=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85])
                if not ok:raise RuntimeError('JPEG encoding failed')
                frame=dict(at=at,tx_id=tx['id'],packet_hex=packet.hex(),sequence=result.get('sequence_local'),
                           jpeg=base64.b64encode(jpg).decode('ascii'),width=image.shape[1],height=image.shape[0],
                           targets=sent_targets(packet,result,geometry),transport=result['transport'],
                           compute_ms=result.get('timings_ms',{}).get('compute'),
                           calibration_status=result.get('calibration_status'),host_time=time.strftime('%Y-%m-%d %H:%M:%S'))
                with self.lock:self.frame=frame
            except Exception as ex:self.error=repr(ex)
            self.stop.wait(self.interval)

    def close(self):
        self.stop.set();self.event.set();self.server.shutdown();self.server.server_close()
        self.worker.join(timeout=2);self.http.join(timeout=2)
