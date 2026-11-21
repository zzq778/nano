"""Latest-frame capture and bounded POSIX serial I/O, no queues of old results."""
import os
import ctypes
import select
import multiprocessing
import queue
import time
import numpy as np
from .astra import NativeAstra
from . import uart
from .packet_snapshot import PacketRecorder

def packet_for_age(result,age_ms,budget_ms):
    """Record latency without starving UART of newly completed valid results.

    Capture keeps only the latest input and the single processing loop sends each
    completed result once. A compute budget is not a protocol validity deadline.
    Unavailable observations are represented by ordinary zero-target packets.
    """
    result['result_age_ms']=age_ms
    result['result_age_over_budget']=age_ms>budget_ms
    return bytes.fromhex(result['uart_hex'])

class SerialSink:
    kind='physical_uart'
    def __init__(self,device,baud=230400,timeout=.05):
        if os.name!='posix':raise RuntimeError('physical serial mode requires Jetson/Linux')
        import termios
        if baud!=230400:raise ValueError('revised protocol requires 230400 baud')
        self.fd=os.open(str(device),os.O_RDWR|os.O_NOCTTY|os.O_NONBLOCK);self.timeout=timeout
        try:
            attrs=termios.tcgetattr(self.fd)
            attrs[0]=0;attrs[1]=0;attrs[2]=termios.CLOCAL|termios.CREAD|termios.CS8;attrs[3]=0
            attrs[4]=termios.B230400;attrs[5]=termios.B230400
            attrs[6][termios.VMIN]=0;attrs[6][termios.VTIME]=0
            termios.tcsetattr(self.fd,termios.TCSANOW,attrs)
        except Exception:os.close(self.fd);raise
    def write(self,packet):
        start=time.perf_counter();offset=0
        while offset<len(packet):
            remaining=self.timeout-(time.perf_counter()-start)
            if remaining<=0:raise TimeoutError('serial partial-write deadline exceeded')
            if not select.select([],[self.fd],[],remaining)[1]:raise TimeoutError('serial not writable')
            try:written=os.write(self.fd,packet[offset:])
            except BlockingIOError:continue
            if written<=0:raise OSError('serial write made no progress')
            offset+=written
        return dict(bytes=offset,write_ms=(time.perf_counter()-start)*1000,kind=self.kind,wire_drain_measured=False)
    def close(self):
        if self.fd is not None:os.close(self.fd);self.fd=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

class LoopbackSink(SerialSink):
    kind='posix_pty_loopback_not_physical_wire'
    def __init__(self):
        import pty
        self.master,slave=pty.openpty()
        try:super().__init__(os.ttyname(slave))
        except Exception:os.close(self.master);raise
        finally:os.close(slave)
    def write(self,packet):
        result=super().write(packet);received=bytearray();end=time.perf_counter()+self.timeout
        while len(received)<len(packet):
            remaining=end-time.perf_counter()
            if remaining<=0 or not select.select([self.master],[],[],remaining)[0]:raise TimeoutError('loopback read deadline exceeded')
            received.extend(os.read(self.master,len(packet)-len(received)))
        if bytes(received)!=packet:raise ValueError('loopback packet differs')
        uart.decode(bytes(received));result['loopback_verified']=True
        return result
    def close(self):
        super().close()
        if self.master is not None:os.close(self.master);self.master=None

def _camera_reader(source,width,height,buffer,sequence,timestamp,frame_lock,stop,ready,failed,errors):
    """One CPU-only process owns all Astra SDK calls, including shutdown."""
    capture=None
    parent_pid=os.getppid()
    try:
        capture=NativeAstra(source['library'],source['uri'],width,height);ready.value=True
        shared=np.frombuffer(buffer,dtype=np.uint8).reshape(height,width,3)
        while not stop.value and os.getppid()==parent_pid:
            try:image=capture.read()
            except TimeoutError:continue
            stamp=time.perf_counter()
            if image.shape!=(height,width,3) or image.dtype!=np.uint8:raise RuntimeError('camera did not supply the configured uint8 BGR size')
            if not frame_lock.acquire(timeout=.2):continue
            try:
                np.copyto(shared,image);sequence.value+=1;timestamp.value=stamp
            finally:frame_lock.release()
    except Exception as error:
        try:errors.put_nowait(str(error)[:256])
        except queue.Full:pass
        failed.value=True;ready.value=True
    finally:
        if capture is not None:capture.close()

def stop_capture_process(process,stop,timeout=1.):
    stop.value=True;process.join(timeout)
    if process.is_alive():process.terminate();process.join(timeout)
    if process.is_alive():
        import signal
        os.kill(process.pid,signal.SIGKILL);process.join(timeout)
    if process.is_alive():raise RuntimeError('capture process did not terminate')

class LatestCamera:
    """Single shared latest frame; spawn avoids inheriting the CUDA context."""
    def __init__(self,source,width,height,library):
        ctx=multiprocessing.get_context('spawn');self.width=width;self.height=height;self.closed=False;self._failure=None
        self.buffer=ctx.RawArray('B',width*height*3);self.sequence=ctx.Value(ctypes.c_int64,0,lock=False);self.timestamp=ctx.Value('d',0,lock=False)
        self.frame_lock=ctx.Lock()
        self.stop=ctx.RawValue(ctypes.c_bool,False);self.ready=ctx.RawValue(ctypes.c_bool,False);self.failed=ctx.RawValue(ctypes.c_bool,False)
        self.errors=ctx.Queue(1)
        descriptor=dict(uri=source,library=str(library))
        self.process=ctx.Process(target=_camera_reader,args=(descriptor,width,height,self.buffer,self.sequence,self.timestamp,self.frame_lock,self.stop,self.ready,self.failed,self.errors),daemon=True)
        self.process.start()
        startup_deadline=time.perf_counter()+10.
        while not self.ready.value and self.process.is_alive() and time.perf_counter()<startup_deadline:time.sleep(.01)
        if not self.ready.value or self.failed.value or not self.process.is_alive():
            self._failure=self._error('camera startup deadline or open failure')
            stop_capture_process(self.process,self.stop);self.failed.value=True
    def _error(self,fallback):
        try:return self.errors.get(timeout=.05)
        except queue.Empty:return fallback
    def get(self,after_sequence,timeout=1.):
        end=time.perf_counter()+timeout
        while True:
            remaining=end-time.perf_counter()
            if remaining<=0:raise TimeoutError('no fresh camera frame')
            acquired=self.frame_lock.acquire(timeout=remaining)
            if acquired:
                try:
                    if self.sequence.value>after_sequence:
                        image=np.frombuffer(self.buffer,dtype=np.uint8).reshape(self.height,self.width,3).copy()
                        return self.sequence.value,image,self.timestamp.value
                finally:self.frame_lock.release()
            if not self.process.is_alive() or self.failed.value:
                if self._failure is None:self._failure=self._error('camera process exited')
                remaining=end-time.perf_counter()
                if remaining>0:time.sleep(remaining)
                raise RuntimeError(self._failure)
            if not acquired:raise TimeoutError('camera publication lock deadline')
            # Condition.wait reacquires an interprocess lock without a deadline.
            # Poll only while no newer frame exists, with bounded lock acquisition.
            time.sleep(min(.005,max(0.,end-time.perf_counter())))
    def close(self):
        if not self.closed:
            self.closed=True
            try:stop_capture_process(self.process,self.stop)
            finally:self.errors.close();self.errors.join_thread()
