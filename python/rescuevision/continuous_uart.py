"""Serialize result writes and report missing progress without replaying targets."""
import threading
import time
from . import uart


class ContinuousSink:
    def __init__(self,sink,interval=1.,on_packet=None):
        if interval<=0:raise ValueError('positive status interval required')
        self.sink=sink;self.kind=sink.kind;self.interval=interval
        self._lock=threading.Lock();self._stop=threading.Event()
        self._error=None;self._last_write=0.;self.status_packets=0
        self._on_packet=on_packet
        self.recording_error=None
        try:
            self.write(uart.encode([]));self.status_packets+=1
            self._worker=threading.Thread(target=self._report_idle,name='uart-progress-status',daemon=True)
            self._worker.start()
        except BaseException:
            sink.close()
            raise

    def write(self,packet):
        with self._lock:
            if self._error is not None:raise self._error
            result=self.sink.write(packet)
            self._last_write=time.monotonic()
            self._record(packet)
            return result

    def _record(self,packet):
        if self._on_packet is None:return
        try:self._on_packet(packet)
        except Exception as error:
            self.recording_error=str(error)
            self._on_packet=None

    def _report_idle(self):
        while not self._stop.is_set():
            remaining=max(0.,self.interval-(time.monotonic()-self._last_write))
            if self._stop.wait(remaining):return
            with self._lock:
                if self._stop.is_set():return
                if time.monotonic()-self._last_write<self.interval:continue
                try:
                    self.sink.write(uart.encode([]))
                    self._last_write=time.monotonic();self.status_packets+=1
                    self._record(uart.encode([]))
                except Exception as error:
                    self._error=error
                    return

    def close(self):
        self._stop.set();self._worker.join()
        self.sink.close()
        if self._error is not None:raise self._error

    def diagnostics(self):
        with self._lock:
            return dict(status_packets=self.status_packets,recording_error=self.recording_error,
                        transport_error=str(self._error) if self._error is not None else None)
