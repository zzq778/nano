"""Optional ROS image output. One latest frame, no camera or UART ownership."""
import importlib
import threading
import time


def _load_ros():
    return importlib.import_module('rospy'),importlib.import_module('sensor_msgs.msg').Image


class RosImagePublisher:
    def __init__(self,topic,hz=5.):
        if not topic or hz<=0:raise ValueError('image topic and positive rate required')
        self.topic=topic;self.period=1./hz
        self._lock=threading.Lock();self._stop=threading.Event()
        self._latest=None;self._sequence=0;self._ros=None
        self.published=0;self.error=None
        self._worker=threading.Thread(target=self._run,name='ros-image-output',daemon=True)
        self._worker.start()

    def submit(self,image):
        # Capture hands over a new owned BGR array; the pipeline only reads it.
        with self._lock:
            self._sequence+=1
            self._latest=(self._sequence,image,time.time())

    def _run(self):
        publisher=None
        try:
            ros,message_type=_load_ros();self._ros=ros
            if self._stop.is_set():return
            ros.init_node('jetson_vision_image',disable_signals=True)
            publisher=ros.Publisher(self.topic,message_type,queue_size=1,tcp_nodelay=True)
            sent=0
            while not self._stop.is_set() and not ros.is_shutdown():
                started=time.monotonic()
                if publisher.get_num_connections():
                    with self._lock:latest=self._latest
                    if latest is not None and latest[0]!=sent:
                        seq,image,stamp=latest
                        message=message_type()
                        message.header.seq=seq
                        message.header.stamp=ros.Time.from_sec(stamp)
                        message.header.frame_id='astra_rgb_optical_frame'
                        message.height,message.width=image.shape[:2]
                        message.encoding='bgr8';message.is_bigendian=0
                        message.step=message.width*3;message.data=image.tobytes()
                        publisher.publish(message);sent=seq;self.published+=1
                self._stop.wait(max(0.,self.period-(time.monotonic()-started)))
        except Exception as error:
            if not self._stop.is_set():self.error=str(error)
        finally:
            if publisher is not None:publisher.unregister()

    def close(self):
        self._stop.set()
        if self._ros is not None:self._ros.signal_shutdown('vision image output stopped')
        self._worker.join(timeout=2.)

    def diagnostics(self):
        return dict(topic=self.topic,published=self.published,error=self.error,
                    stamp_meaning='host handoff time, not exposure',rate_limit_hz=1./self.period)
