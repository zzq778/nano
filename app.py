#!/usr/bin/env python3
"""Standalone Jetson vision: explicit replay/camera source, diagnostics and optional UART."""
import time
MODULE_START=time.perf_counter()
import argparse
import hashlib
import json
import signal
import sys
from collections import deque
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'python'))
import cv2
from rescuevision.pipeline import Pipeline
from rescuevision.runtime import LatestCamera,SerialSink,LoopbackSink,packet_for_age,PacketRecorder
from rescuevision.drawing import annotate
from rescuevision.pose_preview import annotate_pose
from rescuevision import uart
from rescuevision.continuous_uart import ContinuousSink
from rescuevision.ros_image import RosImagePublisher
from rescuevision.sampling import AsyncSampleRecorder
from rescuevision.baseline import fingerprint

IMPORT_READY=time.perf_counter()

def latency_quantiles(samples):
    """Nearest-rank P95 and median from a bounded recent sample window, in ms."""
    if not samples:return None
    values=sorted(samples)
    n=len(values)
    return dict(p50_ms=round((values[(n-1)//2]+values[n//2])/2,3),
                p95_ms=round(values[(95*n+99)//100-1],3),samples=n)


def _stop_signal(signum,frame):
    raise KeyboardInterrupt('termination signal %d'%signum)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'configs/runtime.json'))
    parser.add_argument('--mode','--experiment-frontend',dest='mode',choices=['color','yolo'],help='explicit frontend ablation; default comes from the single baseline config')
    parser.add_argument('--calibration',help='camera/data binding only; does not select another algorithm implementation')
    parser.add_argument('--green-backend',choices=['native','python'],help='explicit reference backend for desktop debugging')
    sources=parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--manifest');sources.add_argument('--video');sources.add_argument('--camera')
    parser.add_argument('--output',required=True);parser.add_argument('--repeat',type=int,default=1)
    parser.add_argument('--warmup',type=int,default=0);parser.add_argument('--max-frames',type=int,default=0)
    parser.add_argument('--rate-hz',type=float,default=0);parser.add_argument('--render',action='store_true')
    parser.add_argument('--preload-replay',action='store_true',help='explicit in-memory BGR replay; not a camera latency test')
    parser.add_argument('--sample-interval',type=float,default=0.,help='seconds between exact paired input/output samples; zero disables')
    parser.add_argument('--sample-limit',type=int,default=120)
    parser.add_argument('--ros-image-topic',help='optional raw ROS Image output from the existing Astra capture')
    parser.add_argument('--ros-image-hz',type=float,default=5.)
    parser.add_argument('--quiet',action='store_true',help='keep JSON evidence without per-frame console output')
    parser.add_argument('--no-frame-log',action='store_true',help='production mode: keep errors/metadata/summary without full per-frame JSON')
    parser.add_argument('--packet-snapshot-per-frame',action='store_true',help='diagnostic I/O ablation; default snapshot is saved on shutdown')
    serial=parser.add_mutually_exclusive_group();serial.add_argument('--uart-device');serial.add_argument('--uart-loopback',action='store_true')
    args=parser.parse_args()
    if args.sample_interval<0 or args.sample_limit<1:parser.error('invalid sampling interval or limit')
    if args.ros_image_topic and (args.camera is None or args.ros_image_hz<=0):parser.error('ROS image output requires camera input and a positive rate')
    if args.repeat<1 or args.warmup<0 or args.max_frames<0 or args.rate_hz<0:parser.error('invalid repeat/warmup/frame/rate value')
    if args.camera is not None and args.camera!='astra' and not args.camera.startswith('2bc5/0402@'):
        parser.error('camera source must be astra or an explicit Astra device URI')
    if args.preload_replay and not args.manifest:parser.error('preload applies only to explicit still-image replay')
    config=json.loads(Path(args.config).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if config['uart_baud']!=230400:parser.error('revised UART protocol requires 230400 baud')
    args.mode=args.mode or config.get('frontend','yolo')
    if args.calibration:config['calibration']=args.calibration
    if args.green_backend:config['green_backend']=args.green_backend
    calibration_scope=json.loads((ROOT/config['calibration']).read_text())
    if calibration_scope.get('simulation_only') and (not args.manifest or args.uart_device):
        parser.error('assumed-camera stress configuration requires recorded images and excludes physical UART')
    if calibration_scope.get('scope')=='approved_astra_six_only' and not args.manifest:
        parser.error('Astra six-image calibration profile is bound to approved still images; do not apply it to the replacement USB camera')
    if args.render:(out/'frames').mkdir(exist_ok=True)
    boot=time.perf_counter();pipeline=None;sink=None;camera=None;capture=None;exit_code=0
    count=0;loop_start=None;image_publisher=None;samples=None;monitor=None
    # Fixed window, so production runs never accumulate unbounded profiling data.
    latency_names=('capture_handoff_age','app_write_result_age','source_wait',
                   'pipeline_compute','app_serial_write','frame_to_write')
    latencies={name:deque(maxlen=1024) for name in latency_names}
    stage_names=('detection','center_only','scheduled_refinement_and_veto',
                 'metric_validation','uart_encode','native_detector',
                 'hsv_and_blue','orange_recovery','green_classic',
                 'green_native_rims','green_association_and_support',
                 'native_preprocess','native_tensorrt_and_transfers','native_postprocess',
                 'monitor_submit','frame_json_log','sample_submit')
    stages={name:deque(maxlen=1024) for name in stage_names}
    camera_skipped_total=0
    startup=dict(module_import_ms=(IMPORT_READY-MODULE_START)*1000,mode='parallel_camera_model' if args.camera is not None else 'model_only')
    startup_pool=None;startup_future=None
    def open_camera():
        start=time.perf_counter();startup['camera_start_since_boot_ms']=(start-boot)*1000
        try:return LatestCamera(args.camera,*calibration_scope['image_size'],ROOT/config['native_camera'])
        finally:
            startup['camera_constructor_ms']=(time.perf_counter()-start)*1000
            startup['camera_ready_since_boot_ms']=(time.perf_counter()-boot)*1000

    camera_warmup_sequence=0;camera_warmup_attempts=0
    packet_recorder=PacketRecorder(out/'last_packet.bin',args.packet_snapshot_per_frame)
    def record_packet(packet):
        if sink is None:packet_recorder.record(packet)
    def observe_packet(packet):
        packet_recorder.record(packet)
        if monitor:monitor.on_packet(packet)
    log=(out/'frames.jsonl').open('w',buffering=1)
    termination_handlers={}
    for name in ('SIGTERM','SIGHUP'):
        signum=getattr(signal,name,None)
        if signum is not None:termination_handlers[signum]=signal.signal(signum,_stop_signal)
    try:
        if args.camera and args.uart_device and config.get('live_monitor',{}).get('enabled',False):
            try:
                from rescuevision.live_monitor import LiveMonitor
                monitor=LiveMonitor(port=config['live_monitor'].get('port',8767),hz=config['live_monitor'].get('hz',5))
            except Exception as monitor_error:
                print('live monitor unavailable:',monitor_error,file=sys.stderr)
        if args.uart_device:sink=SerialSink(args.uart_device,config['uart_baud'])
        elif args.uart_loopback:sink=LoopbackSink()
        if sink:sink=ContinuousSink(sink,on_packet=observe_packet)
        startup['uart_ready_since_boot_ms']=(time.perf_counter()-boot)*1000
        if args.ros_image_topic:image_publisher=RosImagePublisher(args.ros_image_topic,args.ros_image_hz)
        if args.camera is not None:
            from concurrent.futures import ThreadPoolExecutor
            startup_pool=ThreadPoolExecutor(max_workers=1)
            startup_future=startup_pool.submit(open_camera)
        model_start=time.perf_counter()
        pipeline=Pipeline(ROOT,config,args.mode)
        startup['pipeline_init_ms']=(time.perf_counter()-model_start)*1000
        startup['model_ready_since_boot_ms']=(time.perf_counter()-boot)*1000
        startup['native_detector']=getattr(getattr(pipeline,'detector',None),'startup_timings_ms',{})

        image_rows=[]
        if args.manifest:
            manifest_path=Path(args.manifest);manifest=json.loads(manifest_path.read_text())
            cached_bytes=0
            for row in manifest['images']:
                allowed=calibration_scope.get('allowed_rgb_sha256')
                if allowed and row['rgb_sha256'] not in allowed:raise ValueError('input is outside the calibration-bound Astra dataset')
                path=manifest_path.parent/row['path'];im=cv2.imread(str(path))
                if im is None:raise ValueError('input cannot be decoded: '+str(path))
                if hashlib.sha256(cv2.cvtColor(im,cv2.COLOR_BGR2RGB).tobytes()).hexdigest()!=row['rgb_sha256']:raise ValueError('input pixel identity mismatch')
                cached_bytes+=im.nbytes if args.preload_replay else 0
                if cached_bytes>64*1024*1024:raise ValueError('preloaded replay exceeds the explicit 64MiB memory limit')
                image_rows.append((row['id'],path,im if args.preload_replay else None))
            # Warmup consumes only the exact approved image files, not synthetic/other images.
            for _ in range(args.warmup):
                for _,path,cached in image_rows:pipeline.process(cached if cached is not None else cv2.imread(str(path)))
        elif args.camera is not None:
            camera=startup_future.result()
            for warmup_index in range(args.warmup):
                started=time.perf_counter();record=dict(record_kind='camera_warmup',index=warmup_index)
                try:
                    read_start=time.perf_counter()
                    sequence,image,_=camera.get(camera_warmup_sequence)
                    record['fresh_frame_read_ms']=(time.perf_counter()-read_start)*1000
                    process_start=time.perf_counter()
                    camera_warmup_sequence=sequence;record['sequence_local']=sequence
                    pipeline.process(image)
                    record['pipeline_warmup_ms']=(time.perf_counter()-process_start)*1000
                except (TimeoutError,RuntimeError,ValueError,cv2.error) as error:
                    record['error']=str(error)
                packet=uart.encode([]);record['uart_hex']=packet.hex()
                if sink:record['transport']=sink.write(packet)
                record_packet(packet);camera_warmup_attempts+=1
                record['warmup_ms']=(time.perf_counter()-started)*1000
                log.write(json.dumps(record)+'\n')
        else:
            capture=cv2.VideoCapture(args.video)
            if not capture.isOpened():raise RuntimeError('video open failed')
        metadata=dict(mode=args.mode,config=config,calibration=pipeline.geometry.config,
                      baseline=fingerprint(ROOT),active_config_path=str(Path(args.config).resolve()),frontend_override=args.mode!=config.get('frontend','yolo'),
                      startup_and_warmup_ms=(time.perf_counter()-boot)*1000,startup_timings_ms=startup,python=sys.version,opencv=cv2.__version__,
                      camera_warmup_attempts=camera_warmup_attempts,
                      transport=sink.kind if sink else 'disabled',camera_exposure_latency_measured=False,
                      preload_replay=args.preload_replay,preloaded_bgr_bytes=sum(c.nbytes for _,_,c in image_rows if c is not None),
                      per_frame_packet_snapshot=args.packet_snapshot_per_frame,quiet=args.quiet,
                      sampling_timestamp='camera read completion; offline replay start for still images',
                      source=str(args.manifest or args.video or args.camera))
        (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
        if args.sample_interval:samples=AsyncSampleRecorder(out/'samples',args.sample_interval,args.sample_limit,metadata)
        count=0;consecutive_camera_gaps=0;last_sequence=camera_warmup_sequence;deadline=time.perf_counter();loop_start=deadline
        for repeat in range(args.repeat):
            source_index=0
            while True:
                if args.max_frames and count>=args.max_frames:break
                if image_rows and source_index>=len(image_rows):break
                frame_start=time.perf_counter()
                if image_rows:
                    image_id,path,cached=image_rows[source_index];sample_time=frame_start;image=cached if cached is not None else cv2.imread(str(path));sequence=count+1
                    source_kind='preloaded_still_replay_not_camera' if cached is not None else 'repeated_still_image_replay'
                elif args.camera is not None:
                    try:
                        sequence,image,sample_time=camera.get(last_sequence)
                        consecutive_camera_gaps=0
                    except (TimeoutError,RuntimeError) as error:
                        consecutive_camera_gaps+=1
                        event=dict(record_kind='source_frame_drop',error=str(error),consecutive_camera_gaps=consecutive_camera_gaps)
                        if sink:event['transport']=sink.write(uart.encode([]))
                        record_packet(uart.encode([]))
                        log.write(json.dumps(event)+'\n')
                        if getattr(getattr(camera,'failed',None),'value',False) or consecutive_camera_gaps>=3:
                            camera.close()
                            camera=open_camera()
                            last_sequence=0;consecutive_camera_gaps=0
                            log.write(json.dumps(dict(record_kind='camera_reopened',source_sequence_reset=True,
                                startup_failed=bool(camera.failed.value)))+'\n')
                        continue
                    image_id=sequence;source_kind='camera_read_completion_not_exposure'
                else:
                    ok,image=capture.read();sample_time=time.perf_counter();sequence=count+1;image_id=sequence;source_kind='video_decode_completion_not_exposure'
                    if not ok:break
                if image_publisher:image_publisher.submit(image)
                read_done=time.perf_counter()
                try:
                    result=pipeline.process(image)
                except (cv2.error,ValueError,RuntimeError) as error:
                    # Diagnose the failed frame and continue with the same open port.
                    result=dict(objects=[],counts=None,frontend_diagnostics={},
                                timings_ms=dict(compute=(time.perf_counter()-read_done)*1000),
                                uart_hex=uart.encode([]).hex(),
                                diagnostics=['frame_processing_failed'],
                                frame_error=dict(type=type(error).__name__,message=str(error)))
                missed_frames=max(0,sequence-last_sequence-1) if camera else 0
                camera_skipped_total+=missed_frames
                result.update(image_id=image_id,repeat=repeat,sequence_local=sequence,source_kind=source_kind,
                              dropped_camera_frames=missed_frames)
                if args.render and not result.get('frame_error'):
                    preview=annotate(image,result)
                    preview[:image.shape[0]]=annotate_pose(image,result,pipeline.geometry)
                    if not cv2.imwrite(str(out/'frames'/('%02d_%03d.png'%(repeat,image_id))),preview):raise OSError('preview save failed')
                rendered=time.perf_counter();age=(rendered-sample_time)*1000
                packet=packet_for_age(result,age,config['result_age_budget_ms'])
                result['theoretical_wire_ms']=len(packet)*10*1000./config['uart_baud']
                result['transport']=sink.write(packet) if sink else dict(kind='disabled',bytes=0,write_ms=0,wire_drain_measured=False)
                transport_done=time.perf_counter()
                result['result_age_after_app_write_ms']=(transport_done-sample_time)*1000 if sink else None
                result['frame_to_app_write_ms']=(transport_done-frame_start)*1000 if sink else None
                latencies['capture_handoff_age'].append(max(0.,(read_done-sample_time)*1000))
                latencies['source_wait'].append((read_done-frame_start)*1000)
                latencies['pipeline_compute'].append(result['timings_ms']['compute'])
                if sink:
                    latencies['app_write_result_age'].append(result['result_age_after_app_write_ms'])
                    if 'write_ms' in result['transport']:
                        latencies['app_serial_write'].append(result['transport']['write_ms'])
                    latencies['frame_to_write'].append(result['frame_to_app_write_ms'])
                for name in ('detection','center_only','scheduled_refinement_and_veto',
                             'metric_validation','uart_encode'):
                    if name in result['timings_ms']:stages[name].append(result['timings_ms'][name])
                diagnostics=result.get('frontend_diagnostics',{})
                yolo=diagnostics.get('yolo_diagnostics',diagnostics)
                for name in ('native_detector','hsv_and_blue','orange_recovery'):
                    value=yolo.get('timings_ms',{}).get(name)
                    if value is not None:stages[name].append(value)
                for name,field in (('green_classic','classic'),('green_native_rims','native_rims'),
                                   ('green_association_and_support','association_and_support')):
                    value=diagnostics.get('timings_ms',{}).get(field)
                    if value is not None:stages[name].append(value)
                native=yolo.get('native_stages_ms',{})
                for name,field in (('native_preprocess','preprocess'),
                                   ('native_tensorrt_and_transfers','tensorrt_and_transfers'),
                                   ('native_postprocess','postprocess')):
                    if field in native:stages[name].append(native[field])
                record_packet(packet)
                if monitor:
                    monitor_started=time.perf_counter()
                    monitor.submit(image,result,packet,pipeline.geometry)
                    stages['monitor_submit'].append((time.perf_counter()-monitor_started)*1000)
                after_io=time.perf_counter()
                result['timings_ms'].update(source_read=(read_done-frame_start)*1000,render_and_png=(rendered-read_done)*1000-result['timings_ms']['compute'],
                                            read_completion_age_before_write=age,runtime_before_log=(after_io-frame_start)*1000)
                # Own write duration is measured by an independent timing row, avoiding self-referential timings.
                log_start=time.perf_counter()
                if not args.no_frame_log:
                    log.write(json.dumps(result,separators=(',',':'))+'\n')
                    log.flush()
                log_end=time.perf_counter()
                if not args.no_frame_log:
                    stages['frame_json_log'].append((log_end-log_start)*1000)
                timing=dict(record_kind='wall_timing',image_id=image_id,repeat=repeat,sequence_local=sequence,
                            wall_with_json_ms=(log_end-frame_start)*1000,json_write_ms=(log_end-log_start)*1000,
                            real_camera=bool(camera),physical_uart=bool(args.uart_device))
                if not args.no_frame_log:log.write(json.dumps(timing,separators=(',',':'))+'\n')
                if not args.quiet:print('frame',image_id,'repeat',repeat,'counts',result['counts'],'wall_ms',round(timing['wall_with_json_ms'],2),'UART',packet.hex(),flush=True)
                if samples:
                    sample_started=time.perf_counter()
                    event=samples.maybe_save(image,result,packet)
                    stages['sample_submit'].append((time.perf_counter()-sample_started)*1000)
                    if event:log.write(json.dumps(event)+'\n')
                count+=1;source_index+=1;last_sequence=sequence
                if args.rate_hz:
                    deadline+=1./args.rate_hz
                    delay=deadline-time.perf_counter()
                    if delay>0:time.sleep(delay)
                    else:deadline=time.perf_counter()
            if args.max_frames and count>=args.max_frames:break
            if not image_rows:break
        if count==0:raise RuntimeError('no frames processed')
    except (Exception,KeyboardInterrupt) as error:
        event=dict(record_kind='runtime_error',error=str(error),uart_hex=uart.encode([]).hex())
        record_packet(uart.encode([]))
        try:
            if sink:event['transport']=sink.write(uart.encode([]))
        except Exception as serial_error:event['serial_error']=str(serial_error)
        log.write(json.dumps(event)+'\n');print(json.dumps(event),file=sys.stderr);exit_code=2
    finally:
        if startup_pool is not None:
            try:
                initial_camera=startup_future.result()
                if camera is None:camera=initial_camera
            except Exception as cleanup_error:
                log.write(json.dumps(dict(record_kind='camera_startup_cleanup_error',error=str(cleanup_error)))+'\n')
            startup_pool.shutdown(wait=True)
        if loop_start is not None:
            elapsed_ms=(time.perf_counter()-loop_start)*1000
            try:
                (out/'run_summary.json').write_text(json.dumps(dict(frames=count,loop_elapsed_ms=elapsed_ms,
                    mean_loop_ms=elapsed_ms/count if count else None,observed_loop_hz=count*1000./elapsed_ms if elapsed_ms else 0.,
                    includes_rate_pacing=bool(args.rate_hz),requested_rate_hz=args.rate_hz,includes_stdout_and_both_json_rows=not args.no_frame_log,
                    application_results_written=count if sink else 0,
                    camera_sequences_skipped_during_processing=camera_skipped_total if camera else None,
                    latency_window_max_frames=1024,
                    latency_stage_ms={name:latency_quantiles(samples) for name,samples in stages.items()},
                    latency_definitions=dict(capture_handoff_age='read_done - camera process read completion (not exposure); replay starts at application frame start',
                        app_write_result_age='serial write() finished - camera process read completion (not physical wire drain)',
                        source_wait='read_done - loop start; includes waiting for a source frame',
                        pipeline_compute='Pipeline.process compute only, excludes source wait and I/O',
                        app_serial_write='SerialSink.write application-buffer duration; unmeasured when transport disabled',
                        frame_to_write='serial write() finished - loop start; includes source wait and processing'),
                    latency_ms={name:latency_quantiles(samples) for name,samples in latencies.items()}),indent=2))
            except OSError as summary_error:
                print('summary write error:',summary_error,file=sys.stderr);exit_code=2
        for resource,method in ((monitor,'close'),(image_publisher,'close'),(camera,'close'),(capture,'release'),(pipeline,'close'),(sink,'close'),(packet_recorder,'close')):
            if resource:
                try:getattr(resource,method)()
                except Exception as cleanup_error:print('cleanup error:',cleanup_error,file=sys.stderr);exit_code=2
        if samples:
            # Drain the bounded writer before closing the evidence log.
            log.write(json.dumps(samples.close())+'\n')
        if image_publisher:log.write(json.dumps(dict(record_kind='ros_image_summary',**image_publisher.diagnostics()))+'\n')
        if sink:
            log.write(json.dumps(dict(record_kind='uart_link_summary',**sink.diagnostics()))+'\n')
            if sink.recording_error:exit_code=2
        log.close()
        for signum,handler in termination_handlers.items():signal.signal(signum,handler)
    return exit_code

if __name__=='__main__':sys.exit(main())
