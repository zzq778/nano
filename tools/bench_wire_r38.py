"""CPU A/B benchmark of the original and current UART encoders.

Usage: python3 tools/bench_wire_r38.py --baseline-root ../jetson_vision_r37
The baseline root is a separate, unmodified r37 checkout. No UART is opened.
"""
import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def load_uart(root,name):
    spec=importlib.util.spec_from_file_location(name,root/'python/rescuevision/uart.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cases():
    for count in (0,1,15,32):
        yield count,[dict(type=i%8,x_mm=450+i*13,y_mm=-100+i*9,
                          yaw_cdeg=i*37 if i%8 in (2,4,5,6,7) else 0)
                     for i in range(count)]


def quantiles(timings):
    values=sorted(timings)
    return dict(p50_us=round(statistics.median(values),3),
                p95_us=round(values[(95*len(values)+99)//100-1],3))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline-root',required=True,type=Path)
    parser.add_argument('--iterations',type=int,default=500)
    args=parser.parse_args()
    before=load_uart(args.baseline_root,'original_uart')
    after=load_uart(ROOT,'optimized_uart')
    results={}
    for count,targets in cases():
        assert before.encode(targets)==after.encode(targets)
        old_times=[];new_times=[]
        for i in range(args.iterations+30):
            pairs=((before,old_times),(after,new_times)) if i%2 else ((after,new_times),(before,old_times))
            for codec,times in pairs:
                started=time.perf_counter_ns()
                codec.encode(targets)
                if i>=30:times.append((time.perf_counter_ns()-started)/1000)
        results[count]=dict(before=quantiles(old_times),after=quantiles(new_times),
                            p50_saved_pct=round(100*(statistics.median(old_times)-statistics.median(new_times))/statistics.median(old_times),2),
                            bytes=len(after.encode(targets)),matched_bytes=True)
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
