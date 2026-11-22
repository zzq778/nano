#!/usr/bin/env python3
"""Read-only Jetson readiness probe. Never runs capture, inference, sudo or builds.

This probe can be executed after a developer checkout has been verified. It is
safe to run by hand without changing camera/serial ownership or power modes.
"""
from __future__ import print_function
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def probe(root=Path('/')):
    root=Path(root)
    available=shutil.disk_usage(str(root)).free
    mem_avail_kib=None
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                mem_avail_kib=int(line.split()[1]);break
    except (OSError, ValueError, IndexError):pass
    tools={name:bool(shutil.which(name)) for name in ('git','g++','cmake','nvcc','tegrastats','nvpmodel')}
    arch=platform.machine()
    report={'schema_version':1,'arch':arch, 'free_disk_bytes':available,
            'free_disk_gib':round(available/float(1024**3),2),
            'available_ram_mib':round(mem_avail_kib/1024.,1) if mem_avail_kib else None,
            'tools':tools, 'root_checked':str(root),
            'checks':{},'notes':[]}
    report['checks']['jetson_arm64']=arch in ('aarch64','arm64')
    report['checks']['disk_at_least_4_gib']=available >= 4*1024**3
    report['checks']['available_ram_at_least_750_mib']=bool(mem_avail_kib is not None and mem_avail_kib>=750*1024)
    report['checks']['cpp_tools']=bool(tools['g++'] and tools['cmake'])
    report['ready_for_separate_build']=all(report['checks'].values())
    report['notes']=['Read-only environment survey; no camera, UART or CUDA context was opened.',
        'Available disk/RAM gates are conservative capacity checks, not proof of successful CMake/SDK build.',
        'nvcc absent from PATH does not alone prove that CUDA runtime or TensorRT libraries are missing.',
        'Actual camera FPS and thermal throttling require later authorized live capture and sustained tests.']
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path',type=Path,default=Path.home())
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    result=probe(args.path)
    content=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if args.output:args.output.write_text(content)
    print(content)
    return 0

if __name__=='__main__':sys.exit(main())
