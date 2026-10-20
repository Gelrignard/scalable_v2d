"""Small source/version record for uncommitted experimental implementations."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import shutil
import os
import time


def process_cost():
    """CPU time and peak resident memory for this Python process, no dependency."""
    peak = None
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[
                (name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage',
                 'QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        counters=Counters();counters.cb=ctypes.sizeof(counters)
        kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
            peak=counters.PeakWorkingSetSize
    return dict(cpu_seconds=time.process_time(),peak_resident_bytes=peak,
                memory_scope='peak for this Python process including imported runtime',gpu_compute_used=False)


def snapshot_sources(output, paths):
    root, output = Path.cwd(), Path(output)
    sources = []
    for path in paths:
        path = Path(path)
        relative = path.resolve().relative_to(root)
        destination = output/'implementation'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        sources.append(dict(path=relative.as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    versions = {}
    for name in ('numpy', 'scipy', 'mujoco', 'trimesh'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                platform=platform.platform(), versions=versions, source_snapshot=sources)
