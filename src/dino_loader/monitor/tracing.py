"""
dino_loader.monitor.tracing
===========================
Explicit flamegraph tracing context across processes.
"""

import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Optional

class ProcessTracer:
    def __init__(self):
        self.enabled = False
        self.base_path = None
        self._f = None
        self._first = True
        self._lock = threading.Lock()

    def start(self, base_path: str):
        """Starts tracing for this process. Will create a unique file for the PID."""
        self.enabled = True
        self.base_path = base_path
        # Use PID to separate process traces to avoid lock contention / interleaved JSON
        path = f"{base_path}_{os.getpid()}.json"
        self._f = open(path, "w")
        self._f.write("[\n")
        self._first = True

    def stop(self):
        self.enabled = False
        with self._lock:
            if self._f:
                self._f.write("\n]\n")
                self._f.close()
                self._f = None

    def record(self, name: str, cat: str, start_us: int, dur_us: int):
        if not self.enabled or not self._f:
            return
            
        event = {
            "name": name,
            "cat": cat,
            "ph": "X",
            "ts": start_us,
            "dur": dur_us,
            "pid": os.getpid(),
            "tid": threading.get_native_id()
        }
        
        evt_str = json.dumps(event)
        with self._lock:
            if not self._first:
                self._f.write(",\n")
            self._first = False
            self._f.write(evt_str)

_GLOBAL_TRACER = ProcessTracer()

def start_tracing(base_path: str):
    """Start Chrome Trace event collection."""
    _GLOBAL_TRACER.start(base_path)

def stop_tracing():
    """Stop Trace collection."""
    _GLOBAL_TRACER.stop()

@contextmanager
def trace(name: str, cat: str = "default"):
    """
    Context manager to trace the duration of a block of code.
    If tracing is not enabled, this has effectively zero overhead (~ns).
    """
    if not _GLOBAL_TRACER.enabled:
        yield
        return
        
    start_ts = time.perf_counter_ns() // 1000
    try:
        yield
    finally:
        end_ts = time.perf_counter_ns() // 1000
        _GLOBAL_TRACER.record(name, cat, start_ts, end_ts - start_ts)
