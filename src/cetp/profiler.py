"""
System profiler: collects hardware and runtime metrics from the
local machine using psutil. Produces the feature vector used by
the prediction engine.
"""
import logging
import os
import re
import socket
import sys
import threading
import time

import psutil

logger = logging.getLogger(__name__)

DISK_TYPE_CACHE: str | None = None


def detect_disk_type() -> str:
    """
    Detect the storage medium type of the root partition.
    Uses /sys/block/<dev>/queue/rotational on Linux.
    Returns 'HDD', 'SSD', or 'NVMe'.
    Falls back to 'SSD' on non-Linux systems (macOS, Windows).
    Result is cached in DISK_TYPE_CACHE after first detection.
    """
    global DISK_TYPE_CACHE
    if DISK_TYPE_CACHE is not None:
        return DISK_TYPE_CACHE

    if sys.platform != "linux":
        DISK_TYPE_CACHE = "SSD"
        return DISK_TYPE_CACHE

    try:
        partitions = psutil.disk_partitions(all=False)
        root_device = None
        for part in partitions:
            if part.mountpoint == "/":
                root_device = part.device
                break

        if root_device is None:
            DISK_TYPE_CACHE = "SSD"
            return DISK_TYPE_CACHE

        # Strip partition suffix: sda1 → sda, nvme0n1p1 → nvme0n1
        dev_name = re.sub(r"p?\d+$", "", os.path.basename(root_device))
        rotational_path = f"/sys/block/{dev_name}/queue/rotational"

        with open(rotational_path) as fh:
            value = fh.read().strip()

        if value == "1":
            DISK_TYPE_CACHE = "HDD"
        elif "nvme" in dev_name:
            DISK_TYPE_CACHE = "NVMe"
        else:
            DISK_TYPE_CACHE = "SSD"

    except Exception:
        DISK_TYPE_CACHE = "SSD"

    return DISK_TYPE_CACHE


def get_disk_speed_class(disk_type: str) -> int:
    """Returns integer speed class: HDD=1, SSD=2, NVMe=3. Case-insensitive."""
    mapping = {"HDD": 1, "SSD": 2, "NVME": 3}
    return mapping.get(disk_type.upper(), 2)


def get_static_profile() -> dict:
    """
    Returns static hardware characteristics of the current machine.
    Keys: cpu_cores, memory_total_gb, disk_type, disk_speed_class, platform, hostname
    """
    disk_type = detect_disk_type()
    return {
        "cpu_cores": psutil.cpu_count(logical=True),
        "memory_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "disk_type": disk_type,
        "disk_speed_class": get_disk_speed_class(disk_type),
        "platform": sys.platform,
        "hostname": socket.gethostname(),
    }


def take_snapshot() -> dict:
    """
    Captures a point-in-time snapshot before a workload begins.
    Stores start time and disk I/O baseline for delta computation.
    Returns a snapshot dict including static profile fields plus _time_start,
    _disk_read_before, and _disk_write_before.
    """
    disk_type = detect_disk_type()
    disk_io = psutil.disk_io_counters()
    return {
        "cpu_cores": psutil.cpu_count(logical=True),
        "memory_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "disk_type": disk_type,
        "disk_speed_class": get_disk_speed_class(disk_type),
        "_time_start": time.perf_counter(),
        "_disk_read_before": disk_io.read_bytes if disk_io else 0,
        "_disk_write_before": disk_io.write_bytes if disk_io else 0,
    }


def take_end_snapshot(start_snap: dict) -> tuple:
    """
    Computes metrics after a workload completes.
    Returns (disk_read_mb, disk_write_mb, runtime_sec) as a tuple.
    """
    disk_io = psutil.disk_io_counters()
    read_bytes = (disk_io.read_bytes if disk_io else 0) - start_snap["_disk_read_before"]
    write_bytes = (disk_io.write_bytes if disk_io else 0) - start_snap["_disk_write_before"]

    disk_read_mb = round(max(read_bytes, 0) / 1e6, 3)
    disk_write_mb = round(max(write_bytes, 0) / 1e6, 3)
    runtime_sec = round(time.perf_counter() - start_snap["_time_start"], 4)

    return disk_read_mb, disk_write_mb, runtime_sec


def compute_derived_features(
    cpu_cores: int,
    cpu_avg_pct: float,
    memory_avg_gb: float,
    memory_total_gb: float,
    disk_read_mb: float,
    disk_write_mb: float,
    runtime_sec: float,
) -> dict:
    """
    Computes the three derived features from sampled and static values.
    effective_cpu = cpu_cores * (1 - cpu_avg_pct / 100)
    memory_pressure = memory_avg_gb / memory_total_gb, clamped to [0, 1]
    io_intensity = (disk_read_mb + disk_write_mb) / runtime_sec
    """
    effective_cpu = round(cpu_cores * (1.0 - cpu_avg_pct / 100.0), 4)
    if memory_total_gb > 0:
        memory_pressure = round(min(max(memory_avg_gb / memory_total_gb, 0.0), 1.0), 4)
    else:
        memory_pressure = 0.0
    io_intensity = (
        round((disk_read_mb + disk_write_mb) / runtime_sec, 4) if runtime_sec > 0 else 0.0
    )
    return {
        "effective_cpu": effective_cpu,
        "memory_pressure": memory_pressure,
        "io_intensity": io_intensity,
    }


class RuntimeSampler:
    """
    Background thread that samples CPU and memory usage at a fixed interval
    during workload execution. Call start() before the workload, stop() after.
    """

    def __init__(self, interval_sec: float = 0.5) -> None:
        self.interval_sec = interval_sec
        self._cpu_samples: list = []
        self._mem_samples: list = []
        self._stop_event: threading.Event = threading.Event()
        self._thread: threading.Thread = None
        self._started: bool = False

    def start(self) -> None:
        """Begin background sampling. Raises RuntimeError if already running."""
        if self._started:
            raise RuntimeError("RuntimeSampler is already running; call stop() first.")
        self._stop_event.clear()
        self._cpu_samples = []
        self._mem_samples = []
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        """Signal the sampling thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._started = False

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._cpu_samples.append(psutil.cpu_percent(interval=None))
                self._mem_samples.append(psutil.virtual_memory().used / 1e9)
                time.sleep(self.interval_sec)
            except Exception as exc:
                logger.warning("RuntimeSampler: sampling error: %s", exc)

    def get_averages(self) -> dict:
        """Returns dict with cpu_avg_pct, memory_avg_gb, cpu_peak_pct, memory_peak_gb, sample_count."""
        if not self._cpu_samples:
            return {
                "cpu_avg_pct": 0.0,
                "memory_avg_gb": 0.0,
                "cpu_peak_pct": 0.0,
                "memory_peak_gb": 0.0,
                "sample_count": 0,
            }
        n = len(self._cpu_samples)
        return {
            "cpu_avg_pct": round(sum(self._cpu_samples) / n, 2),
            "memory_avg_gb": round(sum(self._mem_samples) / len(self._mem_samples), 3),
            "cpu_peak_pct": round(max(self._cpu_samples), 2),
            "memory_peak_gb": round(max(self._mem_samples), 3),
            "sample_count": n,
        }


def build_feature_row(
    workload_fn: callable,
    workload_type: str,
    workload_name: str,
    workload_complexity: int,
    run_id: int,
    batch_id: str,
) -> dict:
    """
    Runs a workload function while collecting system metrics, then returns
    a complete feature row ready for the dataset CSV.
    """
    sampler = RuntimeSampler(interval_sec=0.5)
    start_snap = take_snapshot()
    sampler.start()
    try:
        workload_fn()
    finally:
        sampler.stop()
    disk_read_mb, disk_write_mb, runtime_sec = take_end_snapshot(start_snap)
    averages = sampler.get_averages()
    derived = compute_derived_features(
        cpu_cores=start_snap["cpu_cores"],
        cpu_avg_pct=averages["cpu_avg_pct"],
        memory_avg_gb=averages["memory_avg_gb"],
        memory_total_gb=start_snap["memory_total_gb"],
        disk_read_mb=disk_read_mb,
        disk_write_mb=disk_write_mb,
        runtime_sec=runtime_sec,
    )
    return {
        "run_id": run_id,
        "batch_id": batch_id,
        "workload_type": workload_type,
        "workload_name": workload_name,
        "workload_complexity": workload_complexity,
        "cpu_cores": start_snap["cpu_cores"],
        "memory_total_gb": start_snap["memory_total_gb"],
        "cpu_avg_pct": averages["cpu_avg_pct"],
        "cpu_peak_pct": averages["cpu_peak_pct"],
        "effective_cpu": derived["effective_cpu"],
        "memory_avg_gb": averages["memory_avg_gb"],
        "memory_peak_gb": averages["memory_peak_gb"],
        "memory_pressure": derived["memory_pressure"],
        "disk_read_mb": disk_read_mb,
        "disk_write_mb": disk_write_mb,
        "io_intensity": derived["io_intensity"],
        "disk_type": start_snap["disk_type"],
        "disk_speed_class": start_snap["disk_speed_class"],
        "runtime_sec": runtime_sec,
    }
