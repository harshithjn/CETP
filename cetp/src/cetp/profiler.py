"""
System profiler: collects hardware and runtime metrics from the
local machine using psutil. Produces the feature vector used by
the prediction engine.
"""
import os
import time
import threading
import psutil

DISK_TYPE_CACHE: str | None = None


def detect_disk_type() -> str:
    """
    Detect the storage medium type of the root partition.
    Uses /sys/block/<dev>/queue/rotational on Linux.
    Returns 'HDD', 'SSD', or 'NVMe'.
    Falls back to 'SSD' on non-Linux systems (e.g. macOS, Windows cloud VMs).
    """
    global DISK_TYPE_CACHE
    if DISK_TYPE_CACHE is not None:
        return DISK_TYPE_CACHE

    if os.name != "posix" or not os.path.exists("/sys/block"):
        DISK_TYPE_CACHE = "SSD"
        return DISK_TYPE_CACHE

    try:
        # Find the device backing the root partition
        partitions = psutil.disk_partitions(all=False)
        root_device = None
        for part in partitions:
            if part.mountpoint == "/":
                root_device = part.device
                break

        if root_device is None:
            DISK_TYPE_CACHE = "SSD"
            return DISK_TYPE_CACHE

        # Strip partition number to get base device name (e.g. /dev/sda1 -> sda)
        dev_name = os.path.basename(root_device).rstrip("0123456789")
        rotational_path = f"/sys/block/{dev_name}/queue/rotational"

        if not os.path.exists(rotational_path):
            # Try nvme pattern — NVMe devices have no rotational flag but name starts with nvme
            if dev_name.startswith("nvme"):
                DISK_TYPE_CACHE = "NVMe"
            else:
                DISK_TYPE_CACHE = "SSD"
            return DISK_TYPE_CACHE

        with open(rotational_path) as fh:
            value = fh.read().strip()

        if value == "1":
            DISK_TYPE_CACHE = "HDD"
        elif dev_name.startswith("nvme"):
            DISK_TYPE_CACHE = "NVMe"
        else:
            DISK_TYPE_CACHE = "SSD"

    except Exception:
        DISK_TYPE_CACHE = "SSD"

    return DISK_TYPE_CACHE


def get_disk_speed_class(disk_type: str) -> int:
    """Returns integer speed class: HDD=1, SSD=2, NVMe=3."""
    mapping = {"HDD": 1, "SSD": 2, "NVME": 3}
    return mapping.get(disk_type.upper(), 2)


def get_static_profile() -> dict:
    """
    Returns static hardware characteristics of the current machine.
    These values do not change during a run.
    Returns dict with keys: cpu_cores, memory_total_gb, disk_type, disk_speed_class
    """
    disk_type = detect_disk_type()
    return {
        "cpu_cores": psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1,
        "memory_total_gb": psutil.virtual_memory().total / (1024 ** 3),
        "disk_type": disk_type,
        "disk_speed_class": get_disk_speed_class(disk_type),
    }


def take_snapshot() -> dict:
    """
    Captures a point-in-time snapshot before a workload begins.
    Stores start time and disk I/O baseline for delta computation.
    Returns a snapshot dict including _time_start and _disk_io_before.
    """
    disk_io = psutil.disk_io_counters()
    return {
        "_time_start": time.perf_counter(),
        "_disk_io_before": {
            "read_bytes": disk_io.read_bytes if disk_io else 0,
            "write_bytes": disk_io.write_bytes if disk_io else 0,
        },
    }


def take_end_snapshot(start_snap: dict) -> tuple:
    """
    Computes metrics after a workload completes.
    Returns (disk_read_mb, disk_write_mb, runtime_sec) as a tuple.
    """
    end_time = time.perf_counter()
    disk_io = psutil.disk_io_counters()

    before = start_snap["_disk_io_before"]
    read_bytes = (disk_io.read_bytes if disk_io else 0) - before["read_bytes"]
    write_bytes = (disk_io.write_bytes if disk_io else 0) - before["write_bytes"]

    disk_read_mb = max(read_bytes / (1024 ** 2), 0.0)
    disk_write_mb = max(write_bytes / (1024 ** 2), 0.0)
    runtime_sec = end_time - start_snap["_time_start"]

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
    Computes the three derived features.
    effective_cpu = cpu_cores * (1 - cpu_avg_pct / 100)
    memory_pressure = memory_avg_gb / memory_total_gb
    io_intensity = (disk_read_mb + disk_write_mb) / runtime_sec
    Returns dict with keys: effective_cpu, memory_pressure, io_intensity
    """
    effective_cpu = cpu_cores * (1.0 - cpu_avg_pct / 100.0)
    memory_pressure = memory_avg_gb / memory_total_gb if memory_total_gb > 0 else 0.0
    io_intensity = (
        (disk_read_mb + disk_write_mb) / runtime_sec if runtime_sec > 0 else 0.0
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
        self._interval = interval_sec
        self._cpu_samples: list[float] = []
        self._mem_samples: list[float] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Begin background sampling."""
        self._stop_event.clear()
        self._cpu_samples = []
        self._mem_samples = []
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the sampling thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 3)
            self._thread = None

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            self._cpu_samples.append(psutil.cpu_percent(interval=None))
            mem = psutil.virtual_memory()
            self._mem_samples.append(mem.used / (1024 ** 3))
            self._stop_event.wait(self._interval)

    def get_averages(self) -> dict:
        """Returns dict with cpu_avg_pct, memory_avg_gb, cpu_peak_pct, memory_peak_gb."""
        if not self._cpu_samples:
            return {
                "cpu_avg_pct": 0.0,
                "memory_avg_gb": 0.0,
                "cpu_peak_pct": 0.0,
                "memory_peak_gb": 0.0,
            }
        return {
            "cpu_avg_pct": sum(self._cpu_samples) / len(self._cpu_samples),
            "memory_avg_gb": sum(self._mem_samples) / len(self._mem_samples),
            "cpu_peak_pct": max(self._cpu_samples),
            "memory_peak_gb": max(self._mem_samples),
        }
