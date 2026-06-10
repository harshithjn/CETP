"""Unit tests for cetp.profiler — Phase 2 comprehensive test suite."""

import time
from unittest.mock import MagicMock, mock_open, patch

import pytest

import cetp.profiler as profiler


def _reset_cache():
    profiler.DISK_TYPE_CACHE = None


# ---------------------------------------------------------------------------
# detect_disk_type
# ---------------------------------------------------------------------------


class TestDetectDiskType:
    def setup_method(self):
        _reset_cache()

    def test_detect_disk_type_hdd(self):
        """On Linux, rotational=1 for /dev/sda1 → HDD."""
        mock_part = MagicMock()
        mock_part.mountpoint = "/"
        mock_part.device = "/dev/sda1"
        with (
            patch("sys.platform", "linux"),
            patch("psutil.disk_partitions", return_value=[mock_part]),
            patch("builtins.open", mock_open(read_data="1\n")),
        ):
            assert profiler.detect_disk_type() == "HDD"

    def test_detect_disk_type_ssd(self):
        """On Linux, rotational=0 and non-nvme device /dev/sda1 → SSD."""
        mock_part = MagicMock()
        mock_part.mountpoint = "/"
        mock_part.device = "/dev/sda1"
        with (
            patch("sys.platform", "linux"),
            patch("psutil.disk_partitions", return_value=[mock_part]),
            patch("builtins.open", mock_open(read_data="0\n")),
        ):
            assert profiler.detect_disk_type() == "SSD"

    def test_detect_disk_type_nvme(self):
        """On Linux, rotational=0 and nvme device /dev/nvme0n1p1 → NVMe."""
        mock_part = MagicMock()
        mock_part.mountpoint = "/"
        mock_part.device = "/dev/nvme0n1p1"
        with (
            patch("sys.platform", "linux"),
            patch("psutil.disk_partitions", return_value=[mock_part]),
            patch("builtins.open", mock_open(read_data="0\n")),
        ):
            assert profiler.detect_disk_type() == "NVMe"

    def test_detect_disk_type_fallback(self):
        """When disk_partitions raises an exception, fall back to SSD."""
        with (
            patch("sys.platform", "linux"),
            patch("psutil.disk_partitions", side_effect=Exception("disk error")),
        ):
            assert profiler.detect_disk_type() == "SSD"

    def test_detect_disk_type_non_linux(self):
        """On macOS (darwin), always return SSD without reading /sys."""
        with patch("sys.platform", "darwin"):
            assert profiler.detect_disk_type() == "SSD"

    def test_cache_used_on_second_call(self):
        """After first detection, DISK_TYPE_CACHE is used; psutil not called again."""
        profiler.DISK_TYPE_CACHE = "NVMe"
        with patch("psutil.disk_partitions") as mock_parts:
            result = profiler.detect_disk_type()
        mock_parts.assert_not_called()
        assert result == "NVMe"


# ---------------------------------------------------------------------------
# get_disk_speed_class
# ---------------------------------------------------------------------------


class TestGetDiskSpeedClass:
    def test_disk_speed_class_hdd(self):
        """HDD maps to speed class 1."""
        assert profiler.get_disk_speed_class("HDD") == 1

    def test_disk_speed_class_ssd(self):
        """SSD maps to speed class 2."""
        assert profiler.get_disk_speed_class("SSD") == 2

    def test_disk_speed_class_nvme_mixed_case(self):
        """NVMe speed class is 3 regardless of case."""
        assert profiler.get_disk_speed_class("NVMe") == 3
        assert profiler.get_disk_speed_class("nvme") == 3
        assert profiler.get_disk_speed_class("NVME") == 3

    def test_disk_speed_class_unknown(self):
        """Unknown disk type falls back to SSD speed class (2)."""
        assert profiler.get_disk_speed_class("OPTANE") == 2


# ---------------------------------------------------------------------------
# get_static_profile
# ---------------------------------------------------------------------------


class TestGetStaticProfile:
    def setup_method(self):
        _reset_cache()

    def test_static_profile_keys(self):
        """get_static_profile() must return all 6 required keys."""
        mock_mem = MagicMock()
        mock_mem.total = 16_000_000_000
        with (
            patch("psutil.cpu_count", return_value=8),
            patch("psutil.virtual_memory", return_value=mock_mem),
            patch("sys.platform", "darwin"),
            patch("socket.gethostname", return_value="test-host"),
        ):
            result = profiler.get_static_profile()
        expected = {
            "cpu_cores",
            "memory_total_gb",
            "disk_type",
            "disk_speed_class",
            "platform",
            "hostname",
        }
        assert expected == set(result.keys())

    def test_static_profile_memory_conversion(self):
        """8_000_000_000 bytes → 8.0 GB (1e9 divisor, not 1024^3)."""
        mock_mem = MagicMock()
        mock_mem.total = 8_000_000_000
        with (
            patch("psutil.cpu_count", return_value=4),
            patch("psutil.virtual_memory", return_value=mock_mem),
            patch("sys.platform", "darwin"),
            patch("socket.gethostname", return_value="host"),
        ):
            result = profiler.get_static_profile()
        assert result["memory_total_gb"] == 8.0


# ---------------------------------------------------------------------------
# take_snapshot
# ---------------------------------------------------------------------------


class TestTakeSnapshot:
    def setup_method(self):
        _reset_cache()

    def test_take_snapshot_keys(self):
        """take_snapshot() must include all required keys."""
        mock_disk_io = MagicMock()
        mock_disk_io.read_bytes = 0
        mock_disk_io.write_bytes = 0
        mock_mem = MagicMock()
        mock_mem.total = 8_000_000_000
        with (
            patch("psutil.cpu_count", return_value=4),
            patch("psutil.virtual_memory", return_value=mock_mem),
            patch("psutil.disk_io_counters", return_value=mock_disk_io),
            patch("sys.platform", "darwin"),
        ):
            snap = profiler.take_snapshot()
        required = {
            "cpu_cores",
            "memory_total_gb",
            "disk_type",
            "disk_speed_class",
            "_time_start",
            "_disk_read_before",
            "_disk_write_before",
        }
        assert required.issubset(set(snap.keys()))

    def test_take_snapshot_types(self):
        """_time_start must be float; _disk_read_before and _disk_write_before must be int."""
        mock_disk_io = MagicMock()
        mock_disk_io.read_bytes = 1000
        mock_disk_io.write_bytes = 500
        mock_mem = MagicMock()
        mock_mem.total = 8_000_000_000
        with (
            patch("psutil.cpu_count", return_value=4),
            patch("psutil.virtual_memory", return_value=mock_mem),
            patch("psutil.disk_io_counters", return_value=mock_disk_io),
            patch("sys.platform", "darwin"),
        ):
            snap = profiler.take_snapshot()
        assert isinstance(snap["_time_start"], float)
        assert isinstance(snap["_disk_read_before"], int)
        assert isinstance(snap["_disk_write_before"], int)


# ---------------------------------------------------------------------------
# take_end_snapshot
# ---------------------------------------------------------------------------


class TestTakeEndSnapshot:
    def test_take_end_snapshot_disk_delta(self):
        """100_000_000 bytes more read than baseline → disk_read_mb == 100.0."""
        start_snap = {
            "_time_start": time.perf_counter(),
            "_disk_read_before": 0,
            "_disk_write_before": 0,
        }
        mock_disk_io = MagicMock()
        mock_disk_io.read_bytes = 100_000_000
        mock_disk_io.write_bytes = 0
        with patch("psutil.disk_io_counters", return_value=mock_disk_io):
            disk_read_mb, disk_write_mb, _ = profiler.take_end_snapshot(start_snap)
        assert disk_read_mb == pytest.approx(100.0, abs=0.001)

    def test_take_end_snapshot_runtime(self):
        """runtime_sec reflects elapsed perf_counter time."""
        start_snap = {
            "_time_start": 100.0,
            "_disk_read_before": 0,
            "_disk_write_before": 0,
        }
        mock_disk_io = MagicMock()
        mock_disk_io.read_bytes = 0
        mock_disk_io.write_bytes = 0
        with (
            patch("psutil.disk_io_counters", return_value=mock_disk_io),
            patch("time.perf_counter", return_value=100.5),
        ):
            _, _, runtime_sec = profiler.take_end_snapshot(start_snap)
        assert runtime_sec == pytest.approx(0.5, abs=0.0001)


# ---------------------------------------------------------------------------
# compute_derived_features
# ---------------------------------------------------------------------------


class TestComputeDerivedFeatures:
    def test_effective_cpu_formula(self):
        """cpu_cores=8, cpu_avg_pct=25.0 → effective_cpu == 6.0."""
        result = profiler.compute_derived_features(
            cpu_cores=8,
            cpu_avg_pct=25.0,
            memory_avg_gb=4.0,
            memory_total_gb=8.0,
            disk_read_mb=0.0,
            disk_write_mb=0.0,
            runtime_sec=1.0,
        )
        assert result["effective_cpu"] == pytest.approx(6.0)

    def test_memory_pressure_formula(self):
        """memory_avg_gb=4.0, memory_total_gb=8.0 → memory_pressure == 0.5."""
        result = profiler.compute_derived_features(
            cpu_cores=4,
            cpu_avg_pct=0.0,
            memory_avg_gb=4.0,
            memory_total_gb=8.0,
            disk_read_mb=0.0,
            disk_write_mb=0.0,
            runtime_sec=1.0,
        )
        assert result["memory_pressure"] == pytest.approx(0.5)

    def test_memory_pressure_clamped(self):
        """memory_avg_gb > memory_total_gb (noise) → memory_pressure clamped to 1.0."""
        result = profiler.compute_derived_features(
            cpu_cores=4,
            cpu_avg_pct=0.0,
            memory_avg_gb=9.0,
            memory_total_gb=8.0,
            disk_read_mb=0.0,
            disk_write_mb=0.0,
            runtime_sec=1.0,
        )
        assert result["memory_pressure"] == pytest.approx(1.0)

    def test_io_intensity_formula(self):
        """(disk_read_mb=100 + disk_write_mb=50) / runtime_sec=10 == 15.0."""
        result = profiler.compute_derived_features(
            cpu_cores=4,
            cpu_avg_pct=0.0,
            memory_avg_gb=2.0,
            memory_total_gb=8.0,
            disk_read_mb=100.0,
            disk_write_mb=50.0,
            runtime_sec=10.0,
        )
        assert result["io_intensity"] == pytest.approx(15.0)

    def test_io_intensity_zero_runtime(self):
        """runtime_sec=0 → io_intensity == 0.0, no ZeroDivisionError."""
        result = profiler.compute_derived_features(
            cpu_cores=4,
            cpu_avg_pct=0.0,
            memory_avg_gb=2.0,
            memory_total_gb=8.0,
            disk_read_mb=100.0,
            disk_write_mb=50.0,
            runtime_sec=0.0,
        )
        assert result["io_intensity"] == 0.0

    def test_zero_memory_total_does_not_raise(self):
        """memory_pressure should be 0 when memory_total_gb == 0."""
        result = profiler.compute_derived_features(
            cpu_cores=4,
            cpu_avg_pct=50.0,
            memory_avg_gb=0.0,
            memory_total_gb=0.0,
            disk_read_mb=0.0,
            disk_write_mb=0.0,
            runtime_sec=1.0,
        )
        assert result["memory_pressure"] == pytest.approx(0.0)

    def test_returns_all_keys(self):
        """compute_derived_features returns exactly effective_cpu, memory_pressure, io_intensity."""
        result = profiler.compute_derived_features(
            cpu_cores=4,
            cpu_avg_pct=50.0,
            memory_avg_gb=4.0,
            memory_total_gb=16.0,
            disk_read_mb=10.0,
            disk_write_mb=5.0,
            runtime_sec=5.0,
        )
        assert set(result.keys()) == {"effective_cpu", "memory_pressure", "io_intensity"}


# ---------------------------------------------------------------------------
# RuntimeSampler
# ---------------------------------------------------------------------------


class TestRuntimeSampler:
    def test_sampler_collects_samples(self):
        """After start/sleep/stop, sample_count >= 1."""
        sampler = profiler.RuntimeSampler(interval_sec=0.05)
        sampler.start()
        time.sleep(0.2)
        sampler.stop()
        assert sampler.get_averages()["sample_count"] >= 1

    def test_sampler_returns_correct_keys(self):
        """get_averages() returns all 5 expected keys."""
        sampler = profiler.RuntimeSampler(interval_sec=0.05)
        sampler.start()
        time.sleep(0.15)
        sampler.stop()
        assert set(sampler.get_averages().keys()) == {
            "cpu_avg_pct",
            "memory_avg_gb",
            "cpu_peak_pct",
            "memory_peak_gb",
            "sample_count",
        }

    def test_sampler_double_start_raises(self):
        """Calling start() twice without stop() raises RuntimeError."""
        sampler = profiler.RuntimeSampler(interval_sec=0.1)
        sampler.start()
        try:
            with pytest.raises(RuntimeError):
                sampler.start()
        finally:
            sampler.stop()

    def test_sampler_averages_correct(self):
        """Injecting known samples produces correct averages."""
        sampler = profiler.RuntimeSampler()
        sampler._cpu_samples = [20.0, 40.0]
        sampler._mem_samples = [2.0, 4.0]
        avgs = sampler.get_averages()
        assert avgs["cpu_avg_pct"] == pytest.approx(30.0)
        assert avgs["memory_avg_gb"] == pytest.approx(3.0)

    def test_sampler_peak_values(self):
        """Peak is the max of injected samples."""
        sampler = profiler.RuntimeSampler()
        sampler._cpu_samples = [20.0, 80.0, 40.0]
        sampler._mem_samples = [1.0, 2.0, 1.5]
        avgs = sampler.get_averages()
        assert avgs["cpu_peak_pct"] == pytest.approx(80.0)

    def test_sampler_empty_returns_zeros(self):
        """get_averages() with no samples returns all zeros and sample_count=0."""
        sampler = profiler.RuntimeSampler()
        avgs = sampler.get_averages()
        assert avgs["cpu_avg_pct"] == 0.0
        assert avgs["memory_avg_gb"] == 0.0
        assert avgs["cpu_peak_pct"] == 0.0
        assert avgs["memory_peak_gb"] == 0.0
        assert avgs["sample_count"] == 0

    def test_get_averages_values_are_numeric(self):
        """All returned values must be numeric (int or float)."""
        sampler = profiler.RuntimeSampler(interval_sec=0.05)
        sampler.start()
        time.sleep(0.15)
        sampler.stop()
        for key, val in sampler.get_averages().items():
            assert isinstance(val, (int, float)), f"{key} is not numeric: {val}"

    def test_peak_gte_average(self):
        """Peak values must be >= average values."""
        sampler = profiler.RuntimeSampler(interval_sec=0.05)
        sampler.start()
        time.sleep(0.2)
        sampler.stop()
        avg = sampler.get_averages()
        assert avg["cpu_peak_pct"] >= avg["cpu_avg_pct"]
        assert avg["memory_peak_gb"] >= avg["memory_avg_gb"]


# ---------------------------------------------------------------------------
# build_feature_row
# ---------------------------------------------------------------------------

_EXPECTED_FEATURE_ROW_KEYS = {
    "run_id",
    "batch_id",
    "workload_type",
    "workload_name",
    "workload_complexity",
    "cpu_cores",
    "memory_total_gb",
    "cpu_avg_pct",
    "cpu_peak_pct",
    "effective_cpu",
    "memory_avg_gb",
    "memory_peak_gb",
    "memory_pressure",
    "disk_read_mb",
    "disk_write_mb",
    "io_intensity",
    "disk_type",
    "disk_speed_class",
    "runtime_sec",
}


class TestBuildFeatureRow:
    def test_build_feature_row_all_keys(self):
        """build_feature_row() result must contain all 19 expected keys."""
        row = profiler.build_feature_row(
            workload_fn=lambda: None,
            workload_type="ML",
            workload_name="test_model",
            workload_complexity=2,
            run_id=1,
            batch_id="batch_001",
        )
        assert _EXPECTED_FEATURE_ROW_KEYS == set(row.keys())

    def test_build_feature_row_runtime_positive(self):
        """runtime_sec must be greater than 0."""
        row = profiler.build_feature_row(
            workload_fn=lambda: None,
            workload_type="DB",
            workload_name="query_test",
            workload_complexity=1,
            run_id=0,
            batch_id="b",
        )
        assert row["runtime_sec"] > 0

    def test_build_feature_row_workload_metadata(self):
        """Metadata fields are passed through to the result unchanged."""
        row = profiler.build_feature_row(
            workload_fn=lambda: None,
            workload_type="WEB",
            workload_name="http_bench",
            workload_complexity=3,
            run_id=42,
            batch_id="batch_xyz",
        )
        assert row["run_id"] == 42
        assert row["batch_id"] == "batch_xyz"
        assert row["workload_type"] == "WEB"
        assert row["workload_name"] == "http_bench"
        assert row["workload_complexity"] == 3
