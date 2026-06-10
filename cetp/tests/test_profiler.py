"""Unit tests for cetp.profiler."""
import time
from unittest.mock import MagicMock, mock_open, patch

import pytest

import cetp.profiler as profiler


# ---------------------------------------------------------------------------
# detect_disk_type
# ---------------------------------------------------------------------------

class TestDetectDiskType:
    def setup_method(self):
        # Reset module-level cache before each test
        profiler.DISK_TYPE_CACHE = None

    def test_returns_valid_type_non_linux(self):
        """On non-Linux (os.name != posix or no /sys/block), falls back to SSD."""
        with patch("os.name", "nt"):
            result = profiler.detect_disk_type()
        assert result in ("HDD", "SSD", "NVMe")

    def test_returns_ssd_on_non_linux(self):
        """Explicitly check that the non-Linux fallback is SSD."""
        profiler.DISK_TYPE_CACHE = None
        with patch("os.name", "nt"):
            result = profiler.detect_disk_type()
        assert result == "SSD"

    def test_cache_is_used_on_second_call(self):
        """detect_disk_type should not re-detect when cache is populated."""
        profiler.DISK_TYPE_CACHE = "NVMe"
        with patch("psutil.disk_partitions") as mock_parts:
            result = profiler.detect_disk_type()
        mock_parts.assert_not_called()
        assert result == "NVMe"

    def test_returns_hdd_when_rotational_is_1(self):
        """Simulate a Linux HDD: rotational file contains '1'."""
        profiler.DISK_TYPE_CACHE = None
        mock_part = MagicMock()
        mock_part.mountpoint = "/"
        mock_part.device = "/dev/sda1"

        with (
            patch("os.name", "posix"),
            patch("os.path.exists", return_value=True),
            patch("psutil.disk_partitions", return_value=[mock_part]),
            patch("builtins.open", mock_open(read_data="1\n")),
        ):
            result = profiler.detect_disk_type()
        assert result == "HDD"

    def test_returns_ssd_when_rotational_is_0(self):
        """Simulate a Linux SSD: rotational file contains '0'."""
        profiler.DISK_TYPE_CACHE = None
        mock_part = MagicMock()
        mock_part.mountpoint = "/"
        mock_part.device = "/dev/sdb1"

        with (
            patch("os.name", "posix"),
            patch("os.path.exists", return_value=True),
            patch("psutil.disk_partitions", return_value=[mock_part]),
            patch("builtins.open", mock_open(read_data="0\n")),
        ):
            result = profiler.detect_disk_type()
        assert result == "SSD"


# ---------------------------------------------------------------------------
# get_disk_speed_class
# ---------------------------------------------------------------------------

class TestGetDiskSpeedClass:
    def test_hdd_returns_1(self):
        assert profiler.get_disk_speed_class("HDD") == 1

    def test_ssd_returns_2(self):
        assert profiler.get_disk_speed_class("SSD") == 2

    def test_nvme_returns_3(self):
        assert profiler.get_disk_speed_class("NVMe") == 3

    def test_unknown_type_returns_2(self):
        """Unknown disk types default to the SSD speed class."""
        assert profiler.get_disk_speed_class("UNKNOWN") == 2

    def test_case_insensitive(self):
        """Speed class lookup should be case-insensitive."""
        assert profiler.get_disk_speed_class("hdd") == 1
        assert profiler.get_disk_speed_class("ssd") == 2
        assert profiler.get_disk_speed_class("nvme") == 3


# ---------------------------------------------------------------------------
# compute_derived_features
# ---------------------------------------------------------------------------

class TestComputeDerivedFeatures:
    def test_effective_cpu(self):
        result = profiler.compute_derived_features(
            cpu_cores=8,
            cpu_avg_pct=25.0,
            memory_avg_gb=4.0,
            memory_total_gb=16.0,
            disk_read_mb=100.0,
            disk_write_mb=50.0,
            runtime_sec=10.0,
        )
        # effective_cpu = 8 * (1 - 25/100) = 8 * 0.75 = 6.0
        assert result["effective_cpu"] == pytest.approx(6.0)

    def test_memory_pressure(self):
        result = profiler.compute_derived_features(
            cpu_cores=4,
            cpu_avg_pct=50.0,
            memory_avg_gb=8.0,
            memory_total_gb=16.0,
            disk_read_mb=0.0,
            disk_write_mb=0.0,
            runtime_sec=5.0,
        )
        # memory_pressure = 8 / 16 = 0.5
        assert result["memory_pressure"] == pytest.approx(0.5)

    def test_io_intensity(self):
        result = profiler.compute_derived_features(
            cpu_cores=4,
            cpu_avg_pct=50.0,
            memory_avg_gb=4.0,
            memory_total_gb=16.0,
            disk_read_mb=90.0,
            disk_write_mb=60.0,
            runtime_sec=10.0,
        )
        # io_intensity = (90 + 60) / 10 = 15.0
        assert result["io_intensity"] == pytest.approx(15.0)

    def test_zero_runtime_does_not_raise(self):
        """io_intensity should be 0 when runtime_sec == 0, not raise ZeroDivisionError."""
        result = profiler.compute_derived_features(
            cpu_cores=4,
            cpu_avg_pct=50.0,
            memory_avg_gb=4.0,
            memory_total_gb=16.0,
            disk_read_mb=10.0,
            disk_write_mb=5.0,
            runtime_sec=0.0,
        )
        assert result["io_intensity"] == pytest.approx(0.0)

    def test_zero_memory_total_does_not_raise(self):
        """memory_pressure should be 0 when memory_total_gb == 0, not raise ZeroDivisionError."""
        result = profiler.compute_derived_features(
            cpu_cores=4,
            cpu_avg_pct=50.0,
            memory_avg_gb=0.0,
            memory_total_gb=0.0,
            disk_read_mb=10.0,
            disk_write_mb=5.0,
            runtime_sec=5.0,
        )
        assert result["memory_pressure"] == pytest.approx(0.0)

    def test_returns_all_keys(self):
        result = profiler.compute_derived_features(
            cpu_cores=4, cpu_avg_pct=50.0, memory_avg_gb=4.0,
            memory_total_gb=16.0, disk_read_mb=10.0, disk_write_mb=5.0, runtime_sec=5.0,
        )
        assert {"effective_cpu", "memory_pressure", "io_intensity"} == set(result.keys())


# ---------------------------------------------------------------------------
# RuntimeSampler
# ---------------------------------------------------------------------------

class TestRuntimeSampler:
    def test_get_averages_returns_expected_keys(self):
        """After a short sampling run, get_averages() must return all four keys."""
        sampler = profiler.RuntimeSampler(interval_sec=0.1)
        sampler.start()
        time.sleep(0.35)  # enough for 2–3 samples
        sampler.stop()
        averages = sampler.get_averages()
        assert "cpu_avg_pct" in averages
        assert "memory_avg_gb" in averages
        assert "cpu_peak_pct" in averages
        assert "memory_peak_gb" in averages

    def test_get_averages_values_are_numeric(self):
        """All returned values must be floats (or ints)."""
        sampler = profiler.RuntimeSampler(interval_sec=0.1)
        sampler.start()
        time.sleep(0.25)
        sampler.stop()
        averages = sampler.get_averages()
        for key, val in averages.items():
            assert isinstance(val, (int, float)), f"{key} is not numeric: {val}"

    def test_get_averages_before_start_returns_zeros(self):
        """Calling get_averages() without start/stop should return all zeros."""
        sampler = profiler.RuntimeSampler(interval_sec=0.1)
        averages = sampler.get_averages()
        assert averages["cpu_avg_pct"] == 0.0
        assert averages["memory_avg_gb"] == 0.0

    def test_peak_gte_average(self):
        """Peak values must be >= average values."""
        sampler = profiler.RuntimeSampler(interval_sec=0.1)
        sampler.start()
        time.sleep(0.35)
        sampler.stop()
        avg = sampler.get_averages()
        assert avg["cpu_peak_pct"] >= avg["cpu_avg_pct"]
        assert avg["memory_peak_gb"] >= avg["memory_avg_gb"]
