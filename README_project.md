# CETP — Cross-Environment Execution Time Prediction

> Predict production runtime from development environment metrics with SLA-aware flagging and SHAP-based explanations.

[![CI](https://github.com/pseudo0244/cetp/actions/workflows/ci.yml/badge.svg)](https://github.com/pseudo0244/cetp/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/cetp.svg)](https://pypi.org/project/cetp/)
[![Coverage](https://codecov.io/gh/pseudo0244/cetp/branch/main/graph/badge.svg)](https://codecov.io/gh/pseudo0244/cetp)

---

## Installation

```bash
pip install cetp
```

For development:

```bash
git clone https://github.com/pseudo0244/cetp.git
cd cetp
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

---

## Quick Start

### Predict production runtime

```bash
cetp predict --workload-type ML --workload-name ml_resnet50 --complexity 3
```

Output:

```
Workload : ml_resnet50 (ML, complexity 3)
Predicted: 13.42 s  [90% CI 11.80 – 15.05 s]
SLA flag : GREEN
```

### Profile your machine

```bash
cetp profile
```

Output:

```
=== CETP System Profile ===
  CPU cores (physical) : 8
  Total RAM            : 16.00 GB
  Disk type            : SSD
  Disk speed class     : 2  (HDD=1, SSD=2, NVMe=3)
```

---

## CLI Reference

| Command | Description | Key Flags |
|---------|-------------|-----------|
| `cetp predict` | Predict production runtime | `--workload-type`, `--workload-name`, `--complexity`, `--sla`, `--fail-on-red`, `--json` |
| `cetp profile` | Show machine hardware specs | — |
| `cetp train` | Train a custom BYOD model | `--data`, `--sla`, `--output-dir` |
| `cetp validate` | Validate CSV against CETP schema | `--data` |
| `cetp schema` | Show or export required CSV schema | `--export` |
| `cetp info` | Show active model and SLA thresholds | — |

### `cetp predict` flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--workload-type` | Choice | required | `ML`, `DB`, or `WEB` |
| `--workload-name` | TEXT | required | e.g. `ml_resnet`, `tpch_q3` |
| `--complexity` | INT (1–5) | required | Workload complexity level |
| `--sla` | PATH | `sla_defaults.json` | SLA thresholds config |
| `--cpu-cores` | INT | auto-detect | Override CPU core count |
| `--memory-gb` | FLOAT | auto-detect | Override total RAM in GB |
| `--disk-type` | Choice | auto-detect | `HDD`, `SSD`, or `NVMe` |
| `--cpu-pct` | FLOAT | 50.0 | Override CPU utilisation % |
| `--mem-used-gb` | FLOAT | 50% of RAM | Override used memory in GB |
| `--fail-on-red` | FLAG | false | Exit code 1 on RED SLA flag |
| `--json` | FLAG | false | Output raw JSON |

### `cetp train` flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data` | PATH | required | Path to company CSV |
| `--sla` | PATH | — | Company SLA JSON file |
| `--output-dir` | PATH | `~/.cetp` | Where to save the model |

---

## Docker

### Run the API server

```bash
docker compose up cetp-api
```

The REST API is available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

### Run CLI via Docker

```bash
docker compose run --rm cetp-cli predict --workload-type ML --workload-name ml_resnet --complexity 3
```

### Build manually

```bash
docker build -t cetp:latest .
docker run -p 8000:8000 cetp:latest
```

---

## REST API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Predict runtime for a workload |
| `GET` | `/health` | Health check with model status |
| `GET` | `/explain` | Global SHAP feature importance |

---

## Development Setup

```bash
# Clone and install
git clone https://github.com/pseudo0244/cetp.git
cd cetp
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -e ".[dev]"

# Lint
ruff check src/ tests/

# Test with coverage
pytest

# Run the API locally
PYTHONPATH=src uvicorn api.main:app --reload
```

### CSV Schema for BYOD Training

Your company CSV must contain at least 500 rows and these columns:

```
run_id, workload_type, workload_name, workload_complexity,
cpu_cores, memory_total_gb, cpu_avg_pct, effective_cpu,
memory_avg_gb, memory_pressure, disk_read_mb, disk_write_mb,
io_intensity, disk_type, disk_speed_class, runtime_sec
```

Export a blank template:

```bash
cetp schema --export template.csv
```

Validate before training:

```bash
cetp validate --data company_runs.csv
```

---

## Links

- [Documentation](#) *(coming soon)*
- [Research Paper](#) *(coming soon)*
- [Issue Tracker](https://github.com/pseudo0244/cetp/issues)

---

*CETP — Team 94, PES University*
