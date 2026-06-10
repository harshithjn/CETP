# CETP — Cross-Environment Execution Time Prediction

> Will your code meet its SLA in production? CETP helps answer that question before deployment.

## Overview

Developers often test applications on local machines that differ significantly from production infrastructure.

A script that takes **45 seconds** on a laptop may run much faster—or slower—on a production server with different CPU, memory, and storage characteristics.

CETP (Cross-Environment Execution Time Prediction) uses machine learning to estimate execution time on a target environment before deployment, helping teams identify potential SLA violations early in the development lifecycle.

---

## Key Features

- Predict execution time on production infrastructure
- Estimate best-case and worst-case runtime ranges
- Evaluate compliance with SLA requirements
- Explain factors influencing predictions
- Train custom models using organization-specific data
- Integrate prediction checks into CI/CD pipelines
- Local-first design — data never leaves your machine

---

## Installation

```bash
pip install cetp
```

---

## Quick Start

### 1. Profile Your Current System

```bash
cetp profile
```

Example output:

```text
=== CETP System Profile ===
CPU cores     : 8
Total RAM     : 16.00 GB
Disk type     : SSD
Disk speed    : 2
```

### 2. Predict Runtime on a Production Environment

```bash
cetp predict \
  --workload-type ML \
  --workload-name ml_resnet \
  --complexity 3 \
  --cpu-cores 64 \
  --memory-gb 256 \
  --disk-type NVMe \
  --cpu-pct 20 \
  --mem-used-gb 48 \
  --sla sla.json
```

Example output:

```text
=== Prediction Result ===
Predicted runtime : 6.3s
Confidence range  : 4.8s – 8.1s
SLA limit         : 20.0s
SLA status        : GREEN ✓

Top factors driving this prediction:
memory_pressure  → reduces runtime   (-3.2s)
effective_cpu    → reduces runtime   (-1.8s)
io_intensity     → increases runtime (+0.9s)
```

---

## Available Commands

| Command | Description |
|----------|------------|
| `cetp profile` | Display hardware specifications of the current machine |
| `cetp predict` | Predict runtime on a target environment |
| `cetp validate --data file.csv` | Validate dataset format |
| `cetp schema` | Display required dataset schema |
| `cetp schema --export template.csv` | Export a blank dataset template |
| `cetp train --data file.csv` | Train a custom prediction model |
| `cetp info` | Show active model and SLA configuration |

---

## SLA Status Levels

Each prediction is categorized into one of three deployment risk levels.

| Status | Description |
|----------|------------|
| GREEN | Predicted runtime is comfortably below the SLA threshold |
| YELLOW | Predicted runtime is approaching the SLA threshold |
| RED | Predicted runtime is likely to violate the SLA |

---

## CI/CD Integration

Use CETP as a deployment gate within GitHub Actions.

```yaml
- name: Runtime Prediction Gate
  run: |
    pip install cetp
    cetp predict \
      --workload-type ML \
      --workload-name ml_resnet \
      --complexity 3 \
      --cpu-cores 64 \
      --memory-gb 256 \
      --disk-type NVMe \
      --fail-on-red
```

If the prediction result is **RED**, CETP exits with a non-zero status code, allowing the pipeline to block deployment automatically.

---

## Training with Organization-Specific Data

Validate your dataset:

```bash
cetp validate --data your_data.csv
```

Train a custom model:

```bash
cetp train \
  --data your_data.csv \
  --sla your_sla.json
```

After training, all future predictions automatically use the custom model.

---

## SLA Configuration

Define SLA thresholds in a JSON configuration file.

```json
{
  "ML": {
    "sla_runtime_sec": 20.0,
    "warn_at_sec": 15.0
  },
  "DB": {
    "sla_runtime_sec": 5.0,
    "warn_at_sec": 3.5
  },
  "WEB": {
    "sla_runtime_sec": 1.0,
    "warn_at_sec": 0.7
  }
}
```

Use the configuration during prediction:

```bash
cetp predict --sla sla.json
```

---

## Docker Usage

Start the REST API:

```bash
docker pull ghcr.io/harshithjn/cetp:latest

docker run -p 8000:8000 \
  ghcr.io/harshithjn/cetp:latest
```

Example API request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "workload_type": "ML",
    "workload_name": "ml_resnet",
    "workload_complexity": 3,
    "cpu_cores": 64,
    "memory_total_gb": 256,
    "disk_type": "NVMe",
    "cpu_avg_pct": 20,
    "memory_avg_gb": 48
  }'
```

---

## Dataset Format

CETP expects a dataset containing execution-time observations collected from different hardware environments.

View the required schema:

```bash
cetp schema
```

Export a blank template:

```bash
cetp schema --export template.csv
```

Supported workload categories:

| Category | Examples |
|-----------|-----------|
| ML | ResNet50 Inference, BERT Inference |
| DB | TPC-H Query 3, TPC-H Query 6 |
| WEB | Low-Concurrency HTTP, High-Concurrency HTTP |

---

## Development Setup

```bash
git clone https://github.com/harshithjn/CETP.git

cd CETP

python -m venv venv

source venv/bin/activate

pip install -e ".[dev]"

pytest
```

---

## Technology Stack

- XGBoost — Runtime prediction model
- SHAP — Prediction explainability
- FastAPI — REST API backend
- Click — Command-line interface
- psutil — System profiling and hardware metrics

---

## Research Context

This project was developed as a capstone research project at PES University under the guidance of Dr. Nivedita Kasturi.

**Paper Title:**  
*Proposed Framework for Cross-Environment Execution Time Prediction*

---

## License

MIT License
