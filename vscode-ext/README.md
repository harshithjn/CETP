# CETP — Runtime Prediction for VS Code

Predict how long your code will take to run in production before you deploy.

## Features

- Right-click any Python, Java, or C++ file → CETP: Predict Runtime
- Specify production server specs (CPU cores, RAM, disk type) interactively
- See predicted runtime, confidence interval, and SLA status in the sidebar
- Top SHAP feature explanations show WHY the prediction is what it is
- Status bar item showing CETP is active

## Requirements

CETP CLI must be installed:
```bash
pip install cetp
```

## Usage

1. Right-click a .py file in the Explorer
2. Select CETP: Predict Runtime
3. Choose workload type (ML, DB, WEB)
4. Choose workload name
5. Choose complexity level (1-5)
6. Choose auto-profile or specify production specs
7. See results in the sidebar

## Settings

- cetp.useLocalCli: Use local CLI (default: true)
- cetp.apiEndpoint: API server URL (default: http://localhost:8000)
- cetp.defaultWorkloadType: Default workload type
- cetp.defaultComplexity: Default complexity level (1-5)
- cetp.slaFilePath: Path to custom sla.json
