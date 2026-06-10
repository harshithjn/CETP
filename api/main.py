"""
CETP REST API: exposes prediction, health check, and explanation endpoints.
"""
import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="CETP Prediction API",
    description="Cross-Environment Execution Time Prediction with SLA-aware flagging",
    version="0.1.0",
)

_SLA_PATH = Path(__file__).parent.parent / "sla_defaults.json"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    workload_type: str = Field(
        ..., description="Category of the workload: ML, DB, or WEB", examples=["ML"]
    )
    workload_name: str = Field(
        ..., description="Identifier for the specific workload", examples=["ml_resnet50"]
    )
    workload_complexity: int = Field(
        ..., ge=1, le=5, description="Complexity level 1 (trivial) to 5 (intensive)"
    )
    cpu_cores: Optional[int] = Field(
        None, ge=1, description="Number of physical CPU cores (auto-detected if omitted)"
    )
    memory_total_gb: Optional[float] = Field(
        None, gt=0, description="Total system RAM in GB (auto-detected if omitted)"
    )
    disk_type: Optional[str] = Field(
        None, description="Storage medium type: HDD, SSD, or NVMe (auto-detected if omitted)"
    )
    disk_speed_class: Optional[int] = Field(
        None, ge=1, le=3, description="Disk speed class: HDD=1, SSD=2, NVMe=3"
    )
    cpu_avg_pct: Optional[float] = Field(
        None, ge=0, le=100, description="Average CPU utilisation percentage"
    )
    memory_avg_gb: Optional[float] = Field(
        None, ge=0, description="Average used memory in GB"
    )
    sla_config_path: Optional[str] = Field(
        None, description="Optional path to a custom SLA JSON file on the server"
    )


class SHAPContribution(BaseModel):
    feature: str
    value: float
    shap_value: float


class PredictResponse(BaseModel):
    workload_name: str
    workload_type: str
    predicted_sec: float = Field(..., description="Point estimate of production runtime in seconds")
    lower_sec: float = Field(..., description="Lower bound of the 90% confidence interval")
    upper_sec: float = Field(..., description="Upper bound of the 90% confidence interval")
    sla_flag: str = Field(..., description="SLA traffic-light flag: GREEN, YELLOW, or RED")
    top_contributors: List[SHAPContribution] = Field(
        default_factory=list, description="Top-3 SHAP feature attributions"
    )


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    custom_model: bool
    sla_thresholds: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictResponse, status_code=200)
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Predict production runtime for the given workload feature vector.

    Returns a point estimate, 90% confidence interval, SLA flag, and
    the top-3 SHAP feature attributions.

    **Note:** Returns 501 until the model artefact (base_model.pkl) is available.
    """
    # TODO: replace 501 with real prediction once base_model.pkl is available.
    #       Steps:
    #       1. Build feature_dict from request fields.
    #       2. Auto-detect missing fields via profiler.get_static_profile().
    #       3. predictor = CETPPredictor(sla_config_path=request.sla_config_path)
    #       4. predictor.load_model()
    #       5. result = predictor.predict(feature_dict)
    #       6. shap = predictor.get_shap_explanation(feature_dict)
    #       7. Return PredictResponse(**result, top_contributors=shap)
    raise HTTPException(
        status_code=501,
        detail=(
            "Prediction endpoint not yet available. "
            "The model artefact (model/artifacts/base_model.pkl) has not been trained yet. "
            "Run `cetp train --data <file.csv>` to create a model."
        ),
    )


@app.get("/health", response_model=HealthResponse, status_code=200)
async def health() -> HealthResponse:
    """
    Health check endpoint.

    Returns the API version, model load status, and active SLA thresholds.
    """
    from cetp.predictor import CETPPredictor, BASE_MODEL_PATH, CETP_DIR

    predictor = CETPPredictor()
    model_info = predictor.get_active_model_info()

    sla_thresholds: dict = {}
    if _SLA_PATH.exists():
        with open(_SLA_PATH) as fh:
            sla_thresholds = json.load(fh)

    model_loaded = (
        BASE_MODEL_PATH.exists() or (CETP_DIR / "model.pkl").exists()
    )

    return HealthResponse(
        status="ok",
        version="0.1.0",
        model_loaded=model_loaded,
        custom_model=model_info["custom_model"],
        sla_thresholds=sla_thresholds,
    )


@app.get("/explain", status_code=200)
async def explain(
    workload_type: str,
    workload_name: str,
    workload_complexity: int,
) -> dict:
    """
    Return global SHAP feature importance for the active model.

    **Note:** Returns 501 until the model artefact (base_model.pkl) is available.
    """
    # TODO: implement once base_model.pkl is available.
    #       Steps:
    #       1. Load the model via CETPPredictor.load_model().
    #       2. Build a representative sample dataset for global SHAP.
    #       3. Instantiate CETExplainer and call explain_global().
    raise HTTPException(
        status_code=501,
        detail=(
            "Explanation endpoint not yet available. "
            "The model artefact (model/artifacts/base_model.pkl) has not been trained yet."
        ),
    )
