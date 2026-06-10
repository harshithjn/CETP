"""
CETP REST API: prediction, explanation, health-check, and schema endpoints.
"""
from typing import Optional, List, Dict, Any
import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn

app = FastAPI(
    title="CETP Prediction API",
    description=(
        "Cross-Environment Execution Time Prediction — predicts production runtime "
        "from hardware metrics with SLA-aware flagging"
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    workload_type: str = Field(..., description="Workload class: ML, DB, or WEB")
    workload_name: str = Field(..., description="Workload name e.g. ml_resnet, tpch_q3")
    workload_complexity: int = Field(..., ge=1, le=5, description="Complexity level 1-5")
    cpu_cores: int = Field(..., gt=0, description="Number of CPU cores")
    memory_total_gb: float = Field(..., gt=0, description="Total RAM in GB")
    cpu_avg_pct: float = Field(..., ge=0, le=100, description="Average CPU utilisation %")
    memory_avg_gb: float = Field(..., ge=0, description="Average memory used in GB")
    disk_read_mb: float = Field(0.0, ge=0, description="Disk read in MB")
    disk_write_mb: float = Field(0.0, ge=0, description="Disk write in MB")
    disk_type: str = Field("SSD", description="Disk type: HDD, SSD, or NVMe")
    disk_speed_class: int = Field(2, ge=1, le=3, description="Disk speed class 1-3")
    sla_config: Optional[Dict[str, Any]] = Field(None, description="Custom SLA thresholds")

    @field_validator("workload_type")
    @classmethod
    def validate_workload_type(cls, v: str) -> str:
        if v not in ("ML", "DB", "WEB"):
            raise ValueError("workload_type must be one of ML, DB, WEB")
        return v

    @field_validator("disk_type")
    @classmethod
    def validate_disk_type(cls, v: str) -> str:
        if v not in ("HDD", "SSD", "NVMe"):
            raise ValueError("disk_type must be one of HDD, SSD, NVMe")
        return v


class SHAPFeature(BaseModel):
    feature: str
    shap_value: float
    direction: str  # "increases_runtime" or "decreases_runtime"


class PredictResponse(BaseModel):
    runtime_sec: float = Field(..., description="Predicted runtime in seconds")
    confidence_interval: List[float] = Field(..., description="[lower, upper] bounds")
    sla_flag: str = Field(..., description="GREEN, YELLOW, or RED")
    sla_threshold_sec: float = Field(..., description="SLA limit for this workload type")
    top_shap_features: List[SHAPFeature] = Field(..., description="Top 3 SHAP contributors")
    model_used: str = Field(..., description="base or custom")
    workload_type: str
    workload_name: str


class HealthResponse(BaseModel):
    status: str
    version: str
    model_status: str  # "base_model_loaded", "custom_model_loaded", or "no_model"
    model_path: Optional[str]
    sla_config_loaded: bool
    sla_thresholds: Optional[Dict[str, Any]]


class ExplainRequest(PredictRequest):
    pass


class ExplainResponse(BaseModel):
    base_value: float = Field(..., description="Mean prediction across training set")
    predicted_value: float
    features: List[Dict[str, Any]] = Field(..., description="All features with SHAP values")
    workload_type: str
    workload_name: str


class ErrorResponse(BaseModel):
    error: str
    detail: str
    status_code: int


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_sla_config(custom_config: Optional[dict] = None) -> dict:
    """
    Load SLA config with priority:
    1. custom_config from request body
    2. ~/.cetp/sla.json
    3. sla_defaults.json in project root
    4. hardcoded defaults
    """
    _HARDCODED: Dict[str, Any] = {
        "ML": {"sla_runtime_sec": 20.0, "warn_at_sec": 15.0},
        "DB": {"sla_runtime_sec": 5.0, "warn_at_sec": 3.5},
        "WEB": {"sla_runtime_sec": 1.0, "warn_at_sec": 0.7},
    }
    if custom_config is not None:
        return custom_config
    user_path = Path.home() / ".cetp" / "sla.json"
    if user_path.exists():
        with open(user_path) as fh:
            return json.load(fh)
    project_path = _PROJECT_ROOT / "sla_defaults.json"
    if project_path.exists():
        with open(project_path) as fh:
            return json.load(fh)
    return _HARDCODED


def get_model_status() -> dict:
    """Check which model artefact is available. Returns status and path."""
    from cetp.predictor import BASE_MODEL_PATH, CETP_DIR

    custom_path = CETP_DIR / "model.pkl"
    if custom_path.exists():
        return {"status": "custom_model_loaded", "path": str(custom_path)}
    if BASE_MODEL_PATH.exists():
        return {"status": "base_model_loaded", "path": str(BASE_MODEL_PATH)}
    return {"status": "no_model", "path": None}


def compute_derived_features(request_data: dict) -> dict:
    """
    Adds effective_cpu, memory_pressure, io_intensity to a copy of request_data.
    Uses the same formulas as profiler.py.
    io_intensity defaults to 0.0 when runtime_sec is absent or zero.
    """
    cpu_cores = request_data["cpu_cores"]
    cpu_avg_pct = request_data["cpu_avg_pct"]
    memory_avg_gb = request_data["memory_avg_gb"]
    memory_total_gb = request_data["memory_total_gb"]
    disk_read_mb = request_data.get("disk_read_mb", 0.0)
    disk_write_mb = request_data.get("disk_write_mb", 0.0)
    runtime_sec = request_data.get("runtime_sec", 0.0)

    effective_cpu = round(cpu_cores * (1.0 - cpu_avg_pct / 100.0), 4)
    if memory_total_gb > 0:
        memory_pressure = round(min(max(memory_avg_gb / memory_total_gb, 0.0), 1.0), 4)
    else:
        memory_pressure = 0.0
    io_intensity = (
        round((disk_read_mb + disk_write_mb) / runtime_sec, 4) if runtime_sec > 0 else 0.0
    )

    result = dict(request_data)
    result["effective_cpu"] = effective_cpu
    result["memory_pressure"] = memory_pressure
    result["io_intensity"] = io_intensity
    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """
    Health check endpoint. Always returns 200.
    Shows model status, version, and loaded SLA thresholds.
    Use this to verify the API is running and check what model is active.
    """
    try:
        model_info = get_model_status()
        sla_conf = load_sla_config()
        return HealthResponse(
            status="ok",
            version="0.1.0",
            model_status=model_info["status"],
            model_path=model_info["path"],
            sla_config_loaded=True,
            sla_thresholds=sla_conf,
        )
    except Exception:
        return HealthResponse(
            status="ok",
            version="0.1.0",
            model_status="no_model",
            model_path=None,
            sla_config_loaded=False,
            sla_thresholds=None,
        )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Predict production runtime for a workload.

    Provide the target production environment's hardware specifications
    and the workload characteristics. Returns predicted runtime in seconds,
    a confidence interval, an SLA flag (GREEN/YELLOW/RED), and the top
    three SHAP features explaining the prediction.

    Returns 503 if no model artefact is available.
    Returns 422 if request validation fails.
    """
    try:
        from cetp.predictor import CETPPredictor

        feature_dict = compute_derived_features(
            request.model_dump(exclude={"sla_config"})
        )
        sla_conf = load_sla_config(request.sla_config)

        predictor = CETPPredictor()
        result = predictor.predict(feature_dict)
        shap_result = predictor.get_shap_explanation(feature_dict)

        wt = request.workload_type
        sla_threshold = sla_conf.get(wt, {}).get("sla_runtime_sec", 0.0)

        top_shap = [
            SHAPFeature(
                feature=item["feature"],
                shap_value=item["shap_value"],
                direction=(
                    "increases_runtime" if item["shap_value"] > 0 else "decreases_runtime"
                ),
            )
            for item in shap_result[:3]
        ]

        model_info = get_model_status()
        model_used = "custom" if model_info["status"] == "custom_model_loaded" else "base"

        return PredictResponse(
            runtime_sec=result["predicted_sec"],
            confidence_interval=[result["lower_sec"], result["upper_sec"]],
            sla_flag=result["sla_flag"],
            sla_threshold_sec=sla_threshold,
            top_shap_features=top_shap,
            model_used=model_used,
            workload_type=request.workload_type,
            workload_name=request.workload_name,
        )
    except (NotImplementedError, FileNotFoundError):
        return JSONResponse(
            status_code=503,
            content={
                "error": "Model not available",
                "detail": (
                    "No model artefact found. Add base_model.pkl to model/artifacts/ "
                    "or run cetp train."
                ),
                "status_code": 503,
            },
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Prediction failed",
                "detail": str(exc),
                "status_code": 500,
            },
        )


@app.get("/predict/profile", tags=["Prediction"])
async def predict_profile():
    """
    Returns the hardware profile of the machine running this API server.
    Useful for verifying which environment the API is deployed on.
    """
    from cetp.profiler import get_static_profile

    return get_static_profile()


@app.post("/explain", response_model=ExplainResponse, tags=["Explanation"])
async def explain(request: ExplainRequest):
    """
    Get a full SHAP explanation for a prediction.

    Returns the base value (mean prediction), the predicted value,
    and SHAP values for ALL features — not just the top 3.
    Use this when you want to understand exactly why the model
    made a particular prediction.

    Returns 503 if no model artefact is available.
    """
    try:
        from cetp.predictor import CETPPredictor

        feature_dict = compute_derived_features(
            request.model_dump(exclude={"sla_config"})
        )

        predictor = CETPPredictor()
        result = predictor.predict(feature_dict)
        shap_result = predictor.get_shap_explanation(feature_dict)

        features = [
            {
                "feature": item["feature"],
                "value": item.get("value", 0.0),
                "shap_value": item["shap_value"],
                "direction": (
                    "increases_runtime" if item["shap_value"] > 0 else "decreases_runtime"
                ),
            }
            for item in shap_result
        ]

        return ExplainResponse(
            base_value=0.0,
            predicted_value=result["predicted_sec"],
            features=features,
            workload_type=request.workload_type,
            workload_name=request.workload_name,
        )
    except (NotImplementedError, FileNotFoundError):
        return JSONResponse(
            status_code=503,
            content={
                "error": "Model not available",
                "detail": (
                    "No model artefact found. Add base_model.pkl to model/artifacts/ "
                    "or run cetp train."
                ),
                "status_code": 503,
            },
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Explanation failed",
                "detail": str(exc),
                "status_code": 500,
            },
        )


@app.get("/schema", tags=["System"])
async def schema():
    """
    Returns the CETP dataset schema.
    Describes all required columns, valid values, and constraints.
    Use this when preparing data for cetp train.
    """
    from cetp.validator import get_schema_info

    return get_schema_info()


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    label = "Not found" if exc.status_code == 404 else str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": label, "detail": str(exc.detail), "status_code": exc.status_code},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "Validation error", "detail": str(exc), "status_code": 422},
    )


@app.exception_handler(Exception)
async def internal_error_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
            "status_code": 500,
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
