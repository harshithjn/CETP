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

from cetp.workload_config import VALID_MODELS

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
    model_name: str = Field(
        ..., description="ML workload: resnet18, resnet50, mobilenet, or distilbert"
    )
    complexity_level: int = Field(..., ge=1, le=5, description="Workload complexity level 1-5")
    cpu_count: int = Field(..., gt=0, description="Target machine CPU core count")
    total_memory_mb: float = Field(..., gt=0, description="Target machine total RAM in MB")
    sla_config: Optional[Dict[str, Any]] = Field(
        None, description="Custom per-model SLA thresholds, overrides defaults"
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if v not in VALID_MODELS:
            raise ValueError(f"model_name must be one of {VALID_MODELS}")
        return v


class SHAPFeature(BaseModel):
    feature: str
    shap_value: float
    direction: str  # "increases_runtime" or "decreases_runtime"


class PredictResponse(BaseModel):
    predicted_runtime_sec: float = Field(..., description="Predicted runtime in seconds")
    confidence_interval: List[float] = Field(..., description="[lower, upper] 90% bounds")
    confidence_status: str = Field(..., description="RELIABLE or EXTRAPOLATED")
    confidence_warnings: List[str] = Field(
        ..., description="Populated when inputs fall outside the training range"
    )
    sla_flag: str = Field(..., description="GREEN, YELLOW, RED, or UNKNOWN")
    sla_threshold_sec: Optional[float] = Field(
        None, description="SLA limit for this model, null if unconfigured"
    )
    top_shap_features: List[SHAPFeature] = Field(..., description="Top 3 SHAP contributors")
    model_artifact: str = Field(..., description="base or custom — which model artefact was used")
    model_used: str
    complexity_level: int


class HealthResponse(BaseModel):
    status: str
    version: str
    model_status: str  # "base_model_loaded", "custom_model_loaded", or "no_model"
    model_path: Optional[str]
    sla_config_loaded: bool
    sla_thresholds: Optional[Dict[str, Any]]
    training_range: Optional[Dict[str, Any]] = Field(
        None, description="Hardware/model ranges the active model artefact was trained on"
    )


class ExplainRequest(PredictRequest):
    pass


class ExplainResponse(BaseModel):
    base_value: float = Field(..., description="Explainer's expected output over the training set")
    predicted_runtime_sec: float
    confidence_status: str = Field(..., description="RELIABLE or EXTRAPOLATED")
    features: List[Dict[str, Any]] = Field(..., description="All features with SHAP values")
    model_used: str
    complexity_level: int


class ErrorResponse(BaseModel):
    error: str
    detail: str
    status_code: int


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def load_sla_config(custom_config: Optional[dict] = None) -> dict:
    """
    Load per-model SLA config with priority:
    1. custom_config from request body
    2. sla_defaults.json in project root
    3. hardcoded defaults (mirrors the shipped sla_defaults.json)
    """
    _HARDCODED: Dict[str, Any] = {
        "resnet18": {"sla_runtime_sec": 393.8, "warn_at_sec": 234.7},
        "resnet50": {"sla_runtime_sec": 431.9, "warn_at_sec": 237.4},
        "mobilenet": {"sla_runtime_sec": 587.9, "warn_at_sec": 249.2},
        "distilbert": {"sla_runtime_sec": 202.9, "warn_at_sec": 183.7},
    }
    if custom_config is not None:
        return custom_config
    project_path = _PROJECT_ROOT / "sla_defaults.json"
    if project_path.exists():
        with open(project_path) as fh:
            return json.load(fh)
    return _HARDCODED


def get_model_status() -> dict:
    """Check which model artefact is available. Returns status and path."""
    from cetp.predictor import BASE_MODEL_PATH, CUSTOM_MODEL_PATH

    if CUSTOM_MODEL_PATH.exists():
        return {"status": "custom_model_loaded", "path": str(CUSTOM_MODEL_PATH)}
    if BASE_MODEL_PATH.exists():
        return {"status": "base_model_loaded", "path": str(BASE_MODEL_PATH)}
    return {"status": "no_model", "path": None}


def get_training_range() -> Optional[dict]:
    """
    Load the training_range.json for whichever model artefact is active
    (custom takes priority over base, matching CETPPredictor's own resolution
    order). Returns None if neither file exists.
    """
    from cetp.predictor import BASE_TRAINING_RANGE_PATH, CUSTOM_TRAINING_RANGE_PATH

    path = (
        CUSTOM_TRAINING_RANGE_PATH
        if CUSTOM_TRAINING_RANGE_PATH.exists()
        else BASE_TRAINING_RANGE_PATH
    )
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


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
        training_range = get_training_range()
        return HealthResponse(
            status="ok",
            version="0.1.0",
            model_status=model_info["status"],
            model_path=model_info["path"],
            sla_config_loaded=True,
            sla_thresholds=sla_conf,
            training_range=training_range,
        )
    except Exception:
        return HealthResponse(
            status="ok",
            version="0.1.0",
            model_status="no_model",
            model_path=None,
            sla_config_loaded=False,
            sla_thresholds=None,
            training_range=None,
        )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Predict production runtime for an ML inference workload.

    Provide the target production environment's hardware specifications
    and the workload characteristics. Returns predicted runtime in seconds,
    a confidence interval, an SLA flag (GREEN/YELLOW/RED/UNKNOWN), and the
    top three SHAP features explaining the prediction.

    Returns 503 if no model artefact is available.
    Returns 422 if request validation fails.
    """
    try:
        from cetp.predictor import CETPPredictor

        predictor = CETPPredictor()
        result = predictor.predict(
            request.model_name, request.complexity_level, request.cpu_count, request.total_memory_mb
        )
        shap_result = predictor.get_shap_explanation(
            request.model_name, request.complexity_level, request.cpu_count, request.total_memory_mb
        )

        sla_conf = load_sla_config(request.sla_config)
        thresholds = sla_conf.get(request.model_name)
        if thresholds is not None:
            upper = result["confidence_interval"][1]
            warn_at = thresholds["warn_at_sec"]
            sla_limit = thresholds["sla_runtime_sec"]
            if upper <= warn_at:
                sla_flag = "GREEN"
            elif upper <= sla_limit:
                sla_flag = "YELLOW"
            else:
                sla_flag = "RED"
            sla_threshold: Optional[float] = sla_limit
        else:
            sla_flag = "UNKNOWN"
            sla_threshold = None

        top_shap = [
            SHAPFeature(
                feature=item["feature"],
                shap_value=item["shap_value"],
                direction=item["direction"],
            )
            for item in shap_result["features"][:3]
        ]

        model_info = get_model_status()
        model_artifact = "custom" if model_info["status"] == "custom_model_loaded" else "base"

        return PredictResponse(
            predicted_runtime_sec=result["predicted_runtime_sec"],
            confidence_interval=result["confidence_interval"],
            confidence_status=result["confidence_status"],
            confidence_warnings=result["confidence_warnings"],
            sla_flag=sla_flag,
            sla_threshold_sec=sla_threshold,
            top_shap_features=top_shap,
            model_artifact=model_artifact,
            model_used=result["model_used"],
            complexity_level=result["complexity_level"],
        )
    except FileNotFoundError:
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
    except ValueError as exc:
        return JSONResponse(
            status_code=422,
            content={"error": "Validation error", "detail": str(exc), "status_code": 422},
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

    Returns the predicted runtime and SHAP values for ALL features —
    not just the top 3. Use this when you want to understand exactly
    why the model made a particular prediction.

    Returns 503 if no model artefact is available.
    """
    try:
        from cetp.predictor import CETPPredictor

        predictor = CETPPredictor()
        result = predictor.predict(
            request.model_name, request.complexity_level, request.cpu_count, request.total_memory_mb
        )
        # top_n set high enough to always cover every post-aggregation feature
        # (cpu_count, total_memory_mb, complexity_level, batch_size,
        # num_iterations, plus the aggregated model feature — 6 total).
        shap_result = predictor.get_shap_explanation(
            request.model_name,
            request.complexity_level,
            request.cpu_count,
            request.total_memory_mb,
            top_n=100,
        )

        features = [
            {
                "feature": item["feature"],
                "shap_value": item["shap_value"],
                "direction": item["direction"],
            }
            for item in shap_result["features"]
        ]

        return ExplainResponse(
            base_value=shap_result["base_value"],
            predicted_runtime_sec=result["predicted_runtime_sec"],
            confidence_status=result["confidence_status"],
            features=features,
            model_used=result["model_used"],
            complexity_level=result["complexity_level"],
        )
    except FileNotFoundError:
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
    except ValueError as exc:
        return JSONResponse(
            status_code=422,
            content={"error": "Validation error", "detail": str(exc), "status_code": 422},
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
