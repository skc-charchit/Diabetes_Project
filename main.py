import logging
import hmac
import json
import os
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.ml_pipeline import predict, sha256_file

logger = logging.getLogger(__name__)
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent / "src" / "models" / "diabetes_pipeline.pkl"
)
MODEL_PATH = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
METRICS_PATH = Path(
    os.getenv(
        "METRICS_PATH",
        Path(__file__).resolve().parent / "src" / "models" / "training_metrics.json",
    )
)

try:
    artifact_metadata = json.loads(METRICS_PATH.read_text())
    expected_hash = artifact_metadata.get("model_sha256")
    if not expected_hash:
        raise ValueError("Model metadata does not contain model_sha256")
    actual_hash = sha256_file(MODEL_PATH)
    if not hmac.compare_digest(actual_hash, expected_hash):
        raise ValueError("Model checksum does not match model metadata")
    diabetes_model = joblib.load(MODEL_PATH)
    if not all(
        hasattr(diabetes_model, method) for method in ("predict", "predict_proba")
    ):
        raise TypeError("Model artifact must implement predict and predict_proba")
    logger.info("Loaded diabetes model from %s", MODEL_PATH)
except Exception as error:
    raise RuntimeError(f"Unable to load model from {MODEL_PATH}") from error

app = FastAPI(title="Diabetes Prediction API", version="1.0.0")


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Pregnancies: int = Field(ge=0, le=20)
    Glucose: float = Field(ge=0, le=300)
    BloodPressure: float = Field(ge=0, le=200)
    SkinThickness: float = Field(ge=0, le=100)
    Insulin: float = Field(ge=0, le=1000)
    BMI: float = Field(ge=0, le=100)
    DiabetesPedigreeFunction: float = Field(ge=0, le=3)
    Age: int = Field(ge=1, le=120)


class HealthResponse(BaseModel):
    status: str
    model: str
    pipeline_version: str
    artifact_sha256: str


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Welcome to the Diabetes Prediction API"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model=MODEL_PATH.name,
        pipeline_version=artifact_metadata["pipeline_version"],
        artifact_sha256=artifact_metadata["model_sha256"],
    )


@app.post("/diabetes_prediction")
def diabetes_prediction(request: PredictRequest) -> dict[str, float | int | str]:
    try:
        values = request.model_dump()
        prediction, probability = predict(diabetes_model, values)
        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "result": "The person is diabetic."
            if prediction
            else "The person is not diabetic.",
        }
    except Exception as error:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.") from error
