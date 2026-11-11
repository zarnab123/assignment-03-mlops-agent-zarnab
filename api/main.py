from contextlib import asynccontextmanager
import os
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(PROJECT_ROOT / "models" / "breast_cancer_classifier.joblib"),
    )
)
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="Exactly 30 breast-cancer feature values.",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file was not found: {MODEL_PATH}")

    app.state.model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(
    title="MLOps Agent Inference API",
    version=APP_VERSION,
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict:
    return {
        "service": "mlops-agent-inference-api",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy",
        "model": "BreastCancerClassifier",
        "version": APP_VERSION,
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    try:
        model = app.state.model
        values = np.array([request.features])
        prediction = int(model.predict(values)[0])
        probability = float(model.predict_proba(values)[0][1])

        return {
            "prediction": prediction,
            "label": "benign" if prediction == 1 else "malignant",
            "benign_probability": round(probability, 4),
            "model": "BreastCancerClassifier",
            "version": APP_VERSION,
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
