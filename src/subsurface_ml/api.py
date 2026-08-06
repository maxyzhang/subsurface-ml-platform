from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel, ConfigDict, Field

from subsurface_ml.config import MODEL_DIR 
from subsurface_ml.modeling import load_model
from pathlib import Path


BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    /"processed"
    / "X_train.parquet"
)

@lru_cache(maxsize=1)
def get_expected_features() -> list[str]:
    """Load training feature names in the original fitted order. """

    if not TRAIN_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Training feature file not found: {TRAIN_FEATURES_PATH}"
        )

    training_features = pd.read_parquet(
        TRAIN_FEATURES_PATH
    )

    return training_features.columns.to_list()

class PredictionRequest(BaseModel):
    """Well-log features used for one lithology prediction."""

    model_config = ConfigDict(extra="forbid")

    GR: float | None = Field(default=None)
    RHOB: float | None = Field(default=None)
    NPHI: float | None = Field(default=None)
    DTC: float | None = Field(default=None)
    RDEP: float | None = Field(default=None)
    CALI: float | None = Field(default=None)
    SP: float | None = Field(default=None)
    PEF: float | None = Field(default=None)
    RMED: float | None = Field(default=None)
    DEPTH_MD: float | None = Field(default=None)


class PredictionResponse(BaseModel):
    """Prediction returned by the model-serving API."""

    predicted_class: int | str
    model_path: str


@lru_cache(maxsize=1)
def get_model() -> Any:
    """Load and cache the selected best model."""

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Best model file not found: {BEST_MODEL_PATH}"
        )

    return load_model(BEST_MODEL_PATH)


app = FastAPI(
    title="Subsurface ML Prediction API",
    version="0.1.0",
    description=(
        "Serve lithology predictions using the automatically "
        "selected best model."
    ),
)


@app.get("/health")
def health() -> dict[str, str]:
    """Return API health status."""

    return {
        "status": "ok",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
) -> PredictionResponse:
    """Predict one lithology class from well-log features."""

    feature_values = request.model_dump()

    if all(
        value is None
        for value in feature_values.values()
    ):
        raise HTTPException(
            status_code=422,
            detail="At least one feature value is required.",
        )

    try:
        model = get_model()

        expected_features = get_expected_features()

        feature_frame = pd.DataFrame(
            [
                {
                    feature_name: feature_values.get(feature_name)
                    for feature_name in expected_features
                }
            ],
            columns=expected_features,
        )

        prediction = model.predict(feature_frame)[0]

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=422,
            detail=f"Prediction failed: {error}",
        ) from error

    if hasattr(prediction, "item"):
        prediction = prediction.item()

    return PredictionResponse(
        predicted_class=prediction,
        model_path=str(BEST_MODEL_PATH),
    )