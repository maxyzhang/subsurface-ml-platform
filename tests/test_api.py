from unittest.mock import Mock

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from subsurface_ml import api


client = TestClient(api.app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_success(monkeypatch) -> None:
    fake_model = Mock()
    fake_model.predict.return_value = np.array([3])

    monkeypatch.setattr(
        api,
        "get_model",
        lambda: fake_model,
    )

    monkeypatch.setattr(
        api,
        "get_expected_features",
        lambda: [
            "GR",
            "RHOB",
            "NPHI",
            "DTC",
            "RDEP",
            "CALI",
            "SP",
            "PEF",
            "RMED",
            "DEPTH_MD",
        ],
    )

    response = client.post(
        "/predict",
        json={
            "GR": 80,
            "RHOB": 2.45,
            "NPHI": 0.18,
            "DTC": 95,
            "RDEP": 20,
            "CALI": 8.5,
            "SP": -40,
            "PEF": 3.2,
            "RMED": 15,
            "DEPTH_MD": 2500,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["predicted_class"] == 3
    assert body["model_path"].endswith(
        "best_model.joblib"
    )

    feature_frame = fake_model.predict.call_args.args[0]

    assert isinstance(feature_frame, pd.DataFrame)
    assert feature_frame.columns.tolist() == [
        "GR",
        "RHOB",
        "NPHI",
        "DTC",
        "RDEP",
        "CALI",
        "SP",
        "PEF",
        "RMED",
        "DEPTH_MD",
    ]


def test_predict_rejects_empty_request() -> None:
    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "At least one feature value is required."
    )


def test_predict_returns_503_when_model_missing(
    monkeypatch,
) -> None:
    def raise_missing_model():
        raise FileNotFoundError(
            "Best model file not found"
        )

    monkeypatch.setattr(
        api,
        "get_model",
        raise_missing_model,
    )

    response = client.post(
        "/predict",
        json={
            "GR": 80,
        },
    )

    assert response.status_code == 503
    assert "Best model file not found" in (
        response.json()["detail"]
    )


def test_predict_rejects_unknown_feature() -> None:
    response = client.post(
        "/predict",
        json={
            "GR": 80,
            "UNKNOWN_FEATURE": 1,
        },
    )

    assert response.status_code == 422