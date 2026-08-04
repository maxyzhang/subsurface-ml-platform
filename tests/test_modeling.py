from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from subsurface_ml.modeling import tune_random_forest

from subsurface_ml.modeling import (
    build_dummy_classifier,
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
    extract_feature_importances,
    fit_classifier,
    load_model,
    predict_classes,
    predict_probabilities,
    save_model,
    validate_training_data,
)


@pytest.fixture
def modeling_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create a small numerical classification dataset."""

    features = pd.DataFrame(
        {
            "GR": [
                45.0,
                50.0,
                55.0,
                80.0,
                85.0,
                90.0,
                None,
                70.0,
                35.0,
            ],
            "RHOB": [
                2.50,
                2.45,
                2.40,
                2.20,
                2.25,
                2.30,
                2.35,
                None,
                2.55,
            ],
            "NPHI": [
                0.10,
                0.12,
                0.14,
                0.30,
                0.28,
                0.26,
                0.20,
                0.22,
                None,
            ],
        }
    )

    target = pd.Series(
        [
            0,
            0,
            0,
            1,
            1,
            1,
            2,
            2,
            2,
        ],
        name="TARGET",
        dtype="int16",
    )

    return features, target


def test_build_dummy_classifier() -> None:
    model = build_dummy_classifier()

    assert model.strategy == "most_frequent"

def test_build_logistic_regression_pipeline() -> None:
    model = build_logistic_regression_pipeline()

    assert isinstance(model, Pipeline)
    assert "imputer" in model.named_steps
    assert "scaler" in model.named_steps
    assert "classifier" in model.named_steps

def test_tune_random_forest_returns_fitted_search() -> None:
    features = pd.DataFrame(
        {
            "GR": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            "RHOB": [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8],
        }
    )
    target = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

    search = tune_random_forest(
        features,
        target,
        n_iter=1,
        cv=2,
        random_state=42,
        n_jobs=1,
    )

    assert hasattr(search, "best_estimator_")
    assert hasattr(search, "best_params_")
    assert hasattr(search, "best_score_")
    assert search.best_estimator_ is not None 

def test_logistic_regression_pipeline_handles_missing_values() -> None:
    features = pd.DataFrame(
        {
            "GR": [10.0, 20.0, np.nan, 40.0, 50.0, 60.0],
            "RHOB": [2.1, np.nan, 2.3, 2.4, 2.5, 2.6],
        }
    )
    target = pd.Series([0, 0, 0, 1, 1, 1])

    model = build_logistic_regression_pipeline(
        max_iter=500,
    )

    fit_classifier(
        model,
        features,
        target,
    )

    predictions = predict_classes(
        model,
        features,
    )

    assert len(predictions) == len(target)
    assert set(predictions).issubset({0, 1}) 

def test_build_random_forest_pipeline() -> None:
    model = build_random_forest_pipeline(
        n_estimators=10,
        max_depth=4,
        n_jobs=1,
    )

    assert isinstance(model, Pipeline)

    assert list(model.named_steps) == [
        "imputer",
        "classifier",
    ]


def test_build_random_forest_pipeline_rejects_estimators() -> None:
    with pytest.raises(
        ValueError,
        match="n_estimators",
    ):
        build_random_forest_pipeline(
            n_estimators=0
        )


def test_validate_training_data(
    modeling_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = modeling_data

    validate_training_data(
        features,
        target,
    )


def test_validate_training_data_rejects_length() -> None:
    features = pd.DataFrame(
        {"GR": [1.0, 2.0]}
    )

    target = pd.Series([0])

    with pytest.raises(
        ValueError,
        match="row counts",
    ):
        validate_training_data(
            features,
            target,
        )


def test_validate_training_data_rejects_text() -> None:
    features = pd.DataFrame(
        {
            "GR": [1.0, 2.0],
            "TEXT": ["A", "B"],
        }
    )

    target = pd.Series([0, 1])

    with pytest.raises(
        ValueError,
        match="Non-numeric",
    ):
        validate_training_data(
            features,
            target,
        )


def test_fit_and_predict_dummy(
    modeling_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = modeling_data

    model = build_dummy_classifier()

    fit_classifier(
        model,
        features,
        target,
    )

    predictions = predict_classes(
        model,
        features,
    )

    assert len(predictions) == len(target)
    assert predictions.name == "PREDICTION"


def test_fit_and_predict_random_forest(
    modeling_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = modeling_data

    model = build_random_forest_pipeline(
        n_estimators=15,
        max_depth=5,
        min_samples_leaf=1,
        n_jobs=1,
    )

    fit_classifier(
        model,
        features,
        target,
    )

    predictions = predict_classes(
        model,
        features,
    )

    probabilities = predict_probabilities(
        model,
        features,
    )

    assert len(predictions) == len(target)
    assert len(probabilities) == len(target)

    assert probabilities.shape[1] == 3

    assert probabilities.sum(axis=1).to_numpy() == pytest.approx(
        1.0
    )


def test_extract_feature_importances(
    modeling_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = modeling_data

    model = build_random_forest_pipeline(
        n_estimators=15,
        max_depth=5,
        min_samples_leaf=1,
        n_jobs=1,
    )

    fit_classifier(
        model,
        features,
        target,
    )

    importances = extract_feature_importances(
        model,
        list(features.columns),
    )

    assert not importances.empty

    assert {
        "feature",
        "importance",
    }.issubset(importances.columns)

    assert importances["importance"].sum() == pytest.approx(
        1.0
    )


def test_save_and_load_model(
    modeling_data: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    features, target = modeling_data

    model = build_dummy_classifier()

    fit_classifier(
        model,
        features,
        target,
    )

    model_path = tmp_path / "model.joblib"

    save_model(
        model,
        model_path,
    )

    loaded_model = load_model(
        model_path,
    )

    predictions = predict_classes(
        loaded_model,
        features,
    )

    assert model_path.exists()
    assert len(predictions) == len(target)


def test_load_model_rejects_missing_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        load_model(
            tmp_path / "missing.joblib"
        )