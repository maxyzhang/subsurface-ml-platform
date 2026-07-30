from pathlib import Path

import pandas as pd
import pytest

from subsurface_ml.evaluation import (
    build_classification_report,
    build_confusion_matrix,
    calculate_classification_metrics,
    save_confusion_matrix_plot,
    save_feature_importance_plot,
    save_metrics_json,
    validate_evaluation_inputs,
)


@pytest.fixture
def prediction_data() -> tuple[pd.Series, pd.Series]:
    actual = pd.Series(
        [0, 0, 1, 1, 2, 2],
        name="TARGET",
    )

    predicted = pd.Series(
        [0, 1, 1, 1, 2, 0],
        name="PREDICTION",
    )

    return actual, predicted


def test_validate_evaluation_inputs(
    prediction_data: tuple[pd.Series, pd.Series],
) -> None:
    actual, predicted = prediction_data

    validate_evaluation_inputs(
        actual,
        predicted,
    )


def test_validate_evaluation_inputs_rejects_length() -> None:
    actual = pd.Series([0, 1])
    predicted = pd.Series([0])

    with pytest.raises(
        ValueError,
        match="lengths",
    ):
        validate_evaluation_inputs(
            actual,
            predicted,
        )


def test_calculate_classification_metrics(
    prediction_data: tuple[pd.Series, pd.Series],
) -> None:
    actual, predicted = prediction_data

    metrics = calculate_classification_metrics(
        actual,
        predicted,
    )

    assert set(metrics) == {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    }

    assert all(
        0 <= value <= 1
        for value in metrics.values()
    )


def test_build_classification_report(
    prediction_data: tuple[pd.Series, pd.Series],
) -> None:
    actual, predicted = prediction_data

    report = build_classification_report(
        actual,
        predicted,
    )

    assert not report.empty

    assert {
        "class",
        "precision",
        "recall",
        "f1-score",
        "support",
    }.issubset(report.columns)


def test_build_confusion_matrix(
    prediction_data: tuple[pd.Series, pd.Series],
) -> None:
    actual, predicted = prediction_data

    matrix = build_confusion_matrix(
        actual,
        predicted,
        labels=[0, 1, 2],
    )

    assert matrix.shape == (3, 3)
    assert matrix.to_numpy().sum() == 6


def test_save_metrics_json(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "metrics.json"

    save_metrics_json(
        {
            "accuracy": 0.75,
            "macro_f1": 0.70,
        },
        output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_confusion_matrix_plot(
    prediction_data: tuple[pd.Series, pd.Series],
    tmp_path: Path,
) -> None:
    actual, predicted = prediction_data

    matrix = build_confusion_matrix(
        actual,
        predicted,
        labels=[0, 1, 2],
        normalize="true",
    )

    output_path = (
        tmp_path / "confusion_matrix.png"
    )

    save_confusion_matrix_plot(
        matrix,
        output_path,
        title="Test Confusion Matrix",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_feature_importance_plot(
    tmp_path: Path,
) -> None:
    importances = pd.DataFrame(
        {
            "feature": [
                "GR",
                "RHOB",
                "NPHI",
            ],
            "importance": [
                0.5,
                0.3,
                0.2,
            ],
        }
    )

    output_path = (
        tmp_path / "feature_importance.png"
    )

    save_feature_importance_plot(
        importances,
        output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0 