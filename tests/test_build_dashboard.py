import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.build_dashboard as dashboard


def test_load_json(tmp_path: Path) -> None:
    json_path = tmp_path / "metrics.json"

    json_path.write_text(
        json.dumps(
            {
                "best_model": "random_forest",
                "accuracy": 0.72,
            }
        ),
        encoding="utf-8",
    )

    result = dashboard.load_json(json_path)

    assert result["best_model"] == "random_forest"
    assert result["accuracy"] == pytest.approx(0.72)


def test_load_json_rejects_missing_file(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.json"

    with pytest.raises(
        FileNotFoundError,
        match="Required JSON file was not found",
    ):
        dashboard.load_json(missing_path)


def test_load_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "leaderboard.csv"

    expected = pd.DataFrame(
        {
            "model": [
                "random_forest",
                "logistic_regression",
            ],
            "balanced_accuracy": [
                0.33,
                0.23,
            ],
        }
    )

    expected.to_csv(
        csv_path,
        index=False,
    )

    result = dashboard.load_csv(csv_path)

    pd.testing.assert_frame_equal(
        result,
        expected,
    )


def test_load_csv_rejects_missing_file(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(
        FileNotFoundError,
        match="Required CSV file was not found",
    ):
        dashboard.load_csv(missing_path)


def test_copy_dashboard_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source" / "chart.png"

    source_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_path.write_bytes(
        b"fake png contents"
    )

    dashboard_dir = tmp_path / "dashboard"

    monkeypatch.setattr(
        dashboard,
        "DASHBOARD_DIR",
        dashboard_dir,
    )

    copied_path = dashboard.copy_dashboard_image(
        source_path
    )

    assert copied_path == (
        dashboard_dir
        / "images"
        / "chart.png"
    )

    assert copied_path.exists()

    assert copied_path.read_bytes() == (
        b"fake png contents"
    )


def test_copy_dashboard_image_rejects_missing_file(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.png"

    with pytest.raises(
        FileNotFoundError,
        match="Required image was not found",
    ):
        dashboard.copy_dashboard_image(
            missing_path
        )


def test_dataframe_to_html() -> None:
    dataframe = pd.DataFrame(
        {
            "model": ["random_forest"],
            "accuracy": [0.7159],
        }
    )

    result = dashboard.dataframe_to_html(
        dataframe
    )

    assert "<table" in result
    assert "data-table" in result
    assert "random_forest" in result
    assert "0.7159" in result


def test_build_prediction_summary() -> None:
    predictions = pd.DataFrame(
        {
            "predicted_class": [
                1,
                1,
                2,
                3,
            ]
        }
    )

    summary = dashboard.build_prediction_summary(
        predictions
    )

    assert list(summary.columns) == [
        "predicted_class",
        "count",
        "percentage",
    ]

    class_one = summary.loc[
        summary["predicted_class"] == 1
    ].iloc[0]

    assert class_one["count"] == 2
    assert class_one["percentage"] == pytest.approx(
        50.0
    )

    assert summary["count"].sum() == 4

    assert summary["percentage"].sum() == (
        pytest.approx(100.0)
    )


def test_build_prediction_summary_rejects_missing_column() -> None:
    predictions = pd.DataFrame(
        {
            "another_column": [1, 2, 3],
        }
    )

    with pytest.raises(
        ValueError,
        match="predicted_class",
    ):
        dashboard.build_prediction_summary(
            predictions
        )


def test_build_dashboard_html() -> None:
    selection = {
        "selection_metric": "balanced_accuracy",
        "best_model": "random_forest",
        "best_balanced_accuracy": 0.3314,
        "best_accuracy": 0.7159,
        "best_macro_f1": 0.3164,
        "best_weighted_f1": 0.7072,
    }

    leaderboard = pd.DataFrame(
        {
            "rank": [1, 2],
            "model": [
                "random_forest",
                "logistic_regression",
            ],
            "balanced_accuracy": [
                0.3314,
                0.2326,
            ],
        }
    )

    feature_importances = pd.DataFrame(
        {
            "feature": [
                "DEPTH_MD",
                "RHOB",
                "GR",
            ],
            "importance": [
                0.14,
                0.12,
                0.11,
            ],
        }
    )

    prediction_summary = pd.DataFrame(
        {
            "predicted_class": [0, 1],
            "count": [80, 20],
            "percentage": [80.0, 20.0],
        }
    )

    result = dashboard.build_dashboard_html(
        selection,
        leaderboard,
        feature_importances,
        prediction_summary,
        "feature_importance.png",
        "confusion_matrix.png",
    )

    assert "<!DOCTYPE html>" in result
    assert "Subsurface ML Evaluation Dashboard" in result
    assert "random_forest" in result
    assert "balanced_accuracy" in result
    assert "0.7159" in result
    assert "0.3314" in result
    assert "DEPTH_MD" in result
    assert "feature_importance.png" in result
    assert "confusion_matrix.png" in result


def test_build_dashboard_html_escapes_text() -> None:
    selection = {
        "selection_metric": "<metric>",
        "best_model": "<script>alert(1)</script>",
        "best_balanced_accuracy": 0.30,
        "best_accuracy": 0.70,
        "best_macro_f1": 0.25,
        "best_weighted_f1": 0.65,
    }

    empty_table = pd.DataFrame()

    result = dashboard.build_dashboard_html(
        selection,
        empty_table,
        empty_table,
        empty_table,
        "feature.png",
        "confusion.png",
    )

    assert (
        "&lt;script&gt;alert(1)&lt;/script&gt;"
        in result
    )

    assert "<script>alert(1)</script>" not in result
    assert "&lt;metric&gt;" in result