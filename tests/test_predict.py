from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.predict as prediction


def test_build_prediction_output() -> None:
    metadata = pd.DataFrame(
        {
            "WELL": ["Well_A", "Well_A", "Well_B"],
            "DEPTH_MD": [1000.0, 1000.5, 1200.0],
        }
    )

    predictions = pd.Series(
        [1, 2, 3],
        name="predicted_class",
    )

    output = prediction.build_prediction_output(
        metadata,
        predictions,
    )

    assert len(output) == 3

    assert list(output.columns) == [
        "WELL",
        "DEPTH_MD",
        "predicted_class",
    ]

    assert list(output["predicted_class"]) == [
        1,
        2,
        3,
    ]

    assert list(output["WELL"]) == [
        "Well_A",
        "Well_A",
        "Well_B",
    ]


def test_build_prediction_output_resets_indexes() -> None:
    metadata = pd.DataFrame(
        {
            "WELL": ["Well_A", "Well_B"],
            "DEPTH_MD": [1000.0, 1200.0],
        },
        index=[10, 20],
    )

    predictions = pd.Series(
        [4, 5],
        index=[100, 200],
    )

    output = prediction.build_prediction_output(
        metadata,
        predictions,
    )

    assert list(output.index) == [0, 1]
    assert list(output["predicted_class"]) == [4, 5]


def test_build_prediction_output_rejects_length_mismatch() -> None:
    metadata = pd.DataFrame(
        {
            "WELL": ["Well_A", "Well_B"],
        }
    )

    predictions = pd.Series([1])

    with pytest.raises(
        ValueError,
        match="same number of rows",
    ):
        prediction.build_prediction_output(
            metadata,
            predictions,
        )


def test_build_prediction_output_accepts_numpy_predictions() -> None:
    metadata = pd.DataFrame(
        {
            "WELL": ["Well_A", "Well_B"],
            "DEPTH_MD": [1000.0, 1001.0],
        }
    )

    predictions = np.array([2, 3])

    output = prediction.build_prediction_output(
        metadata,
        predictions,
    )

    assert list(output["predicted_class"]) == [2, 3]


def test_save_predictions(
    tmp_path: Path,
) -> None:
    output = pd.DataFrame(
        {
            "WELL": ["Well_A", "Well_B"],
            "DEPTH_MD": [1000.0, 1200.0],
            "predicted_class": [1, 2],
        }
    )

    output_path = (
        tmp_path
        / "reports"
        / "predictions"
        / "test_predictions.csv"
    )

    prediction.save_predictions(
        output,
        output_path,
    )

    assert output_path.exists()

    saved_output = pd.read_csv(
        output_path
    )

    pd.testing.assert_frame_equal(
        saved_output,
        output,
    )


def test_save_predictions_creates_parent_directory(
    tmp_path: Path,
) -> None:
    output = pd.DataFrame(
        {
            "predicted_class": [1, 2, 3],
        }
    )

    output_path = (
        tmp_path
        / "new_directory"
        / "nested_directory"
        / "predictions.csv"
    )

    assert not output_path.parent.exists()

    prediction.save_predictions(
        output,
        output_path,
    )

    assert output_path.parent.exists()
    assert output_path.exists()