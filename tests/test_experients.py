import json
from pathlib import Path

import pandas as pd
import pytest

from subsurface_ml.experiments import (
    EXPERIMENT_COLUMNS,
    append_experiment,
    build_experiment_record,
)


def test_build_experiment_record() -> None:
    record = build_experiment_record(
        run_type="baseline",
        model="random_forest",
        metrics={
            "accuracy": 0.7159,
            "balanced_accuracy": 0.3314,
            "macro_f1": 0.3164,
            "weighted_f1": 0.7072,
        },
        training_seconds=70.5,
        parameters={
            "n_estimators": 100,
            "max_depth": 20,
        },
        timestamp_utc="2026-08-05T12:00:00+00:00",
    )

    assert record["run_type"] == "baseline"
    assert record["model"] == "random_forest"
    assert record["accuracy"] == pytest.approx(
        0.7159
    )
    assert record["training_seconds"] == pytest.approx(
        70.5
    )

    assert json.loads(record["parameters"]) == {
        "max_depth": 20,
        "n_estimators": 100,
    }


def test_build_experiment_record_supports_optional_values() -> None:
    record = build_experiment_record(
        run_type="baseline",
        model="dummy",
        metrics={},
    )

    assert record["accuracy"] is None
    assert record["best_cv_score"] is None
    assert json.loads(record["parameters"]) == {}


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("run_type", ""),
        ("model", ""),
    ],
)
def test_build_experiment_record_rejects_empty_names(
    field_name: str,
    field_value: str,
) -> None:
    arguments = {
        "run_type": "baseline",
        "model": "random_forest",
        "metrics": {},
    }

    arguments[field_name] = field_value

    with pytest.raises(
        ValueError,
        match=field_name,
    ):
        build_experiment_record(**arguments)


def test_append_experiment_creates_csv(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "experiments.csv"

    record = build_experiment_record(
        run_type="baseline",
        model="random_forest",
        metrics={
            "accuracy": 0.71,
        },
        timestamp_utc="2026-08-05T12:00:00+00:00",
    )

    result = append_experiment(
        record,
        output_path,
    )

    assert result == output_path
    assert output_path.exists()

    saved = pd.read_csv(output_path)

    assert list(saved.columns) == EXPERIMENT_COLUMNS
    assert len(saved) == 1
    assert saved.loc[0, "model"] == "random_forest"


def test_append_experiment_appends_rows(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "experiments.csv"

    first = build_experiment_record(
        run_type="baseline",
        model="dummy",
        metrics={},
        timestamp_utc="2026-08-05T12:00:00+00:00",
    )

    second = build_experiment_record(
        run_type="tuning",
        model="random_forest_tuned",
        metrics={
            "accuracy": 0.73,
        },
        best_cv_score=0.72,
        timestamp_utc="2026-08-05T12:05:00+00:00",
    )

    append_experiment(first, output_path)
    append_experiment(second, output_path)

    saved = pd.read_csv(output_path)

    assert len(saved) == 2
    assert saved["model"].tolist() == [
        "dummy",
        "random_forest_tuned",
    ]


def test_append_experiment_rejects_missing_columns(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "experiments.csv"

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        append_experiment(
            {
                "model": "random_forest",
            },
            output_path,
        )