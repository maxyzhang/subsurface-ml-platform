import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.select_best_model as selection


def test_build_model_leaderboard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics_by_model = {
        "dummy": {
            "accuracy": 0.60,
            "balanced_accuracy": 0.10,
            "macro_f1": 0.08,
            "weighted_f1": 0.45,
            "training_seconds": 0.01,
        },
        "logistic_regression": {
            "accuracy": 0.69,
            "balanced_accuracy": 0.23,
            "macro_f1": 0.25,
            "weighted_f1": 0.62,
            "training_seconds": 100.0,
        },
        "random_forest": {
            "accuracy": 0.72,
            "balanced_accuracy": 0.33,
            "macro_f1": 0.32,
            "weighted_f1": 0.71,
            "training_seconds": 70.0,
        },
        "random_forest_tuned": {
            "accuracy": 0.71,
            "balanced_accuracy": 0.31,
            "macro_f1": 0.30,
            "weighted_f1": 0.70,
            "tuning_seconds": 110.0,
        },
    }

    model_configs: dict[str, dict[str, Path]] = {}

    for model_name, metrics in metrics_by_model.items():
        metrics_path = tmp_path / f"{model_name}_metrics.json"
        model_path = tmp_path / f"{model_name}.joblib"

        metrics_path.write_text(
            json.dumps(metrics),
            encoding="utf-8",
        )
        model_path.write_bytes(b"model")

        model_configs[model_name] = {
            "metrics": metrics_path,
            "model": model_path,
        }

    monkeypatch.setattr(
        selection,
        "MODEL_CONFIGS",
        model_configs,
    )

    leaderboard = selection.build_model_leaderboard()

    assert isinstance(leaderboard, pd.DataFrame)
    assert not leaderboard.empty

    assert list(leaderboard["rank"]) == [1, 2, 3, 4]

    assert list(leaderboard["model"]) == [
        "random_forest",
        "random_forest_tuned",
        "logistic_regression",
        "dummy",
    ]

    assert leaderboard.iloc[0]["balanced_accuracy"] == pytest.approx(
        0.33
    )


def test_select_best_model() -> None:
    leaderboard = pd.DataFrame(
        [
            {
                "rank": 1,
                "model": "model_b",
                "balanced_accuracy": 0.80,
            },
            {
                "rank": 2,
                "model": "model_a",
                "balanced_accuracy": 0.60,
            },
            {
                "rank": 3,
                "model": "model_c",
                "balanced_accuracy": 0.50,
            },
        ]
    )

    best_model = selection.select_best_model(
        leaderboard
    )

    assert best_model == "model_b"


def test_select_best_model_rejects_empty_leaderboard() -> None:
    empty_leaderboard = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="empty leaderboard",
    ):
        selection.select_best_model(
            empty_leaderboard
        )


def test_copy_best_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_model = tmp_path / "source_model.joblib"
    source_model.write_bytes(b"trained model contents")

    output_model_dir = tmp_path / "models"

    monkeypatch.setattr(
        selection,
        "MODEL_DIR",
        output_model_dir,
    )

    monkeypatch.setattr(
        selection,
        "MODEL_CONFIGS",
        {
            "best_candidate": {
                "metrics": tmp_path / "metrics.json",
                "model": source_model,
            }
        },
    )

    copied_path = selection.copy_best_model(
        "best_candidate"
    )

    assert copied_path == output_model_dir / "best_model.joblib"
    assert copied_path.exists()
    assert copied_path.read_bytes() == b"trained model contents"


def test_copy_best_model_rejects_missing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_model = tmp_path / "missing.joblib"

    monkeypatch.setattr(
        selection,
        "MODEL_CONFIGS",
        {
            "missing_candidate": {
                "metrics": tmp_path / "metrics.json",
                "model": missing_model,
            }
        },
    )

    with pytest.raises(
        FileNotFoundError,
        match="Model file not found",
    ):
        selection.copy_best_model(
            "missing_candidate"
        )


def test_save_selection_summary(
    tmp_path: Path,
) -> None:
    leaderboard = pd.DataFrame(
        [
            {
                "rank": 1,
                "model": "random_forest",
                "accuracy": 0.72,
                "balanced_accuracy": 0.33,
                "macro_f1": 0.32,
                "weighted_f1": 0.71,
            },
            {
                "rank": 2,
                "model": "logistic_regression",
                "accuracy": 0.69,
                "balanced_accuracy": 0.23,
                "macro_f1": 0.25,
                "weighted_f1": 0.62,
            },
        ]
    )

    output_path = (
        tmp_path
        / "reports"
        / "best_model_selection.json"
    )

    selection.save_selection_summary(
        leaderboard,
        "random_forest",
        output_path,
    )

    assert output_path.exists()

    summary = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    assert summary["selection_metric"] == "balanced_accuracy"
    assert summary["best_model"] == "random_forest"
    assert summary["best_balanced_accuracy"] == pytest.approx(0.33)
    assert summary["best_accuracy"] == pytest.approx(0.72)
    assert summary["best_macro_f1"] == pytest.approx(0.32)
    assert summary["best_weighted_f1"] == pytest.approx(0.71)