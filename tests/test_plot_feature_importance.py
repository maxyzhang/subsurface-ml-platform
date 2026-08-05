from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier 
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline

import scripts.plot_feature_importance as feature_plot


def build_fitted_random_forest_pipeline() -> Pipeline:
    """Create a small fitted Random Forest pipeline for testing."""

    features = pd.DataFrame(
        {
            "GR": [40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
            "RHOB": [2.10, 2.20, 2.30, 2.40, 2.50, 2.60],
            "NPHI": [0.10, 0.12, 0.14, 0.20, 0.22, 0.24],
        }
    )

    target = pd.Series(
        [0, 0, 0, 1, 1, 1],
    )

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=10,
                    max_depth=3,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )

    model.fit(
        features,
        target,
    )

    return model


def test_load_best_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_fitted_random_forest_pipeline()

    model_path = tmp_path / "best_model.joblib"

    joblib.dump(
        model,
        model_path,
    )

    monkeypatch.setattr(
        feature_plot,
        "BEST_MODEL_PATH",
        model_path,
    )

    loaded_model = feature_plot.load_best_model()

    assert isinstance(
        loaded_model,
        Pipeline,
    )

    assert list(
        loaded_model.named_steps
    ) == [
        "imputer",
        "classifier",
    ]


def test_build_feature_importance_dataframe() -> None:
    model = build_fitted_random_forest_pipeline()

    feature_names = list(
        model.feature_names_in_
    )

    importance_df = (
        feature_plot.build_feature_importance_dataframe(
            model,
            feature_names,
        )
    )

    assert isinstance(
        importance_df,
        pd.DataFrame,
    )

    assert list(
        importance_df.columns
    ) == [
        "feature",
        "importance",
    ]

    assert not importance_df.empty

    assert importance_df[
        "importance"
    ].is_monotonic_decreasing

    assert importance_df[
        "importance"
    ].sum() == pytest.approx(
        1.0
    )


def test_save_feature_importances(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = (
        tmp_path
        / "reports"
        / "models"
        / "feature_importances.csv"
    )

    monkeypatch.setattr(
        feature_plot,
        "IMPORTANCE_CSV",
        output_path,
    )

    importance_df = pd.DataFrame(
        {
            "feature": [
                "GR",
                "RHOB",
                "NPHI",
            ],
            "importance": [
                0.50,
                0.30,
                0.20,
            ],
        }
    )

    saved_path = (
        feature_plot.save_feature_importances(
            importance_df
        )
    )

    assert saved_path == output_path
    assert output_path.exists()

    saved_df = pd.read_csv(
        output_path
    )

    pd.testing.assert_frame_equal(
        saved_df,
        importance_df,
    )


def test_save_feature_importances_creates_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = (
        tmp_path
        / "new"
        / "nested"
        / "feature_importances.csv"
    )

    monkeypatch.setattr(
        feature_plot,
        "IMPORTANCE_CSV",
        output_path,
    )

    importance_df = pd.DataFrame(
        {
            "feature": ["GR"],
            "importance": [1.0],
        }
    )

    assert not output_path.parent.exists()

    feature_plot.save_feature_importances(
        importance_df
    )

    assert output_path.parent.exists()
    assert output_path.exists()


def test_plot_feature_importances(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    figure_dir = (
        tmp_path
        / "reports"
        / "figures"
    )

    figure_path = (
        figure_dir
        / "feature_importance.png"
    )

    monkeypatch.setattr(
        feature_plot,
        "FIGURE_DIR",
        figure_dir,
    )

    monkeypatch.setattr(
        feature_plot,
        "FIGURE_PATH",
        figure_path,
    )

    importance_df = pd.DataFrame(
        {
            "feature": [
                "GR",
                "RHOB",
                "NPHI",
            ],
            "importance": [
                0.50,
                0.30,
                0.20,
            ],
        }
    )

    saved_path = (
        feature_plot.plot_feature_importances(
            importance_df
        )
    )

    assert saved_path == figure_path
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0


def test_plot_feature_importances_creates_figure_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    figure_dir = (
        tmp_path
        / "missing"
        / "figures"
    )

    figure_path = (
        figure_dir
        / "feature_importance.png"
    )

    monkeypatch.setattr(
        feature_plot,
        "FIGURE_DIR",
        figure_dir,
    )

    monkeypatch.setattr(
        feature_plot,
        "FIGURE_PATH",
        figure_path,
    )

    importance_df = pd.DataFrame(
        {
            "feature": ["GR"],
            "importance": [1.0],
        }
    )

    assert not figure_dir.exists()

    feature_plot.plot_feature_importances(
        importance_df
    )

    assert figure_dir.exists()
    assert figure_path.exists()