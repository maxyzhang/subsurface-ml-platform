from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def validate_evaluation_inputs(
    actual: pd.Series,
    predicted: pd.Series,
) -> None:
    """Validate classification evaluation inputs."""

    if actual.empty:
        raise ValueError("Actual target is empty")

    if predicted.empty:
        raise ValueError("Predicted target is empty")

    if len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted lengths do not match"
        )

    if actual.isna().any():
        raise ValueError("Actual target contains missing values")

    if predicted.isna().any():
        raise ValueError("Predicted target contains missing values")


def calculate_classification_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict[str, float]:
    """Calculate overall multiclass classification metrics."""

    validate_evaluation_inputs(
        actual,
        predicted,
    )

    return {
        "accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(actual, predicted)
        ),
        "macro_f1": float(
            f1_score(
                actual,
                predicted,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                actual,
                predicted,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def build_classification_report(
    actual: pd.Series,
    predicted: pd.Series,
    labels: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Return precision, recall and F1 metrics as a DataFrame."""

    validate_evaluation_inputs(
        actual,
        predicted,
    )

    report = classification_report(
        actual,
        predicted,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    return (
        pd.DataFrame(report)
        .transpose()
        .reset_index()
        .rename(columns={"index": "class"})
    )


def build_confusion_matrix(
    actual: pd.Series,
    predicted: pd.Series,
    labels: Sequence[int] | None = None,
    normalize: str | None = None,
) -> pd.DataFrame:
    """Return a labeled confusion matrix."""

    validate_evaluation_inputs(
        actual,
        predicted,
    )

    if labels is None:
        labels = sorted(
            set(actual.unique())
            | set(predicted.unique())
        )

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=labels,
        normalize=normalize,
    )

    row_names = [
        f"actual_{label}"
        for label in labels
    ]

    column_names = [
        f"predicted_{label}"
        for label in labels
    ]

    return pd.DataFrame(
        matrix,
        index=row_names,
        columns=column_names,
    )


def save_metrics_json(
    metrics: dict[str, float],
    output_path: Path,
) -> None:
    """Save model metrics as JSON."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
            sort_keys=True,
        )


def save_confusion_matrix_plot(
    matrix: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """Save a confusion-matrix heatmap without extra dependencies."""

    if matrix.empty:
        raise ValueError("Confusion matrix is empty")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(10, 8)
    )

    image = axis.imshow(
        matrix.to_numpy(),
        aspect="auto",
    )

    figure.colorbar(
        image,
        ax=axis,
    )

    axis.set_title(title)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")

    axis.set_xticks(
        np.arange(len(matrix.columns))
    )
    axis.set_yticks(
        np.arange(len(matrix.index))
    )

    axis.set_xticklabels(
        matrix.columns,
        rotation=45,
        ha="right",
    )
    axis.set_yticklabels(
        matrix.index,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=150,
    )

    plt.close(figure)


def save_feature_importance_plot(
    feature_importances: pd.DataFrame,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Save a horizontal feature-importance chart."""

    required_columns = {
        "feature",
        "importance",
    }

    if not required_columns.issubset(
        feature_importances.columns
    ):
        raise ValueError(
            "Feature importance data must contain "
            "feature and importance"
        )

    if top_n <= 0:
        raise ValueError("top_n must be greater than zero")

    plot_data = (
        feature_importances
        .head(top_n)
        .sort_values(
            "importance",
            ascending=True,
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(10, 7)
    )

    axis.barh(
        plot_data["feature"],
        plot_data["importance"],
    )

    axis.set_title(
        "Random Forest Feature Importance"
    )
    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=150,
    )

    plt.close(figure)