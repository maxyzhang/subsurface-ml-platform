from pathlib import Path
import json
import shutil

import pandas as pd

from subsurface_ml.config import (
    MODEL_DIR,
    REPORT_DIR,
)


MODEL_REPORT_DIR = REPORT_DIR / "models"

MODEL_CONFIGS = {
    "dummy": {
        "metrics": MODEL_REPORT_DIR
        / "dummy_validation_metrics.json",
        "model": MODEL_DIR
        / "dummy_classifier.joblib",
    },
    "logistic_regression": {
        "metrics": MODEL_REPORT_DIR
        / "logistic_regression_validation_metrics.json",
        "model": MODEL_DIR
        / "logistic_regression.joblib",
    },
    "random_forest": {
        "metrics": MODEL_REPORT_DIR
        / "random_forest_validation_metrics.json",
        "model": MODEL_DIR
        / "random_forest_baseline.joblib",
    },
    "random_forest_tuned": {
        "metrics": MODEL_REPORT_DIR
        / "random_forest_tuned_validation_metrics.json",
        "model": MODEL_DIR
        / "random_forest_tuned.joblib",
    },
}

EXPERIMENT_PATH = REPORT_DIR / "experiments.csv"

def load_metrics(
    metrics_path: Path,
) -> dict[str, float]:
    """Load model validation metrics from JSON."""

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_path}"
        )

    with metrics_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metrics = json.load(file)

    return metrics

def load_experiment_leaderboard(
    experiment_path: Path = EXPERIMENT_PATH,
) -> pd.DataFrame:
    """Build a model leaderboard from experiment history."""

    if not experiment_path.exists():
        raise FileNotFoundError(
            f"Experiment file not found: {experiment_path}"
        )

    experiments = pd.read_csv(experiment_path)

    required_columns = {
        "timestamp_utc",
        "model",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "training_seconds",
    }

    missing_columns = required_columns.difference(
        experiments.columns
    )

    if missing_columns:
        missing_text = ", ".join(
            sorted(missing_columns)
        )
        raise ValueError(
            "Experiment file is missing required columns: "
            f"{missing_text}"
        )

    if experiments.empty:
        raise ValueError(
            "Experiment history is empty."
        )

    experiments = experiments.copy()

    experiments["timestamp_utc"] = pd.to_datetime(
        experiments["timestamp_utc"],
        utc=True,
        errors="coerce",
    )

    experiments["balanced_accuracy"] = pd.to_numeric(
        experiments["balanced_accuracy"],
        errors="coerce",
    )

    valid_experiments = experiments.dropna(
        subset=[
            "timestamp_utc",
            "model",
            "balanced_accuracy",
        ]
    )

    if valid_experiments.empty:
        raise ValueError(
            "No valid experiments contain balanced_accuracy."
        )

    latest_experiments = (
        valid_experiments
        .sort_values("timestamp_utc")
        .drop_duplicates(
            subset="model",
            keep="last",
        )
    )

    leaderboard = (
        latest_experiments[
            [
                "model",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
                "training_seconds",
                "timestamp_utc",
            ]
        ]
        .sort_values(
            by="balanced_accuracy",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    leaderboard.insert(
        0,
        "rank",
        range(1, len(leaderboard) + 1),
    )

    return leaderboard

def build_model_leaderboard() -> pd.DataFrame:
    """Build a leaderboard from all available model metrics."""

    rows: list[dict[str, object]] = []

    for model_name, config in MODEL_CONFIGS.items():
        metrics_path = config["metrics"]

        metrics = load_metrics(metrics_path)

        rows.append(
            {
                "model": model_name,
                "accuracy": metrics.get("accuracy"),
                "balanced_accuracy": metrics.get(
                    "balanced_accuracy"
                ),
                "macro_f1": metrics.get("macro_f1"),
                "weighted_f1": metrics.get(
                    "weighted_f1"
                ),
                "training_seconds": metrics.get(
                    "training_seconds"
                ),
                "tuning_seconds": metrics.get(
                    "tuning_seconds"
                ),
            }
        )

    leaderboard = pd.DataFrame(rows)

    leaderboard = leaderboard.sort_values(
        by="balanced_accuracy",
        ascending=False,
    ).reset_index(drop=True)

    leaderboard.insert(
        0,
        "rank",
        range(1, len(leaderboard) + 1),
    )

    return leaderboard


def select_best_model(
    leaderboard: pd.DataFrame,
) -> str:
    """Select the model with the highest balanced accuracy."""

    if leaderboard.empty:
        raise ValueError(
            "Cannot select a model from an empty leaderboard."
        )

    return str(
        leaderboard.iloc[0]["model"]
    )


def copy_best_model(
    model_name: str,
) -> Path:
    """Copy the selected model to a stable best-model path."""

    source_path = MODEL_CONFIGS[model_name]["model"]

    if not source_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {source_path}"
        )

    destination_path = MODEL_DIR / "best_model.joblib"

    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    ) 

    shutil.copy2(
        source_path,
        destination_path,
    )

    return destination_path


def save_selection_summary(
    leaderboard: pd.DataFrame,
    best_model_name: str,
    output_path: Path,
) -> None:
    """Save the selected model and leaderboard summary."""

    best_row = leaderboard.iloc[0]

    summary = {
        "selection_metric": "balanced_accuracy",
        "best_model": best_model_name,
        "best_balanced_accuracy": float(
            best_row["balanced_accuracy"]
        ),
        "best_accuracy": float(
            best_row["accuracy"]
        ),
        "best_macro_f1": float(
            best_row["macro_f1"]
        ),
        "best_weighted_f1": float(
            best_row["weighted_f1"]
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )


def main() -> None:
    """Compare model results and select the best model."""

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    leaderboard = load_experiment_leaderboard()

    best_model_name = select_best_model(
        leaderboard
    )

    best_model_path = copy_best_model(
        best_model_name
    )

    leaderboard_path = (
        MODEL_REPORT_DIR
        / "model_leaderboard.csv"
    )

    leaderboard.to_csv(
        leaderboard_path,
        index=False,
    )

    summary_path = (
        MODEL_REPORT_DIR
        / "best_model_selection.json"
    )

    save_selection_summary(
        leaderboard,
        best_model_name,
        summary_path,
    )

    print("\nModel leaderboard:")

    print(
        leaderboard.to_string(
            index=False
        )
    )

    print(
        f"\nSelected best model: {best_model_name}"
    )

    print(
        f"Selection metric: balanced_accuracy"
    )

    print("\nGenerated artifacts:")

    project_root = Path(
        __file__
    ).resolve().parents[1]

    for path in [
        best_model_path,
        leaderboard_path,
        summary_path,
    ]:
        print(
            f"- {path.relative_to(project_root)}"
        )


if __name__ == "__main__":
    main()