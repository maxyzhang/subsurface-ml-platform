from pathlib import Path
from time import perf_counter

import json
import pandas as pd

from subsurface_ml.config import (
    MODEL_DIR,
    REPORT_DIR,
)
from subsurface_ml.evaluation import (
    calculate_classification_metrics,
    save_metrics_json,
)
from subsurface_ml.modeling import (
    predict_classes,
    save_model,
    tune_random_forest,
)
from subsurface_ml.preprocessing import load_prepared_split


MODEL_REPORT_DIR = REPORT_DIR / "models"


def save_best_parameters(
    parameters: dict[str, object],
    output_path: Path,
) -> None:
    """Save the best hyperparameters to JSON."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            parameters,
            file,
            indent=2,
        )


def main() -> None:
    """Tune, evaluate, and save a Random Forest model."""

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    MODEL_REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading prepared training data...")

    X_train, y_train, metadata_train = (
        load_prepared_split("train")
    )

    print("Loading prepared validation data...")

    X_validation, y_validation, metadata_validation = (
        load_prepared_split("validation")
    )

    print(
        f"Train rows: {len(X_train):,}"
    )
    print(
        f"Validation rows: {len(X_validation):,}"
    )
    print(
        f"Feature count: {len(X_train.columns)}"
    )

    print(
        "\nStarting randomized hyperparameter search..."
    )

    start_time = perf_counter()

    sampled_features = X_train.sample(
        n=50_000,
        random_state=42,
    )

    sampled_targe = y_train.loc[
        sampled_features.index
    ]

    search = tune_random_forest(
        X_train,
        y_train,
        n_iter=2,
        cv=2,
        random_state=42,
        n_jobs=-1,
    )

    tuning_seconds = (
        perf_counter() - start_time
    )

    print("\nBest parameters:")

    for parameter_name, value in (
        search.best_params_.items()
    ):
        print(
            f"- {parameter_name}: {value}"
        )

    print(
        "\nBest cross-validation "
        f"balanced accuracy: {search.best_score_:.4f}"
    )

    best_model = search.best_estimator_

    predictions = predict_classes(
        best_model,
        X_validation,
    )

    validation_metrics = (
        calculate_classification_metrics(
            y_validation,
            predictions,
        )
    )

    validation_metrics["best_cv_score"] = float(
        search.best_score_
    )
    validation_metrics["tuning_seconds"] = float(
        tuning_seconds
    )

    print("\nTuned Random Forest validation metrics:")

    for metric_name, value in (
        validation_metrics.items()
    ):
        print(
            f"- {metric_name:20}: {value:.4f}"
        )

    save_model(
        best_model,
        MODEL_DIR
        / "random_forest_tuned.joblib",
    )

    save_best_parameters(
        search.best_params_,
        MODEL_REPORT_DIR
        / "random_forest_tuned_best_params.json",
    )

    save_metrics_json(
        validation_metrics,
        MODEL_REPORT_DIR
        / "random_forest_tuned_validation_metrics.json",
    )

    cv_results = pd.DataFrame(
        search.cv_results_
    ).sort_values(
        "rank_test_score"
    )

    cv_results.to_csv(
        MODEL_REPORT_DIR
        / "random_forest_tuning_results.csv",
        index=False,
    )

    print("\nGenerated artifacts:")

    generated_paths = [
        MODEL_DIR
        / "random_forest_tuned.joblib",
        MODEL_REPORT_DIR
        / "random_forest_tuned_best_params.json",
        MODEL_REPORT_DIR
        / "random_forest_tuned_validation_metrics.json",
        MODEL_REPORT_DIR
        / "random_forest_tuning_results.csv",
    ]

    project_root = Path(
        __file__
    ).resolve().parents[1]

    for path in generated_paths:
        print(
            f"- {path.relative_to(project_root)}"
        )


if __name__ == "__main__":
    main()