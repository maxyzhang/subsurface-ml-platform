from pathlib import Path
from time import perf_counter

import pandas as pd

from subsurface_ml.config import (
    FIGURE_DIR,
    MODEL_DIR,
    REPORT_DIR,
)
from subsurface_ml.evaluation import (
    build_classification_report,
    build_confusion_matrix,
    calculate_classification_metrics,
    save_confusion_matrix_plot,
    save_feature_importance_plot,
    save_metrics_json,
)
from subsurface_ml.modeling import (
    build_dummy_classifier,
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
    extract_feature_importances,
    fit_classifier,
    predict_classes,
    save_model,
)
from subsurface_ml.preprocessing import (
    load_prepared_split,
)


MODEL_REPORT_DIR = REPORT_DIR / "models"
MODEL_FIGURE_DIR = FIGURE_DIR / "models"


def evaluate_and_save(
    model_name: str,
    actual: pd.Series,
    predicted: pd.Series,
) -> dict[str, float]:
    """Calculate and save validation metrics."""

    metrics = calculate_classification_metrics(
        actual,
        predicted,
    )

    report = build_classification_report(
        actual,
        predicted,
    )

    matrix = build_confusion_matrix(
        actual,
        predicted,
    )

    normalized_matrix = build_confusion_matrix(
        actual,
        predicted,
        normalize="true",
    )

    save_metrics_json(
        metrics,
        MODEL_REPORT_DIR
        / f"{model_name}_validation_metrics.json",
    )

    report.to_csv(
        MODEL_REPORT_DIR
        / f"{model_name}_validation_classification_report.csv",
        index=False,
    )

    matrix.to_csv(
        MODEL_REPORT_DIR
        / f"{model_name}_validation_confusion_matrix.csv",
    )

    normalized_matrix.to_csv(
        MODEL_REPORT_DIR
        / f"{model_name}_validation_confusion_matrix_normalized.csv",
    )

    save_confusion_matrix_plot(
        normalized_matrix,
        MODEL_FIGURE_DIR
        / f"{model_name}_validation_confusion_matrix.png",
        title=(
            f"{model_name.replace('_', ' ').title()} "
            "Validation Confusion Matrix"
        ),
    )

    return metrics


def print_metrics(
    model_name: str,
    metrics: dict[str, float],
) -> None:
    """Print formatted classification metrics."""

    print(f"\n{model_name} validation metrics:")

    for metric_name, value in metrics.items():
        print(
            f"- {metric_name:20}: {value:.4f}"
        )


def main() -> None:
    """Train and evaluate baseline lithology classifiers."""

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading prepared train data...")

    X_train, y_train, metadata_train = (
        load_prepared_split("train")
    )

    print("Loading prepared validation data...")

    X_validation, y_validation, metadata_validation = (
        load_prepared_split("validation")
    )

    print(
        f"\nTrain rows: {len(X_train):,}"
    )

    print(
        f"Train wells: "
        f"{metadata_train['WELL'].nunique()}"
    )

    print(
        f"Validation rows: {len(X_validation):,}"
    )

    print(
        f"Validation wells: "
        f"{metadata_validation['WELL'].nunique()}"
    )

    print(
        f"Feature count: {len(X_train.columns)}"
    )

    # ---------------------------------------------------------
    # Dummy classifier
    # ---------------------------------------------------------

    print("\nTraining dummy baseline...")

    dummy_model = build_dummy_classifier()

    start_time = perf_counter()

    fit_classifier(
        dummy_model,
        X_train,
        y_train,
    )

    dummy_training_seconds = (
        perf_counter() - start_time
    )

    dummy_predictions = predict_classes(
        dummy_model,
        X_validation,
    )

    dummy_metrics = evaluate_and_save(
        "dummy",
        y_validation,
        dummy_predictions,
    )

    dummy_metrics["training_seconds"] = (
        dummy_training_seconds
    )

    save_metrics_json(
        dummy_metrics,
        MODEL_REPORT_DIR
        / "dummy_validation_metrics.json",
    )

    save_model(
        dummy_model,
        MODEL_DIR / "dummy_classifier.joblib",
    )

    print_metrics(
        "Dummy classifier",
        dummy_metrics,
    )

    print("\nTraining logistic regression baseline...")

    logistic_model = build_logistic_regression_pipeline(
        max_iter=2000,
        random_state=42,
    )

    start_time = perf_counter()

    fit_classifier(
        logistic_model,
        X_train,
        y_train,
    )

    logistic_training_seconds = (
        perf_counter() - start_time
    )

    logistic_predictions = predict_classes(
        logistic_model,
        X_validation,
    )

    logistic_metrics = evaluate_and_save(
        "logistic_regression",
        y_validation,
        logistic_predictions,
    )

    logistic_metrics["training_seconds"] = (
        logistic_training_seconds
    )

    save_metrics_json(
        logistic_metrics,
        MODEL_REPORT_DIR
        / "logistic_regression_validation_metrics.json",
    )

    save_model(
        logistic_model,
        MODEL_DIR
        / "logistic_regression.joblib",
    )

    print_metrics(
        "Logistic Regression",
        logistic_metrics,
    )

    # ---------------------------------------------------------
    # Random forest
    # ---------------------------------------------------------

    print("\nTraining random forest baseline...")
    print(
        "This may take several minutes on the full dataset."
    )

    random_forest_model = (
        build_random_forest_pipeline(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    )

    start_time = perf_counter()

    fit_classifier(
        random_forest_model,
        X_train,
        y_train,
    )

    random_forest_training_seconds = (
        perf_counter() - start_time
    )

    random_forest_predictions = predict_classes(
        random_forest_model,
        X_validation,
    )

    random_forest_metrics = evaluate_and_save(
        "random_forest",
        y_validation,
        random_forest_predictions,
    )

    random_forest_metrics["training_seconds"] = (
        random_forest_training_seconds
    )

    save_metrics_json(
        random_forest_metrics,
        MODEL_REPORT_DIR
        / "random_forest_validation_metrics.json",
    )

    feature_importances = (
        extract_feature_importances(
            random_forest_model,
            list(X_train.columns),
        )
    )

    feature_importances.to_csv(
        MODEL_REPORT_DIR
        / "random_forest_feature_importances.csv",
        index=False,
    )

    save_feature_importance_plot(
        feature_importances,
        MODEL_FIGURE_DIR
        / "random_forest_feature_importance.png",
    )

    save_model(
        random_forest_model,
        MODEL_DIR
        / "random_forest_baseline.joblib",
    )

    print_metrics(
        "Random forest",
        random_forest_metrics,
    )

    # ---------------------------------------------------------
    # Comparison
    # ---------------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "model": "dummy",
                **dummy_metrics,
            },
            {
                "model": "logistic_regression",
                **logistic_metrics,
            },
            {
                "model": "random_forest",
                **random_forest_metrics,
            },
        ]
    )

    comparison.to_csv(
        MODEL_REPORT_DIR
        / "baseline_model_comparison.csv",
        index=False,
    )

    print("\nBaseline model comparison:")

    print(
        comparison.to_string(
            index=False,
        )
    )

    print("\nGenerated artifacts:")

    generated_paths = [
        MODEL_DIR / "dummy_classifier.joblib",
        MODEL_DIR / "logistic_regression_joblib",
        MODEL_DIR / "random_forest_baseline.joblib",
        MODEL_REPORT_DIR
        / "baseline_model_comparison.csv",
        MODEL_REPORT_DIR
        / "logistic_regression_validation_metrics.json",
        MODEL_REPORT_DIR
        / "random_forest_feature_importances.csv",
        MODEL_FIGURE_DIR
        / "dummy_validation_confusion_matrix.png",
        MODEL_FIGURE_DIR
        / "random_forest_validation_confusion_matrix.png",
        MODEL_FIGURE_DIR
        / "random_forest_feature_importance.png",
    ]

    project_root = Path(__file__).resolve().parents[1]

    for path in generated_paths:
        print(
            f"- {path.relative_to(project_root)}"
        )

if __name__ == "__main__":
    main()