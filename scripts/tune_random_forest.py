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

from subsurface_ml.settings import (
    get_config_value,
    load_yaml_config,
)


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

    config = load_yaml_config()

    random_state = get_config_value(
        config,
        "project",
        "random_state",
    )

    tuning_config = get_config_value(
        config,
        "tuning",
    )

    rf_search_config = get_config_value(
        config,
        "tuning",
        "random_forest",
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

    sample_size = min(
        int(tuning_config["sample_size"]),
        len(X_train),

    )

    sampled_features = X_train.sample(
        n=sample_size,
        random_state=random_state,
    )

    sampled_target = y_train.loc[
        sampled_features.index
    ]

    search = tune_random_forest(
    features=sampled_features,
    target=sampled_target,
    n_iter=int(tuning_config["n_iter"]),
    cv=int(tuning_config["cv"]),
    scoring=str(tuning_config["scoring"]),
    random_state=random_state,
    n_jobs=int(tuning_config["n_jobs"]),
    parameter_distributions={
        "classifier__n_estimators": rf_search_config[
            "n_estimators"
        ],
        "classifier__max_depth": rf_search_config[
            "max_depth"
        ],
        "classifier__min_samples_split": rf_search_config[
            "min_samples_split"
        ],
        "classifier__min_samples_leaf": rf_search_config[
            "min_samples_leaf"
        ],
        "classifier__max_features": rf_search_config[
            "max_features"
        ],
    },
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