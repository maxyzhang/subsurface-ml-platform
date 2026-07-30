from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin 
from sklearn.dummy import DummyClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42


def build_dummy_classifier(
    random_state: int = RANDOM_STATE,
) -> DummyClassifier:
    """Create a majority-class baseline classifier."""

    return DummyClassifier(
        strategy="most_frequent",
        random_state=random_state,
    )


def build_random_forest_pipeline(
    n_estimators: int = 100,
    max_depth: int | None = 20,
    min_samples_leaf: int = 5,
    max_features: str | float | int | None = "sqrt",
    class_weight: str | dict[int, float] | None = "balanced_subsample",
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> Pipeline:
    """Create a median-imputation and random-forest pipeline.

    The imputer is fitted only when the pipeline is trained. Validation
    and test data are transformed using training-set statistics.
    """

    if n_estimators <= 0:
        raise ValueError("n_estimators must be greater than zero")

    if max_depth is not None and max_depth <= 0:
        raise ValueError("max_depth must be greater than zero or None")

    if min_samples_leaf <= 0:
        raise ValueError("min_samples_leaf must be greater than zero")

    imputer = SimpleImputer(
        strategy="median",
        add_indicator=True,
    )

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=0,
    )

    return Pipeline(
        steps=[
            ("imputer", imputer),
            ("classifier", classifier),
        ]
    )


def validate_training_data(
    features: pd.DataFrame,
    target: pd.Series,
) -> None:
    """Validate feature and target data before model fitting."""

    if features.empty:
        raise ValueError("Training features are empty")

    if target.empty:
        raise ValueError("Training target is empty")

    if len(features) != len(target):
        raise ValueError(
            "Feature and target row counts do not match"
        )

    if target.isna().any():
        raise ValueError("Training target contains missing values")

    if target.nunique() < 2:
        raise ValueError(
            "Training target must contain at least two classes"
        )

    non_numeric_columns = [
        column
        for column in features.columns
        if not pd.api.types.is_numeric_dtype(features[column])
    ]

    if non_numeric_columns:
        column_text = ", ".join(non_numeric_columns)

        raise ValueError(
            "All modeling features must be numeric. "
            f"Non-numeric columns: {column_text}"
        )


def fit_classifier(
    model: ClassifierMixin | Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
) -> ClassifierMixin | Pipeline:
    """Validate the training data and fit a classifier."""

    validate_training_data(features, target)

    model.fit(features, target)

    return model


def predict_classes(
    model: ClassifierMixin | Pipeline,
    features: pd.DataFrame,
) -> pd.Series:
    """Predict encoded target classes."""

    if features.empty:
        raise ValueError("Prediction features are empty")

    predictions = model.predict(features)

    return pd.Series(
        predictions,
        index=features.index,
        name="PREDICTION",
        dtype="int64",
    )


def predict_probabilities(
    model: ClassifierMixin | Pipeline,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Predict class probabilities when supported by the model."""

    if features.empty:
        raise ValueError("Prediction features are empty")

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "Model does not support probability prediction"
        )

    probabilities = model.predict_proba(features)

    classes = getattr(model, "classes_", None)

    if classes is None and isinstance(model, Pipeline):
        classes = model.named_steps["classifier"].classes_

    columns = [
        f"class_{int(class_value)}"
        for class_value in classes
    ]

    return pd.DataFrame(
        probabilities,
        index=features.index,
        columns=columns,
    )


def extract_feature_importances(
    model: Pipeline,
    original_feature_names: list[str],
) -> pd.DataFrame:
    """Extract random-forest feature importances.

    Missing-value indicator names created by SimpleImputer are included.
    """

    if "imputer" not in model.named_steps:
        raise ValueError("Pipeline is missing an imputer step")

    if "classifier" not in model.named_steps:
        raise ValueError("Pipeline is missing a classifier step")

    classifier = model.named_steps["classifier"]

    if not hasattr(classifier, "feature_importances_"):
        raise ValueError(
            "Classifier does not provide feature importances"
        )

    imputer = model.named_steps["imputer"]

    transformed_feature_names = list(
        imputer.get_feature_names_out(original_feature_names)
    )

    importances = classifier.feature_importances_

    if len(transformed_feature_names) != len(importances):
        raise ValueError(
            "Feature names and importance values do not match"
        )

    return (
        pd.DataFrame(
            {
                "feature": transformed_feature_names,
                "importance": importances,
            }
        )
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def save_model(
    model: Any,
    output_path: Path,
) -> None:
    """Persist a trained model using joblib."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        output_path,
    )


def load_model(
    model_path: Path,
) -> Any:
    """Load a persisted model."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file does not exist: {model_path}"
        )

    return joblib.load(model_path)