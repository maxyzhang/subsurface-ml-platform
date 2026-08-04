from pathlib import Path

import pandas as pd
import numpy as np

from subsurface_ml.config import MODEL_DIR, REPORT_DIR 
from subsurface_ml.modeling import load_model, predict_classes 
from subsurface_ml.preprocessing import load_prepared_split


PREDICTION_REPORT_DIR = REPORT_DIR / "predictions"
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"


def build_prediction_output(
    metadata: pd.DataFrame,
    predictions: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Combine metadata with predicted lithology classes."""

    if len(metadata) != len(predictions):
        raise ValueError(
            "Metadata and predictions must contain the same number of rows."
        )

    output = metadata.reset_index(drop=True).copy()

    output["predicted_class"] = pd.Series(
        predictions
    ).reset_index(drop=True)

    return output


def save_predictions(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save model predictions to CSV."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        output_path,
        index=False,
    )


def main() -> None:
    """Load the selected model and generate test-set predictions."""

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Best model was not found: {BEST_MODEL_PATH}. "
            "Run scripts/select_best_model.py first."
        )

    print("Loading best model...")

    model = load_model(
        BEST_MODEL_PATH
    )

    print("Loading prepared test data...")

    X_test, y_test, metadata_test = (
        load_prepared_split("test")
    )

    print(
        f"Test rows: {len(X_test):,}"
    )
    print(
        f"Feature count: {len(X_test.columns)}"
    )

    print("Generating predictions...")

    raw_predictions = predict_classes(
        model,
        X_test,
    )

    predictions = pd.Series(
        raw_predictions,
        name="predicted_class",
    )

    prediction_output = build_prediction_output(
        metadata_test,
        predictions,
    )

    output_path = (
        PREDICTION_REPORT_DIR
        / "test_predictions.csv"
    )

    save_predictions(
        prediction_output,
        output_path,
    )

    project_root = Path(
        __file__
    ).resolve().parents[1]

    print("\nGenerated artifact:")
    print(
        f"- {output_path.relative_to(project_root)}"
    )


if __name__ == "__main__":
    main()