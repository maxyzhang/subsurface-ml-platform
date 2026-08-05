from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from subsurface_ml.config import MODEL_DIR, REPORT_DIR
from subsurface_ml.modeling import extract_feature_importances

FIGURE_DIR = REPORT_DIR / "figures"

IMPORTANCE_CSV = (
    REPORT_DIR
    / "models"
    / "random_forest_feature_importances.csv"
)

FIGURE_PATH = (
    FIGURE_DIR
    / "feature_importance.png"
)

BEST_MODEL_PATH = (
    MODEL_DIR
    / "best_model.joblib"
)

def load_best_model():
    """Load the selected best model."""
    return joblib.load(BEST_MODEL_PATH)

def build_feature_importance_dataframe(
    model,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Build a sorted feature-importance dataframe.
    """
    importance_df = extract_feature_importances(
        model,
        feature_names,
    )

    importance_df = (
        importance_df
        .sort_values(
            by="importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return importance_df

def save_feature_importances(
        importance_df: pd.DataFrame,
) -> Path:
    """Save feature importances to CSV."""

    IMPORTANCE_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance_df.to_csv(
        IMPORTANCE_CSV,
        index=False,
    )

    return IMPORTANCE_CSV

def plot_feature_importances(
        importance_df: pd.DataFrame,
) -> Path:
    """Create a feature-importance bar chart."""

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(8,5))

    plt.barh(
        importance_df["feature"],
        importance_df["importance"],
    )

    plt.gca().invert_yaxis

    plt.xlabel("Importance")
    plt.title("Random Forest Feature Importance")

    plt.tight_layout()
    plt.savefig(FIGURE_PATH)
    plt.close()

    return FIGURE_PATH

def main() -> None:
    """Load the best model and generate feature-importance artifacts."""

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Best model was not found: {BEST_MODEL_PATH}. "
            "Run scripts/select_best_model.py first."
        )

    if not IMPORTANCE_CSV.exists():
        raise FileNotFoundError(
            f"Feature-importance CSV was not found: {IMPORTANCE_CSV}. "
            "Run scripts/train_baselines.py first."
        )

    print("Loading best model...")

    model = load_best_model()

    print("Loading feature names...")

    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "The selected model does not contain training feature names."
        )

    feature_names = model.feature_names_in_.tolist()

    print("Extracting feature importances...")

    importance_df = build_feature_importance_dataframe(
        model,
        feature_names,
    )

    csv_path = save_feature_importances(
        importance_df
    )

    figure_path = plot_feature_importances(
        importance_df
    )

    project_root = Path(
        __file__
    ).resolve().parents[1]

    print("\nGenerated artifacts:")

    for path in [
        csv_path,
        figure_path,
    ]:
        print(
            f"- {path.relative_to(project_root)}"
        )


if __name__ == "__main__":
    main()