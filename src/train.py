from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "random_forest_model.joblib"

TARGET_COLUMN = "target"


def load_data(file_path: Path) -> pd.DataFrame:
    """Load the processed dataset."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Processed data file was not found: {file_path}"
        )

    dataframe = pd.read_csv(file_path)

    if dataframe.empty:
        raise ValueError("The processed dataset is empty.")

    return dataframe


def split_features_and_target(
    dataframe: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate input features from the prediction target."""

    if target_column not in dataframe.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found. "
            f"Available columns: {list(dataframe.columns)}"
        )

    features = dataframe.drop(columns=[target_column])
    target = dataframe[target_column]

    return features, target


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[RandomForestClassifier, float]:
    """Train a Random Forest classifier and return its test accuracy."""

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy


def save_model(
    model: RandomForestClassifier,
    model_path: Path,
) -> None:
    """Save the trained model to disk."""

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def main() -> None:
    """Run the model-training pipeline."""

    dataframe = load_data(DATA_PATH)

    features, target = split_features_and_target(
        dataframe,
        TARGET_COLUMN,
    )

    model, accuracy = train_model(features, target)

    save_model(model, MODEL_PATH)

    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()