import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from subsurface_ml.config import DATA_DIR, MODEL_DIR 
from subsurface_ml.neural_network import LithologyNeuralNetwork


X_VALIDATION_PATH = DATA_DIR / "processed" / "X_validation.parquet"
Y_VALIDATION_PATH = DATA_DIR / "processed" / "y_validation.parquet"

MODEL_PATH = MODEL_DIR / "neural_network.pt"


def load_validation_data():
    X = pd.read_parquet(X_VALIDATION_PATH)
    y = pd.read_parquet(Y_VALIDATION_PATH)

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    # Clean invalid values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    scaler_mean = np.asarray(checkpoint["scaler_mean"])
    scaler_scale = np.asarray(checkpoint["scaler_scale"])

    X_scaled = (X.to_numpy() - scaler_mean) / scaler_scale

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y.to_numpy(),
        dtype=torch.long,
    )

    return X_tensor, y_tensor, checkpoint


def evaluate():
    X, y, checkpoint = load_validation_data()

    model = LithologyNeuralNetwork(
        input_size=checkpoint["input_size"],
        num_classes=checkpoint["num_classes"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    with torch.no_grad():
        logits = model(X)

        predictions = torch.argmax(
            logits,
            dim=1,
        )

    y_true = y.numpy()
    y_pred = predictions.numpy()

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print()
    print("Neural Network Validation Results")
    print("---------------------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print()
    print("Confusion Matrix")
    print("----------------")
    print(
        confusion_matrix(
            y_true,
            y_pred,
        )
    )

    print()
    print("Classification Report")
    print("---------------------")
    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    evaluate()