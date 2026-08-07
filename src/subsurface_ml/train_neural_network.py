from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler 
from torch.utils.data import DataLoader, TensorDataset

from subsurface_ml.config import DATA_DIR, MODEL_DIR 
from subsurface_ml.neural_network import LithologyNeuralNetwork


X_TRAIN_PATH = DATA_DIR / "processed" / "X_train.parquet"
Y_TRAIN_PATH = DATA_DIR / "processed" / "y_train.parquet"

MODEL_PATH = MODEL_DIR / "neural_network.pt"


def load_training_data():
    X = pd.read_parquet(X_TRAIN_PATH)
    y = pd.read_parquet(Y_TRAIN_PATH)

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    # Replace infinite value with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Fill missing values with the median of each feature
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y.to_numpy(),
        dtype=torch.long,
    )

    return X_tensor, y_tensor, scaler


def train_neural_network(
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 0.001,
):
    X, y, scaler = load_training_data()

    dataset = TensorDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    num_classes = int(torch.max(y).item()) + 1

    model = LithologyNeuralNetwork(
        input_size=X.shape[1],
        num_classes=num_classes,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0

        for features, labels in loader:
            optimizer.zero_grad()

            outputs = model(features)

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(loader)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} "
            f"Loss: {average_loss:.4f}"
        )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": X.shape[1],
            "num_classes": num_classes,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        },
        MODEL_PATH,
    )

    print(f"Model saved to: {MODEL_PATH}")

    return model


if __name__ == "__main__":
    train_neural_network()