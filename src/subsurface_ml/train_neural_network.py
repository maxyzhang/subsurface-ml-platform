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
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.0005,
    patience: int = 5,
):
    X_train, y_train, scaler = load_training_data()

    # Load validation data
    X_val_df = pd.read_parquet(
        DATA_DIR / "processed" / "X_validation.parquet"
    )
    y_val_df = pd.read_parquet(
        DATA_DIR / "processed" / "y_validation.parquet"
    )

    if isinstance(y_val_df, pd.DataFrame):
        y_val_df = y_val_df.iloc[:, 0]

    X_val_df = X_val_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X_val_df = X_val_df.fillna(
        X_val_df.median()
    )

    X_val_scaled = (
        X_val_df.to_numpy() - scaler.mean_
    ) / scaler.scale_

    X_val = torch.tensor(
        X_val_scaled,
        dtype=torch.float32,
    )

    y_val = torch.tensor(
        y_val_df.to_numpy(),
        dtype=torch.long,
    )

    train_dataset = TensorDataset(
        X_train,
        y_train,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    num_classes = int(
        torch.max(y_train).item()
    ) + 1

    model = LithologyNeuralNetwork(
        input_size=X_train.shape[1],
        num_classes=num_classes,
    )

    # -----------------------------------
    # Class weights
    # -----------------------------------

    class_counts = torch.bincount(
        y_train,
        minlength=num_classes,
    ).float()

    class_weights = torch.sqrt(
        len(y_train)
        / (
            num_classes
            * class_counts.clamp(min=1)
        )
    )
    class_weights = class_weights / class_weights.mean()

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(epochs):

        # -------------------------------
        # Training
        # -------------------------------

        model.train()

        total_train_loss = 0.0

        for features, labels in train_loader:

            optimizer.zero_grad()

            outputs = model(features)

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()

            optimizer.step()

            total_train_loss += loss.item()

        train_loss = (
            total_train_loss
            / len(train_loader)
        )

        # -------------------------------
        # Validation
        # -------------------------------

        model.eval()

        with torch.no_grad():

            val_outputs = model(X_val)

            val_loss = criterion(
                val_outputs,
                y_val,
            ).item()

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1:02d}/{epochs} "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"LR: {current_lr:.6f}"
        )

        # -------------------------------
        # Best model checkpoint
        # -------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "input_size":
                        X_train.shape[1],

                    "num_classes":
                        num_classes,

                    "scaler_mean":
                        scaler.mean_,

                    "scaler_scale":
                        scaler.scale_,

                    "best_val_loss":
                        best_val_loss,
                },
                MODEL_PATH,
            )

            print(
                "  Best model saved."
            )

        else:

            epochs_without_improvement += 1

        # -------------------------------
        # Early stopping
        # -------------------------------

        if (
            epochs_without_improvement
            >= patience
        ):

            print(
                f"Early stopping at "
                f"epoch {epoch + 1}"
            )

            break

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Model saved to: "
        f"{MODEL_PATH}"
    )

    return model

if __name__ == "__main__":
    train_neural_network()