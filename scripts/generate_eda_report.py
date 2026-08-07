from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from subsurface_ml.config import DATA_DIR


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

X_TRAIN_PATH = DATA_DIR / "processed" / "X_train.parquet"
Y_TRAIN_PATH = DATA_DIR / "processed" / "y_train.parquet"

REPORT_DIR = Path("reports") / "eda"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

def load_data():
    X = pd.read_parquet(X_TRAIN_PATH)
    y = pd.read_parquet(Y_TRAIN_PATH)

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    return X, y


# ---------------------------------------------------------
# Summary report
# ---------------------------------------------------------

def generate_summary(X: pd.DataFrame, y: pd.Series):
    summary_path = REPORT_DIR / "eda_summary.txt"

    missing = X.isna().sum()
    missing_percent = X.isna().mean() * 100

    class_counts = y.value_counts().sort_index()

    numeric_columns = X.select_dtypes(include=[np.number]).columns

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Subsurface ML Platform - EDA Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write("DATASET SHAPE\n")
        f.write("-" * 50 + "\n")
        f.write(f"Rows: {len(X)}\n")
        f.write(f"Columns: {X.shape[1]}\n")
        f.write(f"Numeric columns: {len(numeric_columns)}\n")
        f.write("\n")

        f.write("FEATURE NAMES\n")
        f.write("-" * 50 + "\n")
        for col in X.columns:
            f.write(f"{col}\n")
        f.write("\n")

        f.write("MISSING VALUES\n")
        f.write("-" * 50 + "\n")

        for col in X.columns:
            f.write(
                f"{col}: "
                f"{missing[col]} "
                f"({missing_percent[col]:.2f}%)\n"
            )

        f.write("\n")

        f.write("CLASS DISTRIBUTION\n")
        f.write("-" * 50 + "\n")

        for label, count in class_counts.items():
            percentage = count / len(y) * 100

            f.write(
                f"Class {label}: "
                f"{count} "
                f"({percentage:.2f}%)\n"
            )

        f.write("\n")

        f.write("NUMERIC FEATURE STATISTICS\n")
        f.write("-" * 50 + "\n")
        f.write(X[numeric_columns].describe().to_string())

    print(f"Saved: {summary_path}")


# ---------------------------------------------------------
# Label distribution
# ---------------------------------------------------------

def plot_label_distribution(y: pd.Series):
    counts = y.value_counts().sort_index()

    plt.figure(figsize=(12, 6))

    counts.plot(kind="bar")

    plt.title("Lithology Class Distribution")
    plt.xlabel("Lithology Class")
    plt.ylabel("Sample Count")

    plt.xticks(rotation=0)
    plt.tight_layout()

    output_path = REPORT_DIR / "label_distribution.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


# ---------------------------------------------------------
# Missing values
# ---------------------------------------------------------

def plot_missing_values(X: pd.DataFrame):
    missing = X.isna().sum()

    missing = missing[missing > 0].sort_values(
        ascending=False
    )

    plt.figure(figsize=(14, 7))

    if len(missing) == 0:
        plt.text(
            0.5,
            0.5,
            "No missing values found",
            horizontalalignment="center",
            verticalalignment="center",
        )

        plt.axis("off")

    else:
        missing.plot(kind="bar")

        plt.title("Missing Values by Feature")
        plt.xlabel("Feature")
        plt.ylabel("Missing Count")

        plt.xticks(
            rotation=75,
            ha="right",
        )

    plt.tight_layout()

    output_path = REPORT_DIR / "missing_values.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


# ---------------------------------------------------------
# Feature distributions
# ---------------------------------------------------------

def plot_feature_distributions(X: pd.DataFrame):
    numeric_columns = list(
        X.select_dtypes(include=[np.number]).columns
    )

    if not numeric_columns:
        return

    # Limit the number of plots so the image stays readable
    columns = numeric_columns[:12]

    n_cols = 3
    n_rows = int(
        np.ceil(len(columns) / n_cols)
    )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15, 4 * n_rows),
    )

    axes = np.array(axes).reshape(-1)

    for index, column in enumerate(columns):
        values = X[column].replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        axes[index].hist(
            values,
            bins=40,
        )

        axes[index].set_title(column)
        axes[index].set_xlabel("Value")
        axes[index].set_ylabel("Frequency")

    for index in range(
        len(columns),
        len(axes),
    ):
        axes[index].axis("off")

    fig.suptitle(
        "Feature Distributions",
        fontsize=16,
    )

    plt.tight_layout()

    output_path = (
        REPORT_DIR /
        "feature_distributions.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


# ---------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------

def plot_correlation_heatmap(X: pd.DataFrame):
    numeric_df = X.select_dtypes(
        include=[np.number]
    ).copy()

    if numeric_df.empty:
        return

    numeric_df = numeric_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    correlation = numeric_df.corr()

    plt.figure(figsize=(14, 12))

    image = plt.imshow(
        correlation,
        aspect="auto",
    )

    plt.colorbar(image)

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=90,
        fontsize=8,
    )

    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns,
        fontsize=8,
    )

    plt.title("Feature Correlation Heatmap")

    plt.tight_layout()

    output_path = (
        REPORT_DIR /
        "correlation_heatmap.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


# ---------------------------------------------------------
# Feature boxplots
# ---------------------------------------------------------

def plot_feature_boxplots(X: pd.DataFrame):
    numeric_columns = list(
        X.select_dtypes(include=[np.number]).columns
    )

    if not numeric_columns:
        return

    columns = numeric_columns[:12]

    cleaned = X[columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    plt.figure(figsize=(15, 7))

    cleaned.boxplot(
        rot=75,
    )

    plt.title("Feature Boxplots")
    plt.xlabel("Feature")
    plt.ylabel("Value")

    plt.tight_layout()

    output_path = (
        REPORT_DIR /
        "feature_boxplots.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    print("Loading training data...")

    X, y = load_data()

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    print()
    print("Generating EDA report...")

    generate_summary(X, y)

    plot_label_distribution(y)

    plot_missing_values(X)

    plot_feature_distributions(X)

    plot_correlation_heatmap(X)

    plot_feature_boxplots(X)

    print()
    print("EDA report complete.")
    print(f"Output directory: {REPORT_DIR.resolve()}")


if __name__ == "__main__":
    main()