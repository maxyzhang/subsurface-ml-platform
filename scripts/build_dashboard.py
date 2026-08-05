from pathlib import Path
import html
import json
import shutil

import pandas as pd

from subsurface_ml.config import REPORT_DIR


MODEL_REPORT_DIR = REPORT_DIR / "models"
FIGURE_DIR = REPORT_DIR / "figures"
PREDICTION_DIR = REPORT_DIR / "predictions"
DASHBOARD_DIR = REPORT_DIR / "dashboard"

LEADERBOARD_PATH = MODEL_REPORT_DIR / "model_leaderboard.csv"
SELECTION_PATH = MODEL_REPORT_DIR / "best_model_selection.json"
FEATURE_IMPORTANCE_PATH = (
    MODEL_REPORT_DIR
    / "random_forest_feature_importances.csv"
)
PREDICTION_PATH = (
    PREDICTION_DIR
    / "test_predictions.csv"
)

FEATURE_IMPORTANCE_FIGURE = (
    FIGURE_DIR
    / "feature_importance.png"
)
CONFUSION_MATRIX_FIGURE = (
    FIGURE_DIR
    / "models"
    / "random_forest_validation_confusion_matrix.png"
)

DASHBOARD_PATH = DASHBOARD_DIR / "index.html"


def load_json(path: Path) -> dict[str, object]:
    """Load a JSON report."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required JSON file was not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV report."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required CSV file was not found: {path}"
        )

    return pd.read_csv(path)


def copy_dashboard_image(
    source_path: Path,
) -> Path:
    """Copy an image into the dashboard directory."""

    if not source_path.exists():
        raise FileNotFoundError(
            f"Required image was not found: {source_path}"
        )

    image_dir = DASHBOARD_DIR / "images"

    image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination_path = (
        image_dir
        / source_path.name
    )

    shutil.copy2(
        source_path,
        destination_path,
    )

    return destination_path


def dataframe_to_html(
    dataframe: pd.DataFrame,
) -> str:
    """Convert a dataframe into a styled HTML table."""

    return dataframe.to_html(
        index=False,
        border=0,
        classes="data-table",
        float_format=lambda value: f"{value:.4f}",
    )


def build_prediction_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize predicted-class counts and percentages."""

    if "predicted_class" not in predictions.columns:
        raise ValueError(
            "Prediction CSV must contain "
            "a 'predicted_class' column."
        )

    summary = (
        predictions["predicted_class"]
        .value_counts(dropna=False)
        .rename_axis("predicted_class")
        .reset_index(name="count")
    )

    summary["percentage"] = (
        summary["count"]
        / len(predictions)
        * 100.0
    )

    return summary


def build_dashboard_html(
    selection: dict[str, object],
    leaderboard: pd.DataFrame,
    feature_importances: pd.DataFrame,
    prediction_summary: pd.DataFrame,
    feature_figure_name: str,
    confusion_figure_name: str,
) -> str:
    """Build the complete dashboard HTML."""

    best_model = html.escape(
        str(selection["best_model"])
    )

    selection_metric = html.escape(
        str(selection["selection_metric"])
    )

    best_balanced_accuracy = float(
        selection["best_balanced_accuracy"]
    )
    best_accuracy = float(
        selection["best_accuracy"]
    )
    best_macro_f1 = float(
        selection["best_macro_f1"]
    )
    best_weighted_f1 = float(
        selection["best_weighted_f1"]
    )

    top_features = feature_importances.head(
        10
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >
    <title>Subsurface ML Evaluation Dashboard</title>

    <style>
        body {{
            margin: 0;
            font-family:
                Arial,
                Helvetica,
                sans-serif;
            background: #f4f6f8;
            color: #1f2933;
        }}

        .container {{
            width: min(1200px, 92%);
            margin: 0 auto;
            padding: 32px 0 48px;
        }}

        h1 {{
            margin-bottom: 6px;
        }}

        .subtitle {{
            margin-top: 0;
            color: #52606d;
        }}

        .cards {{
            display: grid;
            grid-template-columns:
                repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin: 24px 0;
        }}

        .card {{
            background: white;
            border-radius: 10px;
            padding: 18px;
            box-shadow:
                0 2px 8px rgba(0, 0, 0, 0.08);
        }}

        .card-label {{
            color: #52606d;
            font-size: 0.9rem;
        }}

        .card-value {{
            margin-top: 8px;
            font-size: 1.5rem;
            font-weight: bold;
        }}

        .section {{
            background: white;
            border-radius: 10px;
            padding: 22px;
            margin-top: 20px;
            box-shadow:
                0 2px 8px rgba(0, 0, 0, 0.08);
        }}

        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
        }}

        .data-table th,
        .data-table td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #d9e2ec;
        }}

        .data-table th {{
            background: #eef2f6;
        }}

        .figure-grid {{
            display: grid;
            grid-template-columns:
                repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
        }}

        .figure-grid img {{
            width: 100%;
            height: auto;
            border: 1px solid #d9e2ec;
            border-radius: 8px;
        }}

        code {{
            background: #eef2f6;
            padding: 2px 5px;
            border-radius: 4px;
        }}
    </style>
</head>

<body>
    <main class="container">
        <h1>Subsurface ML Evaluation Dashboard</h1>

        <p class="subtitle">
            Automated lithology-classification model report
        </p>

        <section class="cards">
            <div class="card">
                <div class="card-label">Best model</div>
                <div class="card-value">{best_model}</div>
            </div>

            <div class="card">
                <div class="card-label">
                    Selection metric
                </div>
                <div class="card-value">
                    {selection_metric}
                </div>
            </div>

            <div class="card">
                <div class="card-label">Accuracy</div>
                <div class="card-value">
                    {best_accuracy:.4f}
                </div>
            </div>

            <div class="card">
                <div class="card-label">
                    Balanced accuracy
                </div>
                <div class="card-value">
                    {best_balanced_accuracy:.4f}
                </div>
            </div>

            <div class="card">
                <div class="card-label">Macro F1</div>
                <div class="card-value">
                    {best_macro_f1:.4f}
                </div>
            </div>

            <div class="card">
                <div class="card-label">Weighted F1</div>
                <div class="card-value">
                    {best_weighted_f1:.4f}
                </div>
            </div>
        </section>

        <section class="section">
            <h2>Model leaderboard</h2>
            {dataframe_to_html(leaderboard)}
        </section>

        <section class="section">
            <h2>Top feature importances</h2>
            {dataframe_to_html(top_features)}
        </section>

        <section class="section">
            <h2>Prediction summary</h2>
            {dataframe_to_html(prediction_summary)}
        </section>

        <section class="section">
            <h2>Model visualizations</h2>

            <div class="figure-grid">
                <div>
                    <h3>Feature importance</h3>
                    <img
                        src="images/{html.escape(feature_figure_name)}"
                        alt="Feature importance chart"
                    >
                </div>

                <div>
                    <h3>Validation confusion matrix</h3>
                    <img
                        src="images/{html.escape(confusion_figure_name)}"
                        alt="Validation confusion matrix"
                    >
                </div>
            </div>
        </section>

        <section class="section">
            <h2>Reproduce this report</h2>

            <p>
                Run
                <code>python scripts/build_dashboard.py</code>
                from the project root.
            </p>
        </section>
    </main>
</body>
</html>
"""


def main() -> None:
    """Build the model-evaluation dashboard."""

    print("Loading model-selection summary...")

    selection = load_json(
        SELECTION_PATH
    )

    print("Loading model leaderboard...")

    leaderboard = load_csv(
        LEADERBOARD_PATH
    )

    print("Loading feature importances...")

    feature_importances = load_csv(
        FEATURE_IMPORTANCE_PATH
    )

    print("Loading predictions...")

    predictions = load_csv(
        PREDICTION_PATH
    )

    prediction_summary = (
        build_prediction_summary(
            predictions
        )
    )

    print("Copying dashboard images...")

    feature_image_path = copy_dashboard_image(
        FEATURE_IMPORTANCE_FIGURE
    )

    confusion_image_path = copy_dashboard_image(
        CONFUSION_MATRIX_FIGURE
    )

    dashboard_html = build_dashboard_html(
        selection,
        leaderboard,
        feature_importances,
        prediction_summary,
        feature_image_path.name,
        confusion_image_path.name,
    )

    DASHBOARD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DASHBOARD_PATH.write_text(
        dashboard_html,
        encoding="utf-8",
    )

    project_root = Path(
        __file__
    ).resolve().parents[1]

    print("\nGenerated dashboard:")
    print(
        f"- {DASHBOARD_PATH.relative_to(project_root)}"
    )


if __name__ == "__main__":
    main()