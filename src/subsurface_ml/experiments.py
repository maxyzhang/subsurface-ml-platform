from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from subsurface_ml.config import REPORT_DIR


EXPERIMENTS_PATH = REPORT_DIR / "experiments.csv"

EXPERIMENT_COLUMNS = [
    "timestamp_utc",
    "run_type",
    "model",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "training_seconds",
    "best_cv_score",
    "parameters",
]


def build_experiment_record(
    *,
    run_type: str,
    model: str,
    metrics: dict[str, float],
    training_seconds: float | None = None,
    best_cv_score: float | None = None,
    parameters: dict[str, Any] | None = None,
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    """Build one experiment-tracking record."""

    if not run_type.strip():
        raise ValueError("run_type must not be empty")

    if not model.strip():
        raise ValueError("model must not be empty")

    timestamp = timestamp_utc or datetime.now(
        timezone.utc
    ).isoformat()

    return {
        "timestamp_utc": timestamp,
        "run_type": run_type,
        "model": model,
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get(
            "balanced_accuracy"
        ),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "training_seconds": training_seconds,
        "best_cv_score": best_cv_score,
        "parameters": json.dumps(
            parameters or {},
            sort_keys=True,
        ),
    }


def append_experiment(
    record: dict[str, Any],
    output_path: Path = EXPERIMENTS_PATH,
) -> Path:
    """Append one experiment record to a CSV file."""

    missing_columns = [
        column
        for column in EXPERIMENT_COLUMNS
        if column not in record
    ]

    if missing_columns:
        raise ValueError(
            "Experiment record is missing columns: "
            + ", ".join(missing_columns)
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    new_row = pd.DataFrame(
        [record],
        columns=EXPERIMENT_COLUMNS,
    )

    if output_path.exists():
        existing = pd.read_csv(output_path)

        combined = pd.concat(
            [existing, new_row],
            ignore_index=True,
        )
    else:
        combined = new_row

    combined.to_csv(
        output_path,
        index=False,
    )

    return output_path