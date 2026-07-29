from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from subsurface_ml.data_loader import LITHOLOGY_COLUMN


METADATA_COLUMNS = {
    "WELL",
    "SOURCE_FILE",
    LITHOLOGY_COLUMN,
    "FORCE_2020_LITHOFACIES_CONFIDENCE",
}

DEPTH_COLUMN_CANDIDATES = (
    "DEPTH_MD",
    "DEPT",
    "DEPTH",
)


def missing_value_report(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value statistics for every column.

    The returned DataFrame contains:

    - column
    - missing_count
    - missing_pct
    - non_missing_count

    Results are sorted from the highest to the lowest missing percentage.
    """

    if dataframe.empty:
        raise ValueError("Cannot analyze an empty DataFrame")

    row_count = len(dataframe)

    report = pd.DataFrame(
        {
            "column": dataframe.columns,
            "missing_count": dataframe.isna().sum().values,
            "non_missing_count": dataframe.notna().sum().values,
        }
    )

    report["missing_pct"] = (
        report["missing_count"]
        .div(row_count)
        .mul(100)
    )

    return (
        report.sort_values(
            by=["missing_pct", "column"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def lithology_distribution(
    dataframe: pd.DataFrame,
    lithology_column: str = LITHOLOGY_COLUMN,
) -> pd.DataFrame:
    """Return counts and percentages for labeled lithology classes.

    Missing lithology values are excluded because they do not represent
    trainable target classes.
    """

    if dataframe.empty:
        raise ValueError("Cannot analyze an empty DataFrame")

    if lithology_column not in dataframe.columns:
        raise ValueError(
            f"Lithology column is missing: {lithology_column}"
        )

    labeled_values = dataframe[lithology_column].dropna()

    if labeled_values.empty:
        return pd.DataFrame(
            columns=[
                "lithology_code",
                "count",
                "percentage",
            ]
        )

    counts = (
        labeled_values
        .value_counts()
        .sort_index()
        .rename_axis("lithology_code")
        .reset_index(name="count")
    )

    counts["percentage"] = (
        counts["count"]
        .div(counts["count"].sum())
        .mul(100)
    )

    return counts


def find_depth_column(
    dataframe: pd.DataFrame,
    candidates: Iterable[str] = DEPTH_COLUMN_CANDIDATES,
) -> str:
    """Return the first available depth column."""

    for column in candidates:
        if column in dataframe.columns:
            return column

    candidate_text = ", ".join(candidates)

    raise ValueError(
        "No supported depth column was found. "
        f"Expected one of: {candidate_text}"
    )


def well_summary(
    dataframe: pd.DataFrame,
    well_column: str = "WELL",
    lithology_column: str = LITHOLOGY_COLUMN,
) -> pd.DataFrame:
    """Return sample and depth statistics for every well."""

    if dataframe.empty:
        raise ValueError("Cannot analyze an empty DataFrame")

    if well_column not in dataframe.columns:
        raise ValueError(f"Well column is missing: {well_column}")

    depth_column = find_depth_column(dataframe)

    grouped = dataframe.groupby(
        well_column,
        observed=True,
        sort=True,
    )

    summary = grouped.agg(
        rows=(well_column, "size"),
        top_depth=(depth_column, "min"),
        bottom_depth=(depth_column, "max"),
        depth_samples=(depth_column, "count"),
    )

    if lithology_column in dataframe.columns:
        labeled_rows = grouped[lithology_column].count()
        summary["labeled_rows"] = labeled_rows
        summary["unlabeled_rows"] = (
            summary["rows"] - summary["labeled_rows"]
        )

    summary["depth_range"] = (
        summary["bottom_depth"] - summary["top_depth"]
    )

    return summary.reset_index()


def curve_availability_report(
    dataframe: pd.DataFrame,
    well_column: str = "WELL",
    excluded_columns: set[str] | None = None,
) -> pd.DataFrame:
    """Return curve availability for every well.

    One output row is produced for each well and curve combination.

    The ``available`` field is True when at least one non-missing value
    exists for that curve in that well.
    """

    if dataframe.empty:
        raise ValueError("Cannot analyze an empty DataFrame")

    if well_column not in dataframe.columns:
        raise ValueError(f"Well column is missing: {well_column}")

    excluded = set(METADATA_COLUMNS)

    if excluded_columns is not None:
        excluded.update(excluded_columns)

    curve_columns = [
        column
        for column in dataframe.columns
        if column not in excluded
        and column != well_column
    ]

    if not curve_columns:
        return pd.DataFrame(
            columns=[
                "well",
                "curve",
                "available",
                "non_missing_count",
                "missing_pct",
            ]
        )

    grouped = dataframe.groupby(
        well_column,
        observed=True,
        sort=True,
    )

    non_missing_counts = grouped[curve_columns].count()
    total_rows = grouped.size()

    long_report = (
        non_missing_counts
        .stack(future_stack=True)
        .rename("non_missing_count")
        .reset_index()
        .rename(
            columns={
                well_column: "well",
                "level_1": "curve",
            }
        )
    )

    total_rows_by_well = total_rows.rename("well_rows").reset_index()

    long_report = long_report.merge(
        total_rows_by_well,
        left_on="well",
        right_on=well_column,
        how="left",
    )

    if well_column != "well":
        long_report = long_report.drop(columns=[well_column])

    long_report["available"] = (
        long_report["non_missing_count"] > 0
    )

    long_report["missing_pct"] = (
        1
        - long_report["non_missing_count"]
        .div(long_report["well_rows"])
    ).mul(100)

    return (
        long_report[
            [
                "well",
                "curve",
                "available",
                "non_missing_count",
                "missing_pct",
            ]
        ]
        .sort_values(["well", "curve"])
        .reset_index(drop=True)
    )


def dataset_quality_summary(
    dataframe: pd.DataFrame,
    lithology_column: str = LITHOLOGY_COLUMN,
) -> pd.DataFrame:
    """Return a compact one-row dataset-quality summary."""

    if dataframe.empty:
        raise ValueError("Cannot analyze an empty DataFrame")

    if "WELL" not in dataframe.columns:
        raise ValueError("Well column is missing: WELL")

    labeled_rows = 0
    lithology_classes = 0

    if lithology_column in dataframe.columns:
        labeled_rows = int(
            dataframe[lithology_column].notna().sum()
        )
        lithology_classes = int(
            dataframe[lithology_column].nunique(dropna=True)
        )

    total_rows = len(dataframe)

    return pd.DataFrame(
        [
            {
                "rows": total_rows,
                "columns": len(dataframe.columns),
                "wells": int(dataframe["WELL"].nunique()),
                "labeled_rows": labeled_rows,
                "unlabeled_rows": total_rows - labeled_rows,
                "lithology_classes": lithology_classes,
                "duplicate_rows": int(
                    dataframe.duplicated().sum()
                ),
            }
        ]
    )


def save_missing_value_plot(
    report: pd.DataFrame,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Save a horizontal bar chart of missing-value percentages."""

    required_columns = {"column", "missing_pct"}

    if not required_columns.issubset(report.columns):
        raise ValueError(
            "Missing-value report must contain column and missing_pct"
        )

    if top_n <= 0:
        raise ValueError("top_n must be greater than zero")

    plot_data = (
        report.head(top_n)
        .sort_values("missing_pct", ascending=True)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, 8))

    axis.barh(
        plot_data["column"],
        plot_data["missing_pct"],
    )

    axis.set_title(
        f"Top {min(top_n, len(plot_data))} Columns by Missing Values"
    )
    axis.set_xlabel("Missing values (%)")
    axis.set_ylabel("Column")
    axis.set_xlim(0, 100)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_lithology_distribution_plot(
    distribution: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a lithology class-distribution bar chart."""

    required_columns = {
        "lithology_code",
        "count",
    }

    if not required_columns.issubset(distribution.columns):
        raise ValueError(
            "Lithology distribution must contain "
            "lithology_code and count"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(11, 6))

    labels = (
        distribution["lithology_code"]
        .astype(int)
        .astype(str)
    )

    axis.bar(
        labels,
        distribution["count"],
    )

    axis.set_title("FORCE 2020 Lithology Class Distribution")
    axis.set_xlabel("Lithology code")
    axis.set_ylabel("Labeled depth samples")
    axis.tick_params(axis="x", rotation=45)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_well_size_plot(
    summary: pd.DataFrame,
    output_path: Path,
    bins: int = 20,
) -> None:
    """Save a histogram of depth-sample counts per well."""

    if "rows" not in summary.columns:
        raise ValueError("Well summary must contain rows")

    if bins <= 0:
        raise ValueError("bins must be greater than zero")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.hist(
        summary["rows"],
        bins=bins,
    )

    axis.set_title("Distribution of Samples per Well")
    axis.set_xlabel("Depth samples per well")
    axis.set_ylabel("Number of wells")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_well_label_coverage_plot(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save labeled-row percentages for each well."""

    required_columns = {
        "WELL",
        "rows",
        "labeled_rows",
    }

    if not required_columns.issubset(summary.columns):
        raise ValueError(
            "Well summary must contain WELL, rows and labeled_rows"
        )

    plot_data = summary.copy()

    plot_data["labeled_pct"] = (
        plot_data["labeled_rows"]
        .div(plot_data["rows"])
        .mul(100)
    )

    plot_data = plot_data.sort_values(
        "labeled_pct",
        ascending=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, 18))

    axis.barh(
        plot_data["WELL"],
        plot_data["labeled_pct"],
    )

    axis.set_title("Labeled Data Coverage by Well")
    axis.set_xlabel("Labeled samples (%)")
    axis.set_ylabel("Well")
    axis.set_xlim(0, 100)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)