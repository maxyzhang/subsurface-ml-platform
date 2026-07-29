from pathlib import Path

import pandas as pd
import pytest

from subsurface_ml.eda import (
    curve_availability_report,
    dataset_quality_summary,
    find_depth_column,
    lithology_distribution,
    missing_value_report,
    save_lithology_distribution_plot,
    save_missing_value_plot,
    save_well_label_coverage_plot,
    save_well_size_plot,
    well_summary,
)


LITHOLOGY_COLUMN = "FORCE_2020_LITHOFACIES_LITHOLOGY"


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Return a small synthetic well-log dataset."""

    return pd.DataFrame(
        {
            "WELL": [
                "WELL_A",
                "WELL_A",
                "WELL_A",
                "WELL_B",
                "WELL_B",
            ],
            "DEPTH_MD": [
                1000.0,
                1000.5,
                1001.0,
                2000.0,
                2000.5,
            ],
            "GR": [
                45.0,
                None,
                55.0,
                80.0,
                82.0,
            ],
            "RHOB": [
                2.40,
                2.45,
                None,
                2.30,
                2.35,
            ],
            "NPHI": [
                0.20,
                0.21,
                0.22,
                None,
                None,
            ],
            LITHOLOGY_COLUMN: [
                30000.0,
                30000.0,
                65000.0,
                None,
                65000.0,
            ],
            "SOURCE_FILE": [
                "well_a.las",
                "well_a.las",
                "well_a.las",
                "well_b.las",
                "well_b.las",
            ],
        }
    )


def test_missing_value_report(
    sample_dataframe: pd.DataFrame,
) -> None:
    report = missing_value_report(sample_dataframe)

    assert len(report) == len(sample_dataframe.columns)

    assert {
        "column",
        "missing_count",
        "non_missing_count",
        "missing_pct",
    }.issubset(report.columns)

    assert report["missing_pct"].between(0, 100).all()

    gr_row = report.loc[report["column"] == "GR"].iloc[0]

    assert gr_row["missing_count"] == 1
    assert gr_row["non_missing_count"] == 4
    assert gr_row["missing_pct"] == pytest.approx(20.0)


def test_missing_value_report_rejects_empty_dataframe() -> None:
    with pytest.raises(
        ValueError,
        match="empty DataFrame",
    ):
        missing_value_report(pd.DataFrame())


def test_lithology_distribution(
    sample_dataframe: pd.DataFrame,
) -> None:
    distribution = lithology_distribution(sample_dataframe)

    assert distribution["count"].sum() == 4
    assert len(distribution) == 2

    assert distribution["percentage"].sum() == pytest.approx(
        100.0
    )

    codes = set(distribution["lithology_code"])

    assert codes == {30000.0, 65000.0}


def test_lithology_distribution_rejects_missing_column() -> None:
    dataframe = pd.DataFrame(
        {
            "WELL": ["A"],
            "GR": [50.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="Lithology column is missing",
    ):
        lithology_distribution(dataframe)


def test_find_depth_column(
    sample_dataframe: pd.DataFrame,
) -> None:
    assert find_depth_column(sample_dataframe) == "DEPTH_MD"


def test_find_depth_column_rejects_missing_depth() -> None:
    dataframe = pd.DataFrame(
        {
            "WELL": ["A"],
            "GR": [50.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="No supported depth column",
    ):
        find_depth_column(dataframe)


def test_well_summary(
    sample_dataframe: pd.DataFrame,
) -> None:
    summary = well_summary(sample_dataframe)

    assert len(summary) == 2

    well_a = summary.loc[
        summary["WELL"] == "WELL_A"
    ].iloc[0]

    assert well_a["rows"] == 3
    assert well_a["top_depth"] == pytest.approx(1000.0)
    assert well_a["bottom_depth"] == pytest.approx(1001.0)
    assert well_a["depth_range"] == pytest.approx(1.0)
    assert well_a["labeled_rows"] == 3
    assert well_a["unlabeled_rows"] == 0

    well_b = summary.loc[
        summary["WELL"] == "WELL_B"
    ].iloc[0]

    assert well_b["rows"] == 2
    assert well_b["labeled_rows"] == 1
    assert well_b["unlabeled_rows"] == 1


def test_curve_availability_report(
    sample_dataframe: pd.DataFrame,
) -> None:
    report = curve_availability_report(sample_dataframe)

    assert {
        "well",
        "curve",
        "available",
        "non_missing_count",
        "missing_pct",
    }.issubset(report.columns)

    well_b_nphi = report.loc[
        (report["well"] == "WELL_B")
        & (report["curve"] == "NPHI")
    ].iloc[0]

    assert not bool(well_b_nphi["available"])
    assert well_b_nphi["non_missing_count"] == 0
    assert well_b_nphi["missing_pct"] == pytest.approx(100.0)

    well_a_gr = report.loc[
        (report["well"] == "WELL_A")
        & (report["curve"] == "GR")
    ].iloc[0]

    assert bool(well_a_gr["available"])
    assert well_a_gr["non_missing_count"] == 2
    assert well_a_gr["missing_pct"] == pytest.approx(
        100 / 3
    )


def test_dataset_quality_summary(
    sample_dataframe: pd.DataFrame,
) -> None:
    summary = dataset_quality_summary(sample_dataframe)

    assert len(summary) == 1

    row = summary.iloc[0]

    assert row["rows"] == 5
    assert row["wells"] == 2
    assert row["labeled_rows"] == 4
    assert row["unlabeled_rows"] == 1
    assert row["lithology_classes"] == 2
    assert row["duplicate_rows"] == 0


def test_save_missing_value_plot(
    sample_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    report = missing_value_report(sample_dataframe)
    output_path = tmp_path / "missing_values.png"

    save_missing_value_plot(
        report,
        output_path,
        top_n=5,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_lithology_distribution_plot(
    sample_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    distribution = lithology_distribution(
        sample_dataframe
    )

    output_path = (
        tmp_path / "lithology_distribution.png"
    )

    save_lithology_distribution_plot(
        distribution,
        output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_well_size_plot(
    sample_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    summary = well_summary(sample_dataframe)
    output_path = tmp_path / "well_sizes.png"

    save_well_size_plot(
        summary,
        output_path,
        bins=5,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_well_label_coverage_plot(
    sample_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    summary = well_summary(sample_dataframe)

    output_path = (
        tmp_path / "well_label_coverage.png"
    )

    save_well_label_coverage_plot(
        summary,
        output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0 