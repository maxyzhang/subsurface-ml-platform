from pathlib import Path

from subsurface_ml.data_loader import load_force_dataset 
from subsurface_ml.eda import (
    curve_availability_report,
    dataset_quality_summary,
    lithology_distribution,
    missing_value_report,
    save_lithology_distribution_plot,
    save_missing_value_plot,
    save_well_label_coverage_plot,
    save_well_size_plot,
    well_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"


def main() -> None:
    """Run exploratory data analysis for FORCE 2020."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading all FORCE 2020 LAS files...")

    dataframe = load_force_dataset()

    print(
        f"Loaded {len(dataframe):,} rows "
        f"from {dataframe['WELL'].nunique()} wells."
    )

    print("\nGenerating missing-value report...")
    missing_report = missing_value_report(dataframe)

    print("Generating lithology distribution...")
    lithology_report = lithology_distribution(dataframe)

    print("Generating well summary...")
    wells_report = well_summary(dataframe)

    print("Generating curve-availability report...")
    availability_report = curve_availability_report(dataframe)

    print("Generating dataset-quality summary...")
    quality_report = dataset_quality_summary(dataframe)

    missing_report.to_csv(
        REPORT_DIR / "missing_values.csv",
        index=False,
    )

    lithology_report.to_csv(
        REPORT_DIR / "lithology_distribution.csv",
        index=False,
    )

    wells_report.to_csv(
        REPORT_DIR / "well_summary.csv",
        index=False,
    )

    availability_report.to_csv(
        REPORT_DIR / "curve_availability.csv",
        index=False,
    )

    quality_report.to_csv(
        REPORT_DIR / "dataset_quality_summary.csv",
        index=False,
    )

    print("Creating EDA figures...")

    save_missing_value_plot(
        missing_report,
        FIGURE_DIR / "missing_values.png",
    )

    save_lithology_distribution_plot(
        lithology_report,
        FIGURE_DIR / "lithology_distribution.png",
    )

    save_well_size_plot(
        wells_report,
        FIGURE_DIR / "well_sizes.png",
    )

    save_well_label_coverage_plot(
        wells_report,
        FIGURE_DIR / "well_label_coverage.png",
    )

    print("\nDataset quality summary:")
    print(quality_report.to_string(index=False))

    print("\nHighest missing-value percentages:")
    print(
        missing_report.head(15).to_string(index=False)
    )

    print("\nLithology distribution:")
    print(
        lithology_report.to_string(index=False)
    )

    print("\nGenerated reports:")

    generated_files = [
        REPORT_DIR / "missing_values.csv",
        REPORT_DIR / "lithology_distribution.csv",
        REPORT_DIR / "well_summary.csv",
        REPORT_DIR / "curve_availability.csv",
        REPORT_DIR / "dataset_quality_summary.csv",
        FIGURE_DIR / "missing_values.png",
        FIGURE_DIR / "lithology_distribution.png",
        FIGURE_DIR / "well_sizes.png",
        FIGURE_DIR / "well_label_coverage.png",
    ]

    for file_path in generated_files:
        print(f"- {file_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()