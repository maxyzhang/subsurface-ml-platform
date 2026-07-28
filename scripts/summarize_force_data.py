from subsurface_ml.data_loader import (
    LITHOLOGY_COLUMN,
    load_force_dataset,
    summarize_dataset,
)


def main() -> None:
    print("Loading FORCE 2020 LAS files...")

    dataframe = load_force_dataset()

    summary = summarize_dataset(dataframe)

    print("\nDataset summary:")
    for name, value in summary.items():
        print(f"{name:20}: {value}")

    print("\nColumns:")
    for column in dataframe.columns:
        print(column)

    print("\nLithology distribution:")
    print(
        dataframe[LITHOLOGY_COLUMN]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\nMissing-value percentages:")
    missing_percentages = (
        dataframe.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    print(missing_percentages.to_string())


if __name__ == "__main__":
    main()