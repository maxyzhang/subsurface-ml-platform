from subsurface_ml.config import PROCESSED_DATA_DIR 
from subsurface_ml.data_loader import load_force_dataset 
from subsurface_ml.preprocessing import (
    DEFAULT_FEATURE_COLUMNS,
    assert_no_well_leakage,
    build_prepared_dataset,
    save_prepared_splits,
)


def main() -> None:
    """Prepare FORCE 2020 data for model training."""

    print("Loading FORCE 2020 dataset...")

    dataframe = load_force_dataset()

    print(
        f"Raw rows: {len(dataframe):,}"
    )

    print(
        f"Raw wells: {dataframe['WELL'].nunique()}"
    )

    print("\nCleaning and preparing labeled samples...")

    splits, label_mapping = build_prepared_dataset(
        dataframe
    )

    assert_no_well_leakage(splits)

    print("\nSplit summary:")

    split_frames = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }

    for split_name, split_frame in split_frames.items():
        print(
            f"{split_name:12} "
            f"rows={len(split_frame):,} "
            f"wells={split_frame['WELL'].nunique()} "
            f"classes={split_frame['TARGET'].nunique()}"
        )

    print("\nSelected features:")

    for feature in DEFAULT_FEATURE_COLUMNS:
        missing_pct = (
            splits.train[feature]
            .isna()
            .mean()
            * 100
        )

        print(
            f"- {feature:10} "
            f"train missing={missing_pct:6.2f}%"
        )

    print("\nLabel mapping:")

    for original_code, encoded_class in sorted(
        label_mapping.items()
    ):
        print(
            f"- {original_code} -> {encoded_class}"
        )

    print("\nSaving prepared datasets...")

    save_prepared_splits(
        splits,
        label_mapping,
    )

    print(
        f"\nPrepared files saved under: "
        f"{PROCESSED_DATA_DIR}"
    )


if __name__ == "__main__":
    main()