from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from subsurface_ml.config import PROCESSED_DATA_DIR 
from subsurface_ml.data_loader import LITHOLOGY_COLUMN


DEFAULT_FEATURE_COLUMNS = [
    "GR",
    "RHOB",
    "NPHI",
    "DTC",
    "RDEP",
    "RMED",
    "SP",
    "PEF",
    "CALI",
    "DEPTH_MD",
]

WELL_COLUMN = "WELL"
TARGET_COLUMN = LITHOLOGY_COLUMN


@dataclass(frozen=True)
class DatasetSplits:
    """Container for train, validation and test datasets."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def validate_split_fractions(
    validation_size: float,
    test_size: float,
) -> None:
    """Validate requested validation and test fractions."""

    if not 0 < validation_size < 1:
        raise ValueError(
            "validation_size must be between zero and one"
        )

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between zero and one"
        )

    if validation_size + test_size >= 1:
        raise ValueError(
            "validation_size plus test_size must be less than one"
        )


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: Sequence[str],
) -> None:
    """Raise an error when required columns are absent."""

    missing_columns = sorted(
        set(required_columns).difference(dataframe.columns)
    )

    if missing_columns:
        missing_text = ", ".join(missing_columns)

        raise ValueError(
            f"Dataset is missing required columns: {missing_text}"
        )


def remove_unlabeled_rows(
    dataframe: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Remove rows that do not contain a target lithology label."""

    if dataframe.empty:
        raise ValueError("Cannot process an empty DataFrame")

    validate_required_columns(
        dataframe,
        [target_column],
    )

    return (
        dataframe.loc[dataframe[target_column].notna()]
        .copy()
        .reset_index(drop=True)
    )


def remove_duplicate_samples(
    dataframe: pd.DataFrame,
    well_column: str = WELL_COLUMN,
    depth_column: str = "DEPTH_MD",
) -> pd.DataFrame:
    """Remove duplicate samples within the same well and depth."""

    validate_required_columns(
        dataframe,
        [well_column, depth_column],
    )

    return (
        dataframe.drop_duplicates(
            subset=[well_column, depth_column],
            keep="first",
        )
        .reset_index(drop=True)
    )


def select_modeling_columns(
    dataframe: pd.DataFrame,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    target_column: str = TARGET_COLUMN,
    well_column: str = WELL_COLUMN,
) -> pd.DataFrame:
    """Select features, target and grouping metadata."""

    required_columns = [
        *feature_columns,
        target_column,
        well_column,
    ]

    validate_required_columns(
        dataframe,
        required_columns,
    )

    selected_columns = [
        well_column,
        *feature_columns,
        target_column,
    ]

    return dataframe.loc[:, selected_columns].copy()


def coerce_numeric_columns(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Convert supplied columns to numeric values.

    Values that cannot be parsed are converted to missing values.
    """

    validate_required_columns(
        dataframe,
        columns,
    )

    result = dataframe.copy()

    for column in columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    return result

def remove_rows_without_features(
    dataframe: pd.DataFrame,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Remove rows without any usable well-log measurement.

    DEPTH_MD identifies the sample position but is not by itself a
    measured well-log signal. A row containing only depth is therefore
    removed.
    """

    validate_required_columns(
        dataframe,
        feature_columns,
    )

    measurement_columns = [
        column
        for column in feature_columns
        if column != "DEPTH_MD"
    ]

    if not measurement_columns:
        raise ValueError(
            "At least one measurement feature is required"
        )

    return (
        dataframe.dropna(
            subset=measurement_columns,
            how="all",
        )
        .reset_index(drop=True)
    )

def encode_lithology_labels(
    dataframe: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Encode original lithology codes as contiguous class integers.

    Returns the encoded DataFrame and a mapping from original lithology
    code to encoded integer class.
    """

    validate_required_columns(
        dataframe,
        [target_column],
    )

    if dataframe[target_column].isna().any():
        raise ValueError(
            "Target column contains missing values"
        )

    original_codes = sorted(
        int(code)
        for code in dataframe[target_column].unique()
    )

    label_mapping = {
        original_code: encoded_class
        for encoded_class, original_code
        in enumerate(original_codes)
    }

    result = dataframe.copy()

    result["LITHOLOGY_CODE"] = (
        result[target_column]
        .astype(int)
    )

    result["TARGET"] = (
        result["LITHOLOGY_CODE"]
        .map(label_mapping)
        .astype("int16")
    )

    return result, label_mapping


def split_by_well(
    dataframe: pd.DataFrame,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    well_column: str = WELL_COLUMN,
) -> DatasetSplits:
    """Split data into train, validation and test sets by well.

    Entire wells are assigned to only one split.
    """

    if dataframe.empty:
        raise ValueError("Cannot split an empty DataFrame")

    validate_split_fractions(
        validation_size,
        test_size,
    )

    validate_required_columns(
        dataframe,
        [well_column],
    )

    well_count = dataframe[well_column].nunique()

    if well_count < 3:
        raise ValueError(
            "At least three unique wells are required"
        )

    groups = dataframe[well_column]

    first_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_validation_indices, test_indices = next(
        first_splitter.split(
            dataframe,
            groups=groups,
        )
    )

    train_validation = (
        dataframe.iloc[train_validation_indices]
        .copy()
        .reset_index(drop=True)
    )

    test = (
        dataframe.iloc[test_indices]
        .copy()
        .reset_index(drop=True)
    )

    relative_validation_size = (
        validation_size / (1 - test_size)
    )

    second_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_validation_size,
        random_state=random_state + 1,
    )

    train_indices, validation_indices = next(
        second_splitter.split(
            train_validation,
            groups=train_validation[well_column],
        )
    )

    train = (
        train_validation.iloc[train_indices]
        .copy()
        .reset_index(drop=True)
    )

    validation = (
        train_validation.iloc[validation_indices]
        .copy()
        .reset_index(drop=True)
    )

    return DatasetSplits(
        train=train,
        validation=validation,
        test=test,
    )


def assert_no_well_leakage(
    splits: DatasetSplits,
    well_column: str = WELL_COLUMN,
) -> None:
    """Verify that no well appears in more than one split."""

    train_wells = set(splits.train[well_column])
    validation_wells = set(splits.validation[well_column])
    test_wells = set(splits.test[well_column])

    if train_wells.intersection(validation_wells):
        raise ValueError(
            "Well leakage detected between train and validation"
        )

    if train_wells.intersection(test_wells):
        raise ValueError(
            "Well leakage detected between train and test"
        )

    if validation_wells.intersection(test_wells):
        raise ValueError(
            "Well leakage detected between validation and test"
        )


def split_feature_target_metadata(
    dataframe: pd.DataFrame,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    well_column: str = WELL_COLUMN,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Separate features, encoded target and identifying metadata."""

    required_columns = [
        *feature_columns,
        "TARGET",
        "LITHOLOGY_CODE",
        well_column,
    ]

    validate_required_columns(
        dataframe,
        required_columns,
    )

    features = dataframe.loc[:, feature_columns].copy()

    target = (
        dataframe["TARGET"]
        .copy()
        .rename("TARGET")
    )

    metadata = dataframe.loc[
        :,
        [
            well_column,
            "DEPTH_MD",
            "LITHOLOGY_CODE",
        ],
    ].copy()

    return features, target, metadata


def build_prepared_dataset(
    dataframe: pd.DataFrame,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
) -> tuple[DatasetSplits, dict[int, int]]:
    """Run deterministic cleaning, selection and label encoding."""

    cleaned = remove_unlabeled_rows(dataframe)

    cleaned = remove_duplicate_samples(cleaned)

    selected = select_modeling_columns(
        cleaned,
        feature_columns=feature_columns,
    )

    selected = coerce_numeric_columns(
        selected,
        columns=[
            *feature_columns,
            TARGET_COLUMN,
        ],
    )

    selected = remove_rows_without_features(
        selected,
        feature_columns=feature_columns,
    )

    encoded, label_mapping = encode_lithology_labels(
        selected
    )

    splits = split_by_well(encoded)

    assert_no_well_leakage(splits)

    return splits, label_mapping


def save_dataframe(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a DataFrame as a compressed Parquet file."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_parquet(
        output_path,
        index=False,
        compression="snappy",
    )


def save_series(
    series: pd.Series,
    output_path: Path,
) -> None:
    """Save a Series as a single-column Parquet file."""

    save_dataframe(
        series.to_frame(),
        output_path,
    )


def save_label_mapping(
    label_mapping: dict[int, int],
    output_path: Path,
) -> None:
    """Save original-to-encoded lithology mapping as CSV."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mapping_frame = pd.DataFrame(
        [
            {
                "lithology_code": lithology_code,
                "target_class": target_class,
            }
            for lithology_code, target_class
            in sorted(label_mapping.items())
        ]
    )

    mapping_frame.to_csv(
        output_path,
        index=False,
    )


def save_prepared_splits(
    splits: DatasetSplits,
    label_mapping: dict[int, int],
    output_dir: Path = PROCESSED_DATA_DIR,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
) -> None:
    """Save train, validation and test artifacts."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_frames = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }

    for split_name, split_frame in split_frames.items():
        features, target, metadata = (
            split_feature_target_metadata(
                split_frame,
                feature_columns=feature_columns,
            )
        )

        save_dataframe(
            features,
            output_dir / f"X_{split_name}.parquet",
        )

        save_series(
            target,
            output_dir / f"y_{split_name}.parquet",
        )

        save_dataframe(
            metadata,
            output_dir / f"metadata_{split_name}.parquet",
        )

    save_label_mapping(
        label_mapping,
        output_dir / "label_mapping.csv",
    )


def load_prepared_split(
    split_name: str,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load features, target and metadata for one split."""

    allowed_splits = {
        "train",
        "validation",
        "test",
    }

    if split_name not in allowed_splits:
        allowed_text = ", ".join(
            sorted(allowed_splits)
        )

        raise ValueError(
            f"split_name must be one of: {allowed_text}"
        )

    features_path = (
        output_dir / f"X_{split_name}.parquet"
    )

    target_path = (
        output_dir / f"y_{split_name}.parquet"
    )

    metadata_path = (
        output_dir / f"metadata_{split_name}.parquet"
    )

    required_paths = [
        features_path,
        target_path,
        metadata_path,
    ]

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        missing_text = ", ".join(
            str(path)
            for path in missing_paths
        )

        raise FileNotFoundError(
            f"Prepared split files are missing: {missing_text}"
        )

    features = pd.read_parquet(features_path)

    target_frame = pd.read_parquet(target_path)
    target = target_frame["TARGET"]

    metadata = pd.read_parquet(metadata_path)

    return features, target, metadata