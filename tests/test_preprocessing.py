from pathlib import Path

import pandas as pd
import pytest

from subsurface_ml.preprocessing import (
    DatasetSplits,
    assert_no_well_leakage,
    build_prepared_dataset,
    coerce_numeric_columns,
    encode_lithology_labels,
    load_prepared_split,
    remove_duplicate_samples,
    remove_rows_without_features,
    remove_unlabeled_rows,
    save_prepared_splits,
    select_modeling_columns,
    split_by_well,
    split_feature_target_metadata,
    validate_split_fractions,
)


TARGET_COLUMN = "FORCE_2020_LITHOFACIES_LITHOLOGY"

FEATURE_COLUMNS = [
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


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a synthetic multi-well lithology dataset."""

    rows: list[dict[str, object]] = []

    lithology_codes = [
        30000,
        65000,
        65030,
    ]

    for well_index in range(10):
        well_name = f"WELL_{well_index:02d}"

        for sample_index in range(12):
            rows.append(
                {
                    "WELL": well_name,
                    "DEPTH_MD": (
                        1000.0
                        + well_index * 100
                        + sample_index * 0.5
                    ),
                    "GR": 40.0 + sample_index,
                    "RHOB": 2.2 + sample_index * 0.01,
                    "NPHI": 0.15 + sample_index * 0.001,
                    "DTC": 70.0 + sample_index,
                    "RDEP": 1.0 + sample_index * 0.1,
                    "RMED": 1.2 + sample_index * 0.1,
                    "SP": -20.0 + sample_index,
                    "PEF": 2.0 + sample_index * 0.01,
                    "CALI": 8.5 + sample_index * 0.01,
                    TARGET_COLUMN: lithology_codes[
                        sample_index % len(lithology_codes)
                    ],
                }
            )

    rows.append(
        {
            "WELL": "WELL_00",
            "DEPTH_MD": 9999.0,
            "GR": None,
            "RHOB": None,
            "NPHI": None,
            "DTC": None,
            "RDEP": None,
            "RMED": None,
            "SP": None,
            "PEF": None,
            "CALI": None,
            TARGET_COLUMN: 30000,
        }
    )

    rows.append(
        {
            "WELL": "WELL_01",
            "DEPTH_MD": 9998.0,
            "GR": 50.0,
            "RHOB": 2.4,
            "NPHI": 0.2,
            "DTC": 80.0,
            "RDEP": 2.0,
            "RMED": 2.1,
            "SP": -10.0,
            "PEF": 2.5,
            "CALI": 8.6,
            TARGET_COLUMN: None,
        }
    )

    return pd.DataFrame(rows)


def test_validate_split_fractions() -> None:
    validate_split_fractions(
        validation_size=0.15,
        test_size=0.15,
    )


@pytest.mark.parametrize(
    ("validation_size", "test_size"),
    [
        (0.0, 0.15),
        (1.0, 0.15),
        (0.15, 0.0),
        (0.15, 1.0),
        (0.60, 0.50),
    ],
)
def test_validate_split_fractions_rejects_invalid_values(
    validation_size: float,
    test_size: float,
) -> None:
    with pytest.raises(ValueError):
        validate_split_fractions(
            validation_size,
            test_size,
        )


def test_remove_unlabeled_rows(
    sample_dataframe: pd.DataFrame,
) -> None:
    cleaned = remove_unlabeled_rows(
        sample_dataframe
    )

    assert cleaned[TARGET_COLUMN].notna().all()

    assert len(cleaned) == len(sample_dataframe) - 1


def test_remove_duplicate_samples() -> None:
    dataframe = pd.DataFrame(
        {
            "WELL": ["A", "A", "B"],
            "DEPTH_MD": [1000.0, 1000.0, 1000.0],
            "GR": [50.0, 55.0, 60.0],
        }
    )

    cleaned = remove_duplicate_samples(dataframe)

    assert len(cleaned) == 2


def test_select_modeling_columns(
    sample_dataframe: pd.DataFrame,
) -> None:
    selected = select_modeling_columns(
        sample_dataframe,
        feature_columns=FEATURE_COLUMNS,
    )

    assert list(selected.columns) == [
        "WELL",
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    ]


def test_coerce_numeric_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "GR": ["50.0", "invalid"],
        }
    )

    converted = coerce_numeric_columns(
        dataframe,
        ["GR"],
    )

    assert converted.loc[0, "GR"] == pytest.approx(
        50.0
    )

    assert pd.isna(converted.loc[1, "GR"])


def test_remove_rows_without_features(
    sample_dataframe: pd.DataFrame,
) -> None:
    cleaned = remove_rows_without_features(
        sample_dataframe,
        feature_columns=FEATURE_COLUMNS,
    )

    assert len(cleaned) == len(sample_dataframe) - 1


def test_encode_lithology_labels(
    sample_dataframe: pd.DataFrame,
) -> None:
    labeled = remove_unlabeled_rows(
        sample_dataframe
    )

    encoded, mapping = encode_lithology_labels(
        labeled
    )

    assert mapping == {
        30000: 0,
        65000: 1,
        65030: 2,
    }

    assert set(encoded["TARGET"]) == {
        0,
        1,
        2,
    }

    assert encoded["TARGET"].dtype == "int16"


def test_split_by_well(
    sample_dataframe: pd.DataFrame,
) -> None:
    labeled = remove_unlabeled_rows(
        sample_dataframe
    )

    splits = split_by_well(
        labeled,
        validation_size=0.20,
        test_size=0.20,
        random_state=42,
    )

    assert len(splits.train) > 0
    assert len(splits.validation) > 0
    assert len(splits.test) > 0

    assert_no_well_leakage(splits)


def test_assert_no_well_leakage_rejects_overlap() -> None:
    train = pd.DataFrame({"WELL": ["A"]})
    validation = pd.DataFrame({"WELL": ["B"]})
    test = pd.DataFrame({"WELL": ["A"]})

    splits = DatasetSplits(
        train=train,
        validation=validation,
        test=test,
    )

    with pytest.raises(
        ValueError,
        match="train and test",
    ):
        assert_no_well_leakage(splits)


def test_build_prepared_dataset(
    sample_dataframe: pd.DataFrame,
) -> None:
    splits, mapping = build_prepared_dataset(
        sample_dataframe,
        feature_columns=FEATURE_COLUMNS,
    )

    assert mapping == {
        30000: 0,
        65000: 1,
        65030: 2,
    }

    assert_no_well_leakage(splits)

    combined = pd.concat(
        [
            splits.train,
            splits.validation,
            splits.test,
        ],
        ignore_index=True,
    )

    assert combined[TARGET_COLUMN].notna().all()
    assert combined["TARGET"].notna().all()

    assert not combined[FEATURE_COLUMNS].isna().all(
        axis=1
    ).any()


def test_split_feature_target_metadata(
    sample_dataframe: pd.DataFrame,
) -> None:
    labeled = remove_unlabeled_rows(
        sample_dataframe
    )

    selected = select_modeling_columns(
        labeled,
        feature_columns=FEATURE_COLUMNS,
    )

    encoded, _ = encode_lithology_labels(
        selected
    )

    features, target, metadata = (
        split_feature_target_metadata(
            encoded,
            feature_columns=FEATURE_COLUMNS,
        )
    )

    assert list(features.columns) == FEATURE_COLUMNS
    assert target.name == "TARGET"

    assert list(metadata.columns) == [
        "WELL",
        "DEPTH_MD",
        "LITHOLOGY_CODE",
    ]


def test_save_and_load_prepared_splits(
    sample_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    splits, mapping = build_prepared_dataset(
        sample_dataframe,
        feature_columns=FEATURE_COLUMNS,
    )

    save_prepared_splits(
        splits,
        mapping,
        output_dir=tmp_path,
        feature_columns=FEATURE_COLUMNS,
    )

    features, target, metadata = (
        load_prepared_split(
            "train",
            output_dir=tmp_path,
        )
    )

    assert not features.empty
    assert len(features) == len(target)
    assert len(features) == len(metadata)

    assert (
        tmp_path / "label_mapping.csv"
    ).exists()


def test_load_prepared_split_rejects_name(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="split_name",
    ):
        load_prepared_split(
            "unknown",
            output_dir=tmp_path,
        )