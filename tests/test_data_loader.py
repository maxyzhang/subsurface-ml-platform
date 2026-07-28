from pathlib import Path

import pandas as pd
import pytest

from subsurface_ml.data_loader import (
    FORCE_DATA_DIR,
    LITHOLOGY_COLUMN,
    find_las_files,
    load_force_dataset,
    load_las_file,
    summarize_dataset,
    validate_force_dataset,
)

@pytest.mark.skipif(
        not FORCE_DATA_DIR.exists(),
        reason="FROCE 2020 data is not installed",
)
def test_find_las_files_finds_force_files() -> None:
    las_files = find_las_files()

    assert len(las_files) == 118
    assert all(file_path.suffix.lower() == ".las" for file_path in las_files)

@pytest.mark.skipif(
        not FORCE_DATA_DIR.exists(),
        reason="FROCE 2020 data is not installed",
)
def test_load_single_las_file() -> None:
    first_file = find_las_files()[0]

    dataframe = load_las_file(first_file)

    assert not dataframe.empty
    assert "WELL" in dataframe.columns
    assert "SOURCE_FILE" in dataframe.columns
    assert dataframe["WELL"].nunique() == 1

@pytest.mark.skipif(
        not FORCE_DATA_DIR.exists(),
        reason="FROCE 2020 data is not installed",
)
def test_load_force_dataset_with_limit() -> None:
    dataframe = load_force_dataset(limit=2)

    assert not dataframe.empty
    assert dataframe["WELL"].nunique() == 2

@pytest.mark.skipif(
        not FORCE_DATA_DIR.exists(),
        reason="FROCE 2020 data is not installed",
)
def test_validate_force_dataset() -> None:
    dataframe = load_force_dataset(limit=1)

    validate_force_dataset(dataframe)

@pytest.mark.skipif(
        not FORCE_DATA_DIR.exists(),
        reason="FROCE 2020 data is not installed",
)
def test_validate_force_dataset_rejects_missing_columns() -> None:
    dataframe = pd.DataFrame({"WELL": ["example"]})

    with pytest.raises(ValueError, match="missing required columns"):
        validate_force_dataset(dataframe)

@pytest.mark.skipif(
        not FORCE_DATA_DIR.exists(),
        reason="FROCE 2020 data is not installed",
)
def test_summarize_dataset() -> None:
    dataframe = load_force_dataset(limit=2)

    summary = summarize_dataset(dataframe)

    assert summary["rows"] > 0
    assert summary["wells"] == 2
    assert summary["columns"] > 0
    assert summary["labeled_rows"] >= 0

@pytest.mark.skipif(
        not FORCE_DATA_DIR.exists(),
        reason="FROCE 2020 data is not installed",
)
def test_find_las_files_rejects_missing_directory(
    tmp_path: Path,
) -> None:
    missing_directory = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        find_las_files(missing_directory) 