from pathlib import Path

import lasio
import pandas as pd

from subsurface_ml.config import RAW_DATA_DIR


FORCE_DATA_DIR = RAW_DATA_DIR / "force2020_las"

LITHOLOGY_COLUMN = "FORCE_2020_LITHOFACIES_LITHOLOGY"
CONFIDENCE_COLUMN = "FORCE_2020_LITHOFACIES_CONFIDENCE"

DEFAULT_FEATURE_COLUMNS = [
    "CALI",
    "MUDWEIGHT",
    "ROP",
    "RDEP",
    "RSHA",
    "RMED",
    "RXO",
    "SP",
    "DTC",
    "NPHI",
    "PEF",
    "GR",
    "RHOB",
    "DRHO",
    "DEPTH_MD",
    "X_LOC",
    "Y_LOC",
    "Z_LOC",
]


def find_las_files(data_dir: Path = FORCE_DATA_DIR) -> list[Path]:
    """Return all LAS files below the supplied directory."""

    if not data_dir.exists():
        raise FileNotFoundError(f"FORCE data directory does not exist: {data_dir}")

    las_files = sorted(data_dir.rglob("*.las"))

    if not las_files:
        raise FileNotFoundError(f"No LAS files found under: {data_dir}")

    return las_files


def load_las_file(file_path: Path) -> pd.DataFrame:
    """Load one LAS file and return its curves as a DataFrame."""

    if not file_path.exists():
        raise FileNotFoundError(f"LAS file does not exist: {file_path}")

    las = lasio.read(file_path)

    dataframe = las.df().reset_index()

    # LAS depth index is normally named DEPTH.
    if "DEPTH" in dataframe.columns and "DEPTH" != "DEPTH_MD":
        dataframe = dataframe.rename(columns={"DEPTH": "DEPTH"})

    dataframe["WELL"] = file_path.stem
    dataframe["SOURCE_FILE"] = file_path.name

    return dataframe


def load_force_dataset(
    data_dir: Path = FORCE_DATA_DIR,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load and combine FORCE 2020 LAS files into one DataFrame."""

    las_files = find_las_files(data_dir)

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        las_files = las_files[:limit]

    dataframes = [load_las_file(file_path) for file_path in las_files]

    return pd.concat(dataframes, ignore_index=True, sort=False)


def validate_force_dataset(dataframe: pd.DataFrame) -> None:
    """Validate the minimum columns required for lithology modeling."""

    required_columns = {
        "WELL",
        LITHOLOGY_COLUMN,
        "GR",
        "RHOB",
        "NPHI",
        "DTC",
    }

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing_text}")


def summarize_dataset(dataframe: pd.DataFrame) -> dict[str, int | float]:
    """Return basic FORCE dataset statistics."""

    validate_force_dataset(dataframe)

    return {
        "rows": len(dataframe),
        "columns": len(dataframe.columns),
        "wells": dataframe["WELL"].nunique(),
        "labeled_rows": int(dataframe[LITHOLOGY_COLUMN].notna().sum()),
        "unlabeled_rows": int(dataframe[LITHOLOGY_COLUMN].isna().sum()),
        "lithology_classes": int(dataframe[LITHOLOGY_COLUMN].nunique()),
    }