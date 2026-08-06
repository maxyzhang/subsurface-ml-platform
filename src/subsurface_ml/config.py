import os
from pathlib import Path

PROJECT_ROOT = Path(
    os.getenv(
        "SUBSURFACE_ML_PROJECT_ROOT",
        Path(__file__).resolve().parents[2],
    )
).resolve()

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "configs"


def ensure_project_directories() -> None:
    """Create project output directories if they do not exist."""

    directories = [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODEL_DIR,
        REPORT_DIR,
        FIGURE_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True) 